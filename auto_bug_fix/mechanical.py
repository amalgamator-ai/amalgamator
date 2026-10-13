# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Git-native mechanical resolution of cherry-pick conflicts.

Before handing conflicted files to the LLM, try to re-apply the upstream
per-file patch onto the target's version ("ours", index stage 2) with
progressively relaxed matching:

1. ``git apply --ignore-whitespace``      — context drifted only in whitespace
2. ``git apply -C1``                      — outer context lines changed/removed
3. ``git apply -C1 --ignore-whitespace``  — both

This wins exactly the class of conflicts caused by cosmetic divergence of the
surrounding code, and the result is byte-spliced from the real patch instead
of LLM-regenerated — no formatting drift, no scope creep. Anything that
doesn't fully apply is restored to its conflicted state for the LLM.

All file content is handled as bytes so line endings and non-UTF-8 encodings
survive untouched. Whitespace-significant file types never use the
``--ignore-whitespace`` rungs.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

from auto_bug_fix.git_tools import _run_git

_CONFLICT_MARKER_RE = re.compile(rb"^(<{7}|={7}|>{7})(\r?$| )", re.MULTILINE)

_WS_SENSITIVE_SUFFIXES = (".py", ".yaml", ".yml", ".mk")

_RUNGS: list[tuple[str, list[str], bool]] = [
    # (name, git-apply flags, whitespace-lenient)
    ("apply-ignore-ws", ["--ignore-whitespace"], True),
    ("apply-C1", ["-C1"], False),
    ("apply-C1-ignore-ws", ["-C1", "--ignore-whitespace"], True),
]


def _is_ws_sensitive(path: str) -> bool:
    base = os.path.basename(path)
    return path.endswith(_WS_SENSITIVE_SUFFIXES) or base.startswith("Makefile")


def _git_bytes(repo_path: str, args: list[str]) -> bytes | None:
    """Run a git command and return raw stdout bytes, or None on failure."""
    result = subprocess.run(["git"] + args, cwd=repo_path, capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout


def _per_file_patch(repo_path: str, commit: str, path: str) -> bytes | None:
    """The upstream commit's diff for one file, relative to its first parent.

    ``<commit>^1..<commit>`` matches what ``cherry-pick -m1`` applies for
    merge commits and is identical to ``git show`` for ordinary commits.
    """
    return _git_bytes(repo_path, ["diff", f"{commit}^1..{commit}", "--", path])


def _stage_blob(repo_path: str, stage: int, path: str) -> bytes | None:
    """Content of the index stage (1=base, 2=ours, 3=theirs) for a conflicted path."""
    data = _git_bytes(repo_path, ["show", f":{stage}:{path}"])
    if data is None or b"\x00" in data[:8192]:
        return None
    return data


def _restore_conflict(repo_path: str, path: str) -> None:
    """Recreate the conflicted working-tree content from the index stages."""
    _run_git(repo_path, ["checkout", "-m", "--", path])


def try_resolve_file(repo_path: str, commit: str, path: str) -> str | None:
    """Attempt mechanical resolution of one conflicted file.

    Returns the name of the successful rung, or None (with the conflicted
    working-tree state restored).
    """
    ours = _stage_blob(repo_path, 2, path)
    if ours is None:
        return None  # deleted/added on one side, or binary — LLM territory
    patch_bytes = _per_file_patch(repo_path, commit, path)
    if not patch_bytes or not patch_bytes.strip():
        return None

    ws_sensitive = _is_ws_sensitive(path)
    abs_path = os.path.join(repo_path, path)

    patch_file = tempfile.NamedTemporaryFile(  # noqa: SIM115 (delete-on-exit, unlinked in finally)
        mode="wb", suffix=".patch", delete=False
    )
    try:
        patch_file.write(patch_bytes)
        patch_file.close()

        for name, flags, ws_lenient in _RUNGS:
            if ws_sensitive and ws_lenient:
                continue
            # Start from the pristine target-side version each attempt.
            with open(abs_path, "wb") as fh:
                fh.write(ours)
            result = _run_git(repo_path, ["apply", *flags, patch_file.name])
            if not result.success:
                continue
            content: bytes | None
            try:
                with open(abs_path, "rb") as fh:
                    content = fh.read()
            except FileNotFoundError:
                content = None  # the patch deleted the file — a valid outcome
            if content is not None and _CONFLICT_MARKER_RE.search(content):
                continue
            _run_git(repo_path, ["add", "-A", "--", path])
            return name

        _restore_conflict(repo_path, path)
        return None
    finally:
        os.unlink(patch_file.name)


def resolve_mechanically(
    repo_path: str,
    commit: str,
    conflicted_files: list[str],
) -> dict[str, str]:
    """Try mechanical resolution for each conflicted file.

    Returns {path: rung_name} for the files that were fully resolved and
    staged. Files that remain conflicted are left in their original state.
    """
    resolved: dict[str, str] = {}
    for path in conflicted_files:
        rung = try_resolve_file(repo_path, commit, path)
        if rung:
            resolved[path] = rung
    return resolved
