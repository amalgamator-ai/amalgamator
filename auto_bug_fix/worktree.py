# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Worktree management helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from auto_bug_fix.git_tools import _run_git, git_worktree_add, git_worktree_remove

# Build/test scratch that agents commonly create inside the worktree. The
# case build commands point GOCACHE/GOTMPDIR/TMPDIR at $PWD, so without these
# excludes a `git add -A` in a resolver session stages thousands of cache
# files into the candidate patch.
_SCRATCH_EXCLUDES = (
    ".gocache/",
    ".gotmp/",
    ".tmp/",
    ".agent-tmp/",
    ".agent-home/",
    ".pytest_cache/",
    "__pycache__/",
)
# Records which patterns this worktree added to the shared info/exclude.
_ADDED_EXCLUDES_MARKER = "amalgamator-scratch-excludes"


def _exclude_path(worktree_path: str) -> Path | None:
    """The shared ``info/exclude`` file git uses for this worktree."""
    result = _run_git(worktree_path, ["rev-parse", "--git-path", "info/exclude"])
    if not result.success or not result.stdout.strip():
        return None
    path = result.stdout.strip()
    if not os.path.isabs(path):
        path = os.path.join(worktree_path, path)
    return Path(path)


def _git_dir_path(worktree_path: str) -> Path | None:
    """This worktree's own git directory (``.git/worktrees/<name>`` when linked)."""
    result = _run_git(worktree_path, ["rev-parse", "--git-dir"])
    if not result.success or not result.stdout.strip():
        return None
    path = result.stdout.strip()
    if not os.path.isabs(path):
        path = os.path.join(worktree_path, path)
    return Path(path)


def _add_scratch_excludes(worktree_path: str) -> None:
    """Add build-scratch patterns to the repository's shared exclude file.

    The file is shared by every worktree of the repository, so the patterns we
    add are recorded in the worktree's own git directory and removed again by
    ``cleanup_worktree``.
    """
    exclude = _exclude_path(worktree_path)
    if exclude is None:
        return
    try:
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        lines = set(existing.splitlines())
        missing = [p for p in _SCRATCH_EXCLUDES if p not in lines]
        if not missing:
            return
        sep = "\n" if existing and not existing.endswith("\n") else ""
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(
            existing + sep + "".join(f"{p}\n" for p in missing), encoding="utf-8"
        )
        git_dir = _git_dir_path(worktree_path)
        if git_dir is not None:
            git_dir.mkdir(parents=True, exist_ok=True)
            (git_dir / _ADDED_EXCLUDES_MARKER).write_text(
                "".join(f"{p}\n" for p in missing), encoding="utf-8"
            )
    except OSError:
        pass


def _remove_scratch_excludes(worktree_path: str) -> None:
    """Remove only the patterns ``_add_scratch_excludes`` added for this worktree."""
    git_dir = _git_dir_path(worktree_path)
    exclude = _exclude_path(worktree_path)
    if git_dir is None or exclude is None:
        return
    marker = git_dir / _ADDED_EXCLUDES_MARKER
    if not marker.exists():
        return
    try:
        added = set(marker.read_text(encoding="utf-8").splitlines())
        remaining = [
            line
            for line in exclude.read_text(encoding="utf-8").splitlines()
            if line.strip() not in added
        ]
        exclude.write_text(
            "\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8"
        )
        marker.unlink()
    except OSError:
        pass


def cleanup_worktree(repo_path: str, worktree_path: str) -> None:
    """Remove a worktree and prune stale registry entries."""
    if os.path.isdir(worktree_path):
        _remove_scratch_excludes(worktree_path)
        result = git_worktree_remove(repo_path, worktree_path)
        if not result.success:
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=repo_path,
                capture_output=True,
            )
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=repo_path,
        capture_output=True,
    )


def create_fresh_worktree(repo_path: str, worktree_path: str, ref: str) -> None:
    """Remove any stale worktree at *worktree_path* and create a fresh detached one at *ref*.

    Raises RuntimeError if the worktree cannot be created.
    """
    cleanup_worktree(repo_path, worktree_path)
    if os.path.exists(worktree_path):
        shutil.rmtree(worktree_path)
    result = git_worktree_add(repo_path, worktree_path, ref)
    if not result.success:
        raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
    _add_scratch_excludes(worktree_path)
