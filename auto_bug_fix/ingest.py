# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase -1: materialize a patch file as a commit for the existing pipeline."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import NoReturn

from auto_bug_fix.config_loader import ConfigError
from auto_bug_fix.git_tools import (
    COAUTHOR_TRAILERS,
    GitResult,
    _run_git,
    git_commit,
    git_reset_hard,
    git_rev_parse,
)
from auto_bug_fix.worktree import cleanup_worktree, create_fresh_worktree

log = logging.getLogger(__name__)

_REF_PREFIX = "refs/amalgamator/ingest"

# Trailer recording patch provenance on the synthetic commit. run_bug_fix
# surfaces it in the dossier; the reuse check greps the hash marker.
PATCH_TRAILER_KEY = "X-Amalgamator-Patch"


def _hash_marker(patch_hash: str) -> str:
    return f"sha256:{patch_hash}"


def _patch_trailer(patch_basename: str, patch_hash: str) -> str:
    return f"{patch_basename} {_hash_marker(patch_hash)}"


class IngestError(ConfigError):
    """Raised when patch ingest fails."""


@dataclass
class IngestResult:
    synthetic_commit: str
    base_ref: str
    base_sha: str
    method: str
    patch_sha256: str
    warnings: list[str] = field(default_factory=list)


def _slug(issue_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]", "-", issue_id)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "patch"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _mbox_message_count(source_path: str, patch_path: str) -> int:
    """Count mbox messages using git's own parser. 0 = not an mbox (bare diff)."""
    with tempfile.TemporaryDirectory() as d:
        r = _run_git(source_path, ["mailsplit", f"-o{d}", os.path.abspath(patch_path)])
        if not r.success:
            return 0
        try:
            return int(r.stdout.strip())
        except ValueError:
            return 0


def _patch_already_applied(result: GitResult) -> bool:
    combined = (result.stdout + result.stderr).lower()
    return "patch does not apply" not in combined and (
        "already applied" in combined
        or "no changes" in combined
        or "empty patch" in combined
    )


def _truncate_stderr(stderr: str, limit: int = 500) -> str:
    stderr = stderr.strip()
    if len(stderr) <= limit:
        return stderr
    return stderr[:limit] + "..."


def _raise_already_applied(
    reason: str = "Patch appears to already be applied on the base branch.",
) -> NoReturn:
    raise IngestError(
        f"{reason} Find the existing commit and use fix.commit mode instead."
    )


def _raise_does_not_apply(tool: str, stderr: str) -> NoReturn:
    raise IngestError(
        f"Patch does not apply to the declared base. "
        f"Is branches.source the branch this patch was written against?\n"
        f"{tool}: {_truncate_stderr(stderr)}"
    )


def _try_am(worktree: str, patch_path: str) -> tuple[str, list[str]]:
    """Try git am (mbox format). Returns (method, warnings).

    Raises IngestError on failure.
    """
    r = _run_git(worktree, ["am", "--3way", patch_path])
    if r.success:
        return "am", []

    _run_git(worktree, ["am", "--abort"])

    if _patch_already_applied(r):
        _raise_already_applied()

    r2 = _run_git(worktree, ["am", "--3way", "--ignore-whitespace", patch_path])
    if r2.success:
        return "am-ws", ["applied with --ignore-whitespace"]

    _run_git(worktree, ["am", "--abort"])

    _raise_does_not_apply("git am", r2.stderr)


def _try_apply(
    worktree: str,
    patch_path: str,
    issue_id: str,
    bug_description: str,
    patch_basename: str,
    patch_hash: str,
) -> tuple[str, list[str]]:
    """Try git apply (bare diff). Returns (method, warnings).

    Raises IngestError on failure.
    """
    strategies = [
        (["apply", "--index", patch_path], "apply"),
        (["apply", "--index", "--3way", patch_path], "apply-3way"),
        (["apply", "--index", "--ignore-whitespace", patch_path], "apply-ws"),
    ]
    last_stderr = ""
    for args, method in strategies:
        r = _run_git(worktree, args)
        if r.success:
            staged = _run_git(worktree, ["diff", "--cached", "--quiet"])
            if staged.success:
                _raise_already_applied(
                    "Patch produced no changes on the base — likely already applied."
                )

            warnings = (
                ["applied with --ignore-whitespace"] if method == "apply-ws" else []
            )
            message = f"{issue_id}: apply {patch_basename}\n\n{bug_description}"
            trailers = {
                PATCH_TRAILER_KEY: _patch_trailer(patch_basename, patch_hash),
                **COAUTHOR_TRAILERS,
            }
            cr = git_commit(worktree, message, trailers=trailers)
            if not cr.success:
                detail = (cr.stdout + cr.stderr).strip()
                raise IngestError(f"git commit failed after apply: {detail}")
            return method, warnings

        if _patch_already_applied(r):
            _raise_already_applied()

        last_stderr = r.stderr
        git_reset_hard(worktree)

    _raise_does_not_apply("git apply", last_stderr)


def ingest_patch(
    source_path: str,
    patch_path: str,
    base_ref: str,
    issue_id: str,
    bug_description: str,
    workdir: str,
) -> IngestResult:
    """Materialize a patch file as a commit pinned under refs/amalgamator/ingest/.

    Raises IngestError on any failure.
    """
    if not os.path.isfile(patch_path):
        raise IngestError(f"Patch file not found: {patch_path}")

    msg_count = _mbox_message_count(source_path, patch_path)
    if msg_count > 1:
        raise IngestError(
            f"Multi-patch mbox ({msg_count} messages) is not supported in v1. "
            "Split into individual patches or squash before ingesting."
        )

    patch_hash = _sha256_file(patch_path)
    patch_basename = os.path.basename(patch_path)

    try:
        base_sha = git_rev_parse(source_path, base_ref)
    except RuntimeError as e:
        raise IngestError(
            f"Cannot resolve base ref '{base_ref}'. "
            f"Check that branches.source is correct and run 'git fetch' if needed."
        ) from e

    slug = _slug(issue_id)
    ref_name = f"{_REF_PREFIX}/{slug}"

    existing = _run_git(source_path, ["rev-parse", "--verify", ref_name])
    if existing.success:
        existing_sha = existing.stdout.strip()
        parent = _run_git(source_path, ["rev-parse", f"{existing_sha}^"])
        if parent.success and parent.stdout.strip() == base_sha:
            msg = _run_git(source_path, ["log", "-1", "--format=%B", existing_sha])
            if _hash_marker(patch_hash) in msg.stdout:
                log.info("Reusing existing ingest commit %s", existing_sha[:12])
                return IngestResult(
                    synthetic_commit=existing_sha,
                    base_ref=base_ref,
                    base_sha=base_sha,
                    method="reuse",
                    patch_sha256=patch_hash,
                )

    worktree_path = os.path.join(workdir, f"ingest_{slug}")
    os.makedirs(workdir, exist_ok=True)
    try:
        create_fresh_worktree(source_path, worktree_path, base_sha)
    except RuntimeError as e:
        raise IngestError(f"Failed to create ingest worktree: {e}") from e

    try:
        abs_patch = os.path.abspath(patch_path)
        is_mbox = msg_count == 1

        if is_mbox:
            method, warnings = _try_am(worktree_path, abs_patch)
            r = _run_git(worktree_path, ["log", "-1", "--format=%B", "HEAD"])
            if _hash_marker(patch_hash) not in r.stdout:
                trailer_line = (
                    f"{PATCH_TRAILER_KEY}: {_patch_trailer(patch_basename, patch_hash)}"
                )
                _run_git(
                    worktree_path,
                    [
                        "commit",
                        "--amend",
                        "--no-edit",
                        "-m",
                        r.stdout.rstrip() + f"\n\n{trailer_line}",
                    ],
                )
        else:
            method, warnings = _try_apply(
                worktree_path,
                abs_patch,
                issue_id,
                bug_description,
                patch_basename,
                patch_hash,
            )

        sha = git_rev_parse(worktree_path, "HEAD")

        _run_git(source_path, ["update-ref", ref_name, sha])
        log.info("Pinned ingest commit %s at %s", sha[:12], ref_name)

        result = IngestResult(
            synthetic_commit=sha,
            base_ref=base_ref,
            base_sha=base_sha,
            method=method,
            patch_sha256=patch_hash,
            warnings=warnings,
        )

        for w in warnings:
            log.warning("Ingest warning: %s", w)

        return result

    finally:
        cleanup_worktree(source_path, worktree_path)
