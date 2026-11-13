# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Worktree management helpers."""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
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


def _rev_parse_path(worktree_path: str, *args: str) -> Path | None:
    """Resolve a path reported by ``git rev-parse`` (``--git-dir``, ``--git-path``)."""
    result = _run_git(worktree_path, ["rev-parse", *args])
    if not result.success or not result.stdout.strip():
        return None
    path = result.stdout.strip()
    if not os.path.isabs(path):
        path = os.path.join(worktree_path, path)
    return Path(path)


def _exclude_path(worktree_path: str) -> Path | None:
    """The shared ``info/exclude`` file git uses for this worktree."""
    return _rev_parse_path(worktree_path, "--git-path", "info/exclude")


def _other_marker_claims(
    worktree_path: str, own_marker: Path | None
) -> tuple[set[str], set[str]]:
    """Patterns claimed by other live worktrees, plus those orphaned by dead ones.

    Markers of dead worktrees are pruned; their patterns are returned so the
    caller can adopt them instead of leaking them into the shared file.
    """
    common = _rev_parse_path(worktree_path, "--git-common-dir")
    claims: set[str] = set()
    orphans: set[str] = set()
    if common is None or not (common / "worktrees").is_dir():
        return claims, orphans
    for wt_dir in (common / "worktrees").iterdir():
        marker = wt_dir / _ADDED_EXCLUDES_MARKER
        if (own_marker is not None and marker == own_marker) or not marker.exists():
            continue
        try:
            target = (wt_dir / "gitdir").read_text(encoding="utf-8").strip()
        except OSError:
            target = ""
        try:
            lines = set(marker.read_text(encoding="utf-8").splitlines())
        except OSError:
            lines = set()
        # gitdir points at "<worktree>/.git"; dead worktrees leave the marker
        # behind, and a live one must keep the patterns it relies on.
        if target and os.path.exists(target):
            claims |= lines
            continue
        orphans |= lines
        with contextlib.suppress(OSError):
            marker.unlink()
    return claims, orphans


def _add_scratch_excludes(worktree_path: str, adopted: set[str] | None = None) -> None:
    """Add build-scratch patterns to the repository's shared exclude file.

    The file is shared by every worktree of the repository. Each worktree
    records the patterns it relies on in its own git directory, so cleanup only
    drops patterns no other live worktree needs. *adopted* carries claims read
    from a crashed run's marker before ``worktree prune`` deleted it.
    """
    exclude = _exclude_path(worktree_path)
    git_dir = _rev_parse_path(worktree_path, "--git-dir")
    if exclude is None or git_dir is None:
        return
    marker = git_dir / _ADDED_EXCLUDES_MARKER
    try:
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        lines = set(existing.splitlines())
        missing = [p for p in _SCRATCH_EXCLUDES if p not in lines]
        # Pre-existing patterns are ours to remove only when another amalgamator
        # worktree added them; otherwise they belong to the user.
        claims, orphans = _other_marker_claims(worktree_path, marker)
        owned = (adopted or set()) | orphans | claims
        shared = [p for p in _SCRATCH_EXCLUDES if p in lines and p in owned]
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("".join(f"{p}\n" for p in missing + shared), encoding="utf-8")
        if not missing:
            return
        sep = "\n" if existing and not existing.endswith("\n") else ""
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(
            existing + sep + "".join(f"{p}\n" for p in missing), encoding="utf-8"
        )
    except OSError:
        pass


def _remove_scratch_excludes(worktree_path: str) -> None:
    """Remove the patterns ``_add_scratch_excludes`` added for this worktree."""
    git_dir = _rev_parse_path(worktree_path, "--git-dir")
    exclude = _exclude_path(worktree_path)
    if git_dir is None or exclude is None:
        return
    marker = git_dir / _ADDED_EXCLUDES_MARKER
    if not marker.exists():
        return
    try:
        added = set(marker.read_text(encoding="utf-8").splitlines())
        keep, orphans = _other_marker_claims(worktree_path, marker)
        marker.unlink()
        removable = (added | orphans) - keep
        remaining = [
            line
            for line in exclude.read_text(encoding="utf-8").splitlines()
            if line.strip() not in removable
        ]
        exclude.write_text(
            "\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8"
        )
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
    if os.path.isdir(worktree_path):
        _force_rmtree(worktree_path)


def _force_rmtree(path: str) -> None:
    """Remove a tree even when build caches left read-only directories."""
    for root, dirs, files in os.walk(path):
        with contextlib.suppress(OSError):
            os.chmod(root, stat.S_IRWXU)
        for name in dirs + files:
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(root, name), stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):
        shutil.rmtree(path)


def create_fresh_worktree(repo_path: str, worktree_path: str, ref: str) -> None:
    """Remove any stale worktree at *worktree_path* and create a fresh detached one at *ref*.

    Raises RuntimeError if the worktree cannot be created.
    """
    # A crashed run's marker records excludes nobody else owns, and the prune
    # below deletes that marker with its registry entry — read the claim first.
    _, adopted = _other_marker_claims(repo_path, None)
    cleanup_worktree(repo_path, worktree_path)
    if os.path.exists(worktree_path):
        _force_rmtree(worktree_path)
    result = git_worktree_add(repo_path, worktree_path, ref)
    if not result.success:
        raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
    _add_scratch_excludes(worktree_path, adopted)
