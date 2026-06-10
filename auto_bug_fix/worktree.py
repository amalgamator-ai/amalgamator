"""Two-mode worktree management: per-branch (build caching) and per-CVE (rerere-safe)."""
from __future__ import annotations

import os

from auto_bug_fix.git_tools import (
    GitResult,
    git_worktree_add,
    git_worktree_remove,
    _run_git,
)


def setup_per_branch_worktree(
    repo_path: str, branch: str, worktree_root: str,
) -> str:
    """Create a shared worktree for a branch (reuses existing). Returns the worktree path."""
    path = os.path.join(worktree_root, branch)
    if not os.path.isdir(path):
        git_worktree_add(repo_path, path, branch)
    return path


def setup_per_cve_worktree(
    repo_path: str, branch: str, cve_id: str, worktree_root: str,
) -> str:
    """Create an isolated worktree per CVE for rerere-safe conflict resolution."""
    path = os.path.join(worktree_root, branch, cve_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    git_worktree_add(repo_path, path, branch)
    return path


def cleanup_worktree(repo_path: str, worktree_path: str) -> None:
    """Remove a worktree if it exists."""
    if os.path.isdir(worktree_path):
        git_worktree_remove(repo_path, worktree_path)


def configure_rerere(worktree_path: str, enabled: bool) -> None:
    """Enable or disable git rerere in a worktree."""
    value = "true" if enabled else "false"
    _run_git(worktree_path, ["config", "rerere.enabled", value])
