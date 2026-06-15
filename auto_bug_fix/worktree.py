"""Worktree management helpers."""
from __future__ import annotations

import os

from auto_bug_fix.git_tools import git_worktree_remove


def cleanup_worktree(repo_path: str, worktree_path: str) -> None:
    """Remove a worktree if it exists."""
    if os.path.isdir(worktree_path):
        git_worktree_remove(repo_path, worktree_path)
