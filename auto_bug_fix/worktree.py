"""Worktree management helpers."""
from __future__ import annotations

import os
import subprocess

from auto_bug_fix.git_tools import git_worktree_remove


def cleanup_worktree(repo_path: str, worktree_path: str) -> None:
    """Remove a worktree and prune stale registry entries."""
    if os.path.isdir(worktree_path):
        result = git_worktree_remove(repo_path, worktree_path)
        if not result.success:
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=repo_path, capture_output=True,
            )
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=repo_path, capture_output=True,
    )
