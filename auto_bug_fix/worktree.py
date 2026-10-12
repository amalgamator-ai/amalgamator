"""Worktree management helpers."""

from __future__ import annotations

import os
import shutil
import subprocess

from auto_bug_fix.git_tools import git_worktree_add, git_worktree_remove


def cleanup_worktree(repo_path: str, worktree_path: str) -> None:
    """Remove a worktree and prune stale registry entries."""
    if os.path.isdir(worktree_path):
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
