"""Tests for auto_bug_fix.worktree — worktree cleanup and prune."""

import os
import subprocess

import pytest


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo for worktree tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    subprocess.run(
        ["git", "init", "-b", "main"], check=True, cwd=rp, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], check=True, cwd=rp, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        check=True,
        cwd=rp,
        capture_output=True,
    )
    (repo / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], check=True, cwd=rp, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], check=True, cwd=rp, capture_output=True
    )
    return rp


def test_cleanup_removes_worktree(git_repo, tmp_path):
    from auto_bug_fix.worktree import cleanup_worktree

    wt = str(tmp_path / "wt")
    subprocess.run(
        ["git", "worktree", "add", wt, "HEAD"],
        check=True,
        cwd=git_repo,
        capture_output=True,
    )
    assert os.path.isdir(wt)

    cleanup_worktree(git_repo, wt)
    assert not os.path.isdir(wt)


def test_cleanup_prunes_stale_registry(git_repo, tmp_path):
    """Simulate a stale worktree: directory deleted but registry entry remains."""
    import shutil

    from auto_bug_fix.worktree import cleanup_worktree

    wt = str(tmp_path / "wt")
    subprocess.run(
        ["git", "worktree", "add", wt, "HEAD"],
        check=True,
        cwd=git_repo,
        capture_output=True,
    )

    # Simulate stale entry: delete directory without git worktree remove
    shutil.rmtree(wt)
    assert not os.path.isdir(wt)

    # Registry still has the entry
    result = subprocess.run(
        ["git", "worktree", "list"], cwd=git_repo, capture_output=True, text=True
    )
    assert wt in result.stdout

    # cleanup_worktree should prune the stale entry
    cleanup_worktree(git_repo, wt)

    # Registry should be clean now
    result = subprocess.run(
        ["git", "worktree", "list"], cwd=git_repo, capture_output=True, text=True
    )
    assert wt not in result.stdout

    # Should be able to create a new worktree at the same path
    subprocess.run(
        ["git", "worktree", "add", wt, "HEAD"],
        check=True,
        cwd=git_repo,
        capture_output=True,
    )
    assert os.path.isdir(wt)


def test_cleanup_noop_when_no_worktree(git_repo, tmp_path):
    from auto_bug_fix.worktree import cleanup_worktree

    wt = str(tmp_path / "nonexistent")
    # Should not raise
    cleanup_worktree(git_repo, wt)
