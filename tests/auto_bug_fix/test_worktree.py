# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

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


def test_create_fresh_worktree_excludes_build_scratch(git_repo, tmp_path):
    """Build caches created in the worktree must not appear as untracked files."""
    from pathlib import Path

    from auto_bug_fix.worktree import create_fresh_worktree

    wt = str(tmp_path / "wt")
    create_fresh_worktree(git_repo, wt, "main")
    cache = Path(wt) / ".gocache" / "hash"
    cache.parent.mkdir(parents=True)
    cache.write_text("cache")
    (Path(wt) / ".tmp").mkdir()
    (Path(wt) / ".tmp" / "x").write_text("tmp")

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=wt, capture_output=True, text=True
    ).stdout
    assert ".gocache" not in status
    assert ".tmp" not in status


def test_scratch_excludes_are_removed_on_cleanup(git_repo, tmp_path):
    """Only the patterns this worktree added may be removed; user lines stay."""
    from pathlib import Path

    from auto_bug_fix.worktree import cleanup_worktree, create_fresh_worktree

    exclude = Path(git_repo) / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("user-ignore\n__pycache__/\n")

    wt = str(tmp_path / "wt")
    create_fresh_worktree(git_repo, wt, "main")
    after_add = exclude.read_text()
    assert ".gocache/" in after_add
    assert "user-ignore" in after_add

    cleanup_worktree(git_repo, wt)
    after_cleanup = exclude.read_text()
    # Ours are gone, the pre-existing lines remain.
    assert ".gocache/" not in after_cleanup
    assert ".agent-home/" not in after_cleanup
    assert "user-ignore" in after_cleanup
    assert "__pycache__/" in after_cleanup


def test_scratch_excludes_survive_another_live_worktree(git_repo, tmp_path):
    """Cleaning one worktree must not drop patterns a sibling still relies on."""
    from pathlib import Path

    from auto_bug_fix.worktree import cleanup_worktree, create_fresh_worktree

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=git_repo, capture_output=True, text=True
    ).stdout.strip()
    exclude = Path(git_repo) / ".git" / "info" / "exclude"
    wt1, wt2 = str(tmp_path / "wt1"), str(tmp_path / "wt2")
    create_fresh_worktree(git_repo, wt1, head)
    create_fresh_worktree(git_repo, wt2, head)

    cleanup_worktree(git_repo, wt1)
    assert ".gocache/" in exclude.read_text()

    cleanup_worktree(git_repo, wt2)
    assert ".gocache/" not in exclude.read_text()


def test_dead_worktree_marker_is_pruned_and_adopted(git_repo, tmp_path):
    """A crashed run leaves excludes behind; the next worktree must own them."""
    import shutil
    from pathlib import Path

    from auto_bug_fix.worktree import cleanup_worktree, create_fresh_worktree

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=git_repo, capture_output=True, text=True
    ).stdout.strip()
    exclude = Path(git_repo) / ".git" / "info" / "exclude"
    wt1, wt2 = str(tmp_path / "wt1"), str(tmp_path / "wt2")
    create_fresh_worktree(git_repo, wt1, head)
    marker1 = (
        Path(git_repo) / ".git" / "worktrees" / "wt1" / "amalgamator-scratch-excludes"
    )
    assert marker1.exists()

    shutil.rmtree(wt1)  # crashed run: no cleanup
    create_fresh_worktree(git_repo, wt2, head)
    assert not marker1.exists()

    cleanup_worktree(git_repo, wt2)
    assert ".gocache/" not in exclude.read_text()


def test_cleanup_removes_read_only_build_cache(git_repo, tmp_path):
    """Go module caches leave 0555 dirs that plain rmtree cannot delete."""
    import os
    from pathlib import Path

    from auto_bug_fix.worktree import cleanup_worktree, create_fresh_worktree

    wt = str(tmp_path / "wt")
    create_fresh_worktree(git_repo, wt, "main")
    cache = Path(wt) / "cache" / "mod"
    cache.mkdir(parents=True)
    (cache / "file").write_text("x")
    os.chmod(cache, 0o555)

    cleanup_worktree(git_repo, wt)

    assert not os.path.isdir(wt)
    # The same path can be reused immediately.
    create_fresh_worktree(git_repo, wt, "main")
    assert os.path.isdir(wt)
