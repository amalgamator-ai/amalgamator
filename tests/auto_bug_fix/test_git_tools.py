"""Tests for auto_bug_fix.git_tools — thin git command wrappers."""
from __future__ import annotations

import subprocess

from auto_bug_fix.git_tools import (
    git_cat_file_exists,
    git_diff_name_only,
    git_is_ancestor,
    git_log_commits,
    git_log_count,
    git_merge_base,
    git_rev_parse,
    git_show_files,
    git_status_porcelain,
    resolve_merge_commit,
)


def test_git_show_files(tmp_git_repo):
    files = git_show_files(tmp_git_repo.repo_path, tmp_git_repo.fix_sha)
    assert "lib/url.c" in files
    assert "tests/test_fix.c" in files


def test_git_is_ancestor_true(tmp_git_repo):
    # The initial commit (parent of fix) is an ancestor of the source branch.
    rp = tmp_git_repo.repo_path
    src = tmp_git_repo.source_branch
    initial_sha = git_rev_parse(rp, f"{tmp_git_repo.fix_sha}~1")
    assert git_is_ancestor(rp, initial_sha, src)


def test_git_is_ancestor_false(tmp_git_repo):
    # The fix commit is NOT an ancestor of target (target branched before it).
    assert not git_is_ancestor(
        tmp_git_repo.repo_path,
        tmp_git_repo.fix_sha,
        tmp_git_repo.target_branch,
    )


def test_git_cat_file_exists_true(tmp_git_repo):
    tgt = tmp_git_repo.target_branch
    assert git_cat_file_exists(tmp_git_repo.repo_path, tgt, "lib/url.c")


def test_git_cat_file_exists_false(tmp_git_repo):
    # tests/test_fix.c was added in the fix commit, so it does not exist on target.
    tgt = tmp_git_repo.target_branch
    assert not git_cat_file_exists(
        tmp_git_repo.repo_path, tgt, "tests/test_fix.c"
    )


def test_git_diff_name_only(tmp_git_repo):
    rp = tmp_git_repo.repo_path
    # Checkout target and modify lib/url.c so it shows up in the diff.
    subprocess.run(
        ["git", "checkout", tmp_git_repo.target_branch], check=True, cwd=rp,
    )
    url_path = f"{rp}/lib/url.c"
    with open(url_path, "a") as fh:
        fh.write("\n// local edit\n")

    changed = git_diff_name_only(rp, "HEAD")
    assert "lib/url.c" in changed

    # Reset so the repo stays clean for other assertions (fixture is function-scoped
    # anyway, but be explicit).
    subprocess.run(["git", "checkout", "--", "."], check=True, cwd=rp)


def test_git_status_porcelain(tmp_git_repo):
    rp = tmp_git_repo.repo_path
    untracked = f"{rp}/untracked.txt"
    with open(untracked, "w") as fh:
        fh.write("hello\n")

    lines = git_status_porcelain(rp)
    assert any("untracked.txt" in line for line in lines)


def test_git_merge_base(tmp_git_repo):
    rp = tmp_git_repo.repo_path
    src = tmp_git_repo.source_branch
    tgt = tmp_git_repo.target_branch
    base = git_merge_base(rp, src, tgt)
    # Must be a valid 40-char SHA.
    assert len(base) == 40
    assert all(c in "0123456789abcdef" for c in base)


def test_git_rev_parse(tmp_git_repo):
    sha = git_rev_parse(tmp_git_repo.repo_path, "HEAD")
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_git_log_commits(tmp_git_repo):
    shas = git_log_commits(tmp_git_repo.repo_path, tmp_git_repo.source_branch)
    # At least the initial commit and the fix commit.
    assert len(shas) >= 2


def test_git_log_count(tmp_git_repo):
    tgt = tmp_git_repo.target_branch
    src = tmp_git_repo.source_branch
    count = git_log_count(tmp_git_repo.repo_path, f"{tgt}..{src}")
    assert count == 1


def test_resolve_merge_commit_not_a_merge(tmp_git_repo):
    # HEAD is a regular commit
    result = resolve_merge_commit(tmp_git_repo.repo_path, "HEAD")
    expected = git_rev_parse(tmp_git_repo.repo_path, "HEAD")
    assert result == expected


def test_resolve_merge_commit_is_merge(tmp_git_repo):
    from pathlib import Path
    rp = tmp_git_repo.repo_path

    # Create a merge
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=rp, check=True)
    Path(rp, "feature.txt").write_text("feature")
    subprocess.run(["git", "add", "feature.txt"], cwd=rp, check=True)
    subprocess.run(["git", "commit", "-m", "feature"], cwd=rp, check=True)

    subprocess.run(["git", "checkout", "main"], cwd=rp, check=True)
    Path(rp, "main.txt").write_text("main")
    subprocess.run(["git", "add", "main.txt"], cwd=rp, check=True)
    subprocess.run(["git", "commit", "-m", "main"], cwd=rp, check=True)

    subprocess.run(["git", "merge", "feature", "--no-ff", "-m", "merge feature"], cwd=rp, check=True)

    # resolve_merge_commit should return the feature branch head (^2)
    merge_sha = git_rev_parse(rp, "HEAD")
    topic_head = git_rev_parse(rp, "HEAD^2")

    result = resolve_merge_commit(rp, merge_sha)
    assert result == topic_head
    assert result != merge_sha
