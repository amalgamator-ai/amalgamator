"""Tests for auto_bug_fix.allowlist — seed derivation, renames, and enforcement."""
import subprocess
from pathlib import Path

from auto_bug_fix.allowlist import (
    derive_seed,
    resolve_renames,
    enforce_allowlist,
    enforce_test_file_veto,
)
from auto_bug_fix.git_tools import git_merge_base


def test_derive_seed(tmp_git_repo):
    seed = derive_seed(tmp_git_repo.repo_path, tmp_git_repo.fix_sha)
    assert isinstance(seed, list)
    assert len(seed) > 0, "seed should contain files from the fix commit"
    # The fixture modifies lib/url.c and adds tests/test_fix.c
    assert "lib/url.c" in seed
    assert "tests/test_fix.c" in seed


def test_resolve_renames_existing_files(tmp_git_repo):
    rp = tmp_git_repo.repo_path
    seed = derive_seed(rp, tmp_git_repo.fix_sha)
    fork_point = git_merge_base(rp, tmp_git_repo.source_branch, tmp_git_repo.target_branch)
    resolved, escalated = resolve_renames(
        rp, seed, tmp_git_repo.target_branch, fork_point,
    )
    # lib/url.c exists on target (it was in the initial commit).
    # tests/test_fix.c does NOT exist on target, so it won't resolve.
    assert "lib/url.c" in resolved


def test_enforce_allowlist_passes(tmp_git_repo):
    rp = tmp_git_repo.repo_path

    # Checkout target, modify a tracked file, stage it so git diff HEAD shows changes.
    subprocess.run(["git", "checkout", "target"], check=True, cwd=rp, capture_output=True)

    url_path = Path(rp) / "lib" / "url.c"
    url_path.write_text("void vulnerable_func() { /* modified on target */ }")
    subprocess.run(["git", "add", "lib/url.c"], check=True, cwd=rp, capture_output=True)

    ok, violations = enforce_allowlist(rp, allowed_modules=["lib/", "tests/"])
    assert ok is True
    assert violations == []


def test_enforce_allowlist_fails(tmp_git_repo):
    rp = tmp_git_repo.repo_path

    # Checkout target and stage a modification.
    subprocess.run(["git", "checkout", "target"], check=True, cwd=rp, capture_output=True)

    url_path = Path(rp) / "lib" / "url.c"
    url_path.write_text("void vulnerable_func() { /* modified on target */ }")
    subprocess.run(["git", "add", "lib/url.c"], check=True, cwd=rp, capture_output=True)

    ok, violations = enforce_allowlist(rp, allowed_modules=[])
    assert ok is False
    assert len(violations) > 0
    assert "lib/url.c" in violations


def test_enforce_test_file_veto(tmp_git_repo):
    rp = tmp_git_repo.repo_path

    # Checkout target, create and stage a new test file NOT in the seed.
    subprocess.run(["git", "checkout", "target"], check=True, cwd=rp, capture_output=True)

    new_test = Path(rp) / "tests" / "test_extra.c"
    new_test.write_text("void test_extra() { /* extra test */ }")
    subprocess.run(["git", "add", "tests/test_extra.c"], check=True, cwd=rp, capture_output=True)

    # The seed only contains files from the fix commit; tests/test_extra.c is NOT in it.
    seed = derive_seed(rp, tmp_git_repo.fix_sha)
    ok, vetoed = enforce_test_file_veto(rp, allowed_seed=seed)
    assert ok is False, "test file not in seed should be vetoed"
    assert "tests/test_extra.c" in vetoed
