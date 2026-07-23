"""Tests for backport_scanner/triage.py — check_branch() and check_ref() detection cascade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backport_scanner.triage import (
    BranchCheckResult,
    FixStatus,
    _check_file_diff,
    _resolve_ref,
    check_branch,
    check_ref,
    get_tags,
)

REPO = Path("/fake/repo")
BRANCH = "release-1.33"
TAG = "v6.12.25"
FIX = "abc1234567890"
VULN = "def9876543210"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_result(status: FixStatus, ref_type: str = "branch") -> BranchCheckResult:
    name = TAG if ref_type == "tag" else BRANCH
    return BranchCheckResult(
        branch=name, status=status, method="test", detail="",
        changed_files=["foo.go"], ref_type=ref_type,
    )


# ── check_branch: Level 1 (exact commit) ─────────────────────────────────────

@patch("backport_scanner.triage.git_show_files", return_value=["server/foo.go"])
@patch("backport_scanner.triage.git_is_ancestor", return_value=True)
def test_check_branch_exact_match(mock_ancestor, mock_files):
    result = check_branch(REPO, BRANCH, FIX)
    assert result.status == FixStatus.EXACT_MATCH
    assert result.method == "git_is_ancestor"
    mock_ancestor.assert_called_once_with(str(REPO), FIX, f"origin/{BRANCH}")


# ── check_branch: Level 2 (equivalent patch) ─────────────────────────────────

@patch("backport_scanner.triage.git_show_files", return_value=["server/foo.go"])
@patch("backport_scanner.triage.forward_patch_id_check", return_value="deadbeef1234")
@patch("backport_scanner.triage.git_is_ancestor", return_value=False)
def test_check_branch_equivalent(mock_ancestor, mock_fwd, mock_files):
    result = check_branch(REPO, BRANCH, FIX)
    assert result.status == FixStatus.EQUIVALENT
    assert result.method == "forward_patch_id_check"
    assert "deadbeef" in result.detail


# ── check_branch: Level 3 (not affected — branch predates bug) ───────────────

@patch("backport_scanner.triage.git_show_files", return_value=["server/foo.go"])
@patch("backport_scanner.triage.git_cat_file_exists", return_value=False)
@patch("backport_scanner.triage.backward_patch_id_check", return_value=None)
@patch("backport_scanner.triage.forward_patch_id_check", return_value=None)
@patch("backport_scanner.triage.git_is_ancestor", return_value=False)
def test_check_branch_not_affected(mock_ancestor, mock_fwd, mock_bwd, mock_cat, mock_files):
    result = check_branch(REPO, BRANCH, FIX, vuln_commit=VULN)
    assert result.status == FixStatus.NOT_AFFECTED
    assert result.method == "backward_patch_id_check"


@patch("backport_scanner.triage.git_show_files", return_value=["server/foo.go"])
@patch("backport_scanner.triage.forward_patch_id_check", return_value=None)
@patch("backport_scanner.triage.git_is_ancestor", return_value=False)
def test_check_branch_no_vuln_commit_skips_level3(mock_ancestor, mock_fwd, mock_files):
    """Without vuln_commit, level 3 is skipped entirely."""
    with patch("backport_scanner.triage._check_file_diff", return_value=FixStatus.NEEDS_FIX):
        result = check_branch(REPO, BRANCH, FIX, vuln_commit=None)
    assert result.status == FixStatus.NEEDS_FIX


# ── check_branch: Level 4 (file diff heuristic) ──────────────────────────────

@patch("backport_scanner.triage.git_show_files", return_value=["server/foo.go"])
@patch("backport_scanner.triage.git_cat_file_exists", return_value=True)
@patch("backport_scanner.triage.backward_patch_id_check", return_value=None)
@patch("backport_scanner.triage.forward_patch_id_check", return_value=None)
@patch("backport_scanner.triage.git_is_ancestor", return_value=False)
def test_check_branch_needs_fix(mock_ancestor, mock_fwd, mock_bwd, mock_cat, mock_files):
    with patch("backport_scanner.triage._check_file_diff", return_value=FixStatus.NEEDS_FIX):
        result = check_branch(REPO, BRANCH, FIX, vuln_commit=VULN)
    assert result.status == FixStatus.NEEDS_FIX
    assert result.method == "file diff heuristic"


@patch("backport_scanner.triage.git_show_files", return_value=["server/foo.go"])
@patch("backport_scanner.triage.git_cat_file_exists", return_value=True)
@patch("backport_scanner.triage.backward_patch_id_check", return_value=None)
@patch("backport_scanner.triage.forward_patch_id_check", return_value=None)
@patch("backport_scanner.triage.git_is_ancestor", return_value=False)
def test_check_branch_ambiguous(mock_ancestor, mock_fwd, mock_bwd, mock_cat, mock_files):
    with patch("backport_scanner.triage._check_file_diff", return_value=FixStatus.AMBIGUOUS):
        result = check_branch(REPO, BRANCH, FIX, vuln_commit=VULN)
    assert result.status == FixStatus.AMBIGUOUS


# ── _check_file_diff ──────────────────────────────────────────────────────────

def _mock_run_side_effect(fix_diff: str, branch_contents: dict[str, str]):
    """Return a side_effect function for _run that returns canned data."""
    def side_effect(cmd, cwd, timeout=30):
        if "show" in cmd and "--unified=0" in cmd:
            return 0, fix_diff, ""
        # git show origin/branch:file
        for fname, content in branch_contents.items():
            if fname in " ".join(cmd):
                return 0, content, ""
        return 1, "", "not found"
    return side_effect


@patch("backport_scanner.triage._run")
def test_file_diff_likely_present(mock_run):
    fix_diff = "+line one\n+line two\n+line three\n+line four\n+line five"
    branch_content = "line one\nline two\nline three\nline four\nline five\n"
    mock_run.side_effect = _mock_run_side_effect(fix_diff, {"foo.go": branch_content})
    result = _check_file_diff(REPO, BRANCH, FIX, ["foo.go"])
    assert result == FixStatus.LIKELY_PRESENT


@patch("backport_scanner.triage._run")
def test_file_diff_needs_fix(mock_run):
    fix_diff = "+brand_new_function()\n+another_new_call()\n+third_line()"
    branch_content = "old_function()\nsome_other_code()\n"
    mock_run.side_effect = _mock_run_side_effect(fix_diff, {"foo.go": branch_content})
    result = _check_file_diff(REPO, BRANCH, FIX, ["foo.go"])
    assert result == FixStatus.NEEDS_FIX


@patch("backport_scanner.triage._run")
def test_file_diff_multi_file(mock_run):
    """Fix spread across two files — should aggregate content from both."""
    fix_diff = "+line_in_a\n+line_in_b"
    mock_run.side_effect = _mock_run_side_effect(
        fix_diff, {"a.go": "line_in_a\n", "b.go": "line_in_b\n"}
    )
    result = _check_file_diff(REPO, BRANCH, FIX, ["a.go", "b.go"])
    assert result == FixStatus.LIKELY_PRESENT


@patch("backport_scanner.triage._run")
def test_file_diff_no_changed_files(mock_run):
    result = _check_file_diff(REPO, BRANCH, FIX, [])
    assert result == FixStatus.AMBIGUOUS
    mock_run.assert_not_called()


@patch("backport_scanner.triage._run")
def test_file_diff_no_added_lines(mock_run):
    """Fix only removes lines — no additions to match."""
    fix_diff = "-removed_line\n-another_removed"
    mock_run.return_value = (0, fix_diff, "")
    result = _check_file_diff(REPO, BRANCH, FIX, ["foo.go"])
    assert result == FixStatus.AMBIGUOUS


# ── _resolve_ref ──────────────────────────────────────────────────────────────

def test_resolve_ref_branch():
    assert _resolve_ref("release-1.33", "branch") == "origin/release-1.33"


def test_resolve_ref_tag():
    assert _resolve_ref("v6.12.25", "tag") == "v6.12.25"


# ── get_tags ──────────────────────────────────────────────────────────────────

@patch("backport_scanner.triage._run")
def test_get_tags_no_pattern(mock_run):
    mock_run.return_value = (0, "v6.1.100\nv6.6.80\nv6.12.25\n", "")
    tags = get_tags(REPO)
    assert tags == ["v6.1.100", "v6.12.25", "v6.6.80"]


@patch("backport_scanner.triage._run")
def test_get_tags_with_pattern(mock_run):
    mock_run.return_value = (0, "v6.1.100\nv6.6.80\nv6.12.25\nv5.15.10\n", "")
    tags = get_tags(REPO, pattern="v6.*")
    assert tags == ["v6.1.100", "v6.12.25", "v6.6.80"]
    assert "v5.15.10" not in tags


@patch("backport_scanner.triage._run")
def test_get_tags_empty(mock_run):
    mock_run.return_value = (0, "", "")
    tags = get_tags(REPO, pattern="v99.*")
    assert tags == []


# ── check_ref with ref_type="tag" ─────────────────────────────────────────────

@patch("backport_scanner.triage.git_show_files", return_value=["net/core/dev.c"])
@patch("backport_scanner.triage.git_is_ancestor", return_value=True)
def test_check_ref_tag_exact_match(mock_ancestor, mock_files):
    result = check_ref(REPO, TAG, FIX, ref_type="tag")
    assert result.status == FixStatus.EXACT_MATCH
    assert result.ref_type == "tag"
    mock_ancestor.assert_called_once_with(str(REPO), FIX, TAG)


@patch("backport_scanner.triage.git_show_files", return_value=["net/core/dev.c"])
@patch("backport_scanner.triage.forward_patch_id_check", return_value="deadbeef1234")
@patch("backport_scanner.triage.git_is_ancestor", return_value=False)
def test_check_ref_tag_equivalent(mock_ancestor, mock_fwd, mock_files):
    result = check_ref(REPO, TAG, FIX, ref_type="tag")
    assert result.status == FixStatus.EQUIVALENT
    assert result.ref_type == "tag"
    mock_fwd.assert_called_once_with(str(REPO), FIX, TAG)


@patch("backport_scanner.triage.git_show_files", return_value=["net/core/dev.c"])
@patch("backport_scanner.triage.forward_patch_id_check", return_value=None)
@patch("backport_scanner.triage.git_is_ancestor", return_value=False)
def test_check_ref_tag_needs_fix(mock_ancestor, mock_fwd, mock_files):
    with patch("backport_scanner.triage._check_file_diff", return_value=FixStatus.NEEDS_FIX):
        result = check_ref(REPO, TAG, FIX, ref_type="tag")
    assert result.status == FixStatus.NEEDS_FIX
    assert result.ref_type == "tag"


@patch("backport_scanner.triage._run")
def test_file_diff_tag_ref(mock_run):
    """File diff heuristic uses tag name directly (no origin/ prefix)."""
    fix_diff = "+new_function()\n+another_line()"
    def side_effect(cmd, cwd, timeout=30):
        if "--unified=0" in cmd:
            return 0, fix_diff, ""
        cmd_str = " ".join(cmd)
        if f"{TAG}:foo.c" in cmd_str:
            return 0, "old_code()\n", ""
        return 1, "", "not found"
    mock_run.side_effect = side_effect
    result = _check_file_diff(REPO, TAG, FIX, ["foo.c"], ref_type="tag")
    assert result == FixStatus.NEEDS_FIX


def test_check_branch_is_alias_for_check_ref():
    """check_branch() should produce identical results to check_ref(ref_type='branch')."""
    with patch("backport_scanner.triage.git_show_files", return_value=["foo.go"]), \
         patch("backport_scanner.triage.git_is_ancestor", return_value=True):
        r1 = check_branch(REPO, BRANCH, FIX)
        r2 = check_ref(REPO, BRANCH, FIX, ref_type="branch")
    assert r1.status == r2.status == FixStatus.EXACT_MATCH
    assert r1.ref_type == "branch"
