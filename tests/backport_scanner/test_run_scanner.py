"""Tests for run_scanner.load_config() and generate_config modes."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from backport_scanner.scanner import ScanConfig
from backport_scanner.triage import FixStatus, BranchCheckResult


# ── load_config ────────────────────────────────────────────────────────────────

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "scan.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_load_config_basic(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg_file = _write_yaml(tmp_path, f"""
        repository:
          local_path: "{repo}"
        fix:
          commit: "abc123"
          description: "fix a bug"
        scan:
          branch_pattern: "release-1.*"
        llm:
          model: "claude-opus-4-6"
    """)
    from backport_scanner.run_scanner import load_config
    config = load_config(cfg_file)
    assert config.fix_commit == "abc123"
    assert config.fix_description == "fix a bug"
    assert config.branch_pattern == "release-1.*"
    assert config.vuln_commit is None


def test_load_config_with_vuln_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg_file = _write_yaml(tmp_path, f"""
        repository:
          local_path: "{repo}"
        fix:
          commit: "abc123"
          vuln_commit: "def456"
        scan:
          branches:
            - release-1.33
            - release-1.34
        llm:
          model: "claude-sonnet-4-6"
    """)
    from backport_scanner.run_scanner import load_config
    config = load_config(cfg_file)
    assert config.vuln_commit == "def456"
    assert config.branches == ["release-1.33", "release-1.34"]
    assert config.llm_model == "claude-sonnet-4-6"


def test_load_config_missing_repo(tmp_path):
    cfg_file = _write_yaml(tmp_path, """
        repository:
          local_path: "/nonexistent/path"
        fix:
          commit: "abc123"
        scan:
          branch_pattern: "release-*"
        llm:
          enabled: false
    """)
    from backport_scanner.run_scanner import load_config
    with pytest.raises(SystemExit):
        load_config(cfg_file)


# ── mode_verify ────────────────────────────────────────────────────────────────

def test_mode_verify_reports_still_needs(tmp_path, capsys):
    existing = tmp_path / "fix.yaml"
    existing.write_text(yaml.dump({
        "fix": {"commit": "abc123"},
        "branches": {"target": "origin/release-1.33"},
    }))

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = ScanConfig(repo_path=repo, fix_commit="abc123")

    needs_fix_result = BranchCheckResult(
        branch="release-1.33",
        status=FixStatus.NEEDS_FIX,
        method="git cherry",
        detail="not found",
        changed_files=[],
    )

    with patch("backport_scanner.generate_config.check_ref", return_value=needs_fix_result):
        from backport_scanner.generate_config import mode_verify
        mode_verify(cfg, existing)

    out = capsys.readouterr().out
    assert "Still needs fix" in out
    assert "release-1.33" in out


def test_mode_verify_reports_already_fixed(tmp_path, capsys):
    existing = tmp_path / "fix.yaml"
    existing.write_text(yaml.dump({
        "fix": {"commit": "abc123"},
        "branches": {"target": "origin/release-1.33"},
    }))

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = ScanConfig(repo_path=repo, fix_commit="abc123")

    fixed_result = BranchCheckResult(
        branch="release-1.33",
        status=FixStatus.EXACT_MATCH,
        method="git_is_ancestor",
        detail="already there",
        changed_files=[],
    )

    with patch("backport_scanner.generate_config.check_ref", return_value=fixed_result):
        from backport_scanner.generate_config import mode_verify
        mode_verify(cfg, existing)

    out = capsys.readouterr().out
    assert "Already fixed" in out
    assert "release-1.33" in out


# ── mode_add ──────────────────────────────────────────────────────────────────

def test_mode_add_appends_new_branch(tmp_path, capsys):
    existing = tmp_path / "fix.yaml"
    existing.write_text(yaml.dump({
        "fix": {"commit": "abc123"},
        "branches": {"source": "origin/main", "target": "origin/release-1.33"},
        "issue_id": "test",
    }))

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = ScanConfig(repo_path=repo, fix_commit="abc123", branch_pattern="release-1.*")

    new_branch_result = BranchCheckResult(
        branch="release-1.34",
        status=FixStatus.NEEDS_FIX,
        method="git cherry",
        detail="not found",
        changed_files=[],
    )

    mock_report = MagicMock()
    mock_report.needs_fix = [new_branch_result]
    mock_report.results = [new_branch_result]

    with patch("backport_scanner.generate_config.run_scan", return_value=mock_report):
        from backport_scanner.generate_config import mode_add
        mode_add(cfg, existing)

    updated = yaml.safe_load(existing.read_text())
    targets = updated["branches"]["targets"]
    assert "origin/release-1.33" in targets
    assert "origin/release-1.34" in targets

    out = capsys.readouterr().out
    assert "release-1.34" in out


def test_mode_add_no_new_branches(tmp_path, capsys):
    existing = tmp_path / "fix.yaml"
    existing.write_text(yaml.dump({
        "fix": {"commit": "abc123"},
        "branches": {"target": "origin/release-1.33"},
    }))

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = ScanConfig(repo_path=repo, fix_commit="abc123")

    already_covered = BranchCheckResult(
        branch="release-1.33",
        status=FixStatus.NEEDS_FIX,
        method="git cherry",
        detail="not found",
        changed_files=[],
    )

    mock_report = MagicMock()
    mock_report.needs_fix = [already_covered]
    mock_report.results = [already_covered]

    with patch("backport_scanner.generate_config.run_scan", return_value=mock_report):
        from backport_scanner.generate_config import mode_add
        mode_add(cfg, existing)

    out = capsys.readouterr().out
    assert "up to date" in out
