# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/check_pr_metadata.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_pr_metadata.py"


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")
    (path / "file").write_text("base\n")
    _git(path, "add", "file")
    _git(path, "commit", "-qm", "chore: base")
    _git(path, "checkout", "-qb", "head")
    return path


def test_dco_accepts_matching_signoff(repo: Path) -> None:
    _git(repo, "commit", "--allow-empty", "-s", "-m", "feat: signed")
    result = _run(["dco", "--base-ref", "main", "--head-ref", "head"], repo)
    assert result.returncode == 0, result.stderr
    assert "Signed-off-by" in result.stdout


def test_dco_rejects_missing_signoff(repo: Path) -> None:
    _git(repo, "commit", "--allow-empty", "-m", "feat: unsigned")
    result = _run(["dco", "--base-ref", "main", "--head-ref", "head"], repo)
    assert result.returncode == 1
    assert "missing a Signed-off-by" in result.stderr


def test_dco_rejects_mismatched_author(repo: Path) -> None:
    _git(
        repo,
        "commit",
        "--allow-empty",
        "--author",
        "Other <other@example.com>",
        "-s",
        "-m",
        "fix: mismatch",
    )
    result = _run(["dco", "--base-ref", "main", "--head-ref", "head"], repo)
    assert result.returncode == 1
    assert "Other <other@example.com>" in result.stderr


def test_dco_email_is_case_insensitive(repo: Path) -> None:
    _git(
        repo,
        "commit",
        "--allow-empty",
        "--author",
        "Test User <TEST@EXAMPLE.COM>",
        "-s",
        "-m",
        "fix: case",
    )
    result = _run(["dco", "--base-ref", "main", "--head-ref", "head"], repo)
    assert result.returncode == 0, result.stderr


def test_dco_skips_bot_authors(repo: Path) -> None:
    _git(repo, "commit", "--allow-empty", "-m", "feat: unsigned")
    result = _run(
        [
            "dco",
            "--base-ref",
            "main",
            "--head-ref",
            "head",
            "--author",
            "dependabot[bot]",
        ],
        repo,
    )
    assert result.returncode == 0, result.stderr
    assert "Skipping" in result.stdout


def test_dco_fails_closed_on_empty_range(repo: Path) -> None:
    result = _run(["dco", "--base-ref", "main", "--head-ref", "main"], repo)
    assert result.returncode == 1
    assert "no non-merge commits" in result.stderr


def test_dco_fails_closed_on_missing_ref(repo: Path) -> None:
    result = _run(["dco", "--base-ref", "missing", "--head-ref", "head"], repo)
    assert result.returncode == 1
    assert "cannot resolve" in result.stderr
    assert "missing" in result.stderr


def test_dco_checks_every_non_merge_commit(repo: Path) -> None:
    _git(repo, "commit", "--allow-empty", "-s", "-m", "feat: signed")
    _git(repo, "commit", "--allow-empty", "-m", "fix: unsigned")
    result = _run(["dco", "--base-ref", "main", "--head-ref", "head"], repo)
    assert result.returncode == 1
    assert "fix: unsigned" in result.stderr


def test_dco_accepts_the_fetched_pr_head_ref(repo: Path) -> None:
    _git(repo, "commit", "--allow-empty", "-s", "-m", "feat: signed")
    _git(repo, "update-ref", "refs/remotes/origin/pr-head", "head")
    _git(repo, "checkout", "-q", "main")
    result = _run(
        [
            "dco",
            "--base-ref",
            "HEAD",
            "--head-ref",
            "refs/remotes/origin/pr-head",
        ],
        repo,
    )
    assert result.returncode == 0, result.stderr
    assert "Signed-off-by" in result.stdout


@pytest.mark.parametrize(
    "title",
    [
        "feat: add a thing",
        "fix(scope): repair it",
        "chore!: drop the old path",
        "docs: describe the workflow",
        "test: cover the parser",
        "fix(scope)!: break compatibility",
    ],
)
def test_title_accepts_supported_forms(title: str, tmp_path: Path) -> None:
    result = _run(["title", "--title", title, "--author", "user"], tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "title",
    [
        "",
        "Feat: capitalised",
        "revert: undo it",
        "feat: ",
        "feat:  ",
        "fix(scope):missing space",
        "fix(): empty scope",
        "features: not a type",
    ],
)
def test_title_rejects_unsupported_forms(title: str, tmp_path: Path) -> None:
    result = _run(["title", "--title", title, "--author", "user"], tmp_path)
    assert result.returncode == 1
    assert "PR title must" in result.stderr


def test_title_skips_bot_authors(tmp_path: Path) -> None:
    result = _run(
        ["title", "--title", "Bump dependency", "--author", "renovate[bot]"],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "Skipping" in result.stdout
