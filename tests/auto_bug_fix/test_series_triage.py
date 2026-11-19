# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Commit-series mode skips Phase 0.5 and derives its inputs deterministically."""

from __future__ import annotations

import subprocess

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    _merge_series_assessment,
    _series_changed_files,
    _series_skip_files,
    _series_triage_assessment,
)

SOURCE = "int main(void) { return 0; }\n"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd, ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write(repo, rel, content="x\n"):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build_series_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")

    _write(repo, "src/base.c", SOURCE)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base_sha = _sha(rp)

    _git(rp, "branch", "target", base_sha)

    # Series commit 1: code plus CI bookkeeping.
    _write(repo, "src/base.c", SOURCE + "int helper(void) { return 1; }\n")
    _write(repo, ".github/workflows/ci.yml", "on: push\n")
    _write(repo, "Changes", "- upstream entry\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "lib: refactor 1")
    first = _sha(rp)

    # Series commit 2: fuzz harness, docs and a test file.
    _write(repo, "src/base.c", SOURCE + "int helper(void) { return 2; }\n")
    _write(repo, "fuzz/parser.c", "int f(void) { return 0; }\n")
    _write(repo, "doc/reference.html", "<html/>\n")
    _write(repo, "tests/alloc_tests.c", "int t(void) { return 0; }\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "tests: cover helper")
    second = _sha(rp)

    return rp, [first, second]


def _make_config(rp, series):
    return BugFixConfig(
        repo_path=rp,
        build_dir=rp,
        source_branch="main",
        target_branch="target",
        source_fix_commit=series[-1],
        bug_description="series fix",
        issue_id="TEST-1",
        source_fix_commits=series,
    )


def test_changed_files_unions_the_series(tmp_path):
    rp, series = _build_series_repo(tmp_path)
    changed = _series_changed_files(_make_config(rp, series))

    assert changed == [
        ".github/workflows/ci.yml",
        "Changes",
        "doc/reference.html",
        "fuzz/parser.c",
        "src/base.c",
        "tests/alloc_tests.c",
    ]


def test_skip_files_excludes_code_and_tests(tmp_path):
    rp, series = _build_series_repo(tmp_path)
    skip = _series_skip_files(_make_config(rp, series))

    assert skip == [
        ".github/workflows/ci.yml",
        "Changes",
        "doc/reference.html",
        "fuzz/parser.c",
    ]
    assert "src/base.c" not in skip
    assert "tests/alloc_tests.c" not in skip


def test_assessment_fills_the_keys_later_phases_read(tmp_path):
    rp, series = _build_series_repo(tmp_path)
    config = _make_config(rp, series)
    state = PipelineState()
    state.allowed_modules = ["src/base.c", "tests/alloc_tests.c"]
    state.seed = ["src/base.c", "tests/alloc_tests.c"]

    assessment = _series_triage_assessment(config, state)

    assert assessment["mode"] == "commit_series_deterministic"
    assert assessment["files_to_skip"] == _series_skip_files(config)
    assert assessment["test_files"] == ["tests/alloc_tests.c"]
    assert assessment["priority_files"] == state.allowed_modules
    assert assessment["prerequisites"] == []
    assert assessment["estimated_difficulty"] == "unknown"


def test_merge_keeps_the_deterministic_floor_when_triage_returns_nothing():
    deterministic = {
        "files_to_skip": ["docs/x.md"],
        "test_files": ["tests/a.py"],
        "priority_files": ["src/a.c"],
        "estimated_difficulty": "unknown",
    }

    merged = _merge_series_assessment(deterministic, {})

    assert merged == deterministic


def test_merge_unions_skips_and_lets_triage_set_other_keys():
    deterministic = {
        "files_to_skip": ["docs/x.md"],
        "test_files": ["tests/a.py"],
        "priority_files": ["src/a.c"],
        "estimated_difficulty": "unknown",
        "recommendation": "series_mode",
    }
    llm = {
        "files_to_skip": ["fuzz/", "docs/x.md"],
        "priority_files": ["src/b.c"],
        "estimated_difficulty": "EXTREME",
        "recommendation": "MANUAL_REVIEW",
        "clusters": [{"id": "enablers", "commits": ["abc123"]}],
    }

    merged = _merge_series_assessment(deterministic, llm)

    assert merged["files_to_skip"] == ["docs/x.md", "fuzz/"]
    assert merged["test_files"] == ["tests/a.py"]
    assert merged["priority_files"] == ["src/b.c"]
    assert merged["estimated_difficulty"] == "EXTREME"
    assert merged["recommendation"] == "MANUAL_REVIEW"
    assert merged["clusters"] == [{"id": "enablers", "commits": ["abc123"]}]
