# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused pre-RED session (triage + prerequisites + test port)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.claude_utils import BugFixClaudeConfig
from auto_bug_fix.dossier import Dossier
from auto_bug_fix.run_bug_fix import PipelineState, _run_fused_pre_red
from auto_bug_fix.tracker import Tracker


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "base.c").write_text("int x = 1;\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)
    (repo / "base.c").write_text("int x = 2;\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix")
    fix = _sha(rp)
    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "target")
    return rp, fix


def _config(rp, fix, **kwargs):
    return BugFixConfig(
        repo_path=rp,
        build_dir=rp,
        source_branch="main",
        target_branch="target",
        source_fix_commit=fix,
        bug_description="test",
        issue_id="TEST-1",
        **kwargs,
    )


def _state():
    state = PipelineState()
    state.seed = ["base.c", "base_test.c"]
    state.dossier = Dossier(
        issue_id="TEST-1",
        source_branch="main",
        target_branch="target",
        fix_commit="x",
        bug_description="test",
    )
    return state


def _claude(rp):
    return BugFixClaudeConfig(
        model="test-model",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=rp,
    )


def _tracker(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return run_dir, Tracker(
        issue_id="TEST-1", config_dict={}, model="test-model", output_dir=str(run_dir)
    )


def _write_result(run_dir: Path, **overrides) -> dict:
    triage = {
        "commit_type": "bug-fix",
        "files_to_skip": [],
        "regeneration_commands": [],
        "lint_commands": [],
        "verification_commands": [],
        "estimated_difficulty": "EASY",
        "recommendation": "PROCEED",
        "priority_files": ["base.c"],
        "reasoning": "small fix",
    }
    triage.update(overrides.pop("triage", {}))
    result = {
        "triage": triage,
        "test_files": ["base_test.c"],
        "tests_ported": [],
        "prerequisites": [],
        "build": {"command": "make", "exit_code": 1},
    }
    result.update(overrides)
    (run_dir / "pre_red_result.json").write_text(json.dumps(result))
    return result


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_pre_red_reads_result_and_commits_session(mock_claude, tmp_path):
    rp, fix = _repo(tmp_path)
    state = _state()
    run_dir, tracker = _tracker(tmp_path)

    async def fake(claude_config, prompts, tracker=None):
        _write_result(
            run_dir,
            tests_ported=["base_test.c"],
            prerequisites=[
                {
                    "symbol": "ext",
                    "introduced_by": "0" * 40,
                    "carried_by_fix": False,
                    "decision": "failed",
                    "evidence": "imported from an external commit",
                }
            ],
        )
        (run_dir / "fused_pre_red_report.md").write_text("# report\n")
        # A test port commit the session made itself.
        (Path(rp) / "base_test.c").write_text("int t;\n")
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "test-port: TEST-1")
        return []

    mock_claude.side_effect = fake
    result = _run_fused_pre_red(_config(rp, fix), _claude(rp), state, tracker)

    assert result is not None
    assert result["commit_type"] == "bug-fix"
    assert result["test_files"] == ["base_test.c"]
    assert result["prerequisites"][0]["decision"] == "failed"
    assert state.triage_assessment is result
    assert state.ported_test_files == ["base_test.c"]
    assert state.failed_prerequisites == ["ext"]
    # Session artifacts never reach the repository or its commits.
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "pre_red_result.json" not in tracked
    assert "fused_pre_red_report.md" not in tracked
    assert "base_test.c" in tracked


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_pre_red_falls_back_without_result(mock_claude, tmp_path):
    rp, fix = _repo(tmp_path)
    state = _state()
    _, tracker = _tracker(tmp_path)
    before = _sha(rp)

    async def fake(claude_config, prompts, tracker=None):
        (Path(rp) / "base.c").write_text("int x = 99;\n")
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "prereq: oops")
        return []

    mock_claude.side_effect = fake
    result = _run_fused_pre_red(_config(rp, fix), _claude(rp), state, tracker)

    assert result is None
    assert _sha(rp) == before


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_pre_red_rejects_applied_fix(mock_claude, tmp_path):
    rp, fix = _repo(tmp_path)
    state = _state()
    run_dir, tracker = _tracker(tmp_path)
    before = _sha(rp)

    async def fake(claude_config, prompts, tracker=None):
        _write_result(run_dir)
        _git(rp, "cherry-pick", fix)
        return []

    mock_claude.side_effect = fake
    result = _run_fused_pre_red(_config(rp, fix), _claude(rp), state, tracker)

    assert result is None
    assert _sha(rp) == before


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_pre_red_rejects_stray_commit(mock_claude, tmp_path):
    rp, fix = _repo(tmp_path)
    state = _state()
    run_dir, tracker = _tracker(tmp_path)

    async def fake(claude_config, prompts, tracker=None):
        _write_result(run_dir)
        (Path(rp) / "base.c").write_text("int x = 3;\n")
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "implement the fix")
        return []

    mock_claude.side_effect = fake
    result = _run_fused_pre_red(_config(rp, fix), _claude(rp), state, tracker)

    assert result is None


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_pre_red_rejects_ported_fix_ancestor(mock_claude, tmp_path):
    rp, fix = _repo(tmp_path)
    state = _state()
    run_dir, tracker = _tracker(tmp_path)

    async def fake(claude_config, prompts, tracker=None):
        _write_result(
            run_dir,
            prerequisites=[
                {
                    "symbol": "x",
                    "introduced_by": fix,
                    "carried_by_fix": True,
                    "decision": "ported",
                    "evidence": "cherry-picked",
                }
            ],
        )
        (Path(rp) / "prereq.c").write_text("int p;\n")
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", f"prereq: {fix} x")
        return []

    mock_claude.side_effect = fake
    result = _run_fused_pre_red(_config(rp, fix), _claude(rp), state, tracker)

    assert result is None


def _merge_repo(tmp_path):
    """mainline M1 + side S1 merged as fix F: F^1 = M1, S1 is carried by F."""
    repo = tmp_path / "merge-repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "base.c").write_text("base\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)
    (repo / "mainline.c").write_text("mainline\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "mainline")
    mainline = _sha(rp)
    _git(rp, "checkout", "-q", "-b", "side", base)
    (repo / "side.c").write_text("side\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "side")
    side = _sha(rp)
    _git(rp, "checkout", "-q", "main")
    _git(rp, "merge", "--no-ff", "-m", "fix", "side")
    fix = _sha(rp)
    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "target")
    return rp, fix, mainline, side


def test_fused_pre_red_allows_mainline_prereq_rejects_carried_one(tmp_path):
    from auto_bug_fix.run_bug_fix import _validate_fused_pre_red

    rp, fix, mainline, side = _merge_repo(tmp_path)
    config = _config(rp, fix)
    base = _sha(rp)
    (Path(rp) / "mainline_port.c").write_text("m\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", f"prereq: {mainline} mainline")
    # Ancestor of both the fix and its first parent: mainline, may be ported.
    assert _validate_fused_pre_red(config, base) == []
    # Reachable from the fix but not from fix^1: carried by the fix diff.
    (Path(rp) / "side_port.c").write_text("s\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", f"prereq: {side} side")
    problems = _validate_fused_pre_red(config, base)
    assert any(side[:12] in p for p in problems)


def test_restore_protected_refs_undoes_branch_move(tmp_path):
    from auto_bug_fix.run_bug_fix import _protected_refs, _restore_protected_refs

    rp, fix = _repo(tmp_path)  # HEAD is attached to the target branch
    state = _state()
    state.protected_refs = _protected_refs(rp, ("target",))
    assert state.protected_refs
    original = subprocess.run(
        ["git", "rev-parse", "target"], cwd=rp, capture_output=True, text=True
    ).stdout.strip()

    (Path(rp) / "session.c").write_text("session\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "session commit")
    moved = _sha(rp)
    assert moved != original

    _restore_protected_refs(_config(rp, fix), state)

    restored = subprocess.run(
        ["git", "rev-parse", "target"], cwd=rp, capture_output=True, text=True
    ).stdout.strip()
    assert restored == original
    # HEAD keeps the session's commit, detached, with the work preserved.
    assert _sha(rp) == moved
    assert (Path(rp) / "session.c").exists()


def test_strip_session_scratch_drops_test_output_keeps_fix_files(tmp_path):
    from auto_bug_fix.run_bug_fix import _strip_session_scratch

    rp, _fix = _repo(tmp_path)
    base = _sha(rp)
    (Path(rp) / "regression.diffs").write_text("out\n")
    sub = Path(rp) / "contrib" / "fuzzystrmatch"
    sub.mkdir(parents=True)
    (sub / "regression.out").write_text("out\n")
    (Path(rp) / "expected.diffs").write_text("golden\n")
    scratch = Path(rp) / ".gocache" / "x"
    scratch.parent.mkdir()
    scratch.write_text("cache\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "session")

    dropped = _strip_session_scratch(rp, base, keep={"expected.diffs"})

    assert set(dropped) == {
        "regression.diffs",
        "contrib/fuzzystrmatch/regression.out",
        ".gocache/x",
    }
    files = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert files == ["expected.diffs"]


def test_pre_red_facts_classify_paths(tmp_path):
    from auto_bug_fix.run_bug_fix import _pre_red_facts

    repo = tmp_path / "facts-repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "base.c").write_text("1\n")
    (repo / "same.c").write_text("s\n")
    (repo / "moved.c").write_text("m\n")
    (repo / "gone.c").write_text("g\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)
    (repo / "base.c").write_text("2\n")
    (repo / "same.c").write_text("s2\n")
    (repo / "moved.c").write_text("m2\n")
    (repo / "gone.c").write_text("g2\n")
    (repo / "new.c").write_text("n\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix")
    fix = _sha(rp)
    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "target")
    (repo / "base.c").write_text("t\n")
    (repo / "moved.c").unlink()
    (repo / "gone.c").unlink()
    (repo / "other").mkdir()
    (repo / "other" / "moved.c").write_text("m\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "target divergence")

    facts, needs_build = _pre_red_facts(rp, "target", fix)

    assert "single commit" in facts
    assert "A absent" in facts and "new.c" in facts
    assert "M diverged" in facts and "base.c" in facts
    assert "M same" in facts and "same.c" in facts
    assert "M absent" in facts and "moved.c" in facts
    assert "same name on target: other/moved.c" in facts
    # gone.c has no target equivalent: a genuine missing-package candidate.
    assert needs_build is True


def test_fused_pre_red_prompt_states_all_tasks():
    from auto_bug_fix.bug_fix_prompts import FusedPreRedPrompt

    prompt = FusedPreRedPrompt(
        context="ctx",
        fix_diff="diff",
        source_branch="main",
        target_branch="target",
        fix_commit="deadbeef",
        base_head="base",
        build_command="make",
        test_command="make test",
        seed_files=["a.go"],
        result_path="/tmp/result.json",
        report_path="/tmp/report.md",
        facts="M absent internal/linux/linux.go",
    ).prompt()
    for phrase in (
        "SCOPE",
        "TESTS",
        "PREREQUISITES",
        "BOUNDARY",
        "RED VALIDITY",
        "RECORD",
        "git merge-base --is-ancestor",
        "deadbeef^1",
        "/tmp/result.json",
        "never check out",
        "Deterministic patch facts",
        "M absent internal/linux/linux.go",
        "need no prerequisite inspection",
        "At most 10 candidates",
        "Stop by ~40",
        "Run the build",
        "never port a package rename",
    ):
        assert phrase in prompt
    skipped = FusedPreRedPrompt(
        context="ctx",
        fix_diff="diff",
        source_branch="main",
        target_branch="target",
        fix_commit="deadbeef",
        base_head="base",
        build_command="make",
        test_command="make test",
        seed_files=["a.go"],
        result_path="/tmp/result.json",
        report_path="/tmp/report.md",
        facts="(none)",
        skip_build=True,
    ).prompt()
    assert "Skip the build" in skipped


def test_compact_fix_overview_has_fetch_instructions(tmp_path):
    from auto_bug_fix.run_bug_fix import _compact_fix_overview

    rp, fix = _repo(tmp_path)
    overview = _compact_fix_overview(_config(rp, fix), fix)
    assert "git show" in overview
    assert "base.c" in overview
    assert len(overview) < 5000
