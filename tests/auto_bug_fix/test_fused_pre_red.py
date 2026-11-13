# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused pre-RED session (triage + prerequisites + test port)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


def _run_session(tmp_path, action=None, write_result=True, **result):
    """Run one fused pre-RED session; *action* edits the repo first."""
    rp, fix = _repo(tmp_path)
    state = _state()
    run_dir, tracker = _tracker(tmp_path)

    async def fake(claude_config, prompts, tracker=None):
        if action:
            action(rp, fix)
        if write_result:
            _write_result(
                run_dir,
                **{k: v(fix) if callable(v) else v for k, v in result.items()},
            )
        return []

    with patch("auto_bug_fix.run_bug_fix.claude_run", side_effect=fake):
        outcome = _run_fused_pre_red(_config(rp, fix), _claude(rp), state, tracker)
    return outcome, rp, fix, state


def test_fused_pre_red_reads_result_and_commits_session(tmp_path):
    def act(rp, _fix):
        (Path(rp) / "base_test.c").write_text("int t;\n")
        (Path(rp) / ".gocache").mkdir()
        (Path(rp) / ".gocache" / "x").write_text("cache\n")
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "test-port: TEST-1")

    outcome, rp, _, state = _run_session(
        tmp_path,
        act,
        tests_ported=["base_test.c"],
        prerequisites=[
            {
                "symbol": "ext",
                "introduced_by": "0" * 40,
                "carried_by_fix": False,
                "decision": "failed",
                "evidence": "external commit",
            }
        ],
    )

    assert outcome is not None and outcome["commit_type"] == "bug-fix"
    assert outcome["test_files"] == ["base_test.c"]
    assert outcome["prerequisites"][0]["decision"] == "failed"
    assert state.triage_assessment is outcome
    assert state.ported_test_files == ["base_test.c"]
    assert state.failed_prerequisites == ["ext"]
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "base_test.c" in tracked and ".gocache/x" not in tracked
    assert "pre_red_result.json" not in tracked


def _dirty_fix(rp, _fix):
    (Path(rp) / "base.c").write_text("int x = 2;\n")


def _cherry_pick_fix(rp, fix):
    _git(rp, "cherry-pick", fix)


def _stray_commit(rp, _fix):
    (Path(rp) / "base.c").write_text("int x = 3;\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "implement the fix")


def _carried_prereq(rp, fix):
    (Path(rp) / "prereq.c").write_text("int p;\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", f"prereq: {fix} x")


@pytest.mark.parametrize(
    "action,result",
    [
        (_dirty_fix, {"triage": {"test_files": []}, "tests_ported": []}),
        (_cherry_pick_fix, {}),
        (_stray_commit, {}),
        (
            _carried_prereq,
            {
                "prerequisites": lambda fix: [
                    {
                        "symbol": "x",
                        "introduced_by": fix,
                        "carried_by_fix": True,
                        "decision": "ported",
                        "evidence": "cherry-picked",
                    }
                ]
            },
        ),
    ],
    ids=["dirty-fix", "applied-fix", "stray-commit", "carried-prereq"],
)
def test_fused_pre_red_rejects_bad_session(action, result, tmp_path):
    outcome, rp, *_ = _run_session(tmp_path, action, **result)
    assert outcome is None
    target = subprocess.run(
        ["git", "rev-parse", "target"], cwd=rp, capture_output=True, text=True
    ).stdout.strip()
    assert _sha(rp) == target


def test_fused_pre_red_floor_and_adaptation(tmp_path):
    outcome, *_ = _run_session(tmp_path, tests_ported=[])
    assert outcome is None

    adapted = tmp_path / "adapted"
    adapted.mkdir()

    def act(rp, _fix):
        path = Path(rp) / "tests" / "test_adapted.py"
        path.parent.mkdir()
        path.write_text("def test_x():\n    pass\n")
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "test-port: TEST-1")

    outcome, _, _, state = _run_session(
        adapted, act, tests_ported=["tests/test_adapted.py"]
    )
    assert outcome is not None
    assert state.ported_test_files == ["tests/test_adapted.py"]


def test_fused_pre_red_falls_back_without_result(tmp_path):
    def act(rp, _fix):
        (Path(rp) / "base.c").write_text("int x = 99;\n")
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "prereq: oops")

    outcome, rp, *_ = _run_session(tmp_path, act, write_result=False)
    target = subprocess.run(
        ["git", "rev-parse", "target"], cwd=rp, capture_output=True, text=True
    ).stdout.strip()
    assert outcome is None and _sha(rp) == target


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


def test_strip_session_scratch(tmp_path):
    from auto_bug_fix.run_bug_fix import _strip_session_scratch

    rp, _fix = _repo(tmp_path)
    base = _sha(rp)
    for rel in (
        "regression.diffs",
        "contrib/fuzzystrmatch/regression.out",
        "pkg/__pycache__/mod.pyc",
        "web/.nyc_output/out.json",
        ".gocache/x",
    ):
        path = Path(rp) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("out\n")
    (Path(rp) / "expected.diffs").write_text("golden\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "session")

    dropped = _strip_session_scratch(rp, base, keep={"expected.diffs"})

    assert dropped is not None
    assert set(dropped) == {
        "regression.diffs",
        "contrib/fuzzystrmatch/regression.out",
        "pkg/__pycache__/mod.pyc",
        "web/.nyc_output/out.json",
        ".gocache/x",
    }
    files = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert files == ["expected.diffs"]
    assert not (Path(rp) / "regression.diffs").exists()
    assert (Path(rp) / "expected.diffs").exists()


def _scratch_commit(rp):
    (Path(rp) / ".gocache").mkdir()
    (Path(rp) / ".gocache" / "x").write_text("cache\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "session")


def test_strip_session_scratch_fails_closed(tmp_path):
    from auto_bug_fix import run_bug_fix as rbf

    for needle in (["reset", "--soft"], ["commit", "--no-verify"]):
        sub = tmp_path / needle[0].replace("--", "")
        sub.mkdir()
        rp, _fix = _repo(sub)
        base = _sha(rp)
        _scratch_commit(rp)
        real = rbf._run_git

        def fake(repo, args, needle=needle, real=real, **kwargs):
            if list(args[:2]) == needle:
                return MagicMock(success=False, stderr="boom", stdout="")
            return real(repo, args, **kwargs)

        with patch.object(rbf, "_run_git", side_effect=fake):
            assert rbf._strip_session_scratch(rp, base) is None


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
    from dataclasses import replace

    from auto_bug_fix.bug_fix_prompts import FusedPreRedPrompt

    base = FusedPreRedPrompt(
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
    )
    prompt = base.prompt()
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
    assert "Skip the build" in replace(base, skip_build=True).prompt()


def test_fused_check_commands_flattens_string_values():
    from auto_bug_fix.run_bug_fix import _fused_check_commands

    state = PipelineState()
    state.triage_assessment = {
        "lint_commands": "gofmt -l .",
        "verification_commands": ["make check"],
    }
    assert _fused_check_commands(state) == ["gofmt -l .", "make check"]


def test_compact_fix_overview_has_fetch_instructions(tmp_path):
    from auto_bug_fix.run_bug_fix import _compact_fix_overview

    rp, fix = _repo(tmp_path)
    overview = _compact_fix_overview(_config(rp, fix), fix)
    assert "git show" in overview
    assert "base.c" in overview
    assert len(overview) < 5000


def test_validate_fused_pre_red_checks_commit_paths(tmp_path):
    """A prefix is not evidence: test-port and leftovers commits are path-checked."""
    from auto_bug_fix.run_bug_fix import _validate_fused_pre_red

    rp, fix = _repo(tmp_path)
    config = _config(rp, fix)
    base = _sha(rp)

    Path(rp, "base.c").write_text("int x = 3;\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "test-port: TEST-1")
    problems = _validate_fused_pre_red(config, base)
    assert any("test-port commit changed non-test paths" in p for p in problems)

    _git(rp, "reset", "--hard", base)
    Path(rp, "tests").mkdir()
    Path(rp, "tests", "app_test.c").write_text("t\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "test-port: TEST-1")
    assert _validate_fused_pre_red(config, base) == []

    _git(rp, "reset", "--hard", base)
    Path(rp, "base.c").write_text("int x = 4;\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "fused-pre-red: leftovers")
    assert any(
        "leftovers commit changed unclaimed production paths" in p
        for p in _validate_fused_pre_red(config, base)
    )


def test_session_test_files_ignores_deletion(tmp_path):
    """Deleting a test must not satisfy the ported-test floor."""
    from auto_bug_fix.run_bug_fix import _session_test_files

    rp, _fix = _repo(tmp_path)
    test_file = Path(rp, "tests", "app_test.c")
    test_file.parent.mkdir()
    test_file.write_text("t\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "seed test")
    base = _sha(rp)
    test_file.unlink()
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "test-port: TEST-1")

    assert _session_test_files(rp, base) == []


def test_strip_keeps_tracked_scratch_named_file(tmp_path):
    """A tracked file under .gocache/ is content, not build debris."""
    from auto_bug_fix.run_bug_fix import _strip_session_scratch

    rp, _fix = _repo(tmp_path)
    cache = Path(rp, ".gocache", "x")
    cache.parent.mkdir()
    cache.write_text("old\n")
    _git(rp, "add", "-f", ".gocache/x")
    _git(rp, "commit", "-m", "tracked cache")
    base = _sha(rp)
    cache.write_text("old\nnew\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "session")

    assert _strip_session_scratch(rp, base) == []
    status = subprocess.run(
        ["git", "diff", "--name-status", base, "HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert status.startswith("M")


def test_fix_content_applied_counts_src_testing_as_production(tmp_path):
    """A production package named testing/ must not be treated as tests."""
    from auto_bug_fix.run_bug_fix import _fix_content_applied

    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    helper = repo / "src" / "testing" / "helpers.go"
    helper.parent.mkdir(parents=True)
    helper.write_text("v1\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)
    helper.write_text("v2\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix")
    fix = _sha(rp)
    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "target")
    base_head = _sha(rp)
    helper.write_text("v2\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "already carries the fix")

    assert _fix_content_applied(rp, base_head, fix) is True


def test_declared_regression_pair_counts_as_tests(tmp_path):
    """PostgreSQL-style sql/expected pairs have no test naming pattern."""
    from auto_bug_fix.run_bug_fix import _validate_fused_pre_red

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
    sql = repo / "contrib" / "x" / "sql" / "a.sql"
    sql.parent.mkdir(parents=True)
    sql.write_text("select 1;\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix")
    fix = _sha(rp)
    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "target")
    base_head = _sha(rp)
    sql.parent.mkdir(parents=True, exist_ok=True)
    sql.write_text("select 1;\n")
    _git(rp, "add", "-f", "contrib/x/sql/a.sql")
    _git(rp, "commit", "-m", "test-port: TEST-1")

    config = _config(rp, fix)
    assert any(
        "non-test paths" in p for p in _validate_fused_pre_red(config, base_head)
    )
    assert _validate_fused_pre_red(config, base_head, {"contrib/x/sql/a.sql"}) == []
