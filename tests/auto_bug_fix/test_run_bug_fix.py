"""Tests for auto_bug_fix.run_bug_fix — pipeline state, strategies, and helpers."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    PipelineStop,
    PipelineEscalation,
    identify_test_files,
    _resolve_lint_command,
    _run_build_and_test,
    TargetResult,
    run_targets,
    _read_tracker_cost,
    _sanitize_branch_slug,
    _COAUTHOR_TRAILERS,
)
from auto_bug_fix.config_loader import TargetSpec, _sanitize_ref
from auto_bug_fix.dossier import Dossier
from auto_bug_fix.baseline import TIMEOUT_EXIT_CODE


def test_pipeline_state_defaults():
    state = PipelineState()
    assert state.seed == []
    assert state.allowed_modules == []
    assert state.baseline == set()
    assert state.ported_test_files == []
    assert state.cherry_pick_path == ""
    assert state.dossier is None
    assert state.triage_assessment == {}
    assert state.priority_files == []


def test_pipeline_stop_exception():
    try:
        raise PipelineStop("fix already present")
    except PipelineStop as exc:
        assert "fix already present" in str(exc)


def test_pipeline_escalation_exception():
    try:
        raise PipelineEscalation("human intervention required")
    except PipelineEscalation as exc:
        assert "human intervention required" in str(exc)



def test_identify_test_files_go():
    files = [
        "pkg/controller/replicaset/replica_set.go",
        "pkg/controller/replicaset/replica_set_test.go",
        "pkg/controller/controller_utils.go",
        "pkg/controller/controller_utils_test.go",
    ]
    result = identify_test_files(files)
    assert "pkg/controller/replicaset/replica_set_test.go" in result
    assert "pkg/controller/controller_utils_test.go" in result
    assert "pkg/controller/replicaset/replica_set.go" not in result
    assert len(result) == 2


def test_identify_test_files_python():
    files = [
        "src/main.py",
        "tests/test_main.py",
        "tests/conftest.py",
        "src/utils.py",
    ]
    result = identify_test_files(files)
    assert "tests/test_main.py" in result
    assert "tests/conftest.py" in result
    assert "src/main.py" not in result


def test_identify_test_files_none():
    files = ["lib/url.c", "lib/parser.c", "include/header.h"]
    result = identify_test_files(files)
    assert result == []


def test_identify_test_files_spec():
    files = ["app/models/user.rb", "spec/models/user_spec.rb"]
    result = identify_test_files(files)
    assert "spec/models/user_spec.rb" in result
    assert len(result) == 1


def test_identify_test_files_no_false_positives():
    files = ["latest/foo.c", "contest.md", "lib/attestation.go", "pkg/protesting.h"]
    result = identify_test_files(files)
    assert result == []


def _make_config(**overrides):
    defaults = dict(
        repo_path="/tmp/repo",
        build_dir="/tmp/repo",
        source_branch="main",
        target_branch="target",
        source_fix_commit="abc123",
        bug_description="test",
        issue_id="test-1",
        build_command="make",
        test_command="make test",
    )
    defaults.update(overrides)
    return BugFixConfig(**defaults)


# --- _resolve_lint_command tests ---

def test_resolve_lint_yaml_takes_priority():
    config = _make_config(lint_command="make lint")
    state = PipelineState()
    state.triage_assessment = {"lint_commands": ["go vet ./..."]}
    assert _resolve_lint_command(config, state) == "make lint"


def test_resolve_lint_falls_back_to_triage():
    config = _make_config(lint_command="")
    state = PipelineState()
    state.triage_assessment = {"lint_commands": ["go vet ./..."]}
    assert _resolve_lint_command(config, state) == "go vet ./..."


def test_resolve_lint_empty_when_neither():
    config = _make_config(lint_command="")
    state = PipelineState()
    state.triage_assessment = {}
    assert _resolve_lint_command(config, state) == ""


def test_resolve_lint_skips_dangerous_triage_commands():
    config = _make_config(lint_command="")
    state = PipelineState()
    state.triage_assessment = {"lint_commands": ["go vet ./... && rm -rf /"]}
    assert _resolve_lint_command(config, state) == ""


def test_resolve_lint_skips_if_same_as_test_command():
    config = _make_config(lint_command="", test_command="go vet ./...")
    state = PipelineState()
    state.triage_assessment = {"lint_commands": ["go vet ./..."]}
    assert _resolve_lint_command(config, state) == ""


# --- _run_build_and_test lint integration tests ---

@patch("auto_bug_fix.run_bug_fix.enforce_test_file_veto", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.enforce_allowlist", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.run_test_suite")
def test_lint_pass_means_verify_pass(mock_run, mock_allow, mock_veto):
    mock_run.side_effect = [
        (0, "", ""),       # build
        (0, "", ""),       # test
        (0, "", ""),       # lint
    ]
    config = _make_config(lint_command="go vet ./...")
    state = PipelineState()
    passed, _, _, _, _ = _run_build_and_test(config, state)
    assert passed is True
    assert mock_run.call_count == 3


@patch("auto_bug_fix.run_bug_fix.enforce_test_file_veto", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.enforce_allowlist", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.run_test_suite")
def test_lint_fail_means_verify_fail(mock_run, mock_allow, mock_veto):
    mock_run.side_effect = [
        (0, "", ""),                          # build
        (0, "", ""),                          # test
        (1, "unused import", "lint error"),   # lint fails
    ]
    config = _make_config(lint_command="go vet ./...")
    state = PipelineState()
    passed, _, _, test_out, test_err = _run_build_and_test(config, state)
    assert passed is False
    assert "unused import" in test_out


@patch("auto_bug_fix.run_bug_fix.enforce_test_file_veto", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.enforce_allowlist", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.run_test_suite")
def test_lint_fail_preexisting_means_verify_pass(mock_run, mock_allow, mock_veto):
    mock_run.side_effect = [
        (0, "", ""),                          # build
        (0, "", ""),                          # test
        (1, "unused import", "lint error"),   # lint fails
    ]
    config = _make_config(lint_command="go vet ./...")
    state = PipelineState()
    state.baseline_verification = [{"cmd": "go vet ./...", "exit_code": 1, "output": "same error"}]
    passed, _, _, _, _ = _run_build_and_test(config, state)
    assert passed is True


@patch("auto_bug_fix.run_bug_fix.enforce_test_file_veto", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.enforce_allowlist", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.run_test_suite")
def test_no_lint_command_skips_lint(mock_run, mock_allow, mock_veto):
    mock_run.side_effect = [
        (0, "", ""),   # build
        (0, "", ""),   # test
    ]
    config = _make_config(lint_command="")
    state = PipelineState()
    passed, _, _, _, _ = _run_build_and_test(config, state)
    assert passed is True
    assert mock_run.call_count == 2


# --- auto_discover_tests tests ---

def test_auto_discover_false_excludes_verification_commands():
    config = _make_config(auto_discover_tests=False)
    state = PipelineState()
    state.triage_assessment = {
        "lint_commands": ["go vet ./..."],
        "verification_commands": ["go test ./server/...", "go test ./pkg/..."],
    }
    from auto_bug_fix.run_bug_fix import _validate_shell_command
    lint_cmds = [c for c in state.triage_assessment.get("lint_commands", []) if _validate_shell_command(c)]
    verification_cmds = list(lint_cmds)
    if config.auto_discover_tests:
        for c in state.triage_assessment.get("verification_commands", []):
            if _validate_shell_command(c):
                verification_cmds.append(c)
    assert verification_cmds == ["go vet ./..."]
    assert "go test ./server/..." not in verification_cmds


def test_auto_discover_true_includes_verification_commands():
    config = _make_config(auto_discover_tests=True)
    state = PipelineState()
    state.triage_assessment = {
        "lint_commands": ["go vet ./..."],
        "verification_commands": ["go test ./server/...", "go test ./pkg/..."],
    }
    from auto_bug_fix.run_bug_fix import _validate_shell_command
    lint_cmds = [c for c in state.triage_assessment.get("lint_commands", []) if _validate_shell_command(c)]
    verification_cmds = list(lint_cmds)
    if config.auto_discover_tests:
        for c in state.triage_assessment.get("verification_commands", []):
            if _validate_shell_command(c):
                verification_cmds.append(c)
    assert verification_cmds == ["go vet ./...", "go test ./server/...", "go test ./pkg/..."]


def test_auto_discover_default_is_false():
    config = _make_config()
    assert config.auto_discover_tests is False


# --- timeout-as-hard-failure tests ---

@patch("auto_bug_fix.run_bug_fix.enforce_test_file_veto", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.enforce_allowlist", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.run_test_suite")
def test_test_timeout_means_verify_fail(mock_run, mock_allow, mock_veto):
    mock_run.side_effect = [
        (0, "", ""),                                   # build
        (TIMEOUT_EXIT_CODE, "partial", "timed out"),   # test times out
    ]
    config = _make_config()
    state = PipelineState()
    passed, _, _, _, _ = _run_build_and_test(config, state)
    assert passed is False
    assert mock_run.call_count == 2


@patch("auto_bug_fix.run_bug_fix.enforce_test_file_veto", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.enforce_allowlist", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.run_test_suite")
def test_lint_timeout_not_masked_as_preexisting(mock_run, mock_allow, mock_veto):
    mock_run.side_effect = [
        (0, "", ""),                            # build
        (0, "", ""),                            # test
        (TIMEOUT_EXIT_CODE, "", "timed out"),   # lint times out
    ]
    config = _make_config(lint_command="go vet ./...")
    state = PipelineState()
    state.baseline_verification = [
        {"cmd": "go vet ./...", "exit_code": TIMEOUT_EXIT_CODE, "output": "also timed out"}
    ]
    passed, _, _, _, _ = _run_build_and_test(config, state)
    assert passed is False


# --- _sanitize_ref tests (function lives in config_loader, imported here) ---

def test_sanitize_ref():
    assert _sanitize_ref("origin/release-1.34") == "release-1.34"
    assert _sanitize_ref("release/1.33") == "release_1.33"
    assert _sanitize_ref("cc8e8609e") == "cc8e8609e"
    assert _sanitize_ref("origin/feature/deep") == "feature_deep"


# --- run_targets tests ---

def _make_tracker_json(output_dir, issue_id, cost_usd=1.23):
    """Write a fake tracker JSON — reproduces run_pipeline's finally side effect."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{issue_id}_20260701_000000.json"
    path = os.path.join(output_dir, filename)
    data = {
        "run_id": "fake-run-id",
        "issue_id": issue_id,
        "outcome": "success",
        "total_cost_usd": cost_usd,
        "total_input_tokens": 1000,
        "total_output_tokens": 500,
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _make_dossier(issue_id="test-1"):
    return Dossier(
        issue_id=issue_id,
        source_branch="main",
        target_branch="target",
        fix_commit="abc123",
        bug_description="test fix",
    )


_BASE_CFG = dict(
    repo_path="/tmp/repo",
    build_dir="/tmp/repo",
    source_branch="main",
    target_branch="placeholder",
    source_fix_commit="abc123",
    bug_description="test fix",
    issue_id="test-1",
    build_command="make",
    test_command="make test",
)


@patch("auto_bug_fix.run_bug_fix.cleanup_worktree")
@patch("auto_bug_fix.run_bug_fix.create_fresh_worktree")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_continues_on_failure(mock_pipeline, mock_wt_add, mock_cleanup, tmp_path):
    """Headline test: failure on target 1 must not prevent target 2 from running."""
    base_config = BugFixConfig(**_BASE_CFG)

    call_count = [0]

    def pipeline_side_effect(config, claude_config, output_dir):
        call_count[0] += 1
        # Both targets write a tracker JSON (run_pipeline's finally does this)
        _make_tracker_json(output_dir, config.issue_id, cost_usd=1.50 * call_count[0])
        if call_count[0] == 1:
            raise PipelineEscalation("build failed on target 1")
        return _make_dossier(config.issue_id), MagicMock(outcome="success")

    mock_pipeline.side_effect = pipeline_side_effect
    mock_wt_add.return_value = None

    targets = [
        TargetSpec(branch="branch-a", issue_id="fix-a"),
        TargetSpec(branch="branch-b", issue_id="fix-b"),
    ]
    claude_cfg = MagicMock()
    parent_run_dir = str(tmp_path / "runs" / "test_run")
    os.makedirs(parent_run_dir, exist_ok=True)

    results = run_targets(
        targets, base_config, claude_cfg,
        source_path="/tmp/source", workdir=str(tmp_path / "workdir"),
        parent_run_dir=parent_run_dir, build_subdir="",
    )

    assert len(results) == 2
    assert results[0].outcome == "pipeline_escalation"
    assert results[0].issue_id == "fix-a"
    assert "build failed" in results[0].error
    assert results[0].cost_usd == 1.50

    assert results[1].outcome == "success"
    assert results[1].issue_id == "fix-b"
    assert results[1].cost_usd == 3.00

    summary_path = os.path.join(parent_run_dir, "summary.json")
    assert os.path.exists(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary["total_targets"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1


@patch("auto_bug_fix.run_bug_fix.cleanup_worktree")
@patch("auto_bug_fix.run_bug_fix.create_fresh_worktree")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_single_target(mock_pipeline, mock_wt_add, mock_cleanup, tmp_path):
    """Single-element list produces correct directory structure."""
    base_config = BugFixConfig(**_BASE_CFG)

    def pipeline_side_effect(config, claude_config, output_dir):
        _make_tracker_json(output_dir, config.issue_id, cost_usd=2.00)
        return _make_dossier(config.issue_id), MagicMock(outcome="success")

    mock_pipeline.side_effect = pipeline_side_effect
    mock_wt_add.return_value = None

    targets = [TargetSpec(branch="release-1.34")]
    parent_run_dir = str(tmp_path / "runs" / "single")
    os.makedirs(parent_run_dir, exist_ok=True)

    results = run_targets(
        targets, base_config, MagicMock(),
        source_path="/tmp/source", workdir=str(tmp_path / "workdir"),
        parent_run_dir=parent_run_dir, build_subdir="",
    )

    assert len(results) == 1
    assert results[0].outcome == "success"
    target_subdir = os.path.join(parent_run_dir, "release-1.34")
    assert os.path.isdir(target_subdir)
    assert os.path.exists(os.path.join(target_subdir, "run_log.txt"))
    assert os.path.exists(os.path.join(target_subdir, "dossier.md"))


@patch("auto_bug_fix.run_bug_fix.cleanup_worktree")
@patch("auto_bug_fix.run_bug_fix.create_fresh_worktree")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_cost_recovery_on_failure(mock_pipeline, mock_wt_add, mock_cleanup, tmp_path):
    """TargetResult.cost_usd is populated from tracker JSON even when run_pipeline raises."""
    base_config = BugFixConfig(**_BASE_CFG)

    def pipeline_side_effect(config, claude_config, output_dir):
        _make_tracker_json(output_dir, config.issue_id, cost_usd=4.56)
        raise PipelineStop("already fixed")

    mock_pipeline.side_effect = pipeline_side_effect
    mock_wt_add.return_value = None

    targets = [TargetSpec(branch="target-1")]
    parent_run_dir = str(tmp_path / "runs" / "cost")
    os.makedirs(parent_run_dir, exist_ok=True)

    results = run_targets(
        targets, base_config, MagicMock(),
        source_path="/tmp/source", workdir=str(tmp_path / "workdir"),
        parent_run_dir=parent_run_dir, build_subdir="",
    )

    assert len(results) == 1
    assert results[0].outcome == "pipeline_stop"
    assert results[0].cost_usd == 4.56
    assert results[0].tracker_path != ""


@patch("auto_bug_fix.run_bug_fix.cleanup_worktree")
@patch("auto_bug_fix.run_bug_fix.create_fresh_worktree")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_worktree_failure_continues(mock_pipeline, mock_wt_add, mock_cleanup, tmp_path):
    """Worktree failure on target 1 records worktree_error and target 2 still runs."""
    base_config = BugFixConfig(**_BASE_CFG)

    call_count = [0]

    def wt_add_side_effect(source_path, worktree_path, ref):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("git worktree add failed: fatal: invalid ref: bad-branch")

    mock_wt_add.side_effect = wt_add_side_effect

    def pipeline_side_effect(config, claude_config, output_dir):
        _make_tracker_json(output_dir, config.issue_id, cost_usd=3.21)
        return _make_dossier(config.issue_id), MagicMock(outcome="success")

    mock_pipeline.side_effect = pipeline_side_effect

    targets = [
        TargetSpec(branch="bad-branch", issue_id="fix-bad"),
        TargetSpec(branch="good-branch", issue_id="fix-good"),
    ]
    parent_run_dir = str(tmp_path / "runs" / "wt_fail")
    os.makedirs(parent_run_dir, exist_ok=True)

    results = run_targets(
        targets, base_config, MagicMock(),
        source_path="/tmp/source", workdir=str(tmp_path / "workdir"),
        parent_run_dir=parent_run_dir, build_subdir="",
    )

    assert len(results) == 2

    assert results[0].outcome == "worktree_error"
    assert results[0].issue_id == "fix-bad"
    assert results[0].cost_usd == 0.0
    assert results[0].tracker_path == ""
    assert results[0].dossier_path == ""
    assert "fatal: invalid ref" in results[0].error

    assert results[1].outcome == "success"
    assert results[1].issue_id == "fix-good"
    assert results[1].cost_usd == 3.21

    assert mock_pipeline.call_count == 1


# --- Phase 1.5 RED triage tests ---

from auto_bug_fix.run_bug_fix import phase_1_5_red_triage
from auto_bug_fix.bug_fix_prompts import RED_TRIAGE_FILE
from auto_bug_fix.claude_utils import BugFixClaudeConfig


def _make_claude_config():
    return BugFixClaudeConfig(
        model="test-model",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/repo",
    )


def _make_state_with_red_pass():
    state = PipelineState()
    state.red_passed_output = "PASSED tests/test_security.py::test_attr_filter"
    state.ported_test_files = ["tests/test_security.py"]
    state.triage_assessment = {"commit_type": "security", "estimated_difficulty": "MEDIUM"}
    return state


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_red_triage_not_vulnerable(mock_claude, tmp_path):
    config = _make_config(repo_path=str(tmp_path), build_dir=str(tmp_path))
    state = _make_state_with_red_pass()
    triage_json = {
        "verdict": "not_vulnerable",
        "reasoning": "The sandbox blocks this vector via existing SandboxedFormatter",
        "test_analysis": "Test hits call() path which is already protected",
        "replacement_test_file": None,
    }
    triage_file = tmp_path / RED_TRIAGE_FILE
    triage_file.write_text(json.dumps(triage_json))

    result = phase_1_5_red_triage(config, _make_claude_config(), state, None)
    assert result["verdict"] == "not_vulnerable"
    assert mock_claude.called


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_red_triage_vulnerable_with_replacement(mock_claude, tmp_path):
    config = _make_config(repo_path=str(tmp_path), build_dir=str(tmp_path))
    state = _make_state_with_red_pass()
    replacement_path = str(tmp_path / "tests" / "test_new.py")
    os.makedirs(os.path.dirname(replacement_path), exist_ok=True)
    with open(replacement_path, "w") as f:
        f.write("def test_real_vuln(): assert False")

    triage_json = {
        "verdict": "vulnerable",
        "reasoning": "The |attr filter bypasses call() via custom filter path",
        "test_analysis": "Ported test hits call() but real vector bypasses it",
        "replacement_test_file": replacement_path,
    }
    triage_file = tmp_path / RED_TRIAGE_FILE
    triage_file.write_text(json.dumps(triage_json))

    result = phase_1_5_red_triage(config, _make_claude_config(), state, None)
    assert result["verdict"] == "vulnerable"
    assert result["replacement_test_file"] == replacement_path


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_red_triage_mitigated(mock_claude, tmp_path):
    config = _make_config(repo_path=str(tmp_path), build_dir=str(tmp_path))
    state = _make_state_with_red_pass()
    triage_json = {
        "verdict": "mitigated",
        "reasoning": "Root cause present but call() intercepts the exploit",
        "test_analysis": "do_attr uses raw getattr but call() catches format methods",
        "root_cause_present": True,
        "mitigation_mechanism": "SandboxedEnvironment.call() → inspect_format_method()",
        "replacement_test_file": None,
    }
    triage_file = tmp_path / RED_TRIAGE_FILE
    triage_file.write_text(json.dumps(triage_json))

    result = phase_1_5_red_triage(config, _make_claude_config(), state, None)
    assert result["verdict"] == "mitigated"
    assert result["root_cause_present"] is True


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_red_triage_inconclusive(mock_claude, tmp_path):
    config = _make_config(repo_path=str(tmp_path), build_dir=str(tmp_path))
    state = _make_state_with_red_pass()
    triage_json = {
        "verdict": "inconclusive",
        "reasoning": "Cannot determine with confidence",
        "test_analysis": "Unclear code path",
        "replacement_test_file": None,
    }
    triage_file = tmp_path / RED_TRIAGE_FILE
    triage_file.write_text(json.dumps(triage_json))

    result = phase_1_5_red_triage(config, _make_claude_config(), state, None)
    assert result["verdict"] == "inconclusive"


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_red_triage_no_output_file(mock_claude, tmp_path):
    config = _make_config(repo_path=str(tmp_path), build_dir=str(tmp_path))
    state = _make_state_with_red_pass()

    result = phase_1_5_red_triage(config, _make_claude_config(), state, None)
    assert result == {}


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_red_triage_malformed_json(mock_claude, tmp_path):
    config = _make_config(repo_path=str(tmp_path), build_dir=str(tmp_path))
    state = _make_state_with_red_pass()
    triage_file = tmp_path / RED_TRIAGE_FILE
    triage_file.write_text("not valid json {{{")

    result = phase_1_5_red_triage(config, _make_claude_config(), state, None)
    assert result == {}


def test_pipeline_state_red_passed_output_default():
    state = PipelineState()
    assert state.red_passed_output == ""


@patch("auto_bug_fix.run_bug_fix.capture_output")
@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_red_triage_integration_not_vulnerable_raises_stop(mock_claude, mock_capture, tmp_path):
    """End-to-end: RED passes → triage says not_vulnerable → PipelineStop."""
    from auto_bug_fix.run_bug_fix import phase_1_baseline_and_red
    config = _make_config(
        repo_path=str(tmp_path), build_dir=str(tmp_path),
        test_command="pytest tests/", build_command="",
    )
    state = PipelineState()
    state.ported_test_files = ["tests/test_security.py"]
    state.triage_assessment = {"commit_type": "security"}

    mock_capture.return_value = (0, "all tests passed")

    phase_1_baseline_and_red(config, _make_claude_config(), state)
    assert state.red_passed_output == "all tests passed"


@patch("auto_bug_fix.run_bug_fix.cleanup_worktree")
@patch("auto_bug_fix.run_bug_fix.create_fresh_worktree")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_records_fetch_warning_in_summary(
    mock_pipeline, mock_wt_add, mock_cleanup, tmp_path,
):
    """A git fetch failure must be recorded in summary.json, not crash."""
    base_config = BugFixConfig(**_BASE_CFG)
    mock_pipeline.return_value = (_make_dossier("fix-1"), MagicMock(outcome="success"))
    mock_wt_add.return_value = None

    targets = [TargetSpec(branch="branch-a", issue_id="fix-1")]
    claude_cfg = MagicMock()
    parent_run_dir = str(tmp_path / "runs" / "test_run")
    os.makedirs(parent_run_dir, exist_ok=True)
    _make_tracker_json(
        os.path.join(parent_run_dir, "branch-a"), "fix-1", cost_usd=1.0,
    )

    fetch_warning = "git fetch --all failed (exit 128): Permission denied"
    results = run_targets(
        targets, base_config, claude_cfg,
        source_path="/tmp/source", workdir=str(tmp_path / "workdir"),
        parent_run_dir=parent_run_dir, build_subdir="",
        run_warnings=[fetch_warning],
    )

    assert len(results) == 1
    assert results[0].outcome == "success"

    summary_path = os.path.join(parent_run_dir, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    assert "warnings" in summary
    assert any("Permission denied" in w for w in summary["warnings"])


# ---------- _sanitize_branch_slug tests ----------

def test_sanitize_branch_slug_passthrough():
    assert _sanitize_branch_slug("my-issue-123") == "my-issue-123"


def test_sanitize_branch_slug_spaces():
    assert _sanitize_branch_slug("fix the bug") == "fix-the-bug"


def test_sanitize_branch_slug_colons_and_tildes():
    assert _sanitize_branch_slug("refs:HEAD~2") == "refs-HEAD-2"


def test_sanitize_branch_slug_double_dots():
    assert _sanitize_branch_slug("a..b") == "a.b"


def test_sanitize_branch_slug_at_brace():
    assert _sanitize_branch_slug("ref@{0}") == "refat-0}"


def test_sanitize_branch_slug_strips_leading_trailing():
    assert _sanitize_branch_slug("-./slug/.-") == "slug"


def test_sanitize_branch_slug_empty_fallback():
    assert _sanitize_branch_slug("---") == "unknown"


# ---------- _COAUTHOR_TRAILERS tests ----------

def test_coauthor_trailers_canonical_casing():
    assert "Co-authored-by" in _COAUTHOR_TRAILERS
    assert "Co-Authored-By" not in _COAUTHOR_TRAILERS


# ---------- Phase 4b result-file contract ----------

def _git_4b(tmp_path, *args):
    import subprocess
    return subprocess.run(
        ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _phase_4b_fixtures(tmp_path):
    from auto_bug_fix.claude_utils import BugFixClaudeConfig
    _git_4b(tmp_path, "init", "-b", "main")
    _git_4b(tmp_path, "config", "user.name", "t")
    _git_4b(tmp_path, "config", "user.email", "t@t")
    (tmp_path / "src.c").write_text("int a = 1;\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "base")
    config = BugFixConfig(
        repo_path=str(tmp_path),
        build_dir=str(tmp_path),
        source_branch="origin/main",
        target_branch="origin/release",
        source_fix_commit="abc1234",
        bug_description="test bug",
        issue_id="tid-1",
    )
    claude_cfg = BugFixClaudeConfig(
        model="test-model",
        allowed_tools=["Bash"],
        perm_mode="acceptEdits",
        cwd=str(tmp_path),
    )
    state = PipelineState()
    state.dossier = Dossier(
        issue_id="tid-1",
        source_branch="origin/main",
        target_branch="origin/release",
        fix_commit="abc1234",
        bug_description="test bug",
    )
    return config, claude_cfg, state


async def _fake_claude_run(cfg, steps):
    return []


def _marker_writing_claude_run(marker_path, text="SUCCESS\nRetries: 1/3\n"):
    """A fake session that writes a result marker, like a well-behaved agent."""
    async def _run(cfg, steps):
        marker_path.write_text(text)
        return []
    return _run


def test_phase_4b_success_file_accepted(tmp_path):
    """No build/test configured (interpreted projects) — nothing to re-verify.

    The marker is only sufficient in this degenerate case. When commands ARE
    configured they are re-run; see the two tests below.
    """
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    assert not config.build_command and not config.test_command
    head_before = _git_4b(tmp_path, "rev-parse", "HEAD")
    fake = _marker_writing_claude_run(tmp_path / "run_and_fix_success.txt")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=fake):
        phase_4b_run_and_fix(config, claude_cfg, state, None)  # must not raise
    # Marker-only session: nothing to commit, no empty fixup created.
    assert _git_4b(tmp_path, "rev-parse", "HEAD") == head_before


def test_phase_4b_success_commits_recovery_edits(tmp_path):
    """4b edits must become a commit — the patch and preserved branch are commit-based."""
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    head_before = _git_4b(tmp_path, "rev-parse", "HEAD")

    async def session(cfg, steps):
        (tmp_path / "src.c").write_text("int a = 2; /* fixed by 4b */\n")
        (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS\nRetries: 1/3\n")
        return []

    with patch("auto_bug_fix.run_bug_fix._agent_run", new=session):
        phase_4b_run_and_fix(config, claude_cfg, state, None)

    assert _git_4b(tmp_path, "rev-parse", "HEAD") != head_before
    assert "fixup: build/test recovery" in _git_4b(tmp_path, "log", "-1", "--format=%s")
    assert "fixed by 4b" in _git_4b(tmp_path, "show", "HEAD:src.c")
    # The result markers are session plumbing, never part of the backport.
    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD")
    assert "run_and_fix_success.txt" not in tree
    assert _git_4b(tmp_path, "status", "--porcelain") == ""


def test_commit_recovery_edits_artifact_only_dirt_returns_false(tmp_path):
    """Untracked pipeline artifacts alone must not be reported as a recovery commit."""
    from auto_bug_fix.run_bug_fix import _commit_recovery_edits
    from auto_bug_fix.bug_fix_prompts import SEMANTIC_TRIAGE_FILE
    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / SEMANTIC_TRIAGE_FILE).write_text("{}")
    head_before = _git_4b(tmp_path, "rev-parse", "HEAD")
    assert _commit_recovery_edits(config, "Phase 4b") is False
    assert _git_4b(tmp_path, "rev-parse", "HEAD") == head_before


def test_phase_4_5_fixup_commit_does_not_degrade_verdict(tmp_path):
    """A recovery fixup must not turn the equivalence verdict into 'unmatched'."""
    from auto_bug_fix.run_bug_fix import phase_4_5_semantic_equivalence
    config, _, state = _phase_4b_fixtures(tmp_path)
    base_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    # The fix on the source side.
    (tmp_path / "src.c").write_text("int a = 2;\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "fix: bump a")
    fix_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    # The backport: same change ported, then a recovery fixup on top.
    _git_4b(tmp_path, "checkout", "-q", "-b", "target", base_sha)
    (tmp_path / "src.c").write_text("int a = 2;\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "Port fix")
    (tmp_path / "extra.c").write_text("int b = 3;\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "fixup: build/test recovery")
    config.source_fix_commit = fix_sha
    config.target_branch = base_sha
    verdict = phase_4_5_semantic_equivalence(config, state)
    assert verdict != "unmatched", "squashed comparison must see content, not commit count"


def test_phase_4b_stale_success_file_is_ignored(tmp_path):
    """A marker left by an earlier 4b invocation must not vouch for this one."""
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS from a previous run\n")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=_fake_claude_run):
        try:
            phase_4b_run_and_fix(config, claude_cfg, state, None)
            assert False, "expected PipelineEscalation despite the stale success file"
        except PipelineEscalation as exc:
            assert "without writing a result file" in str(exc)


def test_phase_4b_missing_result_file_escalates(tmp_path):
    """A session that ends without writing either marker must not count as success."""
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=_fake_claude_run):
        try:
            phase_4b_run_and_fix(config, claude_cfg, state, None)
            assert False, "expected PipelineEscalation when no result file is written"
        except PipelineEscalation as exc:
            assert "without writing a result file" in str(exc)


# ---------- Backport completeness (partial_success, non-bugfix only) ----------

def test_completeness_bugfix_commits_never_downgrade(tmp_path):
    """The check must fire ONLY for non-bugfix commits."""
    from auto_bug_fix.run_bug_fix import _backport_completeness
    config, _, state = _phase_4b_fixtures(tmp_path)
    state.triage_assessment = {"commit_type": "bugfix", "files_to_skip": ["src.c"]}
    state.failed_prerequisites = ["some_symbol"]
    outcome, detail = _backport_completeness(config, state)
    assert outcome == "success"
    assert detail == ""


def test_completeness_unknown_commit_type_stays_success(tmp_path):
    from auto_bug_fix.run_bug_fix import _backport_completeness
    config, _, state = _phase_4b_fixtures(tmp_path)
    state.triage_assessment = {}
    state.failed_prerequisites = ["some_symbol"]
    assert _backport_completeness(config, state)[0] == "success"


def test_completeness_failed_prereq_downgrades_feature_port(tmp_path):
    from auto_bug_fix.run_bug_fix import _backport_completeness
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.source_fix_commit = _git_4b(tmp_path, "rev-parse", "HEAD")
    state.triage_assessment = {"commit_type": "api-change"}
    state.failed_prerequisites = ["missing_infra_symbol"]
    outcome, detail = _backport_completeness(config, state)
    assert outcome == "partial_success"
    assert "missing_infra_symbol" in detail


def _completeness_repo(tmp_path):
    """Source fix touching a.c+b.c; backport branch carries only a.c."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    base_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "a.c").write_text("int a;\n")
    (tmp_path / "b.c").write_text("int b;\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "feat: add a and b")
    fix_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    _git_4b(tmp_path, "checkout", "-q", "-b", "target", base_sha)
    (tmp_path / "a.c").write_text("int a;\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "Port fix (b.c skipped)")
    config.source_fix_commit = fix_sha
    config.target_branch = base_sha
    state.triage_assessment = {"commit_type": "feature", "files_to_skip": ["b.c"]}
    return config, state


def test_completeness_dropped_files_downgrade_feature_port(tmp_path):
    from auto_bug_fix.run_bug_fix import _backport_completeness
    config, state = _completeness_repo(tmp_path)
    outcome, detail = _backport_completeness(config, state)
    assert outcome == "partial_success"
    assert "b.c" in detail


def test_completeness_regenerated_skip_file_counts_as_ported(tmp_path):
    """A skipped file recovered by regeneration is not a drop."""
    from auto_bug_fix.run_bug_fix import _backport_completeness
    config, state = _completeness_repo(tmp_path)
    (tmp_path / "b.c").write_text("int b;\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "Regenerate files")
    outcome, _ = _backport_completeness(config, state)
    assert outcome == "success"


@patch("auto_bug_fix.run_bug_fix.cleanup_worktree")
@patch("auto_bug_fix.run_bug_fix.create_fresh_worktree")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_propagates_partial_success(mock_pipeline, mock_wt_add, mock_cleanup, tmp_path):
    """A tracker-downgraded outcome must reach TargetResult (and the exit code)."""
    base_config = BugFixConfig(**_BASE_CFG)

    def pipeline_side_effect(config, claude_config, output_dir):
        _make_tracker_json(output_dir, config.issue_id, cost_usd=1.00)
        return _make_dossier(config.issue_id), MagicMock(outcome="partial_success")

    mock_pipeline.side_effect = pipeline_side_effect
    mock_wt_add.return_value = None

    targets = [TargetSpec(branch="release-1.34")]
    parent_run_dir = str(tmp_path / "runs" / "partial")
    os.makedirs(parent_run_dir, exist_ok=True)

    results = run_targets(
        targets, base_config, MagicMock(),
        source_path="/tmp/source", workdir=str(tmp_path / "workdir"),
        parent_run_dir=parent_run_dir, build_subdir="",
    )
    assert len(results) == 1
    assert results[0].outcome == "partial_success"


def test_completeness_uses_seed_for_merge_commit_fixes(tmp_path):
    """Merge-commit fixes have empty `git show --name-only`; state.seed must win."""
    from auto_bug_fix.run_bug_fix import _backport_completeness
    config, state = _completeness_repo(tmp_path)
    state.seed = ["a.c", "b.c"]
    config.source_fix_commit = "0" * 40  # unresolvable — git show yields nothing
    outcome, detail = _backport_completeness(config, state)
    assert outcome == "partial_success"
    assert "b.c" in detail


def test_completeness_sloppy_bugfix_label_never_downgrades(tmp_path):
    from auto_bug_fix.run_bug_fix import _backport_completeness
    config, state = _completeness_repo(tmp_path)
    state.triage_assessment["commit_type"] = " Bug Fix "
    state.failed_prerequisites = ["sym"]
    assert _backport_completeness(config, state)[0] == "success"


def test_squashed_backport_range_is_cached(tmp_path):
    """commit-tree embeds a timestamp, so both callers must reuse one object."""
    from auto_bug_fix.run_bug_fix import _squashed_backport_range
    config, state = _completeness_repo(tmp_path)
    first = _squashed_backport_range(config, state)
    assert first and state.squashed_backport_range == first
    second = _squashed_backport_range(config, state)
    assert second == first
    # Exactly one synthetic commit object was created for the two calls.
    dangling = _git_4b(tmp_path, "fsck", "--dangling")
    assert dangling.count("dangling commit") == 1


# --- review findings: recovery commits, 4b extended verification, relatedness ---

def test_commit_recovery_edits_escalates_when_the_commit_does_not_land(tmp_path):
    """Real edits that fail to commit must never pass as success.

    The deliverables (candidate patch, preserved branch, range-diff) are all
    commit-based, so a verified working tree whose edits never reached a commit
    is exactly the silent-drop this helper exists to prevent.
    """
    from auto_bug_fix.run_bug_fix import _commit_recovery_edits, PipelineEscalation
    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / "recovery.c").write_text("int fixed = 1;\n")
    # A pre-commit hook that always fails stands in for any reason the commit
    # cannot land (hooks, index lock, bad identity).
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    head_before = _git_4b(tmp_path, "rev-parse", "HEAD")
    with pytest.raises(PipelineEscalation, match="could not be committed"):
        _commit_recovery_edits(config, "Phase 4a")
    assert _git_4b(tmp_path, "rev-parse", "HEAD") == head_before


def test_commit_recovery_edits_still_tolerates_artifact_only_dirt(tmp_path):
    """git_commit unstages pipeline artifacts, so they may legitimately remain."""
    from auto_bug_fix.run_bug_fix import _commit_recovery_edits
    from auto_bug_fix.bug_fix_prompts import SEMANTIC_TRIAGE_FILE
    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / SEMANTIC_TRIAGE_FILE).write_text("{}")
    assert _commit_recovery_edits(config, "Phase 4b") is False


def test_phase_4b_success_rejected_while_escalating_command_fails(tmp_path):
    """4b only has to satisfy build+test, so its marker cannot stand alone.

    The command whose failure caused the escalation is re-run; a success marker
    written while it still fails must not clear the gate.
    """
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix, PipelineEscalation
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    fake = _marker_writing_claude_run(tmp_path / "run_and_fix_success.txt")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=fake):
        with pytest.raises(PipelineEscalation, match="extended verification still"):
            phase_4b_run_and_fix(config, claude_cfg, state, None,
                                 extra_verification=["exit 1"])


def test_phase_4b_success_accepted_when_escalating_command_passes(tmp_path):
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    fake = _marker_writing_claude_run(tmp_path / "run_and_fix_success.txt")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=fake):
        phase_4b_run_and_fix(config, claude_cfg, state, None,
                             extra_verification=["true"])  # must not raise


def test_phase_4b_names_the_escalating_commands_to_the_agent(tmp_path):
    """The agent cannot fix what it is not told about."""
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    seen = {}

    async def _capture(cfg, steps, *a, **kw):
        seen["prompt"] = steps[0].prompt
        (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS\n")
        return []

    with patch("auto_bug_fix.run_bug_fix._agent_run", new=_capture):
        phase_4b_run_and_fix(config, claude_cfg, state, None,
                             extra_verification=["echo golangci-lint-stand-in"])
    assert "echo golangci-lint-stand-in" in seen["prompt"]


def test_unrelated_backport_is_not_reported_as_adapted(tmp_path):
    """Forced pairing makes range-diff pair anything; relatedness must catch it."""
    from auto_bug_fix.run_bug_fix import phase_4_5_semantic_equivalence
    config, _, state = _phase_4b_fixtures(tmp_path)
    base_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "src.c").write_text("int a = 2;\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "fix: bump a")
    fix_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    # A "backport" touching nothing the fix touched.
    _git_4b(tmp_path, "checkout", "-q", "-b", "target", base_sha)
    (tmp_path / "unrelated.md").write_text("docs, nothing to do with the fix\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "unrelated work")
    config.source_fix_commit = fix_sha
    config.target_branch = base_sha
    assert phase_4_5_semantic_equivalence(config, state) == "unmatched"


# --- review round 2: 4b build/test gate, non-UTF-8 diffs, content relatedness ---

def test_phase_4b_success_rejected_when_build_still_fails(tmp_path):
    """A success marker is the agent's self-report, not evidence.

    On the cherry-pick-failure and verify-failure paths extra_verification is
    empty, so without a deterministic re-run the basic build/test gate would
    rest entirely on the marker.
    """
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix, PipelineEscalation
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    config.build_command = "exit 1"
    fake = _marker_writing_claude_run(tmp_path / "run_and_fix_success.txt")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=fake):
        with pytest.raises(PipelineEscalation, match="build/test still fails"):
            phase_4b_run_and_fix(config, claude_cfg, state, None)


def test_phase_4b_success_accepted_when_build_passes(tmp_path):
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    config.build_command = "true"
    config.test_command = "true"
    fake = _marker_writing_claude_run(tmp_path / "run_and_fix_success.txt")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=fake):
        phase_4b_run_and_fix(config, claude_cfg, state, None)  # must not raise


LATIN1_FIX_SOURCE = b"/* Copyright Bj\xf8rn Hansen */\nint a = %d;\n"


def _latin1_fix_repo(tmp_path):
    """A repo whose fix commit changes a file that is not valid UTF-8."""
    _git_4b(tmp_path, "init", "-b", "main")
    _git_4b(tmp_path, "config", "user.name", "t")
    _git_4b(tmp_path, "config", "user.email", "t@t")
    (tmp_path / "legacy.c").write_bytes(LATIN1_FIX_SOURCE % 1)
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "base")
    (tmp_path / "legacy.c").write_bytes(LATIN1_FIX_SOURCE % 2)
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "fix")
    return _git_4b(tmp_path, "rev-parse", "HEAD")


def test_compute_fix_diff_survives_non_utf8_content(tmp_path):
    """Prompt text must degrade, not abort the run, on latin-1 sources."""
    from auto_bug_fix.run_bug_fix import _compute_fix_diff
    sha = _latin1_fix_repo(tmp_path)
    config = BugFixConfig(
        repo_path=str(tmp_path), build_dir=str(tmp_path), source_branch="main",
        target_branch="main", source_fix_commit=sha, bug_description="b", issue_id="i",
    )
    assert "int a = 2" in _compute_fix_diff(config)


def test_compute_fix_diff_chunked_survives_non_utf8_content(tmp_path):
    """The chunked path (for_files) takes a different branch — same exposure."""
    from auto_bug_fix.run_bug_fix import _compute_fix_diff
    sha = _latin1_fix_repo(tmp_path)
    config = BugFixConfig(
        repo_path=str(tmp_path), build_dir=str(tmp_path), source_branch="main",
        target_branch="main", source_fix_commit=sha, bug_description="b", issue_id="i",
    )
    _compute_fix_diff(config, for_files=["legacy.c"])  # must not raise


def test_same_file_unrelated_change_is_not_reported_as_adapted(tmp_path):
    """Sharing a filename is not sharing content.

    A backport that touches the right file but rewords a comment beside the
    untouched bug would pass a file-set test; it must not.
    """
    from auto_bug_fix.run_bug_fix import phase_4_5_semantic_equivalence
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text(
        "/* header */\nint compute(void) {\n  return check_bounds();\n}\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "seed src.c")
    base_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    # The real fix changes the code.
    (tmp_path / "src.c").write_text(
        "/* header */\nint compute(void) {\n  return safe_check_bounds();\n}\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "fix: harden bounds check")
    fix_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    # The "backport" touches the same file, but only an unrelated comment.
    _git_4b(tmp_path, "checkout", "-q", "-b", "target", base_sha)
    (tmp_path / "src.c").write_text(
        "/* header, reworded entirely */\nint compute(void) {\n  return check_bounds();\n}\n")
    _git_4b(tmp_path, "add", ".")
    _git_4b(tmp_path, "commit", "-m", "docs: reword comment")
    config.source_fix_commit = fix_sha
    config.target_branch = base_sha
    assert phase_4_5_semantic_equivalence(config, state) == "unmatched"


def test_faithful_port_of_the_same_lines_stays_matched(tmp_path):
    """The content check must not fire on a genuine port."""
    from auto_bug_fix.run_bug_fix import phase_4_5_semantic_equivalence
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int compute(void) {\n  return check_bounds();\n}\n")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "seed")
    base_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "src.c").write_text("int compute(void) {\n  return safe_check_bounds();\n}\n")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "fix")
    fix_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    _git_4b(tmp_path, "checkout", "-q", "-b", "target", base_sha)
    (tmp_path / "src.c").write_text("int compute(void) {\n  return safe_check_bounds();\n}\n")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "Port fix")
    # ...plus a recovery fixup, as a real run would have.
    (tmp_path / "extra.c").write_text("int b = 3;\n")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "fixup: recovery")
    config.source_fix_commit = fix_sha
    config.target_branch = base_sha
    assert phase_4_5_semantic_equivalence(config, state) != "unmatched"


def test_phase_4b_uses_the_configured_model(tmp_path):
    """The config value must reach the agent session, not just the dataclass."""
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    config.phase_4b_model = "claude-opus-5"
    seen = {}

    async def _capture(cfg, steps, *a, **kw):
        seen["model"] = cfg.model
        (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS\n")
        return []

    with patch("auto_bug_fix.run_bug_fix._agent_run", new=_capture):
        phase_4b_run_and_fix(config, claude_cfg, state, None)
    assert seen["model"] == "claude-opus-5"


def test_phase_4b_model_defaults_to_pipeline_model(tmp_path):
    """Unconfigured, 4b uses the pipeline model (no implicit escalation)."""
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    claude_cfg.model = "claude-sonnet-4-6"
    seen = {}

    async def _capture(cfg, steps, *a, **kw):
        seen["model"] = cfg.model
        (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS\n")
        return []

    with patch("auto_bug_fix.run_bug_fix._agent_run", new=_capture):
        phase_4b_run_and_fix(config, claude_cfg, state, None)
    assert seen["model"] == "claude-sonnet-4-6"


# --- review round 3: verification gate, relatedness strength, 4b artifacts ---

def _verify_config(tmp_path, **kw):
    defaults = dict(
        repo_path=str(tmp_path), build_dir=str(tmp_path), source_branch="s",
        target_branch="t", source_fix_commit="abc", bug_description="b", issue_id="i",
        build_command="true",
    )
    defaults.update(kw)
    return BugFixConfig(**defaults)


def test_failing_test_command_does_not_pass_verification(tmp_path):
    """A nonzero test exit with no parseable failures is not a pass.

    parse_test_failures returns an empty set for output it does not recognize,
    and an empty set never regresses against the baseline — so without a
    positive check the gate accepts any failing test command.
    """
    from auto_bug_fix.run_bug_fix import _run_build_and_test
    config = _verify_config(tmp_path, test_command="exit 1")
    state = PipelineState()
    passed, *_ = _run_build_and_test(config, state)
    assert passed is False


def test_failing_test_command_excused_when_baseline_failed_the_same_way(tmp_path):
    """Pre-existing opaque failures stay excused — the lint path's rule."""
    from auto_bug_fix.run_bug_fix import _run_build_and_test
    config = _verify_config(tmp_path, test_command="exit 1")
    state = PipelineState()
    state.baseline_test_exit = 1
    passed, *_ = _run_build_and_test(config, state)
    assert passed is True


def test_clean_test_command_still_passes(tmp_path):
    from auto_bug_fix.run_bug_fix import _run_build_and_test
    config = _verify_config(tmp_path, test_command="true")
    assert _run_build_and_test(config, PipelineState())[0] is True


def test_phase_4b_verification_gate_rejects_a_failing_test_command(tmp_path):
    """The 4b gate inherits _run_build_and_test — it must not be bypassable."""
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix, PipelineEscalation
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    config.test_command = "exit 1"
    fake = _marker_writing_claude_run(tmp_path / "run_and_fix_success.txt")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=fake):
        with pytest.raises(PipelineEscalation, match="build/test still fails"):
            phase_4b_run_and_fix(config, claude_cfg, state, None)


def test_relatedness_needs_more_than_one_shared_line(tmp_path):
    """Any two diffs in a language share boilerplate; one line is not evidence."""
    from auto_bug_fix.run_bug_fix import _shares_content_with_fix
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "a.py").write_text("def f():\n    return []\n")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "seed")
    base_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    # The fix: several distinctive lines in a.py.
    (tmp_path / "a.py").write_text(
        "def f():\n    validate_bounds(ptr, length)\n    audit_the_caller(ctx)\n"
        "    return []\n")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "fix")
    fix_sha = _git_4b(tmp_path, "rev-parse", "HEAD")
    # The "backport": a different file, sharing only the boilerplate line.
    _git_4b(tmp_path, "checkout", "-q", "-b", "target", base_sha)
    (tmp_path / "b.py").write_text("def unrelated():\n    return []\n")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "unrelated")
    config.source_fix_commit = fix_sha
    config.target_branch = base_sha
    assert _shares_content_with_fix(config, state) is False


def test_relatedness_ignores_opposite_polarity(tmp_path):
    """A line the fix adds, appearing as a line the backport removes, is not a match."""
    from auto_bug_fix.run_bug_fix import _changed_payload_lines
    (tmp_path / ".git").mkdir(exist_ok=True)
    _git_4b(tmp_path, "init", "-b", "main")
    _git_4b(tmp_path, "config", "user.name", "t")
    _git_4b(tmp_path, "config", "user.email", "t@t")
    (tmp_path / "f.py").write_text("alpha_beta_gamma = 1\n")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "base")
    (tmp_path / "f.py").write_text("")
    _git_4b(tmp_path, "add", "."); _git_4b(tmp_path, "commit", "-m", "remove it")
    removed = _changed_payload_lines(str(tmp_path), ["HEAD^..HEAD"])
    assert ("-", "alpha_beta_gamma = 1") in removed
    assert ("+", "alpha_beta_gamma = 1") not in removed


def test_phase_4b_artifact_does_not_claim_success_before_verification(tmp_path):
    """The marker is the agent's claim; the artifact must record the verdict."""
    import json as _json
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix, PipelineEscalation
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    config.test_command = "exit 1"
    tracker = MagicMock()
    tracker._output_dir = str(tmp_path / "out")
    os.makedirs(tracker._output_dir, exist_ok=True)
    fake = _marker_writing_claude_run(tmp_path / "run_and_fix_success.txt")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=fake):
        with pytest.raises(PipelineEscalation):
            phase_4b_run_and_fix(config, claude_cfg, state, tracker)
    data = _json.loads((tmp_path / "out" / "phase_4b_input.json").read_text())
    assert data["outcome"] == "verification_failed"
    assert data["agent_reported"] == "success"


def test_phase_4b_artifact_records_success_only_after_verification(tmp_path):
    import json as _json
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    config.build_command = "true"
    config.test_command = "true"
    tracker = MagicMock()
    tracker._output_dir = str(tmp_path / "out")
    os.makedirs(tracker._output_dir, exist_ok=True)
    fake = _marker_writing_claude_run(tmp_path / "run_and_fix_success.txt")
    with patch("auto_bug_fix.run_bug_fix._agent_run", new=fake):
        phase_4b_run_and_fix(config, claude_cfg, state, tracker)
    data = _json.loads((tmp_path / "out" / "phase_4b_input.json").read_text())
    assert data["outcome"] == "success"
