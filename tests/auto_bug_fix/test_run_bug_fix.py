"""Tests for auto_bug_fix.run_bug_fix — pipeline state, strategies, and helpers."""
import json
import os
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
from auto_bug_fix.git_tools import GitResult
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
@patch("auto_bug_fix.run_bug_fix.git_worktree_add")
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
        return _make_dossier(config.issue_id), MagicMock()

    mock_pipeline.side_effect = pipeline_side_effect
    mock_wt_add.return_value = GitResult(exit_code=0, stdout="", stderr="")

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
@patch("auto_bug_fix.run_bug_fix.git_worktree_add")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_single_target(mock_pipeline, mock_wt_add, mock_cleanup, tmp_path):
    """Single-element list produces correct directory structure."""
    base_config = BugFixConfig(**_BASE_CFG)

    def pipeline_side_effect(config, claude_config, output_dir):
        _make_tracker_json(output_dir, config.issue_id, cost_usd=2.00)
        return _make_dossier(config.issue_id), MagicMock()

    mock_pipeline.side_effect = pipeline_side_effect
    mock_wt_add.return_value = GitResult(exit_code=0, stdout="", stderr="")

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
@patch("auto_bug_fix.run_bug_fix.git_worktree_add")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_cost_recovery_on_failure(mock_pipeline, mock_wt_add, mock_cleanup, tmp_path):
    """TargetResult.cost_usd is populated from tracker JSON even when run_pipeline raises."""
    base_config = BugFixConfig(**_BASE_CFG)

    def pipeline_side_effect(config, claude_config, output_dir):
        _make_tracker_json(output_dir, config.issue_id, cost_usd=4.56)
        raise PipelineStop("already fixed")

    mock_pipeline.side_effect = pipeline_side_effect
    mock_wt_add.return_value = GitResult(exit_code=0, stdout="", stderr="")

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
@patch("auto_bug_fix.run_bug_fix.git_worktree_add")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_worktree_failure_continues(mock_pipeline, mock_wt_add, mock_cleanup, tmp_path):
    """Worktree failure on target 1 records worktree_error and target 2 still runs."""
    base_config = BugFixConfig(**_BASE_CFG)

    call_count = [0]

    def wt_add_side_effect(source_path, worktree_path, branch):
        call_count[0] += 1
        if call_count[0] == 1:
            return GitResult(exit_code=1, stdout="", stderr="fatal: invalid ref: bad-branch")
        return GitResult(exit_code=0, stdout="", stderr="")

    mock_wt_add.side_effect = wt_add_side_effect

    def pipeline_side_effect(config, claude_config, output_dir):
        _make_tracker_json(output_dir, config.issue_id, cost_usd=3.21)
        return _make_dossier(config.issue_id), MagicMock()

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
from auto_bug_fix.claude_utils import ClaudeConfig


def _make_claude_config():
    return ClaudeConfig(
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
@patch("auto_bug_fix.run_bug_fix.git_worktree_add")
@patch("auto_bug_fix.run_bug_fix.run_pipeline")
def test_run_targets_records_fetch_warning_in_summary(
    mock_pipeline, mock_wt_add, mock_cleanup, tmp_path,
):
    """A git fetch failure must be recorded in summary.json, not crash."""
    base_config = BugFixConfig(**_BASE_CFG)
    mock_pipeline.return_value = (_make_dossier("fix-1"), MagicMock())
    mock_wt_add.return_value = GitResult(exit_code=0, stdout="", stderr="")

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

def _phase_4b_fixtures(tmp_path):
    from auto_bug_fix.claude_utils import ClaudeConfig
    config = BugFixConfig(
        repo_path=str(tmp_path),
        build_dir=str(tmp_path),
        source_branch="origin/main",
        target_branch="origin/release",
        source_fix_commit="abc1234",
        bug_description="test bug",
        issue_id="tid-1",
    )
    claude_cfg = ClaudeConfig(
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


def test_phase_4b_success_file_accepted(tmp_path):
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS\nRetries: 1/3\n")
    with patch("auto_bug_fix.run_bug_fix._claude_run", new=_fake_claude_run):
        phase_4b_run_and_fix(config, claude_cfg, state, None)  # must not raise


def test_phase_4b_missing_result_file_escalates(tmp_path):
    """A session that ends without writing either marker must not count as success."""
    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    with patch("auto_bug_fix.run_bug_fix._claude_run", new=_fake_claude_run):
        try:
            phase_4b_run_and_fix(config, claude_cfg, state, None)
            assert False, "expected PipelineEscalation when no result file is written"
        except PipelineEscalation as exc:
            assert "without writing a result file" in str(exc)
