"""Tests for auto_bug_fix.run_bug_fix — pipeline state, strategies, and helpers."""
from unittest.mock import patch, MagicMock
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    PipelineStop,
    PipelineEscalation,
    identify_test_files,
    _resolve_lint_command,
    _run_build_and_test,
)
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


# --- timeout-as-hard-failure tests ---

@patch("auto_bug_fix.run_bug_fix.enforce_test_file_veto", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.enforce_allowlist", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.run_test_suite")
def test_test_timeout_means_verify_fail(mock_run, mock_allow, mock_veto):
    # A test command that times out must fail verification, never be masked as
    # "no FAIL lines parsed -> pass".
    mock_run.side_effect = [
        (0, "", ""),                                   # build
        (TIMEOUT_EXIT_CODE, "partial", "timed out"),   # test times out
    ]
    config = _make_config()
    state = PipelineState()
    passed, _, _, _, _ = _run_build_and_test(config, state)
    assert passed is False
    assert mock_run.call_count == 2  # returns at the test step; lint not reached


@patch("auto_bug_fix.run_bug_fix.enforce_test_file_veto", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.enforce_allowlist", return_value=(True, []))
@patch("auto_bug_fix.run_bug_fix.run_test_suite")
def test_lint_timeout_not_masked_as_preexisting(mock_run, mock_allow, mock_veto):
    # Regression guard for the false-success class: a lint command that times
    # out must fail even when the same command also timed out on the baseline.
    # A timeout is never treated as a pre-existing failure.
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
