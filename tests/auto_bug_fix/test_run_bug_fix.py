"""Tests for auto_bug_fix.run_bug_fix — pipeline state, strategies, and helpers."""
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    PipelineStop,
    PipelineEscalation,
    CHERRY_PICK_STRATEGIES,
    identify_test_files,
)


def test_pipeline_state_defaults():
    state = PipelineState()
    assert state.seed == []
    assert state.allowed_modules == []
    assert state.baseline == set()
    assert state.ported_test_files == []
    assert state.s_target == ""
    assert state.bisect_sha is None
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


def test_cherry_pick_strategies_order():
    assert len(CHERRY_PICK_STRATEGIES) == 3
    assert CHERRY_PICK_STRATEGIES[0]["name"] == "default"
    assert CHERRY_PICK_STRATEGIES[1]["name"] == "patience"
    assert CHERRY_PICK_STRATEGIES[2]["name"] == "ort"


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
