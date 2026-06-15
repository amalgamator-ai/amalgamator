"""Tests for auto_bug_fix.tracker — analytics tracker with timing, tokens, and cost."""
import json
import time

import pytest

from auto_bug_fix.tracker import (
    Tracker,
    QueryRecord,
    PhaseRecord,
    PipelineRun,
    compute_cost,
    MODEL_PRICING,
)


@pytest.fixture
def tracker(tmp_path):
    return Tracker(
        issue_id="CVE-2026-45186",
        config_dict={"repo_path": "/tmp/repo", "source_branch": "main"},
        model="claude-opus-4-6",
        output_dir=str(tmp_path),
    )


def test_compute_cost_opus():
    cost = compute_cost("claude-opus-4-6", input_tokens=1000, output_tokens=500)
    expected = (1000 * 5.0 + 500 * 25.0) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_compute_cost_sonnet():
    cost = compute_cost("claude-sonnet-4-6", input_tokens=10000, output_tokens=2000)
    expected = (10000 * 3.0 + 2000 * 15.0) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_compute_cost_unknown_model_defaults_to_opus():
    cost = compute_cost("unknown-model", input_tokens=1000, output_tokens=500)
    expected = compute_cost("claude-opus-4-6", input_tokens=1000, output_tokens=500)
    assert cost == expected


def test_phase_timing(tracker):
    with tracker.phase("Phase 0 — Triage") as phase:
        time.sleep(0.05)
    assert phase.duration_s >= 0.04
    assert phase.outcome == "proceed"


def test_phase_records_exception_outcome(tracker):
    with pytest.raises(ValueError):
        with tracker.phase("Phase 0 — Triage") as phase:
            raise ValueError("test error")
    assert phase.outcome == "ValueError"


def test_gate_recording(tracker):
    with tracker.phase("Phase 0 — Triage"):
        tracker.record_gate("forward_patch_id", "not_found")
        tracker.record_gate("ancestry", "affected")

    phase = tracker._run.phases[0]
    assert phase.gate_decisions == {
        "forward_patch_id": "not_found",
        "ancestry": "affected",
    }


def test_query_recording(tracker):
    with tracker.phase("Phase 0 — Triage"):
        t1 = time.monotonic()
        record = tracker.record_query(
            prompt_name="TriageAgent",
            start_time=t1,
            end_time=t1 + 5.0,
            duration_api_ms=4800,
            num_turns=3,
            input_tokens=3200,
            output_tokens=1500,
            is_error=False,
        )

    assert record.prompt_name == "TriageAgent"
    assert record.duration_s == 5.0
    assert record.input_tokens == 3200
    assert record.output_tokens == 1500
    assert record.cost_usd > 0


def test_query_cost_calculation(tracker):
    with tracker.phase("test"):
        record = tracker.record_query(
            prompt_name="test",
            start_time=0,
            end_time=1,
            input_tokens=1_000_000,
            output_tokens=100_000,
        )
    expected = (1_000_000 * 5.0 + 100_000 * 25.0) / 1_000_000
    assert abs(record.cost_usd - expected) < 0.01


def test_save_creates_json(tracker, tmp_path):
    with tracker.phase("Phase 0"):
        tracker.record_query("test", 0, 1, input_tokens=100, output_tokens=50)

    tracker.set_outcome("success")
    path = tracker.save()

    assert path.endswith(".json")
    with open(path) as f:
        data = json.load(f)

    assert data["issue_id"] == "CVE-2026-45186"
    assert data["outcome"] == "success"
    assert len(data["phases"]) == 1
    assert data["total_queries"] == 1
    assert data["total_input_tokens"] == 100
    assert data["total_output_tokens"] == 50
    assert data["total_cost_usd"] > 0


def test_rollup_arithmetic(tracker):
    with tracker.phase("Phase 0"):
        tracker.record_query("q1", 0, 1, input_tokens=1000, output_tokens=200)
        tracker.record_query("q2", 1, 2, input_tokens=2000, output_tokens=300)

    with tracker.phase("Phase 1"):
        tracker.record_query("q3", 2, 3, input_tokens=500, output_tokens=100)

    tracker._compute_rollups()
    run = tracker._run

    assert run.total_queries == 3
    assert run.total_input_tokens == 3500
    assert run.total_output_tokens == 600


def test_summary_output(tracker):
    with tracker.phase("Phase 0 — Triage"):
        tracker.record_query("TriageAgent", 0, 5, input_tokens=3000, output_tokens=1000)
        tracker.record_gate("ancestry", "affected")

    tracker.set_outcome("success")
    text = tracker.summary()

    assert "Phase 0" in text
    assert "TriageAgent" in text
    assert "TOTAL" in text
    assert "$" in text


def test_multiple_phases_recorded(tracker):
    with tracker.phase("Phase 0"):
        pass
    with tracker.phase("Phase 1"):
        pass
    with tracker.phase("Phase 2"):
        pass

    assert len(tracker._run.phases) == 3
    assert tracker._run.phases[0].phase == "Phase 0"
    assert tracker._run.phases[2].phase == "Phase 2"
