# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_bug_fix.tracker — analytics tracker with timing, tokens, and cost."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from auto_bug_fix.tracker import (
    Tracker,
    compute_cost,
)
from common.model_registry import ModelProfile


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
    with pytest.raises(ValueError), tracker.phase("Phase 0 — Triage") as phase:
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


def test_query_profile_precedence_and_public_metadata(tmp_path, monkeypatch):
    import auto_bug_fix.tracker as module

    local = ModelProfile(
        "local", "litellm", "openai/local", api_base="http://local/v1", api_key="SECRET"
    )
    cloud = ModelProfile("cloud", "litellm", "gpt-4o", api_key="OTHER_SECRET")
    tracker = Tracker("profiles", {}, profile=local, output_dir=str(tmp_path))
    calls = []

    def estimate(model, input_tokens, output_tokens, profile=None):
        calls.append((model, profile))
        return 0 if profile and profile.is_self_hosted() else 1.234567

    monkeypatch.setattr(module, "compute_cost", estimate)
    with tracker.phase("profiles"):
        local_record = tracker.record_query("default", 0, 1)
        cloud_record = tracker.record_query(
            "override", 1, 2, model="ignored", profile=cloud
        )
        bare_record = tracker.record_query("bare", 2, 3, model="different")
        zero = tracker.record_query("zero", 3, 4, profile=cloud, sdk_cost_usd=0.0)
        paid = tracker.record_query(
            "paid", 4, 5, profile=cloud, sdk_cost_usd=0.123456789
        )
    assert calls == [(local.model, local), (cloud.model, cloud), ("different", None)]
    assert local_record.cost_usd == 0
    assert cloud_record.model == cloud.model
    assert cloud_record.profile_id == cloud.id
    assert bare_record.profile_id is None
    assert zero.cost_usd == 0.0 and zero.cost_source == "reported"
    assert paid.cost_usd == 0.123456789
    text = Path(tracker.save()).read_text()
    data = json.loads(text)
    assert "SECRET" not in text
    assert data["total_cost_usd"] == round(
        sum(q.cost_usd for q in tracker._run.phases[0].queries), 6
    )


@pytest.mark.parametrize("reported", [None, 0.0, 0.123456789])
def test_run_llm_accounts_for_active_profile(claude_cfg, monkeypatch, reported):
    from auto_bug_fix import run_bug_fix

    profile = ModelProfile(
        "active", "litellm", "openai/local", api_base="http://local/v1"
    )
    claude_cfg.profile, claude_cfg.model = profile, profile.model
    tracker = Tracker("test", {}, model="gpt-4o")
    monkeypatch.setattr(
        run_bug_fix,
        "_agent_run",
        AsyncMock(
            return_value=[
                {
                    "name": "query",
                    "duration": 1,
                    "input_tokens": 1234,
                    "output_tokens": 567,
                    "cost_usd": reported,
                }
            ]
        ),
    )
    with tracker.phase("test"):
        asyncio.run(run_bug_fix._run_llm(claude_cfg, [], tracker))
    record = tracker._run.phases[0].queries[0]
    assert profile.is_self_hosted()
    assert record.cost_usd == (0.0 if reported is None else reported)
    assert record.profile_id == profile.id
    assert record.cost_source == ("estimated" if reported is None else "reported")
