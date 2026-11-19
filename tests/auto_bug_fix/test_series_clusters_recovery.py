# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A series run must not lose its partition silently."""

from __future__ import annotations

import json

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    _ensure_series_clusters,
)
from common.claude_utils import ClaudeConfig


def _claude():
    return ClaudeConfig(
        model="test", allowed_tools=[], perm_mode="acceptEdits", cwd=None
    )


class _Tracker:
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)
        self.gates = {}

    def record_gate(self, name, value):
        self.gates[name] = value


def _config(tmp_path):
    return BugFixConfig(
        repo_path=str(tmp_path),
        build_dir=str(tmp_path),
        source_branch="main",
        target_branch="main",
        source_fix_commit="abc123",
        bug_description="x",
        issue_id="TEST-1",
        source_fix_commits=["a" * 40, "b" * 40],
    )


def test_existing_partition_is_left_alone(tmp_path, monkeypatch):
    state = PipelineState()
    state.triage_assessment = {"clusters": [{"id": "x", "commits": ["a" * 40]}]}
    tracker = _Tracker(tmp_path)

    called = []
    monkeypatch.setattr(
        "auto_bug_fix.run_bug_fix.create_context_str", lambda *a, **k: ""
    )
    monkeypatch.setattr(
        "auto_bug_fix.run_bug_fix.claude_run",
        lambda *a, **k: called.append(True),
    )

    assert _ensure_series_clusters(_config(tmp_path), _claude(), state, tracker)
    assert called == []


def test_missing_partition_is_requested_and_merged(tmp_path, monkeypatch):
    state = PipelineState()
    state.triage_assessment = {"files_to_skip": []}
    tracker = _Tracker(tmp_path)

    monkeypatch.setattr(
        "auto_bug_fix.run_bug_fix.create_context_str", lambda *a, **k: ""
    )
    monkeypatch.setattr("auto_bug_fix.run_bug_fix._phase_config", lambda *a, **k: None)

    def fake_run(config, prompts, tracker=None):
        path = tmp_path / "series_clusters.json"
        path.write_text(
            json.dumps(
                {
                    "clusters": [{"id": "all", "commits": ["a" * 40, "b" * 40]}],
                    "dropped": [{"commit": "b" * 40, "reason": "docs"}],
                    "public_interface_files": ["api.h"],
                }
            )
        )
        return []

    monkeypatch.setattr("auto_bug_fix.run_bug_fix.claude_run", fake_run)

    assert _ensure_series_clusters(_config(tmp_path), _claude(), state, tracker)
    assert state.triage_assessment["clusters"] == [
        {"id": "all", "commits": ["a" * 40, "b" * 40]}
    ]
    assert state.triage_assessment["dropped"] == [
        {"commit": "b" * 40, "reason": "docs"}
    ]
    assert state.triage_assessment["public_interface_files"] == ["api.h"]
    assert tracker.gates["clusters"] == "recovered"


def test_failed_recovery_records_the_fallback(tmp_path, monkeypatch):
    state = PipelineState()
    state.triage_assessment = {}
    tracker = _Tracker(tmp_path)
    monkeypatch.setattr(
        "auto_bug_fix.run_bug_fix.create_context_str", lambda *a, **k: ""
    )
    monkeypatch.setattr("auto_bug_fix.run_bug_fix._phase_config", lambda *a, **k: None)
    monkeypatch.setattr("auto_bug_fix.run_bug_fix.claude_run", lambda *a, **k: [])

    assert not _ensure_series_clusters(_config(tmp_path), _claude(), state, tracker)
    assert tracker.gates["clusters"] == "fallback"


def test_single_commit_mode_needs_no_partition(tmp_path):
    config = BugFixConfig(
        repo_path=str(tmp_path),
        build_dir=str(tmp_path),
        source_branch="main",
        target_branch="main",
        source_fix_commit="abc123",
        bug_description="x",
        issue_id="TEST-1",
    )
    tracker = _Tracker(tmp_path)

    assert _ensure_series_clusters(config, _claude(), PipelineState(), tracker)
    assert tracker.gates == {}
