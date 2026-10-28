# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the early backport phases (0, 0.5, 0.5b, 0.5c).

These phases decide whether a port is needed, derive the file allowlist, read
the semantic-triage assessment, port prerequisites, and port test files. The
tests exercise the real git-backed triage paths and mock only the LLM boundary
for the two agent phases.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from auto_bug_fix import run_bug_fix
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.bug_fix_prompts import SEMANTIC_TRIAGE_FILE, TEST_PORT_MANIFEST_FILE
from auto_bug_fix.claude_utils import BugFixClaudeConfig
from auto_bug_fix.run_bug_fix import (
    MAX_PREREQUISITE_DEPTH,
    PipelineEscalation,
    PipelineState,
    PipelineStop,
    phase_0_5_semantic_triage,
    phase_0_5c_test_port,
    phase_0_triage,
    port_prerequisite_chain,
)


def _git(cwd: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _sha(cwd: str | Path, ref: str = "HEAD") -> str:
    return _git(cwd, "rev-parse", ref).stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Early Phase Test")
    _git(path, "config", "user.email", "early@example.com")
    (path / "README").write_text("base\n")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "base")
    return path


def _config(repo: str | Path, fix_sha: str, **overrides: Any) -> BugFixConfig:
    values: dict[str, Any] = {
        "repo_path": str(repo),
        "build_dir": str(repo),
        "source_branch": "main",
        "target_branch": "target",
        "source_fix_commit": fix_sha,
        "bug_description": "test fix",
        "issue_id": "EARLY-1",
    }
    values.update(overrides)
    return BugFixConfig(**values)


def _claude_config(repo: str | Path) -> BugFixClaudeConfig:
    return BugFixClaudeConfig(
        model="test-model",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=str(repo),
    )


# --- Phase 0: deterministic triage -----------------------------------------


def test_phase_0_stops_when_fix_is_already_on_target(tmp_git_repo):
    repo = tmp_git_repo.repo_path
    _git(repo, "checkout", "target")
    _git(repo, "cherry-pick", tmp_git_repo.fix_sha)

    with pytest.raises(PipelineStop, match="already present"):
        phase_0_triage(
            _config(repo, tmp_git_repo.fix_sha),
            _claude_config(repo),
            PipelineState(),
        )


def test_phase_0_reports_seed_files_only_for_a_normal_backport(tmp_git_repo):
    repo = tmp_git_repo.repo_path
    state = PipelineState()

    gates = phase_0_triage(
        _config(repo, tmp_git_repo.fix_sha),
        _claude_config(repo),
        state,
    )

    assert gates == {"forward_patch_id": "not_found", "ancestry": "seed_files_only"}
    assert "lib/url.c" in state.seed
    assert "tests/test_fix.c" in state.seed
    # The new test file does not exist on the target, so only the modified
    # source file resolves onto the target allowlist.
    assert state.allowed_modules == ["lib/url.c"]


def test_phase_0_reports_no_seed_files_and_keeps_seed_as_allowlist(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _git(repo, "branch", "target")
    (repo / "new_module.c").write_text("int new_symbol;\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fix: add new module")
    fix_sha = _sha(repo)

    state = PipelineState()
    gates = phase_0_triage(_config(repo, fix_sha), _claude_config(repo), state)

    assert gates == {"forward_patch_id": "not_found", "ancestry": "no_seed_files"}
    assert state.seed == ["new_module.c"]
    assert state.allowed_modules == ["new_module.c"]


def test_phase_0_explicit_allowed_modules_wins_on_no_seed_files(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _git(repo, "branch", "target")
    (repo / "brand_new.c").write_text("int brand_new_symbol;\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fix: add brand new file")
    fix_sha = _sha(repo)

    state = PipelineState()
    gates = phase_0_triage(
        _config(repo, fix_sha, allowed_modules=["custom/"]),
        _claude_config(repo),
        state,
    )

    assert gates == {"forward_patch_id": "not_found", "ancestry": "no_seed_files"}
    # The operator override is authoritative even when the fix's files are
    # absent from the target and the seed fallback would otherwise win.
    assert state.allowed_modules == ["custom/"]


def test_phase_0_respects_explicit_allowed_modules(tmp_git_repo):
    repo = tmp_git_repo.repo_path
    state = PipelineState()

    phase_0_triage(
        _config(repo, tmp_git_repo.fix_sha, allowed_modules=["custom/"]),
        _claude_config(repo),
        state,
    )

    assert state.allowed_modules == ["custom/"]


def test_phase_0_reports_affected_when_only_ancestry_matches(tmp_git_repo, monkeypatch):
    repo = tmp_git_repo.repo_path
    # A fix already in target history normally stops at the forward patch-id
    # gate. Stub the patch-id gates so this test isolates the ancestry status:
    # the fix is an ancestor, so the target is affected even without a
    # patch-id match (the case patch-id cannot see, e.g. a no-op commit).
    monkeypatch.setattr(run_bug_fix, "forward_patch_id_check", lambda *a, **k: None)
    monkeypatch.setattr(run_bug_fix, "backward_patch_id_check", lambda *a, **k: None)
    monkeypatch.setattr(run_bug_fix, "git_is_ancestor", lambda *a, **k: True)

    gates = phase_0_triage(
        _config(repo, tmp_git_repo.fix_sha),
        _claude_config(repo),
        PipelineState(),
    )

    assert gates == {"forward_patch_id": "not_found", "ancestry": "affected"}


# --- Phase 0.5: semantic triage --------------------------------------------


def _patch_claude_run(monkeypatch, writer):
    calls: list[dict[str, Any]] = []

    async def fake_claude_run(claude_config, prompts, tracker=None):
        calls.append(
            {
                "config": claude_config,
                "prompt": prompts[0],
                "tracker": tracker,
            }
        )
        writer(Path(claude_config.cwd))
        return []

    monkeypatch.setattr(run_bug_fix, "claude_run", fake_claude_run)
    return calls


def _write_triage(path: Path, payload: dict | str | None) -> None:
    if payload is None:
        return
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (path / SEMANTIC_TRIAGE_FILE).write_text(text)


@pytest.mark.parametrize("subdir", ["", "runs"])
def test_semantic_triage_reads_json_and_priority_files(tmp_path, monkeypatch, subdir):
    repo = tmp_path / "repo"
    repo.mkdir()
    output_dir = repo / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    assessment = {
        "commit_type": "bugfix",
        "estimated_difficulty": "easy",
        "priority_files": ["lib/url.c"],
        "prerequisites": [],
    }
    calls = _patch_claude_run(
        monkeypatch,
        lambda cwd: _write_triage(Path(cwd) / subdir, assessment),
    )
    state = PipelineState()
    state.allowed_modules = ["lib/default.c"]

    result = phase_0_5_semantic_triage(
        _config(repo, "abc123"),
        _claude_config(repo),
        state,
        tracker=object(),
    )

    assert result == assessment
    assert state.triage_assessment == assessment
    assert state.priority_files == ["lib/url.c"]
    assert len(calls) == 1
    assert calls[0]["config"].cwd == str(repo)
    assert calls[0]["tracker"] is not None
    assert "Semantic Triage Agent (Phase 0.5)" in calls[0]["prompt"]


def test_semantic_triage_falls_back_to_allowed_modules(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _patch_claude_run(
        monkeypatch,
        lambda cwd: _write_triage(Path(cwd), {"commit_type": "bugfix"}),
    )
    state = PipelineState()
    state.allowed_modules = ["lib/default.c"]

    phase_0_5_semantic_triage(
        _config(repo, "abc123"), _claude_config(repo), state, tracker=None
    )

    assert state.priority_files == ["lib/default.c"]


@pytest.mark.parametrize("payload", [None, "{not valid json"])
def test_semantic_triage_ignores_missing_or_malformed_output(
    tmp_path, monkeypatch, payload
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _patch_claude_run(monkeypatch, lambda cwd: _write_triage(Path(cwd), payload))
    state = PipelineState()
    state.allowed_modules = ["lib/default.c"]

    result = phase_0_5_semantic_triage(
        _config(repo, "abc123"), _claude_config(repo), state, tracker=None
    )

    assert result == {}
    assert state.triage_assessment == {}
    assert state.priority_files == ["lib/default.c"]


# --- Phase 0.5c: test port --------------------------------------------------


def _patch_test_port(monkeypatch, writer):
    calls: list[dict[str, Any]] = []

    async def fake_claude_run(claude_config, prompts, tracker=None):
        calls.append({"config": claude_config, "prompt": prompts[0]})
        writer(Path(claude_config.cwd))
        return []

    monkeypatch.setattr(run_bug_fix, "claude_run", fake_claude_run)
    return calls


def _empty_manifest(path: Path) -> None:
    return None


def test_test_port_skips_when_no_test_files(tmp_path, monkeypatch):
    calls = _patch_test_port(monkeypatch, _empty_manifest)
    state = PipelineState()
    state.seed = ["lib/url.c"]

    result = phase_0_5c_test_port(
        _config(tmp_path, "abc123"), _claude_config(tmp_path), state, tracker=None
    )

    assert result == []
    assert state.ported_test_files == []
    assert calls == []


def test_test_port_skips_files_to_skip(tmp_path, monkeypatch):
    calls = _patch_test_port(monkeypatch, _empty_manifest)
    state = PipelineState()
    state.seed = ["tests/test_url.py"]
    state.triage_assessment = {"files_to_skip": ["tests/test_url.py"]}

    result = phase_0_5c_test_port(
        _config(tmp_path, "abc123"), _claude_config(tmp_path), state, tracker=None
    )

    assert result == []
    assert calls == []


def test_test_port_reads_manifest_and_sets_state(tmp_path, monkeypatch):
    monkeypatch.setattr(run_bug_fix, "is_merge_commit", lambda *a: False)
    calls = _patch_test_port(
        monkeypatch,
        lambda cwd: (Path(cwd) / TEST_PORT_MANIFEST_FILE).write_text(
            "tests/test_url.py\n"
        ),
    )
    state = PipelineState()
    state.seed = ["lib/url.c", "tests/test_url.py", "tests/test_other.py"]
    state.triage_assessment = {"files_to_skip": ["tests/test_other.py"]}
    repo = tmp_path / "repo"
    repo.mkdir()

    result = phase_0_5c_test_port(
        _config(repo, "abc123"), _claude_config(repo), state, tracker=None
    )

    assert result == ["tests/test_url.py"]
    assert state.ported_test_files == ["tests/test_url.py"]
    assert len(calls) == 1
    prompt = calls[0]["prompt"]
    assert "tests/test_url.py" in prompt
    assert "tests/test_other.py" not in prompt
    assert str(repo / TEST_PORT_MANIFEST_FILE) in prompt
    assert "git show abc123 --" in prompt


def test_test_port_uses_diff_range_for_merge_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(run_bug_fix, "is_merge_commit", lambda *a: True)
    calls = _patch_test_port(
        monkeypatch,
        lambda cwd: (Path(cwd) / TEST_PORT_MANIFEST_FILE).write_text(
            "tests/test_url.py\n"
        ),
    )
    state = PipelineState()
    state.seed = ["tests/test_url.py"]
    repo = tmp_path / "repo"
    repo.mkdir()

    phase_0_5c_test_port(
        _config(repo, "abc123"), _claude_config(repo), state, tracker=None
    )

    assert "git diff abc123^1..abc123 --" in calls[0]["prompt"]


def test_test_port_missing_manifest_leaves_state_empty(tmp_path, monkeypatch):
    _patch_test_port(monkeypatch, _empty_manifest)
    state = PipelineState()
    state.seed = ["tests/test_url.py"]
    repo = tmp_path / "repo"
    repo.mkdir()

    result = phase_0_5c_test_port(
        _config(repo, "abc123"), _claude_config(repo), state, tracker=None
    )

    assert result == []
    assert state.ported_test_files == []


# --- Phase 0.5b: prerequisite chain ----------------------------------------


def _prereq(symbol: str, sha: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "symbol": symbol,
        "introduced_by": sha,
        "portable": True,
        "subject": f"prereq {symbol}",
        "reason": f"{symbol} is required",
    }
    payload.update(overrides)
    return payload


def test_prerequisite_chain_filters_non_portable_entries(tmp_path, monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_cherry_pick(config, claude_config, state, tracker, **kwargs):
        calls.append(kwargs)
        return "clean"

    monkeypatch.setattr(run_bug_fix, "cherry_pick_and_resolve", fake_cherry_pick)
    state = PipelineState()
    state.triage_assessment = {
        "prerequisites": [
            _prereq("portable", "a" * 40),
            _prereq("not_portable", "b" * 40, portable=False),
            _prereq("no_commit", "c" * 40, introduced_by=""),
        ]
    }

    ported = port_prerequisite_chain(
        _config(tmp_path, "fix-sha"),
        _claude_config(tmp_path),
        state,
        tracker=None,
    )

    assert [c["commit"] for c in calls] == ["a" * 40]
    assert ported == [{"sha": "a" * 40, "symbol": "portable", "result": "clean"}]


def test_prerequisite_chain_ports_and_restores_fix_commit(tmp_path, monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_cherry_pick(config, claude_config, state, tracker, **kwargs):
        calls.append(
            {
                **kwargs,
                "source_fix_commit": config.source_fix_commit,
            }
        )
        return "resolved"

    monkeypatch.setattr(run_bug_fix, "cherry_pick_and_resolve", fake_cherry_pick)
    config = _config(tmp_path, "main-fix-sha")
    state = PipelineState()
    state.triage_assessment = {
        "prerequisites": [_prereq("alpha", "a" * 40), _prereq("beta", "b" * 40)],
        "files_to_skip": ["generated/"],
    }

    ported = port_prerequisite_chain(
        config, _claude_config(tmp_path), state, tracker=None
    )

    assert [c["commit"] for c in calls] == ["a" * 40, "b" * 40]
    # The phase temporarily points the config at each prerequisite so
    # evidence and diffs use the right commit, then restores the main fix.
    assert [c["source_fix_commit"] for c in calls] == ["a" * 40, "b" * 40]
    assert config.source_fix_commit == "main-fix-sha"
    assert [c["files_to_skip"] for c in calls] == [
        ["generated/"],
        ["generated/"],
    ]
    assert ported == [
        {"sha": "a" * 40, "symbol": "alpha", "result": "resolved"},
        {"sha": "b" * 40, "symbol": "beta", "result": "resolved"},
    ]
    labels = [item.label for item in state.dossier.items]
    assert "Prerequisite 1: alpha" in labels
    assert "Prerequisite 2: beta" in labels


def test_prerequisite_chain_records_failure_and_continues(tmp_path, monkeypatch):
    def fake_cherry_pick(config, claude_config, state, tracker, **kwargs):
        if kwargs["commit"] == "a" * 40:
            raise PipelineEscalation("cannot port alpha")
        return "clean"

    monkeypatch.setattr(run_bug_fix, "cherry_pick_and_resolve", fake_cherry_pick)
    config = _config(tmp_path, "main-fix-sha")
    state = PipelineState()
    state.triage_assessment = {
        "prerequisites": [_prereq("alpha", "a" * 40), _prereq("beta", "b" * 40)]
    }

    ported = port_prerequisite_chain(
        config, _claude_config(tmp_path), state, tracker=None
    )

    assert state.failed_prerequisites == ["alpha"]
    assert ported == [{"sha": "b" * 40, "symbol": "beta", "result": "clean"}]
    assert config.source_fix_commit == "main-fix-sha"
    labels = [item.label for item in state.dossier.items]
    assert "Prerequisite failed: alpha" in labels
    assert "Prerequisite 2: beta" in labels


def test_prerequisite_chain_caps_depth_and_records_skips(tmp_path, monkeypatch, caplog):
    calls: list[str] = []

    def fake_cherry_pick(config, claude_config, state, tracker, **kwargs):
        calls.append(kwargs["commit"])
        return "clean"

    monkeypatch.setattr(run_bug_fix, "cherry_pick_and_resolve", fake_cherry_pick)
    total = MAX_PREREQUISITE_DEPTH + 2
    state = PipelineState()
    state.triage_assessment = {
        "prerequisites": [_prereq(f"symbol-{i}", f"{i:040d}") for i in range(total)]
    }

    with caplog.at_level(logging.WARNING):
        ported = port_prerequisite_chain(
            _config(tmp_path, "fix-sha"),
            _claude_config(tmp_path),
            state,
            tracker=None,
        )

    assert len(calls) == MAX_PREREQUISITE_DEPTH
    assert len(ported) == MAX_PREREQUISITE_DEPTH
    # Overflow must be visible to the completeness gate, not silently dropped.
    assert state.skipped_prerequisites == [
        f"symbol-{MAX_PREREQUISITE_DEPTH}",
        f"symbol-{MAX_PREREQUISITE_DEPTH + 1}",
    ]
    skipped_items = [
        item
        for item in state.dossier.items
        if item.label == "Prerequisites skipped (depth cap)"
    ]
    assert len(skipped_items) == 1
    assert "symbol-5" in skipped_items[0].content
    assert "symbol-6" in skipped_items[0].content
    assert any("depth cap" in record.message for record in caplog.records)
