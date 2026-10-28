# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for backport recovery paths.

Covers Phase 3b regeneration (keep/rollback/timeout), Phase 4 verification,
Phase 4a agent-assisted recovery, and the extra-verification helpers used when
extended checks escalate into the final recovery loop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from auto_bug_fix import run_bug_fix
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.claude_utils import BugFixClaudeConfig
from auto_bug_fix.run_bug_fix import (
    PipelineEscalation,
    PipelineState,
    _extra_verification_section,
    _rerun_verification,
    phase_3b_regeneration,
    phase_4_verify,
    phase_4a_build_recovery,
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


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Recovery Test")
    _git(repo, "config", "user.email", "recovery@example.com")
    for name, content in {
        "generated.c": "base\n",
        "tracked.txt": "broken\n",
        "tracked1.txt": "one\n",
        "tracked2.txt": "two\n",
    }.items():
        (repo / name).write_text(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo


def _config(repo: str | Path, **overrides: Any) -> BugFixConfig:
    values: dict[str, Any] = {
        "repo_path": str(repo),
        "build_dir": str(repo),
        "source_branch": "main",
        "target_branch": "main",
        "source_fix_commit": "fix-sha",
        "bug_description": "test recovery",
        "issue_id": "REC-1",
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


def _labels(state: PipelineState) -> list[str]:
    return [item.label for item in state.dossier.items]


def _patch_claude_run(monkeypatch, side_effect=None):
    calls: list[dict[str, Any]] = []

    async def fake_claude_run(claude_config, prompts, tracker=None):
        calls.append(
            {
                "config": claude_config,
                "prompt": prompts[0],
                "tracker": tracker,
            }
        )
        if side_effect is not None:
            side_effect(claude_config, len(calls))
        return []

    monkeypatch.setattr(run_bug_fix, "claude_run", fake_claude_run)
    return calls


# --- Phase 3b: regeneration -------------------------------------------------


def test_regeneration_keeps_and_commits_successful_output(tmp_path):
    repo = _init_repo(tmp_path)
    before = _sha(repo)
    state = PipelineState()

    phase_3b_regeneration(
        _config(repo, build_command="true"),
        state,
        ["printf 'regen\\n' >> generated.c"],
    )

    assert (repo / "generated.c").read_text() == "base\nregen\n"
    assert _sha(repo) != before
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == (
        "Regenerate files for REC-1"
    )
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""
    assert "Regenerated files" in _labels(state)


def test_regeneration_without_changes_does_not_commit(tmp_path):
    repo = _init_repo(tmp_path)
    before = _sha(repo)
    state = PipelineState()

    phase_3b_regeneration(_config(repo, build_command="true"), state, ["true"])

    assert _sha(repo) == before
    assert "Regenerated files" not in _labels(state)


def test_regeneration_keeps_nonzero_output_when_build_still_passes(tmp_path):
    repo = _init_repo(tmp_path)
    state = PipelineState()

    phase_3b_regeneration(
        _config(repo, build_command="true"),
        state,
        ["printf 'partial\\n' >> generated.c; exit 1"],
    )

    assert (repo / "generated.c").read_text() == "base\npartial\n"
    assert any(
        label.startswith("Regeneration partial (kept):") for label in _labels(state)
    )
    assert "Regenerated files" in _labels(state)


def test_regeneration_rolls_back_when_build_breaks(tmp_path):
    repo = _init_repo(tmp_path)
    before = _sha(repo)
    state = PipelineState()

    phase_3b_regeneration(
        _config(repo, build_command="false"),
        state,
        ["printf 'bad\\n' >> generated.c; exit 1"],
    )

    assert (repo / "generated.c").read_text() == "base\n"
    assert _sha(repo) == before
    assert any(
        label.startswith("Regeneration failed (rolled back):")
        for label in _labels(state)
    )
    assert "Regenerated files" not in _labels(state)


def test_regeneration_rolls_back_on_timeout(tmp_path):
    repo = _init_repo(tmp_path)
    before = _sha(repo)
    state = PipelineState()

    phase_3b_regeneration(
        _config(repo, build_command="true", command_timeout=1),
        state,
        ["printf 'late\\n' >> generated.c; sleep 2"],
    )

    assert (repo / "generated.c").read_text() == "base\n"
    assert _sha(repo) == before
    assert any(label.startswith("Regeneration timed out:") for label in _labels(state))


# --- Phase 4: verification --------------------------------------------------


def test_phase_4_verification_passes_clean_build_and_test(tmp_path):
    repo = _init_repo(tmp_path)
    state = PipelineState()

    assert phase_4_verify(
        _config(repo, build_command="true", test_command="true"), state
    )


def test_phase_4_verification_escalates_on_build_failure(tmp_path):
    repo = _init_repo(tmp_path)

    with pytest.raises(PipelineEscalation, match="Phase 4 verify failed"):
        phase_4_verify(
            _config(repo, build_command="false", test_command="true"),
            PipelineState(),
        )


def test_phase_4_verification_escalates_on_new_test_failure(tmp_path):
    repo = _init_repo(tmp_path)

    with pytest.raises(PipelineEscalation, match="Phase 4 verify failed"):
        phase_4_verify(
            _config(repo, build_command="true", test_command="false"),
            PipelineState(),
        )


def test_phase_4_verification_excuses_a_preexisting_test_failure(tmp_path):
    repo = _init_repo(tmp_path)
    state = PipelineState()
    state.baseline_test_exit = 1

    assert phase_4_verify(
        _config(repo, build_command="true", test_command="false"), state
    )


# --- Phase 4a: agent-assisted recovery -------------------------------------


def test_phase_4a_returns_without_the_agent_when_already_green(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    calls = _patch_claude_run(monkeypatch)

    phase_4a_build_recovery(
        _config(repo, build_command="true", test_command="true"),
        _claude_config(repo),
        PipelineState(),
        tracker=None,
        fix_diff="",
    )

    assert calls == []


def test_phase_4a_escalates_when_the_agent_makes_no_changes(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    calls = _patch_claude_run(monkeypatch)

    with pytest.raises(PipelineEscalation, match="agent made no changes"):
        phase_4a_build_recovery(
            _config(
                repo,
                build_command="false",
                test_command="true",
                max_build_test_retries=1,
            ),
            _claude_config(repo),
            PipelineState(),
            tracker=None,
            fix_diff="",
        )

    assert len(calls) == 1


def test_phase_4a_commits_agent_recovery_edits(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    before = _sha(repo)

    def repair(config, _call_number):
        (Path(config.cwd) / "tracked.txt").write_text("fixed\n")

    calls = _patch_claude_run(monkeypatch, repair)

    phase_4a_build_recovery(
        _config(
            repo,
            build_command="grep -q '^fixed$' tracked.txt",
            test_command="true",
            max_build_test_retries=2,
        ),
        _claude_config(repo),
        PipelineState(),
        tracker=None,
        fix_diff="",
    )

    assert len(calls) == 1
    assert _sha(repo) != before
    assert _git(repo, "show", "HEAD:tracked.txt").stdout == "fixed\n"
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == (
        "fixup: build/test recovery for REC-1 (Phase 4a)"
    )
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""


def test_phase_4a_escalates_after_exhausting_retries(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)

    def keep_changing(config, call_number):
        name = "tracked1.txt" if call_number == 1 else "tracked2.txt"
        (Path(config.cwd) / name).write_text("changed\n")

    calls = _patch_claude_run(monkeypatch, keep_changing)

    with pytest.raises(
        PipelineEscalation, match="still failing after 2 recovery attempts"
    ):
        phase_4a_build_recovery(
            _config(
                repo,
                build_command="false",
                test_command="true",
                max_build_test_retries=2,
            ),
            _claude_config(repo),
            PipelineState(),
            tracker=None,
            fix_diff="",
        )

    assert len(calls) == 2


# --- Extended-verification helpers -----------------------------------------


@pytest.mark.parametrize(
    "command,timeout,expected",
    [
        ("true", None, []),
        ("false", None, ["false"]),
        ("sleep 2", 1, ["sleep 2"]),
    ],
)
def test_rerun_verification_returns_still_failing_commands(
    tmp_path, command, timeout, expected
):
    config = _config(tmp_path, command_timeout=timeout)

    assert _rerun_verification(config, [command]) == expected


def test_extra_verification_section_names_the_escalating_commands():
    assert _extra_verification_section([]) == ""

    text = _extra_verification_section(["pytest -q", "make lint"])
    assert "pytest -q" in text
    assert "make lint" in text
    assert "NOT optional" in text


# --- Phase 4.1: inline extended-verification recovery ----------------------


def _run_extended_verification_pipeline(
    tmp_path: Path,
    monkeypatch,
    *,
    recovery_fixes: bool,
    recovery_raises: bool = False,
):
    """Drive only the Phase 4.1 branch of _run_pipeline_phases.

    Every earlier phase is stubbed; the real code under test is the inline
    extended-verification loop, including its Phase 4a retry and Phase 4b
    escalation decisions.
    """
    repo = _init_repo(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    calls: dict[str, Any] = {"4a": 0, "4b": []}

    def fake_triage(config, claude_config, state):
        state.seed = ["generated.c"]
        state.allowed_modules = ["generated.c"]
        return {"forward_patch_id": "not_found", "ancestry": "seed_files_only"}

    def fake_semantic_triage(config, claude_config, state, tracker):
        assessment = {
            "commit_type": "bugfix",
            # One check fails; the other passes and must not be escalated.
            "verification_commands": ["test -f fixed.txt", "true"],
        }
        state.triage_assessment = assessment
        return assessment

    def fake_baseline(config, claude_config, state):
        return None

    def fake_cherry_pick(config, claude_config, state, tracker, **kwargs):
        return "clean"

    def fake_fix_diff(config, *args, **kwargs):
        return ""

    def fake_verify(config, state):
        return True

    def fake_4a(config, claude_config, state, tracker, fix_diff):
        calls["4a"] += 1
        if recovery_raises:
            raise PipelineEscalation("4a failed before the post-recovery sweep")
        if recovery_fixes:
            (Path(config.build_dir) / "fixed.txt").write_text("fixed\n")

    def fake_4b(config, claude_config, state, tracker, extra_verification=None):
        calls["4b"].append(list(extra_verification or []))

    def fake_equivalence(config, state):
        return "identical"

    monkeypatch.setattr(run_bug_fix, "phase_0_triage", fake_triage)
    monkeypatch.setattr(run_bug_fix, "phase_0_5_semantic_triage", fake_semantic_triage)
    monkeypatch.setattr(run_bug_fix, "phase_1_baseline_and_red", fake_baseline)
    monkeypatch.setattr(run_bug_fix, "cherry_pick_and_resolve", fake_cherry_pick)
    monkeypatch.setattr(run_bug_fix, "_compute_fix_diff", fake_fix_diff)
    monkeypatch.setattr(run_bug_fix, "phase_4_verify", fake_verify)
    monkeypatch.setattr(run_bug_fix, "phase_4a_build_recovery", fake_4a)
    monkeypatch.setattr(run_bug_fix, "phase_4b_run_and_fix", fake_4b)
    monkeypatch.setattr(run_bug_fix, "phase_4_5_semantic_equivalence", fake_equivalence)
    monkeypatch.setattr(
        run_bug_fix, "_squashed_backport_range", lambda config, state: None
    )
    monkeypatch.setattr(run_bug_fix, "run_range_diff", lambda *a, **k: "")
    monkeypatch.setattr(run_bug_fix, "git_format_patch_bytes", lambda *a, **k: b"patch")

    config = _config(
        repo,
        build_command="true",
        test_command="true",
        auto_discover_tests=True,
        evidence_checks=False,
    )
    dossier, tracker = run_bug_fix.run_pipeline(
        config, _claude_config(repo), output_dir=str(output_dir)
    )
    return repo, output_dir, calls, dossier, tracker


def test_extended_verification_failure_recovers_through_phase_4a(tmp_path, monkeypatch):
    repo, output_dir, calls, dossier, tracker = _run_extended_verification_pipeline(
        tmp_path, monkeypatch, recovery_fixes=True
    )

    assert calls["4a"] == 1
    assert calls["4b"] == []
    assert (repo / "fixed.txt").exists()
    assert tracker.outcome == "success"

    labels = [item.label for item in dossier.items]
    assert "Extended verification failed: test -f fixed.txt" in labels
    assert "Build recovery (from extended verification)" in labels
    assert (output_dir / "candidate.patch").read_bytes() == b"patch"


def test_extended_verification_still_failing_escalates_to_phase_4b(
    tmp_path, monkeypatch
):
    _, _, calls, dossier, _ = _run_extended_verification_pipeline(
        tmp_path, monkeypatch, recovery_fixes=False
    )

    assert calls["4a"] == 1
    assert calls["4b"] == [["test -f fixed.txt"]]

    labels = [item.label for item in dossier.items]
    assert "Extended verification failed: test -f fixed.txt" in labels
    assert "Extended verification still failing after Phase 4a" in labels


def test_extended_verification_recovery_raises_forwards_only_failed_commands(
    tmp_path, monkeypatch
):
    _, _, calls, dossier, _ = _run_extended_verification_pipeline(
        tmp_path, monkeypatch, recovery_fixes=False, recovery_raises=True
    )

    assert calls["4a"] == 1
    # "true" passed the first sweep; only the command that actually failed may
    # be forwarded. The fallback must use ext_failures, not every non-baseline
    # verification command.
    assert calls["4b"] == [["test -f fixed.txt"]]

    labels = [item.label for item in dossier.items]
    assert "Extended verification failed: test -f fixed.txt" in labels
