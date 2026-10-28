# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end orchestration smoke for the git-first backport pipeline.

Unlike the phase-level tests, this drives ``run_targets`` through the real
orchestrator: real git repositories and worktrees, real artifact writers, and
the real ``agent_run`` dispatcher. Only the two provider runners are mocked, so
the test still exercises ``_run_llm``, provider selection, tracker accounting,
and phase ordering without a model service.

The scenario is a bug-fix commit whose target branch re-indented the same file.
Cherry-pick conflicts, mechanical resolution wins without an LLM edit, and the
deterministic post-passes plus the review agents run on the result.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.claude_utils import BugFixClaudeConfig, PipelineStep
from auto_bug_fix.config_loader import TargetSpec
from auto_bug_fix.run_bug_fix import run_targets
from common import agent_runner
from common.model_registry import ModelProfile

BASE = """\
void vulnerable_func(void)
{
    int a = 1;
    int b = 2;
    int c = 3;
    return;
}
"""

FIXED = BASE.replace("    int c = 3;", "    int c = 30;")

REINDENTED = BASE.replace("    int", "\tint").replace("    return;", "\treturn;")

RESOLVED_BY_LLM = REINDENTED.replace("\tint c = 3;", "\tint c = 30;")


def _git(
    cwd: str | Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _sha(cwd: str | Path, ref: str = "HEAD") -> str:
    return _git(cwd, "rev-parse", ref).stdout.strip()


def _build_repo(tmp_path: Path) -> tuple[Path, str]:
    """Build main (base -> fix) and a target that re-indented the file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib").mkdir()

    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Smoke Test")
    _git(repo, "config", "user.email", "smoke@example.com")

    (repo / "lib" / "url.c").write_text(BASE)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    (repo / "lib" / "url.c").write_text(FIXED)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fix: guard the vulnerable path")
    fix_sha = _sha(repo)

    _git(repo, "branch", "target", "HEAD~1")
    _git(repo, "checkout", "target")
    (repo / "lib" / "url.c").write_text(REINDENTED)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "target: re-indent url.c")
    _git(repo, "checkout", "main")

    return repo, fix_sha


def _step_texts(steps: list[Any]) -> list[str]:
    texts = []
    for step in steps:
        if isinstance(step, PipelineStep):
            texts.append(step.prompt if isinstance(step.prompt, str) else "")
        elif isinstance(step, str):
            texts.append(step)
    return texts


def _timing() -> dict[str, Any]:
    return {
        "name": "smoke-agent",
        "duration": 0.01,
        "input_tokens": 100,
        "output_tokens": 20,
        "cost_usd": 0.25,
    }


def _fake_agent(prompts: list[str]):
    """Emulate the agent's side effects at the runner boundary."""

    async def run(config: Any, steps: list[Any], resume: bool = False) -> list[dict]:
        joined = "\n".join(_step_texts(steps))
        prompts.append(joined)
        repo = Path(config.cwd)

        if "Semantic Triage Agent (Phase 0.5)" in joined:
            (repo / "semantic_triage.json").write_text(
                json.dumps(
                    {
                        "commit_type": "bugfix",
                        "estimated_difficulty": "easy",
                        "recommendation": "proceed",
                        "prerequisites": [],
                        "files_to_skip": [],
                    }
                )
            )
        elif "Narrow Resolution Agent (Phase 3a)" in joined:
            # Only reached if mechanical resolution cannot win; keep the
            # agent's edit honest so the post-passes have something to audit.
            (repo / "lib" / "url.c").write_text(RESOLVED_BY_LLM)
            _git(repo, "add", "lib/url.c")

        return [_timing()]

    return run


@pytest.mark.parametrize("provider", ["anthropic", "litellm"])
def test_run_targets_executes_the_full_pipeline(tmp_path, monkeypatch, provider):
    repo, fix_sha = _build_repo(tmp_path)
    model = "claude-sonnet-4-6" if provider == "anthropic" else "gpt-4o"
    profile = ModelProfile(id=f"smoke-{provider}", provider=provider, model=model)

    prompts: list[str] = []
    fake_agent = _fake_agent(prompts)
    claude_runner = AsyncMock(side_effect=fake_agent)
    litellm_runner = AsyncMock(side_effect=fake_agent)
    monkeypatch.setattr(agent_runner.ClaudeAgentRunner, "run", claude_runner)
    monkeypatch.setattr(agent_runner.LiteLLMAgentRunner, "run", litellm_runner)

    base_config = BugFixConfig(
        repo_path=str(repo),
        build_dir=str(repo),
        source_branch="main",
        target_branch="target",
        source_fix_commit=fix_sha,
        bug_description="smoke: fix the vulnerable path in lib/url.c",
        issue_id="SMOKE-1",
        build_command="true",
        test_command="true",
    )
    claude_config = BugFixClaudeConfig(
        model=model,
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=str(repo),
        profile=profile,
    )

    parent_run_dir = tmp_path / "runs"
    results = run_targets(
        [TargetSpec(branch="target")],
        base_config,
        claude_config,
        source_path=str(repo),
        workdir=str(tmp_path / "worktrees"),
        parent_run_dir=str(parent_run_dir),
        build_subdir="",
    )

    assert len(results) == 1
    result = results[0]
    assert result.outcome == "success", result.error
    assert result.error == ""
    assert result.issue_id == "SMOKE-1"
    assert result.cost_usd > 0
    assert result.dossier_path == "target/dossier.md"
    assert result.tracker_path.startswith("target/SMOKE-1_")

    # The dispatcher selected the runner that matches the profile provider.
    if provider == "anthropic":
        claude_runner.assert_awaited()
        litellm_runner.assert_not_awaited()
    else:
        litellm_runner.assert_awaited()
        claude_runner.assert_not_awaited()

    target_run = parent_run_dir / "target"
    tracker_files = sorted(target_run.glob("SMOKE-1_*.json"))
    assert len(tracker_files) == 1
    tracker = json.loads(tracker_files[0].read_text())

    phase_names = [phase["phase"] for phase in tracker["phases"]]
    assert phase_names == [
        "Phase 0 — Triage",
        "Phase 0.5 — Semantic triage",
        "Phase 1 — Baseline + RED",
        "Phase 2+3a — Cherry-pick and resolve",
        "Phase 4 — Verify",
        "Phase 4.5 — Semantic equivalence",
        "Phase 5 — Completeness check",
    ]

    assert tracker["outcome"] == "success"
    # Every fake query returned the same usage, and _run_llm recorded it.
    assert tracker["total_queries"] == len(prompts)
    assert tracker["total_input_tokens"] == 100 * len(prompts)
    assert tracker["total_output_tokens"] == 20 * len(prompts)
    assert tracker["total_cost_usd"] == pytest.approx(0.25 * len(prompts))
    assert result.cost_usd == pytest.approx(tracker["total_cost_usd"])
    query_profiles = {
        query["profile_id"] for phase in tracker["phases"] for query in phase["queries"]
    }
    assert query_profiles == {profile.id}

    # Deterministic-first: the mechanical rung resolved the conflict, so the
    # narrow LLM conflict agent was never needed. The two review agents did run.
    assert any("Semantic Triage Agent (Phase 0.5)" in p for p in prompts)
    assert any("Post-Resolution Verification Agent (Phase 3a.1)" in p for p in prompts)
    assert any("Independent Review Agent (Phase 3a.2)" in p for p in prompts)
    assert not any("Narrow Resolution Agent (Phase 3a)" in p for p in prompts)

    gates = {
        name: decision
        for phase in tracker["phases"]
        for name, decision in phase.get("gate_decisions", {}).items()
    }
    assert gates["forward_patch_id"] == "not_found"
    # The fix is not an ancestor and its patch-id is absent, but the seed file
    # exists on the target branch — the normal "port me" ancestry outcome.
    assert gates["ancestry"] == "seed_files_only"
    assert gates["cherry_pick"] == "resolved"
    assert gates["mechanical_resolution"] == "1/1"
    assert gates["token_audit"] == "0 flagged"
    assert gates["line_provenance"] == "0 findings"
    assert gates["verify"] == "passed"
    # The target re-indented the file, so the port must differ from the source
    # patch in range-diff terms. "identical" would mean the adaptation was lost.
    assert gates["semantic_equivalence"] == "modified"
    assert gates["completeness"] == "complete"

    # Phase artifacts: semantic triage, candidate patch, dossier, summary.
    triage = json.loads((target_run / "semantic_triage.json").read_text())
    assert triage["commit_type"] == "bugfix"
    patch = (target_run / "candidate.patch").read_text()
    assert "int c = 30;" in patch
    assert "<<<<<<<" not in patch
    assert "semantic_triage.json" not in patch

    dossier = (target_run / "dossier.md").read_text()
    assert "## Triage result" in dossier
    assert "## Semantic triage" in dossier
    assert "## Baseline (clean target branch)" in dossier
    assert "## Candidate patch" in dossier
    assert "## Semantic equivalence" in dossier
    assert "mechanical:" in dossier

    summary = json.loads((parent_run_dir / "summary.json").read_text())
    assert summary["total_targets"] == 1
    assert summary["succeeded"] == 1
    assert summary["failed"] == 0
    assert summary["targets"][0]["outcome"] == "success"
    assert summary["targets"][0]["dossier_path"] == "target/dossier.md"
    assert summary["targets"][0]["tracker_path"].startswith("target/SMOKE-1_")

    # The ported commits survive worktree cleanup on a preserved branch.
    assert not (tmp_path / "worktrees" / "SMOKE-1_target").exists()
    preserved = _sha(repo, "amalgamator/SMOKE-1")
    # The branch is a descendant of the target tip, not a copy of it.
    assert (
        _git(
            repo,
            "merge-base",
            "--is-ancestor",
            "target",
            preserved,
            check=False,
        ).returncode
        == 0
    )
    assert preserved != _sha(repo, "target")
    ported_file = _git(repo, "show", "amalgamator/SMOKE-1:lib/url.c").stdout
    assert "int c = 30;" in ported_file
    assert "<<<<<<<" not in ported_file
