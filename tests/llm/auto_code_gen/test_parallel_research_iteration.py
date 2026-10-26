"""Parallel research iteration (PRI) tests: config flags, launcher guards,
prompt construction, and the investigation-prompt consumption hook.

All tests are CPU-only and never spawn real agent sessions.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm.auto_code_gen import parallel_research_iteration as pri
from llm.auto_code_gen.code_gen_configs import CodeGenConfig
from llm.auto_code_gen.code_gen_prompts import (
    gen_InvestigateRuntimeOutputAndFixCodePrompt,
)


def _minimal_config_json(tmp_path, extra=None):
    cross = tmp_path / "cross.json"
    cross.write_text("{}")
    source = tmp_path / "src"
    source.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    data = {
        "cross_trace_config": str(cross),
        "improvement_id": 1,
        "source_code_dir": str(source),
        "output_dir": str(out),
    }
    data.update(extra or {})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


class _FakeProc:
    pid = 4242


def test_config_defaults():
    config = CodeGenConfig(output_dir="/tmp/x")
    assert config.enable_parallel_research_iteration is True
    assert config.parallel_research_iteration_agent_count == 4


def test_config_from_json_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(CodeGenConfig, "_load_from_cross_trace", lambda self: None)
    config = CodeGenConfig.from_json(str(_minimal_config_json(tmp_path)))
    assert config.enable_parallel_research_iteration is True
    assert config.parallel_research_iteration_agent_count == 4


def test_config_from_json_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(CodeGenConfig, "_load_from_cross_trace", lambda self: None)
    config = CodeGenConfig.from_json(str(_minimal_config_json(tmp_path, {
        "enable_parallel_research_iteration": False,
        "parallel_research_iteration_agent_count": 6,
    })))
    assert config.enable_parallel_research_iteration is False
    assert config.parallel_research_iteration_agent_count == 6


@pytest.mark.parametrize("payload", [
    {"parallel_research_iteration_agent_count": 0},
    {"parallel_research_iteration_agent_count": -1},
    {"parallel_research_iteration_agent_count": "four"},
    {"enable_parallel_research_iteration": "yes"},
])
def test_config_from_json_rejects_bad_values(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(CodeGenConfig, "_load_from_cross_trace", lambda self: None)
    with pytest.raises(ValueError):
        CodeGenConfig.from_json(str(_minimal_config_json(tmp_path, payload)))


def test_research_dir_and_proposals_file(tmp_path):
    research = pri.pri_research_dir(str(tmp_path), 7)
    assert research.endswith(os.path.join("research", "RI_over_V7"))
    # No proposals yet -> None
    assert pri.pri_proposals_file(str(tmp_path), 7) is None
    proposals = Path(research) / "proposals_final.md"
    proposals.parent.mkdir(parents=True)
    proposals.write_text("# proposals\n")
    assert pri.pri_proposals_file(str(tmp_path), 7) == str(proposals)


def test_launch_disabled_returns_none(tmp_path):
    config = CodeGenConfig(output_dir=str(tmp_path))
    config.enable_parallel_research_iteration = False
    assert pri.launch_parallel_research_iteration(config, 3) is None
    assert not (tmp_path / "research").exists()


def test_launch_skips_when_already_ran(tmp_path, monkeypatch):
    config = CodeGenConfig(output_dir=str(tmp_path))

    def _no_spawn(*a, **kw):
        raise AssertionError("Popen must not be called when status.json exists")

    monkeypatch.setattr(pri.subprocess, "Popen", _no_spawn)
    status = tmp_path / "research" / "RI_over_V3" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text('{"phase": "COMPLETE"}')
    assert pri.launch_parallel_research_iteration(config, 3) is None


def test_launch_spawns_detached_orchestrator(tmp_path, monkeypatch):
    config = CodeGenConfig(output_dir=str(tmp_path))
    config.agent_config_path = str(tmp_path / "agent.json")

    recorded = {}

    def _fake_popen(cmd, **kw):
        recorded["cmd"] = cmd
        recorded["kw"] = kw
        return _FakeProc()

    monkeypatch.setattr(pri.subprocess, "Popen", _fake_popen)
    research = pri.launch_parallel_research_iteration(config, 8)

    assert research == str(tmp_path / "research" / "RI_over_V8")
    cmd = recorded["cmd"]
    assert cmd[1] == "-m"
    assert cmd[2] == "llm.auto_code_gen.parallel_research_iteration"
    assert "--orchestrate" in cmd
    assert str(tmp_path) in cmd
    assert "8" in cmd
    assert "4" in cmd  # default agent count
    assert str(tmp_path / "agent.json") in cmd
    kw = recorded["kw"]
    assert kw["cwd"] == pri.REPO_ROOT
    assert kw["start_new_session"] is True
    assert (tmp_path / "research" / "RI_over_V8" / "orchestrator.pid"
            ).read_text().strip() == "4242"
    assert (tmp_path / "research" / "RI_over_V8" / "orchestrator.log").exists()


def test_r1_prompt_contains_grounding_and_rules(tmp_path):
    prompt = pri._r1_prompt(
        2, pri.PERSONAS[1], str(tmp_path), str(tmp_path / "out"), 9)
    assert "A2_r1.md" in prompt
    assert "iterations_history.txt" in prompt
    assert "V8/summary.txt" in prompt  # previous iteration grounding
    assert "V9" in prompt
    assert "HARD RULES" in prompt
    assert "Disable switch" in prompt


def test_agent_count_capped_by_persona_pool():
    assert len(pri.PERSONAS) >= 4
    # The orchestrator caps at the pool size; the launch path passes the
    # config value through, so just verify the pool supports the default.
    assert len(pri.PERSONAS[:4]) == 4


def _investigate_prompt(tmp_path, research_proposals_file=None):
    return gen_InvestigateRuntimeOutputAndFixCodePrompt(
        context="ctx",
        code_trace_files=["sg.py", "vllm.py"],
        code_port_plan_file="plan.md",
        test_plan_file="tests.md",
        runtime_logs_dir=str(tmp_path),
        iteration=9,
        prev_patch_file="runtime/V8/patch.patch",
        prev_summary_file="runtime/V8/summary.txt",
        iteration_history_summary_file="history.txt",
        research_proposals_file=research_proposals_file,
    ).prompt()


def test_investigate_prompt_includes_research_section(tmp_path):
    proposals = tmp_path / "proposals_final.md"
    proposals.write_text("# p")
    text = _investigate_prompt(tmp_path, str(proposals))
    assert "<research_proposals>" in text
    assert str(proposals) in text
    assert "cheap validation plan" in text


def test_investigate_prompt_without_research_section(tmp_path):
    text = _investigate_prompt(tmp_path, None)
    assert "<research_proposals>" not in text


def test_module_runs_standalone():
    rc = subprocess.run(
        [sys.executable, "-m", "llm.auto_code_gen.parallel_research_iteration",
         "--help"],
        cwd=pri.REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert rc.returncode == 0
    assert "--orchestrate" in rc.stdout
    assert "--run-agent" in rc.stdout


# ---------------------------------------------------------------------------
# Orchestrate() state machine, end to end, with the agent subprocess mocked:
# R1 -> SYNC1 -> R2 -> MERGE -> GATE -> ADJUDICATE -> handoff append.
# ---------------------------------------------------------------------------

def _fake_campaign(tmp_path):
    """Minimal campaign layout: runtime/V{N-1} promoted, V{N} running."""
    out = tmp_path / "campaign"
    for it in (7, 8):
        d = out / "runtime" / ("V%d" % it)
        d.mkdir(parents=True)
        (d / "patch.patch").write_text("patch V%d\n" % it)
    (out / "runtime" / "V7" / "summary.txt").write_text("V7 summary\n")
    return str(out)


def _install_fake_runner(monkeypatch, calls, tmp_path, output_dir,
                         complete_iteration_during_merge=True):
    """Mock _run_sub: record the call and materialize its expected output."""

    async def _fake_run_sub(name, prompt_path, log_path, timeout,
                            agent_config_path):
        calls.append(name)
        research = tmp_path / "campaign" / "research" / "RI_over_V8"
        if name.startswith("PRI_A") and name.endswith("_R1"):
            agent = name.split("_")[1]
            (research / "agents" / ("A%s_r1.md" % agent[1:])).write_text(
                "# r1 %s\n" % name)
        elif name == "PRI_MOD_R1":
            (research / "sync" / "digest_r1.md").write_text("# digest\n")
        elif name.startswith("PRI_A") and name.endswith("_R2"):
            agent = name.split("_")[1]
            (research / "agents" / ("A%s_r2.md" % agent[1:])).write_text(
                "# r2 %s\n" % name)
        elif name == "PRI_MOD_MERGE":
            (research / "proposals_draft.md").write_text("# draft\n")
            if complete_iteration_during_merge:
                # The runtime iteration finishes while the PRI works.
                (tmp_path / "campaign" / "runtime" / "V8" /
                 "summary.txt").write_text("V8 summary\n")
        elif name == "PRI_ADJUDICATE":
            (research / "proposals_final.md").write_text("# final\n")
            (research / "proposals.json").write_text("[]\n")
        else:
            raise AssertionError("unexpected agent call %r" % name)
        with open(log_path, "wb") as lf:
            lf.write(b"fake log")
        return True

    monkeypatch.setattr(pri, "_run_sub", _fake_run_sub)
    # Fast gate polling so the test does not wait a minute per poll.
    monkeypatch.setattr(pri, "GATE_POLL_S", 0.02)


def test_orchestrate_full_protocol(tmp_path, monkeypatch):
    import asyncio

    output_dir = _fake_campaign(tmp_path)
    calls = []
    _install_fake_runner(monkeypatch, calls, tmp_path, output_dir)

    asyncio.run(pri.orchestrate(output_dir, 8, 4, ""))

    # Phase order: 4x R1, moderator, 4x R2, merge, adjudicate.
    assert calls[:4] == ["PRI_A1_R1", "PRI_A2_R1", "PRI_A3_R1",
                         "PRI_A4_R1"]
    assert calls[4] == "PRI_MOD_R1"
    assert sorted(calls[5:9]) == ["PRI_A1_R2", "PRI_A2_R2", "PRI_A3_R2",
                                  "PRI_A4_R2"]
    assert calls[9] == "PRI_MOD_MERGE"
    assert calls[10] == "PRI_ADJUDICATE"
    assert len(calls) == 11

    research = tmp_path / "campaign" / "research" / "RI_over_V8"
    status = json.loads((research / "status.json").read_text())
    assert status["phase"] == "COMPLETE"
    # Every prompt was materialized for reproducibility.
    prompts = sorted(p.name for p in (research / "prompts").iterdir())
    assert prompts == ["A1_r1.txt", "A1_r2.txt", "A2_r1.txt", "A2_r2.txt",
                       "A3_r1.txt", "A3_r2.txt", "A4_r1.txt", "A4_r2.txt",
                       "adjudicate.txt", "mod_merge.txt", "mod_r1.txt"]
    # The handoff pointer landed in the iteration summary.
    summary = (tmp_path / "campaign" / "runtime" / "V8" /
               "summary.txt").read_text()
    assert "PARALLEL RESEARCH ITERATION (PRI) HANDOFF" in summary
    assert "proposals_final.md" in summary
    # And the investigation prompt hook sees the proposals.
    assert pri.pri_proposals_file(output_dir, 8) == str(
        research / "proposals_final.md")


def test_orchestrate_survives_failed_agents(tmp_path, monkeypatch):
    """One R1 agent timing out must not kill the protocol."""
    import asyncio

    output_dir = _fake_campaign(tmp_path)
    calls = []
    _install_fake_runner(monkeypatch, calls, tmp_path, output_dir)

    real_fake = pri._run_sub

    async def _flaky_run_sub(name, *args, **kwargs):
        if name == "PRI_A3_R1":
            calls.append(name)
            return False  # timed out; no A3_r1.md
        return await real_fake(name, *args, **kwargs)

    monkeypatch.setattr(pri, "_run_sub", _flaky_run_sub)
    asyncio.run(pri.orchestrate(output_dir, 8, 4, ""))

    research = tmp_path / "campaign" / "research" / "RI_over_V8"
    status = json.loads((research / "status.json").read_text())
    assert status["phase"] == "COMPLETE"
    # A3 has no round outputs; round 2 and the digest ran for the rest.
    assert not (research / "agents" / "A3_r2.md").exists()
    assert (research / "agents" / "A2_r2.md").exists()


def test_orchestrate_gate_timeout_still_adjudicates(tmp_path, monkeypatch):
    """If the runtime iteration never completes, the gate times out and the
    PRI still finalizes (with in-flight results) — no handoff append."""
    import asyncio

    output_dir = _fake_campaign(tmp_path)
    calls = []
    _install_fake_runner(monkeypatch, calls, tmp_path, output_dir,
                         complete_iteration_during_merge=False)
    monkeypatch.setattr(pri, "GATE_MAX_WAIT_S", 0.1)

    asyncio.run(pri.orchestrate(output_dir, 8, 2, ""))

    research = tmp_path / "campaign" / "research" / "RI_over_V8"
    status = json.loads((research / "status.json").read_text())
    assert status["phase"] == "COMPLETE"
    assert "PRI_ADJUDICATE" in calls
    # No summary -> no handoff block anywhere.
    assert not (tmp_path / "campaign" / "runtime" / "V8" /
                "summary.txt").exists()

