# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Agent-config tests deferred from Phase 2.

These require the CodeGenConfig agent section and the PR-command prompt
machinery that land in Phases 4-5 of the model_agnostic merge.
"""

import json
import shlex
from dataclasses import asdict

import pytest

from llm.auto_code_gen.agents.claude import native_config
from llm.auto_code_gen.code_gen_configs import CodeGenConfig


@pytest.mark.parametrize("mode", ["normal", "deep"])
def test_legacy_defaults_unchanged(mode, tmp_path):
    config = CodeGenConfig(output_dir=str(tmp_path), thinking_mode=mode)
    assert asdict(native_config(config.make_agent_config())) == asdict(
        config.make_claude_config()
    )


def test_workload_model_is_independent(tmp_path):
    config = CodeGenConfig(
        output_dir=str(tmp_path),
        model="workload/model",
        agent={"model": "openai/agent"},
    )
    assert config.make_agent_config().model == "openai/agent"
    assert config.model == "workload/model"


def test_agent_override_and_relative_key(tmp_path):
    directory = tmp_path / "with spaces"
    directory.mkdir()
    (directory / "key").write_text("test-secret\n")
    path = directory / "agent.json"
    path.write_text(json.dumps({"model": "openai/override", "api_key_file": "key"}))
    config = CodeGenConfig(output_dir=str(tmp_path), agent={"model": "openai/nested"})
    resolved = config.make_agent_config(str(path))
    assert resolved.model == "openai/override"
    assert resolved.api_key == "test-secret"
    assert "test-secret" not in repr(resolved)
    assert resolved.api_key_file == str(directory / "key")
    assert config.agent_config_path == str(path)
    assert config.make_agent_config().model == "openai/override"


def test_framework_json_loads_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(CodeGenConfig, "_load_from_cross_trace", lambda self: None)
    path = tmp_path / "codegen.json"
    path.write_text(
        json.dumps(
            {
                "cross_trace_config": str(path),
                "source_code_dir": str(tmp_path),
                "improvement_id": 1,
                "output_dir": str(tmp_path),
                "agent": {"model": "openai/test"},
            }
        )
    )
    assert CodeGenConfig.from_json(str(path)).make_agent_config().model == "openai/test"


def test_generated_followups_keep_override(tmp_path):
    from llm.auto_code_gen.code_gen_prompts import gen_GeneratePRCommandsPrompt

    override = str(tmp_path / "agent settings.json")
    prompt = gen_GeneratePRCommandsPrompt(
        "context",
        str(tmp_path),
        str(tmp_path),
        "branch",
        "main",
        str(tmp_path / "pipeline.json"),
        agent_config_path=override,
    ).prompt()
    assert prompt.count("--agent-config " + shlex.quote(override)) == 2
