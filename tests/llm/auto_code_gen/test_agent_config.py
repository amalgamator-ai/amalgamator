import json
from dataclasses import asdict
from pathlib import Path

import pytest

from llm.auto_code_gen.agents.config import AgentConfig, resolve_agent_config


def test_missing_explicit_key_fails(tmp_path):
    with pytest.raises(ValueError, match="Cannot read"):
        resolve_agent_config({"model": "openai/test", "api_key_file": "missing"}, config_path=str(tmp_path / "pipeline.json"))


@pytest.mark.parametrize("section", [
    [], {"unknown": 1}, {"model": ""}, {"model": "openai/test", "allowed_tools": ["mcp__x"]},
    {"model": "openai/test", "allowed_tools": "Bash"},
    {"model": "openai/test", "model_params": {"messages": []}},
    {"model": "openai/test", "summary_model_params": []},
    {"model": "openai/test", "summary_model_params": {"stream": False}},
    {"model": "openai/test", "write_model_params": []},
    {"model": "openai/test", "write_model_params": {"stream": False}},
    {"model": "openai/test", "model_params": {"max_tokens": 0}},
    {"backend": "claude_sdk", "model": "vertex_ai/claude-test"},
    {"model": "openai/test", "context_length": True},
    {"model": "openai/test", "max_agent_iterations": 1.5},
    {"model": "openai/test", "perm_mode": "plan"},
    {"model": "claude-test", "api_base": "https://example.invalid"},
])
def test_invalid_config_rejected(section):
    with pytest.raises(ValueError):
        resolve_agent_config(section)


def test_no_mutable_config_leak():
    section = {"model": "openai/test", "model_params": {"extra_body": {"a": 1}},
               "summary_model_params": {"extra_body": {"summary": True}},
               "write_model_params": {"extra_body": {"write": True}}}
    first = resolve_agent_config(section)
    first.model_params["extra_body"]["a"] = 2
    first.summary_model_params["extra_body"]["summary"] = False
    first.write_model_params["extra_body"]["write"] = False
    second = resolve_agent_config(section)
    assert second.model_params["extra_body"]["a"] == 1
    assert second.summary_model_params["extra_body"]["summary"] is True
    assert second.write_model_params["extra_body"]["write"] is True
    assert resolve_agent_config({"model": "openai/other"}).api_base is None

