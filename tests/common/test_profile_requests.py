"""Capture actual request boundaries, without making model API calls."""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from common import agent_runner, llm_client
from common.model_registry import ModelProfile, RegistryError, load_registry


@pytest.fixture
def capture_request(monkeypatch, tmp_path):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    request = Mock(return_value=response)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=request))
    monkeypatch.setattr(agent_runner, "_litellm_completion_with_streaming", request)
    monkeypatch.setattr(llm_client, "_api_base", "http://stale-global/v1")
    monkeypatch.setattr(llm_client, "_api_key", "STALE_KEY")
    monkeypatch.setattr(llm_client, "_model_params", {"temperature": 0.99})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)

    def capture(path, profile):
        request.reset_mock()
        if path == "completion":
            llm_client.completion(profile.model, [], profile=profile)
        elif path == "agent":
            asyncio.run(agent_runner.LiteLLMAgentRunner()._agent_loop(
                model=profile.model, system_prompt="s", user_prompt="u", cwd=str(tmp_path),
                context_length=128000, profile=profile,
            ))
        else:
            agent_runner._summarize_stale(profile.model, [], 128000, profile=profile)
        request.assert_called_once()
        return request.call_args.kwargs
    return capture


@pytest.mark.parametrize("path", ["completion", "agent", "summary"])
@pytest.mark.parametrize("model", ["gpt-4o", "openai/gpt-4o", "anthropic/claude-sonnet-4-6",
                                  "together_ai/example", "ollama_chat/example"])
@pytest.mark.parametrize("endpoint", [None, "http://custom/v1"])
def test_profile_transport_forwarded_for_each_provider(capture_request, path, model, endpoint):
    profile = ModelProfile("test", "litellm", model, api_base=endpoint, api_key="PROFILE_KEY")
    sent = capture_request(path, profile)
    assert sent["api_key"] == "PROFILE_KEY"
    assert sent.get("api_base") == endpoint
    assert "temperature" not in sent


@pytest.mark.parametrize("path", ["completion", "agent", "summary"])
@pytest.mark.parametrize("model,endpoint,expected", [
    ("gpt-4o", None, None), ("openai/gpt-4o", None, None),
    ("openai/local", "http://local/v1", "dummy"), ("ollama/test", "http://ollama", None),
])
def test_dummy_key_only_for_custom_openai(capture_request, path, model, endpoint, expected):
    sent = capture_request(path, ModelProfile("test", "litellm", model, api_base=endpoint))
    assert sent.get("api_key") == expected


@pytest.mark.parametrize("field", ["model", "provider", "api_base", "api_key", "api_key_file"])
def test_registry_rejects_routing_fields_in_params(tmp_path, field):
    path = tmp_path / "models.yaml"
    path.write_text(f"models:\n  a:\n    model: gpt-4o\n    params:\n      {field}: SECRET\ndefault: a\n")
    with pytest.raises(RegistryError, match="top-level") as exc:
        load_registry(path)
    assert "SECRET" not in str(exc.value)


def test_explicit_claude_transport_does_not_inherit_vertex(capture_request, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    profile = ModelProfile("direct", "litellm", "claude-sonnet-4-6", api_key="DIRECT")
    assert capture_request("completion", profile)["model"] == profile.model
