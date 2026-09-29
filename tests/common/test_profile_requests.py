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


@pytest.mark.parametrize("path,default", [("completion", 4096), ("agent", 16384), ("summary", 32768)])
@pytest.mark.parametrize("limit", [None, "max_tokens", "max_completion_tokens"])
def test_shared_parameters_and_output_limits(capture_request, path, default, limit):
    params = {"temperature": 0.2, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if limit:
        params[limit] = 512
    profile = ModelProfile("test", "litellm", "openai/test", api_base="http://local/v1", params=params)
    sent = capture_request(path, profile)
    assert sent["temperature"] == 0.2
    assert sent["extra_body"] == params["extra_body"]
    assert sent[limit or "max_tokens"] == (512 if limit else default)
    if limit == "max_completion_tokens":
        assert "max_tokens" not in sent
    if path == "summary":
        assert "tools" not in sent
    sent["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
    assert profile.params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("path", ["completion", "agent", "summary"])
def test_ollama_defaults_do_not_mutate_nested_params(capture_request, path):
    profile = ModelProfile("test", "litellm", "ollama/test", params={"extra_body": {"nested": []}})
    sent = capture_request(path, profile)
    assert sent["extra_body"]["think"] is False
    sent["extra_body"]["nested"].append("changed")
    assert profile.params == {"extra_body": {"nested": []}}


def test_request_override_precedence_and_isolation(monkeypatch):
    monkeypatch.setattr(llm_client, "_model_params", {"extra_body": {"think": True}})
    overrides = {"temperature": 0.9, "api_key": "OVERRIDE", "extra_body": {"think": True}}
    profile = ModelProfile("test", "litellm", "ollama/test", api_key="PROFILE",
                           params={"temperature": 0.1, "max_tokens": 100})
    sent = llm_client.build_request_kwargs(profile.model, profile=profile, default_max_tokens=200, overrides=overrides)
    assert sent["api_key"] == "OVERRIDE"
    assert sent["temperature"] == 0.9
    assert sent["max_tokens"] == 100
    sent["extra_body"]["think"] = False
    assert overrides["extra_body"]["think"] is True
    legacy = llm_client.build_request_kwargs("ollama/test", default_max_tokens=200)
    legacy["extra_body"]["think"] = False
    assert llm_client.get_model_params()["extra_body"]["think"] is True
