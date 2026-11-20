# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The reasoning_effort config knob reaches the provider request."""

import asyncio

from common import agent_runner
from common.agent_runner import LiteLLMAgentRunner
from common.claude_utils import ClaudeConfig


class _Message:
    tool_calls = None

    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    usage = None

    def __init__(self, content):
        self.choices = [_Choice(content)]


def _run(monkeypatch, **kwargs):
    seen: list[dict] = []

    def fake_completion(**request):
        seen.append(request)
        return _Response("done")

    def fake_build(model, *, profile=None, default_max_tokens, overrides=None):
        return {"overrides": overrides}

    monkeypatch.setattr(
        agent_runner, "_litellm_completion_with_streaming", fake_completion
    )
    monkeypatch.setattr("common.llm_client.build_request_kwargs", fake_build)
    monkeypatch.setattr(agent_runner, "_estimate_tokens", lambda *_a, **_k: 0)

    runner = LiteLLMAgentRunner()
    asyncio.run(
        runner._agent_loop(
            model="openai/test-model",
            system_prompt="system",
            user_prompt="user",
            cwd=".",
            allowed_tools=["Bash"],
            context_length=128000,
            **kwargs,
        )
    )
    return seen


def test_reasoning_effort_is_sent_as_request_override(monkeypatch):
    seen = _run(monkeypatch, reasoning_effort="medium")
    assert seen and seen[0]["overrides"] == {"reasoning_effort": "medium"}


def test_empty_reasoning_effort_keeps_provider_default(monkeypatch):
    seen = _run(monkeypatch)
    assert seen and seen[0]["overrides"] is None


def test_runner_passes_config_reasoning_effort(monkeypatch):
    import common.agent_runner as ar

    captured = {}

    async def fake_loop(self, **kwargs):
        captured.update(kwargs)
        return ar.StepTiming(name="")

    monkeypatch.setattr(LiteLLMAgentRunner, "_agent_loop", fake_loop)
    config = ClaudeConfig(
        model="openai/test-model",
        allowed_tools=["Bash"],
        perm_mode="acceptEdits",
        cwd=".",
        reasoning_effort="low",
    )
    asyncio.run(LiteLLMAgentRunner().run(config, ["do the thing"]))
    assert captured["reasoning_effort"] == "low"
