import asyncio
from unittest.mock import AsyncMock

from common import agent_runner
from common.model_registry import ModelProfile


def test_explicit_provider_wins_over_model_inference(claude_cfg, monkeypatch):
    claude_cfg.profile = ModelProfile("proxy", "litellm", claude_cfg.model)
    generic = AsyncMock(return_value=[])
    native = AsyncMock(return_value=[])
    monkeypatch.setattr(agent_runner.LiteLLMAgentRunner, "run", generic)
    monkeypatch.setattr(agent_runner.ClaudeAgentRunner, "run", native)
    asyncio.run(agent_runner.agent_run(claude_cfg, []))
    generic.assert_awaited_once()
    native.assert_not_awaited()
