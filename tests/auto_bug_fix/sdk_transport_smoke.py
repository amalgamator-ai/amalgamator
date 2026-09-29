"""Real SDK adapter smoke check; stop at process creation before any network I/O."""
import asyncio
import os
import sys
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from claude_agent_sdk._internal.transport import subprocess_cli
from common.claude_utils import ClaudeConfig, build_claude_options
from common.model_registry import ModelProfile, RegistryError


class CapturedSpawn(BaseException):
    pass


async def check():
    profile = ModelProfile("direct", "anthropic", "claude-sonnet-4-6",
                           api_key="PROFILE_KEY", api_base="https://profile.invalid")
    config = ClaudeConfig(profile.model, [], "acceptEdits", os.getcwd(), profile=profile)
    ambient = {"ANTHROPIC_API_KEY": "AMBIENT", "ANTHROPIC_BASE_URL": "https://ambient.invalid",
               "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1"}
    with patch.dict(os.environ, ambient, clear=True):
        options = replace(build_claude_options(config), cli_path=sys.executable)
        spawn = AsyncMock(side_effect=CapturedSpawn)
        with patch.object(subprocess_cli.anyio, "open_process", spawn):
            try:
                await subprocess_cli.SubprocessCLITransport("smoke", options).connect()
            except CapturedSpawn:
                pass
        spawn.assert_awaited_once()
        sent = spawn.call_args.kwargs["env"]
        assert sent["ANTHROPIC_API_KEY"] == "PROFILE_KEY"
        assert sent["ANTHROPIC_BASE_URL"] == "https://profile.invalid"
        assert any(arg in ("--setting-sources", "--setting-sources=") for arg in spawn.call_args.args[0])
        assert os.environ["ANTHROPIC_API_KEY"] == "AMBIENT"
        with patch.dict(os.environ, {"CLAUDE_CODE_USE_VERTEX": "1"}):
            try:
                build_claude_options(config)
            except RegistryError:
                pass
            else:
                raise AssertionError("Conflicting backend accepted")


if __name__ == "__main__":
    asyncio.run(check())
