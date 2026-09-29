import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import agent_runner, claude_utils
from common.model_registry import ModelProfile, RegistryError, load_registry


@pytest.fixture(autouse=True)
def clean_sdk_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_")):
            monkeypatch.delenv(name)


def test_sdk_dispatch_applies_key_file_and_isolates_clients(tmp_path, claude_cfg, monkeypatch):
    (tmp_path / "key").write_text("  PROFILE_KEY\n")
    path = tmp_path / "models.yaml"
    path.write_text(f"models:\n  direct:\n    model: claude-sonnet-4-6\n    api_base: https://proxy.invalid\n    api_key_file: {tmp_path / 'key'}\ndefault: direct\n")
    profiles = [load_registry(path).default_profile, ModelProfile("other", "anthropic", "claude-opus-4-6", api_key="OTHER")]
    captured = []
    class Client:
        def __init__(self, options): captured.append(options)
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
    monkeypatch.setattr(claude_utils, "ClaudeAgentOptions", SimpleNamespace)
    monkeypatch.setattr(claude_utils, "ClaudeSDKClient", Client)
    before = dict(os.environ)
    for profile in profiles:
        claude_cfg.profile, claude_cfg.model = profile, profile.model
        asyncio.run(agent_runner.agent_run(claude_cfg, []))
    assert captured[0].env["ANTHROPIC_API_KEY"] == "PROFILE_KEY"
    assert captured[0].env["ANTHROPIC_BASE_URL"] == "https://proxy.invalid"
    assert captured[1].env["ANTHROPIC_API_KEY"] == "OTHER"
    assert "ANTHROPIC_BASE_URL" not in captured[1].env
    assert captured[0].setting_sources == []
    assert dict(os.environ) == before


@pytest.mark.parametrize("conflict", ["CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_BEDROCK",
                                    "CLAUDE_CODE_USE_FOUNDRY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"])
def test_explicit_transport_rejects_conflicting_modes(claude_cfg, monkeypatch, conflict):
    monkeypatch.setenv(conflict, "1")
    claude_cfg.profile = ModelProfile("direct", "anthropic", claude_cfg.model, api_key="SECRET")
    with pytest.raises(RegistryError, match=conflict):
        claude_utils.build_claude_options(claude_cfg)


def test_environment_only_profile_preserves_sdk_defaults(claude_cfg, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setattr(claude_utils, "ClaudeAgentOptions", SimpleNamespace)
    claude_cfg.profile = ModelProfile("ambient", "anthropic", claude_cfg.model)
    options = claude_utils.build_claude_options(claude_cfg)
    assert options.env == {"ANTHROPIC_MODEL": claude_cfg.model, "MAX_THINKING_TOKENS": "1048576"}
    assert not hasattr(options, "setting_sources")


def test_installed_sdk_subprocess_transport_smoke():
    # Fresh interpreter bypasses conftest's SDK import stubs. No CLI is spawned.
    script = Path(__file__).with_name("sdk_transport_smoke.py")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
