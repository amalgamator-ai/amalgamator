"""Provider-neutral agent configuration helpers for ``auto_analyze``.

The implementation is shared with ``auto_code_gen`` so analysis and code
generation have identical provider selection, LiteLLM transport, tool-loop,
resume, and event-log behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.agents import AgentConfig, resolve_agent_config

DEFAULT_AGENT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "claude_config.json"
)


def add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the generic option while retaining the old spelling as an alias."""
    parser.add_argument(
        "--agent-config",
        "--claude-config",
        dest="agent_config",
        default=None,
        help=(
            "Path to an agent JSON config. --claude-config is accepted as a "
            "deprecated alias. If omitted, the config's 'agent' section is used; "
            "otherwise the legacy Claude default is used."
        ),
    )


def make_agent_config(
    config, cwd: str, agent_config_path: str | None = None
) -> AgentConfig:
    """Resolve an analysis config's embedded or external agent settings."""
    section = getattr(config, "agent", None)
    config_path = getattr(config, "config_json_path", "")
    mode = getattr(config, "thinking_mode", "deep")

    # Existing auto_analyze configs have no agent section. Preserve their
    # behavior by loading the checked-in Claude config only in that case.
    if agent_config_path is None and section is None:
        agent_config_path = str(DEFAULT_AGENT_CONFIG)

    return resolve_agent_config(
        section,
        cwd=str(Path(cwd).resolve()),
        mode=mode,
        config_path=config_path,
        agent_config_path=agent_config_path,
    )


def log_agent_config(config: AgentConfig) -> None:
    """Print provider-neutral, non-secret run configuration."""
    print(f"  Agent backend: {config.backend}")
    print(f"  Agent model:   {config.model}")
    print(f"  Tools:         {config.allowed_tools}")
    print(f"  CWD:           {config.cwd}")
