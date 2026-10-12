"""Shared agent configuration with no process-global provider settings."""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from common.claude_utils import PROFILING_SYSTEM_PROMPT

# Single source of truth: the codegen persona is the profiling persona the
# Claude SDK path has always run with (see common.claude_utils).
SYSTEM_PROMPT = PROFILING_SYSTEM_PROMPT
MODE_PARAMS = {
    "deep": {
        "model": "claude-opus-4-6[1m]",
        "thinking": {"type": "adaptive"},
        "effort": "max",
        "max_thinking_tokens": 1048576,
    },
    "normal": {
        "model": "claude-sonnet-4-6",
        "thinking": {"type": "adaptive"},
        "effort": "medium",
        "max_thinking_tokens": 65536,
    },
}
TOOL_NAMES = {"Read": "read_file", "Write": "write_file", "Bash": "bash"}
RESERVED_PARAMS = {
    "model",
    "messages",
    "tools",
    "tool_choice",
    "api_base",
    "api_key",
    "stream",
    "stream_options",
    "timeout",
    "num_retries",
    "max_retries",
}


@dataclass
class AgentConfig:
    model: str
    cwd: str | None
    backend: str = "claude_sdk"
    allowed_tools: list[str] = field(default_factory=lambda: ["Read", "Write", "Bash"])
    perm_mode: str = "acceptEdits"
    thinking: dict | None = None
    effort: str = "max"
    max_thinking_tokens: int = 1048576
    system_prompt: str = SYSTEM_PROMPT
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_key_file: str | None = None
    model_params: dict = field(default_factory=dict)
    summary_model_params: dict = field(default_factory=dict)
    write_model_params: dict = field(default_factory=dict)
    context_length: int | None = None
    command_timeout: float | None = 1800
    request_timeout: float = 300
    max_agent_iterations: int | None = 100
    max_retries: int = 2
    stream: bool = True
    billing: str = "provider"

    def validate(self):
        if self.backend not in {"claude_sdk", "litellm"}:
            raise ValueError("agent.backend must be claude_sdk or litellm")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("agent.model must be a non-empty string")
        if not isinstance(self.allowed_tools, list) or any(
            not isinstance(t, str) for t in self.allowed_tools
        ):
            raise ValueError("agent.allowed_tools must be a list of strings")
        if not isinstance(self.model_params, dict):
            raise ValueError("agent.model_params must be a mapping")
        if not isinstance(self.summary_model_params, dict):
            raise ValueError("agent.summary_model_params must be a mapping")
        if not isinstance(self.write_model_params, dict):
            raise ValueError("agent.write_model_params must be a mapping")
        if RESERVED_PARAMS.intersection(self.model_params):
            raise ValueError("agent.model_params contains reserved transport fields")
        if RESERVED_PARAMS.intersection(self.summary_model_params):
            raise ValueError(
                "agent.summary_model_params contains reserved transport fields"
            )
        if RESERVED_PARAMS.intersection(self.write_model_params):
            raise ValueError(
                "agent.write_model_params contains reserved transport fields"
            )
        for name in (
            "context_length",
            "command_timeout",
            "request_timeout",
            "max_agent_iterations",
            "max_retries",
        ):
            value = getattr(self, name)
            if value is None and name in {
                "context_length",
                "command_timeout",
                "max_agent_iterations",
            }:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "max_retries")
            ):
                raise ValueError(
                    f"agent.{name} must be positive (max_retries may be zero)"
                )
            if name in {
                "context_length",
                "max_agent_iterations",
                "max_retries",
            } and not isinstance(value, int):
                raise ValueError(f"agent.{name} must be an integer")
        if not isinstance(self.stream, bool):
            raise ValueError("agent.stream must be boolean")
        for name in ("api_base", "api_key", "api_key_file"):
            if getattr(self, name) is not None and not isinstance(
                getattr(self, name), str
            ):
                raise ValueError(f"agent.{name} must be a string")
        if self.billing not in {"provider", "self_hosted"}:
            raise ValueError("agent.billing must be provider or self_hosted")
        if self.backend == "claude_sdk":
            if not self.model.startswith("claude-"):
                raise ValueError("claude_sdk requires a bare claude-* model name")
            if (
                self.api_base
                or self.api_key
                or self.api_key_file
                or self.model_params
                or self.summary_model_params
                or self.write_model_params
            ):
                raise ValueError(
                    "Claude SDK uses its existing environment authentication; agent transport settings require litellm"
                )
            if self.system_prompt != SYSTEM_PROMPT:
                raise ValueError(
                    "The Claude adapter preserves the existing codegen system prompt"
                )
        else:
            unknown = (
                set(self.allowed_tools) - set(TOOL_NAMES) - set(TOOL_NAMES.values())
            )
            if unknown:
                raise ValueError(
                    f"Unsupported generic agent tools: {', '.join(sorted(unknown))}"
                )
            if "[1m]" in self.model:
                raise ValueError(
                    "[1m] is Claude SDK syntax, not a LiteLLM model identifier"
                )
            if self.perm_mode != "acceptEdits":
                raise ValueError(
                    "Generic agents support acceptEdits only: configured tools execute with process permissions"
                )
        max_tokens = self.model_params.get("max_tokens", 8192)
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("agent.model_params.max_tokens must be a positive integer")
        if not isinstance(self.system_prompt, str) or not self.system_prompt:
            raise ValueError("agent.system_prompt must be a non-empty string")
        return self


def resolve_agent_config(
    section=None, *, cwd=None, mode="deep", config_path="", agent_config_path=None
):
    if mode not in MODE_PARAMS:
        raise ValueError("thinking-mode must be normal or deep")
    if agent_config_path:
        owner = Path(agent_config_path).expanduser().resolve()
        with owner.open() as handle:
            section = json.load(handle)
        if not isinstance(section, dict):
            raise ValueError("Agent config file must contain an object")
        section = section.get("agent", section)
    else:
        owner = (
            Path(config_path).resolve() if config_path else Path.cwd() / "config.json"
        )
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("agent must be an object")
    section = copy.deepcopy(section)
    valid = {f.name for f in fields(AgentConfig)} - {"cwd"}
    unknown = set(section) - valid
    if unknown:
        raise ValueError(f"Unknown agent fields: {', '.join(sorted(unknown))}")
    model = section.get("model", MODE_PARAMS[mode]["model"])
    backend = section.get(
        "backend",
        (
            "claude_sdk"
            if isinstance(model, str) and model.startswith("claude-")
            else "litellm"
        ),
    )
    defaults = (
        copy.deepcopy(MODE_PARAMS[mode])
        if backend == "claude_sdk"
        else {"model": model}
    )
    # Fields are runtime-validated against AgentConfig above (unknown -> error), so
    # this dynamic merge cannot be verified statically.
    config = AgentConfig(
        cwd=cwd,
        **{  # pyrefly: ignore [bad-argument-type]
            **defaults,
            **section,
            "backend": backend,
        },
    )
    config.validate()
    if config.api_key_file:
        key_path = Path(os.path.expandvars(config.api_key_file)).expanduser()
        if not key_path.is_absolute():
            key_path = owner.parent / key_path
        config.api_key_file = str(key_path.resolve())
        if config.api_key is None:
            try:
                config.api_key = key_path.read_text().strip()
            except OSError as exc:
                raise ValueError(f"Cannot read agent API key file: {key_path}") from exc
            if not config.api_key:
                raise ValueError("Agent API key file is empty")
    return config


def add_agent_argument(parser):
    parser.add_argument(
        "--agent-config", help="Agent JSON file overriding the pipeline's agent section"
    )
