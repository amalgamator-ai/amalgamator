# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared Claude Agent SDK glue: the agent runner and its config/step types.

This is the common ground every pipeline builds on. A package that wants its
own agent persona subclasses :class:`ClaudeConfig` and overrides
``system_prompt`` — see ``auto_bug_fix/claude_utils.py`` and
``llm/auto_code_gen/claude_utils.py``. Everything else (resume handling, message
printing, timing, token accounting) is identical across pipelines and lives
here only.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, cast

from colorama import Fore, Style

if TYPE_CHECKING:
    from common.model_registry import ModelProfile


def make_anthropic_client():
    """Create a raw Anthropic client for simple one-shot API calls.

    .. deprecated::
        Use :func:`common.llm_client.completion` instead. This function
        is kept for backward compatibility with callers that directly use
        the Anthropic ``messages.create`` interface.
    """
    import anthropic

    if os.environ.get("CLAUDE_CODE_USE_VERTEX"):
        from anthropic import AnthropicVertex

        return AnthropicVertex(
            project_id=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", ""),
            region=os.environ.get("CLOUD_ML_REGION", "us-east5"),
        )
    return anthropic.Anthropic()


try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ContentBlock,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )
    from claude_agent_sdk.types import StreamEvent

    _HAS_CLAUDE_SDK = True
except ImportError:
    _HAS_CLAUDE_SDK = False

# Background-agent tracking in _run_steps. Imported separately because older
# SDK releases lack these; the step loop degrades to plain result handling
# when they are unavailable.
try:
    from claude_agent_sdk.types import (
        TERMINAL_TASK_STATUSES,
        TaskNotificationMessage,
        TaskStartedMessage,
        TaskUpdatedMessage,
    )

    _HAS_TASK_TRACKING = True
except ImportError:
    _HAS_TASK_TRACKING = False


# Deliberately domain-neutral: a package that never picks a persona gets a
# sane generalist rather than silently inheriting another domain's expert.
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software engineer working in a large codebase."
)

PROFILING_SYSTEM_PROMPT = (
    "You are a benchmarking and profiling expert of LLMs that run via vLLM, "
    "SGLang and TRT"
)


@dataclass
class ClaudeConfig:
    model: str
    allowed_tools: list[str]
    perm_mode: str
    cwd: str | None
    thinking: dict | None = None
    effort: str = "max"
    max_thinking_tokens: int = 1048576
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    command_timeout: int | None = None
    context_length: int | None = None
    profile: ModelProfile | None = None
    # Provider compatibility mode: some LiteLLM models emit Claude-style
    # XML tool invocations as plain assistant text.  Off by default —
    # assistant prose (including fenced examples) must never execute.
    # When enabled, only a message whose entire content is XML tool blocks
    # is treated as an invocation.
    xml_tool_fallback: bool = False

    def __post_init__(self):
        if self.profile:
            self.model = self.profile.model


@dataclass
class ProfilingClaudeConfig(ClaudeConfig):
    """Persona for the GPU profiling and trace-analysis pipelines."""

    system_prompt: str = PROFILING_SYSTEM_PROMPT


@dataclass
class PipelineStep:
    name: str
    prompt: str | dict
    output_files: list[str] = field(default_factory=list)


def _all_steps_complete(steps: list[PipelineStep], output_dir: str) -> bool:
    for step in steps:
        if not step.output_files:
            continue
        for f in step.output_files:
            path = os.path.join(output_dir, f)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                return False
    return True


def _find_resume_index(steps: list[PipelineStep], output_dir: str) -> int:
    for i, step in enumerate(steps):
        if not step.output_files:
            continue
        all_exist = all(
            os.path.isfile(os.path.join(output_dir, f))
            and os.path.getsize(os.path.join(output_dir, f)) > 0
            for f in step.output_files
        )
        if not all_exist:
            # Include immediately preceding setup/cleanup steps, but never
            # delete and rerun the last successfully completed artifact step.
            boundary = i
            while boundary > 0 and not steps[boundary - 1].output_files:
                boundary -= 1
            return boundary
    return len(steps)


def _step_output_files_complete(step: PipelineStep, cwd: str) -> bool:
    """Return True if all output_files for a step exist and are non-empty."""
    if not step.output_files:
        return True
    for f in step.output_files:
        path = f if os.path.isabs(f) else os.path.join(cwd, f)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
    return True


def print_claude_config(config: ClaudeConfig) -> None:
    from common.model_identity import is_claude_model
    from common.model_registry import public_config_snapshot

    class_name = config.__class__.__name__
    is_claude = is_claude_model(config.model)

    # Fields only relevant to Claude — suppress for other models.
    _claude_only = {"thinking", "max_thinking_tokens"}

    print(f"{class_name}:")
    for f in fields(config):
        name = f.name
        if not is_claude and name in _claude_only:
            continue
        value = public_config_snapshot(getattr(config, name))
        print(f"  {name}: {value}")


def print_dict(d, indent):
    for key, val in d.items():
        if isinstance(val, dict):
            print(f"{indent}{key} : ")
            print_dict(val, indent + "    ")
        else:
            print(f"{indent}{key} : {val}")


def print_ContentBlock(prefix, content):
    if isinstance(content, TextBlock):
        print(f"{prefix} {content.text}")

    elif isinstance(content, ThinkingBlock):
        print(f"{prefix} Thinking: {content.signature} {content.thinking}")

    elif isinstance(content, ToolUseBlock):
        print(f"{prefix} ToolUse: name = {content.name}")

        print_dict(content.input, indent="    ")

    elif isinstance(content, ToolResultBlock):
        print(f"{prefix} ToolResult: err = {content.is_error}")

        print()
        print(content.content)

    else:
        raise Exception(f"Unexpected type(content) = {type(content)}")
    print()


def print_UserMessage(msg: UserMessage):
    prefix = f"{Fore.GREEN}User:{Style.RESET_ALL}"
    if isinstance(msg.content, str):
        print(f"{prefix} {msg.content}")
    else:
        for block in msg.content:
            assert isinstance(block, ContentBlock)
            print_ContentBlock(prefix, block)


def print_AssistantMessage(msg: AssistantMessage):
    prefix = f"{Fore.CYAN}Claude:{Style.RESET_ALL}"

    for block in msg.content:
        assert isinstance(block, ContentBlock)
        print_ContentBlock(prefix, block)

    if msg.error is not None:
        print(f"    -- ERROR = {msg.error}")


def print_SystemMessage(msg: SystemMessage):
    print(f"System: subtype = {msg.subtype} data = {msg.data}")


def print_ResultMessage(msg: ResultMessage):
    print(
        f"Result: subtype = {msg.subtype} dur = {msg.duration_ms} dur_api = {msg.duration_api_ms} err = {msg.is_error} num_turns = {msg.num_turns}"
    )

    if msg.usage is not None:
        print(f"    -- usage = {msg.usage}")
    if msg.total_cost_usd is not None:
        print(f"    -- cost = ${msg.total_cost_usd:.4f}")
    if msg.result is not None:
        print(f"    -- result = {msg.result}")


def print_StreamEvent(msg: StreamEvent):
    print(f"Stream: event = {msg.event}")


def print_Message(msg):
    if isinstance(msg, UserMessage):
        print_UserMessage(msg)
    elif isinstance(msg, AssistantMessage):
        print_AssistantMessage(msg)
    elif isinstance(msg, SystemMessage):
        print_SystemMessage(msg)
    elif isinstance(msg, ResultMessage):
        print_ResultMessage(msg)
    elif isinstance(msg, StreamEvent):
        print_StreamEvent(msg)
    else:
        raise Exception(f"Unexpected type(msg) = {type(msg)}")


def build_claude_options(config: ClaudeConfig):
    """Apply profile transport only to this SDK client's child process."""
    from common.model_registry import RegistryError

    env = {
        "ANTHROPIC_MODEL": config.model,
        "MAX_THINKING_TOKENS": str(config.max_thinking_tokens),
    }
    extra = {}
    profile = config.profile
    if profile and (profile.api_key or profile.api_base):
        backends = (
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_FOUNDRY",
        )
        auth_modes = (
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
            "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
        )
        conflicts = [
            key for key in backends if os.environ.get(key, "").lower() in ("1", "true")
        ]
        conflicts.extend(key for key in auth_modes if os.environ.get(key))
        if conflicts:
            raise RegistryError(
                "Explicit Anthropic profile transport conflicts with: "
                + ", ".join(conflicts)
            )
        if profile.model.startswith(("vertex_ai/", "bedrock/", "azure_ai/")):
            raise RegistryError(
                "Direct Anthropic transport requires a direct Anthropic model"
            )
        if profile.api_key:
            env["ANTHROPIC_API_KEY"] = profile.api_key
        if profile.api_base:
            env["ANTHROPIC_BASE_URL"] = profile.api_base
        # Explicit direct transport must not be rerouted by user/project settings.
        extra["setting_sources"] = []
    return ClaudeAgentOptions(
        system_prompt=config.system_prompt,
        allowed_tools=config.allowed_tools,
        permission_mode=cast(Any, config.perm_mode),
        cwd=config.cwd,
        env=env,
        model=config.model,
        thinking=cast(Any, config.thinking or {"type": "adaptive"}),
        effort=cast(Any, config.effort),
        **extra,  # pyrefly: ignore [bad-argument-type] (dynamic SDK option passthrough)
    )


@asynccontextmanager
async def claude_session(claude_config: ClaudeConfig):
    """Create a reusable Claude session that can be passed to claude_run().

    The session is built with the same options ``claude_run()`` would use
    (including profile transport), so a session-aware caller behaves exactly
    like the one-shot path minus per-step session recreation.
    """
    if not _HAS_CLAUDE_SDK:
        raise ImportError(
            "claude_agent_sdk is required for claude_session(). "
            "Install it or run inside the Claude Code environment."
        )
    assert claude_config.cwd is not None, "claude_config must have CWD set"
    options = build_claude_options(claude_config)
    async with ClaudeSDKClient(options=options) as client:
        yield client


async def claude_run(
    claude_config: ClaudeConfig,
    steps: list,
    resume: bool = False,
    client=None,
):
    if not _HAS_CLAUDE_SDK:
        raise ImportError(
            "claude_agent_sdk is required for claude_run(). "
            "Install it or run inside the Claude Code environment."
        )
    assert claude_config.cwd is not None, "claude_config must have CWD set"

    # Support legacy callers that pass raw strings/dicts
    wrapped_steps = []
    for i, s in enumerate(steps):
        if isinstance(s, PipelineStep):
            wrapped_steps.append(s)
        else:
            wrapped_steps.append(
                PipelineStep(
                    name=f"step_{i + 1}",
                    prompt=s,
                )
            )
    steps = wrapped_steps

    print_claude_config(claude_config)

    start_index = 0
    if resume:
        start_index = _find_resume_index(steps, claude_config.cwd)
        print(f"\n{Fore.YELLOW}=== RESUME SUMMARY ==={Style.RESET_ALL}\n")
        print(f"Total pipeline steps: {len(steps)}")

        if start_index > 0:
            # Log completed steps
            print(
                f"\n{Fore.GREEN}Completed steps (will be skipped):{Style.RESET_ALL}\n"
            )
            for i in range(start_index):
                s = steps[i]
                if s.output_files:
                    files_str = ", ".join(s.output_files)
                    print(f"  [{i + 1}] {s.name} -> {files_str}")
                else:
                    print(f"  [{i + 1}] {s.name} (command)")

            # Log the step being re-executed (potentially corrupted)
            restart_step = steps[start_index]
            print(
                f"\n{Fore.YELLOW}Re-executing step (last completed, potentially corrupted):{Style.RESET_ALL}\n"
            )
            print(f"  [{start_index + 1}] {restart_step.name}")
            for f in restart_step.output_files:
                path = os.path.join(claude_config.cwd, f)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    print(f"    Deleting: {f} ({size} bytes)")
                    os.remove(path)

            # Log remaining steps
            remaining = len(steps) - start_index
            print(
                f"\n{Fore.CYAN}Remaining steps to execute ({remaining}):{Style.RESET_ALL}\n"
            )
            for i in range(start_index, len(steps)):
                s = steps[i]
                if s.output_files:
                    files_str = ", ".join(s.output_files)
                    print(f"  [{i + 1}] {s.name} -> {files_str}")
                else:
                    print(f"  [{i + 1}] {s.name} (command)")
        else:
            print(
                f"{Fore.YELLOW}No completed steps found, starting from beginning{Style.RESET_ALL}"
            )

        print(f"\n{Fore.YELLOW}=== END RESUME SUMMARY ==={Style.RESET_ALL}")
        print()

    wall_start = time.time()

    if client is not None:
        result = await _run_steps(client, steps, start_index, cwd=claude_config.cwd)
        wall_duration = time.time() - wall_start
        for t in result:
            t["wall_overhead"] = wall_duration - sum(
                cast(float, s["duration"]) for s in result
            )
        return result

    options = build_claude_options(claude_config)

    session_start = time.time()
    async with ClaudeSDKClient(options=options) as new_client:
        session_create_time = time.time() - session_start
        result = await _run_steps(new_client, steps, start_index, cwd=claude_config.cwd)
        wall_duration = time.time() - wall_start
        for t in result:
            t["session_create_time"] = session_create_time
            t["wall_overhead"] = wall_duration - sum(
                cast(float, s["duration"]) for s in result
            )
        return result


async def _run_steps(active_client, steps, start_index, cwd=None):
    """Run pipeline steps against an active SDK client.

    Retries a step (up to 3 attempts) when the API reports an error and the
    step's output files are missing; if files were written despite the error
    the result is accepted. Tracks background agents spawned during a step
    when the SDK exposes task-tracking types.
    """
    step_timings = []
    for i, step in enumerate(steps):
        if i < start_index:
            continue

        prompt = step.prompt
        if isinstance(prompt, dict):
            if "fn" not in prompt:
                raise ValueError(
                    f"Unsupported dict step {step.name!r}: only callable 'fn' steps are "
                    "allowed. The former {'cmd': '<source>'} form was executed "
                    "via eval() and has been removed."
                )
            fn_name = prompt.get("fn_name", str(prompt["fn"]))
            print(f"{Fore.GREEN} RUN CMD: {fn_name} {Style.RESET_ALL}")
            prompt["fn"]()
            continue

        start_time = time.time()
        step_usage: dict[str, float | None] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": None,
        }
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(
                    f"{Fore.YELLOW}RETRY STEP {i + 1} ({step.name}) — attempt {attempt}/{max_retries}{Style.RESET_ALL}"
                )

            print(
                f"{Fore.GREEN} STEP {i + 1} ({step.name}): SEND QUERY PROMPT:{Style.RESET_ALL}"
            )
            print(
                f"{Fore.GREEN}============================================{Style.RESET_ALL}"
            )
            print(prompt)
            print(
                f"{Fore.CYAN}============================================{Style.RESET_ALL}"
            )
            print(f"{Fore.CYAN} RESPONSES:{Style.RESET_ALL}")
            print(
                f"{Fore.CYAN}============================================{Style.RESET_ALL}"
            )
            await active_client.query(prompt)

            # Track background agents spawned during this step. We must not advance
            # to the next step until every spawned agent reaches a terminal state,
            # otherwise their results (file writes) are lost when the next prompt
            # resets the session's active work.
            pending_tasks: set[str] = set()
            result_received = False
            api_error = False

            async for msg in active_client.receive_response():
                print_Message(msg)

                if isinstance(msg, ResultMessage):
                    if msg.usage:
                        step_usage["input_tokens"] += msg.usage.get("input_tokens", 0)
                        step_usage["output_tokens"] += msg.usage.get("output_tokens", 0)
                    if msg.total_cost_usd is not None:
                        previous_cost = step_usage["cost_usd"]
                        step_usage["cost_usd"] = (
                            previous_cost if previous_cost is not None else 0.0
                        ) + msg.total_cost_usd
                    # getattr: minimal test stubs may not carry is_error.
                    if getattr(msg, "is_error", False):
                        api_error = True
                    result_received = True

                elif _HAS_TASK_TRACKING and isinstance(msg, TaskStartedMessage):
                    pending_tasks.add(msg.task_id)
                    print(
                        f"{Fore.YELLOW}  [background] task started: {msg.task_id[:12]}{Style.RESET_ALL}"
                    )

                elif _HAS_TASK_TRACKING and isinstance(
                    msg, (TaskNotificationMessage, TaskUpdatedMessage)
                ):
                    if msg.status in TERMINAL_TASK_STATUSES:
                        pending_tasks.discard(msg.task_id)
                        print(
                            f"{Fore.YELLOW}  [background] task {msg.task_id[:12]}: {msg.status}{Style.RESET_ALL}"
                        )

                print(
                    f"{Fore.CYAN}--------------------------------------------{Style.RESET_ALL}"
                )

                # Only advance to the next step when the main turn is complete
                # AND all background agents have reached a terminal state.
                if result_received and not pending_tasks:
                    break

            print(
                f"{Fore.CYAN}============================================{Style.RESET_ALL}"
            )

            # If the API reported an error AND output files are missing, retry.
            # If files were written despite the error (truncated after writes),
            # accept the result and move on — no retry needed.
            if api_error and not _step_output_files_complete(step, cwd or ""):
                missing = [
                    f
                    for f in (step.output_files or [])
                    if not os.path.isfile(
                        f if os.path.isabs(f) else os.path.join(cwd or "", f)
                    )
                ]
                print(
                    f"{Fore.RED}API error on step {i + 1} ({step.name}) and output files missing: {missing}{Style.RESET_ALL}"
                )
                if attempt < max_retries:
                    print(f"{Fore.YELLOW}Retrying...{Style.RESET_ALL}")
                    continue
                else:
                    print(
                        f"{Fore.RED}Max retries reached for step {i + 1} ({step.name}). Proceeding anyway.{Style.RESET_ALL}"
                    )
            break  # success or no retry condition met

        duration_time = time.time() - start_time
        step_timings.append(
            {
                "name": step.name,
                "duration": duration_time,
                "start_time": start_time,
                "end_time": start_time + duration_time,
                "input_tokens": step_usage["input_tokens"],
                "output_tokens": step_usage["output_tokens"],
                "cost_usd": step_usage["cost_usd"],
            }
        )
        mins, secs = divmod(int(duration_time), 60)
        hrs, mins = divmod(mins, 60)
        human_time = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
        print(
            f"FINISHED STEP {i + 1} ({step.name}): duration = {duration_time:.1f}s ({human_time})"
        )

    return step_timings


# ── Model-agnostic dispatch ──────────────────────────────────────────────────
# Import here (not at module top) to avoid circular deps since agent_runner
# imports PipelineStep and print_claude_config from this module.


async def agent_run(config, steps, resume=False):
    """Model-agnostic agent entry point.

    Dispatches to the Claude Agent SDK for Claude models, or to a LiteLLM
    tool-use loop for everything else.  Drop-in replacement for
    ``claude_run()`` — same async signature, same return type.
    """
    from common.agent_runner import agent_run as _agent_run

    return await _agent_run(config, steps, resume=resume)
