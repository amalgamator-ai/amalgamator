"""Model-agnostic agent runner with tool-use loop.

Provides two concrete runners behind a common interface:

* **ClaudeAgentRunner** — delegates to the existing ``claude_run()`` via the
  Claude Agent SDK.  Zero behaviour change for Claude users.
* **LiteLLMAgentRunner** — implements a tool-use loop on top of LiteLLM's
  function-calling support.  Works with any model that speaks the OpenAI
  tool-calling protocol (Qwen, Llama, Mistral, GPT, Gemini, …).

The top-level :func:`agent_run` dispatches automatically based on the model
name: Claude models go through the SDK, everything else through LiteLLM.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from colorama import Fore, Style

log = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 50
BASH_TIMEOUT_SECONDS = 300


# ── Tool definitions (OpenAI function-calling schema) ────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return stdout + stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


# ── Tool execution ───────────────────────────────────────────────────────────

def _normalize_args(name: str, arguments: dict) -> dict:
    """Fix common malformed argument patterns from weaker models.

    Some models emit ``{"argument_name": "command", "argument_value": "ls"}``
    instead of ``{"command": "ls"}``.  Detect and repair these.
    """
    if "argument_name" in arguments and "argument_value" in arguments:
        return {arguments["argument_name"]: arguments["argument_value"]}
    return arguments


def _execute_tool(name: str, arguments: dict, cwd: str) -> str:
    """Execute a single tool call and return the result as a string."""
    arguments = _normalize_args(name, arguments)
    try:
        if name == "read_file":
            return _tool_read_file(arguments["path"], cwd)
        elif name == "write_file":
            return _tool_write_file(arguments["path"], arguments["content"], cwd)
        elif name == "bash":
            return _tool_bash(arguments["command"], cwd)
        else:
            return f"Error: unknown tool '{name}'"
    except KeyError as e:
        return f"Error: missing required argument {e} for tool '{name}'. Expected args: {_expected_args(name)}"


def _expected_args(name: str) -> str:
    specs = {
        "read_file": "path",
        "write_file": "path, content",
        "bash": "command",
    }
    return specs.get(name, "unknown")


def _tool_read_file(path: str, cwd: str) -> str:
    resolved = Path(path) if Path(path).is_absolute() else Path(cwd) / path
    try:
        content = resolved.read_text(errors="replace")
        if len(content) > 50000:
            return content[:50000] + "\n\n[... truncated at 50000 chars]"
        return content
    except Exception as e:
        return f"Error reading {resolved}: {e}"


def _tool_write_file(path: str, content: str, cwd: str) -> str:
    resolved = Path(path) if Path(path).is_absolute() else Path(cwd) / path
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Successfully wrote {len(content)} chars to {resolved}"
    except Exception as e:
        return f"Error writing {resolved}: {e}"


def _tool_bash(command: str, cwd: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SECONDS,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        if len(output) > 30000:
            output = output[:30000] + "\n\n[... truncated at 30000 chars]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {BASH_TIMEOUT_SECONDS}s"
    except Exception as e:
        return f"Error running command: {e}"


# ── Step timing record ───────────────────────────────────────────────────────

@dataclass
class StepTiming:
    name: str
    duration: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


# ── Abstract base ────────────────────────────────────────────────────────────

class AgentRunner(ABC):
    @abstractmethod
    async def run(
        self,
        config,
        steps: list,
        resume: bool = False,
    ) -> list[dict]:
        """Run a list of pipeline steps, returning per-step timing dicts."""
        ...


# ── Claude Agent SDK runner ──────────────────────────────────────────────────

class ClaudeAgentRunner(AgentRunner):
    """Thin wrapper around the existing ``claude_run()``."""

    async def run(self, config, steps, resume=False):
        from common.claude_utils import claude_run as _claude_run
        return await _claude_run(config, steps, resume=resume)


# ── LiteLLM agent runner (generic tool-use loop) ────────────────────────────

class LiteLLMAgentRunner(AgentRunner):
    """Agentic tool-use loop powered by LiteLLM function calling.

    Works with any model that supports OpenAI-compatible tool/function
    calling: Qwen 2.5+, Llama 3.1+, Mistral, GPT-4o, Gemini, etc.
    """

    async def run(self, config, steps, resume=False):
        from common.claude_utils import PipelineStep, print_claude_config
        from common.llm_client import _resolve_model

        wrapped_steps = []
        for i, s in enumerate(steps):
            if isinstance(s, PipelineStep):
                wrapped_steps.append(s)
            else:
                wrapped_steps.append(PipelineStep(
                    name=f"step_{i + 1}",
                    prompt=s,
                ))
        steps = wrapped_steps

        print_claude_config(config)
        effective_model = _resolve_model(config.model)

        step_timings = []

        for i, step in enumerate(steps):
            prompt = step.prompt

            if isinstance(prompt, dict):
                if "fn" not in prompt:
                    raise ValueError(
                        f"Unsupported dict step {step.name!r}: only callable "
                        f"'fn' steps are allowed."
                    )
                fn_name = prompt.get("fn_name", str(prompt["fn"]))
                print(f"{Fore.GREEN} RUN CMD: {fn_name} {Style.RESET_ALL}")
                prompt["fn"]()
                continue

            start_time = time.time()
            print(f"{Fore.GREEN} STEP {i + 1} ({step.name}): "
                  f"SEND QUERY PROMPT:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'=' * 44}{Style.RESET_ALL}")
            print(prompt)
            print(f"{Fore.CYAN}{'=' * 44}{Style.RESET_ALL}")

            timing = await self._agent_loop(
                model=effective_model,
                system_prompt=config.system_prompt,
                user_prompt=prompt,
                cwd=config.cwd,
            )
            timing.name = step.name

            duration = time.time() - start_time
            timing.duration = duration
            mins, secs = divmod(int(duration), 60)
            hrs, mins = divmod(mins, 60)
            human = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
            print(f"FINISHED STEP {i + 1} ({step.name}): "
                  f"duration = {duration:.1f}s ({human})")

            step_timings.append({
                "name": timing.name,
                "duration": timing.duration,
                "input_tokens": timing.input_tokens,
                "output_tokens": timing.output_tokens,
                "cost_usd": timing.cost_usd,
            })

        return step_timings

    async def _agent_loop(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        cwd: str,
    ) -> StepTiming:
        """Core tool-use loop: prompt -> tool calls -> execute -> repeat.

        Includes duplicate-call detection to break out of loops where a
        smaller model keeps calling the same tool with the same arguments.
        """
        import litellm
        from common.llm_client import (
            _is_ollama_model, _strip_thinking,
            get_api_base, get_api_key, get_model_params,
        )

        # Start with global model_params, then layer on transport config.
        extra_kwargs: dict = {**get_model_params()}
        if _is_ollama_model(model):
            extra_kwargs.setdefault("extra_body", {})
            extra_kwargs["extra_body"].setdefault("think", False)

        api_base = get_api_base()
        if api_base and model.startswith("openai/"):
            extra_kwargs.setdefault("api_base", api_base)
            extra_kwargs.setdefault("api_key", get_api_key() or "dummy")

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        timing = StepTiming(name="")
        iteration = 0
        last_tool_sig: str | None = None
        repeat_count = 0

        while iteration < MAX_AGENT_ITERATIONS:
            iteration += 1
            log.info("Agent iteration %d/%d (model=%s)",
                     iteration, MAX_AGENT_ITERATIONS, model)

            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    max_tokens=16384,
                    **extra_kwargs,
                )
            except Exception as e:
                log.error("LLM call failed on iteration %d: %s", iteration, e)
                print(f"{Fore.RED}LLM error: {e}{Style.RESET_ALL}")
                break

            usage = response.usage
            if usage:
                timing.input_tokens += getattr(usage, "prompt_tokens", 0)
                timing.output_tokens += getattr(usage, "completion_tokens", 0)

            choice = response.choices[0]
            message = choice.message

            content_text = _strip_thinking(message.content or "")

            assistant_msg: dict = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            if content_text:
                print(f"{Fore.CYAN}Agent:{Style.RESET_ALL} {content_text[:500]}")

            if not message.tool_calls:
                log.info("Agent finished (no tool calls) on iteration %d", iteration)
                break

            tool_sig = "|".join(
                f"{tc.function.name}:{tc.function.arguments}"
                for tc in message.tool_calls
            )
            if tool_sig == last_tool_sig:
                repeat_count += 1
                if repeat_count >= 3:
                    log.warning("Breaking loop: same tool call repeated %d times", repeat_count + 1)
                    print(f"{Fore.RED}Agent loop detected — same tool call repeated "
                          f"{repeat_count + 1} times, stopping.{Style.RESET_ALL}")
                    messages.append({
                        "role": "user",
                        "content": "You have already called the same tool with the same "
                                   "arguments multiple times. Please provide your final "
                                   "answer now based on the information you have gathered.",
                    })
                    try:
                        final = litellm.completion(
                            model=model, messages=messages, max_tokens=4096,
                            **extra_kwargs,
                        )
                        final_text = _strip_thinking(
                            final.choices[0].message.content or "")
                        if final_text:
                            print(f"{Fore.CYAN}Agent:{Style.RESET_ALL} {final_text[:500]}")
                        if final.usage:
                            timing.input_tokens += getattr(final.usage, "prompt_tokens", 0)
                            timing.output_tokens += getattr(final.usage, "completion_tokens", 0)
                    except Exception:
                        pass
                    break
            else:
                repeat_count = 0
            last_tool_sig = tool_sig

            for tc in message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                print(f"{Fore.YELLOW}  Tool: {fn_name}({_summarize_args(fn_args)}){Style.RESET_ALL}")
                result = _execute_tool(fn_name, fn_args, cwd)
                print(f"  Result: {result[:200]}{'...' if len(result) > 200 else ''}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            log.warning("Agent hit max iterations (%d)", MAX_AGENT_ITERATIONS)

        return timing


def _summarize_args(args: dict) -> str:
    """Short summary of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


# ── Dispatch ─────────────────────────────────────────────────────────────────

_claude_runner = ClaudeAgentRunner()
_litellm_runner = LiteLLMAgentRunner()


async def agent_run(config, steps, resume=False):
    """Dispatch to the correct runner based on the model name.

    Claude models → Claude Agent SDK (existing behaviour).
    Everything else → LiteLLM tool-use loop.
    """
    from common.llm_client import is_claude_model

    if is_claude_model(config.model):
        log.info("Using Claude Agent SDK for model %s", config.model)
        return await _claude_runner.run(config, steps, resume=resume)
    else:
        log.info("Using LiteLLM agent loop for model %s", config.model)
        return await _litellm_runner.run(config, steps, resume=resume)
