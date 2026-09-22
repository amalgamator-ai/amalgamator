"""Model-agnostic agent runner with tool-use loop.

Provides two concrete runners behind a common interface:

* **ClaudeAgentRunner** — delegates to the existing ``claude_run()`` via the
  Claude Agent SDK.  Zero behaviour change for Claude users.
* **LiteLLMAgentRunner** — implements a tool-use loop on top of LiteLLM's
  function-calling support.  Works with any model that speaks the OpenAI
  tool-calling protocol (Qwen, Llama, Mistral, GPT, Gemini, …).

The top-level :func:`agent_run` dispatches automatically based on the model
name: Claude models go through the SDK, everything else through LiteLLM.

Context window management
~~~~~~~~~~~~~~~~~~~~~~~~~
The LiteLLM loop implements adaptive context management so it works within
any model's context limit (100K, 262K, etc.):

* **Tier 1** – tool output caps scale with the context budget.
* **Tier 2** – a pre-call token estimate triggers compaction at 80%.
* **Tier 3** – anchored iterative summarization compresses the stale
  middle of the conversation while preserving the system prompt, original
  task, and recent tail.
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

MAX_AGENT_ITERATIONS = 9999  # Effectively unlimited — agent stops when it has no more tool calls


def _litellm_completion_with_streaming(stream: bool = True, **kwargs):
    """Call litellm.completion(), using streaming to avoid proxy timeouts.

    When stream=True (default), collects all chunks and reconstructs a
    response object that is API-compatible with the non-streaming response.
    This prevents OpenShift/nginx proxy 504 timeouts because the gateway sees
    data flowing from the first token onward.

    Falls back to non-streaming if the model/endpoint doesn't support it.
    """
    import litellm

    if not stream:
        return litellm.completion(**kwargs)

    try:
        response_stream = litellm.completion(stream=True, **kwargs)
        return litellm.stream_chunk_builder(
            list(response_stream),
            messages=kwargs.get("messages", []),
        )
    except Exception as e:
        # If streaming fails (model doesn't support it), fall back gracefully
        log.warning("Streaming failed (%s), retrying without streaming", e)
        return litellm.completion(**kwargs)


DEFAULT_BASH_TIMEOUT = 600
DEFAULT_CONTEXT_LENGTH = 128_000
COMPACTION_THRESHOLD_PCT = 0.80
KEEP_RECENT_TOKENS = 40_000


# ── Tool definitions (OpenAI function-calling schema) ────────────────────────

_CLAUDE_TO_LITELLM_TOOL = {
    "Read": "read_file",
    "Write": "write_file",
    "Bash": "bash",
}

ALL_TOOL_SCHEMAS = [
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


def _filter_tools(allowed_tools: list[str] | None) -> list[dict]:
    """Return tool schemas filtered by the config's allowed_tools list.

    Maps Claude SDK names (Read, Write, Bash) to LiteLLM names
    (read_file, write_file, bash).  Returns all tools if the allowlist
    is empty or None.
    """
    if not allowed_tools:
        return ALL_TOOL_SCHEMAS
    allowed = {_CLAUDE_TO_LITELLM_TOOL.get(t, t) for t in allowed_tools}
    filtered = [s for s in ALL_TOOL_SCHEMAS if s["function"]["name"] in allowed]
    return filtered or ALL_TOOL_SCHEMAS


# ── Context window management ────────────────────────────────────────────────

def _resolve_context_length(model: str, configured: int | None) -> int:
    """Determine the context window size for the model.

    Priority: explicit config > litellm model info > default 128K.
    """
    if configured and configured > 0:
        return configured
    try:
        import litellm
        info = litellm.get_model_info(model)
        max_input = info.get("max_input_tokens") or info.get("max_tokens")
        if max_input and max_input > 0:
            return max_input
    except Exception:
        pass
    return DEFAULT_CONTEXT_LENGTH


def _tool_output_caps(context_length: int) -> tuple[int, int]:
    """Return (read_cap, bash_cap) in chars, scaled to context budget."""
    if context_length >= 200_000:
        return 50_000, 30_000
    if context_length >= 100_000:
        return 20_000, 15_000
    return 10_000, 8_000


def _estimate_tokens(model: str, messages: list[dict]) -> int:
    """Fast local token estimate via LiteLLM (tiktoken fallback)."""
    try:
        import litellm
        return litellm.token_counter(model=model, messages=messages)
    except Exception:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        for m in messages:
            for tc in m.get("tool_calls", []):
                total_chars += len(tc.get("function", {}).get("arguments", ""))
        return total_chars // 4


def _find_pair_boundary(messages: list[dict], idx: int) -> int:
    """Walk backward from *idx* to the start of its tool_call group.

    If *idx* points at a ``tool`` result, walk back past consecutive
    ``tool`` messages to find the parent ``assistant`` message (the one
    with ``tool_calls``).  Return that index so the whole group is
    included in the tail.

    Stops after finding the *current* group's assistant — does **not**
    continue through earlier groups.
    """
    if idx <= 0 or idx >= len(messages):
        return max(idx, 0)

    if messages[idx]["role"] != "tool":
        return idx

    while idx > 0 and messages[idx]["role"] == "tool":
        idx -= 1

    return idx


def _compact_messages(
    model: str,
    messages: list[dict],
    context_length: int,
    keep_recent_tokens: int = KEEP_RECENT_TOKENS,
    timing: StepTiming | None = None,
    profile=None,
) -> list[dict]:
    """Compress the message history to fit within the context budget.

    Strategy (anchored iterative summarization):
    1. Protect the first 2 messages (system + original user task).
    2. Walk backward from the end to identify a "recent tail" that fits
       within *keep_recent_tokens*.  Never split tool_call/tool pairs.
    3. Summarize the "stale middle" via a cheap LLM call.
    4. If summarization fails, fall back to a simple sliding window that
       drops oldest tool_call/tool pairs until under budget.

    Returns a new list; the original is not mutated.
    """
    budget = int(context_length * COMPACTION_THRESHOLD_PCT)

    # ── 1. Identify protected head (system + user) ───────────────────────
    head = messages[:2]

    # ── 2. Walk backward to find recent tail ─────────────────────────────
    tail_start = len(messages)
    tail_tokens = 0
    while tail_start > 2:
        candidate = tail_start - 1
        msg = messages[candidate]
        msg_tokens = _estimate_tokens(model, [msg])
        if tail_tokens + msg_tokens > keep_recent_tokens:
            break
        tail_tokens += msg_tokens
        tail_start = candidate

    tail_start = _find_pair_boundary(messages, tail_start)
    if tail_start <= 2:
        log.warning("Context compaction: recent tail covers entire history, "
                    "nothing to compact")
        return messages

    tail = messages[tail_start:]
    stale = messages[2:tail_start]

    if not stale:
        return messages

    log.info("Context compaction: %d stale messages → summarize, "
             "keeping %d recent tail messages", len(stale), len(tail))

    # ── 3. Summarize the stale middle ────────────────────────────────────
    summary_text = _summarize_stale(model, stale, context_length,
                                    timing=timing, profile=profile)

    if summary_text:
        compacted = head + [
            {"role": "user", "content": (
                "[Context Compaction — earlier conversation summarized]\n\n"
                + summary_text
            )},
        ] + tail
        new_estimate = _estimate_tokens(model, compacted)
        if new_estimate < budget:
            log.info("Context compaction: %d tokens after summarization "
                     "(budget %d)", new_estimate, budget)
            return compacted
        log.warning("Context compaction: summarized result still %d tokens "
                    "(budget %d), falling back to sliding window",
                    new_estimate, budget)

    # ── 4. Fallback: sliding window ──────────────────────────────────────
    return _sliding_window_compact(model, messages, budget)


def _summarize_stale(
    model: str,
    stale: list[dict],
    context_length: int,
    timing: StepTiming | None = None,
    profile=None,
) -> str | None:
    """Call the LLM to summarize stale conversation turns.

    If *timing* is provided, the token usage of the summarization call is
    added to it so cost tracking stays accurate.
    """
    conversation_text = []
    for msg in stale:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "assistant" and msg.get("tool_calls"):
            calls = ", ".join(
                tc["function"]["name"]
                for tc in msg.get("tool_calls", [])
            )
            conversation_text.append(f"[assistant called: {calls}]")
        elif role == "tool":
            preview = content[:500] if len(content) > 500 else content
            conversation_text.append(f"[tool result: {preview}]")
        else:
            preview = content[:1000] if len(content) > 1000 else content
            conversation_text.append(f"[{role}]: {preview}")

    joined = "\n".join(conversation_text)
    max_input_chars = min(len(joined), int(context_length * 0.4 * 4))
    if len(joined) > max_input_chars:
        joined = joined[:max_input_chars] + "\n[... truncated for summarization]"

    summarize_prompt = (
        "Summarize the following agent conversation for continuity. "
        "The agent is doing a code backport task. Preserve ALL of the following:\n\n"
        "CODE DETAILS (most critical — do not lose these):\n"
        "- Exact function signatures seen or written (name, parameters, return types)\n"
        "- Exact variable names, struct fields, interface methods\n"
        "- Exact import paths and package names\n"
        "- Any code blocks the agent wrote or decided on — quote them verbatim if short\n"
        "- File paths read, written, or modified and what was found in them\n"
        "- Exact conflict resolutions decided (what was kept vs discarded)\n\n"
        "TASK STATE:\n"
        "- Current state (what succeeded, what failed, what is in progress)\n"
        "- Exact error messages, compiler errors, and exit codes\n"
        "- Key decisions made and their rationale\n"
        "- Any commands run and their outcomes\n"
        "- What files still need to be modified\n\n"
        "FORMAT: Write structured bullet points per file/topic. "
        "Do NOT compress code details into vague prose — "
        "exact names and signatures matter for the agent to continue correctly.\n\n"
        "CONVERSATION:\n"
        + joined
    )
    try:
        import litellm
        from common.llm_client import get_api_base, get_api_key

        extra = {}
        if profile:
            if profile.api_base and model.startswith("openai/"):
                extra["api_base"] = profile.api_base
                extra["api_key"] = profile.api_key or "dummy"
        else:
            api_base = get_api_base()
            if api_base and model.startswith("openai/"):
                extra["api_base"] = api_base
                extra["api_key"] = get_api_key() or "dummy"

        response = _litellm_completion_with_streaming(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise summarizer."},
                {"role": "user", "content": summarize_prompt},
            ],
            max_tokens=32768,
            **extra,
        )
        if timing and response.usage:
            timing.input_tokens += getattr(response.usage, "prompt_tokens", 0)
            timing.output_tokens += getattr(response.usage, "completion_tokens", 0)
        text = response.choices[0].message.content or ""
        log.info("Context compaction summary: %d chars", len(text))
        return text.strip()
    except Exception as e:
        log.warning("Context compaction summarization failed: %s", e)
        return None


def _sliding_window_compact(
    model: str, messages: list[dict], budget: int,
) -> list[dict]:
    """Drop oldest tool_call/tool pairs until messages fit within budget.

    Always preserves the first 2 messages (system + user).
    """
    result = list(messages)
    idx = 2
    while idx < len(result) - 2:
        est = _estimate_tokens(model, result)
        if est <= budget:
            break
        msg = result[idx]
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            n_tools = len(msg["tool_calls"])
            del result[idx:idx + 1 + n_tools]
            log.debug("Sliding window: dropped assistant + %d tool results "
                      "at idx %d", n_tools, idx)
            continue
        if msg["role"] in ("assistant", "user") and not msg.get("tool_calls"):
            del result[idx]
            continue
        idx += 1

    est = _estimate_tokens(model, result)
    log.info("Sliding window compaction: %d messages, %d tokens (budget %d)",
             len(result), est, budget)
    return result


# ── Tool execution ───────────────────────────────────────────────────────────

def _normalize_args(name: str, arguments: dict) -> dict:
    """Fix common malformed argument patterns from weaker models.

    Some models emit ``{"argument_name": "command", "argument_value": "ls"}``
    instead of ``{"command": "ls"}``.  Detect and repair these.
    """
    if "argument_name" in arguments and "argument_value" in arguments:
        return {arguments["argument_name"]: arguments["argument_value"]}
    return arguments


def _execute_tool(name: str, arguments: dict, cwd: str,
                   bash_timeout: int = DEFAULT_BASH_TIMEOUT,
                   read_cap: int = 50_000,
                   bash_cap: int = 30_000) -> str:
    """Execute a single tool call and return the result as a string."""
    arguments = _normalize_args(name, arguments)
    try:
        if name == "read_file":
            return _tool_read_file(arguments["path"], cwd, max_chars=read_cap)
        elif name == "write_file":
            return _tool_write_file(arguments["path"], arguments["content"], cwd)
        elif name == "bash":
            return _tool_bash(arguments["command"], cwd, timeout=bash_timeout,
                              max_chars=bash_cap)
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


def _tool_read_file(path: str, cwd: str, max_chars: int = 50_000) -> str:
    resolved = Path(path) if Path(path).is_absolute() else Path(cwd) / path
    try:
        content = resolved.read_text(errors="replace")
        if len(content) > max_chars:
            return (
                content[:max_chars]
                + f"\n\n[... truncated at {max_chars} chars — "
                  f"file is {len(content)} chars total]"
            )
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


def _tool_bash(command: str, cwd: str, timeout: int = DEFAULT_BASH_TIMEOUT,
               max_chars: int = 30_000) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        if len(output) > max_chars:
            head_size = min(2000, max_chars // 4)
            tail_size = max_chars - head_size - 80
            output = (
                output[:head_size]
                + f"\n\n[... {len(output) - head_size - tail_size} chars omitted ...]\n\n"
                + output[-tail_size:]
            )
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
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
                allowed_tools=getattr(config, "allowed_tools", None),
                bash_timeout=getattr(config, "command_timeout", None),
                context_length=getattr(config, "context_length", None),
                profile=getattr(config, "profile", None),
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
        allowed_tools: list[str] | None = None,
        bash_timeout: int | None = None,
        context_length: int | None = None,
        profile=None,
    ) -> StepTiming:
        """Core tool-use loop: prompt -> tool calls -> execute -> repeat.

        When *profile* is provided, transport config (api_base, api_key,
        params) comes from it instead of the module-level globals.

        Includes:
        - Duplicate-call detection to break out of loops.
        - Adaptive context window management (Tiers 1–3) to stay within
          the model's context limit.
        """
        import litellm
        from common.llm_client import (
            _is_ollama_model, _strip_thinking,
            get_api_base, get_api_key, get_model_params,
        )

        # ── Resolve context budget ───────────────────────────────────────
        ctx_limit = _resolve_context_length(model, context_length)
        ctx_budget = int(ctx_limit * COMPACTION_THRESHOLD_PCT)
        read_cap, bash_cap = _tool_output_caps(ctx_limit)
        log.info("Context window: limit=%d, budget=%d (%.0f%%), "
                 "tool caps read=%d bash=%d",
                 ctx_limit, ctx_budget, COMPACTION_THRESHOLD_PCT * 100,
                 read_cap, bash_cap)

        if profile:
            extra_kwargs: dict = {**profile.params}
            p_api_base = profile.api_base
            p_api_key = profile.api_key
        else:
            extra_kwargs: dict = {**get_model_params()}
            p_api_base = get_api_base()
            p_api_key = get_api_key()

        extra_kwargs.setdefault("max_tokens", 16384)

        if _is_ollama_model(model):
            extra_kwargs.setdefault("extra_body", {})
            extra_kwargs["extra_body"].setdefault("think", False)

        if p_api_base and model.startswith("openai/"):
            extra_kwargs.setdefault("api_base", p_api_base)
            extra_kwargs.setdefault("api_key", p_api_key or "dummy")

        tools = _filter_tools(allowed_tools)
        allowed_names = {t["function"]["name"] for t in tools}
        effective_timeout = bash_timeout or DEFAULT_BASH_TIMEOUT

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        timing = StepTiming(name="")
        iteration = 0
        last_tool_sig: str | None = None
        repeat_count = 0
        compaction_count = 0

        while iteration < MAX_AGENT_ITERATIONS:
            iteration += 1
            log.info("Agent iteration %d/%d (model=%s)",
                     iteration, MAX_AGENT_ITERATIONS, model)

            # ── Tier 2: pre-call budget check ────────────────────────────
            est_tokens = _estimate_tokens(model, messages)
            if est_tokens > ctx_budget:
                compaction_count += 1
                log.warning(
                    "Context compaction #%d triggered: %d tokens > budget %d "
                    "(limit %d)", compaction_count, est_tokens, ctx_budget,
                    ctx_limit,
                )
                print(f"{Fore.MAGENTA}[Context compaction #{compaction_count}: "
                      f"{est_tokens:,} tokens → compacting to fit "
                      f"{ctx_budget:,} budget]{Style.RESET_ALL}")
                messages = _compact_messages(
                    model, messages, ctx_limit,
                    keep_recent_tokens=KEEP_RECENT_TOKENS,
                    timing=timing,
                    profile=profile,
                )
                est_tokens = _estimate_tokens(model, messages)
                log.info("Post-compaction: %d tokens, %d messages",
                         est_tokens, len(messages))

            try:
                response = _litellm_completion_with_streaming(
                    model=model,
                    messages=messages,
                    tools=tools,
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
                        final_kwargs = {**extra_kwargs, "max_tokens": 4096}
                        final = _litellm_completion_with_streaming(
                            model=model, messages=messages,
                            **final_kwargs,
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

                if fn_name not in allowed_names:
                    log.warning("Rejecting tool call %s — not in allowed_tools", fn_name)
                    result = f"Error: tool '{fn_name}' is not allowed by the pipeline configuration."
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    continue
                print(f"{Fore.YELLOW}  Tool: {fn_name}({_summarize_args(fn_args)}){Style.RESET_ALL}")
                result = _execute_tool(fn_name, fn_args, cwd,
                                       bash_timeout=effective_timeout,
                                       read_cap=read_cap, bash_cap=bash_cap)
                print(f"  Result: {result[:200]}{'...' if len(result) > 200 else ''}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            log.warning("Agent hit max iterations (%d)", MAX_AGENT_ITERATIONS)

        if compaction_count:
            log.info("Context compaction was triggered %d time(s) during "
                     "this agent loop", compaction_count)

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
    """Dispatch to the correct runner based on the model's provider.

    When the config carries a profile, uses ``profile.provider`` for dispatch.
    Otherwise falls back to the ``is_claude_model()`` heuristic.

    ``provider: anthropic`` → Claude Agent SDK (existing behaviour).
    ``provider: litellm``   → LiteLLM tool-use loop.
    """
    profile = getattr(config, "profile", None)
    if profile:
        use_claude = profile.provider == "anthropic"
    else:
        from common.llm_client import is_claude_model
        use_claude = is_claude_model(config.model)

    if use_claude:
        log.info("Using Claude Agent SDK for model %s", config.model)
        return await _claude_runner.run(config, steps, resume=resume)
    else:
        log.info("Using LiteLLM agent loop for model %s", config.model)
        return await _litellm_runner.run(config, steps, resume=resume)
