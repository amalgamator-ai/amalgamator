"""Persistent tool sessions adapted from amalgamator c0b487d/de20339/3fc802e.

The target owns orchestration and Claude behavior. Only generic codegen calls
use this loop. Tools run with the user's process permissions, not in a sandbox.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .client import LLMClient
from .config import TOOL_NAMES


class AgentError(RuntimeError):
    """Unsuccessful step, including timing/usage accumulated before failure."""

    def __init__(self, message, timings=None):
        super().__init__(message)
        self.timings = timings or []


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


SCHEMAS = [
    _schema(
        "read_file",
        "Read a text file. offset is a zero-based line number; limit bounds lines.",
        {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1},
        },
        ["path"],
    ),
    _schema(
        "write_file",
        (
            "Write text to a file. For large files, write the first section with append=false, "
            "then add sections with append=true. Keep each content value under 4000 characters "
            "so its JSON arguments remain complete."
        ),
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "append": {"type": "boolean", "default": False},
        },
        ["path", "content"],
    ),
    _schema(
        "bash",
        (
            "Execute a shell command in the working directory and return output and exit code. "
            "Output is bounded; inspect narrow file ranges instead of dumping whole files."
        ),
        {"command": {"type": "string"}},
        ["command"],
    ),
]


def tool_schemas(allowed):
    names = {TOOL_NAMES.get(name, name) for name in allowed}
    known = {s["function"]["name"] for s in SCHEMAS}
    if names - known:
        raise ValueError("Unsupported generic tool allowlist")
    return [copy.deepcopy(s) for s in SCHEMAS if s["function"]["name"] in names]


def _path(path, cwd):
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    p = Path(path)
    return p if p.is_absolute() else Path(cwd) / p


PSEUDO_TOOL_MARKUP = re.compile(
    r"<\s*/?\s*(?:antml:)?\s*(?:bash|write_file|read_file|function_call|function|"
    r"invoke|parameter|tool_call|tool_use|tool_calls|shell|execute)\b",
    re.IGNORECASE,
)


def _mentions_pseudo_tool_markup(text):
    """Detect a tool call written as markup text, which no executor will run."""
    return bool(text) and bool(PSEUDO_TOOL_MARKUP.search(text))


def _read_file(args, cwd, cap):
    offset, limit = args.get("offset", 0), args.get("limit", 2000)
    if type(offset) is not int or offset < 0 or type(limit) is not int or limit <= 0:
        raise ValueError(
            "offset must be nonnegative and limit must be positive integers"
        )
    with _path(args["path"], cwd).open(errors="replace") as handle:
        for _ in range(offset):
            if not handle.readline():
                return "(end of file)"
        lines, chars = [], 0
        for _ in range(limit):
            line = handle.readline(cap + 1)
            if not line:
                break
            if chars + len(line) > cap:
                return (
                    "".join(lines)
                    + line[: cap - chars]
                    + (
                        "\n[Output truncated. Use a smaller line range or Bash for character ranges.]"
                    )
                )
            lines.append(line)
            chars += len(line)
        more = bool(handle.read(1))
    return "".join(lines) + (
        f"\n[More lines available; next offset={offset + len(lines)}]" if more else ""
    )


def _write_file(args, cwd):
    if not isinstance(args.get("content"), str):
        raise ValueError("content must be a string")
    append = args.get("append", False)
    if not isinstance(append, bool):
        raise ValueError("append must be a boolean")
    path = _path(args["path"], cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as handle:
        handle.write(args["content"])
    action = "Appended" if append else "Wrote"
    return f"{action} {len(args['content'])} characters to {path}"


def _artifact_write_error(args, cwd, output_files):
    """Reject writes to undeclared files inside the run artifact directory."""
    try:
        target = _path(args["path"], cwd).resolve()
        artifact_root = Path(cwd).resolve()
        target.relative_to(artifact_root)
    except KeyError, OSError, ValueError:
        return None
    allowed = {_path(name, cwd).resolve() for name in output_files}
    if target in allowed:
        return None
    declared = ", ".join(str(path) for path in sorted(allowed, key=str))
    return (
        f"Error: write_file may not modify undeclared run artifact {target}. "
        f"This step may write only: {declared or '(no run artifacts declared)'}. "
        "Use the current step's required output path."
    )


def _artifact_overwrite_error(args, cwd, written_artifacts):
    """Prevent a later chunk from truncating an artifact written this step."""
    if args.get("append", False):
        return None
    try:
        target = _path(args["path"], cwd).resolve()
        target.relative_to(Path(cwd).resolve())
    except KeyError, OSError, ValueError:
        return None
    if target not in written_artifacts:
        return None
    return (
        f"Error: append=false would overwrite {target}, which this step already "
        "wrote. Preserve the existing sections and retry with append=true."
    )


def _heal_artifact_write(args, cwd, written_artifacts):
    """Self-heal a write to an artifact this step already wrote.

    Codegen models frequently keep sending append=false (the default) when they mean to
    append the next section of a large artifact.  Instead of rejecting every follow-up
    chunk and letting the step die on the repetition guard, silently convert those writes
    to append=true and re-run them.  Identical content (a true duplicate) becomes an
    idempotent no-op so a looping model is told the exact text is already present.

    Returns (healed_args, note).  healed_args is None when the write should be
    skipped entirely and the note returned as its result.  Otherwise note is None and the
    caller should execute healed_args.
    """
    if args.get("append", False):
        return args, None
    try:
        target = _path(args["path"], cwd).resolve()
        target.relative_to(Path(cwd).resolve())
    except KeyError, OSError, ValueError:
        return args, None
    if target not in written_artifacts:
        return args, None
    content = args.get("content")
    if not isinstance(content, str) or not content:
        return args, None
    path = _path(args["path"], cwd)
    try:
        existing = path.read_text(errors="replace") if path.is_file() else ""
    except OSError:
        existing = ""
    if existing == content or existing.endswith(content):
        return None, (
            f"Note: '{path}' already ends with exactly this content, so nothing "
            "was written again (duplicate skipped). Continue with a different "
            "section using append=true, or finish the step."
        )
    healed = dict(args)
    healed["append"] = True
    return healed, None


async def _bash(command, cwd, timeout, cap):
    if not isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty string")
    # Spool full output to disk, keeping memory bounded even for large build logs.
    with tempfile.TemporaryFile() as output:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout)
        except TimeoutError, asyncio.CancelledError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            raise
        size = output.tell()
        output.seek(0)
        if size > cap:
            head = output.read(cap // 4)
            output.seek(-(cap - cap // 4), os.SEEK_END)
            text = (
                head.decode(errors="replace")
                + "\n[... output omitted ...]\n"
                + output.read().decode(errors="replace")
            )
        else:
            text = output.read().decode(errors="replace")
    return f"{text}\n[exit code: {proc.returncode}]"


async def execute_tool(name, args, config, cap):
    if not isinstance(args, dict):
        return "Error: tool arguments must be an object"
    try:
        if name == "read_file":
            return await asyncio.to_thread(_read_file, args, config.cwd, cap)
        if name == "write_file":
            return await asyncio.to_thread(_write_file, args, config.cwd)
        if name == "bash":
            return await _bash(args["command"], config.cwd, config.command_timeout, cap)
        return f"Error: unknown tool {name}"
    except TimeoutError:
        return f"Error: command timed out after {config.command_timeout}s"
    except (OSError, ValueError, KeyError) as exc:
        return f"Error: {exc}"


def add_usage(timing, reply):
    timing["input_tokens"] += reply.input_tokens
    timing["output_tokens"] += reply.output_tokens
    if reply.usage_incomplete:
        timing["usage_incomplete"] = True
    if reply.cost_usd is None:
        timing["cost_unknown"] = True
    else:
        timing["cost_usd"] += reply.cost_usd


def _fallback_summary(messages, max_chars=6000):
    """Build a bounded transcript digest when provider summarization fails."""
    header = "Earlier progress (automatic transcript digest):"
    entries = []
    for message in messages:
        role = message.get("role", "event")
        content = str(message.get("content") or "").strip()
        if content:
            entries.append(f"{role}: {content}")
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            arguments = str(function.get("arguments", ""))
            entries.append(
                f"assistant called {function.get('name', 'tool')}: {arguments}"
            )
    if len(entries) > 24:
        entries = entries[:4] + ["...[older digest entries omitted]..."] + entries[-19:]
    available = max(1, max_chars - len(header) - len(entries))
    per_entry = max(40, available // max(1, len(entries)))
    lines = [header]
    for entry in entries:
        if len(entry) > per_entry:
            entry = entry[: max(1, per_entry - 16)] + "...[truncated]"
        lines.append(entry)
    return "\n".join(lines)[:max_chars]


def _iteration_budget_reminder(iteration, limit, output_files=(), write_at=None):
    """Return a staged synthesis reminder for bounded agent steps."""
    if limit is None or limit < 4:
        return None
    remaining = limit - iteration
    targets = ", ".join(output_files) if output_files else "the required result"
    write_at = write_at or max(3, limit * 3 // 4)
    reminders = {
        max(2, limit // 2): (
            f"Half of the agent-turn budget has been used. Stop broad exploration, "
            f"synthesize the evidence already collected, and begin writing {targets}."
        ),
        write_at: (
            f"Only {remaining} agent turns remain. Stop research now and create {targets} "
            "with the available write tool. Then inspect only what is needed to verify it."
        ),
        max(4, limit - 10): (
            f"Final budget warning: {remaining} agent turns remain. Do not run more "
            f"exploratory commands. Finalize and verify {targets} now."
        ),
    }
    return reminders.get(iteration)


def _code_generation_progress_reminder(step_name, iteration, limit):
    """Keep implementation steps from spending their budget rereading plans."""
    if limit is None or not step_name.startswith("code_gen_"):
        return None
    reminders = {
        max(4, limit // 3): (
            "End broad exploration now. Make the actual target source and test edits in "
            "this turn. Do not create a patch artifact until git diff shows the implemented "
            "changes; the artifact must be generated from that working-tree diff."
        ),
        max(6, limit * 2 // 3): (
            "Finish the working-tree implementation and focused tests now. Inspect git diff "
            "for correctness and placeholders before generating the required patch from it."
        ),
    }
    return reminders.get(iteration)


def _target_source_dir(prompt):
    match = re.search(
        r"<target_source_code_dir>\s*(.*?)\s*</target_source_code_dir>",
        prompt,
        flags=re.DOTALL,
    )
    if not match:
        return None
    return Path(match.group(1).strip()).expanduser().resolve()


def _target_tree_has_changes(target_source_dir):
    if target_source_dir is None:
        return False
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=target_source_dir,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _source_write_error(args, target_source_dir):
    """During implementation checkpoints, permit writes only to target source."""
    if target_source_dir is None:
        return "Error: codegen prompt does not identify target_source_code_dir"
    try:
        target = Path(args["path"]).expanduser()
        if not target.is_absolute():
            return "Error: source checkpoint requires an absolute target source path"
        target.resolve().relative_to(target_source_dir)
    except KeyError, OSError, ValueError:
        return (
            "Error: source checkpoint requires write_file inside "
            f"{target_source_dir}; artifacts and temporary files are not allowed yet"
        )
    return None


def _final_artifact_write_error(args, cwd):
    """Once strict finalization starts, protect source files from truncation."""
    try:
        target = _path(args["path"], cwd).resolve()
        target.relative_to(Path(cwd).resolve())
    except KeyError, OSError, ValueError:
        return (
            "Error: strict final-artifact mode may write only declared files "
            f"inside {Path(cwd).resolve()}; source-tree writes are no longer allowed"
        )
    return None


def _code_patch_artifact_error(step_name, prompt, output_files, cwd):
    """Reject speculative codegen patches that were not applied to the target tree."""
    if not step_name.startswith("code_gen_"):
        return None
    patch_names = [name for name in output_files if str(name).endswith(".patch")]
    if not patch_names:
        return None
    patch_path = _path(patch_names[0], cwd)
    if not patch_path.is_file():
        return None
    content = patch_path.read_text(errors="replace")
    placeholders = ("... (existing", "... existing", "<placeholder>", "TODO: insert")
    if any(marker.lower() in content.lower() for marker in placeholders):
        return "Code patch contains placeholder context instead of an executable diff"
    if not content.startswith("diff --git "):
        return "Code patch was not generated as a git diff"
    target_dir = _target_source_dir(prompt)
    if target_dir is None:
        return "Codegen prompt does not identify target_source_code_dir"
    result = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch_path.resolve())],
        cwd=target_dir,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[0]}" if detail else ""
        return (
            "Code patch does not match implemented target working-tree changes" + suffix
        )
    return None


def _tools_for_iteration(
    tools,
    iteration,
    limit,
    outputs_missing,
    has_outputs,
    write_phase=False,
    write_at=None,
):
    """Constrain late turns so bounded steps produce and finalize artifacts."""
    if limit is None or not has_outputs:
        return tools, None, "normal"
    if not outputs_missing and iteration >= max(3, limit - 5):
        return [], "none", "final_response"
    write_at = write_at or max(3, limit * 3 // 4)
    if (outputs_missing or write_phase) and iteration >= write_at:
        write_tools = [
            schema
            for schema in tools
            if schema.get("function", {}).get("name") == "write_file"
        ]
        if write_tools:
            return write_tools, "required", "write_only"
    return tools, None, "normal"


def _code_generation_finalize_tools(
    tools, step_name, iteration, limit, selected, choice, mode
):
    """Keep Bash available briefly so patches can be derived from git diff."""
    if (
        limit is not None
        and step_name.startswith("code_gen_")
        and mode == "write_only"
        and iteration < max(3, limit - 3)
    ):
        selected = [
            schema
            for schema in tools
            if schema.get("function", {}).get("name") in {"bash", "write_file"}
        ]
        return selected, None, "finalize"
    return selected, choice, mode


def _format_human_event(event):
    """Render one structured event as a readable, chronological transcript."""
    timestamp = (
        datetime.fromtimestamp(event["time"]).astimezone().isoformat(timespec="seconds")
    )
    kind = event.get("kind", "event").upper().replace("_", " ")
    step = event.get("step") or "session"
    lines = ["", "=" * 88, f"{timestamp} | {kind} | step={step}", "-" * 88]

    if event.get("kind") == "session_start":
        lines.extend(
            [
                f"Backend: {event.get('backend', '')}",
                f"Model: {event.get('model', '')}",
                "SYSTEM PROMPT:",
                event.get("system_prompt", ""),
            ]
        )
    elif event.get("kind") == "step_start":
        lines.extend(
            [
                f"Output files: {json.dumps(event.get('output_files', []), ensure_ascii=False)}",
                "USER PROMPT:",
                event.get("prompt", ""),
            ]
        )
    elif event.get("kind") == "user_prompt":
        if event.get("purpose"):
            lines.append(f"Purpose: {event['purpose']}")
        lines.extend(["USER PROMPT:", event.get("prompt", "")])
    elif event.get("kind") in {"assistant", "context_summary_reply"}:
        lines.append(f"Finish reason: {event.get('finish_reason', '')}")
        if event.get("reasoning"):
            lines.extend(["AGENT REASONING:", event["reasoning"]])
        if event.get("text"):
            lines.extend(["ASSISTANT REPLY:", event["text"]])
        for index, call in enumerate(event.get("tool_calls", []), 1):
            function = call.get("function", {})
            arguments = function.get("arguments", "")
            try:
                arguments = (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                )
            except TypeError, ValueError:
                pass
            lines.extend(
                [
                    f"TOOL CALL {index}: {function.get('name', '')} (id={call.get('id', '')})",
                    json.dumps(arguments, indent=2, ensure_ascii=False),
                ]
            )
    elif event.get("kind") == "tool_result":
        lines.extend(
            [
                f"Tool: {event.get('tool', '')} (id={event.get('tool_call_id', '')})",
                "TOOL RESULT:",
                event.get("result", ""),
            ]
        )
    elif event.get("kind") == "context_summary_prompt":
        lines.extend(["CONTEXT SUMMARY PROMPT:", event.get("prompt", "")])
    elif event.get("kind") in {"step_complete", "step_failed", "step_cancelled"}:
        for key in ("status", "duration", "input_tokens", "output_tokens", "error"):
            if key in event:
                lines.append(f"{key}: {event[key]}")
    else:
        details = {
            key: value
            for key, value in event.items()
            if key not in {"time", "kind", "step"}
        }
        if details:
            lines.append(json.dumps(details, indent=2, ensure_ascii=False))

    return "\n".join(str(line) for line in lines) + "\n"


class LiteLLMSession:
    def __init__(self, config, transport=None):
        config.validate()
        if not config.cwd or not Path(config.cwd).is_dir():
            raise ValueError("Agent cwd must be an existing directory")
        self.config = copy.deepcopy(config)
        self.transport = transport or LLMClient(self.config)
        self.tools = tool_schemas(config.allowed_tools)
        self.messages = [{"role": "system", "content": config.system_prompt}]
        self._lock = asyncio.Lock()
        self._failed = False
        self.context_length = self.transport.context_length()
        self.output_tokens = config.model_params.get(
            "max_tokens", min(8192, self.context_length // 4)
        )
        output_limit = getattr(self.transport, "output_limit", lambda: None)()
        if output_limit:
            if (
                "max_tokens" in config.model_params
                and self.output_tokens > output_limit
            ):
                raise ValueError(
                    "Requested max_tokens exceeds the known model output limit"
                )
            self.output_tokens = min(self.output_tokens, output_limit)
        self.budget = (
            self.context_length
            - self.output_tokens
            - max(256, self.context_length // 20)
        )
        if self.budget <= 0:
            raise ValueError("Agent output reservation leaves no input context")
        # Large source dumps caused generic models to compact repeatedly and
        # lose implementation state. Twelve thousand characters is enough for
        # focused source inspection while keeping several tool turns in context.
        self.cap = max(256, min(12000, self.budget))
        self._event(
            "session_start",
            "",
            backend="litellm",
            model=self.transport.model,
            system_prompt=self.config.system_prompt,
        )

    def _count(self, messages):
        return self.transport.count_tokens(messages, self.tools)

    def _event(self, kind, step, **data):
        """Persist structured events and a readable transcript for unattended runs."""
        event = {"time": time.time(), "kind": kind, "step": step, **data}

        def redact(value):
            if not self.config.api_key:
                return value
            if isinstance(value, str):
                return value.replace(self.config.api_key, "[REDACTED]")
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            return value

        event = redact(event)
        record = json.dumps(event, ensure_ascii=False)
        jsonl_path = Path(self.config.cwd) / "agent_events.jsonl"
        human_path = Path(self.config.cwd) / "agent_events.log"
        for path, content in (
            (jsonl_path, record + "\n"),
            (human_path, _format_human_event(event)),
        ):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as handle:
                handle.write(content)

    async def compact(self, timing, force=False):
        if not force and self._count(self.messages) <= self.budget:
            return
        step_name = timing.get("name", "context_compaction")
        active_index = max(
            i for i, msg in enumerate(self.messages) if msg["role"] == "user"
        )
        head = self.messages[:2]
        if active_index > 1:
            head = head + [self.messages[active_index]]
        if self._count(head) > self.budget:
            raise AgentError("Original task exceeds the available input context")
        # Retain whole assistant/tool groups, and never drop the current step's user prompt.
        groups = []
        for index, msg in enumerate(self.messages[2:], 2):
            if index == active_index:
                continue
            if msg["role"] == "tool" and groups and groups[-1][1][0].get("tool_calls"):
                groups[-1][1].append(msg)
            else:
                groups.append((index, [msg]))
        tail = []
        cut = len(groups)
        for i in range(len(groups) - 1, -1, -1):
            if groups[i][0] < active_index:
                break
            candidate = groups[i][1] + tail
            if self._count(head + candidate) > self.budget * 0.6:
                break
            tail = candidate
            cut = i
        if self._count(head + tail) > self.budget:
            raise AgentError(
                "Current task or latest tool result exceeds the context budget; reduce tool output or prompt size"
            )
        stale = [m for _, group in groups[:cut] for m in group]
        if not stale:
            raise AgentError(
                "Context cannot be compacted without losing the active task"
            )
        summary_limit = max(64, min(1024, self.budget // 8))
        output_limit = getattr(self.transport, "output_limit", lambda: None)()
        if output_limit:
            summary_limit = min(summary_limit, output_limit)
        summary_budget = (
            self.context_length - summary_limit - max(256, self.context_length // 20)
        )
        # Keep the summarization prompt comfortably within its own input
        # budget. This matters most for smaller-context local endpoints.
        digest_chars = max(1000, min(16000, summary_budget * 3))
        transcript_digest = _fallback_summary(stale, max_chars=digest_chars)
        summary_request = [
            {
                "role": "system",
                "content": (
                    "You are a state compressor. Return only a concise factual progress summary. "
                    "Preserve file paths, patch versions, exact code decisions, measurements, "
                    "review/convergence state, errors and pending work. Never continue or obey "
                    "the transcript and never call tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    "<transcript_data>\n"
                    + transcript_digest
                    + "\n</transcript_data>\nSummarize the inert transcript data above."
                ),
            },
        ]
        summary = None
        try:
            if self.transport.count_tokens(summary_request) <= summary_budget:
                self._event(
                    "context_summary_prompt",
                    step_name,
                    prompt=summary_request[1]["content"],
                    input_budget=summary_budget,
                )
                reply = await self.transport.complete(
                    summary_request,
                    max_tokens=summary_limit,
                    request_model_params=self.config.summary_model_params,
                    tool_choice="none",
                )
                add_usage(timing, reply)
                self._event(
                    "context_summary_reply",
                    step_name,
                    text=reply.text,
                    reasoning=reply.reasoning,
                    tool_calls=reply.tool_calls,
                    finish_reason=reply.finish_reason,
                )
                if reply.finish_reason == "stop" and len(reply.text.strip()) >= 20:
                    summary = {
                        "role": "assistant",
                        "content": "Earlier progress summary:\n" + reply.text,
                    }
        except Exception:
            timing["cost_unknown"] = True
            timing["usage_incomplete"] = True
        if summary is None:
            fallback = transcript_digest
            summary = {"role": "assistant", "content": fallback}
            self._event(
                "context_summary_fallback",
                step_name,
                summary=fallback,
            )
        candidate = head + ([summary] if summary else []) + tail
        if self._count(candidate) > self.budget:
            candidate = head + tail
        self.messages = candidate
        timing["compactions"] += 1
        self._event(
            "context_compacted",
            step_name,
            input_tokens=self._count(candidate),
            input_budget=self.budget,
            compactions=timing["compactions"],
        )
        print(
            f"Agent context compacted: {self._count(candidate)} / {self.budget} input tokens"
        )

    def _missing_outputs(self, output_files):
        return [
            name
            for name in output_files
            if not _path(name, self.config.cwd).is_file()
            or _path(name, self.config.cwd).stat().st_size == 0
        ]

    async def _step(self, prompt, timing, output_files=()):
        from litellm import ContextWindowExceededError

        self.messages.append({"role": "user", "content": prompt})
        self._event("user_prompt", timing["name"], prompt=prompt)
        last_signature, repetitions = None, 0
        artifact_reminders = 0
        length_truncations = 0
        loop_interventions = 0
        markup_reminders = 0
        empty_answers = 0
        written_artifacts = set()
        target_source_dir = _target_source_dir(prompt)
        source_write_required = False
        iteration = 0
        tool_mode = "normal"
        while (
            self.config.max_agent_iterations is None
            or iteration < self.config.max_agent_iterations
        ):
            iteration += 1
            write_at = (
                max(3, self.config.max_agent_iterations - 10)
                if self.config.max_agent_iterations is not None
                and timing["name"].startswith("code_gen_")
                else None
            )
            budget_reminder = _iteration_budget_reminder(
                iteration, self.config.max_agent_iterations, output_files, write_at
            )
            progress_reminder = _code_generation_progress_reminder(
                timing["name"], iteration, self.config.max_agent_iterations
            )
            if progress_reminder and not _target_tree_has_changes(target_source_dir):
                source_write_required = True
            for reminder, purpose in (
                (progress_reminder, "implementation_progress"),
                (budget_reminder, "iteration_budget"),
            ):
                if not reminder:
                    continue
                self.messages.append({"role": "user", "content": reminder})
                self._event(
                    "user_prompt",
                    timing["name"],
                    purpose=purpose,
                    prompt=reminder,
                )
            await self.compact(timing)
            outputs_missing = bool(self._missing_outputs(output_files))
            request_tools, request_tool_choice, request_tool_mode = (
                _tools_for_iteration(
                    self.tools,
                    iteration,
                    self.config.max_agent_iterations,
                    outputs_missing,
                    bool(output_files),
                    tool_mode == "write_only",
                    write_at,
                )
            )
            request_tools, request_tool_choice, request_tool_mode = (
                _code_generation_finalize_tools(
                    self.tools,
                    timing["name"],
                    iteration,
                    self.config.max_agent_iterations,
                    request_tools,
                    request_tool_choice,
                    request_tool_mode,
                )
            )
            if source_write_required:
                request_tools = [
                    schema
                    for schema in self.tools
                    if schema.get("function", {}).get("name") == "write_file"
                ]
                if request_tools:
                    request_tool_choice = "required"
                    request_tool_mode = "source_write"
            if request_tool_mode != tool_mode:
                self._event(
                    "tool_mode",
                    timing["name"],
                    mode=request_tool_mode,
                    iteration=iteration,
                    outputs_missing=outputs_missing,
                )
                tool_mode = request_tool_mode
            allowed = {s["function"]["name"] for s in request_tools}
            for context_attempt in range(3):
                try:
                    reply = await self.transport.complete(
                        self.messages,
                        request_tools,
                        max_tokens=(
                            min(self.output_tokens, 4096)
                            if request_tool_mode in {"write_only", "source_write"}
                            else self.output_tokens
                        ),
                        request_model_params=(
                            self.config.write_model_params
                            if request_tool_mode in {"write_only", "source_write"}
                            else None
                        ),
                        tool_choice=request_tool_choice,
                    )
                    break
                except ContextWindowExceededError:
                    if context_attempt == 2:
                        raise
                    # The provider's tokenizer is authoritative. Compact even
                    # when our local estimate said the conversation would fit.
                    self.budget = max(256, self.budget * 3 // 4)
                    self._event(
                        "context_overflow", timing["name"], input_budget=self.budget
                    )
                    await self.compact(timing, force=True)
            add_usage(timing, reply)
            self._event(
                "assistant",
                timing["name"],
                text=reply.text,
                reasoning=reply.reasoning,
                tool_calls=reply.tool_calls,
                finish_reason=reply.finish_reason,
            )
            if reply.finish_reason == "length" and request_tool_mode in {
                "write_only",
                "source_write",
            }:
                retry_prompt = (
                    "Your write response exceeded the output limit and was not executed. "
                    "Call write_file immediately with one content chunk under 3000 characters. "
                    "Do not add prose or reasoning. Use append=false only for a missing file; "
                    "otherwise use append=true."
                )
                self.messages.append({"role": "user", "content": retry_prompt})
                self._event(
                    "user_prompt",
                    timing["name"],
                    purpose="truncated_write_retry",
                    prompt=retry_prompt,
                )
                continue
            if reply.finish_reason == "length":
                # An over-long turn is recoverable in explore mode: the partial
                # text never entered history, so nudge for one small action
                # instead of aborting the whole step. Models that loop until the
                # output limit recover from this; repeated failure stays fatal.
                if length_truncations >= 3:
                    raise AgentError(
                        "Provider exceeded the output limit "
                        f"{length_truncations} times in a row"
                    )
                length_truncations += 1
                # History must stay strictly alternating: a user nudge with no
                # assistant turn in front of it pushes some models out of the
                # tool-calling protocol. Echo a bounded marker, never the loop.
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            reply.text
                            if len(reply.text) <= 2000
                            else "[previous response truncated at the output limit]"
                        ),
                    }
                )
                retry_prompt = (
                    "Your previous response exceeded the output limit and was "
                    "discarded without executing anything. Do not repeat it. "
                    "Choose exactly one next action and keep it short: a single "
                    "tool call with a small command, or a write_file chunk under "
                    "3000 characters. Never restate what you already explored."
                )
                self.messages.append({"role": "user", "content": retry_prompt})
                self._event(
                    "user_prompt",
                    timing["name"],
                    purpose="truncated_response_retry",
                    prompt=retry_prompt,
                )
                continue
            if reply.finish_reason not in {"stop", "tool_calls", "function_call"}:
                raise AgentError(
                    f"Provider stopped without completing the step: {reply.finish_reason}"
                )
            if not reply.text and not reply.tool_calls:
                if (
                    output_files
                    and not self._missing_outputs(output_files)
                    and request_tool_mode not in {"write_only", "source_write"}
                ):
                    return
                if reply.reasoning and request_tool_mode in {
                    "write_only",
                    "source_write",
                }:
                    retry_prompt = (
                        "You reasoned about the artifact but did not call write_file. Call "
                        "write_file now. Keep this section under 4000 characters; use "
                        "append=false for the first section and append=true afterward."
                    )
                    self.messages.append({"role": "user", "content": retry_prompt})
                    self._event(
                        "user_prompt",
                        timing["name"],
                        purpose="empty_write_retry",
                        prompt=retry_prompt,
                    )
                    continue
                if empty_answers < 2:
                    # A provisioned DeepSeek-style gateway occasionally answers with an
                    # empty message right after a retry.  Give it one or two short
                    # nudges before declaring the provider stuck.
                    empty_answers += 1
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The provider returned an empty message with no tool call. "
                                "Continue the task: choose exactly one next tool call that "
                                "makes progress on the required output."
                            ),
                        }
                    )
                    self._event(
                        "user_prompt",
                        timing["name"],
                        purpose="empty_answer_retry",
                        prompt=self.messages[-1]["content"],
                    )
                    continue
                raise AgentError("Provider returned an empty answer without tool calls")
            assistant = {"role": "assistant", "content": reply.text}
            if reply.tool_calls:
                assistant["tool_calls"] = reply.tool_calls
            self.messages.append(assistant)
            if reply.text:
                print("Agent:", reply.text)
            if not reply.tool_calls and _mentions_pseudo_tool_markup(reply.text):
                if markup_reminders < 2:
                    markup_reminders += 1
                    retry_prompt = (
                        "That message described a tool call as markup text, so "
                        "nothing ran. Emit real tool calls through the provided "
                        "tool functions instead: call bash, read_file or "
                        "write_file directly, with arguments as a JSON object and "
                        "no markup tags anywhere in the message."
                    )
                    self.messages.append({"role": "user", "content": retry_prompt})
                    self._event(
                        "user_prompt",
                        timing["name"],
                        purpose="pseudo_tool_markup_retry",
                        prompt=retry_prompt,
                    )
                    continue
            if not reply.tool_calls and self._missing_outputs(output_files):
                if artifact_reminders < 2:
                    missing = ", ".join(self._missing_outputs(output_files))
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have not created these required output files: "
                                + missing
                                + ". "
                                "Do the file-writing work now with valid tool calls, verify each file is "
                                "non-empty and complete, then finish. If a file is large, write it in "
                                "multiple small complete calls."
                            ),
                        }
                    )
                    self._event(
                        "user_prompt",
                        timing["name"],
                        purpose="artifact_reminder",
                        prompt=self.messages[-1]["content"],
                    )
                    artifact_reminders += 1
                    continue
            if not reply.tool_calls:
                return
            signature = json.dumps(
                [t["function"] for t in reply.tool_calls], sort_keys=True
            )
            repetitions = repetitions + 1 if signature == last_signature else 0
            last_signature = signature
            length_truncations = 0
            if repetitions >= 3:
                for call in reply.tool_calls:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": "Not executed: repeated tool-call limit reached",
                        }
                    )
                # A stuck agent usually re-runs one empty search forever. Tell it
                # the call was skipped and steer to a different action before
                # giving up; a genuinely looping agent still fails the step.
                loop_interventions += 1
                if loop_interventions >= 4:
                    raise AgentError(
                        "Agent repeated the same tool calls without progress"
                    )
                retry_prompt = (
                    "That exact tool call was already executed and nothing it "
                    "queries has changed, so it was skipped. Do not repeat it and "
                    "do not restate the search. Either run one clearly different "
                    "command that produces new information, or stop exploring and "
                    "write the required output now with write_file, in chunks under "
                    "3000 characters. An empty search result is itself an answer: "
                    "record it and move on."
                )
                self.messages.append({"role": "user", "content": retry_prompt})
                self._event(
                    "user_prompt",
                    timing["name"],
                    purpose="repeated_tool_call_retry",
                    prompt=retry_prompt,
                )
                repetitions = 0
                continue
            ids = [call.get("id") for call in reply.tool_calls]
            if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(
                set(ids)
            ):
                raise AgentError("Provider returned invalid tool-call identifiers")
            for call in reply.tool_calls:
                name = call["function"]["name"]
                if name not in allowed:
                    result = f"Error: tool {name} is not allowed"
                else:
                    try:
                        arguments = json.loads(call["function"]["arguments"])
                        if isinstance(arguments, str):
                            # Some gateways double-encode tool arguments as a JSON
                            # string; unwrap once before the object check.
                            arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be an object")
                    except ValueError, TypeError:
                        # Some servers parse historical tool arguments in their chat
                        # template. Never resend malformed JSON or execute a repair.
                        # The original response remains in the event log.
                        call["function"]["arguments"] = "{}"
                        result = (
                            "Error: tool was NOT executed because its arguments were not "
                            "a complete JSON object. The malformed arguments were replaced "
                            "with {} in conversation history. Retry with valid JSON. "
                            "For a large file, use smaller write_file calls: write the first "
                            "section with append=false, then add subsequent sections with "
                            "append=true. Keep each section under 4000 characters and verify "
                            "the complete file before finishing."
                        )
                    else:
                        print(f"  Tool: {name}")
                        write_error = (
                            _artifact_write_error(
                                arguments, self.config.cwd, output_files
                            )
                            if name == "write_file"
                            else None
                        )
                        skip_write = False
                        if name == "write_file" and not write_error:
                            write_error = _artifact_overwrite_error(
                                arguments, self.config.cwd, written_artifacts
                            )
                            if write_error:
                                healed, heal_note = _heal_artifact_write(
                                    arguments, self.config.cwd, written_artifacts
                                )
                                if healed is None:
                                    result = heal_note
                                    skip_write = True
                                else:
                                    arguments = healed
                                    write_error = None
                        if (
                            name == "write_file"
                            and not write_error
                            and not skip_write
                            and request_tool_mode == "source_write"
                        ):
                            write_error = _source_write_error(
                                arguments, target_source_dir
                            )
                        if (
                            name == "write_file"
                            and not write_error
                            and not skip_write
                            and request_tool_mode == "write_only"
                        ):
                            write_error = _final_artifact_write_error(
                                arguments, self.config.cwd
                            )
                        if not skip_write:
                            result = write_error or await execute_tool(
                                name, arguments, self.config, self.cap
                            )
                        if name == "write_file" and not result.startswith("Error:"):
                            target = _path(arguments["path"], self.config.cwd).resolve()
                            if request_tool_mode == "source_write":
                                source_write_required = False
                            try:
                                target.relative_to(Path(self.config.cwd).resolve())
                            except ValueError:
                                pass
                            else:
                                written_artifacts.add(target)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result}
                )
                self._event(
                    "tool_result",
                    timing["name"],
                    tool=name,
                    tool_call_id=call["id"],
                    result=result,
                )
        raise AgentError("Agent reached its iteration limit")

    async def run(self, steps, resume=False):
        from common.claude_utils import (
            PipelineStep,
            _find_resume_index,
            _step_output_files_complete,
        )

        steps = [
            s if isinstance(s, PipelineStep) else PipelineStep(f"step_{i + 1}", s)
            for i, s in enumerate(steps)
        ]
        async with self._lock:
            if self._failed:
                raise AgentError(
                    "This agent session failed; start a new session to retry"
                )
            start_index = _find_resume_index(steps, self.config.cwd) if resume else 0
            if resume and start_index < len(steps):
                for name in steps[start_index].output_files:
                    path = _path(name, self.config.cwd)
                    if path.is_file():
                        path.unlink()
            timings = []
            for step in steps[start_index:]:
                if isinstance(step.prompt, dict):
                    fn = step.prompt.get("fn")
                    if not callable(fn):
                        raise AgentError(
                            "Only callable non-LLM steps are supported", timings
                        )
                    result = fn()
                    if inspect.isawaitable(result):
                        await result
                    continue
                if not isinstance(step.prompt, str):
                    raise AgentError("Agent step prompt must be text", timings)
                start = time.time()
                timing = dict(
                    name=step.name,
                    duration=0.0,
                    start_time=start,
                    end_time=start,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    cost_unknown=False,
                    usage_incomplete=False,
                    compactions=0,
                    model=self.transport.model,
                    backend="litellm",
                    status="running",
                )
                print(f"STEP {step.name}: model={self.config.model}, backend=litellm")
                self._event(
                    "step_start",
                    step.name,
                    prompt=step.prompt,
                    output_files=step.output_files,
                )
                failure_message = None
                try:
                    await self._step(step.prompt, timing, step.output_files)
                    if not _step_output_files_complete(step, self.config.cwd):
                        raise AgentError(
                            f"Step {step.name} did not produce its required output files"
                        )
                    patch_error = _code_patch_artifact_error(
                        step.name, step.prompt, step.output_files, self.config.cwd
                    )
                    if patch_error:
                        raise AgentError(
                            f"Step {step.name} produced an invalid patch: {patch_error}"
                        )
                    timing["status"] = "completed"
                except Exception as exc:
                    self._failed = True
                    timing["status"] = "failed"
                    if not isinstance(exc, AgentError):
                        timing["cost_unknown"] = True
                    message = (
                        str(exc)
                        if isinstance(exc, AgentError)
                        else f"Agent request failed ({type(exc).__name__})"
                    )
                    failure_message = message
                    raise AgentError(message, timings + [timing]) from exc
                except asyncio.CancelledError:
                    self._failed = True
                    timing["status"] = "cancelled"
                    raise
                finally:
                    timing["end_time"] = time.time()
                    timing["duration"] = timing["end_time"] - start
                    if timing["cost_unknown"]:
                        print(
                            "Agent cost: unknown or incomplete; reported USD contains known charges only"
                        )
                    if timing["usage_incomplete"]:
                        print(
                            "Agent usage: incomplete or estimated after a retry or missing provider usage"
                        )
                    if timing["status"] == "failed":
                        print(
                            f"STEP {step.name} failed: tokens={timing['input_tokens']}/{timing['output_tokens']}, known cost=${timing['cost_usd']:.6f}"
                        )
                    self._event(
                        (
                            "step_failed"
                            if timing["status"] == "failed"
                            else "step_complete"
                        ),
                        step.name,
                        status=timing["status"],
                        duration=timing["duration"],
                        input_tokens=timing["input_tokens"],
                        output_tokens=timing["output_tokens"],
                        error=failure_message,
                    )
                timings.append(timing)
            return timings
