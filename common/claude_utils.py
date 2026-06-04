import os
import time
from dataclasses import dataclass, field, fields
from colorama import Fore, Style

from claude_agent_sdk import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ContentBlock,
    ResultMessage,
    SystemMessage,
    UserMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
)

from claude_agent_sdk.types import StreamEvent

import common.utils  # noqa: F401

@dataclass
class ClaudeConfig:
    model: str
    allowed_tools: list[str]
    perm_mode: str
    cwd: str | None
    thinking: dict | None = None
    effort: str = "max"
    max_thinking_tokens: int = 1048576


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
    last_completed = -1
    for i, step in enumerate(steps):
        if not step.output_files:
            continue
        all_exist = all(
            os.path.isfile(os.path.join(output_dir, f))
            and os.path.getsize(os.path.join(output_dir, f)) > 0
            for f in step.output_files
        )
        if all_exist:
            last_completed = i

    if last_completed < 0:
        return 0
    # Re-run from last_completed since its files may be corrupted
    return last_completed


def print_claude_config(config: ClaudeConfig) -> None:
    class_name = config.__class__.__name__
    print(f"{class_name}:")

    for field in fields(config):
        name = field.name
        value = getattr(config, name)
        print(f"  {name}: {value}")


def profile_system_prompt():
    return "You are a benchmarking and profiling expert of LLMs that run via vLLM, SGLang and TRT"


def print_dict(d, indent):
    for key, val in d.items():
        if isinstance(val, dict):
            print("{}{} : ".format(indent, key))
            print_dict(val, indent + "    ")
        else:
            print("{}{} : {}".format(indent, key, val))


def print_ContentBlock(prefix, content):
    if isinstance(content, TextBlock):
        print("{} {}".format(prefix, content.text))

    elif isinstance(content, ThinkingBlock):
        print("{} Thinking: {} {}".format(prefix, content.signature, content.thinking))

    elif isinstance(content, ToolUseBlock):
        print(
            "{} ToolUse: name = {}".format(
                prefix,
                content.name,
            )
        )

        print_dict(content.input, indent="    ")

    elif isinstance(content, ToolResultBlock):
        print(
            "{} ToolResult: err = {}".format(
                prefix,
                content.is_error,
            )
        )

        print()
        print(content.content)

    else:
        raise Exception("Unexpected type(content) = {}".format(type(content)))
    print()


def print_UserMessage(msg: UserMessage):
    prefix = f"{Fore.GREEN}User:{Style.RESET_ALL}"
    if isinstance(msg.content, str):
        print("{} {}".format(prefix, msg.content))
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
        print("    -- ERROR = {}".format(msg.error))


def print_SystemMessage(msg: SystemMessage):
    print("System: subtype = {} data = {}".format(msg.subtype, msg.data))


def print_ResultMessage(msg: ResultMessage):
    print(
        "Result: subtype = {} dur = {} dur_api = {} err = {} num_turns = {}".format(
            msg.subtype,
            msg.duration_ms,
            msg.duration_api_ms,
            msg.is_error,
            msg.num_turns,
        )
    )

    if msg.usage is not None:
        print("    -- usage = {}".format(msg.usage))
    if msg.total_cost_usd is not None:
        print("    -- cost = ${:.4f}".format(msg.total_cost_usd))
    if msg.result is not None:
        print("    -- result = {}".format(msg.result))


def print_StreamEvent(msg: StreamEvent):
    print("Stream: event = {}".format(msg.event))


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
        raise Exception("Unexpected type(msg) = {}".format(type(msg)))


async def claude_run(
    claude_config: ClaudeConfig,
    steps: list,
    resume: bool = False,
):
    assert claude_config.cwd is not None, "claude_config must have CWD set"

    # Support legacy callers that pass raw strings/dicts
    wrapped_steps = []
    for i, s in enumerate(steps):
        if isinstance(s, PipelineStep):
            wrapped_steps.append(s)
        else:
            wrapped_steps.append(PipelineStep(
                name="step_{}".format(i + 1),
                prompt=s,
            ))
    steps = wrapped_steps

    print_claude_config(claude_config)

    start_index = 0
    if resume:
        start_index = _find_resume_index(steps, claude_config.cwd)
        print("\n{}=== RESUME SUMMARY ==={}\n".format(Fore.YELLOW, Style.RESET_ALL))
        print("Total pipeline steps: {}".format(len(steps)))

        if start_index > 0:
            # Log completed steps
            print("\n{}Completed steps (will be skipped):{}\n".format(Fore.GREEN, Style.RESET_ALL))
            for i in range(start_index):
                s = steps[i]
                if s.output_files:
                    files_str = ", ".join(s.output_files)
                    print("  [{}] {} -> {}".format(i + 1, s.name, files_str))
                else:
                    print("  [{}] {} (command)".format(i + 1, s.name))

            # Log the step being re-executed (potentially corrupted)
            restart_step = steps[start_index]
            print("\n{}Re-executing step (last completed, potentially corrupted):{}\n".format(
                Fore.YELLOW, Style.RESET_ALL
            ))
            print("  [{}] {}".format(start_index + 1, restart_step.name))
            for f in restart_step.output_files:
                path = os.path.join(claude_config.cwd, f)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    print("    Deleting: {} ({} bytes)".format(f, size))
                    os.remove(path)

            # Log remaining steps
            remaining = len(steps) - start_index
            print("\n{}Remaining steps to execute ({}):{}\n".format(
                Fore.CYAN, remaining, Style.RESET_ALL
            ))
            for i in range(start_index, len(steps)):
                s = steps[i]
                if s.output_files:
                    files_str = ", ".join(s.output_files)
                    print("  [{}] {} -> {}".format(i + 1, s.name, files_str))
                else:
                    print("  [{}] {} (command)".format(i + 1, s.name))
        else:
            print("{}No completed steps found, starting from beginning{}".format(
                Fore.YELLOW, Style.RESET_ALL
            ))

        print("\n{}=== END RESUME SUMMARY ==={}".format(Fore.YELLOW, Style.RESET_ALL))
        print()

    thinking = claude_config.thinking if claude_config.thinking else {"type": "adaptive"}

    options = ClaudeAgentOptions(
        system_prompt=profile_system_prompt(),
        allowed_tools=claude_config.allowed_tools,
        permission_mode=claude_config.perm_mode,
        cwd=claude_config.cwd,
        env={
            "ANTHROPIC_MODEL": claude_config.model,
            "MAX_THINKING_TOKENS": str(claude_config.max_thinking_tokens),
        },
        model=claude_config.model,
        thinking=thinking,
        effort=claude_config.effort,
    )

    step_timings = []

    async with ClaudeSDKClient(options=options) as client:
        prompt_step = 0
        for i, step in enumerate(steps):
            if i < start_index:
                continue

            prompt = step.prompt
            if isinstance(prompt, dict):
                if "fn" in prompt:
                    fn_name = prompt.get("fn_name", str(prompt["fn"]))
                    print("{} RUN CMD: {} {}".format(Fore.GREEN, fn_name, Style.RESET_ALL))
                    prompt["fn"]()
                else:
                    cmd = prompt["cmd"]
                    print("{} RUN CMD: {} {}".format(Fore.GREEN, cmd, Style.RESET_ALL))
                    eval(cmd)
                continue

            prompt_step += 1
            start_time = time.time()

            print("{} STEP {} ({}): SEND QUERY PROMPT:{}".format(
                Fore.GREEN, i + 1, step.name, Style.RESET_ALL
            ))
            print(
                "{}============================================{}".format(
                    Fore.GREEN, Style.RESET_ALL
                )
            )
            print(prompt)
            print(
                "{}============================================{}".format(
                    Fore.CYAN, Style.RESET_ALL
                )
            )
            print("{} RESPONSES:{}".format(Fore.CYAN, Style.RESET_ALL))
            print(
                "{}============================================{}".format(
                    Fore.CYAN, Style.RESET_ALL
                )
            )
            await client.query(prompt)
            step_usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            async for msg in client.receive_response():
                print_Message(msg)
                if isinstance(msg, ResultMessage):
                    if msg.usage:
                        step_usage["input_tokens"] += msg.usage.get("input_tokens", 0)
                        step_usage["output_tokens"] += msg.usage.get("output_tokens", 0)
                    if msg.total_cost_usd:
                        step_usage["cost_usd"] += msg.total_cost_usd
                print(
                    "{}--------------------------------------------{}".format(
                        Fore.CYAN, Style.RESET_ALL
                    )
                )
            print(
                "{}============================================{}".format(
                    Fore.CYAN, Style.RESET_ALL
                )
            )

            duration_time = time.time() - start_time
            step_timings.append({
                "name": step.name,
                "duration": duration_time,
                "input_tokens": step_usage["input_tokens"],
                "output_tokens": step_usage["output_tokens"],
                "cost_usd": step_usage["cost_usd"],
            })
            mins, secs = divmod(int(duration_time), 60)
            hrs, mins = divmod(mins, 60)
            human_time = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
            print("FINISHED STEP {} ({}): duration = {:.1f}s ({})".format(
                i + 1, step.name, duration_time, human_time
            ))

    return step_timings
