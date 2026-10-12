import asyncio
import copy
import json
import subprocess
from contextlib import asynccontextmanager

import pytest

from common.claude_utils import PipelineStep
from llm.auto_code_gen.agents import agent_run, agent_session
from llm.auto_code_gen.agents.client import Reply
from llm.auto_code_gen.agents.config import resolve_agent_config
from llm.auto_code_gen.agents.runner import (
    AgentError,
    LiteLLMSession,
    _artifact_overwrite_error,
    _artifact_write_error,
    _code_generation_finalize_tools,
    _code_generation_progress_reminder,
    _code_patch_artifact_error,
    _fallback_summary,
    _final_artifact_write_error,
    _format_human_event,
    _heal_artifact_write,
    _iteration_budget_reminder,
    _source_write_error,
    _target_tree_has_changes,
    _tools_for_iteration,
    execute_tool,
    tool_schemas,
)


class FakeTransport:
    model = "openai/test"

    def __init__(self, replies=(), context=4096):
        self.replies = list(replies)
        self.requests = []
        self.context = context

    def context_length(self):
        return self.context

    def count_tokens(self, messages, tools=None):
        return len(json.dumps([messages, tools or []])) // 4

    async def complete(self, messages, tools=None, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        reply = (
            self.replies.pop(0)
            if self.replies
            else Reply(text="done", input_tokens=10, output_tokens=2)
        )
        if isinstance(reply, Exception):
            raise reply
        return reply


def cfg(tmp_path, **kw):
    return resolve_agent_config(
        {"model": "openai/test", "model_params": {"max_tokens": 256}, **kw},
        cwd=str(tmp_path),
    )


def call(name, args, id="call_1"):
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_human_event_formatter_contains_transcript_sections():
    rendered = _format_human_event(
        {
            "time": 0,
            "kind": "assistant",
            "step": "trace",
            "text": "I inspected the source.",
            "reasoning": "I need to inspect the repository before answering.",
            "finish_reason": "tool_calls",
            "tool_calls": [call("bash", {"command": "git status"})],
        }
    )
    assert "AGENT REASONING:" in rendered
    assert "I need to inspect the repository before answering." in rendered
    assert "ASSISTANT REPLY:" in rendered
    assert "I inspected the source." in rendered
    assert "TOOL CALL 1: bash" in rendered
    assert '"command": "git status"' in rendered


def test_fallback_summary_is_bounded_and_keeps_both_ends():
    messages = [
        {"role": "assistant", "content": "first-decision " + "x" * 3000},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("read_file", {"path": "/important/path"})],
        },
        {"role": "tool", "content": "last-result " + "y" * 3000},
    ]
    summary = _fallback_summary(messages, max_chars=1000)
    assert len(summary) <= 1030
    assert "first-decision" in summary
    assert "last-result" in summary
    assert "[truncated]" in summary


def test_iteration_budget_reminders_force_artifact_synthesis():
    assert _iteration_budget_reminder(49, 100, ["trace.txt"]) is None
    assert "begin writing trace.txt" in _iteration_budget_reminder(
        50, 100, ["trace.txt"]
    )
    assert "Stop research now" in _iteration_budget_reminder(75, 100, ["trace.txt"])
    assert "Finalize and verify trace.txt" in _iteration_budget_reminder(
        90, 100, ["trace.txt"]
    )
    assert _iteration_budget_reminder(50, None, ["trace.txt"]) is None


def test_late_iteration_tools_force_write_then_final_response():
    tools = tool_schemas(["Read", "Write", "Bash"])
    selected, choice, mode = _tools_for_iteration(tools, 74, 100, True, True)
    assert selected == tools and choice is None and mode == "normal"
    selected, choice, mode = _tools_for_iteration(tools, 75, 100, True, True)
    assert [tool["function"]["name"] for tool in selected] == ["write_file"]
    assert choice == "required" and mode == "write_only"
    selected, choice, mode = _tools_for_iteration(tools, 95, 100, False, True)
    assert selected == [] and choice == "none" and mode == "final_response"
    selected, choice, mode = _tools_for_iteration(
        tools, 80, 100, False, True, write_phase=True
    )
    assert [tool["function"]["name"] for tool in selected] == ["write_file"]
    assert choice == "required" and mode == "write_only"


def test_code_generation_can_delay_the_write_only_boundary():
    tools = tool_schemas(["Read", "Write", "Bash"])
    assert (
        _iteration_budget_reminder(45, 60, ["code_gen/V1.patch"], write_at=50) is None
    )
    selected, choice, mode = _tools_for_iteration(
        tools, 49, 60, True, True, write_at=50
    )
    assert selected == tools and choice is None and mode == "normal"
    selected, choice, mode = _tools_for_iteration(
        tools, 50, 60, True, True, write_at=50
    )
    assert [tool["function"]["name"] for tool in selected] == ["write_file"]
    assert choice == "required" and mode == "write_only"


def test_code_generation_progress_reminders_require_working_tree_edits():
    assert "actual target source" in _code_generation_progress_reminder(
        "code_gen_V1", 20, 60
    )
    assert "Finish the working-tree" in _code_generation_progress_reminder(
        "code_gen_review_V1", 40, 60
    )
    assert _code_generation_progress_reminder("test_plan_V1", 20, 60) is None


def test_source_checkpoint_allows_only_absolute_target_writes(tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    assert _source_write_error({"path": str(target / "module.py")}, target) is None
    assert "absolute" in _source_write_error({"path": "module.py"}, target)
    assert "temporary files" in _source_write_error(
        {"path": str(tmp_path / "artifact.patch")}, target
    )


def test_source_checkpoint_detects_existing_working_tree_changes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert _target_tree_has_changes(tmp_path) is False
    (tmp_path / "new.py").write_text("change\n")
    assert _target_tree_has_changes(tmp_path) is True


def test_strict_finalization_cannot_overwrite_source(tmp_path):
    assert (
        _final_artifact_write_error({"path": str(tmp_path / "V1.patch")}, tmp_path)
        is None
    )
    assert "source-tree writes" in _final_artifact_write_error(
        {"path": str(tmp_path.parent / "repo" / "module.py")}, tmp_path
    )


def test_codegen_finalization_keeps_bash_until_last_three_turns():
    tools = tool_schemas(["Read", "Write", "Bash"])
    write_only = [tool for tool in tools if tool["function"]["name"] == "write_file"]
    selected, choice, mode = _code_generation_finalize_tools(
        tools, "code_gen_V1", 50, 60, write_only, "required", "write_only"
    )
    assert [tool["function"]["name"] for tool in selected] == ["write_file", "bash"]
    assert choice is None and mode == "finalize"
    selected, choice, mode = _code_generation_finalize_tools(
        tools, "code_gen_V1", 57, 60, write_only, "required", "write_only"
    )
    assert selected == write_only and choice == "required" and mode == "write_only"


def test_resume_starts_at_first_incomplete_artifact_step(tmp_path):
    from common.claude_utils import _find_resume_index

    (tmp_path / "V1.patch").write_text("complete")
    steps = [
        PipelineStep("cleanup", {"fn": lambda: None}),
        PipelineStep("codegen", "prompt", output_files=["V1.patch"]),
        PipelineStep("review", "prompt", output_files=["V1_review.patch"]),
    ]
    assert _find_resume_index(steps, str(tmp_path)) == 2


def test_resume_includes_cleanup_before_incomplete_codegen(tmp_path):
    from common.claude_utils import _find_resume_index

    steps = [
        PipelineStep("cleanup", {"fn": lambda: None}),
        PipelineStep("codegen", "prompt", output_files=["V1.patch"]),
        PipelineStep("review", "prompt", output_files=["V1_review.patch"]),
    ]
    assert _find_resume_index(steps, str(tmp_path)) == 0


def test_not_converged_verdict_is_not_a_convergence_marker(tmp_path):
    from common.iteration_utils import check_converged

    summary = tmp_path / "summary.txt"
    summary.write_text("NOT CONVERGED\nissues remain\n")
    assert check_converged(str(tmp_path), "summary.txt") is False
    summary.write_text("CONVERGED\n")
    assert check_converged(str(tmp_path), "summary.txt") is True


def test_code_patch_artifact_must_match_implemented_tree(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    source = repo / "file.py"
    source.write_text("old\n")
    subprocess.run(["git", "add", "file.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    source.write_text("new\n")
    patch = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    (out / "V1.patch").write_text(patch)
    prompt = f"<target_source_code_dir>{repo}</target_source_code_dir>"
    assert _code_patch_artifact_error("code_gen_V1", prompt, ["V1.patch"], out) is None
    source.write_text("old\n")
    assert "does not match" in _code_patch_artifact_error(
        "code_gen_V1", prompt, ["V1.patch"], out
    )


def test_session_caps_tool_output_to_preserve_implementation_context(tmp_path):
    config = resolve_agent_config(
        {"backend": "litellm", "model": "openai/test", "api_key": "dummy"},
        cwd=str(tmp_path),
    )
    session = LiteLLMSession(config, transport=FakeTransport(context=65536))
    assert session.cap == 12000


def test_run_artifact_writes_are_limited_to_current_step_outputs(tmp_path):
    assert (
        _artifact_write_error(
            {"path": "current/result.txt"}, tmp_path, ["current/result.txt"]
        )
        is None
    )
    error = _artifact_write_error(
        {"path": "completed/trace.txt"}, tmp_path, ["current/result.txt"]
    )
    assert "undeclared run artifact" in error
    assert (
        _artifact_write_error(
            {"path": str(tmp_path.parent / "target.py")},
            tmp_path,
            ["current/result.txt"],
        )
        is None
    )


def test_later_artifact_chunks_cannot_truncate_same_file(tmp_path):
    target = (tmp_path / "result.txt").resolve()
    assert (
        _artifact_overwrite_error(
            {"path": "result.txt", "append": False}, tmp_path, set()
        )
        is None
    )
    assert "append=true" in _artifact_overwrite_error(
        {"path": "result.txt", "append": False}, tmp_path, {target}
    )
    assert (
        _artifact_overwrite_error(
            {"path": "result.txt", "append": True}, tmp_path, {target}
        )
        is None
    )


def test_heal_artifact_write_converts_or_dedupes(tmp_path):
    first = {
        "path": str(tmp_path / "result.txt"),
        "content": "section one\n",
        "append": True,
    }
    with open(tmp_path / "result.txt", "w") as handle:
        handle.write(first["content"])
    target = (tmp_path / "result.txt").resolve()
    healed, note = _heal_artifact_write(
        {"path": "result.txt", "content": "section two\n", "append": False},
        tmp_path,
        {target},
    )
    assert healed is not None and (tmp_path / healed["path"]).resolve() == target
    assert healed["append"] is True and note is None
    healed, note = _heal_artifact_write(
        {"path": "result.txt", "content": "section one\n", "append": False},
        tmp_path,
        {target},
    )
    assert healed is None and "duplicate" in note
    healed, note = _heal_artifact_write(
        {"path": "result.txt", "content": "section two\n", "append": True},
        tmp_path,
        {target},
    )
    assert healed is not None and healed["append"] is True


@pytest.mark.asyncio
async def test_write_file_can_append_large_artifact_sections(tmp_path):
    config = cfg(tmp_path)
    first = await execute_tool(
        "write_file",
        {
            "path": "trace.txt",
            "content": "section one\n",
            "append": False,
        },
        config,
        1000,
    )
    second = await execute_tool(
        "write_file",
        {
            "path": "trace.txt",
            "content": "section two\n",
            "append": True,
        },
        config,
        1000,
    )
    assert first.startswith("Wrote") and second.startswith("Appended")
    assert (tmp_path / "trace.txt").read_text() == "section one\nsection two\n"


@pytest.mark.asyncio
async def test_write_then_finish_and_history_across_calls(tmp_path):
    transport = FakeTransport(
        [
            Reply(tool_calls=[call("write_file", {"path": "result", "content": "ok"})]),
            Reply(text="saved"),
        ]
    )
    session = LiteLLMSession(cfg(tmp_path), transport)
    first = await agent_run(
        session.config,
        [PipelineStep("first", "write the result", ["result"])],
        client=session,
    )
    await agent_run(session.config, ["review the result"], client=session)
    assert (tmp_path / "result").read_text() == "ok"
    assert any(m.get("content") == "write the result" for m in transport.requests[-1])
    assert any(m.get("content") == "saved" for m in transport.requests[-1])
    assert first[0]["status"] == "completed"
    assert first[0]["end_time"] >= first[0]["start_time"]
    assert first[0]["cost_unknown"] is True


@pytest.mark.asyncio
async def test_failed_tool_result_is_available_for_unattended_diagnosis(tmp_path):
    transport = FakeTransport(
        [
            Reply(
                reasoning="I should read the requested file.",
                tool_calls=[call("read_file", {"path": "missing.txt"})],
            ),
            Reply(text="Cannot read missing.txt"),
        ]
    )
    await LiteLLMSession(cfg(tmp_path), transport).run(["read the input"])
    events = [
        json.loads(line)
        for line in (tmp_path / "agent_events.jsonl").read_text().splitlines()
    ]
    assistant = next(event for event in events if event["kind"] == "assistant")
    result = next(event for event in events if event["kind"] == "tool_result")
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"
    assert assistant["reasoning"] == "I should read the requested file."
    assert "No such file" in result["result"]
    assert result["result"] == transport.requests[-1][-1]["content"]
    assert (tmp_path / "agent_events.jsonl").stat().st_mode & 0o777 == 0o600
    human_log = (tmp_path / "agent_events.log").read_text()
    assert "USER PROMPT:" in human_log
    assert "AGENT REASONING:" in human_log
    assert "I should read the requested file." in human_log
    assert "TOOL CALL 1: read_file" in human_log
    assert "TOOL RESULT:" in human_log
    assert (tmp_path / "agent_events.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_arguments", ['{"path": "report", "content": "unfinished', "[]"]
)
async def test_invalid_tool_arguments_are_not_executed_or_resent(
    tmp_path, bad_arguments
):
    malformed = call("write_file", {})
    malformed["function"]["arguments"] = bad_arguments
    transport = FakeTransport(
        [
            Reply(tool_calls=[malformed]),
            Reply(
                tool_calls=[
                    call(
                        "write_file", {"path": "report", "content": "complete"}, "retry"
                    )
                ]
            ),
            Reply(text="saved"),
        ]
    )
    await LiteLLMSession(cfg(tmp_path), transport).run(
        [PipelineStep("write", "write report", ["report"])]
    )
    history = transport.requests[1]
    assert history[-2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert "NOT executed" in history[-1]["content"]
    assert (tmp_path / "report").read_text() == "complete"
    events = [
        json.loads(line)
        for line in (tmp_path / "agent_events.jsonl").read_text().splitlines()
    ]
    assistant = next(event for event in events if event["kind"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["arguments"] == bad_arguments


@pytest.mark.asyncio
async def test_generic_does_not_open_sdk(tmp_path, monkeypatch):
    import common.claude_utils as native

    monkeypatch.setattr(
        native, "ClaudeSDKClient", lambda **kw: pytest.fail("SDK was constructed")
    )
    monkeypatch.setattr(
        "llm.auto_code_gen.agents.runner.LLMClient", lambda config: FakeTransport()
    )
    async with agent_session(cfg(tmp_path)) as client:
        assert (await agent_run(client.config, ["hello"], client=client))[0][
            "status"
        ] == "completed"


@pytest.mark.asyncio
async def test_claude_adapter_preserves_native_calls(tmp_path, monkeypatch):
    import common.claude_utils as native

    seen = []
    token = object()

    @asynccontextmanager
    async def session(config):
        seen.append(config)
        yield token

    async def run(config, steps, resume=False, client=None):
        assert client is token and resume
        return [{"name": "native"}]

    monkeypatch.setattr(native, "claude_session", session)
    monkeypatch.setattr(native, "claude_run", run)
    config = resolve_agent_config(cwd=str(tmp_path))
    async with agent_session(config) as client:
        assert await agent_run(config, ["hello"], resume=True, client=client) == [
            {"name": "native"}
        ]
    assert seen[0].model == config.model
    assert not hasattr(seen[0], "api_key")


@pytest.mark.asyncio
async def test_mismatched_session_rejected(tmp_path):
    session = LiteLLMSession(cfg(tmp_path), FakeTransport())
    with pytest.raises(ValueError, match="match"):
        await agent_run(
            cfg(tmp_path, api_base="http://example.invalid"), ["hi"], client=session
        )
    with pytest.raises(ValueError):
        await agent_run(resolve_agent_config(cwd=str(tmp_path)), ["hi"], client=session)


@pytest.mark.asyncio
async def test_disallowed_tool_never_executes(tmp_path):
    transport = FakeTransport(
        [
            Reply(
                tool_calls=[call("write_file", {"path": "forbidden", "content": "no"})]
            ),
            Reply(text="done"),
        ]
    )
    await LiteLLMSession(cfg(tmp_path, allowed_tools=["Read"]), transport).run(
        ["read only"]
    )
    assert not (tmp_path / "forbidden").exists()
    assert "not allowed" in transport.requests[-1][-1]["content"]
    assert tool_schemas([]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replies, match",
    [
        ([RuntimeError("unavailable")], "request failed"),
        ([Reply(), Reply(), Reply()], "empty answer"),
    ],
)
async def test_provider_failure_is_not_success(tmp_path, replies, match):
    session = LiteLLMSession(cfg(tmp_path), FakeTransport(replies))
    with pytest.raises(AgentError, match=match) as error:
        await session.run(["work"])
    assert error.value.timings[-1]["status"] == "failed"
    with pytest.raises(AgentError, match="start a new session"):
        await session.run(["retry"])


@pytest.mark.asyncio
async def test_empty_answer_recovers_with_a_nudge(tmp_path):
    transport = FakeTransport(
        [Reply(), Reply(text="done", input_tokens=10, output_tokens=2)]
    )
    session = LiteLLMSession(cfg(tmp_path), transport)
    timings = await session.run(["work"])
    assert timings[0]["status"] == "completed"
    assert "empty" in transport.requests[-1][-1]["content"].lower()


@pytest.mark.asyncio
async def test_truncated_response_retries_with_a_nudge(tmp_path):
    transport = FakeTransport(
        [
            Reply(text="loop" * 200, finish_reason="length"),
            Reply(tool_calls=[call("read_file", {"path": "input.txt"})]),
            Reply(text="complete"),
        ]
    )
    (tmp_path / "input.txt").write_text("data")
    result = await LiteLLMSession(cfg(tmp_path), transport).run(["work"])
    assert result[0]["status"] == "completed"
    assert "exceeded the output limit" in transport.requests[1][-1]["content"]


@pytest.mark.asyncio
async def test_repeatedly_truncated_response_fails(tmp_path):
    replies = [Reply(text="loop" * 200, finish_reason="length") for _ in range(4)]
    with pytest.raises(AgentError, match="exceeded the output limit") as error:
        await LiteLLMSession(cfg(tmp_path), FakeTransport(replies)).run(["work"])
    assert error.value.timings[-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_missing_artifact_fails(tmp_path):
    with pytest.raises(AgentError, match="required output"):
        await LiteLLMSession(cfg(tmp_path), FakeTransport()).run(
            [PipelineStep("write", "write", ["missing"])]
        )


@pytest.mark.asyncio
async def test_agent_is_reminded_to_write_missing_required_artifact(tmp_path):
    transport = FakeTransport(
        [
            Reply(text="I will write the report now."),
            Reply(
                tool_calls=[call("write_file", {"path": "report", "content": "done"})]
            ),
            Reply(text="complete"),
        ]
    )
    result = await LiteLLMSession(cfg(tmp_path), transport).run(
        [
            PipelineStep("trace", "trace code", ["report"]),
        ]
    )
    assert result[0]["status"] == "completed"
    assert (tmp_path / "report").read_text() == "done"
    assert "required output files" in transport.requests[1][-1]["content"]


@pytest.mark.asyncio
async def test_iteration_limit_retains_usage(tmp_path):
    response = Reply(
        tool_calls=[call("read_file", {"path": "missing"})],
        input_tokens=12,
        output_tokens=4,
        cost_usd=0.01,
    )
    with pytest.raises(AgentError, match="iteration limit") as error:
        await LiteLLMSession(
            cfg(tmp_path, max_agent_iterations=1), FakeTransport([response])
        ).run(["work"])
    assert error.value.timings[0]["input_tokens"] == 12
    assert error.value.timings[0]["cost_usd"] == 0.01


@pytest.mark.asyncio
async def test_unbounded_agent_iterations_can_finish(tmp_path):
    responses = [
        Reply(tool_calls=[call("read_file", {"path": "missing"})]),
        Reply(text="done"),
    ]
    result = await LiteLLMSession(
        cfg(tmp_path, max_agent_iterations=None), FakeTransport(responses)
    ).run(["work"])
    assert result[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_resume_skips_all_completed_steps(tmp_path):
    (tmp_path / "one").write_text("ok")
    (tmp_path / "two").write_text("ok")
    transport = FakeTransport()
    steps = [
        PipelineStep("cleanup", {"fn": lambda: pytest.fail("replayed cleanup")}),
        PipelineStep("one", "one", ["one"]),
        PipelineStep("two", "two", ["two"]),
    ]
    timings = await LiteLLMSession(cfg(tmp_path), transport).run(steps, resume=True)
    assert timings == []
    assert (tmp_path / "two").read_text() == "ok"
    assert transport.requests == []


@pytest.mark.asyncio
async def test_callable_step_runs_without_model(tmp_path):
    seen = []
    transport = FakeTransport()
    assert (
        await LiteLLMSession(cfg(tmp_path), transport).run(
            [{"fn": lambda: seen.append(True)}]
        )
        == []
    )
    assert seen == [True] and not transport.requests


@pytest.mark.asyncio
async def test_compaction_preserves_active_prompt_and_tool_pairs(tmp_path):
    transport = FakeTransport(context=2048)
    session = LiteLLMSession(cfg(tmp_path), transport)
    session.messages.append({"role": "user", "content": "original task"})
    for i in range(16):
        session.messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [call("read_file", {"path": "code"}, str(i))],
                },
                {"role": "tool", "tool_call_id": str(i), "content": "code" * 250},
            ]
        )
    session.messages.append({"role": "user", "content": "current review request"})
    timings = await session.run(["continue review"])
    request = transport.requests[-1]
    assert any(m["content"] == "original task" for m in request)
    assert any(m["content"] == "continue review" for m in request)
    assert session._count(request) <= session.budget
    assert timings[0]["compactions"] >= 1
    assert timings[0]["input_tokens"] == 20
    calls = {c["id"] for m in request for c in m.get("tool_calls", [])}
    assert all(m["tool_call_id"] in calls for m in request if m["role"] == "tool")


@pytest.mark.asyncio
async def test_provider_context_rejection_forces_compaction_when_estimate_is_low(
    tmp_path,
):
    from litellm import ContextWindowExceededError

    transport = FakeTransport(
        [
            ContextWindowExceededError(
                message="too large", model="test", llm_provider="openai"
            ),
            Reply(text="Earlier work is saved in report.txt"),
            Reply(text="current review complete"),
        ],
        context=16384,
    )
    transport.count_tokens = lambda *args, **kwargs: 10
    session = LiteLLMSession(cfg(tmp_path), transport)
    session.messages.extend(
        [
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "old detail " * 1000},
        ]
    )
    timings = await session.run(["review saved report"])
    assert timings[0]["compactions"] == 1
    retry = transport.requests[-1]
    assert any(m["content"] == "review saved report" for m in retry)
    assert not any("old detail " in m.get("content", "") for m in retry)
    assert any("report.txt" in m.get("content", "") for m in retry)


@pytest.mark.asyncio
async def test_oversized_original_fails_without_request(tmp_path):
    transport = FakeTransport(context=1024)
    with pytest.raises(AgentError, match="Original task"):
        await LiteLLMSession(cfg(tmp_path), transport).run(["x" * 10000])
    assert not transport.requests


@pytest.mark.asyncio
async def test_chunked_read_and_shell_exit(tmp_path):
    (tmp_path / "code").write_text("first\nsecond\nthird\n")
    config = cfg(tmp_path)
    result = await execute_tool(
        "read_file", {"path": "code", "offset": 1, "limit": 1}, config, 1000
    )
    assert "second" in result and "first" not in result and "next offset=2" in result
    result = await execute_tool(
        "bash", {"command": "echo problem; exit 7"}, config, 1000
    )
    assert "problem" in result and "exit code: 7" in result


@pytest.mark.asyncio
async def test_command_timeout(tmp_path):
    result = await execute_tool(
        "bash", {"command": "sleep 10"}, cfg(tmp_path, command_timeout=1), 1000
    )
    assert "timed out" in result


@pytest.mark.asyncio
async def test_cancel_command(tmp_path):
    task = asyncio.create_task(
        execute_tool("bash", {"command": "sleep 10"}, cfg(tmp_path), 1000)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_compaction_within_current_step_and_summary_failure(tmp_path):
    transport = FakeTransport([RuntimeError("summary unavailable")], context=2048)
    session = LiteLLMSession(cfg(tmp_path), transport)
    session.messages.append({"role": "user", "content": "original"})
    for i in range(15):
        session.messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [call("read_file", {"path": "code"}, str(i))],
                },
                {"role": "tool", "tool_call_id": str(i), "content": "x" * 900},
            ]
        )
    timing = dict(
        input_tokens=0, output_tokens=0, cost_usd=0, compactions=0, cost_unknown=False
    )
    await session.compact(timing)
    assert session._count(session.messages) <= session.budget
    assert session.messages[1]["content"] == "original"
    assert timing["cost_unknown"] and timing["usage_incomplete"]
    for index, message in enumerate(session.messages):
        if message["role"] == "tool":
            assert session.messages[index - 1].get("tool_calls")


@pytest.mark.asyncio
async def test_repeat_limit_keeps_pairs_complete(tmp_path):
    reply = Reply(tool_calls=[call("read_file", {"path": "missing"})])
    session = LiteLLMSession(cfg(tmp_path), FakeTransport([reply] * 16))
    with pytest.raises(AgentError, match="repeated"):
        await session.run(["work"])
    requested = {
        c["id"]
        for m in session.messages
        if m["role"] == "assistant"
        for c in m.get("tool_calls") or []
    }
    answered = {m["tool_call_id"] for m in session.messages if m["role"] == "tool"}
    assert requested and answered == requested


def test_known_output_limit_checked(tmp_path):
    transport = FakeTransport()
    transport.output_limit = lambda: 32
    with pytest.raises(ValueError, match="output limit"):
        LiteLLMSession(cfg(tmp_path), transport)


"""Pipeline-integration agent tests (deferred from test_agent_runner).

These exercise run_pipeline(client=...) and the LLMFrameworkUseCase PR-commands
phase; they activate once Phases 4-5 of the model_agnostic merge land.
"""


@pytest.mark.asyncio
async def test_pipeline_routes_all_phases_and_resumes(tmp_path, monkeypatch):
    import common.claude_utils as native
    from llm.auto_code_gen.code_gen_configs import CodeGenConfig
    from llm.auto_code_gen.run_code_gen import run_pipeline

    monkeypatch.setattr(
        native, "ClaudeSDKClient", lambda **kw: pytest.fail("unexpected SDK")
    )

    class WritingTransport(FakeTransport):
        async def complete(self, messages, tools=None, **kwargs):
            self.requests.append(copy.deepcopy(messages))
            if messages[-1]["role"] == "user":
                name = messages[-1]["content"]
                return Reply(
                    tool_calls=[
                        call("write_file", {"path": name, "content": "verified"})
                    ]
                )
            return Reply(text="saved")

    class FixtureUseCase:
        def create_context_str(self, *args):
            return "fixture"

        def gen_code_trace_steps(self, *args):
            return [PipelineStep("trace", "trace.txt", ["trace.txt"])], ["trace.txt"]

        def skip_test_plan_phase(self, config):
            return False

        def steps(self, name):
            return [
                PipelineStep(name, name, [name]),
                PipelineStep("unused review", "unused"),
            ], None

        def gen_code_port_plan_iter_steps(self, *args):
            return self.steps("plan.txt")

        def gen_test_plan_iter_steps(self, *args):
            return self.steps("test-plan.txt")

        def gen_code_gen_iter_steps(self, *args):
            return self.steps("patch.diff")

        async def run_runtime_iterations(
            self, context, config, agent, *args, resume=False, client=None
        ):
            assert client is session
            return [], await agent_run(
                agent,
                [PipelineStep("runtime", "runtime.txt", ["runtime.txt"])],
                client=client,
            )

    config = CodeGenConfig(
        output_dir=str(tmp_path),
        code_port_plan_skip_review=True,
        test_plan_skip_review=True,
        code_gen_skip_review=True,
    )
    agent = cfg(tmp_path)
    transport = WritingTransport(context=16384)
    session = LiteLLMSession(agent, transport)
    phases, timings = await run_pipeline(
        config, agent, FixtureUseCase(), client=session
    )
    assert [t["name"] for t in timings] == [
        "trace",
        "plan.txt",
        "test-plan.txt",
        "patch.diff",
        "runtime",
    ]
    assert len(phases) == 4
    assert all(
        (tmp_path / name).is_file()
        for name in [
            "trace.txt",
            "plan.txt",
            "test-plan.txt",
            "patch.diff",
            "runtime.txt",
        ]
    )
    assert any(m.get("content") == "trace.txt" for m in transport.requests[-1])
    _, resumed = await run_pipeline(
        config, agent, FixtureUseCase(), resume=True, client=session
    )
    assert [t["name"] for t in resumed] == ["runtime"]


@pytest.mark.asyncio
async def test_repeated_tool_calls_get_steered_not_killed(tmp_path):
    (tmp_path / "input.txt").write_text("data")
    same = [call("read_file", {"path": "input.txt"})]
    transport = FakeTransport(
        [Reply(tool_calls=same) for _ in range(4)] + [Reply(text="done")]
    )
    result = await LiteLLMSession(cfg(tmp_path), transport).run(["work"])
    assert result[0]["status"] == "completed"
    assert "That exact tool call" in transport.requests[-1][-1]["content"]


@pytest.mark.asyncio
async def test_agent_stuck_forever_still_fails(tmp_path):
    (tmp_path / "input.txt").write_text("data")
    same = [call("read_file", {"path": "input.txt"})]
    transport = FakeTransport([Reply(tool_calls=same) for _ in range(16)])
    with pytest.raises(AgentError, match="repeated the same tool calls") as error:
        await LiteLLMSession(cfg(tmp_path), transport).run(["work"])
    assert error.value.timings[-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_truncation_nudge_keeps_alternation(tmp_path):
    transport = FakeTransport(
        [
            Reply(text="loop" * 4000, finish_reason="length"),
            Reply(tool_calls=[call("read_file", {"path": "input.txt"})]),
            Reply(text="complete"),
        ]
    )
    (tmp_path / "input.txt").write_text("data")
    await LiteLLMSession(cfg(tmp_path), transport).run(["work"])
    history = transport.requests[1]
    assert [m["role"] for m in history[-2:]] == ["assistant", "user"]
    assert "exceeded the output limit" in history[-1]["content"]
    assert len(history[-2]["content"]) <= 2000


@pytest.mark.asyncio
async def test_tool_call_written_as_markup_is_retried(tmp_path):
    (tmp_path / "input.txt").write_text("data")
    transport = FakeTransport(
        [
            Reply(text='<bash>\ncommand="cat input.txt"\n</bash>'),
            Reply(tool_calls=[call("read_file", {"path": "input.txt"})]),
            Reply(text="complete"),
        ]
    )
    result = await LiteLLMSession(cfg(tmp_path), transport).run(["work"])
    assert result[0]["status"] == "completed"
    assert "markup text" in transport.requests[1][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("descs_only", [True, False])
async def test_pr_commands_phase_forwards_agent_client(
    tmp_path, monkeypatch, descs_only
):
    from llm.auto_code_gen.code_gen_configs import CodeGenConfig
    from llm.auto_code_gen.use_cases.llm_framework import LLMFrameworkUseCase

    config = cfg(tmp_path)
    token = object()
    seen = []

    async def run(config, steps, client=None):
        seen.append((client, [step.name for step in steps]))
        return [{"status": "completed"}]

    monkeypatch.setattr("llm.auto_code_gen.use_cases.llm_framework.agent_run", run)
    results, timings = await LLMFrameworkUseCase().run_generate_pr_commands(
        "ctx",
        CodeGenConfig(output_dir=str(tmp_path)),
        config,
        descs_only=descs_only,
        client=token,
    )
    expected = ["update_commit_and_pr_descs"]
    if not descs_only:
        expected.append("generate_pr_commands")
    assert [name for _, names in seen for name in names] == expected
    assert len(seen) == len(expected)
    assert all(client is token for client, _ in seen)
    assert results[0]["name"] == "Generate PR commands"
