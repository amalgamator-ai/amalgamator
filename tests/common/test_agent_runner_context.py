# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic history elision for long LiteLLM agent sessions."""

from common.agent_runner import _elide_stale_history


def _conversation(pairs: int = 20, chars: int = 4000) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task prompt"},
    ]
    for i in range(pairs):
        call_id = f"call_{i}"
        messages.append(
            {
                "role": "assistant",
                "content": f"step {i}",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "x"}'},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": "r" * chars}
        )
    return messages


def test_elides_stale_output_and_keeps_recent_tail_verbatim():
    messages = _conversation()

    elided_msgs, count = _elide_stale_history(
        "test-model", messages, keep_recent_tokens=2500
    )

    assert count > 0
    assert len(elided_msgs) == len(messages)
    # Head is protected.
    assert elided_msgs[0] == messages[0]
    assert elided_msgs[1] == messages[1]
    # The recent tail is untouched.
    assert elided_msgs[-1] == messages[-1]
    assert elided_msgs[-2] == messages[-2]
    # Old tool output becomes a marker, the call/result pairing survives.
    old_tool = elided_msgs[3]
    assert old_tool["role"] == "tool"
    assert old_tool["tool_call_id"] == "call_0"
    assert old_tool["content"].startswith("[earlier tool output elided:")
    old_assistant = elided_msgs[2]
    assert old_assistant["tool_calls"] == messages[2]["tool_calls"]
    assert old_assistant["content"] == "step 0"


def test_nothing_elided_below_keep_budget():
    messages = _conversation(pairs=2)

    elided_msgs, count = _elide_stale_history(
        "test-model", messages, keep_recent_tokens=1_000_000
    )

    assert count == 0
    assert elided_msgs is messages


def test_second_pass_is_a_no_op():
    messages = _conversation()
    once, _ = _elide_stale_history("test-model", messages, keep_recent_tokens=2500)

    twice, count = _elide_stale_history("test-model", once, keep_recent_tokens=2500)

    assert count == 0
    assert twice is once


def test_xml_tool_results_are_elided():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {
            "role": "user",
            "content": "<tool_results>\n" + "x" * 20_000 + "\n</tool_results>",
        },
        *[{"role": "assistant", "content": "r" * 1000} for _ in range(10)],
    ]

    elided_msgs, count = _elide_stale_history(
        "test-model", messages, keep_recent_tokens=500
    )

    assert count >= 1
    assert elided_msgs[2]["content"].startswith("[earlier tool results elided:")


def test_max_iterations_stops_the_loop(monkeypatch):
    import asyncio

    from common import agent_runner
    from common.agent_runner import LiteLLMAgentRunner

    class _Fn:
        name = "bash"
        arguments = '{"command": "true"}'

    class _Call:
        id = "call_1"
        function = _Fn()

    class _Message:
        content = ""
        tool_calls = [_Call()]

    class _Response:
        usage = None

        def __init__(self):
            self.choices = [type("C", (), {"message": _Message()})()]

    calls = []

    def fake_completion(**_kwargs):
        calls.append(1)
        return _Response()

    monkeypatch.setattr(
        agent_runner, "_litellm_completion_with_streaming", fake_completion
    )
    monkeypatch.setattr(agent_runner, "_execute_tool", lambda *_a, **_k: "ok")
    monkeypatch.setattr(agent_runner, "_estimate_tokens", lambda *_a, **_k: 0)
    monkeypatch.setattr("common.llm_client.build_request_kwargs", lambda *_a, **_k: {})

    asyncio.run(
        LiteLLMAgentRunner()._agent_loop(
            model="openai/test-model",
            system_prompt="s",
            user_prompt="u",
            cwd=".",
            allowed_tools=["Bash"],
            context_length=128000,
            max_iterations=3,
        )
    )

    assert len(calls) == 3


def test_wrapup_message_is_injected_once(monkeypatch):
    import asyncio

    from common import agent_runner
    from common.agent_runner import LiteLLMAgentRunner

    counter = {"n": 0}

    class _Fn:
        name = "bash"

        @property
        def arguments(self):
            counter["n"] += 1
            return '{"command": "true %d"}' % counter["n"]

    class _Call:
        id = "call_1"
        function = _Fn()

    class _Message:
        content = ""
        tool_calls = [_Call()]

    class _Response:
        usage = None

        def __init__(self):
            self.choices = [type("C", (), {"message": _Message()})()]

    requests = []

    def fake_completion(**kwargs):
        requests.append(list(kwargs.get("messages", [])))
        return _Response()

    monkeypatch.setattr(
        agent_runner, "_litellm_completion_with_streaming", fake_completion
    )
    monkeypatch.setattr(agent_runner, "_execute_tool", lambda *_a, **_k: "ok")
    monkeypatch.setattr(agent_runner, "_estimate_tokens", lambda *_a, **_k: 0)
    monkeypatch.setattr("common.llm_client.build_request_kwargs", lambda *_a, **_k: {})

    asyncio.run(
        LiteLLMAgentRunner()._agent_loop(
            model="openai/test-model",
            system_prompt="s",
            user_prompt="u",
            cwd=".",
            allowed_tools=["Bash"],
            context_length=128000,
            max_iterations=4,
            wrapup_at=2,
            wrapup_message="finish now",
        )
    )

    first = [m for m in requests[0] if m.get("content") == "finish now"]
    last = [m for m in requests[-1] if m.get("content") == "finish now"]
    assert first == []
    assert len(last) == 1
