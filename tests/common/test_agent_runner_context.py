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
