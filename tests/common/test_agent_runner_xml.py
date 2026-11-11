# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the opt-in XML tool fallback.

The fallback must never execute examples quoted in ordinary assistant
text.  Only a message whose entire content is XML tool blocks is an
invocation, and only when the provider compatibility mode is enabled.
"""

import asyncio

from common import agent_runner
from common.agent_runner import (
    LiteLLMAgentRunner,
    _has_repeating_xml_sig,
    _parse_xml_tool_message,
)
from common.claude_utils import ClaudeConfig


class _Message:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    usage = None

    def __init__(self, content):
        self.choices = [_Choice(content)]


def _run_loop(monkeypatch, contents, *, enabled, allowed_tools=None):
    responses = [_Response(c) for c in contents]
    calls = []
    requests = []

    def fake_completion(**kwargs):
        if not responses:
            raise AssertionError("loop requested more responses than expected")
        requests.append(list(kwargs.get("messages", [])))
        return responses.pop(0)

    def fake_execute(name, args, _cwd, **_kwargs):
        calls.append((name, args))
        return "tool output"

    monkeypatch.setattr(
        agent_runner, "_litellm_completion_with_streaming", fake_completion
    )
    monkeypatch.setattr(agent_runner, "_execute_tool", fake_execute)
    monkeypatch.setattr(agent_runner, "_estimate_tokens", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        "common.llm_client.build_request_kwargs",
        lambda *_a, **_k: {},
    )

    config = ClaudeConfig(
        model="openai/test-model",
        allowed_tools=allowed_tools or ["Bash", "Read", "Write"],
        perm_mode="acceptEdits",
        cwd=None,
        xml_tool_fallback=enabled,
    )
    runner = LiteLLMAgentRunner()
    asyncio.run(
        runner._agent_loop(
            model=config.model,
            system_prompt="system",
            user_prompt="user",
            cwd=".",
            allowed_tools=config.allowed_tools,
            context_length=128000,
            xml_tool_fallback=config.xml_tool_fallback,
        )
    )
    return calls, responses, requests


def test_parser_rejects_fenced_example():
    text = (
        "Here is an example, do not execute it:\n"
        "```xml\n"
        "<bash><command>touch /tmp/pwned</command></bash>\n"
        "```\n"
    )
    assert _parse_xml_tool_message(text) == ([], [])


def test_parser_rejects_prose_around_block():
    text = (
        "I would run this to inspect the tree:\n"
        "<bash><command>git status</command></bash>\n"
        "but only after you confirm."
    )
    assert _parse_xml_tool_message(text) == ([], [])


def test_parser_accepts_whole_message_invocation():
    calls, errors = _parse_xml_tool_message(
        "<bash><command>git status</command></bash>"
    )
    assert calls == [("bash", {"command": "git status"})]
    assert errors == []


def test_parser_allows_fenced_content_inside_a_block():
    text = (
        "<write_file><path>notes.md</path>"
        "<content>```\ncode\n```</content></write_file>"
    )
    calls, errors = _parse_xml_tool_message(text)
    assert calls == [("write_file", {"path": "notes.md", "content": "```\ncode\n```"})]
    assert errors == []


def test_parser_accepts_multiple_blocks_only():
    text = (
        "<read_file><path>a.c</path></read_file>\n"
        "<bash><command>make</command></bash>"
    )
    calls, errors = _parse_xml_tool_message(text)
    assert calls == [
        ("read_file", {"path": "a.c"}),
        ("bash", {"command": "make"}),
    ]
    assert errors == []


def test_parser_reports_malformed_block_instead_of_silently_dropping_it():
    calls, errors = _parse_xml_tool_message(
        "<bash><command>git status</command></bash>\n<read_file></read_file>"
    )
    assert calls == [("bash", {"command": "git status"})]
    assert errors == ["Malformed <read_file> tool block: missing <path>."]


def test_parser_rejects_malformed_block_next_to_prose():
    text = (
        "Here is an example:\n"
        "<bash><command>git status</command></bash>\n"
        "<read_file></read_file>"
    )
    assert _parse_xml_tool_message(text) == ([], [])


def test_repeating_sig_detector_catches_alternation():
    assert _has_repeating_xml_sig(["a", "", "a", "", "a", "", "a", ""])
    assert _has_repeating_xml_sig(["a", "b", "c"] * 4)
    assert not _has_repeating_xml_sig(["a", "b", "c", "d", "e", "f", "g", "h"])


def test_loop_does_not_execute_quoted_xml_when_disabled(monkeypatch):
    text = (
        "Here is an example, do not execute it:\n"
        "```xml\n"
        "<bash><command>touch /tmp/pwned</command></bash>\n"
        "```\n"
    )
    calls, remaining, _requests = _run_loop(monkeypatch, [text], enabled=False)
    assert calls == []
    assert remaining == []


def test_loop_executes_whole_message_when_enabled(monkeypatch):
    calls, remaining, _requests = _run_loop(
        monkeypatch,
        ["<bash><command>git status</command></bash>", "done"],
        enabled=True,
    )
    assert calls == [("bash", {"command": "git status"})]
    assert remaining == []


def test_loop_feeds_malformed_sibling_block_back_to_model(monkeypatch):
    calls, remaining, requests = _run_loop(
        monkeypatch,
        [
            "<bash><command>git status</command></bash>\n<read_file></read_file>",
            "done",
        ],
        enabled=True,
    )
    assert calls == [("bash", {"command": "git status"})]
    assert remaining == []
    assert len(requests) == 2
    feedback = requests[1][-1]["content"]
    assert "Malformed <read_file> tool block: missing <path>." in feedback


def test_loop_feeds_malformed_only_block_back_to_model(monkeypatch):
    calls, remaining, requests = _run_loop(
        monkeypatch,
        ["<read_file></read_file>", "done"],
        enabled=True,
    )
    assert calls == []
    assert remaining == []
    assert "Malformed <read_file> tool block: missing <path>." in (
        requests[1][-1]["content"]
    )


def test_loop_rejects_disallowed_xml_tool_with_error_feedback(monkeypatch):
    calls, remaining, requests = _run_loop(
        monkeypatch,
        ["<read_file><path>secret</path></read_file>", "done"],
        enabled=True,
        allowed_tools=["Bash"],
    )
    assert calls == []
    assert remaining == []
    assert "not allowed by the pipeline configuration" in (requests[1][-1]["content"])


def test_xml_repeat_break_requests_final_answer(monkeypatch):
    invoke = "<bash><command>git status</command></bash>"
    calls, remaining, requests = _run_loop(
        monkeypatch,
        [invoke, invoke, invoke, invoke, "final answer"],
        enabled=True,
    )
    assert calls == [("bash", {"command": "git status"})] * 3
    assert remaining == []
    assert len(requests) == 5
    assert requests[-1][-1]["content"].startswith("You have already called")


def test_xml_alternating_valid_and_malformed_breaks_pattern(monkeypatch):
    valid = "<bash><command>git status</command></bash>"
    malformed = "<read_file></read_file>"
    calls, remaining, requests = _run_loop(
        monkeypatch,
        [valid, malformed] * 4 + ["final answer"],
        enabled=True,
    )
    assert calls == [("bash", {"command": "git status"})] * 4
    assert remaining == []
    assert len(requests) == 9
    assert requests[-1][-1]["content"].startswith("You have already called")


def test_xml_distinct_calls_are_not_treated_as_a_pattern(monkeypatch):
    contents = [f"<bash><command>echo {i}</command></bash>" for i in range(8)]
    calls, remaining, _requests = _run_loop(
        monkeypatch,
        contents + ["done"],
        enabled=True,
    )
    assert calls == [("bash", {"command": f"echo {i}"}) for i in range(8)]
    assert remaining == []


def test_loop_parses_xml_after_think_block(monkeypatch):
    tag = "think"
    thinking = f"<{tag}>\nI should inspect the tree.\n</{tag}>\n"
    calls, remaining, _requests = _run_loop(
        monkeypatch,
        [
            thinking + "<bash><command>git status</command></bash>",
            "done",
        ],
        enabled=True,
    )
    assert calls == [("bash", {"command": "git status"})]
    assert remaining == []


def test_config_defaults_to_disabled():
    config = ClaudeConfig(
        model="openai/test-model",
        allowed_tools=["Bash"],
        perm_mode="acceptEdits",
        cwd=None,
    )
    assert config.xml_tool_fallback is False
