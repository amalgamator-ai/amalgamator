import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web

from common.claude_utils import PipelineStep
from llm.auto_code_gen.agents.client import LLMClient, _merge_params, _message_reasoning
from llm.auto_code_gen.agents.config import resolve_agent_config
from llm.auto_code_gen.agents.runner import AgentError, LiteLLMSession


def test_unknown_model_does_not_use_silent_fallback_tokenizer(monkeypatch):
    import litellm

    calls = []

    def unknown(model):
        calls.append(model)
        raise ValueError("unknown model")

    monkeypatch.setattr(litellm, "get_model_info", unknown)
    monkeypatch.setattr(
        litellm, "token_counter", lambda **kwargs: pytest.fail("wrong tokenizer")
    )
    client = LLMClient(resolve_agent_config({"model": "openai/custom-model"}))
    assert client.count_tokens([{"role": "user", "content": "x" * 2000}]) > 1000
    assert client.output_limit() is None
    assert client.context_length() == 32768
    assert len(calls) == 1


@asynccontextmanager
async def endpoint(handler, port):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        await runner.cleanup()


def response(message, finish="stop"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", **message},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SimpleNamespace(reasoning="primary", reasoning_content="alternate"),
            "primary",
        ),
        (SimpleNamespace(reasoning=None, reasoning_content="alternate"), "alternate"),
        (
            SimpleNamespace(
                reasoning=None,
                reasoning_content=None,
                provider_specific_fields={"reasoning": "provider"},
            ),
            "provider",
        ),
    ],
)
def test_message_reasoning_normalizes_provider_fields(message, expected):
    assert _message_reasoning(message) == expected


def test_request_model_params_merge_without_mutating_defaults():
    defaults = {"temperature": 0.2, "extra_body": {"existing": True}}
    override = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    merged = _merge_params(defaults, override)
    assert merged == {
        "temperature": 0.2,
        "extra_body": {
            "existing": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    assert defaults == {"temperature": 0.2, "extra_body": {"existing": True}}


@pytest.mark.asyncio
async def test_real_litellm_local_tool_protocol(tmp_path, unused_tcp_port):
    requests = []

    async def handler(request):
        assert request.headers["Authorization"] == "Bearer test-only"
        body = await request.json()
        requests.append(body)
        assert body["model"] == "test-model"
        if len(requests) == 1:
            return web.json_response(
                response(
                    {
                        "content": None,
                        "reasoning": "I should write the requested file.",
                        "tool_calls": [
                            {
                                "id": "write-1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps(
                                        {"path": "result.txt", "content": "verified"}
                                    ),
                                },
                            }
                        ],
                    },
                    "tool_calls",
                )
            )
        assert body["messages"][-1]["role"] == "tool"
        assert body["messages"][-1]["tool_call_id"] == "write-1"
        return web.json_response(response({"content": "done"}))

    async with endpoint(handler, unused_tcp_port) as url:
        config = resolve_agent_config(
            {
                "model": "openai/test-model",
                "api_base": url,
                "api_key": "test-only",
                "stream": False,
                "billing": "self_hosted",
                "context_length": 8192,
                "model_params": {"max_tokens": 512},
            },
            cwd=str(tmp_path),
        )
        result = await LiteLLMSession(config).run(
            [PipelineStep("write", "write result", ["result.txt"])]
        )
    assert (tmp_path / "result.txt").read_text() == "verified"
    assert result[0]["input_tokens"] == 20
    assert result[0]["output_tokens"] == 10
    assert result[0]["cost_usd"] == 0 and not result[0]["cost_unknown"]
    assert len(requests) == 2
    human_log = (tmp_path / "agent_events.log").read_text()
    assert "AGENT REASONING:" in human_log
    assert "I should write the requested file." in human_log


@pytest.mark.asyncio
async def test_real_litellm_stream_assembly(tmp_path, unused_tcp_port):
    async def handler(request):
        body = await request.json()
        assert body["stream"] is True
        events = []
        for delta, finish in [
            (
                {"role": "assistant", "content": "hello ", "reasoning": "I should "},
                None,
            ),
            ({"content": "world", "reasoning": "reply briefly."}, "stop"),
        ]:
            chunk: dict[str, object] = {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if finish:
                chunk["usage"] = {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }
            events.append("data: " + json.dumps(chunk) + "\n\n")
        return web.Response(
            text="".join(events) + "data: [DONE]\n\n", content_type="text/event-stream"
        )

    async with endpoint(handler, unused_tcp_port) as url:
        config = resolve_agent_config(
            {"model": "openai/test-model", "api_base": url, "api_key": "dummy"},
            cwd=str(tmp_path),
        )
        reply = await LLMClient(config).complete(
            [{"role": "user", "content": "hi"}], max_tokens=64
        )
    assert reply.text == "hello world"
    assert reply.reasoning == "I should reply briefly."
    assert reply.finish_reason == "stop"


@pytest.mark.asyncio
async def test_auth_failure_is_not_retried(tmp_path, unused_tcp_port):
    requests = []

    async def handler(request):
        requests.append(await request.json())
        return web.json_response(
            {
                "error": {
                    "message": "invalid key",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
            status=401,
        )

    async with endpoint(handler, unused_tcp_port) as url:
        config = resolve_agent_config(
            {
                "model": "openai/test-model",
                "api_base": url,
                "api_key": "dummy",
                "stream": False,
            },
            cwd=str(tmp_path),
        )
        with pytest.raises(AgentError, match="AuthenticationError"):
            await LiteLLMSession(config).run(["hello"])
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_transport_precedence_and_isolation(monkeypatch):
    import litellm

    requests = []

    async def completion(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return litellm.ModelResponse(**response({"content": "ok"}))

    monkeypatch.setattr(litellm, "acompletion", completion)
    one = LLMClient(
        resolve_agent_config(
            {
                "model": "openai/test",
                "api_key": "first",
                "stream": False,
                "model_params": {"max_tokens": 100},
            }
        )
    )
    two = LLMClient(
        resolve_agent_config(
            {"model": "openai/test", "api_key": "second", "stream": False}
        )
    )
    await one.complete([{"role": "user", "content": "hi"}], max_tokens=17)
    await two.complete([{"role": "user", "content": "hi"}])
    assert requests[0]["api_key"] == "first" and requests[0]["max_tokens"] == 17
    assert requests[1]["api_key"] == "second" and requests[1]["max_tokens"] == 8192
    assert "api_base" not in requests[1]


@pytest.mark.asyncio
async def test_streaming_tool_argument_fragments(tmp_path, unused_tcp_port):
    async def handler(request):
        chunks = []
        pieces = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "out",',
                        },
                    }
                ],
            },
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '"content": "ok"}'}}
                ]
            },
            {},
        ]
        for i, delta in enumerate(pieces):
            chunk: dict[str, object] = {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": "tool_calls" if i == 2 else None,
                    }
                ],
            }
            chunks.append("data: " + json.dumps(chunk) + "\n\n")
        return web.Response(
            text="".join(chunks) + "data: [DONE]\n\n", content_type="text/event-stream"
        )

    async with endpoint(handler, unused_tcp_port) as url:
        config = resolve_agent_config(
            {"model": "openai/test-model", "api_base": url, "api_key": "dummy"}
        )
        reply = await LLMClient(config).complete(
            [{"role": "user", "content": "hi"}], max_tokens=64
        )
    assert reply.tool_calls[0]["function"]["name"] == "write_file"
    assert json.loads(reply.tool_calls[0]["function"]["arguments"]) == {
        "path": "out",
        "content": "ok",
    }
    assert reply.usage_incomplete


@pytest.mark.asyncio
async def test_transient_retry_and_stream_only_fallback(monkeypatch):
    import litellm

    seen = []

    async def completion(**kwargs):
        seen.append(kwargs)
        if len(seen) == 1:
            raise litellm.APIConnectionError(
                message="temporary", llm_provider="openai", model="test"
            )
        if kwargs["stream"]:
            raise litellm.UnsupportedParamsError(
                message="stream is unsupported", llm_provider="openai", model="test"
            )
        return litellm.ModelResponse(**response({"content": "ok"}))

    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(litellm, "acompletion", completion)
    monkeypatch.setattr("llm.auto_code_gen.agents.client.asyncio.sleep", no_sleep)
    config = resolve_agent_config({"model": "openai/test", "api_key": "dummy"})
    reply = await LLMClient(config).complete([{"role": "user", "content": "hi"}])
    assert reply.text == "ok" and reply.usage_incomplete
    assert [r["stream"] for r in seen] == [True, True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("billing", "replays"), [("self_hosted", True), ("provider", False)]
)
async def test_aborted_stream_replays_only_when_not_billed(
    monkeypatch, tmp_path, billing, replays
):
    import litellm

    attempts = []

    class Stream:
        def __init__(self, chunks=(), failure=None):
            self.items = list(chunks)
            self.failure = failure

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.items:
                return self.items.pop(0)
            if self.failure:
                error, self.failure = self.failure, None
                raise error
            raise StopAsyncIteration

    final = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="recovered", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )

    async def fake_acompletion(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return Stream(
                chunks=[
                    SimpleNamespace(
                        usage=None,
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="partial"),
                                finish_reason=None,
                            )
                        ],
                    )
                ],
                failure=litellm.ServiceUnavailableError(
                    message="repetition guard",
                    llm_provider="openai",
                    model="openai/test",
                ),
            )
        return Stream(chunks=[SimpleNamespace(usage=final.usage, choices=[])])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        litellm, "stream_chunk_builder", lambda chunks, messages=None: final
    )
    client = LLMClient(
        resolve_agent_config(
            {
                "model": "openai/test",
                "billing": billing,
                "stream": True,
                "model_params": {"max_tokens": 64},
            },
            cwd=str(tmp_path),
        )
    )

    if replays:
        reply = await client.complete([{"role": "user", "content": "go"}])
        assert reply.text == "recovered"
        assert reply.usage_incomplete
        assert len(attempts) == 2
    else:
        with pytest.raises(litellm.ServiceUnavailableError):
            await client.complete([{"role": "user", "content": "go"}])
        assert len(attempts) == 1
