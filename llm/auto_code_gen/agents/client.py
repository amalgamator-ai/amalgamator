# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM transport adapted from amalgamator c0b487d/de20339/3fc802e.

Configuration is per client; native Claude SDK behavior stays in its adapter.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import os
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    usage_incomplete: bool = False
    reasoning: str = ""


def _message_reasoning(message):
    """Return provider reasoning without depending on one API field name."""
    for name in ("reasoning", "reasoning_content"):
        value = getattr(message, name, None)
        if value:
            return str(value)
    for container_name in ("provider_specific_fields", "additional_kwargs"):
        container = getattr(message, container_name, None)
        if not isinstance(container, dict):
            continue
        for name in ("reasoning", "reasoning_content"):
            if container.get(name):
                return str(container[name])
    return ""


def _merge_params(base, override):
    """Recursively merge per-request provider parameters into a copy."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_params(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class LLMClient:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.model = config.model
        self._info_loaded = False
        self._info: Any = {}
        if self.model.startswith("claude-") and os.environ.get(
            "CLAUDE_CODE_USE_VERTEX"
        ):
            self.model = "vertex_ai/" + self.model

    def _model_info(self):
        if self._info_loaded:
            return self._info
        try:
            import litellm

            self._info = litellm.get_model_info(self.model) or {}
        except Exception:
            self._info = {}
        self._info_loaded = True
        return self._info

    def context_length(self):
        if self.config.context_length:
            return self.config.context_length
        info = self._model_info()
        return info.get("max_input_tokens") or info.get("max_tokens") or 32768

    def output_limit(self):
        return self._model_info().get("max_output_tokens")

    def count_tokens(self, messages, tools=None):
        import litellm

        try:
            if not self._model_info():
                # LiteLLM may silently use an unrelated tokenizer for unknown
                # OpenAI-compatible models. Prefer our conservative estimate.
                raise ValueError("Unknown model tokenizer")
            return litellm.token_counter(
                model=self.model, messages=messages, tools=tools or []
            )
        except Exception:
            import json

            # Conservative fallback for code and unknown tokenizers.
            return (
                len(
                    json.dumps([messages, tools or []], ensure_ascii=False).encode(
                        "utf-8"
                    )
                )
                // 2
                + 1
            )

    async def complete(
        self,
        messages,
        tools=None,
        *,
        max_tokens=None,
        request_model_params=None,
        tool_choice=None,
    ):
        import litellm

        params = _merge_params(self.config.model_params, request_model_params)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.setdefault("max_tokens", 8192)
        output_limit = self.output_limit()
        if output_limit and params["max_tokens"] > output_limit:
            raise ValueError(
                "Requested max_tokens exceeds the known model output limit"
            )
        if self.config.api_base:
            params["api_base"] = self.config.api_base.rstrip("/")
        if self.config.api_key is not None:
            params["api_key"] = self.config.api_key
        if self.model.startswith(("ollama/", "ollama_chat/")):
            params.setdefault("extra_body", {}).setdefault("think", False)
        request = dict(
            model=self.model,
            messages=messages,
            timeout=self.config.request_timeout,
            num_retries=0,
            **params,
        )
        if tools:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        streaming = self.config.stream
        usage_incomplete = False
        response = None
        for attempt in range(self.config.max_retries + 1):
            chunks = []
            try:
                try:
                    response = await litellm.acompletion(**request, stream=streaming)
                except litellm.UnsupportedParamsError as exc:
                    if not streaming or "stream" not in str(exc).lower():
                        raise
                    streaming = False
                    response = await litellm.acompletion(**request, stream=False)
                if streaming:
                    try:
                        async for chunk in cast(Any, response):
                            chunks.append(chunk)
                    finally:
                        close = getattr(response, "aclose", None)
                        if close:
                            await close()
                    response = cast(
                        Any, litellm.stream_chunk_builder(chunks, messages=messages)
                    )
                    if not any(getattr(chunk, "usage", None) for chunk in chunks):
                        usage_incomplete = True
                break
            except (
                litellm.RateLimitError,
                litellm.Timeout,
                litellm.APIConnectionError,
                litellm.ServiceUnavailableError,
            ) as exc:
                # A partially consumed response has uncertain billing; replaying
                # it is only safe when the provider is not billed per token. A
                # self-hosted server aborts mid-stream on its own repetition
                # guard, and resampling the same prompt is the recovery.
                if chunks and self.config.billing != "self_hosted":
                    raise
                if attempt == self.config.max_retries:
                    raise
                if chunks or not isinstance(exc, litellm.RateLimitError):
                    usage_incomplete = True
                await asyncio.sleep(min(2**attempt, 8))
        response = cast(Any, response)
        if not response or not response.choices:
            raise RuntimeError("Provider returned no choices")
        choice = response.choices[0]
        msg = choice.message
        usage = response.usage
        cost = None
        if self.config.billing == "self_hosted":
            cost = 0.0
        else:
            with contextlib.suppress(Exception):
                cost = litellm.completion_cost(
                    completion_response=response, model=self.model
                )
        return Reply(
            text=msg.content or "",
            reasoning=_message_reasoning(msg),
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
            finish_reason=choice.finish_reason or "stop",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
            usage_incomplete=usage_incomplete or usage is None,
        )
