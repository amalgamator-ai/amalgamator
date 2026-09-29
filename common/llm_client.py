"""Model-agnostic LLM completion layer.

Wraps LiteLLM to provide a single ``completion()`` call that works with any
provider: Anthropic (direct or Vertex), OpenAI, Ollama, vLLM, Together AI, etc.

Model names follow the LiteLLM convention::

    claude-opus-4-6              Anthropic (auto-routes via Vertex when
                                 CLAUDE_CODE_USE_VERTEX is set)
    vertex_ai/claude-opus-4-6    Anthropic via Google Vertex (explicit)
    ollama/qwen3:14b             Local Ollama
    openai/<model-name>          OpenAI-compatible endpoint (vLLM, TGI, etc.)
                                 Requires api_base to be set via
                                 ``configure(api_base=...)`` or the
                                 OPENAI_API_BASE environment variable.
    gpt-4o                       OpenAI (official)
    together_ai/meta-llama/...   Together AI

The module also exposes ``is_claude_model()`` so callers can decide whether
to use the Claude Agent SDK (for autonomous tool-use) or the generic LiteLLM
agent loop.
"""
from __future__ import annotations

import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Compatibility exports for existing callers.
from common.model_identity import is_claude_model, is_ollama_model as _is_ollama_model

if TYPE_CHECKING:
    from common.model_registry import ModelProfile

log = logging.getLogger(__name__)

# ── Module-level configuration ───────────────────────────────────────────────

_api_base: str | None = None
_api_key: str | None = None
_model_params: dict = {}


def configure(
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    model_params: dict | None = None,
) -> None:
    """Set module-level defaults (typically called once from config_loader).

    *api_base* is forwarded to LiteLLM for ``openai/`` prefixed models so
    users can point at a self-hosted vLLM / TGI / Ollama server without
    needing environment variables.

    *api_key* overrides the environment-variable lookup for API keys.
    Useful when the key is read from a file via ``api_key_file``
    in the YAML config.

    *model_params* is a dict of extra parameters (``temperature``,
    ``max_tokens``, ``top_p``, ``reasoning_effort``, …) merged into every
    LiteLLM call.  Explicit kwargs passed to ``completion()`` take
    precedence over these defaults.
    """
    global _api_base, _api_key, _model_params
    if api_base:
        _api_base = api_base.rstrip("/")
    if api_key is not None:
        _api_key = api_key
    if model_params is not None:
        _model_params = dict(model_params)


def get_api_base() -> str | None:
    """Return the effective API base URL (explicit config > env var)."""
    return _api_base or os.environ.get("OPENAI_API_BASE")


def get_api_key() -> str | None:
    """Return the effective API key (explicit config > env var)."""
    return _api_key or os.environ.get("OPENAI_API_KEY")


def get_model_params() -> dict:
    """Return the configured model parameters dict (may be empty)."""
    return dict(_model_params)


@dataclass
class CompletionResult:
    """Normalized result of a single LLM completion call."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


def is_self_hosted(model: str, profile: "ModelProfile | None" = None) -> bool:
    """Return True when the model is running on self-hosted infrastructure.

    Self-hosted means no per-token cloud cost: local Ollama, a custom
    vLLM/TGI server reached via ``api_base``, etc.

    When *profile* is provided, uses its api_base; otherwise falls back
    to the global/env api_base.
    """
    if profile:
        return profile.is_self_hosted()
    if _is_ollama_model(model):
        return True
    if model.startswith("openai/") and get_api_base():
        return True
    return False


def build_request_kwargs(
    model: str,
    *,
    profile: "ModelProfile | None" = None,
    default_max_tokens: int,
    overrides: dict | None = None,
) -> dict:
    """Build isolated per-call options; explicit transport is provider-independent."""
    if profile is not None:
        merged = deepcopy(profile.params)
        api_base, api_key = profile.api_base, profile.api_key
    else:
        merged = deepcopy(_model_params)
        api_base, api_key = _api_base, _api_key
        if model.startswith("openai/"):
            api_base, api_key = get_api_base(), get_api_key()
    if api_base:
        merged["api_base"] = api_base
    if api_key:
        merged["api_key"] = api_key
    merged.update(deepcopy(overrides or {}))
    if merged.get("api_base") and model.startswith("openai/") and not merged.get("api_key"):
        merged["api_key"] = os.environ.get("OPENAI_API_KEY") or "dummy"
    if not any(key in merged for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")):
        merged["max_tokens"] = default_max_tokens
    if _is_ollama_model(model):
        merged.setdefault("extra_body", {})
        merged["extra_body"].setdefault("think", False)
    return merged


def completion(
    model: str,
    messages: list[dict],
    max_tokens: int = 4096,
    *,
    profile: "ModelProfile | None" = None,
    **kwargs,
) -> CompletionResult:
    """Model-agnostic completion via LiteLLM.

    When *profile* is provided, transport config (api_base, api_key, params)
    comes from it — no global state is consulted.  When *profile* is None,
    falls back to the module-level globals / env vars for backward
    compatibility with callers that haven't adopted profiles yet.

    Global ``model_params`` (set via :func:`configure`) are merged as
    defaults — explicit *kwargs* take precedence.  The caller's
    *max_tokens* is used only when neither *kwargs* nor *model_params*
    supply it, avoiding a ``TypeError`` on duplicate keyword arguments.
    """
    import litellm

    effective_model = _resolve_model(profile.model if profile else model, profile=profile)
    merged = build_request_kwargs(
        effective_model, profile=profile, default_max_tokens=max_tokens, overrides=kwargs,
    )

    log.debug("LLM completion: model=%s (effective=%s), max_tokens=%s, profile=%s",
              model, effective_model, merged.get("max_tokens"),
              profile.id if profile else "legacy")

    response = litellm.completion(
        model=effective_model,
        messages=messages,
        **merged,
    )

    choice = response.choices[0]
    usage = response.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
    raw_text = choice.message.content or ""

    return CompletionResult(
        text=_strip_thinking(raw_text),
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=getattr(usage, "completion_tokens", 0),
        model=effective_model,
    )


def _resolve_model(model: str, profile: "ModelProfile | None" = None) -> str:
    """Translate legacy env-var conventions into a LiteLLM model string.

    If ``CLAUDE_CODE_USE_VERTEX`` is set and the caller passes a bare
    Claude model name (e.g. ``claude-opus-4-6``), prepend
    ``vertex_ai/`` so LiteLLM routes through Vertex.
    """
    explicit_transport = profile is not None and (profile.api_key or profile.api_base)
    if (not explicit_transport and os.environ.get("CLAUDE_CODE_USE_VERTEX", "").lower() in ("1", "true")
            and model.startswith("claude-")):
        return f"vertex_ai/{model}"
    return model


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Strip ``<think>…</think>`` blocks emitted by models with native
    thinking support (Qwen 3, DeepSeek-R1, etc.).

    If the entire response is inside a think block with no visible answer,
    returns the thinking content as a fallback.
    """
    stripped = _THINK_RE.sub("", text).strip()
    if stripped:
        return stripped
    return text.strip()
