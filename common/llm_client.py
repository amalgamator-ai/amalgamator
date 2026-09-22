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
from dataclasses import dataclass
from typing import TYPE_CHECKING

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


# ── Model classification ────────────────────────────────────────────────────

_CLAUDE_PREFIXES = (
    "claude-",
    "anthropic/",
    "vertex_ai/claude-",
)


def is_claude_model(model: str) -> bool:
    """Return True if *model* refers to an Anthropic Claude model.

    Used to decide between the native Claude Agent SDK path and the
    generic LiteLLM agent loop.
    """
    return any(model.startswith(p) for p in _CLAUDE_PREFIXES)


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

    effective_model = _resolve_model(model)

    if profile:
        base_params = dict(profile.params)
        p_api_base = profile.api_base
        p_api_key = profile.api_key
    else:
        base_params = dict(_model_params)
        p_api_base = get_api_base()
        p_api_key = get_api_key()

    merged = {**base_params, **kwargs}
    merged.setdefault("max_tokens", max_tokens)

    log.debug("LLM completion: model=%s (effective=%s), max_tokens=%s, profile=%s",
              model, effective_model, merged.get("max_tokens"),
              profile.id if profile else "legacy")

    if _is_ollama_model(effective_model):
        merged.setdefault("extra_body", {})
        merged["extra_body"].setdefault("think", False)

    if p_api_base and effective_model.startswith("openai/"):
        merged.setdefault("api_base", p_api_base)
        merged.setdefault("api_key", p_api_key or "dummy")

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


def _resolve_model(model: str) -> str:
    """Translate legacy env-var conventions into a LiteLLM model string.

    If ``CLAUDE_CODE_USE_VERTEX`` is set and the caller passes a bare
    Claude model name (e.g. ``claude-opus-4-6``), prepend
    ``vertex_ai/`` so LiteLLM routes through Vertex.
    """
    if os.environ.get("CLAUDE_CODE_USE_VERTEX") and model.startswith("claude-"):
        return f"vertex_ai/{model}"
    return model


def _is_ollama_model(model: str) -> bool:
    return model.startswith("ollama/") or model.startswith("ollama_chat/")


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
