"""Fetch and resolve vLLM serving recipes from recipes.vllm.ai."""

from __future__ import annotations

import json
import logging
import shlex
import urllib.request
from urllib.error import HTTPError, URLError

from llm.auto_config.models import ResolvedRecipe

log = logging.getLogger(__name__)

RECIPES_BASE_URL = "https://recipes.vllm.ai"

ADAPTER_HANDLED_FLAGS = {
    "--tensor-parallel-size",
    "--port",
    "--max-model-len",
    "--no-enable-prefix-caching",
    "--trust-remote-code",
}

DISABLED_FEATURES = {"tool_calling", "reasoning"}


class RecipeFetchError(Exception):
    """Failed to fetch from recipes.vllm.ai."""


class RecipeNotFoundError(RecipeFetchError):
    """Model not found in recipes.vllm.ai."""


class HardwareNotSupportedError(RecipeFetchError):
    """Hardware target not available for this model/variant."""


def _fetch_json(path: str) -> dict:
    url = f"{RECIPES_BASE_URL}{path}"
    log.info("Fetching %s", url)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        raise RecipeFetchError(f"HTTP {e.code} fetching {url}") from e
    except URLError as e:
        raise RecipeFetchError(f"Failed to connect to {url}: {e.reason}") from e


def _parse_args_list(args: list[str]) -> list[tuple[str, str | None]]:
    """Parse a flat list of args into (flag, value) pairs."""
    pairs = []
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            i += 1
            continue
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            pairs.append((token, args[i + 1]))
            i += 2
        else:
            pairs.append((token, None))
            i += 1
    return pairs


def _parse_argv(argv: list[str]) -> list[tuple[str, str | None]]:
    """Parse a vllm serve argv, skipping the 'vllm serve <model>' prefix."""
    i = 0
    while i < len(argv) and not argv[i].startswith("--"):
        i += 1
    return _parse_args_list(argv[i:])


def _extract_model_id(argv: list[str]) -> str | None:
    """Extract the model ID from argv (element after 'serve')."""
    for i, token in enumerate(argv):
        if token == "serve" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _build_feature_strip_set(features: dict) -> set[str]:
    """Collect flag names from disabled features for stripping."""
    strip = set()
    for feature_name in DISABLED_FEATURES:
        feature = features.get(feature_name)
        if feature and "args" in feature:
            for arg in feature["args"]:
                if arg.startswith("--"):
                    strip.add(arg)
    return strip


def _pairs_to_flags_str(pairs: list[tuple[str, str | None]]) -> str:
    """Convert (flag, value) pairs to a shell-safe flags string."""
    parts = []
    for flag, value in pairs:
        if value is None:
            parts.append(flag)
        elif any(c in value for c in " {}[]\"'"):
            parts.append(f"{flag} {shlex.quote(value)}")
        else:
            parts.append(f"{flag} {value}")
    return " ".join(parts)


def fetch_recipe(
    model: str,
    hardware: str,
    spec_decoding: str = "none",
) -> ResolvedRecipe:
    """Fetch and resolve a vLLM recipe into extra_flags, extra_env, and docker_image.

    Args:
        model: HuggingFace model ID (e.g. "nvidia/GLM-5.2-NVFP4").
        hardware: GPU target (e.g. "b200", "h200", "mi355x").
        spec_decoding: Speculative decoding mode ("none", "mtp", "dspark").

    Returns:
        ResolvedRecipe ready for use in a profiling config.
    """
    # Step 1: find model in /models.json
    models_list = _fetch_json("/models.json")
    model_entry = None
    for entry in models_list:
        if entry.get("hf_id") == model:
            model_entry = entry
            break

    if model_entry is None:
        raise RecipeNotFoundError(
            f"Model {model!r} not found on recipes.vllm.ai. "
            f"Browse available models at https://recipes.vllm.ai/browse"
        )

    json_path = model_entry["json"]
    log.info("Resolved %s → %s", model, json_path)

    # Step 2: fetch the recipe JSON
    recipe = _fetch_json(json_path)

    # Step 3: find and fetch the hardware-specific endpoint
    by_hardware = recipe.get("recommended_command", {}).get("by_hardware", {})
    if hardware not in by_hardware:
        available_hw = sorted(by_hardware.keys())
        raise HardwareNotSupportedError(
            f"Hardware {hardware!r} not available for {model}. "
            f"Supported: {available_hw}"
        )

    hw_data = _fetch_json(by_hardware[hardware])

    argv = hw_data.get("argv", [])
    env = hw_data.get("env", {})
    docker_image = hw_data.get("docker_image", "")

    model_id = _extract_model_id(argv) or model

    # Step 4: parse argv into flag pairs
    pairs = _parse_argv(argv)

    # Step 5: strip adapter-handled flags
    pairs = [(f, v) for f, v in pairs if f not in ADAPTER_HANDLED_FLAGS]

    # Step 6: strip disabled feature flags
    features = recipe.get("features", {})
    feature_strip = _build_feature_strip_set(features)
    pairs = [(f, v) for f, v in pairs if f not in feature_strip]

    # Step 7: add speculative decoding args if requested
    if spec_decoding != "none":
        spec_feature = features.get("spec_decoding", {})
        modes = spec_feature.get("modes", {})
        if spec_decoding not in modes:
            available_modes = sorted(modes.keys())
            raise ValueError(
                f"Speculative decoding mode {spec_decoding!r} not available for {model}. "
                f"Available: {available_modes}"
            )
        spec_args = modes[spec_decoding].get("args", [])
        pairs.extend(_parse_args_list(spec_args))

    extra_flags = _pairs_to_flags_str(pairs)

    log.info(
        "Resolved recipe for %s on %s: flags=%r env=%r docker=%s",
        model_id,
        hardware,
        extra_flags,
        env,
        docker_image,
    )

    return ResolvedRecipe(
        model_id=model_id,
        extra_flags=extra_flags,
        extra_env=env,
        docker_image=docker_image,
    )
