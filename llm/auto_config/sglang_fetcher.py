"""Fetch and resolve SGLang serving recipes from the SGLang cookbook on GitHub."""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from urllib.error import HTTPError, URLError

from llm.auto_config.models import ResolvedRecipe
from llm.auto_config.vllm_fetcher import (
    HardwareNotSupportedError,
    RecipeFetchError,
    RecipeNotFoundError,
    _pairs_to_flags_str,
)

log = logging.getLogger(__name__)

SGLANG_RAW_BASE = (
    "https://raw.githubusercontent.com/sgl-project/sglang/main"
    "/docs/src/snippets/configs"
)

SGLANG_ADAPTER_HANDLED_FLAGS = {
    "--model-path",
    "--tp",
    "--host",
    "--port",
    "--trust-remote-code",
    "--enable-metrics",
    "--disable-radix-cache",
    "--dist-timeout",
}

SGLANG_SPEC_FLAGS = {
    "--speculative-algorithm",
    "--speculative-num-steps",
    "--speculative-eagle-topk",
    "--speculative-num-draft-tokens",
}

QUANT_SUFFIXES = ["-FP8", "-NVFP4", "-MXFP4", "-BF16", "-INT4", "-GPTQ", "-AWQ"]


def _fetch_raw(url: str) -> str:
    log.info("Fetching %s", url)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode()
    except HTTPError as e:
        raise RecipeFetchError(f"HTTP {e.code} fetching {url}") from e
    except URLError as e:
        raise RecipeFetchError(f"Failed to connect to {url}: {e.reason}") from e


# ── JSX parsing ──────────────────────────────────────────────────────────


def _extract_balanced(text: str, start: int, open_ch: str, close_ch: str) -> str:
    """Extract text between balanced brackets starting at start."""
    depth = 0
    in_string = None
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_string:
            i += 2
            continue
        if ch in ('"', "'", "`"):
            if in_string is None:
                in_string = ch
            elif in_string == ch:
                in_string = None
        elif in_string is None:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return ""


def _strip_js_comments(text: str) -> str:
    """Remove JS single-line comments, preserving strings."""
    lines = []
    for line in text.split("\n"):
        result = []
        in_string = None
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and in_string:
                result.append(line[i : i + 2])
                i += 2
                continue
            if ch in ('"', "'"):
                if in_string is None:
                    in_string = ch
                elif in_string == ch:
                    in_string = None
            elif (
                ch == "/"
                and i + 1 < len(line)
                and line[i + 1] == "/"
                and in_string is None
            ):
                break
            result.append(ch)
            i += 1
        lines.append("".join(result))
    return "\n".join(lines)


def _js_to_json(text: str) -> str:
    """Convert a JS object/array literal to valid JSON."""
    text = _strip_js_comments(text)
    text = re.sub(r"(?<=[{,\n])\s*(\w[\w]*)\s*:", r' "\1":', text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _extract_section(content: str, key: str, bracket: str) -> str:
    """Extract a top-level section value from the config JSX."""
    close = "}" if bracket == "{" else "]"
    pattern = rf"(?:^|\n)\s*{key}\s*:\s*{re.escape(bracket)}"
    m = re.search(pattern, content)
    if not m:
        return ""
    bracket_pos = m.end() - 1
    return _extract_balanced(content, bracket_pos, bracket, close)


def _parse_config(content: str) -> tuple[dict, dict, list]:
    """Parse modelNames, dockerImages, and cells from a SGLang config JSX."""
    content = _strip_js_comments(content)
    model_names_raw = _extract_section(content, "modelNames", "{")
    docker_images_raw = _extract_section(content, "dockerImages", "{")
    cells_raw = _extract_section(content, "cells", "[")

    if not cells_raw:
        raise RecipeFetchError("Could not parse cells from SGLang config")

    model_names = json.loads(_js_to_json(model_names_raw)) if model_names_raw else {}
    docker_images = (
        json.loads(_js_to_json(docker_images_raw)) if docker_images_raw else {}
    )
    cells = json.loads(_js_to_json(cells_raw)) if cells_raw else []

    return model_names, docker_images, cells


# ── Config file discovery ────────────────────────────────────────────────


def _model_to_slug(model: str) -> str:
    """Strip quant suffix from model name to get the base config slug."""
    _, name = model.split("/", 1)
    for suffix in QUANT_SUFFIXES:
        if name.upper().endswith(suffix.upper()):
            name = name[: len(name) - len(suffix)]
            break
    return name.lower()


def _find_config_via_llms_txt(slug: str) -> str | None:
    """Discover the config JSX URL by searching llms.txt for the model's cookbook page."""
    try:
        llms_txt = _fetch_raw("https://docs.sglang.io/llms.txt")
    except RecipeFetchError:
        return None

    for line in llms_txt.split("\n"):
        m = re.search(r"\((https://docs\.sglang\.io/cookbook/[^)]+\.md)\)", line)
        if not m:
            continue
        page_url = m.group(1)
        model_in_url = page_url.rsplit("/", 1)[-1].replace(".md", "").lower()
        if model_in_url != slug:
            continue

        mdx_path = page_url.replace("https://docs.sglang.io/", "docs/").replace(
            ".md", ".mdx"
        )
        mdx_url = (
            f"https://raw.githubusercontent.com/sgl-project/sglang/main/{mdx_path}"
        )
        try:
            mdx_content = _fetch_raw(mdx_url)
        except RecipeFetchError:
            continue

        m2 = re.search(r'import\s+\{[^}]*config[^}]*\}\s+from\s+"([^"]+)"', mdx_content)
        if m2:
            import_path = m2.group(1).lstrip("/")
            return (
                f"https://raw.githubusercontent.com/sgl-project/sglang/main"
                f"/docs/{import_path}"
            )

    return None


def _find_config(model: str) -> str:
    """Find and return the raw JSX content of the SGLang config for a model."""
    org, _ = model.split("/", 1)
    slug = _model_to_slug(model)

    # Try the model's own org first
    url = f"{SGLANG_RAW_BASE}/{org}/{slug}.jsx"
    try:
        content = _fetch_raw(url)
        if model in content:
            return content
    except RecipeFetchError:
        pass

    # Discover the config path via llms.txt → MDX import
    config_url = _find_config_via_llms_txt(slug)
    if config_url:
        try:
            content = _fetch_raw(config_url)
            if model in content:
                return content
        except RecipeFetchError:
            pass

    raise RecipeNotFoundError(
        f"No SGLang cookbook recipe found for {model!r}. "
        f"Browse available recipes at https://docs.sglang.io/cookbook/intro"
    )


# ── Flag parsing ─────────────────────────────────────────────────────────


def _parse_cell_flags(flags: list[str]) -> list[tuple[str, str | None]]:
    """Parse SGLang cell flags (each is a '--flag value' string)."""
    pairs = []
    for flag_str in flags:
        parts = flag_str.split(None, 1)
        flag = parts[0]
        value = parts[1] if len(parts) > 1 else None
        pairs.append((flag, value))
    return pairs


def _parse_env_list(env_list: list[str]) -> dict[str, str]:
    """Convert ['KEY=val', ...] to {KEY: val}."""
    env = {}
    for entry in env_list:
        key, _, value = entry.partition("=")
        if key:
            env[key] = value
    return env


# ── Public API ───────────────────────────────────────────────────────────


def fetch_sglang_recipe(
    model: str,
    hardware: str,
    strategy: str = "low-latency",
    spec_decoding: str = "none",
) -> ResolvedRecipe:
    """Fetch and resolve a SGLang recipe.

    Args:
        model: HuggingFace model ID (e.g. "nvidia/GLM-5.2-NVFP4").
        hardware: GPU target (e.g. "h200", "b200", "mi355x").
        strategy: Serving strategy ("low-latency", "balanced", "high-throughput").
        spec_decoding: "none" strips spec flags, "keep" preserves what the cell has.

    Returns:
        ResolvedRecipe ready for use in a profiling config.
    """
    content = _find_config(model)
    model_names, docker_images, cells = _parse_config(content)

    variant = None
    quant = None
    for key, mid in model_names.items():
        if mid == model:
            parts = key.split("|", 1)
            variant = parts[0]
            quant = parts[1] if len(parts) > 1 else "fp8"
            break

    if variant is None:
        available = list(model_names.values())
        raise RecipeNotFoundError(
            f"Model {model!r} not in SGLang config modelNames. "
            f"Available: {available}"
        )

    log.info("Resolved %s → variant=%s quant=%s", model, variant, quant)

    matching_cell = None
    for cell in cells:
        m = cell.get("match", {})
        if (
            m.get("hw") == hardware
            and m.get("variant") == variant
            and m.get("quant") == quant
            and m.get("strategy") == strategy
            and m.get("nodes") == "single"
        ):
            matching_cell = cell
            break

    if matching_cell is None:
        available_hw = sorted(
            {
                c["match"]["hw"]
                for c in cells
                if c["match"].get("variant") == variant
                and c["match"].get("quant") == quant
                and c["match"].get("nodes") == "single"
            }
        )
        available_strategies = sorted(
            {
                c["match"]["strategy"]
                for c in cells
                if c["match"].get("hw") == hardware
                and c["match"].get("variant") == variant
                and c["match"].get("quant") == quant
                and c["match"].get("nodes") == "single"
            }
        )
        if not available_hw:
            raise HardwareNotSupportedError(
                f"Hardware {hardware!r} not available for {model} (quant={quant}). "
                f"No single-node cells found for this variant/quant."
            )
        if not available_strategies:
            raise HardwareNotSupportedError(
                f"Strategy {strategy!r} not available for {model} on {hardware}. "
                f"Available strategies: {available_strategies}"
            )
        raise HardwareNotSupportedError(
            f"No cell for {model} on {hardware} with strategy={strategy!r}. "
            f"Available hardware: {available_hw}, strategies: {available_strategies}"
        )

    docker_image = docker_images.get(f"{hardware}|{quant}") or docker_images.get(
        hardware, "lmsysorg/sglang:latest"
    )

    pairs = _parse_cell_flags(matching_cell.get("flags", []))

    pairs = [(f, v) for f, v in pairs if f not in SGLANG_ADAPTER_HANDLED_FLAGS]

    if spec_decoding == "none":
        pairs = [(f, v) for f, v in pairs if f not in SGLANG_SPEC_FLAGS]

    extra_flags = _pairs_to_flags_str(pairs)
    extra_env = _parse_env_list(matching_cell.get("env", []))

    log.info(
        "Resolved SGLang recipe for %s on %s (%s): flags=%r env=%r docker=%s",
        model,
        hardware,
        strategy,
        extra_flags,
        extra_env,
        docker_image,
    )

    return ResolvedRecipe(
        model_id=model,
        extra_flags=extra_flags,
        extra_env=extra_env,
        docker_image=docker_image,
    )
