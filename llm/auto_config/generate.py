"""Generate profiling configs for llm.auto_profile_server_client."""

from __future__ import annotations

import json
import logging
import re
import shlex
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

from llm.auto_config.vllm_fetcher import (
    fetch_recipe,
    _parse_args_list,
    _pairs_to_flags_str,
)
from llm.auto_config.sglang_fetcher import fetch_sglang_recipe

log = logging.getLogger(__name__)

DOCKERHUB_TAGS_URL = "https://hub.docker.com/v2/repositories/{repo}/tags/"

LATEST_IMAGE_REPOS = {
    "vllm": "vllm/vllm-openai",
    "sgl": "lmsysorg/sglang",
}


def _fetch_latest_stable_tag(repo: str) -> str | None:
    """Query DockerHub for the latest stable versioned tag (vN.N.N)."""
    url = f"{DOCKERHUB_TAGS_URL.format(repo=repo)}?page_size=25&ordering=last_updated&name=v0"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError, json.JSONDecodeError):
        return None

    for tag in data.get("results", []):
        name = tag.get("name", "")
        if re.fullmatch(r"v\d+\.\d+\.\d+", name):
            return name
    return None


def _resolve_latest_image(fw: str, recipe_image: str) -> str:
    """Replace the recipe's pinned image tag with the latest stable release."""
    repo = LATEST_IMAGE_REPOS.get(fw)
    if not repo:
        return recipe_image

    tag = _fetch_latest_stable_tag(repo)
    if not tag:
        log.warning("Could not fetch latest tag for %s, using recipe image", repo)
        return recipe_image

    latest = f"{repo}:{tag}"
    if latest != recipe_image:
        log.info("Docker image for %s: %s → %s (latest)", fw, recipe_image, latest)
    return latest


def _merge_flags(recipe_flags: str, user_flags: str) -> str:
    """Merge recipe and user flags. User flags win on conflict."""
    if not user_flags:
        return recipe_flags
    if not recipe_flags:
        return user_flags

    recipe_pairs = _parse_args_list(shlex.split(recipe_flags))
    user_pairs = _parse_args_list(shlex.split(user_flags))

    merged = {}
    for flag, value in recipe_pairs:
        merged[flag] = value
    for flag, value in user_pairs:
        merged[flag] = value

    return _pairs_to_flags_str(list(merged.items()))


def _fetch_fw_recipe(fw, model, hardware, strategy, spec_decoding):
    """Fetch the recipe for a given framework."""
    if fw == "vllm":
        return fetch_recipe(model, hardware, spec_decoding)
    if fw == "sgl":
        return fetch_sglang_recipe(model, hardware, strategy, spec_decoding)
    return None


def generate_config(
    model: str,
    hardware: str,
    num_gpus: int,
    frameworks: list[str] | None = None,
    strategy: str = "low-latency",
    spec_decoding: str = "none",
    concurrencies: list[int] | None = None,
    exec_mode: str = "pure_decode",
    input_len: int = 4,
    output_len: int = 1024,
    enable_trace: bool = True,
    extra_flags: str = "",
    extra_env: dict[str, str] | None = None,
    output_path: str | None = None,
) -> dict:
    """Generate a profiling config by fetching recipes from framework cookbooks.

    Args:
        model: HuggingFace model ID (e.g. "nvidia/GLM-5.2-NVFP4").
        hardware: GPU target (e.g. "b200", "h200", "mi355x").
        num_gpus: Number of GPUs.
        frameworks: Frameworks to include (default: ["vllm"]).
        strategy: SGLang serving strategy ("low-latency", "balanced", "high-throughput").
        spec_decoding: Speculative decoding mode ("none", "mtp", "dspark", or "keep").
        concurrencies: Concurrency levels (default: [1]).
        exec_mode: Execution mode name (default: "pure_decode").
        input_len: Input token length for the exec mode.
        output_len: Output token length for the exec mode.
        enable_trace: Enable NSYS tracing (default: True).
        extra_flags: Additional server flags to append (override recipe on conflict).
        extra_env: Additional env vars to merge (override recipe on conflict).
        output_path: If set, write the config JSON to this path.

    Returns:
        Config dict matching auto_profile_server_client's RunConfig JSON schema.
    """
    if frameworks is None:
        frameworks = ["vllm"]
    if concurrencies is None:
        concurrencies = [1]
    if extra_env is None:
        extra_env = {}

    framework_entries = []
    docker_images = {}
    resolved_model_id = model

    for fw in frameworks:
        recipe = _fetch_fw_recipe(fw, model, hardware, strategy, spec_decoding)

        if recipe is not None:
            merged_flags = _merge_flags(recipe.extra_flags, extra_flags)
            merged_env = {**recipe.extra_env, **extra_env}
            resolved_model_id = recipe.model_id

            framework_entries.append({
                "name": fw,
                "enable_trace": enable_trace,
                "server_args": {
                    "extra_flags": merged_flags,
                    "extra_env": merged_env,
                },
                "client_args": {"extra_flags": ""},
            })
            docker_images[fw] = _resolve_latest_image(fw, recipe.docker_image)
        else:
            framework_entries.append({
                "name": fw,
                "enable_trace": enable_trace,
                "server_args": {"extra_flags": "", "extra_env": {}},
                "client_args": {"extra_flags": ""},
            })
            docker_images.setdefault(fw, f"<SET_{fw.upper()}_DOCKER_IMAGE>")

    config = {
        "profiles": [{
            "model": resolved_model_id,
            "num_gpus": num_gpus,
            "exec_mode": exec_mode,
            "frameworks": framework_entries,
        }],
        "concurrencies": concurrencies,
        "docker_images": docker_images,
        "exec_modes": {
            exec_mode: {"input_len": input_len, "output_len": output_len},
        },
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        log.info("Wrote config to %s", path)

    return config
