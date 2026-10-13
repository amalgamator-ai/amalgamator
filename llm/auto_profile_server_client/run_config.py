# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate JSON run configs for the server-client benchmark pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from llm.auto_profile_server_client.config import (
    DEFAULT_MAX_STABILITY_RETRIES,
    DEFAULT_METRICS_POLL_INTERVAL_SEC,
    DEFAULT_STABILITY_THRESHOLD_PCT,
    PROFILING_BUFFER,
    VALID_FRAMEWORKS,
)
from llm.auto_profile_server_client.models import (
    ClientArgs,
    ExecMode,
    FrameworkEntry,
    Profile,
    RunConfig,
    ServerArgs,
)

log = logging.getLogger(__name__)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _parse_server_args(raw: dict | None) -> ServerArgs:
    if raw is None:
        return ServerArgs()
    return ServerArgs(
        extra_flags=raw.get("extra_flags", ""),
        extra_env=raw.get("extra_env", {}),
        prepare_cmds=raw.get("prepare_cmds", []),
    )


def _parse_client_args(raw: dict | None) -> ClientArgs:
    if raw is None:
        return ClientArgs()
    return ClientArgs(extra_flags=raw.get("extra_flags", ""))


def _parse_framework_entry(raw: dict, idx: int) -> FrameworkEntry:
    name = raw.get("name")
    _require(
        isinstance(name, str) and name in VALID_FRAMEWORKS,
        f"profiles[{idx}].frameworks: invalid name {name!r}. "
        f"Must be one of {list(VALID_FRAMEWORKS)}",
    )
    assert isinstance(name, str)
    profiler_type = raw.get("profiler_type", "nsys")
    _require(
        profiler_type in ("nsys", "torch"),
        f"profiles[{idx}].frameworks: profiler_type must be 'nsys' or 'torch', "
        f"got {profiler_type!r}",
    )
    return FrameworkEntry(
        name=name,
        enable_trace=raw.get("enable_trace", False),
        profiler_type=profiler_type,
        server_args=_parse_server_args(raw.get("server_args")),
        client_args=_parse_client_args(raw.get("client_args")),
    )


def _resolve_exec_mode(
    profile_raw: dict, exec_modes: dict[str, ExecMode], idx: int
) -> tuple[str, int, int, int]:
    if "exec_mode" in profile_raw:
        mode_name = profile_raw["exec_mode"]
        _require(
            mode_name in exec_modes,
            f"profiles[{idx}].exec_mode: {mode_name!r} not found. "
            f"Available: {list(exec_modes)}",
        )
        em = exec_modes[mode_name]
        input_len = em.input_len
        if em.kv_cache_fill > 0:
            output_len = em.kv_cache_fill + PROFILING_BUFFER
            kv_cache_fill = em.kv_cache_fill
        else:
            output_len = em.output_len
            kv_cache_fill = 0
        return mode_name, input_len, output_len, kv_cache_fill

    input_len = profile_raw.get("input_len")
    _require(
        isinstance(input_len, int) and input_len > 0,
        f"profiles[{idx}].input_len must be a positive integer",
    )
    assert isinstance(input_len, int)

    if "kv_cache_fill" in profile_raw:
        kv_cache_fill = profile_raw["kv_cache_fill"]
        _require(
            isinstance(kv_cache_fill, int) and kv_cache_fill > input_len,
            f"profiles[{idx}].kv_cache_fill must be > input_len",
        )
        assert isinstance(input_len, int)
        assert isinstance(kv_cache_fill, int)
        output_len = kv_cache_fill + PROFILING_BUFFER
        return "", input_len, output_len, kv_cache_fill

    output_len = profile_raw.get("output_len")
    _require(
        isinstance(output_len, int) and output_len > 0,
        f"profiles[{idx}].output_len must be a positive integer",
    )
    assert isinstance(input_len, int)
    assert isinstance(output_len, int)
    return "", input_len, output_len, 0


def _parse_exec_modes(raw: dict | None) -> dict[str, ExecMode]:
    if raw is None:
        return {}
    modes = {}
    for name, cfg in raw.items():
        _require(isinstance(cfg, dict), f"exec_modes.{name} must be an object")
        input_len = cfg.get("input_len", 0)
        _require(
            isinstance(input_len, int) and input_len > 0,
            f"exec_modes.{name}.input_len must be a positive integer",
        )
        output_len = cfg.get("output_len", 0)
        kv_cache_fill = cfg.get("kv_cache_fill", 0)
        modes[name] = ExecMode(
            input_len=input_len,
            output_len=output_len,
            kv_cache_fill=kv_cache_fill,
        )
    return modes


def _parse_profile(raw: dict, exec_modes: dict[str, ExecMode], idx: int) -> Profile:
    model = raw.get("model")
    _require(
        isinstance(model, str) and len(model) > 0,
        f"profiles[{idx}].model must be a non-empty string",
    )
    assert isinstance(model, str)

    num_gpus = raw.get("num_gpus")
    _require(
        isinstance(num_gpus, int) and num_gpus > 0,
        f"profiles[{idx}].num_gpus must be a positive integer",
    )
    assert isinstance(num_gpus, int)

    mode_name, input_len, output_len, kv_cache_fill = _resolve_exec_mode(
        raw, exec_modes, idx
    )

    frameworks_raw = raw.get("frameworks", [])
    _require(
        isinstance(frameworks_raw, list) and len(frameworks_raw) > 0,
        f"profiles[{idx}].frameworks must be a non-empty list",
    )

    seen = set()
    frameworks = []
    for fw_raw in frameworks_raw:
        entry = _parse_framework_entry(fw_raw, idx)
        _require(
            entry.name not in seen,
            f"profiles[{idx}].frameworks has duplicate framework {entry.name!r}",
        )
        seen.add(entry.name)
        frameworks.append(entry)

    return Profile(
        model=model,
        num_gpus=num_gpus,
        exec_mode_name=mode_name,
        input_len=input_len,
        output_len=output_len,
        kv_cache_fill=kv_cache_fill,
        frameworks=frameworks,
    )


def load_config(config_path: str | Path) -> RunConfig:
    """Load and validate a run config JSON file."""
    config_path = Path(config_path)
    _require(config_path.is_file(), f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = json.load(f)

    _require(isinstance(raw, dict), "Config must be a JSON object")

    exec_modes = _parse_exec_modes(raw.get("exec_modes"))

    profiles_raw = raw.get("profiles", [])
    _require(
        isinstance(profiles_raw, list) and len(profiles_raw) > 0,
        '"profiles" must be a non-empty list',
    )

    profiles = [_parse_profile(p, exec_modes, i) for i, p in enumerate(profiles_raw)]

    concurrencies = raw.get("concurrencies", [])
    _require(
        isinstance(concurrencies, list)
        and len(concurrencies) > 0
        and all(isinstance(c, int) and c > 0 for c in concurrencies),
        '"concurrencies" must be a non-empty list of positive integers',
    )

    docker_images = raw.get("docker_images", {})
    _require(isinstance(docker_images, dict), '"docker_images" must be an object')

    needed_frameworks = {fw.name for p in profiles for fw in p.frameworks}
    missing = needed_frameworks - set(docker_images)
    _require(len(missing) == 0, f"docker_images missing entries for: {missing}")

    stability_threshold = raw.get(
        "stability_threshold_pct", DEFAULT_STABILITY_THRESHOLD_PCT
    )
    max_retries = raw.get("max_stability_retries", DEFAULT_MAX_STABILITY_RETRIES)
    metrics_interval = raw.get(
        "metrics_poll_interval_sec", DEFAULT_METRICS_POLL_INTERVAL_SEC
    )
    output_dir = raw.get("output_dir", "")

    config = RunConfig(
        profiles=profiles,
        concurrencies=concurrencies,
        docker_images=docker_images,
        exec_modes=exec_modes,
        stability_threshold_pct=stability_threshold,
        max_stability_retries=max_retries,
        metrics_poll_interval_sec=metrics_interval,
        output_dir=output_dir,
    )

    log.info(
        "Loaded config from %s: %d profiles, concurrencies=%s",
        config_path,
        len(profiles),
        concurrencies,
    )
    return config
