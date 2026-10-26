"""Data models for configs, results, and metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ── Config models ────────────────────────────────────────────────────────────


@dataclass
class ServerArgs:
    extra_flags: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)
    prepare_cmds: list[str] = field(default_factory=list)


@dataclass
class ClientArgs:
    extra_flags: str = ""


@dataclass
class FrameworkEntry:
    name: str
    enable_trace: bool = False
    profiler_type: str = "nsys"
    server_args: ServerArgs = field(default_factory=ServerArgs)
    client_args: ClientArgs = field(default_factory=ClientArgs)


@dataclass
class ExecMode:
    input_len: int
    output_len: int = 0
    kv_cache_fill: int = 0


@dataclass
class Profile:
    model: str
    num_gpus: int
    exec_mode_name: str
    input_len: int
    output_len: int
    kv_cache_fill: int
    frameworks: list[FrameworkEntry]


@dataclass
class RunConfig:
    profiles: list[Profile]
    concurrencies: list[int]
    docker_images: dict[str, str]
    exec_modes: dict[str, ExecMode]
    stability_threshold_pct: float = 5.0
    max_stability_retries: int = 3
    metrics_poll_interval_sec: float = 1.0
    num_prompts_multiplier: int = 4
    output_dir: str = ""


# ── Result models ────────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    throughput_tok_per_sec: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    tpot_p90_ms: float = 0.0
    tpot_p99_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    num_prompts: int = 0
    total_time_sec: float = 0.0
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricsSample:
    timestamp: float
    generation_tokens_total: float = 0.0
    requests_running: float = 0.0
    kv_cache_usage_pct: float = 0.0
    queue_size: float = 0.0


@dataclass
class StabilityResult:
    stable: bool
    tpot_diff_pct: float
    ttft_diff_pct: float
    result_a: BenchmarkResult
    result_b: BenchmarkResult
    message: str = ""


@dataclass
class ProfileWindow:
    """Defines when/where to capture the nsys profile."""
    decode_start_sec: float = 0.0
    decode_end_sec: float = 0.0
    midpoint_sec: float = 0.0
    profile_start_sec: float = 0.0
    profile_duration_sec: float = 0.0
    start_iter: int = 0
    end_iter: int = 0
    method: str = "time"


@dataclass
class StepSummary:
    step_name: str
    status: str
    details: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    duration_sec: float = 0.0


# ── Test naming helpers ──────────────────────────────────────────────────────


def make_test_name(framework: str, model: str, num_gpus: int, concurrency: int,
                   input_len: int, output_len: int, mode: str = "") -> str:
    model_slug = model.replace("/", "__")
    name = f"{framework}-{model_slug}-tp_{num_gpus}-isl_{input_len}-osl_{output_len}-b_{concurrency}"
    if mode:
        name += f"-mode_{mode}"
    return name


def make_test_id(model: str, num_gpus: int, input_len: int, output_len: int,
                 mode: str = "") -> str:
    model_slug = model.replace("/", "__")
    tid = f"{model_slug}-tp_{num_gpus}-isl_{input_len}-osl_{output_len}"
    if mode:
        tid += f"-mode_{mode}"
    return tid


def make_test_id_with_batch(model: str, num_gpus: int, concurrency: int,
                            input_len: int, output_len: int,
                            mode: str = "") -> str:
    return make_test_id(model, num_gpus, input_len, output_len, mode) + f"-b_{concurrency}"


def save_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str)
