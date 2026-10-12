"""SGLang framework adapter."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from llm.auto_profile_server_client.frameworks.base import FrameworkAdapter
from llm.auto_profile_server_client.models import BenchmarkResult

log = logging.getLogger(__name__)


class SglangAdapter(FrameworkAdapter):
    name = "sgl"
    default_port = 30000

    def build_server_cmd(
        self,
        model: str,
        tp: int,
        port: int,
        extra_flags: str = "",
        extra_env: dict[str, str] | None = None,
    ) -> str:
        env_prefix = ""
        if extra_env:
            env_prefix = " ".join(f"{k}={v}" for k, v in extra_env.items()) + " "

        return (
            f"{env_prefix}"
            f"python -m sglang.launch_server "
            f"--model-path {model} "
            f"--host 0.0.0.0 "
            f"--port {port} "
            f"--tp {tp} "
            f"--trust-remote-code "
            f"--enable-metrics "
            f"--disable-radix-cache "
            f"--dist-timeout 1800 "
            f"{extra_flags}"
        ).strip()

    def build_client_cmd(
        self,
        model: str,
        port: int,
        num_prompts: int,
        input_len: int,
        output_len: int,
        concurrency: int,
        output_path: str,
        extra_flags: str = "",
    ) -> str:
        return (
            f"python -m sglang.bench_serving "
            f"--backend sglang "
            f"--port {port} "
            f"--model {model} "
            f"--dataset-name random "
            f"--random-input-len {input_len} "
            f"--random-output-len {output_len} "
            f"--num-prompts {num_prompts} "
            f"--max-concurrency {concurrency} "
            f"--warmup-requests 0 "
            f"--output-file {output_path} "
            f"{extra_flags}"
        ).strip()

    def parse_benchmark_results(self, json_path: str, log_path: str) -> BenchmarkResult:
        result = BenchmarkResult()

        if Path(json_path).is_file():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                result.raw = data
                result.ttft_ms = data.get("mean_ttft_ms", 0)
                result.tpot_ms = data.get("mean_tpot_ms", 0)
                result.throughput_tok_per_sec = data.get("output_throughput", 0)
                result.ttft_p50_ms = data.get("p50_ttft_ms", 0)
                result.ttft_p90_ms = data.get("p90_ttft_ms", 0)
                result.ttft_p99_ms = data.get("p99_ttft_ms", 0)
                result.tpot_p50_ms = data.get("p50_tpot_ms", 0)
                result.tpot_p90_ms = data.get("p90_tpot_ms", 0)
                result.tpot_p99_ms = data.get("p99_tpot_ms", 0)
                result.e2e_latency_ms = data.get("mean_e2el_ms", 0)
                result.total_time_sec = data.get("duration", 0)
                result.num_prompts = data.get("num_requests", 0)
                log.info(
                    "Parsed SGLang results: TTFT=%.2fms TPOT=%.2fms throughput=%.1f tok/s",
                    result.ttft_ms,
                    result.tpot_ms,
                    result.throughput_tok_per_sec,
                )
            except Exception:
                log.warning("Failed to parse SGLang JSON: %s", json_path, exc_info=True)

        if result.ttft_ms == 0 and Path(log_path).is_file():
            self._parse_from_log(log_path, result)

        return result

    def _parse_from_log(self, log_path: str, result: BenchmarkResult) -> None:
        try:
            text = Path(log_path).read_text()
            for pattern, attr in [
                (r"Mean TTFT.*?:\s*([\d.]+)\s*ms", "ttft_ms"),
                (r"Mean TPOT.*?:\s*([\d.]+)\s*ms", "tpot_ms"),
                (r"Output throughput.*?:\s*([\d.]+)\s*tok", "throughput_tok_per_sec"),
            ]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    setattr(result, attr, float(m.group(1)))
        except Exception:
            log.warning("Failed to parse SGLang log: %s", log_path, exc_info=True)

    def build_profile_env(self, start_iter: int, num_iters: int) -> dict[str, str]:
        return {}

    def operational_flags(self, concurrency: int) -> str:
        return f"--max-running-requests {concurrency}"

    def uses_metrics_based_profiling(self) -> bool:
        return True

    def profiling_start_request(
        self, port: int, delay: int, active: int, profiler_type: str = "nsys"
    ) -> list[str]:
        import json

        if profiler_type == "nsys":
            activities = ["CUDA_PROFILER"]
        else:
            activities = ["CPU", "GPU"]
        body = json.dumps(
            {
                "num_steps": active,
                "activities": activities,
            }
        )
        return [
            "curl",
            "-sf",
            "-X",
            "POST",
            f"http://localhost:{port}/start_profile",
            "-H",
            "Content-Type: application/json",
            "-d",
            body,
        ]

    def server_extra_flags_for_metrics(self) -> str:
        return "--enable-metrics"
