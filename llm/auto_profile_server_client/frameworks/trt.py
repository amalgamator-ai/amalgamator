"""TRT-LLM framework adapter."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from llm.auto_profile_server_client.frameworks.base import FrameworkAdapter
from llm.auto_profile_server_client.models import BenchmarkResult

log = logging.getLogger(__name__)


class TrtAdapter(FrameworkAdapter):
    name = "trt"
    default_port = 8000

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
            f"trtllm-serve {model} "
            f"--backend pytorch "
            f"--tp_size {tp} "
            f"--port {port} "
            f"--host 0.0.0.0 "
            f"--kv_cache_free_gpu_memory_fraction 0.9 "
            f"--trust_remote_code "
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
        result_dir = str(Path(output_path).parent)
        return (
            f"python -m tensorrt_llm.serve.scripts.benchmark_serving "
            f"--backend openai "
            f"--model {model} "
            f"--base-url http://localhost:{port} "
            f"--dataset-name random "
            f"--random-input-len {input_len} "
            f"--random-output-len {output_len} "
            f"--num-prompts {num_prompts} "
            f"--max-concurrency {concurrency} "
            f"--ignore-eos "
            f"--save-result "
            f"--result-dir {result_dir} "
            f"--result-filename {Path(output_path).name} "
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
                log.info("Parsed TRT results: TTFT=%.2fms TPOT=%.2fms throughput=%.1f tok/s",
                         result.ttft_ms, result.tpot_ms, result.throughput_tok_per_sec)
            except Exception:
                log.warning("Failed to parse TRT JSON: %s", json_path, exc_info=True)

        if result.ttft_ms == 0 and Path(log_path).is_file():
            self._parse_from_log(log_path, result)

        return result

    def _parse_from_log(self, log_path: str, result: BenchmarkResult) -> None:
        try:
            text = Path(log_path).read_text()
            for pattern, attr in [
                (r"Mean TTFT.*?:\s*([\d.]+)\s*ms", "ttft_ms"),
                (r"Mean TPOT.*?:\s*([\d.]+)\s*ms", "tpot_ms"),
                (r"Output throughput.*?:\s*([\d.]+)", "throughput_tok_per_sec"),
                (r"request_output_throughput_tok_per_s.*?:\s*([\d.]+)", "throughput_tok_per_sec"),
            ]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m and getattr(result, attr) == 0:
                    setattr(result, attr, float(m.group(1)))
        except Exception:
            log.warning("Failed to parse TRT log: %s", log_path, exc_info=True)

    def build_profile_env(self, start_iter: int, num_iters: int) -> dict[str, str]:
        end_iter = start_iter + num_iters
        return {"TLLM_PROFILE_START_STOP": f"{start_iter}-{end_iter}"}

    def metrics_url(self, port: int) -> str:
        return f"http://localhost:{port}/prometheus/metrics"
