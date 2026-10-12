"""Abstract base for framework adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm.auto_profile_server_client.models import BenchmarkResult


class FrameworkAdapter(ABC):
    """Interface each framework implements for server-client benchmarking."""

    name: str
    default_port: int

    @abstractmethod
    def build_server_cmd(
        self,
        model: str,
        tp: int,
        port: int,
        extra_flags: str = "",
        extra_env: dict[str, str] | None = None,
    ) -> str:
        """Return the shell command to start the serving endpoint."""

    @abstractmethod
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
        """Return the client benchmark command."""

    @abstractmethod
    def parse_benchmark_results(self, json_path: str, log_path: str) -> BenchmarkResult:
        """Parse framework-specific output into structured TTFT/TPOT/throughput."""

    @abstractmethod
    def build_profile_env(self, start_iter: int, num_iters: int) -> dict[str, str]:
        """Return env vars needed for profiling (e.g., TRT's TLLM_PROFILE_START_STOP)."""

    def health_check_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def metrics_url(self, port: int) -> str:
        return f"http://localhost:{port}/metrics"

    def server_extra_flags_for_metrics(self) -> str:
        """Extra server flags needed to enable metrics."""
        return ""

    def operational_flags(self, concurrency: int) -> str:
        """Flags the pipeline sets automatically per-run (not from the recipe config)."""
        return ""

    def uses_metrics_based_profiling(self) -> bool:
        """If True, profiling uses metrics polling to detect decode midpoint
        instead of iteration-based delay."""
        return False

    def profiling_server_extra_flags(
        self, delay: int, active: int, profiler_type: str = "nsys"
    ) -> str:
        """Extra server flags for profiling. Only needed if the framework
        requires server-side config (e.g. vLLM --profiler-config)."""
        return ""

    def profiling_start_request(
        self, port: int, delay: int, active: int, profiler_type: str = "nsys"
    ) -> list[str]:
        """Build the curl command to arm the profiler via HTTP."""
        return ["curl", "-sf", "-X", "POST", f"http://localhost:{port}/start_profile"]

    def profiling_stop_request(self, port: int) -> list[str]:
        """Build the curl command to stop the profiler via HTTP."""
        return ["curl", "-sf", "-X", "POST", f"http://localhost:{port}/stop_profile"]
