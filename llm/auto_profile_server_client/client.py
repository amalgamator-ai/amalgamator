"""Client benchmark runner — executes the framework's bench command and captures output."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from llm.auto_profile_server_client.frameworks.base import FrameworkAdapter
from llm.auto_profile_server_client.models import BenchmarkResult

log = logging.getLogger(__name__)


def run_client_benchmark(
    adapter: FrameworkAdapter,
    model: str,
    port: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    concurrency: int,
    output_dir: str,
    run_label: str,
    extra_flags: str = "",
    timeout: int = 3600,
) -> tuple[BenchmarkResult, str, str]:
    """Run a single client benchmark and return parsed results.

    Returns:
        (BenchmarkResult, json_path, log_path)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = str(out_dir / f"{run_label}_bench.json")
    log_path = str(out_dir / f"{run_label}_bench.log")

    cmd = adapter.build_client_cmd(
        model=model,
        port=port,
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        concurrency=concurrency,
        output_path=json_path,
        extra_flags=extra_flags,
    )

    log.info("[%s] Running client benchmark: %s", run_label, cmd[:200])
    start = time.time()

    import shlex
    header = f"{'=' * 80}\nCLIENT COMMAND ({run_label})\n{'=' * 80}\n{cmd}\n{'=' * 80}\n\n"
    Path(log_path).write_text(header)
    log_cmd = f"{cmd} 2>&1 | tee -a {shlex.quote(log_path)} > /dev/null; exit ${{PIPESTATUS[0]}}"

    result = subprocess.run(
        ["bash", "-c", log_cmd],
        timeout=timeout,
    )

    duration = time.time() - start
    log.info("[%s] Client benchmark completed in %.1fs (exit=%d)",
             run_label, duration, result.returncode)

    if result.returncode != 0:
        tail = Path(log_path).read_text()[-2000:] if Path(log_path).is_file() else ""
        log.error("[%s] Client benchmark failed:\n%s", run_label, tail)
        raise RuntimeError(
            f"Client benchmark '{run_label}' failed with exit code {result.returncode}. "
            f"See log: {log_path}"
        )

    bench_result = adapter.parse_benchmark_results(json_path, log_path)
    bench_result.total_time_sec = duration
    return bench_result, json_path, log_path
