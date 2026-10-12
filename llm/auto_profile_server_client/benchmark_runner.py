"""Per-framework benchmark orchestrator.

Single-pod flow:
  Benchmark: server start → warmup → benchmark run (with metrics) → stop
  Profiling: nsys server start → warmup → arm profiler → capture run (with metrics) → stop

All output files use step-numbered prefixes (step_01_, step_02_, ...).
Benchmark and profiling outputs go into separate subdirectories.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from llm.auto_profile_server_client.client import run_client_benchmark
from llm.auto_profile_server_client.config import (
    DEFAULT_NUM_PROMPTS_MULTIPLIER,
    NSYS_DEFAULT_FLAGS,
    get_next_port,
)
from llm.auto_profile_server_client.frameworks.base import FrameworkAdapter
from llm.auto_profile_server_client.models import (
    BenchmarkResult,
    FrameworkEntry,
    Profile,
    make_test_name,
    save_json,
)
from llm.auto_profile_server_client.profiler import export_nsys_to_sqlite
from llm.auto_profile_server_client.server import (
    ServerProcess,
    start_server_in_container,
)
from llm.auto_profile_server_client.summary import SummaryWriter

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark + profiling for one framework at one concurrency."""

    def __init__(
        self,
        adapter: FrameworkAdapter,
        profile: Profile,
        fw_entry: FrameworkEntry,
        concurrency: int,
        output_dir: str,
        stability_threshold_pct: float,
        max_stability_retries: int,
        summary_writer: SummaryWriter,
    ):
        self.adapter = adapter
        self.profile = profile
        self.fw_entry = fw_entry
        self.concurrency = concurrency
        self.output_dir = output_dir
        self.stability_threshold_pct = stability_threshold_pct
        self.max_stability_retries = max_stability_retries
        self.summary = summary_writer

        num_gpus = profile.num_gpus
        self.test_name = make_test_name(
            adapter.name,
            profile.model,
            num_gpus,
            concurrency,
            profile.input_len,
            profile.output_len,
        )
        self.port = get_next_port(adapter.name)
        self.num_prompts = concurrency * DEFAULT_NUM_PROMPTS_MULTIPLIER
        self._step = 0

    def _next_step(self, label: str) -> str:
        self._step += 1
        return f"step_{self._step:02d}_{label}"

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute benchmark + profiling in a single pod."""
        log.info("=" * 60)
        log.info("START: %s", self.test_name)
        log.info("=" * 60)

        bench_dir = f"{self.output_dir}/benchmark"
        prof_dir = f"{self.output_dir}/profiling"
        Path(bench_dir).mkdir(parents=True, exist_ok=True)

        overall_start = time.time()
        results = {"test_name": self.test_name, "framework": self.adapter.name}

        # ── Benchmark phase ─────────────────────────────────────────────
        self._step = 0
        bench_result = self._run_benchmark(bench_dir)
        results["benchmark"] = bench_result.to_dict()
        save_json(results, f"{bench_dir}/results.json")

        # ── Profiling phase ──────────────────────────────────────────────
        if self.fw_entry.enable_trace:
            Path(prof_dir).mkdir(parents=True, exist_ok=True)
            self._step = 0
            self._run_profiling(
                prof_dir, benchmark_throughput=bench_result.throughput_tok_per_sec
            )
            results["profiling"] = "completed"
        else:
            results["profiling"] = "skipped"

        duration = time.time() - overall_start
        results["total_duration_sec"] = duration
        save_json(results, f"{self.output_dir}/results.json")

        # Write metadata describing what was tested
        metadata = {
            "model": self.profile.model,
            "framework": self.adapter.name,
            "num_gpus": self.profile.num_gpus,
            "concurrency": self.concurrency,
            "num_prompts": self.num_prompts,
            "input_len": self.profile.input_len,
            "output_len": self.profile.output_len,
            "exec_mode": self.profile.exec_mode_name,
            "extra_flags": self.fw_entry.server_args.extra_flags,
            "extra_env": self.fw_entry.server_args.extra_env,
            "enable_trace": self.fw_entry.enable_trace,
            "profiler_type": self.fw_entry.profiler_type,
            "port": self.port,
        }
        save_json(metadata, f"{self.output_dir}/metadata.json")

        # Write single summary
        summary_lines = [
            "=" * 72,
            f"RUN SUMMARY: {self.test_name}",
            "=" * 72,
            "",
            "Configuration:",
            f"  Model:       {self.profile.model}",
            f"  Framework:   {self.adapter.name}",
            f"  GPUs:        {self.profile.num_gpus}",
            f"  Concurrency: {self.concurrency}",
            f"  Prompts:     {self.num_prompts} (ISL={self.profile.input_len}, OSL={self.profile.output_len})",
            f"  Extra flags: {self.fw_entry.server_args.extra_flags}",
            "",
            "Benchmark Results:",
            f"  TTFT:       {bench_result.ttft_ms:.1f} ms",
            f"  TPOT:       {bench_result.tpot_ms:.2f} ms",
            f"  Throughput: {bench_result.throughput_tok_per_sec:.1f} tok/s",
            f"  E2E:        {bench_result.e2e_latency_ms:.0f} ms",
            "",
            f"Profiling:    {results['profiling']}",
            f"Duration:     {duration:.1f}s",
            "=" * 72,
        ]
        Path(f"{self.output_dir}/summary.txt").write_text("\n".join(summary_lines))

        log.info("DONE: %s (%.1fs)", self.test_name, duration)
        return results

    # ── Benchmark phase ──────────────────────────────────────────────────

    def _run_benchmark(self, out_dir: str) -> BenchmarkResult:
        """Start server → warmup → benchmark run."""
        server = self._start_server(out_dir, "server")
        try:
            # Step 1: Warmup
            self._run_warmup(out_dir)

            # Step 2: Benchmark run
            step = self._next_step("benchmark")
            log.info("[%s] Running benchmark", self.test_name)

            result, _, _ = run_client_benchmark(
                adapter=self.adapter,
                model=self.profile.model,
                port=self.port,
                num_prompts=self.num_prompts,
                input_len=self.profile.input_len,
                output_len=self.profile.output_len,
                concurrency=self.concurrency,
                output_dir=out_dir,
                run_label=step,
                extra_flags=self.fw_entry.client_args.extra_flags,
            )

            log.info(
                "[%s] Benchmark: TTFT=%.1fms TPOT=%.2fms throughput=%.1f tok/s",
                self.test_name,
                result.ttft_ms,
                result.tpot_ms,
                result.throughput_tok_per_sec,
            )
            return result
        finally:
            server.stop()

    # ── Profiling phase ──────────────────────────────────────────────────

    @staticmethod
    def _compute_profile_iterations(
        num_prompts: int, output_len: int, concurrency: int, active: int = 50
    ) -> tuple[int, int]:
        """Compute delay_iterations and active_iterations.

        Targets the pure-decode midpoint of the middle prompt/wave.
        """
        iters_per_prompt = output_len + 1

        if concurrency >= num_prompts:
            mid = output_len // 2
        else:
            mid_prompt = num_prompts // 2
            mid = mid_prompt * iters_per_prompt + output_len // 2

        return mid, active

    # ── Time-based profiling helpers ─────────────────────────────────────

    def _run_client(self, out_dir: str, step_prefix: str) -> None:
        """Run client benchmark and wait for completion. No metrics polling."""
        import shlex

        client_cmd = self.adapter.build_client_cmd(
            model=self.profile.model,
            port=self.port,
            num_prompts=self.num_prompts,
            input_len=self.profile.input_len,
            output_len=self.profile.output_len,
            concurrency=self.concurrency,
            output_path=f"{out_dir}/{step_prefix}_bench.json",
            extra_flags=self.fw_entry.client_args.extra_flags,
        )

        bench_log = f"{out_dir}/{step_prefix}_bench.log"
        header = f"{'=' * 80}\nCLIENT COMMAND ({step_prefix})\n{'=' * 80}\n{client_cmd}\n{'=' * 80}\n\n"
        Path(bench_log).write_text(header)
        log_cmd = (
            f"{client_cmd} 2>&1 | tee -a {shlex.quote(bench_log)}"
            f" > /dev/null; exit ${{PIPESTATUS[0]}}"
        )

        log.info("[%s] Running client (%s)", self.test_name, step_prefix)
        result = subprocess.run(["bash", "-c", log_cmd], timeout=600)
        log.info(
            "[%s] Client finished (%s, exit=%d)",
            self.test_name,
            step_prefix,
            result.returncode,
        )

    @staticmethod
    def _compute_fire_delay_sec(
        ttft_ms: float,
        tpot_ms: float,
        output_len: int,
        num_prompts: int,
        concurrency: int,
    ) -> float:
        """Seconds from client start to midpoint of pure decode."""
        import math

        request_ms = ttft_ms + output_len * tpot_ms
        decode_half_ms = output_len * tpot_ms / 2

        if concurrency >= num_prompts:
            return (ttft_ms + decode_half_ms) / 1000

        if concurrency == 1:
            mid = num_prompts // 2
            return (mid * request_ms + ttft_ms + decode_half_ms) / 1000

        num_waves = math.ceil(num_prompts / concurrency)
        mid_wave = num_waves // 2
        return (mid_wave * request_ms + ttft_ms + decode_half_ms) / 1000

    def _run_client_with_timed_fire(
        self,
        out_dir: str,
        step_prefix: str,
        fire_delay_sec: float,
        active: int,
        profiler_type: str = "nsys",
    ) -> dict:
        """Run capture client, sleep for fire_delay, then POST /start_profile."""
        import shlex

        client_cmd = self.adapter.build_client_cmd(
            model=self.profile.model,
            port=self.port,
            num_prompts=self.num_prompts,
            input_len=self.profile.input_len,
            output_len=self.profile.output_len,
            concurrency=self.concurrency,
            output_path=f"{out_dir}/{step_prefix}_bench.json",
            extra_flags=self.fw_entry.client_args.extra_flags,
        )

        bench_log = f"{out_dir}/{step_prefix}_bench.log"
        header = f"{'=' * 80}\nCLIENT COMMAND ({step_prefix})\n{'=' * 80}\n{client_cmd}\n{'=' * 80}\n\n"
        Path(bench_log).write_text(header)
        log_cmd = (
            f"{client_cmd} 2>&1 | tee -a {shlex.quote(bench_log)}"
            f" > /dev/null; exit ${{PIPESTATUS[0]}}"
        )

        log.info(
            "[%s] Starting capture client, will fire in %.1fs (%s)",
            self.test_name,
            fire_delay_sec,
            step_prefix,
        )
        client_proc = subprocess.Popen(["bash", "-c", log_cmd])
        client_start = time.time()

        time.sleep(fire_delay_sec)
        elapsed = time.time() - client_start

        if client_proc.poll() is not None:
            log.warning(
                "[%s] Client finished before fire delay (exit=%d)",
                self.test_name,
                client_proc.returncode,
            )
            return {}

        log.info(
            "[%s] Fire delay reached (%.1fs) — firing /start_profile",
            self.test_name,
            elapsed,
        )
        start_cmd = self.adapter.profiling_start_request(
            self.port, 0, active, profiler_type
        )
        r = subprocess.run(start_cmd, capture_output=True, text=True, timeout=30)

        fire_result = {
            "fire_delay_sec": fire_delay_sec,
            "actual_elapsed_sec": round(elapsed, 2),
            "start_profile_exit": r.returncode,
            "active_iterations": active,
            "profiler_type": profiler_type,
        }

        # Wait for client to finish
        timeout = fire_delay_sec + 120
        remaining = timeout - elapsed
        deadline = time.time() + remaining
        while time.time() < deadline:
            if client_proc.poll() is not None:
                log.info(
                    "[%s] Client finished (exit=%d)",
                    self.test_name,
                    client_proc.returncode,
                )
                break
            time.sleep(5)
        else:
            log.info("[%s] Capture timeout — killing client", self.test_name)
            client_proc.kill()
            client_proc.wait(timeout=10)

        return fire_result

    # ── Profiling ────────────────────────────────────────────────────────

    def _run_profiling(self, out_dir: str, benchmark_throughput: float = 0) -> None:
        """Profiling server → warmup → capture → finalize trace.

        Supports two profiler types:
          nsys:  nsys-wrapped server, SIGINT finalization, .nsys-rep output
          torch: normal server, graceful stop, .pt.trace.json output

        Supports two capture methods:
          time-based (SGLang):  dry-run → validate → timed fire
          iteration-based (vLLM): arm profiler → run client
        """
        delay, active = self._compute_profile_iterations(
            self.num_prompts,
            self.profile.output_len,
            self.concurrency,
        )
        use_time_based = self.adapter.uses_metrics_based_profiling()
        profiler_type = self.fw_entry.profiler_type

        log.info(
            "[%s] Profiling: profiler=%s, active_iterations=%d, method=%s, "
            "benchmark_throughput=%.1f tok/s",
            self.test_name,
            profiler_type,
            active,
            "time-based" if use_time_based else f"delay_iterations={delay}",
            benchmark_throughput,
        )

        server = self._start_profiling_server(out_dir, delay, active, profiler_type)

        try:
            # Step 1: Warmup
            self._run_warmup(out_dir)

            if use_time_based:
                self._profiling_time_based_path(
                    out_dir, benchmark_throughput, active, profiler_type
                )
            else:
                self._profiling_iteration_path(out_dir, delay, active, profiler_type)

            # Stop profiler (cleanup — may fail if auto-stopped)
            step_stop = self._next_step("profile_stop")
            stop_cmd = self.adapter.profiling_stop_request(self.port)
            try:
                r = subprocess.run(stop_cmd, capture_output=True, text=True, timeout=30)
                save_json(
                    {"action": "stop_profile", "curl_exit": r.returncode},
                    f"{out_dir}/{step_stop}.json",
                )
            except subprocess.TimeoutExpired:
                save_json(
                    {"action": "stop_profile", "timeout": True},
                    f"{out_dir}/{step_stop}.json",
                )

            # Finalize trace
            if profiler_type == "nsys":
                self._finalize_nsys_trace(out_dir, server)
            else:
                self._finalize_torch_trace(out_dir, server)

            step_meta = self._next_step("profile_meta")
            save_json(
                {
                    "profiler_type": profiler_type,
                    "method": "time-based" if use_time_based else "iteration-based",
                    "delay_iterations": delay,
                    "active_iterations": active,
                    "num_prompts": self.num_prompts,
                    "output_len": self.profile.output_len,
                    "concurrency": self.concurrency,
                    "benchmark_throughput": benchmark_throughput,
                },
                f"{out_dir}/{step_meta}.json",
            )

            log.info("[%s] Profiling complete", self.test_name)

        except Exception:
            server.stop()
            raise
        finally:
            if server.is_running:
                server.stop()

    def _profiling_time_based_path(
        self, out_dir: str, benchmark_throughput: float, active: int, profiler_type: str
    ) -> None:
        """Time-based profiling: dry-run → compute fire delay → timed capture."""
        # Step 2: Dry-run — run client to get profiler-adjusted TTFT/TPOT
        step_dry = self._next_step("dryrun")
        log.info(
            "[%s] Dry-run: running client to measure adjusted timing", self.test_name
        )
        dryrun_start = time.time()
        self._run_client(out_dir, step_dry)
        dryrun_duration = time.time() - dryrun_start

        dryrun_json = f"{out_dir}/{step_dry}_bench.json"
        dryrun_result = self.adapter.parse_benchmark_results(
            dryrun_json, f"{out_dir}/{step_dry}_bench.log"
        )

        ttft = dryrun_result.ttft_ms
        tpot = dryrun_result.tpot_ms
        if ttft <= 0 or tpot <= 0:
            raise RuntimeError(
                f"Dry-run produced invalid TTFT={ttft:.1f}ms / TPOT={tpot:.2f}ms. "
                f"Cannot compute fire delay. See {dryrun_json}"
            )

        # Step 3: Validate — compute fire delay and sanity-check
        step_validate = self._next_step("validate")
        fire_delay = self._compute_fire_delay_sec(
            ttft, tpot, self.profile.output_len, self.num_prompts, self.concurrency
        )

        log.info(
            "[%s] Dry-run: TTFT=%.1fms TPOT=%.2fms → fire_delay=%.1fs "
            "(dryrun_duration=%.1fs)",
            self.test_name,
            ttft,
            tpot,
            fire_delay,
            dryrun_duration,
        )

        if fire_delay >= dryrun_duration:
            save_json(
                {
                    "status": "FAILED",
                    "reason": "fire_delay exceeds dry-run duration",
                    "ttft_ms": ttft,
                    "tpot_ms": tpot,
                    "fire_delay_sec": fire_delay,
                    "dryrun_duration_sec": round(dryrun_duration, 1),
                },
                f"{out_dir}/{step_validate}.json",
            )
            raise RuntimeError(
                f"Fire delay {fire_delay:.1f}s >= dry-run duration "
                f"{dryrun_duration:.1f}s. Profiler would fire after workload ends."
            )

        save_json(
            {
                "status": "OK",
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "fire_delay_sec": round(fire_delay, 2),
                "dryrun_duration_sec": round(dryrun_duration, 1),
                "dryrun_throughput_tok_s": dryrun_result.throughput_tok_per_sec,
                "benchmark_throughput_tok_s": benchmark_throughput,
            },
            f"{out_dir}/{step_validate}.json",
        )

        # Step 4: Live capture — start client, sleep fire_delay, POST /start_profile
        step_capture = self._next_step("capture")
        log.info(
            "[%s] Live capture: will fire /start_profile at %.1fs",
            self.test_name,
            fire_delay,
        )
        fire_result = self._run_client_with_timed_fire(
            out_dir, step_capture, fire_delay, active, profiler_type
        )

        step_fire = self._next_step("profile_fire")
        if fire_result:
            log.info(
                "[%s] Profile fired at %.1fs (gen_tokens=%.0f, running=%.0f)",
                self.test_name,
                fire_result.get("actual_elapsed_sec", 0),
                fire_result.get("gen_tokens_at_fire", 0),
                fire_result.get("requests_running_at_fire", 0),
            )
            save_json(fire_result, f"{out_dir}/{step_fire}.json")
        else:
            save_json(
                {"status": "FAILED", "reason": "fire delay not reached"},
                f"{out_dir}/{step_fire}.json",
            )
            log.warning("[%s] Live capture: fire delay not reached", self.test_name)

    def _profiling_iteration_path(
        self, out_dir: str, delay: int, active: int, profiler_type: str = "nsys"
    ) -> None:
        """Iteration-based profiling (vLLM): arm profiler → run client → wait."""
        import shlex

        step_arm = self._next_step("profile_arm")
        log.info("[%s] Arming profiler (POST /start_profile)", self.test_name)
        start_cmd = self.adapter.profiling_start_request(
            self.port, delay, active, profiler_type
        )
        r = subprocess.run(start_cmd, capture_output=True, text=True, timeout=30)
        save_json(
            {
                "action": "start_profile",
                "curl_exit": r.returncode,
                "delay_iterations": delay,
                "active_iterations": active,
            },
            f"{out_dir}/{step_arm}.json",
        )

        step_cap = self._next_step("capture")
        log.info("[%s] Starting capture client", self.test_name)

        client_cmd = self.adapter.build_client_cmd(
            model=self.profile.model,
            port=self.port,
            num_prompts=self.num_prompts,
            input_len=self.profile.input_len,
            output_len=self.profile.output_len,
            concurrency=self.concurrency,
            output_path=f"{out_dir}/{step_cap}_bench.json",
            extra_flags=self.fw_entry.client_args.extra_flags,
        )

        bench_log = f"{out_dir}/{step_cap}_bench.log"
        header = f"{'=' * 80}\nCLIENT COMMAND ({step_cap})\n{'=' * 80}\n{client_cmd}\n{'=' * 80}\n\n"
        Path(bench_log).write_text(header)
        log_cmd = (
            f"{client_cmd} 2>&1 | tee -a {shlex.quote(bench_log)}"
            f" > /dev/null; exit ${{PIPESTATUS[0]}}"
        )
        client_proc = subprocess.Popen(["bash", "-c", log_cmd])

        estimated_sec = (delay + active) * 0.01 + 30
        log.info(
            "[%s] Waiting up to %.0fs for capture to complete",
            self.test_name,
            estimated_sec,
        )

        deadline = time.time() + estimated_sec
        while time.time() < deadline:
            if client_proc.poll() is not None:
                log.info(
                    "[%s] Client finished (exit=%d)",
                    self.test_name,
                    client_proc.returncode,
                )
                break
            time.sleep(5)
        else:
            log.info("[%s] Capture timeout — killing client", self.test_name)
            client_proc.kill()
            client_proc.wait(timeout=10)

    def _finalize_nsys_trace(self, out_dir: str, server: ServerProcess) -> None:
        """SIGINT for nsys trace finalization + wait for file to stabilize."""
        import os
        import signal

        log.info(
            "[%s] Sending SIGINT to server for nsys trace finalization", self.test_name
        )
        try:
            os.kill(server.process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

        nsys_rep = f"{out_dir}/trace.nsys-rep"
        log.info("[%s] Waiting for nsys trace at %s", self.test_name, nsys_rep)
        last_size = -1
        stable_count = 0
        for _ in range(60):
            if Path(nsys_rep).is_file():
                size = Path(nsys_rep).stat().st_size
                if size > 0 and size == last_size:
                    stable_count += 1
                    if stable_count >= 3:
                        break
                else:
                    stable_count = 0
                last_size = size
            time.sleep(2)

        if Path(nsys_rep).is_file():
            log.info(
                "[%s] nsys trace: %s (%.1f MB)",
                self.test_name,
                nsys_rep,
                Path(nsys_rep).stat().st_size / 1e6,
            )
            export_nsys_to_sqlite(nsys_rep)
        else:
            log.warning("[%s] No .nsys-rep file at %s", self.test_name, nsys_rep)

    def _finalize_torch_trace(self, out_dir: str, server: ServerProcess) -> None:
        """Copy torch trace files from /tmp to output dir, then stop server.

        Must collect files BEFORE stopping — they live on ephemeral storage.
        SGLang writes to /tmp with pattern: sglang_profile_<id>_<rank>.json
        vLLM writes to /tmp/.vllm/profiles/ with pattern: *.pt.trace.json*
        """
        import glob
        import shutil

        log.info("[%s] Collecting torch trace files", self.test_name)

        search_patterns = [
            "/tmp/*.trace.json.gz",
            "/tmp/*.trace.json",
            "/tmp/vllm_traces/**/*.json*",
        ]

        traces = []
        for pattern in search_patterns:
            traces.extend(glob.glob(pattern, recursive=True))

        if traces:
            for src in traces:
                dst = f"{out_dir}/{Path(src).name}"
                shutil.copy2(src, dst)
                log.info(
                    "[%s] Torch trace: %s (%.1f MB)",
                    self.test_name,
                    dst,
                    Path(dst).stat().st_size / 1e6,
                )
        else:
            log.warning("[%s] No torch trace files found in /tmp", self.test_name)

        server.stop()

    # ── Server management ────────────────────────────────────────────────

    def _start_server(self, out_dir: str, log_name: str) -> ServerProcess:
        log.info("[%s] Starting server", self.test_name)
        start = time.time()

        op_flags = self.adapter.operational_flags(self.concurrency)
        extra = self.fw_entry.server_args.extra_flags
        if op_flags:
            extra = f"{extra} {op_flags}".strip()

        cmd = self.adapter.build_server_cmd(
            model=self.profile.model,
            tp=self.profile.num_gpus,
            port=self.port,
            extra_flags=extra,
            extra_env=self.fw_entry.server_args.extra_env,
        )

        health_url = self.adapter.health_check_url(self.port)
        server_log = f"{out_dir}/{log_name}.log"
        server = start_server_in_container(
            cmd, self.port, health_url, log_path=server_log
        )
        server.wait_until_ready()

        log.info("[%s] Server ready in %.1fs", self.test_name, time.time() - start)
        return server

    def _start_profiling_server(
        self, out_dir: str, delay_iters: int, active_iters: int, profiler_type: str
    ) -> ServerProcess:
        log.info(
            "[%s] Starting %s profiling server (delay=%d, active=%d)",
            self.test_name,
            profiler_type,
            delay_iters,
            active_iters,
        )
        start = time.time()

        op_flags = self.adapter.operational_flags(self.concurrency)
        extra = self.fw_entry.server_args.extra_flags
        if op_flags:
            extra = f"{extra} {op_flags}".strip()

        prof_flags = self.adapter.profiling_server_extra_flags(
            delay_iters, active_iters, profiler_type
        )
        if prof_flags:
            extra = f"{extra} {prof_flags}".strip()

        server_cmd = self.adapter.build_server_cmd(
            model=self.profile.model,
            tp=self.profile.num_gpus,
            port=self.port,
            extra_flags=extra,
            extra_env=self.fw_entry.server_args.extra_env,
        )

        if profiler_type == "nsys":
            nsys_flags = " ".join(NSYS_DEFAULT_FLAGS)
            cmd = f"nsys profile {nsys_flags} -o {out_dir}/trace {server_cmd}"
        else:
            cmd = server_cmd

        server_log = f"{out_dir}/server.log"
        server = start_server_in_container(
            cmd,
            self.port,
            self.adapter.health_check_url(self.port),
            log_path=server_log,
        )
        server.wait_until_ready()
        log.info(
            "[%s] %s server ready in %.1fs",
            self.test_name,
            profiler_type,
            time.time() - start,
        )
        return server

    # ── Helpers ───────────────────────────────────────────────────────────

    def _run_warmup(self, out_dir: str) -> None:
        log.info("[%s] Running warmup", self.test_name)
        start = time.time()
        step = self._next_step("warmup")

        run_client_benchmark(
            adapter=self.adapter,
            model=self.profile.model,
            port=self.port,
            num_prompts=self.num_prompts,
            input_len=self.profile.input_len,
            output_len=self.profile.output_len,
            concurrency=self.concurrency,
            output_dir=out_dir,
            run_label=step,
            extra_flags=self.fw_entry.client_args.extra_flags,
        )

        log.info("[%s] Warmup done in %.1fs", self.test_name, time.time() - start)
