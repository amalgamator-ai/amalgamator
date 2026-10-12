"""Top-level pipeline: orchestrate benchmarks across all profiles and concurrencies.

Entry point for both CLI usage and LLM-driven execution.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from llm.auto_profile_server_client.benchmark_runner import BenchmarkRunner
from llm.auto_profile_server_client.config import FRAMEWORK_PORTS
from llm.auto_profile_server_client.docker_runner import (
    DockerRunner,
    wait_for_gpu_memory_release,
)
from llm.auto_profile_server_client.frameworks import get_adapter
from llm.auto_profile_server_client.models import RunConfig, StepSummary, save_json
from llm.auto_profile_server_client.run_config import load_config
from llm.auto_profile_server_client.summary import SummaryWriter

log = logging.getLogger(__name__)


def run_pipeline(config: RunConfig) -> dict:
    """Execute the full benchmark pipeline.

    When running locally (outside docker), this launches docker containers
    that run the benchmark_runner internally.

    When running inside docker (e.g., cluster pod), this calls
    benchmark_runner directly.
    """
    output_dir = config.output_dir or _default_output_dir()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    summary = SummaryWriter(output_dir)
    pipeline_start = time.time()
    all_results = []

    log.info("=" * 72)
    log.info("AUTO PROFILE SERVER-CLIENT PIPELINE")
    log.info("=" * 72)
    log.info("Output dir: %s", output_dir)
    log.info("Profiles: %d", len(config.profiles))
    log.info("Concurrencies: %s", config.concurrencies)
    log.info("Stability threshold: %.1f%%", config.stability_threshold_pct)
    log.info("Max stability retries: %d", config.max_stability_retries)

    inside_docker = _is_inside_docker()

    if inside_docker:
        log.info("Running INSIDE docker — direct execution mode")
        all_results = _run_direct(config, output_dir, summary)
    else:
        log.info("Running OUTSIDE docker — will launch containers")
        all_results = _run_via_docker(config, output_dir, summary)

    pipeline_duration = time.time() - pipeline_start

    results_path = Path(output_dir) / "pipeline_results.json"
    save_json(
        {"results": all_results, "duration_sec": pipeline_duration}, str(results_path)
    )

    log.info("=" * 72)
    log.info("PIPELINE COMPLETE (%.1fs)", pipeline_duration)
    log.info("Results: %s", results_path)
    log.info("=" * 72)

    return {"results": all_results, "output_dir": output_dir}


def _run_direct(
    config: RunConfig, output_dir: str, summary: SummaryWriter
) -> list[dict]:
    """Run benchmarks directly (inside docker or local with GPUs)."""
    all_results = []

    for profile in config.profiles:
        for fw_entry in profile.frameworks:
            adapter = get_adapter(fw_entry.name)

            for concurrency in config.concurrencies:
                runner = BenchmarkRunner(
                    adapter=adapter,
                    profile=profile,
                    fw_entry=fw_entry,
                    concurrency=concurrency,
                    output_dir=output_dir,
                    stability_threshold_pct=config.stability_threshold_pct,
                    max_stability_retries=config.max_stability_retries,
                    summary_writer=summary,
                )
                try:
                    result = runner.run()
                    all_results.append(result)
                except Exception as e:
                    log.error(
                        "Benchmark failed: %s — %s", runner.test_name, e, exc_info=True
                    )
                    summary.write_step_summary(
                        StepSummary(
                            step_name=runner.test_name,
                            status="failed",
                            issues=[str(e)],
                        )
                    )

    return all_results


def _run_via_docker(
    config: RunConfig, output_dir: str, summary: SummaryWriter
) -> list[dict]:
    """Launch docker containers to run benchmarks."""
    docker = DockerRunner()
    all_results = []

    # Update docker images
    for _fw_name, image in config.docker_images.items():
        docker.ensure_image_updated(image)

    fw_order = []
    for profile in config.profiles:
        for fw_entry in profile.frameworks:
            fw_order.append((profile, fw_entry))

    for i, (profile, fw_entry) in enumerate(fw_order):
        if i > 0:
            wait_for_gpu_memory_release()

        image = config.docker_images[fw_entry.name]
        container_name = (
            f"{fw_entry.name}_auto_profile_{os.environ.get('USER', 'user')}"
        )

        # Build the inner command that runs inside docker
        inner_config_path = f"/tmp/run_config_{fw_entry.name}.json"
        inner_cmd = (
            f"python -m llm.auto_profile_server_client.pipeline "
            f"--config {inner_config_path} "
            f"--framework {fw_entry.name} "
            f"--model {profile.model}"
        )

        # Prepare commands
        prepare = ""
        if fw_entry.server_args.prepare_cmds:
            cmds = " && ".join(fw_entry.server_args.prepare_cmds)
            prepare = f"{cmds} && "

        full_cmd = f"{prepare}{inner_cmd}"

        port = FRAMEWORK_PORTS.get(fw_entry.name, 8000)
        env = dict(fw_entry.server_args.extra_env)

        log.info("Launching docker container: %s", container_name)
        result = docker.run(
            image=image,
            name=container_name,
            cmd=full_cmd,
            ports={port: port},
            env=env,
        )

        if result.returncode == 0:
            log.info("Container %s completed successfully", container_name)
        else:
            log.error(
                "Container %s failed (exit=%d)", container_name, result.returncode
            )
            summary.write_step_summary(
                StepSummary(
                    step_name=f"docker_{fw_entry.name}",
                    status="failed",
                    issues=[f"Container exited with code {result.returncode}"],
                )
            )

    return all_results


def _is_inside_docker() -> bool:
    return (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or "container" in Path("/proc/1/cgroup").read_text()
        if Path("/proc/1/cgroup").exists()
        else False
    )


def _default_output_dir() -> str:
    return f"llm/auto_profile_server_client/results/run_{int(time.time())}"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Server-client benchmark and profiling pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to run config JSON")
    parser.add_argument(
        "--framework", help="Run only this framework (for docker inner mode)"
    )
    parser.add_argument("--model", help="Run only this model (for docker inner mode)")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    # Filter to specific framework/model if running in inner docker mode
    if args.framework:
        for profile in config.profiles:
            profile.frameworks = [
                fw for fw in profile.frameworks if fw.name == args.framework
            ]
        config.profiles = [p for p in config.profiles if p.frameworks]

    if args.model:
        config.profiles = [p for p in config.profiles if p.model == args.model]

    run_pipeline(config)


if __name__ == "__main__":
    main()
