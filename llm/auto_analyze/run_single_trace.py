# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single-Trace Analysis Entry Point
==================================

Analyzes a single framework's GPU trace to extract transformer block structure,
correlate GPU operations to high-level operations, and produce performance
analysis with improvement proposals.

Usage:
    python -m llm.auto_analyze.run_single_trace --config <config.json> [--no-clean]
    python -m llm.auto_analyze.run_single_trace --config <config.json> --agent-config <agent.json>

Arguments:
    --config          Path to single-trace config JSON (required)
    --agent-config    Optional provider-neutral agent config JSON
    --no-clean        Do not clean output directory before running

Config JSON fields:
    Required:
      framework_name, model, gpu_type, batch_size_range, prefill_size_range,
      output_size_range, trace_file, framework_source_code, output_dir

    Optional:
      commit_id           Git commit to checkout (default: "HEAD")
      run_command         Command used to run the framework (default: null).
                          When null, auto-extracted from the first line of run_log
                          that looks like a command. Prints a warning if not found.
      run_log             Path to framework run log file
      trace_gpu_focus     GPU to focus on: "ALL", GPU ID, or null (default: null).
                          Use null for PyTorch traces that are single-GPU.
      high_level_focus    Focus guidance for high-level block analysis (default: null)
      perf_analysis_focus Focus areas for performance analysis
      skip_perf_analysis  Skip perf analysis and trace generation (default: false)
      max_gpu_ops         Max GPU ops to extract (default: 2000)

Outputs (in output_dir):
    Always produced:
      - transformer_block_high_level_ops.txt  High-level transformer block operations
      - gpu_ops.txt                           GPU operations extracted from trace
      - gpu_ops_to_blocks.txt                 GPU ops correlated to transformer blocks
      - median_block.txt                      Selected median transformer block
      - run_params.txt                        Run parameters for downstream tools
      - run_originals/                        Copies of trace file, run command, run log

    When skip_perf_analysis is false (default):
      - perf_analysis_single_trace.txt        Detailed performance analysis
      - single_trace_transformer_block.json   Chrome trace JSON for Perfetto
      - single_trace_transformer_block.txt    Human-readable trace summary
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import asyncio
import os
import shutil
import sys
import time
from dataclasses import asdict

from common.claude_utils import PipelineStep, _all_steps_complete
from common.iteration_utils import check_converged, resolve_output_file
from common.utils import safe_clean_dir, setup_logging
from llm.agents import agent_run, agent_session
from llm.auto_analyze.agent_support import (
    add_agent_arguments,
    log_agent_config,
    make_agent_config,
)
from llm.auto_analyze.configs.single_trace_config import (
    GPU_OPS_TO_BLOCKS_FILE,
    GPU_OPS_TXT_FILE,
    HIGH_LEVEL_OPS_FILE,
    MEDIAN_BLOCK_FILE,
    PERF_ANALYSIS_SINGLE_FILE,
    SingleTraceConfig,
)
from llm.auto_analyze.prompts.chrome_trace_prompts import build_single_trace_json_prompt
from llm.auto_analyze.prompts.single_trace_prompts import (
    GpuOpsPrompt,
    gen_correlation_iter_steps,
    gen_high_level_ops_iter_steps,
    gen_perf_analysis_iter_steps,
)


def log_config(config: SingleTraceConfig):
    print("=" * 70)
    print("  SINGLE-TRACE ANALYSIS")
    print("=" * 70)
    print(f"  Framework:     {config.framework_name}")
    print(f"  Model:         {config.model}")
    print(f"  HF URL:        https://huggingface.co/{config.model}")
    print(f"  GPU:           {config.gpu_type}")
    print(f"  Batch size:    {config.batch_size_range}")
    print(f"  Prefill size:  {config.prefill_size_range}")
    print(f"  Output size:   {config.output_size_range}")
    print(f"  Trace file:    {config.trace_file}")
    print(f"  Trace type:    {config.trace_file_type}")
    print(f"  GPU focus:     {config.trace_gpu_focus or '(none - single GPU trace)'}")
    print(f"  Source code:   {config.framework_source_code}")
    if config.commit_id:
        print(f"  Commit ID:     {config.commit_id}")
    if config.run_log:
        print(f"  Run log:       {config.run_log}")
    print(f"  Run command:   {config.run_command or '(not provided)'}")
    if config.high_level_focus:
        print(f"  Focus:         {config.high_level_focus}")
    print(f"  Output dir:    {config.output_dir}")
    print(f"  Skip perf analysis: {config.skip_perf_analysis}")

    print("=" * 70)


def save_run_artifacts(config: SingleTraceConfig):
    output_dir = config.output_dir

    # Save run_params.txt
    params_path = os.path.join(output_dir, "run_params.txt")
    with open(params_path, "w") as f:
        f.write("SINGLE TRACE ANALYSIS PARAMETERS\n")
        f.write("=" * 50 + "\n")
        for key, value in asdict(config).items():
            f.write(f"{key}: {value}\n")
    print(f"  Saved run parameters: {params_path}")

    # Create run_originals subdirectory
    originals_dir = os.path.join(output_dir, "run_originals")
    os.makedirs(originals_dir, exist_ok=True)

    # Copy trace file (prefer the original gzipped version to save space)
    trace_to_copy = getattr(config, "original_trace_file", config.trace_file)
    if trace_to_copy and os.path.exists(trace_to_copy):
        dst = os.path.join(originals_dir, os.path.basename(trace_to_copy))
        if os.path.abspath(trace_to_copy) != os.path.abspath(dst):
            shutil.copy2(trace_to_copy, dst)
            print(f"  Copied trace file: {dst}")
        else:
            print("  Trace file already in output dir, skipping copy")

    # Save run_command.txt
    if config.run_command:
        cmd_path = os.path.join(originals_dir, "run_command.txt")
        with open(cmd_path, "w") as f:
            f.write(config.run_command + "\n")
        print(f"  Saved run command: {cmd_path}")

    # Copy run log
    if config.run_log and os.path.exists(config.run_log):
        dst = os.path.join(originals_dir, "run_log.txt")
        shutil.copy2(config.run_log, dst)
        print(f"  Copied run log: {dst}")


def log_outputs(output_dir, output_files):
    print()
    print("=" * 70)
    print("  OUTPUT FILES")
    print("=" * 70)
    for label, filename in output_files.items():
        path = (
            filename if os.path.isabs(filename) else os.path.join(output_dir, filename)
        )
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {filename}")
    print("=" * 70)


def _format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    raw = f"{seconds:.1f}s"
    human = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
    return raw, human


def _format_tokens(count):
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _print_phase_summary(
    phase_results, all_step_timings, total_duration, output_dir=None
):
    """Print a summary table of all phases with iteration counts and convergence status."""

    rows = []
    for phase in phase_results:
        if phase["max_iterations"] <= 1:
            converged_str = "N/A"
        elif phase["converged"]:
            converged_str = "Yes (iter {})".format(phase["iterations"])
        else:
            converged_str = "No"
        raw, human = _format_duration(phase["duration"])
        rows.append(
            {
                "name": phase["name"],
                "iters": "{}/{}".format(phase["iterations"], phase["max_iterations"]),
                "conv": converged_str,
                "dur_raw": raw,
                "dur_human": human,
            }
        )

    total_raw, total_human = _format_duration(total_duration)
    rows.append(
        {
            "name": "TOTAL",
            "iters": "",
            "conv": "",
            "dur_raw": total_raw,
            "dur_human": total_human,
        }
    )

    w = 100
    lines = []
    lines.append("\n" + "=" * w)
    lines.append("PHASE SUMMARY")
    lines.append("=" * w)
    lines.append(
        "{:<30s} {:<10s} {:<18s} {:<25s}".format(
            "Phase", "Iters", "Converged", "Duration"
        )
    )
    lines.append("-" * w)

    for i, r in enumerate(rows):
        if i == len(rows) - 1:
            lines.append("-" * w)
        dur_str = "{:<10s}  ({})".format(r["dur_raw"], r["dur_human"])
        lines.append(
            "{:<30s} {:<10s} {:<18s} {:<25s}".format(
                r["name"],
                r["iters"],
                r["conv"],
                dur_str,
            )
        )

    lines.append("=" * w)

    for line in lines:
        print(line)

    if output_dir:
        output_file = os.path.join(output_dir, "output.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(lines) + "\n")


async def run_pipeline(config, agent_config, resume=False):
    """Run the single-trace analysis pipeline with a shared agent session."""
    output_dir = os.path.abspath(config.output_dir)

    phase_results = []
    all_step_timings = []

    print(f"\nStarting analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    async with agent_session(agent_config) as client:

        # ===================================================================
        # Phase: Step 2 — High-Level Ops (iteration loop)
        # ===================================================================
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("PHASE: HIGH-LEVEL OPS EXTRACTION")
        print("=" * 80)

        skip_review = config.high_level_ops_skip_review
        max_iters = 1 if skip_review else config.num_high_level_ops_iterations
        prev_output_file = None
        prev_output_summary_file = None
        converged = False
        actual_iters = 0

        for i in range(max_iters):
            iteration = i + 1
            actual_iters = iteration

            steps, review_step = gen_high_level_ops_iter_steps(
                config, prev_output_file, prev_output_summary_file, iteration
            )
            run_steps = steps[:-1] if skip_review else steps

            if resume and _all_steps_complete(run_steps, output_dir):
                print(
                    f"Resume: skipping completed iteration 'High-level ops V{iteration}'"
                )
                if not skip_review and check_converged(
                    output_dir, review_step.output_files[1]
                ):
                    converged = True
                    break
                prev_output_file = (
                    review_step.output_files[0]
                    if not skip_review
                    else run_steps[-1].output_files[0]
                )
                prev_output_summary_file = (
                    review_step.output_files[1]
                    if not skip_review
                    else run_steps[-1].output_files[1]
                )
                continue

            timings = await agent_run(
                agent_config, run_steps, resume=resume, client=client
            )
            all_step_timings.extend(timings)

            if skip_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = run_steps[-1].output_files[1]
            else:
                prev_output_file = review_step.output_files[0]
                prev_output_summary_file = review_step.output_files[1]
                if check_converged(output_dir, review_step.output_files[1]):
                    converged = True
                    break

        high_level_ops_file = resolve_output_file(output_dir, HIGH_LEVEL_OPS_FILE)

        phase_results.append(
            {
                "name": "High-level ops",
                "iterations": actual_iters,
                "max_iterations": max_iters,
                "converged": converged,
                "duration": time.time() - phase_start,
            }
        )
        print(
            f"High-level ops: {actual_iters}/{max_iters} iterations, converged={converged}"
        )

        # ===================================================================
        # Phase: Step 3 — GPU Ops Extraction (flat, no iterations)
        # ===================================================================
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("PHASE: GPU OPS EXTRACTION")
        print("=" * 80)

        gpu_ops_txt_file = os.path.join(output_dir, GPU_OPS_TXT_FILE)

        gpu_ops_prompt = GpuOpsPrompt(
            params=config,
            output_txt_file=gpu_ops_txt_file,
        )
        gpu_ops_step = PipelineStep(
            name="gpu_ops_extraction",
            prompt=gpu_ops_prompt.prompt(),
            output_files=[GPU_OPS_TXT_FILE],
        )

        if resume and _all_steps_complete([gpu_ops_step], output_dir):
            print("Resume: skipping completed step 'GPU ops extraction'")
        else:
            timings = await agent_run(
                agent_config, [gpu_ops_step], resume=resume, client=client
            )
            all_step_timings.extend(timings)

        phase_results.append(
            {
                "name": "GPU ops extraction",
                "iterations": 1,
                "max_iterations": 1,
                "converged": False,
                "duration": time.time() - phase_start,
            }
        )

        # ===================================================================
        # Phase: Step 4 — Correlation (iteration loop)
        # ===================================================================
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("PHASE: GPU OPS TO TRANSFORMER BLOCKS CORRELATION")
        print("=" * 80)

        skip_review = config.correlation_skip_review
        max_iters = 1 if skip_review else config.num_correlation_iterations
        prev_output_file = None
        prev_output_summary_file = None
        converged = False
        actual_iters = 0

        for i in range(max_iters):
            iteration = i + 1
            actual_iters = iteration

            steps, review_step = gen_correlation_iter_steps(
                config,
                high_level_ops_file,
                gpu_ops_txt_file,
                prev_output_file,
                prev_output_summary_file,
                iteration,
            )
            run_steps = steps[:-1] if skip_review else steps

            if resume and _all_steps_complete(run_steps, output_dir):
                print(
                    f"Resume: skipping completed iteration 'Correlation V{iteration}'"
                )
                if not skip_review and check_converged(
                    output_dir, review_step.output_files[1]
                ):
                    converged = True
                    break
                prev_output_file = (
                    review_step.output_files[0]
                    if not skip_review
                    else run_steps[-1].output_files[0]
                )
                prev_output_summary_file = (
                    review_step.output_files[1]
                    if not skip_review
                    else run_steps[-1].output_files[1]
                )
                continue

            timings = await agent_run(
                agent_config, run_steps, resume=resume, client=client
            )
            all_step_timings.extend(timings)

            if skip_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = run_steps[-1].output_files[1]
            else:
                prev_output_file = review_step.output_files[0]
                prev_output_summary_file = review_step.output_files[1]
                if check_converged(output_dir, review_step.output_files[1]):
                    converged = True
                    break

        gpu_ops_to_blocks_file = resolve_output_file(output_dir, GPU_OPS_TO_BLOCKS_FILE)
        median_block_file = resolve_output_file(output_dir, MEDIAN_BLOCK_FILE)

        phase_results.append(
            {
                "name": "Correlation",
                "iterations": actual_iters,
                "max_iterations": max_iters,
                "converged": converged,
                "duration": time.time() - phase_start,
            }
        )
        print(
            f"Correlation: {actual_iters}/{max_iters} iterations, converged={converged}"
        )

        # ===================================================================
        # Phase: Step 5 — Performance Analysis (iteration loop, conditional)
        # ===================================================================
        if not config.skip_perf_analysis:
            phase_start = time.time()
            print("\n" + "=" * 80)
            print("PHASE: PERFORMANCE ANALYSIS")
            print("=" * 80)

            skip_review = config.perf_analysis_skip_review
            max_iters = 1 if skip_review else config.num_perf_analysis_iterations
            prev_output_file = None
            prev_output_summary_file = None
            converged = False
            actual_iters = 0

            for i in range(max_iters):
                iteration = i + 1
                actual_iters = iteration

                steps, review_step = gen_perf_analysis_iter_steps(
                    config,
                    median_block_file,
                    high_level_ops_file,
                    gpu_ops_to_blocks_file,
                    gpu_ops_txt_file,
                    prev_output_file,
                    prev_output_summary_file,
                    iteration,
                )
                run_steps = steps[:-1] if skip_review else steps

                if resume and _all_steps_complete(run_steps, output_dir):
                    print(
                        f"Resume: skipping completed iteration 'Perf analysis V{iteration}'"
                    )
                    if not skip_review and check_converged(
                        output_dir, review_step.output_files[1]
                    ):
                        converged = True
                        break
                    prev_output_file = (
                        review_step.output_files[0]
                        if not skip_review
                        else run_steps[-1].output_files[0]
                    )
                    prev_output_summary_file = (
                        review_step.output_files[1]
                        if not skip_review
                        else run_steps[-1].output_files[1]
                    )
                    continue

                timings = await agent_run(
                    agent_config, run_steps, resume=resume, client=client
                )
                all_step_timings.extend(timings)

                if skip_review:
                    prev_output_file = run_steps[-1].output_files[0]
                    prev_output_summary_file = run_steps[-1].output_files[1]
                else:
                    prev_output_file = review_step.output_files[0]
                    prev_output_summary_file = review_step.output_files[1]
                    if check_converged(output_dir, review_step.output_files[1]):
                        converged = True
                        break

            phase_results.append(
                {
                    "name": "Perf analysis",
                    "iterations": actual_iters,
                    "max_iterations": max_iters,
                    "converged": converged,
                    "duration": time.time() - phase_start,
                }
            )
            print(
                f"Perf analysis: {actual_iters}/{max_iters} iterations, converged={converged}"
            )

        # ===================================================================
        # Phase: Step 6 — Chrome Trace (flat, no iterations)
        # ===================================================================
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("PHASE: CHROME TRACE GENERATION")
        print("=" * 80)

        trace_prompt = build_single_trace_json_prompt(config)
        trace_step = PipelineStep(
            name="chrome_trace",
            prompt=trace_prompt.prompt(),
            output_files=[
                os.path.basename(trace_prompt.output_json_file),
                os.path.basename(trace_prompt.output_txt_file),
            ],
        )

        if resume and _all_steps_complete([trace_step], output_dir):
            print("Resume: skipping completed step 'Chrome trace generation'")
        else:
            timings = await agent_run(
                agent_config, [trace_step], resume=resume, client=client
            )
            all_step_timings.extend(timings)

        phase_results.append(
            {
                "name": "Chrome trace",
                "iterations": 1,
                "max_iterations": 1,
                "converged": False,
                "duration": time.time() - phase_start,
            }
        )

    return phase_results, all_step_timings


if __name__ == "__main__":
    setup_logging("single_trace")

    parser = argparse.ArgumentParser(
        description="Run single-trace analysis on a framework execution trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to single-trace config JSON file",
    )
    add_agent_arguments(parser)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean output directory before running (default: clean)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last incomplete step/iteration instead of starting from scratch.",
    )
    args = parser.parse_args()

    config = SingleTraceConfig.from_json(args.config)

    if config.run_command is None:
        config.try_extract_run_command()

    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    if not args.no_clean and not args.resume and os.path.exists(config.output_dir):
        safe_clean_dir(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    config.prepare_source_code()
    config.prepare_trace_file()

    log_config(config)

    print("\nSaving run artifacts...")
    save_run_artifacts(config)

    agent_config = make_agent_config(config, config.output_dir, args.agent_config)
    log_agent_config(agent_config)

    output_dir = os.path.abspath(config.output_dir)

    start_time = time.time()
    phase_results, all_step_timings = asyncio.run(
        run_pipeline(config, agent_config, resume=args.resume)
    )
    total_duration = time.time() - start_time

    print(f"\nAnalysis completed in {total_duration:.1f}s")

    config.cleanup_decompressed_trace()
    config.save_result_metadata()

    # Build output_files dict for log_outputs
    output_files = {
        "high_level_ops": resolve_output_file(output_dir, HIGH_LEVEL_OPS_FILE),
        "gpu_ops_txt": os.path.join(output_dir, GPU_OPS_TXT_FILE),
        "gpu_ops_to_blocks": resolve_output_file(output_dir, GPU_OPS_TO_BLOCKS_FILE),
        "median_block": resolve_output_file(output_dir, MEDIAN_BLOCK_FILE),
    }
    if not config.skip_perf_analysis:
        output_files["perf_analysis"] = resolve_output_file(
            output_dir, PERF_ANALYSIS_SINGLE_FILE
        )
    trace_prompt = build_single_trace_json_prompt(config)
    output_files["chrome_trace_json"] = os.path.basename(trace_prompt.output_json_file)
    output_files["chrome_trace_txt"] = os.path.basename(trace_prompt.output_txt_file)

    # Don't clean versioned files — they should persist for inspection
    log_outputs(config.output_dir, output_files)
    print(f"\nResult metadata saved to: {config.output_dir}/result_metadata.json")

    _print_phase_summary(phase_results, all_step_timings, total_duration, output_dir)
