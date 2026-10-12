"""
Cross-Trace Analysis Entry Point
=================================

Compares multiple single-trace analysis results to find performance
differences and generate an improvement plan for the target trace.

Automatically infers analysis mode from the traces:
  - cross-framework:  Traces from different frameworks (e.g., vLLM vs SGLang)
  - cross-commit:     Traces from the same framework, different commits

Usage:
    python -m llm.auto_analyze.run_cross_trace --config <config.json> [--no-clean] [--resume]
    python -m llm.auto_analyze.run_cross_trace --config <config.json> --agent-config <agent.json>

Arguments:
    --config          Path to cross-trace config JSON (required)
    --agent-config    Optional provider-neutral agent config JSON
    --no-clean        Do not clean output directory before running
    --resume          Resume from last incomplete step instead of starting from scratch

Config JSON fields:
    traces                  List of objects, each with "result_dir" pointing to a single-trace output
    target_trace_id         Index into traces list (the trace to optimize/fix)
    output_dir              Output directory for cross-trace results
    make_improvement_plan   If true, generate improvement plan for target trace (default: false)

    Iteration control (all optional):
    num_matching_iterations          Max gen=>review iterations for matching (default: 3)
    num_compare_iterations           Max gen=>review iterations for comparison (default: 3)
    num_improvement_plan_iterations  Max gen=>review iterations for improvement plan (default: 3)
    matching_skip_review             Skip review for matching (default: false)
    compare_skip_review              Skip review for comparison (default: false)
    improvement_plan_skip_review     Skip review for improvement plan (default: false)

    All other parameters (framework name, model, GPU, commit ID, execution params)
    are inferred from each trace's run_params.txt file. The traces must share the
    same model, GPU type, batch size, prefill size, and output size.

Outputs (in output_dir):
    Always produced:
      - cross_matching_blocks.txt      Operation-by-operation matching across block traces
      - cross_compare_blocks.txt       Performance comparison across block traces
      - run_params_cross.txt           Cross-trace run parameters for downstream tools

    When make_improvement_plan is true:
      - cross_improvement_plan.txt     Improvement plan for target trace with coding guides
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import asyncio
import os
import sys
import time

from common.claude_utils import _all_steps_complete
from common.iteration_utils import (
    check_converged as _check_converged,
)
from common.iteration_utils import (
    resolve_output_file,
)
from common.utils import safe_clean_dir, setup_logging
from llm.agents import agent_run, agent_session
from llm.auto_analyze.agent_support import (
    add_agent_arguments,
    log_agent_config,
    make_agent_config,
)
from llm.auto_analyze.configs.cross_trace_config import (
    CROSS_COMPARE_FILE,
    CROSS_IMPROVEMENT_FILE,
    CROSS_MATCHING_FILE,
    CrossTraceConfig,
)
from llm.auto_analyze.prompts.cross_trace_prompts import (
    gen_compare_iter_steps,
    gen_improvement_plan_iter_steps,
    gen_matching_iter_steps,
)


def log_config(config: CrossTraceConfig):
    analysis_type = config.infer_analysis_type()
    mode_label = (
        "Cross-Framework Comparison"
        if analysis_type == "cross-framework"
        else "Same-Framework Cross-Commit Analysis"
    )

    target = config.get_target_result()

    print("=" * 70)
    print("  CROSS-TRACE ANALYSIS")
    print("=" * 70)
    print(f"  Mode:          {mode_label}")
    print(f"  Model:         {target.model}")
    print(f"  GPU:           {target.gpu_type}")
    print(f"  Target:        [{config.target_trace_id}] {target.trace_id}")
    print(f"  Output dir:    {config.output_dir}")
    print()
    print("  Traces:")
    for i, tr in enumerate(config.traces):
        marker = " (*)" if i == config.target_trace_id else ""
        print(f"    [{i}] {tr.trace_id}{marker}")
        print(f"        Framework:   {tr.framework_name}")
        print(f"        Commit:      {tr.commit_id[:12] if tr.commit_id else 'N/A'}")
        print(f"        Source code: {tr.framework_source_code}")
        print(f"        Result dir:  {tr.result_dir}")
        print(
            f"        Exec params: BS={tr.batch_size_range} ISL={tr.prefill_size_range} OSL={tr.output_size_range}"
        )

    print()
    print("  Iteration settings:")
    print(
        f"    Matching:         max_iters={config.num_matching_iterations}, skip_review={config.matching_skip_review}"
    )
    print(
        f"    Compare:          max_iters={config.num_compare_iterations}, skip_review={config.compare_skip_review}"
    )
    print(
        f"    Improvement plan: max_iters={config.num_improvement_plan_iterations}, skip_review={config.improvement_plan_skip_review}"
    )
    print("=" * 70)


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


def _print_phase_summary(phase_results, total_duration):
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

    max_dur_len = max(len(r["dur_raw"]) for r in rows)
    w = 80

    lines = []
    lines.append("\n" + "=" * w)
    lines.append("PHASE SUMMARY")
    lines.append("=" * w)
    lines.append(
        "{:<25s} {:<10s} {:<16s} {}".format("Phase", "Iters", "Converged", "Duration")
    )
    lines.append("-" * w)

    for i, r in enumerate(rows):
        if i == len(rows) - 1:
            lines.append("-" * w)
        dur_str = "{:<{}}  ({})".format(r["dur_raw"], max_dur_len, r["dur_human"])
        lines.append(
            "{:<25s} {:<10s} {:<16s} {}".format(
                r["name"],
                r["iters"],
                r["conv"],
                dur_str,
            )
        )

    lines.append("=" * w)

    print("\n".join(lines))


async def run_pipeline(config, agent_config, resume=False):
    """Run the cross-trace analysis pipeline with gen=>review iteration loops."""
    output_dir = os.path.abspath(config.output_dir)
    traces_context = (
        ""  # passed to gen_*_iter_steps but unused (prompts build it internally)
    )

    async with agent_session(agent_config) as client:
        return await _run_pipeline_with_client(
            config,
            agent_config,
            client,
            output_dir,
            traces_context,
            resume,
        )


async def _run_pipeline_with_client(
    config, agent_config, client, output_dir, traces_context, resume
):
    phase_results = []

    # ------------------------------------------------------------------
    # Phase 1: Matching iterations
    # ------------------------------------------------------------------
    phase_start = time.time()
    skip_review = config.matching_skip_review
    max_iters = 1 if skip_review else config.num_matching_iterations
    prev_output_file = None
    prev_output_summary_file = None
    converged = False
    actual_iters = 0

    print("\n" + "=" * 70)
    print("PHASE 1: MATCHING MEDIAN TRANSFORMER BLOCKS")
    print("=" * 70)

    for i in range(max_iters):
        iteration = i + 1
        actual_iters = iteration
        steps, review_prompt = gen_matching_iter_steps(
            config,
            traces_context,
            prev_output_file,
            prev_output_summary_file,
            iteration,
        )
        run_steps = steps[:-1] if skip_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print(f"Resume: skipping completed iteration 'Matching V{iteration}'")
            if skip_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1
                    else None
                )
            else:
                prev_output_file = review_prompt.output_file
                prev_output_summary_file = review_prompt.output_summary_file
                if _check_converged(
                    output_dir, os.path.basename(review_prompt.output_summary_file)
                ):
                    print(f"Matching CONVERGED at iteration {iteration}")
                    converged = True
                    break
            continue

        await agent_run(agent_config, run_steps, resume=resume, client=client)

        if skip_review:
            prev_output_file = run_steps[-1].output_files[0]
            prev_output_summary_file = (
                run_steps[-1].output_files[1]
                if len(run_steps[-1].output_files) > 1
                else None
            )
        else:
            prev_output_file = review_prompt.output_file
            prev_output_summary_file = review_prompt.output_summary_file
            if _check_converged(
                output_dir, os.path.basename(review_prompt.output_summary_file)
            ):
                print(f"Matching CONVERGED at iteration {iteration}")
                converged = True
                break

    phase_results.append(
        {
            "name": "Matching",
            "iterations": actual_iters,
            "max_iterations": max_iters,
            "converged": converged,
            "duration": time.time() - phase_start,
        }
    )

    matching_file = resolve_output_file(output_dir, CROSS_MATCHING_FILE)

    # ------------------------------------------------------------------
    # Phase 2: Compare iterations
    # ------------------------------------------------------------------
    phase_start = time.time()
    skip_review = config.compare_skip_review
    max_iters = 1 if skip_review else config.num_compare_iterations
    prev_output_file = None
    prev_output_summary_file = None
    converged = False
    actual_iters = 0

    print("\n" + "=" * 70)
    print("PHASE 2: COMPARING PERFORMANCE ACROSS TRACES")
    print("=" * 70)

    for i in range(max_iters):
        iteration = i + 1
        actual_iters = iteration
        steps, review_prompt = gen_compare_iter_steps(
            config,
            traces_context,
            matching_file,
            prev_output_file,
            prev_output_summary_file,
            iteration,
        )
        run_steps = steps[:-1] if skip_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print(f"Resume: skipping completed iteration 'Compare V{iteration}'")
            if skip_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1
                    else None
                )
            else:
                prev_output_file = review_prompt.output_file
                prev_output_summary_file = review_prompt.output_summary_file
                if _check_converged(
                    output_dir, os.path.basename(review_prompt.output_summary_file)
                ):
                    print(f"Compare CONVERGED at iteration {iteration}")
                    converged = True
                    break
            continue

        await agent_run(agent_config, run_steps, resume=resume, client=client)

        if skip_review:
            prev_output_file = run_steps[-1].output_files[0]
            prev_output_summary_file = (
                run_steps[-1].output_files[1]
                if len(run_steps[-1].output_files) > 1
                else None
            )
        else:
            prev_output_file = review_prompt.output_file
            prev_output_summary_file = review_prompt.output_summary_file
            if _check_converged(
                output_dir, os.path.basename(review_prompt.output_summary_file)
            ):
                print(f"Compare CONVERGED at iteration {iteration}")
                converged = True
                break

    phase_results.append(
        {
            "name": "Compare",
            "iterations": actual_iters,
            "max_iterations": max_iters,
            "converged": converged,
            "duration": time.time() - phase_start,
        }
    )

    compare_file = resolve_output_file(output_dir, CROSS_COMPARE_FILE)

    # ------------------------------------------------------------------
    # Phase 3: Improvement plan iterations (conditional)
    # ------------------------------------------------------------------
    if config.make_improvement_plan:
        phase_start = time.time()
        skip_review = config.improvement_plan_skip_review
        max_iters = 1 if skip_review else config.num_improvement_plan_iterations
        prev_output_file = None
        prev_output_summary_file = None
        converged = False
        actual_iters = 0

        print("\n" + "=" * 70)
        print("PHASE 3: GENERATING IMPROVEMENT PLAN")
        print("=" * 70)

        for i in range(max_iters):
            iteration = i + 1
            actual_iters = iteration
            steps, review_prompt = gen_improvement_plan_iter_steps(
                config,
                traces_context,
                compare_file,
                prev_output_file,
                prev_output_summary_file,
                iteration,
            )
            run_steps = steps[:-1] if skip_review else steps

            if resume and _all_steps_complete(run_steps, output_dir):
                print(
                    f"Resume: skipping completed iteration 'Improvement plan V{iteration}'"
                )
                if skip_review:
                    prev_output_file = run_steps[-1].output_files[0]
                    prev_output_summary_file = (
                        run_steps[-1].output_files[1]
                        if len(run_steps[-1].output_files) > 1
                        else None
                    )
                else:
                    prev_output_file = review_prompt.output_file
                    prev_output_summary_file = review_prompt.output_summary_file
                    if _check_converged(
                        output_dir, os.path.basename(review_prompt.output_summary_file)
                    ):
                        print(f"Improvement plan CONVERGED at iteration {iteration}")
                        converged = True
                        break
                continue

            await agent_run(agent_config, run_steps, resume=resume, client=client)

            if skip_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1
                    else None
                )
            else:
                prev_output_file = review_prompt.output_file
                prev_output_summary_file = review_prompt.output_summary_file
                if _check_converged(
                    output_dir, os.path.basename(review_prompt.output_summary_file)
                ):
                    print(f"Improvement plan CONVERGED at iteration {iteration}")
                    converged = True
                    break

        phase_results.append(
            {
                "name": "Improvement plan",
                "iterations": actual_iters,
                "max_iterations": max_iters,
                "converged": converged,
                "duration": time.time() - phase_start,
            }
        )

    return phase_results


if __name__ == "__main__":
    setup_logging("cross_trace")

    parser = argparse.ArgumentParser(
        description="Run cross-trace analysis comparing multiple single-trace results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to cross-trace config JSON file",
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
        help="Resume from last incomplete step instead of starting from scratch",
    )
    args = parser.parse_args()

    config = CrossTraceConfig.from_json(args.config)

    # Load parameters from each trace's result directory
    config.load_all_trace_params()

    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    if not args.no_clean and not args.resume and os.path.exists(config.output_dir):
        safe_clean_dir(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    log_config(config)

    print("\nSaving cross-trace run parameters...")
    config.save_run_params()

    agent_config = make_agent_config(config, config.output_dir, args.agent_config)
    log_agent_config(agent_config)

    start_time = time.time()
    print(f"\nStarting analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    phase_results = asyncio.run(run_pipeline(config, agent_config, resume=args.resume))

    total_duration = time.time() - start_time
    mins, secs = divmod(int(total_duration), 60)
    hrs, mins = divmod(mins, 60)
    human_time = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
    print(f"\nAnalysis completed in {total_duration:.1f}s ({human_time})")

    output_files = {
        "cross_matching": resolve_output_file(config.output_dir, CROSS_MATCHING_FILE),
        "cross_compare": resolve_output_file(config.output_dir, CROSS_COMPARE_FILE),
    }
    if config.make_improvement_plan:
        output_files["cross_improvement"] = resolve_output_file(
            config.output_dir, CROSS_IMPROVEMENT_FILE
        )

    log_outputs(config.output_dir, output_files)
    _print_phase_summary(phase_results, total_duration)
