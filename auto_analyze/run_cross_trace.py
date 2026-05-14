"""
Cross-Trace Analysis Entry Point
=================================

Compares multiple single-trace analysis results to find performance
differences and generate an improvement plan for the target trace.

Automatically infers analysis mode from the traces:
  - cross-framework:  Traces from different frameworks (e.g., vLLM vs SGLang)
  - cross-commit:     Traces from the same framework, different commits

Usage:
    python -m auto_analyze.run_cross_trace --config <config.json> [--no-clean]
    python -m auto_analyze.run_cross_trace --config <config.json> --claude-config <claude.json>

Arguments:
    --config          Path to cross-trace config JSON (required)
    --claude-config   Path to Claude config JSON (default: auto_analyze/configs/claude_config.json)
    --no-clean        Do not clean output directory before running

Config JSON fields:
    traces                  List of objects, each with "result_dir" pointing to a single-trace output
    target_trace_id         Index into traces list (the trace to optimize/fix)
    output_dir              Output directory for cross-trace results
    make_improvement_plan   If true, generate improvement plan for target trace (default: false)

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

import sys as _sys
import os as _os
_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import json
import os
import sys
import time
import asyncio
import argparse

from common.utils import setup_logging, safe_clean_dir, clean_output_dir
from common.claude_utils import claude_run, ClaudeConfig

from auto_analyze.configs.cross_trace_config import CrossTraceConfig
from auto_analyze.prompts.cross_trace_prompts import gen_cross_trace_prompts


def load_claude_config(claude_config_path, output_dir):
    with open(claude_config_path) as f:
        data = json.load(f)
    return ClaudeConfig(
        model=data["model"],
        allowed_tools=data["allowed_tools"],
        perm_mode="acceptEdits",
        cwd=output_dir,
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
        print(f"        Exec params: BS={tr.batch_size_range} ISL={tr.prefill_size_range} OSL={tr.output_size_range}")
    print("=" * 70)


def log_outputs(output_dir, output_files):
    print()
    print("=" * 70)
    print("  OUTPUT FILES")
    print("=" * 70)
    for label, filename in output_files.items():
        path = filename if os.path.isabs(filename) else os.path.join(output_dir, filename)
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {filename}")
    print("=" * 70)


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
    parser.add_argument(
        "--claude-config",
        type=str,
        default="auto_analyze/configs/claude_config.json",
        help="Path to Claude config JSON file (default: auto_analyze/configs/claude_config.json)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean output directory before running (default: clean)",
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

    if not args.no_clean and os.path.exists(config.output_dir):
        safe_clean_dir(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    log_config(config)

    print("\nSaving cross-trace run parameters...")
    config.save_run_params()

    claude_config = load_claude_config(args.claude_config, config.output_dir)
    prompts, output_files = gen_cross_trace_prompts(config)

    start_time = time.time()
    print(f"\nStarting analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {len([p for p in prompts if isinstance(p, str)])} analysis steps...\n")

    asyncio.run(claude_run(claude_config, prompts))

    duration = time.time() - start_time
    mins, secs = divmod(int(duration), 60)
    hrs, mins = divmod(mins, 60)
    human_time = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
    print(f"\nAnalysis completed in {duration:.1f}s ({human_time})")

    clean_output_dir(config.output_dir, output_files)
    log_outputs(config.output_dir, output_files)
