# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Create Cross-Trace Analysis Config
====================================

Helper script that generates a cross-trace analysis config JSON file
from the provided parameters. Each trace references a single-trace
analysis result directory produced by run_single_trace.py.

Supports both analysis modes (auto-detected from the traces):
  - Cross-framework: comparing different frameworks running the same model
    (e.g. vLLM vs SGLang)
  - Cross-commit: comparing different commits of the same framework to
    identify performance differences between versions

Usage:
    python -m llm.auto_analyze.create_cross_trace_config \
        --trace-result-dir /path/to/llm/auto_analyze/results/results_analyze_deepseek_r1_nvfp4/test_results/test_1/vllm \
        --trace-result-dir /path/to/llm/auto_analyze/results/results_analyze_deepseek_r1_nvfp4/test_results/test_1/sglang \
        --target-trace-id 0 \
        --analyze-output-dir /path/to/llm/auto_analyze/results/results_analyze_deepseek_r1_nvfp4/test_results/test_1/cross_trace \
        --output-config-file /path/to/cross_config

    The output file path should NOT include the .json extension — it is
    added automatically.

Required parameters:
    --trace-result-dir      Path to a single-trace result directory (repeat for
                            each trace; at least 2 required). Each directory must
                            contain run_params.txt from a completed single-trace
                            analysis.
    --target-trace-id       Index (0-based) of the target trace to optimize/fix
    --analyze-output-dir    Directory where cross-trace analysis outputs will be stored
    --output-config-file    Output path for the config JSON (without .json suffix)

Advanced parameters:
    --make-improvement-plan  Generate an improvement plan for the target trace
                             with step-by-step coding guides (default: disabled)
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import json
import os
import sys

from common.iteration_utils import resolve_output_file
from llm.auto_analyze.configs.single_trace_config import (
    HIGH_LEVEL_OPS_FILE,
    MEDIAN_BLOCK_FILE,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a cross-trace analysis config JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    required = parser.add_argument_group("required parameters")
    required.add_argument(
        "--trace-result-dir",
        required=True,
        action="append",
        dest="trace_result_dirs",
        help="Path to a single-trace result directory "
        "(repeat for each trace; at least 2 required)",
    )
    required.add_argument(
        "--target-trace-id",
        required=True,
        type=int,
        help="Index (0-based) of the target trace to optimize/fix",
    )
    required.add_argument(
        "--analyze-output-dir",
        required=True,
        help="Directory where cross-trace analysis outputs will be stored",
    )
    required.add_argument(
        "--output-config-file",
        required=True,
        help="Output path for the config JSON (without .json suffix)",
    )

    advanced = parser.add_argument_group("advanced parameters")
    advanced.add_argument(
        "--make-improvement-plan",
        action="store_true",
        default=False,
        help="Generate an improvement plan for the target trace "
        "with step-by-step coding guides (default: disabled)",
    )

    args = parser.parse_args()

    errors = []

    # Validate trace count
    if len(args.trace_result_dirs) < 2:
        errors.append("at least 2 --trace-result-dir entries are required")

    # Validate each trace result directory
    for i, result_dir in enumerate(args.trace_result_dirs):
        if not os.path.isdir(result_dir):
            errors.append(f"trace [{i}] result directory not found: {result_dir}")
            continue

        params_file = os.path.join(result_dir, "run_params.txt")
        if not os.path.exists(params_file):
            errors.append(f"trace [{i}] missing run_params.txt: {params_file}")

        median_file = resolve_output_file(result_dir, MEDIAN_BLOCK_FILE)
        if not os.path.exists(median_file):
            errors.append(f"trace [{i}] missing {MEDIAN_BLOCK_FILE}: {median_file}")

        high_level_file = resolve_output_file(result_dir, HIGH_LEVEL_OPS_FILE)
        if not os.path.exists(high_level_file):
            errors.append(
                f"trace [{i}] missing {HIGH_LEVEL_OPS_FILE}: {high_level_file}"
            )

    # Validate target trace ID
    if args.target_trace_id < 0 or args.target_trace_id >= len(args.trace_result_dirs):
        errors.append(
            f"target-trace-id {args.target_trace_id} is out of range "
            f"(0..{len(args.trace_result_dirs) - 1})"
        )

    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # --- Print summary ---
    print(f"  Traces: {len(args.trace_result_dirs)}")
    for i, result_dir in enumerate(args.trace_result_dirs):
        marker = " (TARGET)" if i == args.target_trace_id else ""
        print(f"    [{i}]{marker} {result_dir}")
    print(
        f"  Improvement plan: {'enabled' if args.make_improvement_plan else 'disabled'}"
    )

    # --- Resolve paths ---
    output_json = args.output_config_file
    if not output_json.endswith(".json"):
        output_json += ".json"

    # --- Build config ---
    config = {
        "traces": [{"result_dir": os.path.abspath(d)} for d in args.trace_result_dirs],
        "target_trace_id": args.target_trace_id,
        "output_dir": os.path.abspath(args.analyze_output_dir),
        "make_improvement_plan": args.make_improvement_plan,
    }

    # --- Write config ---
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(config, f, indent=4)

    print()
    print(f"Config written to: {output_json}")
    print()
    print("=" * 70)
    print("  CROSS-TRACE CONFIG")
    print("=" * 70)
    for i, t in enumerate(config["traces"]):
        marker = " (TARGET)" if i == config["target_trace_id"] else ""
        print(f"  trace [{i}]{marker}: {t['result_dir']}")
    print(f"  target_trace_id: {config['target_trace_id']}")
    print(f"  output_dir: {config['output_dir']}")
    print(f"  make_improvement_plan: {config['make_improvement_plan']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
