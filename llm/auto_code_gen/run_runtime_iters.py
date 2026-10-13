# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run runtime iterations for a specific model configuration.

Standalone entry point for runtime iterations after the code gen pipeline
has produced a patch. By default, runs with the original benchmarked model.
Can also run with a smaller auto-detected model, or a specific model from
additional_benchmark_configs.

All runtime results (logs, success file, patches) go into the
runtime/ subdirectory inside output_dir.

Usage:

    # Default: run with original model
    python -m llm.auto_code_gen.run_runtime_iters --config <config.json>

    # Run with auto-detected smaller model
    python -m llm.auto_code_gen.run_runtime_iters --config <config.json> --use-smaller-model

    # Run with a specific additional model by index (0-based)
    python -m llm.auto_code_gen.run_runtime_iters --config <config.json> --model-index 0

    # Run ALL models (original + smaller if enabled + all additional)
    python -m llm.auto_code_gen.run_runtime_iters --config <config.json> --all-models

    # Resume from last incomplete iteration
    python -m llm.auto_code_gen.run_runtime_iters --config <config.json> --resume
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

from common.utils import setup_logging
from llm.auto_code_gen.code_gen_configs import CodeGenConfig
from llm.auto_code_gen.code_gen_prompts import (
    CODE_PORT_PLAN_DIR,
    RUNTIME_DIR,
    RUNTIME_RESULTS_MANIFEST,
    TEST_PLAN_DIR,
    code_trace_filename,
)
from llm.auto_code_gen.run_code_gen import (
    _check_runtime_success,
    _print_detailed_table,
    _print_phase_summary,
    _print_timeline_and_costs,
    run_record_learnings,
)
from llm.auto_code_gen.use_cases.llm_framework import LLMFrameworkUseCase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run runtime iterations (standalone mode). "
        "Runs after run_code_gen.py has produced a code patch."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last incomplete runtime iteration.",
    )

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--use-smaller-model",
        action="store_true",
        help="Run with an auto-detected smaller model instead of the original.",
    )
    model_group.add_argument(
        "--model-index",
        type=int,
        default=None,
        help="Run a specific additional model by index (0-based from additional_benchmark_configs).",
    )
    model_group.add_argument(
        "--all-models",
        action="store_true",
        help="Run ALL models: original, smaller (if enabled), and all additional.",
    )
    from llm.auto_code_gen.agents import add_agent_argument

    add_agent_argument(parser)
    return parser.parse_args()


def _build_model_list(args, config, use_case):
    """Build the list of model configs to run based on CLI args."""
    models = []

    if args.all_models:
        # Original model
        models.append(
            {
                "model": config.model,
                "label": "original",
                "execution_command": config.target_run_command,
                "smaller_model_file": None,
                "disable_new_feature": config.disable_new_feature_for_runtime,
            }
        )
        # Smaller model (if enabled in config)
        if config.use_smaller_model_for_runtime:
            models.append(
                {
                    "model": "smaller",
                    "label": "smaller",
                    "execution_command": config.target_run_command,
                    "smaller_model_file": "__pending__",
                    "disable_new_feature": config.disable_new_feature_for_runtime,
                }
            )
        # All additional models
        for bench_cfg in config.additional_benchmark_configs:
            model = bench_cfg.get("model", "")
            num_gpus = bench_cfg.get("num_gpus", 8)
            label = bench_cfg.get("label", model)
            exec_cmd = bench_cfg.get("run_command", "")
            if not exec_cmd:
                exec_cmd = use_case._derive_run_command(
                    config.target_run_command, model, num_gpus
                )
            models.append(
                {
                    "model": model,
                    "label": label,
                    "execution_command": exec_cmd,
                    "smaller_model_file": None,
                    "disable_new_feature": config.disable_new_feature_for_runtime,
                }
            )

    elif args.use_smaller_model:
        models.append(
            {
                "model": "smaller",
                "label": "smaller",
                "execution_command": config.target_run_command,
                "smaller_model_file": "__pending__",
                "disable_new_feature": config.disable_new_feature_for_runtime,
            }
        )

    elif args.model_index is not None:
        idx = args.model_index
        if idx < 0 or idx >= len(config.additional_benchmark_configs):
            print(
                f"ERROR: model-index {idx} out of range (0-{len(config.additional_benchmark_configs) - 1})"
            )
            sys.exit(1)
        bench_cfg = config.additional_benchmark_configs[idx]
        model = bench_cfg.get("model", "")
        num_gpus = bench_cfg.get("num_gpus", 8)
        label = bench_cfg.get("label", model)
        exec_cmd = bench_cfg.get("run_command", "")
        if not exec_cmd:
            exec_cmd = use_case._derive_run_command(
                config.target_run_command, model, num_gpus
            )
        models.append(
            {
                "model": model,
                "label": label,
                "execution_command": exec_cmd,
                "smaller_model_file": None,
                "disable_new_feature": config.disable_new_feature_for_runtime,
            }
        )

    else:
        # Default: original model
        models.append(
            {
                "model": config.model,
                "label": "original",
                "execution_command": config.target_run_command,
                "smaller_model_file": None,
                "disable_new_feature": config.disable_new_feature_for_runtime,
            }
        )

    return models


def _find_code_trace_files(output_dir, code_gen_config):
    source_fw = code_gen_config.source_framework
    target_fw = code_gen_config.target_framework
    files = [
        os.path.join(output_dir, code_trace_filename(source_fw)),
        os.path.join(output_dir, code_trace_filename(target_fw)),
    ]
    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"Code trace file not found: {f}\n"
                "  Run the code generation pipeline (run_code_gen.py) first."
            )
    return files


def _find_latest_plan_file(output_dir, subdir):
    """Find the latest plan file in a phase subdirectory.

    Looks for V{N}_review.txt first (preferred), then V{N}.txt as fallback.
    Returns a relative path like 'code_port_plan/V2_review.txt', or None.
    """
    import re as _re

    phase_dir = os.path.join(output_dir, subdir)
    if not os.path.isdir(phase_dir):
        return None
    review_pat = _re.compile(r"^V(\d+)_review\.txt$")
    plain_pat = _re.compile(r"^V(\d+)\.txt$")
    latest = None
    latest_ver = 0
    for f in os.listdir(phase_dir):
        m = review_pat.match(f)
        if m:
            ver = int(m.group(1))
            if ver > latest_ver:
                latest_ver = ver
                latest = os.path.join(subdir, f)
    if latest is None:
        for f in os.listdir(phase_dir):
            m = plain_pat.match(f)
            if m:
                ver = int(m.group(1))
                if ver > latest_ver:
                    latest_ver = ver
                    latest = os.path.join(subdir, f)
    return latest


async def run_standalone_runtime_iterations(
    code_gen_config, claude_config, resume=False, run_models=None, client=None
):
    use_case = LLMFrameworkUseCase()
    context = use_case.create_context_str(claude_config, code_gen_config)
    output_dir = code_gen_config.output_dir

    # Verify target branch
    print("Verifying target framework branch...")
    if not code_gen_config.verify_target_branch():
        print("\nERROR: Target repo is not on the expected branch.")
        print(f"  Expected branch: {code_gen_config.get_target_branch_name()}")
        print("  Please clean the repo and retry.")
        sys.exit(1)

    # Find code trace files from previous pipeline run
    code_trace_files = _find_code_trace_files(output_dir, code_gen_config)

    # Find latest plan files
    code_port_plan_file = _find_latest_plan_file(output_dir, CODE_PORT_PLAN_DIR) or ""
    test_plan_file = _find_latest_plan_file(output_dir, TEST_PLAN_DIR) or ""

    phase_results, all_step_timings = await use_case.run_runtime_iterations(
        context,
        code_gen_config,
        claude_config,
        code_trace_files=code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        resume=resume,
        run_models=run_models,
        client=client,
    )

    return phase_results, all_step_timings


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_agent_config(args.agent_config)
    use_case = LLMFrameworkUseCase()

    setup_logging("runtime_iters")

    os.makedirs(code_gen_config.output_dir, exist_ok=True)

    run_models = _build_model_list(args, code_gen_config, use_case)

    print("\n" + "=" * 80)
    print("STANDALONE RUNTIME ITERATIONS")
    print("=" * 80)
    print(f"  Output directory: {code_gen_config.output_dir}")
    print(f"  Models to run: {len(run_models)}")
    for m in run_models:
        print("    - {} ({})".format(m["label"], m["model"]))
    print("=" * 80 + "\n")

    async def _main_with_session():
        from llm.auto_code_gen.agents import agent_session

        async with agent_session(claude_config) as client:
            results = await run_standalone_runtime_iterations(
                code_gen_config,
                claude_config,
                resume=args.resume,
                run_models=run_models,
                client=client,
            )
            # Record learnings within the same session
            try:
                await run_record_learnings(
                    code_gen_config.output_dir, claude_config, client=client
                )
            except Exception as e:
                print(f"\nWarning: Could not record learnings: {e}")
            return results

    start_time = time.time()
    phase_results, all_step_timings = asyncio.run(_main_with_session())
    total_duration = time.time() - start_time

    if all_step_timings:
        _print_detailed_table(all_step_timings, total_duration)
    if phase_results:
        _print_phase_summary(phase_results, all_step_timings, total_duration)
    if all_step_timings:
        _print_timeline_and_costs(
            all_step_timings,
            total_duration,
            start_time,
            code_gen_config.output_dir,
        )

    # Check results per model
    runtime_dir = os.path.join(code_gen_config.output_dir, RUNTIME_DIR)
    for m in run_models:
        if _check_runtime_success(runtime_dir):
            print("\n  {} — SUCCEEDED".format(m["label"]))
        else:
            print("\n  {} — DID NOT SUCCEED".format(m["label"]))

    manifest_path = os.path.join(
        code_gen_config.output_dir, RUNTIME_DIR, RUNTIME_RESULTS_MANIFEST
    )
    if os.path.isfile(manifest_path):
        print(f"\nResults manifest: {manifest_path}")
