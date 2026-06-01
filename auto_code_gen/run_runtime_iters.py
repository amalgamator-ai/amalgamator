import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import sys
import time
import argparse
import asyncio

from common.utils import setup_logging
from common.claude_utils import claude_run, PipelineStep

from auto_code_gen.code_gen_configs import CodeGenConfig

from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_IterationHistorySummaryPrompt,
    gen_FindSmallerModelPrompt,
    code_trace_filename,
    CODE_GEN_FILE_PREFIX,
    RUNTIME_FILE_PREFIX,
    RUNTIME_SUCCESS_FILE,
    RUNTIME_LOGS_DIR,
    RUNTIME_SMALLER_MODEL_FILE,
)

from auto_code_gen.run_code_gen import (
    run_runtime_iterations,
    _find_latest_patch,
    _find_latest_runtime_iteration,
    _check_runtime_success,
    _format_duration,
    _format_tokens,
    _print_detailed_table,
    _print_phase_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run runtime iterations (standalone mode). "
        "This is used after run_code_gen.py has completed its code gen iterations."
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
    return parser.parse_args()


def _find_code_trace_files(output_dir, code_gen_config):
    """Find the framework code trace files from previous code gen runs."""
    source_fw = code_gen_config.source_framework
    target_fw = code_gen_config.target_framework

    source_trace = os.path.join(output_dir, code_trace_filename(source_fw))
    target_trace = os.path.join(output_dir, code_trace_filename(target_fw))

    files = [source_trace, target_trace]
    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(
                "Code trace file not found: {}\n"
                "  Run the code generation pipeline (run_code_gen.py) first.".format(f)
            )
    return files


def _find_latest_plan_file(output_dir, prefix):
    """Find the latest review file for a plan prefix (code_port_plan or test_plan)."""
    import re
    pattern = re.compile(r'^{}_review_V(\d+)\.txt$'.format(re.escape(prefix)))
    latest = None
    latest_ver = 0
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            ver = int(m.group(1))
            if ver > latest_ver:
                latest_ver = ver
                latest = f
    return latest


async def run_standalone_runtime_iterations(code_gen_config, claude_config, resume=False):
    output_dir = code_gen_config.output_dir
    context = create_context_str(claude_config, code_gen_config)

    # Verify target branch
    print("Verifying target framework branch...")
    if not code_gen_config.verify_target_branch():
        print("\nERROR: Target repo is not on the expected branch.")
        print("  Expected branch: {}".format(code_gen_config.get_target_branch_name()))
        print("  Please clean the repo and retry.")
        sys.exit(1)

    # Find code trace files
    framework_code_trace_files = _find_code_trace_files(output_dir, code_gen_config)

    # Find the latest code port plan and test plan
    code_port_plan_file = _find_latest_plan_file(output_dir, "code_port_plan")
    if code_port_plan_file is None:
        print("ERROR: No code port plan found in {}".format(output_dir))
        print("  Run the code generation pipeline first.")
        sys.exit(1)

    test_plan_file = _find_latest_plan_file(output_dir, "test_plan")
    if test_plan_file is None:
        print("ERROR: No test plan found in {}".format(output_dir))
        print("  Run the code generation pipeline first.")
        sys.exit(1)

    # Find the latest patch
    latest_patch = _find_latest_patch(output_dir)
    if latest_patch is None:
        print("ERROR: No code patch found in {}".format(output_dir))
        print("  Run the code generation pipeline first.")
        sys.exit(1)

    # Check if already succeeded
    if _check_runtime_success(output_dir):
        print("Runtime iterations already succeeded (found {})".format(
            RUNTIME_SUCCESS_FILE
        ))
        return [], []

    # Determine starting iteration
    last_runtime_iter = _find_latest_runtime_iteration(output_dir)
    start_iteration = last_runtime_iter + 1

    print("\n" + "=" * 80)
    print("STANDALONE RUNTIME ITERATIONS")
    print("=" * 80)
    print("  Output directory: {}".format(output_dir))
    print("  Latest patch: {}".format(latest_patch))
    print("  Code port plan: {}".format(code_port_plan_file))
    print("  Test plan: {}".format(test_plan_file))
    print("  Starting iteration: {}".format(start_iteration))
    print("  Execution command: {}".format(
        code_gen_config.target_run_command[:120]
    ))
    print("=" * 80 + "\n")

    all_step_timings = []

    # Phase 0: Generate iteration history summary (always runs to capture latest state)
    print("Generating iteration history summary...")
    history_prompt = gen_IterationHistorySummaryPrompt(
        context=context,
        code_gen_output_dir=output_dir,
    )
    steps_history = [
        PipelineStep(
            name="iteration_history_summary",
            prompt=history_prompt.prompt(),
            output_files=[history_prompt.output_file],
        ),
    ]

    timings = await claude_run(claude_config, steps_history)
    all_step_timings.extend(timings)

    smaller_model_file = None
    if code_gen_config.use_smaller_model_for_runtime:
        smaller_model_path = os.path.join(output_dir, RUNTIME_SMALLER_MODEL_FILE)
        if resume and os.path.isfile(smaller_model_path):
            print("Resume: reusing existing smaller model selection ({})".format(
                RUNTIME_SMALLER_MODEL_FILE
            ))
            smaller_model_file = RUNTIME_SMALLER_MODEL_FILE
        else:
            print("Finding smaller model for runtime iterations...")
            smaller_prompt = gen_FindSmallerModelPrompt(
                context=context,
                framework_code_trace_files=framework_code_trace_files,
                code_port_plan_file=code_port_plan_file,
            )
            steps_smaller = [
                PipelineStep(
                    name="find_smaller_model",
                    prompt=smaller_prompt.prompt(),
                    output_files=[smaller_prompt.output_file],
                ),
            ]
            timings = await claude_run(claude_config, steps_smaller)
            all_step_timings.extend(timings)
            smaller_model_file = smaller_prompt.output_file

    # Run the runtime iteration loop
    phase_results, runtime_timings = await run_runtime_iterations(
        context, framework_code_trace_files, code_gen_config, claude_config,
        code_port_plan_file, test_plan_file,
        history_prompt.output_file,
        resume=resume, start_iteration=start_iteration,
        smaller_model_file=smaller_model_file,
        disable_new_feature=code_gen_config.disable_new_feature_for_runtime,
    )
    all_step_timings.extend(runtime_timings)

    return phase_results, all_step_timings


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_claude_config()

    setup_logging("runtime_iters")

    os.makedirs(code_gen_config.output_dir, exist_ok=True)

    start_time = time.time()
    phase_results, all_step_timings = asyncio.run(
        run_standalone_runtime_iterations(
            code_gen_config, claude_config, resume=args.resume
        )
    )
    total_duration = time.time() - start_time

    if all_step_timings:
        _print_detailed_table(all_step_timings, total_duration)
    if phase_results:
        _print_phase_summary(phase_results, all_step_timings, total_duration)

    if _check_runtime_success(code_gen_config.output_dir):
        print("\nRUNTIME ITERATIONS COMPLETED SUCCESSFULLY")
        success_file = os.path.join(
            code_gen_config.output_dir, RUNTIME_SUCCESS_FILE
        )
        print("Results: {}".format(success_file))
    else:
        print("\nRUNTIME ITERATIONS DID NOT SUCCEED")
        print("Check the logs and summaries in: {}".format(
            code_gen_config.output_dir
        ))
