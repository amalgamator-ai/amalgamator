import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import re
import sys
import time
import argparse
import asyncio

from common.utils import setup_logging, safe_clean_dir, clear_vllm_source_tree
from common.claude_utils import claude_run, PipelineStep, _all_steps_complete

from auto_code_gen.code_gen_configs import load_config_and_use_case

from auto_code_gen.code_gen_prompts import (
    gen_ApplyCodeAndCompilePrompt,
    gen_RunAndLogPrompt,
    gen_InvestigateRuntimeOutputAndFixCodePrompt,
    gen_RunLMEvalPrompt,
    RUNTIME_FILE_PREFIX,
    RUNTIME_SUCCESS_FILE,
    RUNTIME_LOGS_DIR,
)


CONVERGENCE_MARKER = "CONVERGED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI code generation pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last incomplete step instead of starting from scratch.",
    )
    return parser.parse_args()


def _check_converged(output_dir, summary_file):
    path = os.path.join(output_dir, summary_file)
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        first_line = f.readline().strip()
    return CONVERGENCE_MARKER in first_line


def _find_latest_patch(output_dir):
    """Find the latest code patch file from code gen or runtime iterations."""
    _runtime_pat = re.compile(r'^code_gen_runtime_V(\d+)\.patch$')
    _codegen_pat = re.compile(r'^code_gen_review_V(\d+)\.patch$')

    latest_runtime = None
    latest_runtime_ver = 0
    latest_codegen = None
    latest_codegen_ver = 0

    for f in os.listdir(output_dir):
        m = _runtime_pat.match(f)
        if m:
            ver = int(m.group(1))
            if ver > latest_runtime_ver:
                latest_runtime_ver = ver
                latest_runtime = f
        m = _codegen_pat.match(f)
        if m:
            ver = int(m.group(1))
            if ver > latest_codegen_ver:
                latest_codegen_ver = ver
                latest_codegen = f

    if latest_runtime:
        return latest_runtime
    return latest_codegen


def _find_latest_runtime_iteration(output_dir):
    """Find the latest runtime iteration number (0 if none exist)."""
    pattern = re.compile(r'^code_gen_runtime_V(\d+)\.patch$')
    max_ver = 0
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            ver = int(m.group(1))
            if ver > max_ver:
                max_ver = ver
    return max_ver


def _check_apply_succeeded(output_dir, iteration):
    """Check if apply and compile succeeded for the given runtime iteration."""
    status_file = os.path.join(
        output_dir, "runtime_apply_status_V{}.txt".format(iteration)
    )
    if not os.path.isfile(status_file):
        return False
    with open(status_file) as f:
        status = f.read().strip()
    return status == "SUCCESS"


def _check_runtime_success(output_dir):
    """Check if runtime iteration resulted in a successful benchmark run."""
    return os.path.isfile(os.path.join(output_dir, RUNTIME_SUCCESS_FILE))


def _check_run_succeeded(runtime_logs_dir, iteration):
    """Check if the RunAndLog patched run completed successfully."""
    status_file = os.path.join(
        runtime_logs_dir, "runtime_run_status_V{}.txt".format(iteration)
    )
    if not os.path.isfile(status_file):
        return False
    with open(status_file) as f:
        return f.read().strip() == "COMPLETED"




def _format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    raw = "{:.1f}s".format(seconds)
    if hrs:
        human = "{}h {}m {}s".format(hrs, mins, secs)
    else:
        human = "{}m {}s".format(mins, secs)
    return raw, human


def _format_tokens(count):
    if count >= 1_000_000:
        return "{:.1f}M".format(count / 1_000_000)
    if count >= 1_000:
        return "{:.1f}K".format(count / 1_000)
    return str(count)


def _print_detailed_table(all_step_timings, total_duration):
    total_input = sum(s["input_tokens"] for s in all_step_timings)
    total_output = sum(s["output_tokens"] for s in all_step_timings)
    total_cost = sum(s["cost_usd"] for s in all_step_timings)

    rows = []
    for s in all_step_timings:
        raw, human = _format_duration(s["duration"])
        rows.append({
            "name": s["name"],
            "dur_raw": raw, "dur_human": human,
            "input": _format_tokens(s["input_tokens"]),
            "output": _format_tokens(s["output_tokens"]),
            "cost": "${:.4f}".format(s["cost_usd"]),
        })

    total_raw, total_human = _format_duration(total_duration)
    rows.append({
        "name": "TOTAL",
        "dur_raw": total_raw, "dur_human": total_human,
        "input": _format_tokens(total_input),
        "output": _format_tokens(total_output),
        "cost": "${:.4f}".format(total_cost),
    })

    max_dur_len = max(len(r["dur_raw"]) for r in rows)
    max_in_len = max(len(r["input"]) for r in rows)
    max_out_len = max(len(r["output"]) for r in rows)
    w = 120

    print("\n" + "=" * w)
    print("DETAILED STEP TIMINGS")
    print("=" * w)
    print("{:<35s} {:<25s} {:<12s} {:<12s} {}".format(
        "Step", "Duration", "In Tokens", "Out Tokens", "Cost"
    ))
    print("-" * w)

    for i, r in enumerate(rows):
        if i == len(rows) - 1:
            print("-" * w)
        dur_str = "{:<{}}  ({})".format(r["dur_raw"], max_dur_len, r["dur_human"])
        print("{:<35s} {:<25s} {:<12s} {:<12s} {}".format(
            r["name"], dur_str,
            r["input"], r["output"], r["cost"],
        ))

    print("=" * w)


def _print_phase_summary(phase_results, all_step_timings, total_duration):
    phase_usage = {}
    for s in all_step_timings:
        name = s["name"]
        _PHASE_STEP_PREFIXES = {
            "Code traces": ["code_trace_"],
            "Code port plan": ["code_port_plan"],
            "Test plan": ["test_plan"],
            "Code gen": ["code_gen_V", "code_gen_review", "clear_target_dir_code_gen"],
            "Runtime": ["apply_compile_", "run_log_", "investigate_runtime_", "clear_runtime_"],
        }
        for phase_name, prefixes in _PHASE_STEP_PREFIXES.items():
            if any(name.startswith(p) for p in prefixes):
                phase_usage.setdefault(phase_name, {"input": 0, "output": 0, "cost": 0.0})
                phase_usage[phase_name]["input"] += s["input_tokens"]
                phase_usage[phase_name]["output"] += s["output_tokens"]
                phase_usage[phase_name]["cost"] += s["cost_usd"]
                break

    total_input = sum(s["input_tokens"] for s in all_step_timings)
    total_output = sum(s["output_tokens"] for s in all_step_timings)
    total_cost = sum(s["cost_usd"] for s in all_step_timings)

    rows = []
    for phase in phase_results:
        if phase["max_iterations"] <= 1:
            converged_str = "N/A"
        elif phase["converged"]:
            converged_str = "Yes (iter {})".format(phase["iterations"])
        else:
            converged_str = "No"
        raw, human = _format_duration(phase["duration"])
        usage = phase_usage.get(phase["name"], {"input": 0, "output": 0, "cost": 0.0})
        rows.append({
            "name": phase["name"],
            "iters": "{}/{}".format(phase["iterations"], phase["max_iterations"]),
            "conv": converged_str,
            "dur_raw": raw, "dur_human": human,
            "input": _format_tokens(usage["input"]),
            "output": _format_tokens(usage["output"]),
            "cost": "${:.4f}".format(usage["cost"]),
        })

    total_raw, total_human = _format_duration(total_duration)
    rows.append({
        "name": "TOTAL", "iters": "", "conv": "",
        "dur_raw": total_raw, "dur_human": total_human,
        "input": _format_tokens(total_input),
        "output": _format_tokens(total_output),
        "cost": "${:.4f}".format(total_cost),
    })

    max_dur_len = max(len(r["dur_raw"]) for r in rows)
    w = 120

    print("\n" + "=" * w)
    print("PHASE SUMMARY")
    print("=" * w)
    print("{:<20s} {:<10s} {:<14s} {:<25s} {:<12s} {:<12s} {}".format(
        "Phase", "Iters", "Converged", "Duration", "In Tokens", "Out Tokens", "Cost"
    ))
    print("-" * w)

    for i, r in enumerate(rows):
        if i == len(rows) - 1:
            print("-" * w)
        dur_str = "{:<{}}  ({})".format(r["dur_raw"], max_dur_len, r["dur_human"])
        print("{:<20s} {:<10s} {:<14s} {:<25s} {:<12s} {:<12s} {}".format(
            r["name"], r["iters"], r["conv"], dur_str,
            r["input"], r["output"], r["cost"],
        ))

    print("=" * w)


async def run_pipeline(config, claude_config, use_case, resume=False):
    output_dir = config.output_dir
    context = use_case.create_context_str(claude_config, config)
    phase_results = []
    all_step_timings = []

    # Phase 1: Code traces
    phase_start = time.time()
    code_trace_steps, code_trace_files = use_case.gen_code_trace_steps(
        context, config
    )
    if resume and _all_steps_complete(code_trace_steps, output_dir):
        print("Resume: skipping completed phase 'Code traces'")
    else:
        timings = await claude_run(claude_config, code_trace_steps)
        all_step_timings.extend(timings)
    phase_results.append({
        "name": "Code traces",
        "iterations": 1,
        "max_iterations": 1,
        "converged": False,
        "duration": time.time() - phase_start,
    })

    # Phase 2: Code port plan iterations
    phase_start = time.time()
    skip_code_port_review = config.code_port_plan_skip_review
    max_code_port_plan_iterations = 1 if skip_code_port_review else config.num_code_port_plan_iterations
    prev_output_file = None
    prev_output_summary_file = None
    code_port_converged = False
    code_port_iterations = 0
    for i in range(max_code_port_plan_iterations):
        iteration = i + 1
        code_port_iterations = iteration
        steps, review_prompt = use_case.gen_code_port_plan_iter_steps(
            context, config, code_trace_files,
            prev_output_file, prev_output_summary_file, iteration,
        )
        run_steps = steps[:-1] if skip_code_port_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print("Resume: skipping completed iteration 'Code port plan V{}'".format(iteration))
            if skip_code_port_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = run_steps[-1].output_files[1] if len(run_steps[-1].output_files) > 1 else None
            else:
                prev_output_file = review_prompt.output_file
                prev_output_summary_file = review_prompt.output_summary_file
                if _check_converged(output_dir, review_prompt.output_summary_file):
                    print("Code port plan CONVERGED at iteration {}".format(iteration))
                    code_port_converged = True
                    break
            continue

        timings = await claude_run(claude_config, run_steps)
        all_step_timings.extend(timings)

        if skip_code_port_review:
            prev_output_file = run_steps[-1].output_files[0]
            prev_output_summary_file = run_steps[-1].output_files[1] if len(run_steps[-1].output_files) > 1 else None
        else:
            prev_output_file = review_prompt.output_file
            prev_output_summary_file = review_prompt.output_summary_file
            if _check_converged(output_dir, review_prompt.output_summary_file):
                print("Code port plan CONVERGED at iteration {}".format(iteration))
                code_port_converged = True
                break

    final_code_port_plan_file = prev_output_file
    phase_results.append({
        "name": "Code port plan",
        "iterations": code_port_iterations,
        "max_iterations": max_code_port_plan_iterations,
        "converged": code_port_converged,
        "duration": time.time() - phase_start,
    })

    # Phase 3: Test plan iterations (skipped when code port plan already includes the test plan)
    if use_case.skip_test_plan_phase(config):
        print("Skipping test plan phase (combined with code port plan)")
        final_test_plan_file = final_code_port_plan_file
    else:
        phase_start = time.time()
        skip_test_review = config.test_plan_skip_review
        max_test_plan_iterations = 1 if skip_test_review else config.num_test_plan_iterations
        prev_output_file = None
        prev_output_summary_file = None
        test_plan_converged = False
        test_plan_iterations = 0
        for i in range(max_test_plan_iterations):
            iteration = i + 1
            test_plan_iterations = iteration
            steps, review_prompt = use_case.gen_test_plan_iter_steps(
                context, config, code_trace_files, final_code_port_plan_file,
                prev_output_file, prev_output_summary_file, iteration,
            )
            run_steps = steps[:-1] if skip_test_review else steps

            if resume and _all_steps_complete(run_steps, output_dir):
                print("Resume: skipping completed iteration 'Test plan V{}'".format(iteration))
                if skip_test_review:
                    prev_output_file = run_steps[-1].output_files[0]
                    prev_output_summary_file = run_steps[-1].output_files[1] if len(run_steps[-1].output_files) > 1 else None
                else:
                    prev_output_file = review_prompt.output_file
                    prev_output_summary_file = review_prompt.output_summary_file
                    if _check_converged(output_dir, review_prompt.output_summary_file):
                        print("Test plan CONVERGED at iteration {}".format(iteration))
                        test_plan_converged = True
                        break
                continue

            timings = await claude_run(claude_config, run_steps)
            all_step_timings.extend(timings)

            if skip_test_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = run_steps[-1].output_files[1] if len(run_steps[-1].output_files) > 1 else None
            else:
                prev_output_file = review_prompt.output_file
                prev_output_summary_file = review_prompt.output_summary_file
                if _check_converged(output_dir, review_prompt.output_summary_file):
                    print("Test plan CONVERGED at iteration {}".format(iteration))
                    test_plan_converged = True
                    break

        final_test_plan_file = prev_output_file
        phase_results.append({
            "name": "Test plan",
            "iterations": test_plan_iterations,
            "max_iterations": max_test_plan_iterations,
            "converged": test_plan_converged,
            "duration": time.time() - phase_start,
        })

    # Phase 4: Code gen iterations
    phase_start = time.time()
    skip_code_gen_review = config.code_gen_skip_review
    max_code_gen_iterations = 1 if skip_code_gen_review else config.num_code_gen_iterations
    prev_output_patch_file = None
    prev_output_summary_file = None
    code_gen_converged = False
    code_gen_iterations = 0
    for i in range(max_code_gen_iterations):
        iteration = i + 1
        code_gen_iterations = iteration
        steps, code_review_prompt = use_case.gen_code_gen_iter_steps(
            context, config, code_trace_files,
            final_code_port_plan_file, final_test_plan_file,
            prev_output_patch_file, prev_output_summary_file, iteration,
        )
        run_steps = steps[:-1] if skip_code_gen_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print("Resume: skipping completed iteration 'Code gen V{}'".format(iteration))
            if skip_code_gen_review:
                prev_output_patch_file = run_steps[-1].output_files[0]
                prev_output_summary_file = run_steps[-1].output_files[1] if len(run_steps[-1].output_files) > 1 else None
            else:
                prev_output_patch_file = code_review_prompt.output_patch_file
                prev_output_summary_file = code_review_prompt.output_summary_file
                if _check_converged(output_dir, code_review_prompt.output_summary_file):
                    print("Code gen CONVERGED at iteration {}".format(iteration))
                    code_gen_converged = True
                    break
            continue

        timings = await claude_run(claude_config, run_steps)
        all_step_timings.extend(timings)

        if skip_code_gen_review:
            prev_output_patch_file = run_steps[-1].output_files[0]
            prev_output_summary_file = run_steps[-1].output_files[1] if len(run_steps[-1].output_files) > 1 else None
        else:
            prev_output_patch_file = code_review_prompt.output_patch_file
            prev_output_summary_file = code_review_prompt.output_summary_file
            if _check_converged(output_dir, code_review_prompt.output_summary_file):
                print("Code gen CONVERGED at iteration {}".format(iteration))
                code_gen_converged = True
                break

    phase_results.append({
        "name": "Code gen",
        "iterations": code_gen_iterations,
        "max_iterations": max_code_gen_iterations,
        "converged": code_gen_converged,
        "duration": time.time() - phase_start,
    })

    # Phase 5: Runtime iterations
    runtime_phase_results, runtime_timings = await use_case.run_runtime_iterations(
        context, config, claude_config, code_trace_files,
        final_code_port_plan_file, final_test_plan_file,
        resume=resume,
    )
    phase_results.extend(runtime_phase_results)
    all_step_timings.extend(runtime_timings)

    return phase_results, all_step_timings


async def run_runtime_iterations(
    context, code_trace_files, config, claude_config,
    code_port_plan_file, test_plan_file, iteration_history_summary_file,
    resume=False, start_iteration=None, smaller_model_file=None,
    disable_new_feature=False,
):
    """Run the runtime iteration loop: apply patch, run benchmark, investigate.

    This function can be called either as Phase 5 of the full pipeline
    (from run_pipeline) or standalone (from run_runtime_iters.py).

    Returns (phase_results, all_step_timings).
    """
    output_dir = config.output_dir
    max_runtime_iterations = config.num_runtime_iterations
    runtime_logs_dir = os.path.join(output_dir, RUNTIME_LOGS_DIR)
    os.makedirs(runtime_logs_dir, exist_ok=True)

    phase_start = time.time()
    all_step_timings = []
    runtime_succeeded = False
    runtime_iterations = 0

    if resume and _check_runtime_success(output_dir):
        print("Resume: runtime iterations already succeeded (found {})".format(
            RUNTIME_SUCCESS_FILE
        ))
        return [{
            "name": "Runtime",
            "iterations": 0,
            "max_iterations": max_runtime_iterations,
            "converged": True,
            "duration": 0.0,
        }], []

    if start_iteration is not None:
        iter_start = start_iteration
    else:
        iter_start = _find_latest_runtime_iteration(output_dir) + 1
        if resume and iter_start > 1:
            print("Resume: continuing runtime iterations from V{}".format(iter_start))

    latest_patch = _find_latest_patch(output_dir)
    if latest_patch is None:
        print("ERROR: No code patch found in {}. Cannot run runtime iterations.".format(
            output_dir
        ))
        return [{
            "name": "Runtime",
            "iterations": 0,
            "max_iterations": max_runtime_iterations,
            "converged": False,
            "duration": time.time() - phase_start,
        }], []

    prev_patch_file = latest_patch
    prev_summary_file = None

    if iter_start > 1:
        prev_summary = "{}_summary_V{}.txt".format(
            RUNTIME_FILE_PREFIX, iter_start - 1
        )
        if os.path.isfile(os.path.join(output_dir, prev_summary)):
            prev_summary_file = prev_summary

    if not config.target_run_command:
        print("ERROR: No execution command found in target trace run_params.txt.")
        print("  Ensure the single-trace analysis has a valid run_command.")
        return [{
            "name": "Runtime",
            "iterations": 0,
            "max_iterations": max_runtime_iterations,
            "converged": False,
            "duration": time.time() - phase_start,
        }], []

    print("\n" + "=" * 80)
    print("PHASE 5: RUNTIME ITERATIONS")
    print("=" * 80)
    print("  Max iterations: {}".format(max_runtime_iterations))
    print("  Starting from iteration: {}".format(iter_start))
    print("  Initial patch: {}".format(prev_patch_file))
    print("  Execution command: {}".format(config.target_run_command))
    print("  Runtime logs dir: {}".format(runtime_logs_dir))
    print("=" * 80 + "\n")

    for i in range(iter_start - 1, max_runtime_iterations):
        iteration = i + 1
        runtime_iterations = iteration

        print("\n--- Runtime Iteration {} / {} ---".format(
            iteration, max_runtime_iterations
        ))
        print("  Applying patch: {}".format(prev_patch_file))

        # Step 1: Clear target repo + Apply patch + Compile
        clear_cmd = {
            "fn": lambda: clear_vllm_source_tree(config.source_code_dir),
            "fn_name": 'clear_vllm_source_tree("{}")'.format(
                config.source_code_dir
            ),
        }
        apply_prompt = gen_ApplyCodeAndCompilePrompt(
            context=context,
            patch_file=prev_patch_file,
            iteration=iteration,
        )
        steps_apply = [
            PipelineStep(
                name="clear_runtime_V{}".format(iteration),
                prompt=clear_cmd,
            ),
            PipelineStep(
                name="apply_compile_V{}".format(iteration),
                prompt=apply_prompt.prompt(),
                output_files=[apply_prompt.output_status_file],
            ),
        ]

        timings = await claude_run(claude_config, steps_apply)
        all_step_timings.extend(timings)

        if not _check_apply_succeeded(output_dir, iteration):
            print("Runtime iteration {}: Patch apply/compile FAILED, stopping.".format(
                iteration
            ))
            break

        # Step 2: Run benchmark and capture output
        run_prompt = gen_RunAndLogPrompt(
            context=context,
            execution_command=config.target_run_command,
            runtime_logs_dir=runtime_logs_dir,
            iteration=iteration,
            gpu_wait_timeout_minutes=config.gpu_wait_timeout_minutes,
            smaller_model_file=smaller_model_file,
            disable_new_feature=disable_new_feature,
        )
        steps_run = [
            PipelineStep(
                name="run_log_V{}".format(iteration),
                prompt=run_prompt.prompt(),
            ),
        ]

        timings = await claude_run(claude_config, steps_run)
        all_step_timings.extend(timings)

        # Step 2.5: Run lm_eval correctness check (only when patched run succeeded and feature is enabled)
        lm_eval_result_file = None
        if (not disable_new_feature
                and _check_run_succeeded(runtime_logs_dir, iteration)):
            print("Runtime iteration {}: Patched run succeeded, running lm_eval correctness check...".format(
                iteration
            ))
            lm_eval_prompt = gen_RunLMEvalPrompt(
                context=context,
                execution_command=config.target_run_command,
                runtime_logs_dir=runtime_logs_dir,
                iteration=iteration,
                gpu_wait_timeout_minutes=config.gpu_wait_timeout_minutes,
                smaller_model_file=smaller_model_file,
            )
            steps_lm_eval = [
                PipelineStep(
                    name="lm_eval_V{}".format(iteration),
                    prompt=lm_eval_prompt.prompt(),
                    output_files=[lm_eval_prompt.output_file],
                ),
            ]

            timings = await claude_run(claude_config, steps_lm_eval)
            all_step_timings.extend(timings)
            lm_eval_result_file = lm_eval_prompt.output_file

        # Step 3: Investigate runtime output
        investigate_prompt = gen_InvestigateRuntimeOutputAndFixCodePrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            test_plan_file=test_plan_file,
            runtime_logs_dir=runtime_logs_dir,
            iteration=iteration,
            prev_patch_file=prev_patch_file,
            prev_summary_file=prev_summary_file,
            iteration_history_summary_file=iteration_history_summary_file,
            smaller_model_file=smaller_model_file,
            lm_eval_result_file=lm_eval_result_file,
        )
        steps_investigate = [
            PipelineStep(
                name="investigate_runtime_V{}".format(iteration),
                prompt=investigate_prompt.prompt(),
            ),
        ]

        timings = await claude_run(claude_config, steps_investigate)
        all_step_timings.extend(timings)

        # Check if the run succeeded
        if _check_runtime_success(output_dir):
            print("\nRuntime iteration {}: SUCCESS! Results written to {}".format(
                iteration, RUNTIME_SUCCESS_FILE
            ))
            runtime_succeeded = True
            break

        # Check if a new fixed patch was generated
        new_patch = "{}_V{}.patch".format(RUNTIME_FILE_PREFIX, iteration)
        new_summary = "{}_summary_V{}.txt".format(RUNTIME_FILE_PREFIX, iteration)
        if not os.path.isfile(os.path.join(output_dir, new_patch)):
            print("Runtime iteration {}: No new patch generated, stopping.".format(
                iteration
            ))
            break

        prev_patch_file = new_patch
        prev_summary_file = new_summary
        print("Runtime iteration {}: Errors fixed, new patch: {}".format(
            iteration, new_patch
        ))

    phase_results = [{
        "name": "Runtime",
        "iterations": runtime_iterations,
        "max_iterations": max_runtime_iterations,
        "converged": runtime_succeeded,
        "duration": time.time() - phase_start,
    }]

    return phase_results, all_step_timings


if __name__ == "__main__":
    args = parse_args()

    config, use_case = load_config_and_use_case(args.config)
    claude_config = config.make_claude_config()

    setup_logging("code_gen")

    os.makedirs(config.output_dir, exist_ok=True)
    if not args.resume:
        safe_clean_dir(config.output_dir)

    if hasattr(config, 'prepare_branches'):
        print("Preparing source code branches...")
        config.prepare_branches()

    start_time = time.time()
    phase_results, all_step_timings = asyncio.run(run_pipeline(
        config, claude_config, use_case, resume=args.resume
    ))
    total_duration = time.time() - start_time

    _print_detailed_table(all_step_timings, total_duration)
    _print_phase_summary(phase_results, all_step_timings, total_duration)
