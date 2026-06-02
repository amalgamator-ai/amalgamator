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

from auto_code_gen.code_gen_configs import CodeGenConfig

from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_CodePortPlanPrompt,
    gen_ReviewCodePortPlanPrompt,
    gen_TestPlanPrompt,
    gen_ReviewTestPlanPrompt,
    gen_CodeGenPrompt,
    gen_ReviewCodeGenPrompt,
    gen_ApplyCodeAndCompilePrompt,
    gen_RunAndLogPrompt,
    gen_InvestigateRuntimeOutputAndFixCodePrompt,
    gen_IterationHistorySummaryPrompt,
    gen_FindSmallerModelPrompt,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
    TEST_PLAN_PREFIX,
    RUNTIME_FILE_PREFIX,
    RUNTIME_SUCCESS_FILE,
    RUNTIME_LOGS_DIR,
    RUNTIME_SMALLER_MODEL_FILE,
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


def _gen_code_trace_steps(context, code_gen_config):
    clear_target_dir_cmd = {
        "fn": lambda: clear_vllm_source_tree(code_gen_config.source_code_dir),
        "fn_name": 'clear_vllm_source_tree("{}")'.format(code_gen_config.source_code_dir),
    }

    source_fw = code_gen_config.source_framework
    target_fw = code_gen_config.target_framework

    source_code_trace_prompt = gen_CodeTracePrompt(context=context, framework=source_fw)
    target_code_trace_prompt = gen_CodeTracePrompt(context=context, framework=target_fw)

    framework_code_trace_files = [
        source_code_trace_prompt.output_file,
        target_code_trace_prompt.output_file,
    ]

    steps = [
        PipelineStep(
            name="clear_target_dir",
            prompt=clear_target_dir_cmd,
        ),
        PipelineStep(
            name="code_trace_{}".format(source_fw),
            prompt=source_code_trace_prompt.prompt(),
            output_files=[source_code_trace_prompt.output_file],
        ),
        PipelineStep(
            name="code_trace_{}".format(target_fw),
            prompt=target_code_trace_prompt.prompt(),
            output_files=[target_code_trace_prompt.output_file],
        ),
    ]

    return steps, framework_code_trace_files


def _gen_code_port_plan_iteration_steps(
    context, framework_code_trace_files, code_gen_config,
    prev_output_file, prev_output_summary_file, iteration,
):
    code_port_plan_prompt = gen_CodePortPlanPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_disallowed_modules=code_gen_config.code_port_disallowed_modules,
        output_file="{}_V{}.txt".format(CODE_PORT_PLAN_FILE_PREFIX, iteration),
        output_summary_file="{}_summary_V{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, iteration
        ),
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
    )

    review_code_port_plan_prompt = gen_ReviewCodePortPlanPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        input_file=code_port_plan_prompt.output_file,
        input_summary_file=code_port_plan_prompt.output_summary_file,
        output_file="{}_review_V{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, iteration
        ),
        output_summary_file="{}_review_summary_V{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, iteration
        ),
        iteration=iteration,
    )

    steps = [
        PipelineStep(
            name="code_port_plan_V{}".format(iteration),
            prompt=code_port_plan_prompt.prompt(),
            output_files=[code_port_plan_prompt.output_file, code_port_plan_prompt.output_summary_file],
        ),
        PipelineStep(
            name="code_port_plan_review_V{}".format(iteration),
            prompt=review_code_port_plan_prompt.prompt(),
            output_files=[review_code_port_plan_prompt.output_file, review_code_port_plan_prompt.output_summary_file],
        ),
    ]

    return steps, review_code_port_plan_prompt


def _gen_test_plan_iteration_steps(
    context, framework_code_trace_files, code_port_plan_file,
    prev_output_file, prev_output_summary_file, iteration,
):
    test_plan_prompt = gen_TestPlanPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        output_file="{}_V{}.txt".format(TEST_PLAN_PREFIX, iteration),
        output_summary_file="{}_summary_V{}.txt".format(
            TEST_PLAN_PREFIX, iteration
        ),
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
    )

    review_test_plan_prompt = gen_ReviewTestPlanPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        input_file=test_plan_prompt.output_file,
        input_summary_file=test_plan_prompt.output_summary_file,
        output_file="{}_review_V{}.txt".format(TEST_PLAN_PREFIX, iteration),
        output_summary_file="{}_review_summary_V{}.txt".format(
            TEST_PLAN_PREFIX, iteration
        ),
        iteration=iteration,
    )

    steps = [
        PipelineStep(
            name="test_plan_V{}".format(iteration),
            prompt=test_plan_prompt.prompt(),
            output_files=[test_plan_prompt.output_file, test_plan_prompt.output_summary_file],
        ),
        PipelineStep(
            name="test_plan_review_V{}".format(iteration),
            prompt=review_test_plan_prompt.prompt(),
            output_files=[review_test_plan_prompt.output_file, review_test_plan_prompt.output_summary_file],
        ),
    ]

    return steps, review_test_plan_prompt


def _gen_code_gen_iteration_steps(
    context, framework_code_trace_files, code_gen_config,
    code_port_plan_file, test_plan_file,
    prev_output_patch_file, prev_output_summary_file, iteration,
):
    clear_target_dir_cmd = {
        "fn": lambda: clear_vllm_source_tree(code_gen_config.source_code_dir),
        "fn_name": 'clear_vllm_source_tree("{}")'.format(code_gen_config.source_code_dir),
    }

    code_gen_prompt = gen_CodeGenPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        output_patch_file="{}_V{}.patch".format(
            CODE_GEN_FILE_PREFIX, iteration
        ),
        output_summary_file="{}_summary_V{}.txt".format(
            CODE_GEN_FILE_PREFIX, iteration
        ),
        prev_output_patch_file=prev_output_patch_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
    )

    code_review_prompt = gen_ReviewCodeGenPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        input_patch_file=code_gen_prompt.output_patch_file,
        input_summary_file=code_gen_prompt.output_summary_file,
        output_patch_file="{}_review_V{}.patch".format(
            CODE_GEN_FILE_PREFIX, iteration
        ),
        output_summary_file="{}_review_summary_V{}.txt".format(
            CODE_GEN_FILE_PREFIX, iteration
        ),
        iteration=iteration,
    )

    steps = [
        PipelineStep(
            name="clear_target_dir_code_gen_V{}".format(iteration),
            prompt=clear_target_dir_cmd,
        ),
        PipelineStep(
            name="code_gen_V{}".format(iteration),
            prompt=code_gen_prompt.prompt(),
            output_files=[code_gen_prompt.output_patch_file, code_gen_prompt.output_summary_file],
        ),
        PipelineStep(
            name="code_gen_review_V{}".format(iteration),
            prompt=code_review_prompt.prompt(),
            output_files=[code_review_prompt.output_patch_file, code_review_prompt.output_summary_file],
        ),
    ]

    return steps, code_review_prompt


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


async def run_pipeline(code_gen_config, claude_config, resume=False):
    output_dir = code_gen_config.output_dir
    context = create_context_str(claude_config, code_gen_config)
    phase_results = []
    all_step_timings = []

    # Phase 1: Code traces
    phase_start = time.time()
    code_trace_steps, framework_code_trace_files = _gen_code_trace_steps(
        context, code_gen_config
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
    max_code_port_plan_iterations = code_gen_config.num_code_port_plan_iterations
    prev_output_file = None
    prev_output_summary_file = None
    code_port_converged = False
    code_port_iterations = 0
    for i in range(max_code_port_plan_iterations):
        iteration = i + 1
        code_port_iterations = iteration
        steps, review_prompt = _gen_code_port_plan_iteration_steps(
            context, framework_code_trace_files, code_gen_config,
            prev_output_file, prev_output_summary_file, iteration,
        )

        if resume and _all_steps_complete(steps, output_dir):
            print("Resume: skipping completed iteration 'Code port plan V{}'".format(iteration))
            prev_output_file = review_prompt.output_file
            prev_output_summary_file = review_prompt.output_summary_file
            if _check_converged(output_dir, review_prompt.output_summary_file):
                print("Code port plan CONVERGED at iteration {}".format(iteration))
                code_port_converged = True
                break
            continue

        timings = await claude_run(claude_config, steps)
        all_step_timings.extend(timings)

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

    # Phase 3: Test plan iterations
    phase_start = time.time()
    max_test_plan_iterations = code_gen_config.num_test_plan_iterations
    prev_output_file = None
    prev_output_summary_file = None
    test_plan_converged = False
    test_plan_iterations = 0
    for i in range(max_test_plan_iterations):
        iteration = i + 1
        test_plan_iterations = iteration
        steps, review_prompt = _gen_test_plan_iteration_steps(
            context, framework_code_trace_files, final_code_port_plan_file,
            prev_output_file, prev_output_summary_file, iteration,
        )

        if resume and _all_steps_complete(steps, output_dir):
            print("Resume: skipping completed iteration 'Test plan V{}'".format(iteration))
            prev_output_file = review_prompt.output_file
            prev_output_summary_file = review_prompt.output_summary_file
            if _check_converged(output_dir, review_prompt.output_summary_file):
                print("Test plan CONVERGED at iteration {}".format(iteration))
                test_plan_converged = True
                break
            continue

        timings = await claude_run(claude_config, steps)
        all_step_timings.extend(timings)

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
    max_code_gen_iterations = code_gen_config.num_code_gen_iterations
    prev_output_patch_file = None
    prev_output_summary_file = None
    code_gen_converged = False
    code_gen_iterations = 0
    for i in range(max_code_gen_iterations):
        iteration = i + 1
        code_gen_iterations = iteration
        steps, code_review_prompt = _gen_code_gen_iteration_steps(
            context, framework_code_trace_files, code_gen_config,
            final_code_port_plan_file, final_test_plan_file,
            prev_output_patch_file, prev_output_summary_file, iteration,
        )

        if resume and _all_steps_complete(steps, output_dir):
            print("Resume: skipping completed iteration 'Code gen V{}'".format(iteration))
            prev_output_patch_file = code_review_prompt.output_patch_file
            prev_output_summary_file = code_review_prompt.output_summary_file
            if _check_converged(output_dir, code_review_prompt.output_summary_file):
                print("Code gen CONVERGED at iteration {}".format(iteration))
                code_gen_converged = True
                break
            continue

        timings = await claude_run(claude_config, steps)
        all_step_timings.extend(timings)

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

    # Phase 5: Iteration history summary + Runtime iterations
    print("Generating iteration history summary before runtime iterations...")
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
        print("Finding smaller model for runtime iterations...")
        smaller_prompt = gen_FindSmallerModelPrompt(
            context=context,
            framework_code_trace_files=framework_code_trace_files,
            code_port_plan_file=final_code_port_plan_file,
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

    runtime_phase_results, runtime_timings = await run_runtime_iterations(
        context, framework_code_trace_files, code_gen_config, claude_config,
        final_code_port_plan_file, final_test_plan_file,
        history_prompt.output_file, resume=resume,
        smaller_model_file=smaller_model_file,
        disable_new_feature=code_gen_config.disable_new_feature_for_runtime,
    )
    phase_results.extend(runtime_phase_results)
    all_step_timings.extend(runtime_timings)

    return phase_results, all_step_timings


async def run_runtime_iterations(
    context, framework_code_trace_files, code_gen_config, claude_config,
    code_port_plan_file, test_plan_file, iteration_history_summary_file,
    resume=False, start_iteration=None, smaller_model_file=None,
    disable_new_feature=False,
):
    """Run the runtime iteration loop: apply patch, run benchmark, investigate.

    This function can be called either as Phase 5 of the full pipeline
    (from run_pipeline) or standalone (from run_runtime_iters.py).

    Returns (phase_results, all_step_timings).
    """
    output_dir = code_gen_config.output_dir
    max_runtime_iterations = code_gen_config.num_runtime_iterations
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

    if not code_gen_config.target_run_command:
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
    print("  Execution command: {}".format(code_gen_config.target_run_command))
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
            "fn": lambda: clear_vllm_source_tree(code_gen_config.source_code_dir),
            "fn_name": 'clear_vllm_source_tree("{}")'.format(
                code_gen_config.source_code_dir
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
            execution_command=code_gen_config.target_run_command,
            runtime_logs_dir=runtime_logs_dir,
            iteration=iteration,
            gpu_wait_timeout_minutes=code_gen_config.gpu_wait_timeout_minutes,
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

        # Step 3: Investigate runtime output
        investigate_prompt = gen_InvestigateRuntimeOutputAndFixCodePrompt(
            context=context,
            framework_code_trace_files=framework_code_trace_files,
            code_port_plan_file=code_port_plan_file,
            test_plan_file=test_plan_file,
            runtime_logs_dir=runtime_logs_dir,
            iteration=iteration,
            prev_patch_file=prev_patch_file,
            prev_summary_file=prev_summary_file,
            iteration_history_summary_file=iteration_history_summary_file,
            smaller_model_file=smaller_model_file,
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
            print("\nRuntime iteration {}: SUCCESS! Performance results written to {}".format(
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

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_claude_config()

    setup_logging("code_gen")

    os.makedirs(code_gen_config.output_dir, exist_ok=True)
    if not args.resume:
        safe_clean_dir(code_gen_config.output_dir)

    print("Preparing source code branches...")
    code_gen_config.prepare_branches()

    start_time = time.time()
    phase_results, all_step_timings = asyncio.run(run_pipeline(
        code_gen_config, claude_config, resume=args.resume
    ))
    total_duration = time.time() - start_time

    _print_detailed_table(all_step_timings, total_duration)
    _print_phase_summary(phase_results, all_step_timings, total_duration)
