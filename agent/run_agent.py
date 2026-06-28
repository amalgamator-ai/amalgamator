import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import re
import json
import time
import argparse
import asyncio

from common.utils import setup_logging, safe_clean_dir
from common.claude_utils import claude_run, PipelineStep, _all_steps_complete

from agent.agent_config import AgentConfig
from agent.use_case import FeaturePortingUseCase
from agent.agent_prompts import (
    DETECTED_FEATURES_JSON,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
    TEST_PLAN_PREFIX,
    RUNTIME_SUCCESS_FILE,
)


CONVERGENCE_MARKER = "CONVERGED"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hermes->OpenClaw feature porting pipeline"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the agent JSON config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last incomplete step.",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Run only the feature detection phase, then stop.",
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        default=None,
        help="Override the feature_id from config (1-based rank from detection results).",
    )
    return parser.parse_args()


def _check_converged(output_dir, summary_file):
    path = os.path.join(output_dir, summary_file)
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        first_line = f.readline().strip()
    return CONVERGENCE_MARKER in first_line


def _load_feature_description(output_dir, feature_id):
    json_path = os.path.join(output_dir, DETECTED_FEATURES_JSON)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            "Feature detection results not found at {}.\n"
            "Run with --detect-only first, or let the full pipeline run.".format(
                json_path
            )
        )
    with open(json_path) as f:
        data = json.load(f)

    features = data.get("detected_features", [])
    if not features:
        raise ValueError("No features detected. Check the detection results.")

    if feature_id < 1 or feature_id > len(features):
        raise ValueError(
            "feature_id {} is out of range. Detected {} features (1-{}).".format(
                feature_id, len(features), len(features),
            )
        )

    feature = features[feature_id - 1]
    return (
        "Feature: {name}\n"
        "Category: {category}\n"
        "Summary: {summary}\n"
        "Porting complexity: {complexity}\n"
        "OpenClaw status: {status}\n"
        "Key Hermes files: {files}"
    ).format(
        name=feature.get("feature_name", "Unknown"),
        category=feature.get("category", "Unknown"),
        summary=feature.get("summary", "No summary available"),
        complexity=feature.get("porting_complexity", "Unknown"),
        status=feature.get("openclaw_status", "Unknown"),
        files=", ".join(feature.get("key_hermes_files", [])),
    )


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


def _print_phase_summary(phase_results, all_step_timings, total_duration, output_dir=None):
    total_input = sum(s["input_tokens"] for s in all_step_timings)
    total_output = sum(s["output_tokens"] for s in all_step_timings)
    total_cost = sum(s["cost_usd"] for s in all_step_timings)

    w = 120
    lines = []
    lines.append("\n" + "=" * w)
    lines.append("PHASE SUMMARY")
    lines.append("=" * w)
    lines.append("{:<25s} {:<10s} {:<14s} {:<25s} {:<12s} {:<12s} {}".format(
        "Phase", "Iters", "Converged", "Duration", "In Tokens", "Out Tokens", "Cost"
    ))
    lines.append("-" * w)

    for phase in phase_results:
        if phase["max_iterations"] <= 1:
            converged_str = "N/A"
        elif phase["converged"]:
            converged_str = "Yes (iter {})".format(phase["iterations"])
        else:
            converged_str = "No"
        raw, human = _format_duration(phase["duration"])
        dur_str = "{}  ({})".format(raw, human)
        lines.append("{:<25s} {:<10s} {:<14s} {:<25s}".format(
            phase["name"],
            "{}/{}".format(phase["iterations"], phase["max_iterations"]),
            converged_str,
            dur_str,
        ))

    total_raw, total_human = _format_duration(total_duration)
    lines.append("-" * w)
    lines.append("{:<25s} {:<10s} {:<14s} {:<25s} {:<12s} {:<12s} {}".format(
        "TOTAL", "", "",
        "{}  ({})".format(total_raw, total_human),
        _format_tokens(total_input),
        _format_tokens(total_output),
        "${:.4f}".format(total_cost),
    ))
    lines.append("=" * w)

    for line in lines:
        print(line)

    if output_dir:
        output_file = os.path.join(output_dir, "output.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(lines) + "\n")


async def run_pipeline(config, claude_config, use_case, resume=False,
                       detect_only=False, feature_id=None):
    output_dir = config.output_dir
    context = use_case.create_context_str(config)
    phase_results = []
    all_step_timings = []

    # Phase 0: Feature Detection
    phase_start = time.time()
    print("\n" + "=" * 80)
    print("PHASE 0: FEATURE DETECTION")
    print("=" * 80)

    detection_steps = use_case.gen_feature_detection_steps(context, config)

    if resume and _all_steps_complete(detection_steps, output_dir):
        print("Resume: skipping completed phase 'Feature Detection'")
    else:
        timings = await claude_run(claude_config, detection_steps)
        all_step_timings.extend(timings)

    phase_results.append({
        "name": "Feature Detection",
        "iterations": 1,
        "max_iterations": 1,
        "converged": False,
        "duration": time.time() - phase_start,
    })

    # Print detection results
    _print_detection_results(output_dir)

    if detect_only:
        print("\n--detect-only flag set. Stopping after feature detection.")
        return phase_results, all_step_timings

    # Determine which feature to port
    effective_feature_id = feature_id if feature_id is not None else config.feature_id
    if effective_feature_id == 0:
        effective_feature_id = 1
    feature_description = _load_feature_description(output_dir, effective_feature_id)

    print("\n" + "=" * 80)
    print("SELECTED FEATURE (rank #{}) FOR PORTING:".format(effective_feature_id))
    print(feature_description)
    print("=" * 80)

    # Phase 1: Code Traces
    phase_start = time.time()
    print("\n" + "=" * 80)
    print("PHASE 1: CODE TRACES")
    print("=" * 80)

    code_trace_steps, code_trace_files = use_case.gen_code_trace_steps(
        context, config, feature_description,
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

    # Phase 2: Code Port Plan Iterations
    phase_start = time.time()
    print("\n" + "=" * 80)
    print("PHASE 2: CODE PORT PLAN ITERATIONS")
    print("=" * 80)

    skip_code_port_review = config.code_port_plan_skip_review
    max_code_port_iters = 1 if skip_code_port_review else config.num_code_port_plan_iterations
    prev_output_file = None
    prev_output_summary_file = None
    code_port_converged = False
    code_port_iterations = 0

    for i in range(max_code_port_iters):
        iteration = i + 1
        code_port_iterations = iteration
        steps, review_prompt = use_case.gen_code_port_plan_iter_steps(
            context, config, code_trace_files, feature_description,
            prev_output_file, prev_output_summary_file, iteration,
        )
        run_steps = steps[:-1] if skip_code_port_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print("Resume: skipping iteration 'Code port plan V{}'".format(iteration))
            if skip_code_port_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1 else None
                )
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
            prev_output_summary_file = (
                run_steps[-1].output_files[1]
                if len(run_steps[-1].output_files) > 1 else None
            )
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
        "max_iterations": max_code_port_iters,
        "converged": code_port_converged,
        "duration": time.time() - phase_start,
    })

    # Phase 3: Test Plan Iterations
    phase_start = time.time()
    print("\n" + "=" * 80)
    print("PHASE 3: TEST PLAN ITERATIONS")
    print("=" * 80)

    skip_test_review = config.test_plan_skip_review
    max_test_plan_iters = 1 if skip_test_review else config.num_test_plan_iterations
    prev_output_file = None
    prev_output_summary_file = None
    test_plan_converged = False
    test_plan_iterations = 0

    for i in range(max_test_plan_iters):
        iteration = i + 1
        test_plan_iterations = iteration
        steps, review_prompt = use_case.gen_test_plan_iter_steps(
            context, config, code_trace_files,
            final_code_port_plan_file, feature_description,
            prev_output_file, prev_output_summary_file, iteration,
        )
        run_steps = steps[:-1] if skip_test_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print("Resume: skipping iteration 'Test plan V{}'".format(iteration))
            if skip_test_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1 else None
                )
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
            prev_output_summary_file = (
                run_steps[-1].output_files[1]
                if len(run_steps[-1].output_files) > 1 else None
            )
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
        "max_iterations": max_test_plan_iters,
        "converged": test_plan_converged,
        "duration": time.time() - phase_start,
    })

    # Phase 4: Code Gen Iterations
    phase_start = time.time()
    print("\n" + "=" * 80)
    print("PHASE 4: CODE GEN ITERATIONS")
    print("=" * 80)

    skip_code_gen_review = config.code_gen_skip_review
    max_code_gen_iters = 1 if skip_code_gen_review else config.num_code_gen_iterations
    prev_output_patch_file = None
    prev_output_summary_file = None
    code_gen_converged = False
    code_gen_iterations = 0

    for i in range(max_code_gen_iters):
        iteration = i + 1
        code_gen_iterations = iteration
        steps, code_review_prompt = use_case.gen_code_gen_iter_steps(
            context, config, code_trace_files,
            final_code_port_plan_file, final_test_plan_file,
            feature_description,
            prev_output_patch_file, prev_output_summary_file, iteration,
        )
        run_steps = steps[:-1] if skip_code_gen_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print("Resume: skipping iteration 'Code gen V{}'".format(iteration))
            if skip_code_gen_review:
                prev_output_patch_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1 else None
                )
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
            prev_output_summary_file = (
                run_steps[-1].output_files[1]
                if len(run_steps[-1].output_files) > 1 else None
            )
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
        "max_iterations": max_code_gen_iters,
        "converged": code_gen_converged,
        "duration": time.time() - phase_start,
    })

    # Phase 5: Runtime Iterations
    runtime_phase_results, runtime_timings = await use_case.run_runtime_iterations(
        context, config, claude_config, code_trace_files,
        final_code_port_plan_file, final_test_plan_file,
        feature_description, resume=resume,
    )
    phase_results.extend(runtime_phase_results)
    all_step_timings.extend(runtime_timings)

    return phase_results, all_step_timings


def _print_detection_results(output_dir):
    json_path = os.path.join(output_dir, DETECTED_FEATURES_JSON)
    report_path = os.path.join(output_dir, "detected_features.txt")

    print("\n" + "=" * 80)
    print("FEATURE DETECTION RESULTS")
    print("=" * 80)

    if os.path.isfile(report_path):
        with open(report_path) as f:
            report = f.read()
        print(report)

    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
        features = data.get("detected_features", [])
        print("\n--- DETECTED FEATURES SUMMARY ({} candidates) ---\n".format(
            len(features)
        ))
        for feat in features:
            print("  #{rank}: {name} (score: {score})".format(
                rank=feat.get("rank", "?"),
                name=feat.get("feature_name", "Unknown"),
                score=feat.get("overall_score", "?"),
            ))
            print("      OpenClaw status: {}".format(
                feat.get("openclaw_status", "?")
            ))
            print("      Complexity: {}".format(
                feat.get("porting_complexity", "?")
            ))
            issues = feat.get("openclaw_issues", [])
            if issues:
                print("      Community issues: {}".format(
                    ", ".join("#{}".format(i.get("number", "?")) for i in issues[:5])
                ))
            print()
    else:
        print("  (No structured results found)")

    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()

    config = AgentConfig.from_json(args.config)
    claude_config = config.make_claude_config()
    use_case = FeaturePortingUseCase()

    setup_logging("agent")

    os.makedirs(config.output_dir, exist_ok=True)
    if not args.resume:
        safe_clean_dir(config.output_dir)

    feature_id = args.feature_id if args.feature_id is not None else None

    start_time = time.time()
    phase_results, all_step_timings = asyncio.run(run_pipeline(
        config, claude_config, use_case,
        resume=args.resume,
        detect_only=args.detect_only,
        feature_id=feature_id,
    ))
    total_duration = time.time() - start_time

    _print_phase_summary(phase_results, all_step_timings, total_duration, config.output_dir)
