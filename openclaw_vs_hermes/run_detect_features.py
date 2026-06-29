#!/usr/bin/env python3
"""Standalone script to run ONLY the feature detection phase.

Usage:
    python -m openclaw_vs_hermes.run_detect_features --config agent/configs/default_config.json
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import json
import time
import argparse
import asyncio

from common.utils import setup_logging, safe_clean_dir
from common.claude_utils import claude_run, _all_steps_complete

from openclaw_vs_hermes.agent_config import AgentConfig
from openclaw_vs_hermes.use_case import FeaturePortingUseCase
from openclaw_vs_hermes.agent_prompts import DETECTED_FEATURES_JSON


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run feature detection: identify Hermes features to port to OpenClaw"
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
    return parser.parse_args()


def _format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return "{:.1f}s ({}h {}m {}s)".format(seconds, hrs, mins, secs)
    return "{:.1f}s ({}m {}s)".format(seconds, mins, secs)


def _format_tokens(count):
    if count >= 1_000_000:
        return "{:.1f}M".format(count / 1_000_000)
    if count >= 1_000:
        return "{:.1f}K".format(count / 1_000)
    return str(count)


def print_results(output_dir):
    json_path = os.path.join(output_dir, DETECTED_FEATURES_JSON)
    report_path = os.path.join(output_dir, "detected_features.txt")

    print("\n" + "=" * 100)
    print("FEATURE DETECTION RESULTS")
    print("=" * 100)

    if os.path.isfile(report_path):
        with open(report_path) as f:
            print(f.read())

    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)

        features = data.get("detected_features", [])
        total_analyzed = data.get("total_candidates_analyzed", "?")
        total_with_demand = data.get("total_candidates_with_demand", "?")

        print("\n" + "=" * 100)
        print("STRUCTURED SUMMARY")
        print("=" * 100)
        print("Total Hermes features analyzed: {}".format(total_analyzed))
        print("Features with community demand: {}".format(total_with_demand))
        print("Top candidates for porting: {}".format(len(features)))
        print()

        for feat in features:
            rank = feat.get("rank", "?")
            name = feat.get("feature_name", "Unknown")
            score = feat.get("overall_score", "?")
            category = feat.get("category", "?")
            oc_status = feat.get("openclaw_status", "?")
            complexity = feat.get("porting_complexity", "?")
            demand = feat.get("community_demand_score", "?")
            gap = feat.get("gap_severity_score", "?")
            impact = feat.get("impact_score", "?")
            feasibility = feat.get("feasibility_score", "?")
            summary = feat.get("summary", "")

            print("-" * 100)
            print("#{} | {} | Overall Score: {}".format(rank, name, score))
            print("   Category: {} | OpenClaw Status: {} | Complexity: {}".format(
                category, oc_status, complexity,
            ))
            print("   Scores: demand={}, gap={}, impact={}, feasibility={}".format(
                demand, gap, impact, feasibility,
            ))
            if summary:
                print("   Summary: {}".format(summary))

            issues = feat.get("openclaw_issues", [])
            if issues:
                issue_strs = []
                for iss in issues[:5]:
                    num = iss.get("number", "?")
                    title = iss.get("title", "")
                    reactions = iss.get("reactions", 0)
                    issue_strs.append("#{} ({} reactions): {}".format(
                        num, reactions, title[:60],
                    ))
                print("   Community Issues:")
                for s in issue_strs:
                    print("     - {}".format(s))

            hermes_files = feat.get("key_hermes_files", [])
            if hermes_files:
                print("   Key Hermes Files: {}".format(", ".join(hermes_files[:5])))
            print()

        print("=" * 100)
        print()
        print("Results saved to:")
        print("  Report: {}".format(report_path))
        print("  JSON:   {}".format(json_path))
        print()
        print("To run the full porting pipeline for a specific feature:")
        print("  python -m openclaw_vs_hermes.run_agent --config {} --feature-id <N>".format(
            "<config_path>"
        ))
    else:
        print("  ERROR: No structured results found at {}".format(json_path))

    print("=" * 100)


async def run_detection(config, claude_config, use_case, resume=False):
    output_dir = config.output_dir
    context = use_case.create_context_str(config)
    all_step_timings = []

    detection_steps = use_case.gen_feature_detection_steps(context, config)

    if resume and _all_steps_complete(detection_steps, output_dir):
        print("Resume: all feature detection steps already complete.")
    else:
        timings = await claude_run(claude_config, detection_steps)
        all_step_timings.extend(timings)

    return all_step_timings


if __name__ == "__main__":
    args = parse_args()

    config = AgentConfig.from_json(args.config)
    claude_config = config.make_claude_config()
    use_case = FeaturePortingUseCase()

    setup_logging("agent_detect")

    os.makedirs(config.output_dir, exist_ok=True)
    if not args.resume:
        safe_clean_dir(config.output_dir)

    print("\n" + "=" * 100)
    print("HERMES -> OPENCLAW FEATURE DETECTION")
    print("=" * 100)
    print("Hermes source:  {}".format(config.hermes_source_dir))
    print("OpenClaw source: {}".format(config.openclaw_source_dir))
    print("Output dir:     {}".format(config.output_dir))
    print("Max features:   {}".format(config.max_features_to_detect))
    print("Thinking mode:  {}".format(config.thinking_mode))
    print("=" * 100 + "\n")

    start_time = time.time()
    all_step_timings = asyncio.run(
        run_detection(config, claude_config, use_case, resume=args.resume)
    )
    total_duration = time.time() - start_time

    print_results(config.output_dir)

    # Print timing summary
    total_input = sum(s["input_tokens"] for s in all_step_timings)
    total_output = sum(s["output_tokens"] for s in all_step_timings)
    total_cost = sum(s["cost_usd"] for s in all_step_timings)

    print("\n--- Timing Summary ---")
    print("Total duration: {}".format(_format_duration(total_duration)))
    print("Input tokens:   {}".format(_format_tokens(total_input)))
    print("Output tokens:  {}".format(_format_tokens(total_output)))
    print("Total cost:     ${:.4f}".format(total_cost))
    for s in all_step_timings:
        print("  {}: {} | in={} out={} cost=${:.4f}".format(
            s["name"],
            _format_duration(s["duration"]),
            _format_tokens(s["input_tokens"]),
            _format_tokens(s["output_tokens"]),
            s["cost_usd"],
        ))
