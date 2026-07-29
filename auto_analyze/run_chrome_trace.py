"""
Chrome Trace JSON Generator
=============================

Generates Chrome trace JSON files for Perfetto visualization of median
transformer block(s).

Supports two input modes:
  (A) Single-trace: visualize one median transformer block from a single framework
  (B) Cross-trace:  visualize multiple median blocks side by side for comparison

Usage:
    python -m auto_analyze.run_chrome_trace --single-config <config.json>
    python -m auto_analyze.run_chrome_trace --cross-config <config.json>

Arguments:
    --single-config   Path to single-trace config JSON (mode A)
    --cross-config    Path to cross-trace config JSON (mode B)
    --claude-config   Path to Claude config JSON (default: auto_analyze/configs/claude_config.json)

    Exactly one of --single-config or --cross-config is required.

Outputs:
    Mode A (in single-trace output_dir):
      - single_trace_transformer_block.json   Chrome trace JSON
      - single_trace_transformer_block.txt    Human-readable summary

    Mode B (in cross-trace output_dir):
      - cross_trace_transformer_blocks.json   Chrome trace JSON with all traces
      - cross_trace_transformer_blocks.txt    Human-readable summary
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

from common.utils import setup_logging
from common.claude_utils import claude_run, ProfilingClaudeConfig

from auto_analyze.configs.single_trace_config import SingleTraceConfig
from auto_analyze.configs.cross_trace_config import CrossTraceConfig
from auto_analyze.prompts.chrome_trace_prompts import (
    build_single_trace_json_prompt,
    build_cross_trace_json_prompt,
)


def load_claude_config(claude_config_path, cwd):
    with open(claude_config_path) as f:
        data = json.load(f)
    return ProfilingClaudeConfig(
        model=data["model"],
        allowed_tools=data["allowed_tools"],
        perm_mode="acceptEdits",
        cwd=cwd,
    )


if __name__ == "__main__":
    setup_logging("trace_json")

    parser = argparse.ArgumentParser(
        description="Generate Chrome trace JSON for Perfetto visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--single-config",
        type=str,
        help="Path to single-trace config JSON (mode A)",
    )
    group.add_argument(
        "--cross-config",
        type=str,
        help="Path to cross-trace config JSON (mode B)",
    )
    parser.add_argument(
        "--claude-config",
        type=str,
        default="auto_analyze/configs/claude_config.json",
        help="Path to Claude config JSON file (default: auto_analyze/configs/claude_config.json)",
    )
    args = parser.parse_args()

    if args.single_config:
        config = SingleTraceConfig.from_json(args.single_config)
        prompt_obj = build_single_trace_json_prompt(config)
        mode = "single"

        print("=" * 70)
        print("  SINGLE TRACE JSON GENERATION")
        print("=" * 70)
        print(f"  Framework:     {config.framework_name}")
        print(f"  Model:         {config.model}")
        print(f"  Output dir:    {prompt_obj.output_dir}")
        print(f"  Output file:   {prompt_obj.output_json_file}")
        print("=" * 70)
    else:
        config = CrossTraceConfig.from_json(args.cross_config)
        config.load_all_trace_params()
        prompt_obj = build_cross_trace_json_prompt(config)
        mode = "cross"

        print("=" * 70)
        print("  CROSS TRACE JSON GENERATION")
        print("=" * 70)
        print(f"  Output dir:    {prompt_obj.output_dir}")
        print(f"  Output file:   {prompt_obj.output_json_file}")
        print(f"  Traces:        {len(config.traces)}")
        for i, tr in enumerate(config.traces):
            marker = " (*)" if i == config.target_trace_id else ""
            print(f"    [{i}] {tr.trace_id}{marker}: {tr.result_dir}")
        print("=" * 70)

    output_dir = prompt_obj.output_dir
    output_file = prompt_obj.output_json_file

    claude_config = load_claude_config(args.claude_config, output_dir)

    start_time = time.time()
    print(f"\nStarting trace JSON generation ({mode}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    asyncio.run(claude_run(claude_config, [prompt_obj.prompt()]))

    duration = time.time() - start_time
    mins, secs = divmod(int(duration), 60)
    print(f"\nTrace JSON generation completed in {duration:.1f}s ({mins}m {secs}s)")

    if os.path.exists(output_file):
        print(f"  [OK] {output_file}")
    else:
        print(f"  [MISSING] {output_file}")
