# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Redo cross-trace analysis with accumulated code-gen learnings.

Re-analyzes performance gaps using the original trace data PLUS learnings
from code-gen runs, runtime iterations, diagnostic profiles, and the
knowledge base. Produces an updated improvement plan with corrected
estimates, implemented proposals marked as done, and new proposals for
emerging bottlenecks.

Usage:
    python -m llm.auto_analyze.run_redo_cross_trace \
        --config <cross_trace_config.json> \
        --code-gen-dirs <dir1> [<dir2> ...]
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import asyncio
import os
import time

from common.utils import setup_logging
from llm.agents import agent_run, agent_session
from llm.auto_analyze.agent_support import (
    add_agent_arguments,
    log_agent_config,
    make_agent_config,
)
from llm.auto_analyze.configs.cross_trace_config import CrossTraceConfig
from llm.auto_analyze.prompts.cross_trace_prompts import (
    REDO_IMPROVEMENT_PREFIX,
    gen_redo_cross_trace_steps,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redo cross-trace analysis with code-gen learnings"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the cross-trace config JSON file.",
    )
    parser.add_argument(
        "--code-gen-dirs",
        nargs="+",
        required=True,
        help="Paths to code-gen output directories (one per attempted proposal).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration number for versioned output (auto-detected if omitted).",
    )
    parser.add_argument(
        "--full-decode-step-analysis",
        action="store_true",
        help="Enable full decode step analysis to explain the complete TPOT difference.",
    )
    add_agent_arguments(parser)
    return parser.parse_args()


def _find_next_iteration(output_dir):
    """Find the next redo iteration number."""
    import glob
    import re

    max_ver = 0
    for path in glob.glob(
        os.path.join(output_dir, f"{REDO_IMPROVEMENT_PREFIX}_V*.txt")
    ):
        basename = os.path.basename(path)
        if "_summary_" in basename or "_review_" in basename:
            continue
        match = re.search(r"_V(\d+)\.txt$", basename)
        if match:
            ver = int(match.group(1))
            if ver > max_ver:
                max_ver = ver
    return max_ver + 1


if __name__ == "__main__":
    args = parse_args()

    setup_logging("redo_cross_trace")

    config = CrossTraceConfig.from_json(args.config)
    config.load_all_trace_params()

    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    for d in args.code_gen_dirs:
        if not os.path.isdir(d):
            print(f"ERROR: code-gen directory not found: {d}")
            raise SystemExit(1)

    iteration = args.iteration
    if iteration is None:
        iteration = _find_next_iteration(os.path.abspath(config.output_dir))

    steps, output_files = gen_redo_cross_trace_steps(
        config=config,
        code_gen_dirs=args.code_gen_dirs,
        iteration=iteration,
        full_decode_step_analysis=args.full_decode_step_analysis,
    )

    print("=" * 70)
    print("  REDO CROSS-TRACE ANALYSIS")
    print("=" * 70)
    print(f"  Config:       {args.config}")
    print(f"  Output dir:   {os.path.abspath(config.output_dir)}")
    print(f"  Iteration:    V{iteration}")
    print("  Code-gen dirs:")
    for d in args.code_gen_dirs:
        print(f"    - {d}")
    print("=" * 70)

    agent_config = make_agent_config(
        config, os.path.abspath(config.output_dir), args.agent_config
    )
    log_agent_config(agent_config)

    async def _main():
        async with agent_session(agent_config) as client:
            return await agent_run(agent_config, steps, client=client)

    start_time = time.time()
    timings = asyncio.run(_main())
    duration = time.time() - start_time

    mins, secs = divmod(int(duration), 60)
    print("\n" + "=" * 70)
    print(f"  REDO COMPLETE — {mins}m {secs}s")
    print("=" * 70)
    for f in output_files:
        if os.path.isfile(f):
            print(f"  [OK] {os.path.basename(f)}")
        else:
            print(f"  [MISSING] {os.path.basename(f)}")
    print("=" * 70)
