"""
JIRA Task Creator
==================

Creates JIRA tasks from analysis improvement proposals. Each improvement
proposal becomes a sub-task under a master JIRA task.

Supports two input modes:
  (A) Single-trace: create tasks from perf_analysis_single_trace.txt proposals
  (B) Cross-trace:  create tasks from cross_compare_blocks.txt improvement plan

Usage:
    python -m llm.auto_analyze.run_jiras --single-config <config.json>
    python -m llm.auto_analyze.run_jiras --cross-config <config.json>

Arguments:
    --single-config   Path to single-trace config JSON (mode A)
    --cross-config    Path to cross-trace config JSON (mode B)
    --agent-config    Optional agent config JSON (JIRA MCP currently requires Claude SDK)

    Exactly one of --single-config or --cross-config is required.
    Requires JIRA MCP server to be configured.

Output:
    jira_tasks_output.txt in the analysis results directory.
"""

import sys as _sys
import os as _os
_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import sys
import time
import asyncio
import argparse

from common.utils import setup_logging
from common.claude_utils import PipelineStep
from llm.agents import agent_run
from llm.auto_analyze.agent_support import add_agent_arguments, make_agent_config

from llm.auto_analyze.prompts.jira_prompts import JiraTasksPrompt, JIRA_TASKS_FILE
from llm.auto_analyze.configs.single_trace_config import PERF_ANALYSIS_SINGLE_FILE
from llm.auto_analyze.configs.cross_trace_config import CROSS_COMPARE_FILE
from common.iteration_utils import resolve_output_file


if __name__ == "__main__":
    setup_logging("create_jiras")

    parser = argparse.ArgumentParser(
        description="Create JIRA tasks from analysis improvement proposals",
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
    add_agent_arguments(parser)
    args = parser.parse_args()

    if args.single_config:
        from llm.auto_analyze.configs.single_trace_config import SingleTraceConfig
        config = SingleTraceConfig.from_json(args.single_config)
        results_dir = os.path.abspath(config.output_dir)
        mode = "single"
        run_params_file = os.path.join(results_dir, "run_params.txt")
        analysis_file = resolve_output_file(results_dir, PERF_ANALYSIS_SINGLE_FILE)
    else:
        from llm.auto_analyze.configs.cross_trace_config import CrossTraceConfig
        config = CrossTraceConfig.from_json(args.cross_config)
        config.load_all_trace_params()
        results_dir = os.path.abspath(config.output_dir)
        mode = "cross"
        run_params_file = os.path.join(results_dir, "run_params_cross.txt")
        analysis_file = resolve_output_file(results_dir, CROSS_COMPARE_FILE)

    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    if not os.path.exists(analysis_file):
        print(f"Error: analysis file not found: {analysis_file}")
        sys.exit(1)

    output_file = os.path.join(results_dir, JIRA_TASKS_FILE)

    print("=" * 70)
    print("  JIRA TASK CREATION")
    print("=" * 70)
    print(f"  Mode:           {mode}")
    print(f"  Results dir:    {results_dir}")
    print(f"  Analysis file:  {analysis_file}")
    print(f"  Output file:    {output_file}")
    print("=" * 70)

    prompt_obj = JiraTasksPrompt(
        results_dir=results_dir,
        mode=mode,
        run_params_file=run_params_file,
        analysis_file=analysis_file,
        output_file=output_file,
    )

    agent_config = make_agent_config(config, results_dir, args.agent_config)
    if agent_config.backend != "claude_sdk":
        raise SystemExit(
            "JIRA task creation requires the Claude SDK backend because the "
            "generic agent runner does not expose MCP tools."
        )
    agent_config.allowed_tools.append("mcp__mcp-atlassian__*")

    start_time = time.time()
    print(f"\nStarting JIRA task creation at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    step = PipelineStep(
        name="jira_tasks",
        prompt=prompt_obj.prompt(),
        output_files=[os.path.basename(output_file)],
    )
    asyncio.run(agent_run(agent_config, [step]))

    duration = time.time() - start_time
    mins, secs = divmod(int(duration), 60)
    print(f"\nJIRA task creation completed in {duration:.1f}s ({mins}m {secs}s)")

    if os.path.exists(output_file):
        print(f"  [OK] {output_file}")
    else:
        print(f"  [MISSING] {output_file}")
