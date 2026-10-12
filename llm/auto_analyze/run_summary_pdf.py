"""
Summary PDF Report Generator
==============================

Generates a professional PDF report from analysis results.

Supports two input modes:
  (A) Single-trace: report from one framework's perf analysis
  (B) Cross-trace:  report from cross-framework/cross-commit comparison

Usage:
    python -m llm.auto_analyze.run_summary_pdf --single-config <config.json>
    python -m llm.auto_analyze.run_summary_pdf --cross-config <config.json>

Arguments:
    --single-config   Path to single-trace config JSON (mode A)
    --cross-config    Path to cross-trace config JSON (mode B)
    --agent-config    Optional provider-neutral agent config JSON

    Exactly one of --single-config or --cross-config is required.

Output:
    summary_report.pdf in the analysis results directory.
    - Mode A reads: run_params.txt, perf_analysis_single_trace.txt, median_block.txt, etc.
    - Mode B reads: run_params_cross.txt, cross_compare_blocks.txt, cross_matching_blocks.txt
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

from common.claude_utils import PipelineStep
from common.utils import setup_logging
from llm.agents import agent_run
from llm.auto_analyze.agent_support import add_agent_arguments, make_agent_config
from llm.auto_analyze.prompts.summary_pdf_prompts import (
    SUMMARY_PDF_FILE,
    SummaryPDFPrompt,
)

if __name__ == "__main__":
    setup_logging("summary_pdf")

    parser = argparse.ArgumentParser(
        description="Generate summary PDF report from analysis results",
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
    else:
        from llm.auto_analyze.configs.cross_trace_config import CrossTraceConfig

        config = CrossTraceConfig.from_json(args.cross_config)
        config.load_all_trace_params()
        results_dir = os.path.abspath(config.output_dir)
        mode = "cross"

    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    output_file = os.path.join(results_dir, SUMMARY_PDF_FILE)

    print("=" * 70)
    print("  SUMMARY PDF GENERATION")
    print("=" * 70)
    print(f"  Mode:          {mode}")
    print(f"  Results dir:   {results_dir}")
    print(f"  Output file:   {output_file}")
    print("=" * 70)

    prompt_obj = SummaryPDFPrompt(
        results_dir=results_dir,
        mode=mode,
        output_file=output_file,
    )

    agent_config = make_agent_config(config, results_dir, args.agent_config)

    start_time = time.time()
    print(f"\nStarting PDF generation at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    step = PipelineStep(
        name="summary_pdf",
        prompt=prompt_obj.prompt(),
        output_files=[os.path.basename(output_file)],
    )
    asyncio.run(agent_run(agent_config, [step]))

    duration = time.time() - start_time
    mins, secs = divmod(int(duration), 60)
    print(f"\nPDF generation completed in {duration:.1f}s ({mins}m {secs}s)")

    if os.path.exists(output_file):
        print(f"  [OK] {output_file}")
    else:
        print(f"  [MISSING] {output_file}")
