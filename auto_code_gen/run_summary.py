import sys
import time
import argparse
import asyncio

from pathlib import Path
from common.utils import setup_logging
from common.claude_utils import claude_run

from auto_code_gen.code_gen_configs import CodeGenConfig
from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_SummarizeCodeGenProcessPrompt,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize code gen process")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--framework-code-trace-files",
        nargs="+",
        required=True,
        help="List of framework code trace file paths.",
    )
    parser.add_argument(
        "--code-port-plan-file",
        required=True,
        type=Path,
        help="Path to the code port plan file.",
    )
    parser.add_argument(
        "--test-plan-file",
        required=True,
        type=Path,
        help="Path to the test plan file.",
    )
    parser.add_argument(
        "--code-port-plan-review-evolution-file",
        required=True,
        type=Path,
        help="Path to the code port plan review evolution file.",
    )
    parser.add_argument(
        "--code-pr-info-file",
        required=True,
        type=Path,
        help="Path to the code PR info file.",
    )
    parser.add_argument(
        "--code-pr-file",
        required=True,
        type=Path,
        help="Path to the code PR file.",
    )

    parser.add_argument(
        "--code-pr-review-evolution-file",
        required=True,
        type=Path,
        help="Path to the code PR review evolution file.",
    )

    parser.add_argument(
        "--issue-desc-files",
        nargs="+",
        required=True,
        help="Issue description files.",
    )
    parser.add_argument(
        "--issue-fix-review-evolution-files",
        nargs="+",
        required=True,
        help="Issue review evolution files.",
    )

    parser.add_argument(
        "--auto-analyze-project-brief",
        required=True,
        type=Path,
        help="Path to the auto-analyze presentation brief.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Path to the output file",
    )
    return parser.parse_args()


def gen_prompts(args, code_gen_config, claude_config):
    context = create_context_str(claude_config, code_gen_config)

    summary_prompt = gen_SummarizeCodeGenProcessPrompt(
        context=context,
        framework_code_trace_files=args.framework_code_trace_files,
        code_port_plan_file=args.code_port_plan_file,
        test_plan_file=args.test_plan_file,
        code_port_plan_review_evolution_file=args.code_port_plan_review_evolution_file,
        code_pr_info_file=args.code_pr_info_file,
        code_pr_file=args.code_pr_file,
        code_pr_review_evolution_file=args.code_pr_review_evolution_file,
        issue_desc_files=args.issue_desc_files,
        issue_fix_review_evolution_files=args.issue_fix_review_evolution_files,
        auto_analyze_project_brief=args.auto_analyze_project_brief,
        output_file=args.output_file
    )

    return [summary_prompt.prompt()]


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_claude_config()

    setup_logging("summary")

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(args, code_gen_config, claude_config)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
