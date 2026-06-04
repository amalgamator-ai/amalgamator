import sys
import time
import argparse
import asyncio

from pathlib import Path
from common.utils import setup_logging, clear_vllm_source_tree
from common.claude_utils import claude_run

from auto_code_gen.code_gen_configs import CodeGenConfig
from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_InvestigateIssuePrompt,
    gen_ReviewInvestigatedIssuePrompt,
)


NUM_ITERS = 2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate issue in PR")
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
        "--issue-desc-file",
        required=True,
        type=Path,
        help="Issue description file.",
    )
    parser.add_argument(
        "--output-file-prefix",
        required=True,
        type=Path,
        help="Path to the output file prefix",
    )
    return parser.parse_args()


def gen_prompts(args, code_gen_config, claude_config):
    clear_target_dir_cmd = {
        "fn": lambda: clear_vllm_source_tree(code_gen_config.source_code_dir),
        "fn_name": 'clear_vllm_source_tree("{}")'.format(code_gen_config.source_code_dir),
    }

    context = create_context_str(claude_config, code_gen_config)
    code_pr_file_prefix = str(args.code_pr_file).strip(".patch")

    prompts = []
    num_reviews = NUM_ITERS
    issue_fix_previous_attempt_file = ""
    issue_fix_previous_attempt_review_evolution_file = ""
    code_pr_file_prev = args.code_pr_file
    for i in range(num_reviews):
        issue_fix_file = "{}_fix_V{}.txt".format(args.output_file_prefix, i + 1)
        code_pr_fixed_file = "{}_V{}_FIXED.patch".format(code_pr_file_prefix, i + 1)
        code_pr_review_fixed_file = "{}_V{}_REVIEW_FIXED.patch".format(
            code_pr_file_prefix, i + 1
        )

        investigate_issue_prompt = gen_InvestigateIssuePrompt(
            context=context,
            code_trace_files=args.code_trace_files,
            code_port_plan_file=args.code_port_plan_file,
            test_plan_file=args.test_plan_file,
            code_port_plan_review_evolution_file=args.code_port_plan_review_evolution_file,
            code_pr_info_file=args.code_pr_info_file,
            code_pr_file=code_pr_file_prev,
            code_pr_review_evolution_file=args.code_pr_review_evolution_file,
            issue_desc_file=args.issue_desc_file,
            issue_fix_previous_attempt_file=issue_fix_previous_attempt_file,
            issue_fix_previous_attempt_review_evolution_file=issue_fix_previous_attempt_review_evolution_file,
            issue_fix_file=issue_fix_file,
            code_pr_fixed_file=code_pr_fixed_file,
        )
        code_pr_file_prev = code_pr_fixed_file

        issue_fix_review_file = "{}_fix_V{}_REVIEW.txt".format(
            args.output_file_prefix, i + 1
        )
        issue_fix_fixed_file = "{}_fix_V{}_FIXED.txt".format(
            args.output_file_prefix, i + 1
        )

        issue_fix_review_evolution_file = "{}_fix_V{}_REVIEW_EVOLUTION.txt".format(
            args.output_file_prefix, i + 1
        )

        review_investigated_issue_prompt = gen_ReviewInvestigatedIssuePrompt(
            context=context,
            code_trace_files=args.code_trace_files,
            code_port_plan_file=args.code_port_plan_file,
            test_plan_file=args.test_plan_file,
            code_port_plan_review_evolution_file=args.code_port_plan_review_evolution_file,
            code_pr_info_file=args.code_pr_info_file,
            code_pr_file=code_pr_file_prev,
            code_pr_review_evolution_file=args.code_pr_review_evolution_file,
            issue_desc_file=args.issue_desc_file,
            issue_fix_file=issue_fix_file,
            issue_fix_review_file=issue_fix_review_file,
            issue_fix_fixed_file=issue_fix_fixed_file,
            issue_fix_review_evolution_file=issue_fix_review_evolution_file,
            code_pr_review_fixed_file=code_pr_review_fixed_file,
        )
        code_pr_file_prev = code_pr_review_fixed_file

        issue_fix_previous_attempt_file = issue_fix_fixed_file
        issue_fix_previous_attempt_review_evolution_file = (
            issue_fix_review_evolution_file
        )

        prompts.append(investigate_issue_prompt.prompt())
        prompts.append(review_investigated_issue_prompt.prompt())

    return prompts


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_claude_config()

    setup_logging("investigate_issue")

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(args, code_gen_config, claude_config)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
