import sys
import time
import argparse
import asyncio

from pathlib import Path
from common.utils import setup_logging
from common.claude_utils import claude_run

from auto_code_gen.code_gen_configs import CodeGenConfig
from auto_code_gen.code_gen_prompts import gen_FixIssuePrompt



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix issue in PR"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--high_level_code_plan_file",
        required=True,
        type=Path,
        help="Path to the high-level code plan file.",
    )
    parser.add_argument(
        "--prs_dir",
        required=True,
        type=Path,
        help="Path to the directory containing PR files.",
    )
    parser.add_argument(
        "--issue_to_fix_file",
        required=True,
        type=Path,
        help="Path to the issue-to-fix file.",
    )

    parser.add_argument(
        "--issue_cwd",
        required=True,
        type=Path,
        help="Path to the issue CWD.",
    )
    return parser.parse_args()


def gen_prompts(args, code_gen_config, claude_config):
    fix_issue_prompt = gen_FixIssuePrompt(
        claude_config=claude_config,
        code_gen_config=code_gen_config,
        high_level_code_plan_file=args.high_level_code_plan_file,
        prs_dir=args.prs_dir,
        issue_to_fix_file=args.issue_to_fix_file,
        issue_cwd=args.issue_cwd,
    )

    prompts = [fix_issue_prompt.prompt()]
    return prompts


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_claude_config()

    setup_logging("fix_issue")

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(args, code_gen_config, claude_config)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
