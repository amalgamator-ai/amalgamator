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
    gen_WorkItemsPrompt,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute work items")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--code-gen-dir",
        required=True,
        type=Path,
        help="Path to the code generation results directory.",
    )

    parser.add_argument(
        "--work-items-file",
        required=True,
        type=Path,
        help="Path to the work items file.",
    )

    return parser.parse_args()


def gen_prompts(args, code_gen_config, claude_config):
    context = create_context_str(claude_config, code_gen_config)

    work_items_prompt = gen_WorkItemsPrompt(
        context=context,
        code_gen_dir=args.code_gen_dir,
        work_items_file=args.work_items_file,
    )

    return [work_items_prompt.prompt()]


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_claude_config()
    claude_config.cwd = str(args.code_gen_dir)

    setup_logging("work_items")

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(args, code_gen_config, claude_config)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
