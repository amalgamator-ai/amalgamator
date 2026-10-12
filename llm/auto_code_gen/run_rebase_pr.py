"""Rebase the patched branch onto the latest main branch.

Creates a backup of the current branch, rebases onto the latest main,
resolves conflicts, verifies compilation, and saves the rebased patch
to the output directory.

Usage:

    python -m llm.auto_code_gen.run_rebase_pr \
        --config <config.json> \
        --latest-main-branch <branch_name>

The latest-main-branch is typically created by pr_gen_step_7_get_latest_main.sh
and stored in pr-generate/latest_main_branch.txt.

Requires the patched branch to have a clean committed state (run the
commit step first).
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

from common.claude_utils import PipelineStep
from common.utils import setup_logging
from llm.auto_code_gen.agents import agent_run
from llm.auto_code_gen.code_gen_configs import CodeGenConfig
from llm.auto_code_gen.code_gen_prompts import (
    gen_RebasePRPrompt,
)
from llm.auto_code_gen.use_cases.llm_framework import LLMFrameworkUseCase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebase the patched branch onto the latest main branch."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--latest-main-branch",
        required=True,
        help="Name of the local branch holding the latest fetched main "
        "(created by pr_gen_step_7_get_latest_main.sh).",
    )
    from llm.auto_code_gen.agents import add_agent_argument

    add_agent_argument(parser)
    return parser.parse_args()


async def run_rebase(code_gen_config, claude_config, latest_main_branch):
    use_case = LLMFrameworkUseCase()
    context = use_case.create_context_str(claude_config, code_gen_config)

    branch_name = code_gen_config.get_target_branch_name()

    print("\n" + "=" * 80)
    print("REBASE PR")
    print("=" * 80)
    print(f"  Target repo: {code_gen_config.source_code_dir}")
    print(f"  Patch branch: {branch_name}")
    print(f"  Latest main branch: {latest_main_branch}")
    print(f"  Output dir: {code_gen_config.output_dir}")
    print("=" * 80 + "\n")

    prompt = gen_RebasePRPrompt(
        context=context,
        target_repo_dir=code_gen_config.source_code_dir,
        branch_name=branch_name,
        pr_base_branch=code_gen_config.pr_base_branch,
        pr_remote=code_gen_config.pr_remote,
        code_gen_output_dir=code_gen_config.output_dir,
        latest_main_branch=latest_main_branch,
    )

    steps = [
        PipelineStep(
            name="rebase_pr",
            prompt=prompt.prompt(),
        ),
    ]

    timings = await agent_run(claude_config, steps)
    return timings


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_agent_config(args.agent_config)

    setup_logging("rebase_pr")

    start_time = time.time()
    timings = asyncio.run(
        run_rebase(code_gen_config, claude_config, args.latest_main_branch)
    )
    total_duration = time.time() - start_time

    import re

    runtime_patches = sorted(
        [
            f
            for f in os.listdir(code_gen_config.output_dir)
            if re.match(r"^code_gen_runtime_V\d+\.patch$", f)
        ]
    )
    if runtime_patches:
        latest = runtime_patches[-1]
        print(
            f"\nRebase patch saved as runtime iteration: {os.path.join(code_gen_config.output_dir, latest)}"
        )
    else:
        print("\nWARNING: No rebase patch was generated.")

    print(f"Duration: {total_duration:.1f}s")
