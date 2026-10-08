"""Generate PR description files and command scripts from a successful pipeline run.

Reads the pipeline output files (performance results, lm_eval scores,
code port plan, iteration history) and generates:
1. commit_desc.txt and pr_desc.txt — the up-to-date commit and PR descriptions
2. Individual shell scripts in a pr-generate/ subdirectory for each PR step

The user reviews the scripts and runs them one by one — no git or gh
commands are executed by this tool.

Can run as Phase 6 of the full pipeline (when generate_pr_commands=true
in the config) or standalone after a successful run:

    python -m llm.auto_code_gen.run_generate_pr_commands --config <config.json>

To regenerate only the description files (after additional fixes/iterations):

    python -m llm.auto_code_gen.run_generate_pr_commands --config <config.json> --descs-only

Requires runtime_success_result.txt to exist in output_dir (i.e., the
runtime iterations must have succeeded).
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import sys
import time
import argparse
import asyncio

from common.utils import setup_logging

from llm.auto_code_gen.code_gen_configs import CodeGenConfig
from llm.auto_code_gen.use_cases.llm_framework import LLMFrameworkUseCase

from llm.auto_code_gen.code_gen_prompts import (
    RUNTIME_DIR,
    RUNTIME_SUCCESS_FILE,
    PR_COMMANDS_DIR,
    PR_COMMIT_DESC_FILE,
    PR_PR_DESC_FILE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PR description files and command scripts (standalone mode)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--descs-only",
        action="store_true",
        help="Only regenerate commit_desc.txt and pr_desc.txt (skip script generation).",
    )
    from llm.auto_code_gen.agents import add_agent_argument
    add_agent_argument(parser)
    return parser.parse_args()


async def run_standalone_generate_pr_commands(code_gen_config, claude_config,
                                              descs_only=False):
    output_dir = code_gen_config.output_dir
    use_case = LLMFrameworkUseCase()
    context = use_case.create_context_str(claude_config, code_gen_config)

    has_success = os.path.isfile(
        os.path.join(output_dir, RUNTIME_DIR, RUNTIME_SUCCESS_FILE)
    )
    if not has_success:
        print("ERROR: No successful runtime iterations found (no {})".format(
            RUNTIME_SUCCESS_FILE
        ))
        print("  Run the pipeline first and ensure runtime iterations succeed.")
        sys.exit(1)

    pr_dir = os.path.join(output_dir, PR_COMMANDS_DIR)

    if descs_only:
        print("\n" + "=" * 80)
        print("REGENERATE PR DESCRIPTIONS")
        print("=" * 80)
        print("  Output directory: {}".format(output_dir))
        print("  PR dir: {}".format(pr_dir))
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("GENERATE PR COMMANDS")
        print("=" * 80)
        print("  Output directory: {}".format(output_dir))
        print("  PR scripts dir: {}".format(pr_dir))
        print("  Target repo: {}".format(code_gen_config.source_code_dir))
        print("  Push: {}/{}".format(code_gen_config.pr_remote, code_gen_config.get_target_branch_name()))
        print("  PR: {} <- {}".format(code_gen_config.pr_base_branch, code_gen_config.get_target_branch_name()))
        print("=" * 80 + "\n")

    phase_results, all_step_timings = await use_case.run_generate_pr_commands(
        context, code_gen_config, claude_config, descs_only=descs_only,
    )

    return phase_results, all_step_timings


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_agent_config(args.agent_config)

    setup_logging("generate_pr_commands")

    start_time = time.time()
    phase_results, all_step_timings = asyncio.run(
        run_standalone_generate_pr_commands(
            code_gen_config, claude_config, descs_only=args.descs_only,
        )
    )
    total_duration = time.time() - start_time

    pr_dir = os.path.join(code_gen_config.output_dir, PR_COMMANDS_DIR)

    if args.descs_only:
        for f in [PR_COMMIT_DESC_FILE, PR_PR_DESC_FILE]:
            path = os.path.join(pr_dir, f)
            if os.path.isfile(path):
                print("  Updated: {}".format(path))
    else:
        if os.path.isdir(pr_dir) and os.listdir(pr_dir):
            print("\nPR command scripts generated in: {}".format(pr_dir))
            scripts = sorted(f for f in os.listdir(pr_dir) if f.endswith(".sh"))
            for s in scripts:
                print("  {}".format(s))
            print("\nRun them in order:")
            for s in scripts:
                print("  bash {}/{}".format(pr_dir, s))
        else:
            print("\nWARNING: PR command scripts were not generated.")

    print("Duration: {:.1f}s".format(total_duration))
