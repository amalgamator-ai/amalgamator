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

from common.utils import setup_logging, safe_clean_dir, output_dir_from_run_config, PROFILE_RESULTS_DIR
from common.claude_utils import claude_run

from common.claude_utils import ProfilingClaudeConfig
from llm.auto_profile.parse_prompts import ParseResultsPrompt

claude_config = ProfilingClaudeConfig(
    model="claude-opus-4-6",
    # model="claude-opus-4-5",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",  # "bypassPermissions",
    cwd=None,
)


def gen_parse_results_prompts(results_dir, output_dir):
    parse_results_prompt = ParseResultsPrompt(
        results_dir=results_dir,
        output_dir=output_dir,
    )

    prompts = [parse_results_prompt.prompt()]
    return prompts


if __name__ == "__main__":
    setup_logging("parse")

    # Parse args
    parser = argparse.ArgumentParser(description="Parse auto_profile results")

    parser.add_argument(
        "--run-config",
        required=True,
        type=str,
        help="Path to run config JSON file (run_<test_name>.json)",
    )
    parser.add_argument(
        "--override-output-dir",
        action="store_true",
        help="If set, allow output directory to exist and clean it before proceeding",
    )
    args = parser.parse_args()

    output_dir = output_dir_from_run_config(args.run_config)
    if os.path.exists(output_dir):
        if args.override_output_dir:
            safe_clean_dir(output_dir)
        else:
            print(f"Error: output directory already exists: {output_dir}")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Set CWD
    claude_config.cwd = os.getcwd()

    # Run
    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_parse_results_prompts(PROFILE_RESULTS_DIR, output_dir)))
    duration_time = time.time() - start_time
    print("FINISHED ALL: total_duration = {}".format(duration_time))
