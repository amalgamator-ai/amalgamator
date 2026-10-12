# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Manually (re)generate the PR title and description files.

Thin entrypoint around llm.auto_code_gen.pr_flow.pr_descs.update_pr_descs, which is
the same refresh the runtime-iteration runners trigger automatically after a
passing iteration. Use it to re-run a failed refresh or to force an update:

    python -m llm.auto_code_gen.run_pr_descs_update --config <code_gen.json>

Creates pr-generate/commit_desc.txt (first line = PR title) and
pr-generate/pr_desc.txt if absent, updates them in place otherwise. No git or
gh commands are run; the publish scripts pick the files up (step 2 at PR
creation, step 4 via gh pr edit whenever the content changed).

Requires a passing result: runtime/success_result.txt (normal pipeline) or a
runtime/V<n>/verify_verdict.json with overall == PASS (PR flow).

Exit codes:
    0   descriptions written/updated
    1   no passing result, or the agent wrote nothing
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import asyncio
import sys
import time

from common.utils import setup_logging
from llm.auto_code_gen.code_gen_configs import load_config_and_use_case
from llm.auto_code_gen.pr_flow.pr_descs import PrDescsError, update_pr_descs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or update commit_desc.txt / pr_desc.txt "
        "(PR title and body) from the latest passing results."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    from llm.auto_code_gen.agents import add_agent_argument

    add_agent_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config, use_case = load_config_and_use_case(args.config)
    claude_config = config.make_agent_config(args.agent_config)

    setup_logging("pr_descs_update")

    start_time = time.time()
    try:
        desc_path, body_path = asyncio.run(
            update_pr_descs(config, claude_config, use_case)
        )
    except PrDescsError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Duration: {time.time() - start_time:.1f}s")
    if not desc_path:
        return 1

    for label, path in (("title+commit message", desc_path), ("PR body", body_path)):
        if path:
            print(f"  {label}: {path}")
    print("\nThe publish steps pick these up automatically:")
    print(
        "  bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_2_push.sh "
        f"--config {args.config} --remote-branch perf/<short-name>   # PR not yet created"
    )
    print(
        "  bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_4_update_pr.sh "
        f"--config {args.config}                                     # PR already published"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
