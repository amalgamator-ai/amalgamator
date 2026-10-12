"""Fix PR review comments using Claude.

Reads the review comments fetched by step2, uses Claude Agent SDK to
analyze each comment, make code fixes in the repository, and commit
each logical fix separately.

Usage:

    python -m llm.auto_code_gen.pr_review.step3_fix_review_comments \
        --output-dir /path/to/pr_review_output \
        --repo-dir /path/to/vllm \
        --mode deep
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import asyncio
import os
import sys
import time

from common.claude_utils import ClaudeConfig, PipelineStep
from common.utils import setup_logging
from llm.auto_code_gen.agents import agent_run
from llm.auto_code_gen.code_gen_configs import CodeGenConfig
from llm.auto_code_gen.pr_review.prompts import gen_FixReviewCommentsPrompt
from llm.auto_code_gen.pr_review.step2_fetch_review_comments import (
    PR_METADATA_FILE,
    REVIEW_COMMENTS_FILE,
    REVIEW_COMMENTS_SUMMARY_FILE,
)

FIX_SUMMARY_FILE = "fix_summary.txt"

_MODE_PARAMS = CodeGenConfig._MODE_PARAMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix PR review comments using Claude.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing step2 outputs (review_comments_summary.txt, etc.).",
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Path to the local git repository with the PR branch checked out.",
    )
    parser.add_argument(
        "--mode",
        default="deep",
        choices=["deep", "normal"],
        help="Thinking mode: 'deep' (Opus, max effort) or 'normal' (Sonnet, medium). Default: deep.",
    )
    from llm.auto_code_gen.agents import add_agent_argument

    add_agent_argument(parser)
    return parser.parse_args()


def make_claude_config(repo_dir: str, mode: str = "deep") -> ClaudeConfig:
    params = _MODE_PARAMS[mode]
    return ClaudeConfig(
        model=params["model"],
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=repo_dir,
        thinking=params["thinking"],
        effort=params["effort"],
        max_thinking_tokens=params["max_thinking_tokens"],
    )


def _validate_inputs(output_dir: str, repo_dir: str):
    errors = []

    if not os.path.isdir(output_dir):
        errors.append(f"Output directory not found: {output_dir}")

    required_files = [
        PR_METADATA_FILE,
        REVIEW_COMMENTS_FILE,
        REVIEW_COMMENTS_SUMMARY_FILE,
    ]
    for f in required_files:
        path = os.path.join(output_dir, f)
        if not os.path.isfile(path):
            errors.append(f"Required file not found: {path}")
        elif os.path.getsize(path) == 0:
            errors.append(f"Required file is empty: {path}")

    if not os.path.isdir(repo_dir):
        errors.append(f"Repository directory not found: {repo_dir}")

    if errors:
        print("ERROR: Cannot proceed:")
        for e in errors:
            print(f"  - {e}")
        print("\nRun step2_fetch_review_comments first to generate the required files.")
        sys.exit(1)


async def run_fix_review_comments(
    output_dir: str, repo_dir: str, mode: str = "deep", agent_config_path=None
):
    output_dir = os.path.abspath(output_dir)
    repo_dir = os.path.abspath(repo_dir)

    _validate_inputs(output_dir, repo_dir)

    from llm.auto_code_gen.agents import resolve_agent_config

    claude_config = resolve_agent_config(
        cwd=repo_dir, mode=mode, agent_config_path=agent_config_path
    )

    print("\n" + "=" * 70)
    print("FIX PR REVIEW COMMENTS")
    print("=" * 70)
    print(f"  Repo: {repo_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Mode: {mode} ({claude_config.model})")
    print("=" * 70 + "\n")

    prompt_obj = gen_FixReviewCommentsPrompt(
        repo_dir=repo_dir,
        review_comments_summary_file=os.path.join(
            output_dir, REVIEW_COMMENTS_SUMMARY_FILE
        ),
        review_comments_json_file=os.path.join(output_dir, REVIEW_COMMENTS_FILE),
        pr_metadata_file=os.path.join(output_dir, PR_METADATA_FILE),
        output_dir=output_dir,
    )

    steps = [
        PipelineStep(
            name="fix_review_comments",
            prompt=prompt_obj.prompt(),
            output_files=[FIX_SUMMARY_FILE],
        ),
    ]

    timings = await agent_run(claude_config, steps)
    return timings


if __name__ == "__main__":
    args = parse_args()

    setup_logging("pr_review_fix")

    start_time = time.time()
    timings = asyncio.run(
        run_fix_review_comments(
            args.output_dir, args.repo_dir, args.mode, args.agent_config
        )
    )
    total_duration = time.time() - start_time

    fix_summary_path = os.path.join(os.path.abspath(args.output_dir), FIX_SUMMARY_FILE)
    if os.path.isfile(fix_summary_path):
        print(f"\nFix summary: {fix_summary_path}")
    else:
        print("\nWARNING: Fix summary was not generated.")

    mins, secs = divmod(int(total_duration), 60)
    hrs, mins = divmod(mins, 60)
    human_time = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
    print(f"Total duration: {total_duration:.1f}s ({human_time})")

    if timings:
        for t in timings:
            print(
                "  Step '{}': {:.1f}s, in={} out={} cost=${:.4f}".format(
                    t["name"],
                    t["duration"],
                    t["input_tokens"],
                    t["output_tokens"],
                    t["cost_usd"],
                )
            )
