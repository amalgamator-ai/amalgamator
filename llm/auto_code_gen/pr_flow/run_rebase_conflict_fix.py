"""Resolve rebase conflicts in a paused worktree -- the AI half of a rebase.

The human script starts the rebase and leaves the repository paused. This module
lets the agent edit the conflicted files until nothing is left in conflict; it
never stages, commits, continues or aborts, because those git writes belong to
the human script.

    1. snapshot the conflict state (rebase/CONFLICTS.md) for the agent
    2. agent resolves the markers and writes rebase_resolution.md
    3. verify: every file resolved, no markers left, nothing still unmerged
    4. hand back to pr_flow_step_r1_rebase.sh --continue

Exit codes:
    0   all conflicts resolved -- the human script may continue the rebase
    30  no rebase paused, or conflicts still outstanding
    50  attempts exhausted -- human decision required (see escalation file)
    51  precondition problem

Usage:
    python -m llm.auto_code_gen.pr_flow.run_rebase_conflict_fix --config <code_gen.json> \
        [--stage 1] [--max-attempts 3] [--dry-run]
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import asyncio
import json
import os
import sys
import time

from common.utils import setup_logging
from common.claude_utils import PipelineStep
from llm.auto_code_gen.agents import agent_run, agent_session
from llm.auto_code_gen.code_gen_configs import load_config_and_use_case
from llm.auto_code_gen.code_gen_prompts import PR_COMMANDS_DIR, PR_PR_DESC_FILE

from llm.auto_code_gen.run_code_gen import _read_feature_disable_env_var

from llm.auto_code_gen.pr_flow import checks, pr_state, rebase
from llm.auto_code_gen.pr_flow.prompts import (
    REBASE_RESOLUTION_FILE,
    REBASE_SUMMARY_DRAFT_FILE,
    gen_RebaseConflictFixPrompt,
)
from llm.auto_code_gen.pr_flow.run_pr_review_iters import _repo_dir, _write

EXIT_SUCCESS = 0
EXIT_REBASE = 30
EXIT_EXHAUSTED = 50
EXIT_PRECONDITION = 51


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve rebase conflicts (no git writes).")
    parser.add_argument("--config", required=True, help="code-gen JSON config")
    parser.add_argument("--stage", type=int, default=None,
                        help="rebase stage number (default: next free R<n>)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--allow-git-network", action="store_true",
                        help="do not block git network transports for this process")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan, resolve nothing")
    from llm.auto_code_gen.agents import add_agent_argument
    add_agent_argument(parser)
    return parser.parse_args()


def existing_stages(output_dir):
    root = rebase.rebase_dir(output_dir)
    if not os.path.isdir(root):
        return []
    # A stage counts only once it holds a snapshot, so an empty directory left
    # behind by an interrupted run cannot push the work forward.
    return sorted(int(n[1:]) for n in os.listdir(root)
                  if n.startswith("R") and n[1:].isdigit()
                  and os.path.isfile(os.path.join(root, n, rebase.CONFLICTS_FILE)))


def default_stage(output_dir):
    """The stage the human script is currently on, or a new one.

    pr_flow_step_r1_rebase.sh snapshots conflicts into R<n> when it pauses, so
    the newest R<n> without a resolution report is the work in progress -- it
    must be filled in, not skipped over.
    """
    for stage in reversed(existing_stages(output_dir)):
        if not os.path.isfile(os.path.join(
                rebase.rebase_dir(output_dir, stage), REBASE_RESOLUTION_FILE)):
            return stage
    return (existing_stages(output_dir)[-1] + 1) if existing_stages(output_dir) else 1


async def run_stage(config, claude_config, use_case, state, args):
    output_dir = config.output_dir
    repo_dir = _repo_dir(config, state)
    stage = args.stage or default_stage(output_dir)
    stage_dir = rebase.rebase_dir(output_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    disable_var = _read_feature_disable_env_var(output_dir) or "UNKNOWN"
    context = use_case.create_context_str(claude_config, config)
    pr_desc = os.path.join(output_dir, PR_COMMANDS_DIR, PR_PR_DESC_FILE)

    if args.dry_run:
        # Nothing is written in a dry run -- not even the conflict brief.
        conflicts = rebase.conflicted_files(repo_dir)
        print("  rebase state: {}".format(rebase.current_step(repo_dir)))
        print("  conflicts   : {}".format(", ".join(conflicts) or "none"))
        print("[dry-run] would snapshot the conflicts into {} and ask the agent to".format(
            rebase.rebase_dir(output_dir, stage)))
        print("[dry-run] resolve them and write {}".format(REBASE_RESOLUTION_FILE))
        print("[dry-run] nothing executed.")
        return EXIT_SUCCESS

    snapshot = rebase.record(output_dir, repo_dir, state, stage)
    conflicts_path = os.path.join(stage_dir, rebase.CONFLICTS_FILE)
    resolution_path = os.path.join(stage_dir, REBASE_RESOLUTION_FILE)

    print("=" * 78)
    print("REBASE CONFLICT FIX -- stage R{}".format(stage))
    print("=" * 78)
    print("  repo        : {}".format(repo_dir))
    print("  rebase state: {}".format(rebase.current_step(repo_dir)))
    print("  conflicts   : {}".format(", ".join(snapshot["conflict_files"]) or "none"))
    print("  brief       : {}".format(conflicts_path))
    print("=" * 78 + "\n")

    if not snapshot["conflict_files"]:
        print("Nothing is in conflict -- the rebase is clean.")
        print("NEXT: bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_r1_rebase.sh "
              "--config {} --continue".format(args.config))
        return EXIT_SUCCESS

    notes = []
    async with agent_session(claude_config) as client:
        for attempt in range(1, args.max_attempts + 1):
            print("\n--- conflict resolution attempt {}/{} (stage R{}) ---".format(
                attempt, args.max_attempts, stage))
            previous = os.path.join(
                rebase.rebase_dir(output_dir, stage - 1), REBASE_RESOLUTION_FILE) \
                if stage > 1 and attempt == 1 else ""
            prompt = gen_RebaseConflictFixPrompt(
                context=context, repo_dir=repo_dir, output_dir=output_dir,
                stage_dir=stage_dir, attempt=attempt, conflicts_file=conflicts_path,
                prev_attempt_file=previous or "", pr_description_file=pr_desc,
                feature_toggle=disable_var,
            )
            start = time.time()
            await agent_run(claude_config, [
                PipelineStep(
                    name="resolve_rebase_conflicts_R{}_{}".format(stage, attempt),
                    prompt=prompt.prompt(),
                    output_files=[os.path.join(
                        os.path.relpath(stage_dir, output_dir), REBASE_RESOLUTION_FILE)]),
            ], client=client)
            print("  resolution attempt took {:.0f}s".format(time.time() - start))

            try:
                outcome = checks.check_rebase_resolved(state, resolution_path)
            except checks.CheckError as exc:
                notes.append("attempt {}: {}".format(attempt, exc))
                print("  still outstanding: {}".format(exc))
                rebase.record(output_dir, repo_dir, state, stage)
                continue

            summary = rebase.parse_resolution(resolution_path)
            _write(os.path.join(stage_dir, "resolution_summary.json"),
                   json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("\n  {}".format(outcome))
            print("  Resolved by the agent; the rebase is still paused so nothing is committed.")
            print("  NEXT (human, local git):")
            print("    bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_r1_rebase.sh "
                  "--config {} --continue".format(args.config))
            return EXIT_SUCCESS

    escalation = os.path.join(rebase.rebase_dir(output_dir),
                              "escalation_stage_R{}.md".format(stage))
    _write(escalation, _escalation_text(stage, state, notes, repo_dir))
    print("\n  Conflicts still outstanding after {} attempt(s).".format(args.max_attempts))
    print("  The rebase is still paused -- nothing was lost:")
    print("    inspect : {}".format(rebase.current_step(repo_dir)))
    print("    give up : bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_r1_rebase.sh "
          "--config {} --abort".format(args.config))
    print("  See {}".format(escalation))
    return EXIT_EXHAUSTED


def _escalation_text(stage, state, notes, repo_dir):
    lines = ["# Rebase stage R{} -- human decision required".format(stage), ""]
    lines.append("PR: {} (#{})".format(state.get("pr_url") or "not published yet",
                                       state.get("pr_number") or "-"))
    lines.append("Repo: {}".format(repo_dir))
    lines.append("")
    lines.append("## Attempts")
    for note in notes:
        lines.append("- {}".format(note))
    lines.append("")
    lines.append("The repository is still paused mid-rebase. Resolve the remaining conflicts by")
    lines.append("hand and run the continue step, or abort to return to the pre-rebase head:")
    lines.append("  git -C {} rebase --abort".format(repo_dir))
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    config, use_case = load_config_and_use_case(args.config)
    claude_config = config.make_agent_config(args.agent_config)
    claude_config.cwd = config.source_code_dir
    setup_logging("rebase_conflict_fix")

    state_path = pr_state.state_path(config.output_dir)
    if not os.path.isfile(state_path):
        print("ERROR: {} missing -- run scripts/pr_flow_step_0_prepare.sh".format(state_path))
        return EXIT_PRECONDITION
    state = pr_state.load_state(state_path)
    state["_path"] = state_path

    if not args.allow_git_network:
        os.environ["GIT_SSH_COMMAND"] = "/bin/false"
        os.environ["GIT_TERMINAL_PROMPT"] = "0"
        for token in ("GH_TOKEN", "GITHUB_TOKEN"):
            os.environ.pop(token, None)
        print("  Network guard: git transports disabled, GitHub tokens cleared")

    try:
        if not args.dry_run:
            print("  " + checks.check_rebase_paused(state))
        return asyncio.run(run_stage(config, claude_config, use_case, state, args))
    except checks.CheckError as exc:
        print("REFUSED ({}): {}".format(exc.code, exc))
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
