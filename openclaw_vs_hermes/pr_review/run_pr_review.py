#!/usr/bin/env python3
"""Automated PR review: fix comments, rebuild, retest, commit, summarize.

Usage:
    python -m openclaw_vs_hermes.pr_review.run_pr_review --config agent/pr_review/configs/pr_review_config.json
    python -m openclaw_vs_hermes.pr_review.run_pr_review --config agent/pr_review/configs/pr_review_config.json --round 2
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import subprocess
import sys
import time
import argparse
import asyncio

from common.utils import setup_logging
from common.claude_utils import claude_run, PipelineStep

from openclaw_vs_hermes.pr_review.pr_review_config import PRReviewConfig
from openclaw_vs_hermes.pr_review.pr_review_prompts import (
    create_pr_review_context_str,
    _build_artifact_context_section,
    _detect_max_code_gen_version,
    FixPRCommentsPrompt,
    ReviewPRFixPrompt,
    SummarizePRFixesPrompt,
    pr_comments_filename,
    pr_fix_patch_filename,
    pr_fix_summary_filename,
    pr_fixes_report_filename,
    code_gen_pr_patch_filename,
    code_gen_pr_summary_filename,
    code_gen_pr_review_patch_filename,
    code_gen_pr_review_summary_filename,
    detect_current_round_id,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Address PR review comments for a ported feature"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the PR review JSON config file.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Explicit round ID. If omitted, auto-detects from highest PR_comments_N.txt.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1: Verify branch
# ---------------------------------------------------------------------------

def _verify_branch(config):
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=config.openclaw_source_dir,
        capture_output=True, text=True, check=False,
    )
    current = result.stdout.strip()
    if current != config.branch_name:
        print("ERROR: OpenClaw repo is on branch '{}', expected '{}'.".format(
            current, config.branch_name,
        ))
        print("  Run: cd {} && git checkout {}".format(
            config.openclaw_source_dir, config.branch_name,
        ))
        sys.exit(1)

    head = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=config.openclaw_source_dir,
        capture_output=True, text=True, check=False,
    )
    print("Branch: {}".format(current))
    print("HEAD:   {}".format(head.stdout.strip()))


# ---------------------------------------------------------------------------
# Phase 2: Build
# ---------------------------------------------------------------------------

def _run_build(config):
    print("Running: {}".format(config.build_command))
    result = subprocess.run(
        config.build_command, shell=True,
        cwd=config.openclaw_source_dir, check=False,
    )
    if result.returncode != 0:
        print("ERROR: Build failed (exit code {}).".format(result.returncode))
        print("Fix build errors before running PR review.")
        sys.exit(1)
    print("Build: PASS")


# ---------------------------------------------------------------------------
# Phase 2b: Lint
# ---------------------------------------------------------------------------

def _run_lint(config):
    if not config.lint_command:
        print("Lint: SKIPPED (no lint_command configured)")
        return
    print("Running: {}".format(config.lint_command))
    result = subprocess.run(
        config.lint_command, shell=True,
        cwd=config.openclaw_source_dir, check=False,
    )
    if result.returncode != 0:
        print("ERROR: Lint failed (exit code {}).".format(result.returncode))
        print("Fix lint errors before running PR review.")
        sys.exit(1)
    print("Lint: PASS")


# ---------------------------------------------------------------------------
# Phase 3: Test
# ---------------------------------------------------------------------------

def _run_tests(config):
    print("Running: {}".format(config.test_command))
    result = subprocess.run(
        config.test_command, shell=True,
        cwd=config.openclaw_source_dir, check=False,
    )
    if result.returncode != 0:
        print("ERROR: Tests failed (exit code {}).".format(result.returncode))
        print("Fix test failures before running PR review.")
        sys.exit(1)
    print("Tests: PASS")


# ---------------------------------------------------------------------------
# Phase 4: Fix PR comments (Claude loop)
# ---------------------------------------------------------------------------

async def _run_fix_iterations(config, claude_config, context, artifact_context,
                               round_id):
    all_step_timings = []
    prev_patch_file = None
    prev_summary_file = None
    final_patch_file = None
    final_summary_file = None

    comments_file = pr_comments_filename(round_id)
    base_version = _detect_max_code_gen_version(config.output_dir)

    for i in range(config.max_fix_iterations):
        iteration = i + 1
        code_gen_version = base_version + iteration

        # --- Fix step: generate code_gen_V{N}_pr_r_{R}.patch ---
        gen_patch = code_gen_pr_patch_filename(code_gen_version, round_id)
        gen_summary = code_gen_pr_summary_filename(code_gen_version, round_id)

        fix_prompt = FixPRCommentsPrompt(
            context=context,
            artifact_context=artifact_context,
            pr_comments_file=comments_file,
            branch_name=config.branch_name,
            round_id=round_id,
            build_command=config.build_command,
            lint_command=config.lint_command,
            test_command=config.test_command,
            output_patch_file=gen_patch,
            output_summary_file=gen_summary,
            prev_output_patch_file=prev_patch_file,
            prev_output_summary_file=prev_summary_file,
            iteration=iteration,
        )

        fix_steps = [
            PipelineStep(
                name="fix_pr_comments_R{}_V{}".format(round_id, iteration),
                prompt=fix_prompt.prompt(),
                output_files=[gen_patch, gen_summary],
            ),
        ]

        print("\n--- Fix Iteration {} (Round {}, code_gen V{}) ---".format(
            iteration, round_id, code_gen_version,
        ))
        timings = await claude_run(claude_config, fix_steps)
        all_step_timings.extend(timings)

        gen_patch_path = os.path.join(config.output_dir, gen_patch)
        if not os.path.isfile(gen_patch_path):
            print("Iteration {}: No patch generated. Stopping.".format(iteration))
            break

        # --- Review step: generate code_gen_review_V{N}_pr_r_{R}.patch ---
        review_patch = code_gen_pr_review_patch_filename(code_gen_version, round_id)
        review_summary = code_gen_pr_review_summary_filename(code_gen_version, round_id)

        review_prompt = ReviewPRFixPrompt(
            context=context,
            artifact_context=artifact_context,
            pr_comments_file=comments_file,
            input_patch_file=gen_patch,
            input_summary_file=gen_summary,
            branch_name=config.branch_name,
            round_id=round_id,
            build_command=config.build_command,
            lint_command=config.lint_command,
            test_command=config.test_command,
            output_patch_file=review_patch,
            output_summary_file=review_summary,
        )

        review_steps = [
            PipelineStep(
                name="review_pr_fix_R{}_V{}".format(round_id, iteration),
                prompt=review_prompt.prompt(),
                output_files=[review_patch, review_summary],
            ),
        ]

        print("--- Review of Fix V{} (Round {}) ---".format(code_gen_version, round_id))
        timings = await claude_run(claude_config, review_steps)
        all_step_timings.extend(timings)

        review_patch_path = os.path.join(config.output_dir, review_patch)
        review_summary_path = os.path.join(config.output_dir, review_summary)

        if os.path.isfile(review_patch_path):
            final_patch_file = review_patch
            final_summary_file = review_summary
            prev_patch_file = review_patch
            prev_summary_file = review_summary
        else:
            final_patch_file = gen_patch
            final_summary_file = gen_summary
            prev_patch_file = gen_patch
            prev_summary_file = gen_summary

        # Check convergence from review summary
        if os.path.isfile(review_summary_path):
            with open(review_summary_path) as f:
                first_line = f.readline().strip()
            if "CONVERGED" in first_line:
                print("Review CONVERGED at iteration {} — no further fixes needed.".format(
                    iteration,
                ))
                break

    return all_step_timings, final_patch_file, final_summary_file


# ---------------------------------------------------------------------------
# Phase 5: Commit
# ---------------------------------------------------------------------------

def _extract_findings_from_file(path):
    """Extract [P1]/[P2] findings from a summary file, handling multiple formats."""
    import re
    findings = []
    seen = set()
    patterns = [
        re.compile(r'^\[P\d\]\s+(.+?)\s+—\s+(.+)$'),
        re.compile(r'^###\s+Comment\s+\d+:\s+\[P\d\]\s+(.+)$'),
        re.compile(r'^COMMENT\s+\d+:\s+\[P\d\]\s+(.+)$'),
        re.compile(r'^-\s+\[P\d\]\s+(.+?)\s+—\s+(.+)$'),
        re.compile(r'^-\s+\[P\d\]\s+(.+)$'),
    ]
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            for pat in patterns:
                m = pat.match(stripped)
                if m:
                    desc = m.group(1).strip()
                    if len(m.groups()) > 1:
                        location = m.group(2).strip()
                        desc = "{} ({})".format(desc, location)
                    key = desc.lower()
                    if key not in seen:
                        seen.add(key)
                        findings.append("- {}".format(desc))
                    break
    return findings


def _build_commit_message(config, round_id, final_summary_file):
    """Build a commit message by extracting what was fixed from the summary."""
    bullet_lines = []

    if final_summary_file:
        summary_path = os.path.join(config.output_dir, final_summary_file)
        if os.path.isfile(summary_path):
            bullet_lines = _extract_findings_from_file(summary_path)

    if not bullet_lines:
        gen_summary = final_summary_file.replace("_review_", "_") if final_summary_file else None
        if gen_summary:
            gen_path = os.path.join(config.output_dir, gen_summary)
            if os.path.isfile(gen_path):
                bullet_lines = _extract_findings_from_file(gen_path)

    if not bullet_lines:
        bullet_lines.append("- Address reviewer feedback")

    title = "fix: address PR review feedback"
    body = "\n".join(bullet_lines)
    return "{}\n\n{}".format(title, body)


def _commit_fixes(config, round_id, final_summary_file):
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=config.openclaw_source_dir,
        capture_output=True, text=True, check=False,
    )
    changes = [l for l in status.stdout.strip().splitlines() if l and not l.startswith("??")]
    if not changes:
        print("No staged/modified files to commit. Skipping commit.")
        return False

    subprocess.run(
        ["git", "add", "-A"],
        cwd=config.openclaw_source_dir, check=True,
    )

    msg = _build_commit_message(config, round_id, final_summary_file)
    result = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=config.openclaw_source_dir, check=False,
    )
    if result.returncode != 0:
        print("WARNING: git commit failed (exit code {}).".format(result.returncode))
        return False

    head = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=config.openclaw_source_dir,
        capture_output=True, text=True, check=False,
    )
    print("Committed: {}".format(head.stdout.strip()))
    return True


# ---------------------------------------------------------------------------
# Phase 6: Summarize
# ---------------------------------------------------------------------------

async def _run_summarize(config, claude_config, context, round_id,
                          final_patch_file, final_summary_file):
    report_file = pr_fixes_report_filename(round_id)

    summarize_prompt = SummarizePRFixesPrompt(
        context=context,
        pr_comments_file=pr_comments_filename(round_id),
        fix_patch_file=final_patch_file,
        fix_summary_file=final_summary_file,
        round_id=round_id,
        output_file=report_file,
    )

    steps = [
        PipelineStep(
            name="summarize_pr_fixes_R{}".format(round_id),
            prompt=summarize_prompt.prompt(),
            output_files=[report_file],
        ),
    ]

    timings = await claude_run(claude_config, steps)
    return timings


# ---------------------------------------------------------------------------
# Phase 9: Upstream update instructions
# ---------------------------------------------------------------------------

def _print_upstream_instructions(config, round_id):
    report_file = pr_fixes_report_filename(round_id)
    report_path = os.path.join(config.output_dir, report_file)

    print("All PR review fixes for round {} have been committed locally.".format(round_id))
    print()
    print("To update the upstream PR, run the following commands:")
    print()
    print("  1. Push the updated branch to your fork:")
    print("     cd {} && git push myfork {}".format(
        config.openclaw_source_dir, config.branch_name,
    ))
    print()
    print("  2. (Optional) Force-push if you amended or rebased:")
    print("     cd {} && git push --force-with-lease myfork {}".format(
        config.openclaw_source_dir, config.branch_name,
    ))
    print()
    print("  3. Post a comment on the PR summarizing the fixes.")
    print("     The summary is available at:")
    print("     {}".format(report_path))
    print()
    print("     You can copy-paste it directly, or post via gh CLI:")
    print("     gh pr comment <PR_NUMBER> --repo openclaw/openclaw \\")
    print("       --body-file {}".format(report_path))
    print()
    print("When new reviewer comments arrive, run the next round:")
    print("  1. ./openclaw_vs_hermes/pr_review/fetch_pr_comments.sh <PR_URL> {} {}".format(
        config.output_dir, round_id + 1,
    ))
    print("  2. python -m openclaw_vs_hermes.pr_review.run_pr_review --config <config> --round {}".format(
        round_id + 1,
    ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return "{:.1f}s ({}h {}m {}s)".format(seconds, hrs, mins, secs)
    return "{:.1f}s ({}m {}s)".format(seconds, mins, secs)


def _format_tokens(count):
    if count >= 1_000_000:
        return "{:.1f}M".format(count / 1_000_000)
    if count >= 1_000:
        return "{:.1f}K".format(count / 1_000)
    return str(count)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pr_review(config, round_id):
    claude_config = config.make_claude_config()
    context = create_pr_review_context_str(config)
    artifact_context = _build_artifact_context_section(config.output_dir, round_id)

    all_step_timings = []

    # Pre-check: PR comments file must exist
    comments_file = pr_comments_filename(round_id)
    comments_path = os.path.join(config.output_dir, comments_file)
    if not os.path.isfile(comments_path):
        print("ERROR: {} not found in {}.".format(comments_file, config.output_dir))
        print("Run fetch_pr_comments.sh first:")
        print("  ./openclaw_vs_hermes/pr_review/fetch_pr_comments.sh <PR_URL> {} {}".format(
            config.output_dir, round_id,
        ))
        sys.exit(1)
    with open(comments_path) as f:
        content = f.read()
    if len(content.strip()) < 50:
        print("ERROR: {} appears empty or malformed ({} bytes).".format(
            comments_file, len(content),
        ))
        print("Re-run fetch_pr_comments.sh to fetch comments.")
        sys.exit(1)
    print("PR comments file: {} ({} bytes)".format(comments_file, len(content)))

    # Phase 1: Verify
    print("\n" + "=" * 80)
    print("PHASE 1: VERIFY BRANCH")
    print("=" * 80)
    _verify_branch(config)

    # Phase 2: Build + Lint
    print("\n" + "=" * 80)
    print("PHASE 2: BUILD")
    print("=" * 80)
    _run_build(config)
    _run_lint(config)

    # Phase 3: Test
    print("\n" + "=" * 80)
    print("PHASE 3: TEST (verify baseline)")
    print("=" * 80)
    _run_tests(config)

    # Phase 4: Fix PR Comments
    print("\n" + "=" * 80)
    print("PHASE 4: FIX PR COMMENTS (Round {})".format(round_id))
    print("=" * 80)
    fix_timings, final_patch, final_summary = await _run_fix_iterations(
        config, claude_config, context, artifact_context, round_id,
    )
    all_step_timings.extend(fix_timings)

    if final_patch is None:
        print("ERROR: No fixes were generated. Exiting.")
        sys.exit(1)

    # Phase 5: Rebuild + Lint after fixes
    print("\n" + "=" * 80)
    print("PHASE 5: REBUILD (verify fixes compile)")
    print("=" * 80)
    _run_build(config)
    _run_lint(config)

    # Phase 6: Retest after fixes
    print("\n" + "=" * 80)
    print("PHASE 6: RETEST (verify fixes pass)")
    print("=" * 80)
    _run_tests(config)

    # Phase 7: Commit
    print("\n" + "=" * 80)
    print("PHASE 7: COMMIT")
    print("=" * 80)
    _commit_fixes(config, round_id, final_summary)

    # Phase 8: Summarize
    print("\n" + "=" * 80)
    print("PHASE 8: SUMMARIZE")
    print("=" * 80)
    summarize_timings = await _run_summarize(
        config, claude_config, context, round_id, final_patch, final_summary,
    )
    all_step_timings.extend(summarize_timings)

    # Phase 9: Print upstream update instructions
    print("\n" + "=" * 80)
    print("PHASE 9: UPSTREAM UPDATE INSTRUCTIONS")
    print("=" * 80)
    _print_upstream_instructions(config, round_id)

    return all_step_timings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    config = PRReviewConfig.from_json(args.config)

    setup_logging("pr_review")

    if args.round is not None:
        round_id = args.round
    else:
        round_id = detect_current_round_id(config.output_dir)
        if round_id == 0:
            print("ERROR: No PR_comments_N.txt found in {}.".format(config.output_dir))
            print("Run fetch_pr_comments.sh first to create PR_comments_1.txt.")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("PR REVIEW AUTOMATION — Round {}".format(round_id))
    print("=" * 80)
    print("Output dir:      {}".format(config.output_dir))
    print("OpenClaw source: {}".format(config.openclaw_source_dir))
    print("Branch:          {}".format(config.branch_name))
    print("Comments file:   {}".format(pr_comments_filename(round_id)))
    print("Max iterations:  {}".format(config.max_fix_iterations))
    print("Thinking mode:   {}".format(config.thinking_mode))
    print("=" * 80 + "\n")

    start_time = time.time()
    all_step_timings = asyncio.run(run_pr_review(config, round_id))
    total_duration = time.time() - start_time

    total_input = sum(s["input_tokens"] for s in all_step_timings)
    total_output = sum(s["output_tokens"] for s in all_step_timings)
    total_cost = sum(s["cost_usd"] for s in all_step_timings)

    print("\n" + "=" * 80)
    print("PR REVIEW COMPLETE — Round {}".format(round_id))
    print("=" * 80)
    print("Duration:      {}".format(_format_duration(total_duration)))
    print("Input tokens:  {}".format(_format_tokens(total_input)))
    print("Output tokens: {}".format(_format_tokens(total_output)))
    print("Cost:          ${:.4f}".format(total_cost))
    print()

    report_path = os.path.join(config.output_dir, pr_fixes_report_filename(round_id))
    if os.path.isfile(report_path):
        print("Results saved to: {}".format(report_path))
    else:
        print("WARNING: Summary report was not generated.")
    print("=" * 80)
