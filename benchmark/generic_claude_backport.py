"""
Generic single-shot Claude backporter — the "raw Claude" baseline evaluator.

Given the original fix commit (on the source branch) and a target worktree
already checked out at BACKPORT_SHA~1, asks Claude in a single call to
produce the backported diff.  No cherry-pick, no conflict resolution loop,
no build verification.

Usage:
  uv run python -m benchmark.generic_claude_backport \\
      --repo          /path/to/target-worktree   ← checked out at BACKPORT_SHA~1
      --source-repo   /path/to/source-repo       ← where orig SHA lives (can be same)
      --orig          <ORIG_SHA>                  ← fix commit on source branch
      --target-branch <BRANCH_NAME>              ← e.g. "6.1" or "release-1.33"
      --model         claude-opus-4-6            ← any Claude model
      --out           /path/to/output.diff
"""

import argparse
import os
import subprocess
import sys
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_git(cmd: list[str], cwd: str) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("git command failed: %s\n%s", " ".join(cmd), result.stderr[:500])
        return ""
    return result.stdout


def _make_completion(model: str, messages: list[dict], max_tokens: int = 8192):
    """Model-agnostic completion via LiteLLM."""
    from common.llm_client import completion as llm_completion
    return llm_completion(model=model, messages=messages, max_tokens=max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-shot Claude backporter (baseline)")
    parser.add_argument("--repo",          required=True,  help="Path to the git repository")
    parser.add_argument("--orig",          required=True,  help="Original fix commit SHA (on source branch)")
    parser.add_argument("--target-sha",    required=True,  help="SHA of BACKPORT_SHA~1 (pre-fix state on target)")
    parser.add_argument("--target-branch", required=True,  help="Target branch name (e.g. 5.9, release-1.33)")
    parser.add_argument("--model",         default="claude-opus-4-6", help="Claude model to use")
    parser.add_argument("--out",           required=True,  help="Path to write output diff")
    args = parser.parse_args()

    # 1. Get the original fix diff
    log.info("Fetching original fix diff for %s", args.orig[:12])
    orig_diff = run_git(["git", "show", "--unified=10", args.orig], cwd=args.repo)
    if not orig_diff:
        log.error("Could not get diff for %s", args.orig)
        sys.exit(1)

    # 2. Find which files the fix changed
    changed_files = run_git(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", args.orig],
        cwd=args.repo,
    ).strip().splitlines()

    # 3. Read those files at the pre-fix SHA (BACKPORT_SHA~1) using git show
    #    — no worktree needed, just point git show at the right commit
    target_context = ""
    for f in changed_files[:5]:
        content = run_git(["git", "show", f"{args.target_sha}:{f}"], cwd=args.repo)
        if content:
            target_context += f"\n\n=== {f} (state on {args.target_branch} before fix) ===\n{content[:6000]}"

    if not target_context:
        log.warning("Could not read any changed files at %s", args.target_sha[:8])

    # 4. Ask Claude for the backport
    prompt = f"""You are an expert software engineer performing a git backport.

TASK: Apply the following fix to the target branch `{args.target_branch}` and produce a unified diff.

## Original Fix Diff (from source branch)
```diff
{orig_diff[:10000]}
```

## Current State of Changed Files on `{args.target_branch}`
{target_context}

## Instructions
1. Understand what the original fix does (the intent, not just the syntax).
2. Adapt the fix to the current code on `{args.target_branch}` — context may differ between branches.
3. Preserve the original intent: fix the same root cause, keep the same behaviour.
4. Output ONLY a valid unified diff in `git diff` format.
5. Do not include any explanation or commentary — just the raw diff starting with `diff --git`.

Begin your response with `diff --git` and end when the diff is complete."""

    log.info("Calling LLM (%s) for single-shot backport…", args.model)
    result = _make_completion(args.model, messages=[{"role": "user", "content": prompt}])

    raw = result.text.strip()

    # Strip markdown fences if Claude wrapped the diff
    if "```" in raw:
        lines = raw.splitlines()
        in_block, diff_lines = False, []
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                diff_lines.append(line)
        raw = "\n".join(diff_lines)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(raw)

    log.info("Done — %d lines written to %s", raw.count("\n"), args.out)
    log.info("Token usage — input: %d, output: %d", result.input_tokens, result.output_tokens)


if __name__ == "__main__":
    main()
