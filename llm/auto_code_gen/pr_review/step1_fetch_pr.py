"""Fetch a GitHub PR to a local branch.

Checks out the PR branch locally using gh CLI (preferred) or falls back
to raw git fetch if gh is not authenticated.

Usage:

    python -m llm.auto_code_gen.pr_review.step1_fetch_pr \
        --pr-url https://github.com/vllm-project/vllm/pull/44891 \
        --repo-dir /path/to/vllm

    python -m llm.auto_code_gen.pr_review.step1_fetch_pr \
        --pr-url https://github.com/vllm-project/vllm/pull/44891 \
        --repo-dir /path/to/vllm \
        --branch-name my-review-branch
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import os
import subprocess
import sys

from llm.auto_code_gen.pr_review import check_gh_auth, parse_pr_url, run_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a GitHub PR to a local branch.")
    parser.add_argument(
        "--pr-url",
        required=True,
        help="Full GitHub PR URL (e.g. https://github.com/owner/repo/pull/123).",
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Path to the local git repository.",
    )
    parser.add_argument(
        "--branch-name",
        default=None,
        help="Local branch name (default: pr-<number>).",
    )
    return parser.parse_args()


def fetch_pr(pr_url: str, repo_dir: str, branch_name: str | None = None):
    owner, repo, number = parse_pr_url(pr_url)
    if branch_name is None:
        branch_name = f"pr-{number}"

    repo_dir = os.path.abspath(repo_dir)

    if not os.path.isdir(repo_dir):
        print(f"ERROR: Repository directory not found: {repo_dir}")
        sys.exit(1)

    result = run_git(["rev-parse", "--is-inside-work-tree"], repo_dir, check=False)
    if result.returncode != 0:
        print(f"ERROR: {repo_dir} is not a git repository.")
        sys.exit(1)

    print("=" * 70)
    print("FETCH PR")
    print("=" * 70)
    print(f"  PR: {owner}/{repo} #{number}")
    print(f"  Repo: {repo_dir}")
    print(f"  Branch: {branch_name}")
    print("=" * 70)

    existing = run_git(["branch", "--list", branch_name], repo_dir, check=False)
    if branch_name in existing.stdout:
        print(f"\nBranch '{branch_name}' already exists locally.")
        current = run_git(["branch", "--show-current"], repo_dir).stdout.strip()
        if current != branch_name:
            print(f"Switching to branch '{branch_name}'...")
            run_git(["checkout", branch_name], repo_dir)
        head = run_git(["rev-parse", "--short", "HEAD"], repo_dir).stdout.strip()
        print(f"HEAD: {head}")
        return

    if check_gh_auth():
        print(f"\nUsing gh to checkout PR #{number}...")
        result = subprocess.run(
            ["gh", "pr", "checkout", str(number), "--branch", branch_name],
            cwd=repo_dir,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            head = run_git(["rev-parse", "--short", "HEAD"], repo_dir).stdout.strip()
            print(f"Checked out PR #{number} to branch '{branch_name}' (HEAD: {head})")
            return
        else:
            print(f"gh pr checkout failed: {result.stderr.strip()}")
            print("Falling back to git fetch...")

    print(f"\nUsing git fetch to get PR #{number}...")
    fetch_result = run_git(
        ["fetch", "origin", f"pull/{number}/head:{branch_name}"],
        repo_dir,
        check=False,
    )
    if fetch_result.returncode != 0:
        print(f"ERROR: git fetch failed: {fetch_result.stderr.strip()}")
        sys.exit(1)

    run_git(["checkout", branch_name], repo_dir)
    head = run_git(["rev-parse", "--short", "HEAD"], repo_dir).stdout.strip()
    print(f"Checked out PR #{number} to branch '{branch_name}' (HEAD: {head})")


if __name__ == "__main__":
    args = parse_args()
    fetch_pr(args.pr_url, args.repo_dir, args.branch_name)
