"""Push fix commits to the remote PR branch.

Pushes any new commits on the current branch to the remote. Uses SSH
for authentication — the password/passphrase prompt is passed through
to the terminal for manual entry.

Usage:

    python -m llm.auto_code_gen.pr_review.step4_push_fixes \
        --repo-dir /path/to/vllm

    python -m llm.auto_code_gen.pr_review.step4_push_fixes \
        --repo-dir /path/to/vllm \
        --remote origin \
        --branch pr-44891
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import sys
import argparse
import subprocess

from llm.auto_code_gen.pr_review import run_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push fix commits to the remote PR branch."
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Path to the local git repository.",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote name (default: origin).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch name to push (default: current branch).",
    )
    return parser.parse_args()


def push_fixes(repo_dir: str, remote: str = "origin", branch: str = None):
    repo_dir = os.path.abspath(repo_dir)

    if not os.path.isdir(repo_dir):
        print("ERROR: Repository directory not found: {}".format(repo_dir))
        sys.exit(1)

    result = run_git(["rev-parse", "--is-inside-work-tree"], repo_dir, check=False)
    if result.returncode != 0:
        print("ERROR: {} is not a git repository.".format(repo_dir))
        sys.exit(1)

    if branch is None:
        branch = run_git(["branch", "--show-current"], repo_dir).stdout.strip()
        if not branch:
            print("ERROR: Not on any branch (detached HEAD).")
            sys.exit(1)

    print("=" * 70)
    print("PUSH FIXES")
    print("=" * 70)
    print("  Repo: {}".format(repo_dir))
    print("  Remote: {}".format(remote))
    print("  Branch: {}".format(branch))
    print("=" * 70)

    tracking = run_git(
        ["rev-parse", "--abbrev-ref", "{}@{{upstream}}".format(branch)],
        repo_dir,
        check=False,
    )
    has_upstream = tracking.returncode == 0

    if has_upstream:
        upstream = tracking.stdout.strip()
        log_result = run_git(
            ["log", "--oneline", "{}..{}".format(upstream, branch)],
            repo_dir,
            check=False,
        )
        commits = log_result.stdout.strip()
    else:
        log_result = run_git(
            ["log", "--oneline", "-20", branch],
            repo_dir,
            check=False,
        )
        commits = log_result.stdout.strip()
        print("\nWARNING: No upstream tracking branch set.")
        print("Showing last 20 commits on '{}':".format(branch))

    if commits:
        print("\nCommits to push:")
        for line in commits.split("\n"):
            print("  {}".format(line))
    else:
        print("\nNo new commits to push.")
        return

    commit_count = len(commits.strip().split("\n"))
    print("\nPushing {} commit(s) to {}/{}...".format(commit_count, remote, branch))
    print("(SSH passphrase may be requested)\n")

    push_result = subprocess.run(
        ["git", "push", remote, branch],
        cwd=repo_dir,
    )

    if push_result.returncode == 0:
        print("\nPush successful.")
    else:
        print("\nERROR: Push failed (exit code {}).".format(push_result.returncode))
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    push_fixes(args.repo_dir, args.remote, args.branch)
