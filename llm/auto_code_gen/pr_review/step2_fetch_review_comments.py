"""Fetch PR review comments from GitHub and generate a summary file.

Fetches all inline review comments, top-level reviews, and issue comments
for a PR, saves the raw JSON, and generates a human-readable summary
for Claude to consume.

Usage:

    python -m llm.auto_code_gen.pr_review.step2_fetch_review_comments \
        --pr-url https://github.com/vllm-project/vllm/pull/44891 \
        --output-dir /path/to/pr_review_output
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import json
import os
from collections import defaultdict

from llm.auto_code_gen.pr_review import parse_pr_url, run_gh

PR_METADATA_FILE = "pr_metadata.json"
REVIEW_COMMENTS_FILE = "review_comments.json"
REVIEWS_FILE = "reviews.json"
ISSUE_COMMENTS_FILE = "issue_comments.json"
REVIEW_COMMENTS_SUMMARY_FILE = "review_comments_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch PR review comments and generate a summary."
    )
    parser.add_argument(
        "--pr-url",
        required=True,
        help="Full GitHub PR URL.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output files.",
    )
    return parser.parse_args()


def _fetch_json(api_path: str, paginate: bool = False) -> object:
    args = ["api"]
    if paginate:
        args.append("--paginate")
    args.append(api_path)
    result = run_gh(args)
    return json.loads(result.stdout)


def _fetch_pr_metadata(owner: str, repo: str, number: int) -> dict:
    return _fetch_json(f"repos/{owner}/{repo}/pulls/{number}")


def _fetch_review_comments(owner: str, repo: str, number: int) -> list:
    return _fetch_json(
        f"repos/{owner}/{repo}/pulls/{number}/comments",
        paginate=True,
    )


def _fetch_reviews(owner: str, repo: str, number: int) -> list:
    return _fetch_json(
        f"repos/{owner}/{repo}/pulls/{number}/reviews",
        paginate=True,
    )


def _fetch_issue_comments(owner: str, repo: str, number: int) -> list:
    return _fetch_json(
        f"repos/{owner}/{repo}/issues/{number}/comments",
        paginate=True,
    )


def _build_comment_threads(comments: list) -> list[list[dict]]:
    by_id = {}
    for c in comments:
        by_id[c["id"]] = c

    roots = []
    replies_map = defaultdict(list)
    for c in comments:
        reply_to = c.get("in_reply_to_id")
        if reply_to and reply_to in by_id:
            replies_map[reply_to].append(c)
        else:
            roots.append(c)

    threads = []
    for root in roots:
        thread = [root]
        _collect_replies(root["id"], replies_map, thread)
        threads.append(thread)
    return threads


def _collect_replies(parent_id: int, replies_map: dict, thread: list):
    for reply in replies_map.get(parent_id, []):
        thread.append(reply)
        _collect_replies(reply["id"], replies_map, thread)


def _generate_summary(
    metadata: dict,
    review_comments: list,
    reviews: list,
    issue_comments: list,
    pr_url: str,
) -> str:
    lines = []
    lines.append("# PR Review Comments Summary")
    lines.append(
        "## PR: {} (#{})".format(metadata.get("title", ""), metadata.get("number", ""))
    )
    lines.append(f"## URL: {pr_url}")
    lines.append(
        "## Author: {}  |  Base: {} <- Head: {}".format(
            metadata.get("user", {}).get("login", ""),
            metadata.get("base", {}).get("ref", ""),
            metadata.get("head", {}).get("ref", ""),
        )
    )
    lines.append("")

    top_level_reviews = [r for r in reviews if r.get("body") and r["body"].strip()]
    if top_level_reviews:
        lines.append("=" * 70)
        lines.append("TOP-LEVEL REVIEWS")
        lines.append("=" * 70)
        for i, r in enumerate(top_level_reviews, 1):
            lines.append("")
            lines.append(
                "--- Top-Level Review {} (by {}, state: {}) ---".format(
                    i,
                    r.get("user", {}).get("login", "unknown"),
                    r.get("state", "UNKNOWN"),
                )
            )
            lines.append(r["body"].strip())
        lines.append("")

    threads = _build_comment_threads(review_comments)

    if threads:
        lines.append("=" * 70)
        lines.append("INLINE REVIEW COMMENTS")
        lines.append("=" * 70)
        for i, thread in enumerate(threads, 1):
            root = thread[0]
            lines.append("")
            lines.append(
                "--- Inline Comment {} (by {}, {}) ---".format(
                    i,
                    root.get("user", {}).get("login", "unknown"),
                    root.get("created_at", "")[:10],
                )
            )
            lines.append("File: {}".format(root.get("path", "")))
            if root.get("line"):
                lines.append("Line: {}".format(root["line"]))
            elif root.get("original_line"):
                lines.append("Original line: {}".format(root["original_line"]))

            diff_hunk = root.get("diff_hunk", "")
            if diff_hunk:
                lines.append("Diff context:")
                for hunk_line in diff_hunk.split("\n"):
                    lines.append(f"  {hunk_line}")

            lines.append("")
            lines.append("Comment: {}".format(root.get("body", "").strip()))

            for reply in thread[1:]:
                lines.append("")
                lines.append(
                    "  [Reply by {} ({}):".format(
                        reply.get("user", {}).get("login", "unknown"),
                        reply.get("created_at", "")[:10],
                    )
                )
                lines.append("   {}]".format(reply.get("body", "").strip()))
        lines.append("")

    if issue_comments:
        non_bot_comments = [
            c
            for c in issue_comments
            if c.get("body")
            and c["body"].strip()
            and not c.get("user", {}).get("login", "").endswith("[bot]")
        ]
        if non_bot_comments:
            lines.append("=" * 70)
            lines.append("GENERAL PR COMMENTS")
            lines.append("=" * 70)
            for i, c in enumerate(non_bot_comments, 1):
                lines.append("")
                lines.append(
                    "--- General Comment {} (by {}, {}) ---".format(
                        i,
                        c.get("user", {}).get("login", "unknown"),
                        c.get("created_at", "")[:10],
                    )
                )
                lines.append(c["body"].strip())
            lines.append("")

    return "\n".join(lines)


def fetch_review_comments(pr_url: str, output_dir: str):
    owner, repo, number = parse_pr_url(pr_url)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("FETCH PR REVIEW COMMENTS")
    print("=" * 70)
    print(f"  PR: {owner}/{repo} #{number}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    print("\nFetching PR metadata...")
    metadata = _fetch_pr_metadata(owner, repo, number)
    metadata_path = os.path.join(output_dir, PR_METADATA_FILE)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    print("Fetching inline review comments...")
    review_comments = _fetch_review_comments(owner, repo, number)
    rc_path = os.path.join(output_dir, REVIEW_COMMENTS_FILE)
    with open(rc_path, "w") as f:
        json.dump(review_comments, f, indent=2)
    print(f"  Saved: {rc_path} ({len(review_comments)} comments)")

    print("Fetching top-level reviews...")
    reviews = _fetch_reviews(owner, repo, number)
    reviews_path = os.path.join(output_dir, REVIEWS_FILE)
    with open(reviews_path, "w") as f:
        json.dump(reviews, f, indent=2)
    print(f"  Saved: {reviews_path} ({len(reviews)} reviews)")

    print("Fetching general PR comments...")
    issue_comments = _fetch_issue_comments(owner, repo, number)
    ic_path = os.path.join(output_dir, ISSUE_COMMENTS_FILE)
    with open(ic_path, "w") as f:
        json.dump(issue_comments, f, indent=2)
    print(f"  Saved: {ic_path} ({len(issue_comments)} comments)")

    print("Generating review comments summary...")
    summary = _generate_summary(
        metadata, review_comments, reviews, issue_comments, pr_url
    )
    summary_path = os.path.join(output_dir, REVIEW_COMMENTS_SUMMARY_FILE)
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")

    total_inline = len(review_comments)
    total_reviews = len([r for r in reviews if r.get("body") and r["body"].strip()])
    threads = _build_comment_threads(review_comments)
    print("\nSummary:")
    print(f"  Inline comment threads: {len(threads)}")
    print(f"  Total inline comments: {total_inline}")
    print(f"  Top-level reviews with body: {total_reviews}")
    print(f"  General comments: {len(issue_comments)}")


if __name__ == "__main__":
    args = parse_args()
    fetch_review_comments(args.pr_url, args.output_dir)
