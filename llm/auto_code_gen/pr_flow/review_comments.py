"""Turn fetched GitHub review artifacts into bounded, actionable input for the AI.

The human fetch script writes raw JSON (gh api output) into a round directory;
this module rebuilds comment threads, drops what the previous round already saw,
and emits two files: a human-readable NEW_COMMENTS.md for the agent to read and
pending_comments.json for tooling. Nothing here talks to GitHub.

Round directory (written by scripts/pr_flow_step_3_fetch_reviews.sh):
    pr_view.json        gh pr view --json (number,url,headRefOid,state,...)
    comments.json       REST /pulls/{n}/comments   (inline review comments)
    reviews.json        REST /pulls/{n}/reviews
    issue_comments.json REST /issues/{n}/comments
    checks.json         check runs + status for the current head
    threads.json        GraphQL reviewThreads (isResolved) -- optional
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

REVIEW_SUMMARY_FILE = "review_summary.txt"
NEW_COMMENTS_FILE = "NEW_COMMENTS.md"
PENDING_COMMENTS_FILE = "pending_comments.json"

RAW_FILES = ("pr_view.json", "comments.json", "reviews.json", "issue_comments.json")


def _load_json(path, default):
    try:
        with open(path) as f:
            text = f.read().strip()
        return json.loads(text) if text else default
    except (OSError, json.JSONDecodeError):
        return default


def flatten_pages(payload):
    """Normalize 'gh api --paginate --slurp' output (a list of pages) to one list.

    Every raw artifact is read through this so the loaders below do not care
    whether the fetch script used --slurp or a plain paginated call.
    """
    if payload is None or isinstance(payload, dict):
        return payload
    if not isinstance(payload, list):
        return payload
    if any(isinstance(item, list) for item in payload):
        flat = []
        for page in payload:
            if isinstance(page, list):
                flat.extend(page)
            elif page is not None:
                flat.append(page)
        return flat
    return payload


def load_round(inbox_dir):
    """Load everything the fetch script wrote for one round."""
    if not os.path.isdir(inbox_dir):
        raise SystemExit("round directory not found: {}".format(inbox_dir))
    return {
        "pr_view": _load_json(os.path.join(inbox_dir, "pr_view.json"), {}),
        "comments": flatten_pages(_load_json(os.path.join(inbox_dir, "comments.json"), [])),
        "reviews": flatten_pages(_load_json(os.path.join(inbox_dir, "reviews.json"), [])),
        "issue_comments": flatten_pages(
            _load_json(os.path.join(inbox_dir, "issue_comments.json"), [])),
        "checks": flatten_pages(_load_json(os.path.join(inbox_dir, "checks.json"), [])),
        "threads": _load_json(os.path.join(inbox_dir, "threads.json"), None),
    }


def build_threads(comments):
    """Group inline comments into threads keyed by their root comment."""
    by_id = {c["id"]: c for c in comments if isinstance(c, dict) and "id" in c}
    replies = defaultdict(list)
    roots = []
    for comment in by_id.values():
        parent = comment.get("in_reply_to_id")
        if parent and parent in by_id:
            replies[parent].append(comment)
        else:
            roots.append(comment)

    threads = []
    for root in sorted(roots, key=lambda c: c["id"]):
        thread = [root]
        pending = list(replies.get(root["id"], []))
        seen = {root["id"]}
        while pending:
            node = pending.pop(0)
            if node["id"] in seen:
                continue
            seen.add(node["id"])
            thread.append(node)
            pending.extend(replies.get(node["id"], []))
        threads.append(thread)
    return threads


def resolved_thread_ids(payload):
    """Comment ids that belong to a resolved review thread (GraphQL, optional)."""
    resolved = set()
    if not payload:
        return resolved
    nodes = payload
    if isinstance(payload, dict):
        nodes = (payload.get("data", {}).get("repository", {})
                 .get("pullRequest", {}).get("reviewThreads", {})
                 .get("nodes", []))
    for thread in nodes or []:
        if not isinstance(thread, dict) or not thread.get("isResolved"):
            continue
        for comment in (thread.get("comments", {}) or {}).get("nodes", []) or []:
            if isinstance(comment, dict) and "databaseId" in comment:
                resolved.add(comment["databaseId"])
    return resolved


def _is_bot(user):
    login = (user or {}).get("login", "") if isinstance(user, dict) else ""
    return login.endswith("[bot]") or login in ("github-actions", "coveralls", "codecov")


def _location(comment):
    path = comment.get("path") or comment.get("filename") or "(no file)"
    line = comment.get("line") or comment.get("original_line")
    return "{}:{}".format(path, line) if line else path


def select_pending(data, cursor=None, our_logins=()):
    """Comments the agent still owes an answer to.

    Excluded: our own comments, bot comments, threads a reviewer resolved, and
    anything at or below the cursor left by the previous fetch.
    """
    cursor = cursor or {}
    max_seen = int(cursor.get("max_comment_id") or 0)
    our_logins = {str(x).lower() for x in our_logins if x}
    resolved = resolved_thread_ids(data.get("threads"))

    pending = []
    for thread in build_threads(data.get("comments") or []):
        root = thread[0]
        ids = [c["id"] for c in thread]
        if any(cid in resolved for cid in ids):
            continue
        if max(ids) <= max_seen:
            continue
        authors = {(c.get("user") or {}).get("login", "").lower() for c in thread}
        others = authors - our_logins - {""}
        if not others or all(_is_bot(c.get("user")) for c in thread):
            continue
        last = thread[-1]
        pending.append({
            "comment_id": root["id"],
            "thread_ids": ids,
            "author": (root.get("user") or {}).get("login", "unknown"),
            "path": root.get("path") or "",
            "line": root.get("line") or root.get("original_line"),
            "location": _location(root),
            "created_at": root.get("created_at", ""),
            "updated_at": last.get("updated_at") or last.get("created_at", ""),
            "body": (root.get("body") or "").strip(),
            "replies": [{"author": (c.get("user") or {}).get("login", ""),
                         "body": (c.get("body") or "").strip()} for c in thread[1:]],
        })
    pending.sort(key=lambda item: item["comment_id"])
    return pending


def _format_checks(checks):
    if not checks:
        return ["CI checks: none recorded"]
    lines = []
    for item in checks if isinstance(checks, list) else [checks]:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("context") or "check"
        status = item.get("status") or item.get("state") or "?"
        conclusion = item.get("conclusion") or item.get("state") or ""
        lines.append("CI check {}: {} {}".format(name, status, conclusion).rstrip())
    return lines or ["CI checks: none recorded"]


def render_new_comments(data, pending, round_no):
    view = data.get("pr_view") or {}
    lines = ["# New review input -- round {}".format(round_no), ""]
    lines.append("PR: {}  (#{})".format(view.get("url", ""), view.get("number", "?")))
    lines.append("Head SHA: {}".format(view.get("headRefOid", "unknown")))
    lines.append("State: {}  Mergeable: {}  Review decision: {}".format(
        view.get("state", "?"), view.get("mergeable", "?"),
        view.get("reviewDecision", "?")))
    lines.append("Pending comments: {}".format(len(pending)))
    lines.append("")
    lines.extend(_format_checks(data.get("checks")))
    lines.append("")
    lines.append("=" * 72)
    lines.append("COMMENTS TO ADDRESS")
    lines.append("=" * 72)

    for item in pending:
        lines.append("")
        lines.append("## Comment {}".format(item["comment_id"]))
        lines.append("File: {}  Author: {}  Date: {}".format(
            item["location"], item["author"], item["created_at"][:10]))
        lines.append("")
        lines.append(item["body"] or "(empty comment body)")
        for reply in item["replies"]:
            lines.append("")
            lines.append("[Reply by {}] {}".format(reply["author"], reply["body"]))
        lines.append("")
        lines.append("-- end comment {} --".format(item["comment_id"]))

    reviews = [r for r in (data.get("reviews") or [])
               if isinstance(r, dict) and (r.get("body") or "").strip()]
    if reviews:
        lines.append("")
        lines.append("=" * 72)
        lines.append("TOP-LEVEL REVIEWS")
        lines.append("=" * 72)
        for review in reviews:
            lines.append("")
            lines.append("--- review by {} ({}) ---".format(
                (review.get("user") or {}).get("login", "unknown"),
                review.get("state", "?")))
            lines.append(review["body"].strip())

    issue_comments = [c for c in (data.get("issue_comments") or [])
                      if (c.get("body") or "").strip() and not _is_bot(c.get("user"))]
    if issue_comments:
        lines.append("")
        lines.append("=" * 72)
        lines.append("GENERAL PR COMMENTS")
        lines.append("=" * 72)
        for comment in issue_comments:
            lines.append("")
            lines.append("--- comment {} by {} ---".format(
                comment.get("id"), (comment.get("user") or {}).get("login", "unknown")))
            lines.append(comment["body"].strip())

    return "\n".join(lines) + "\n"


def summarize(inbox_dir, our_logins=(), round_no=None):
    """Write review_summary.txt / NEW_COMMENTS.md / pending_comments.json."""
    round_no = round_no or os.path.basename(os.path.abspath(inbox_dir)).replace("round_", "")
    data = load_round(inbox_dir)
    pending = select_pending(data, our_logins=our_logins)

    new_path = os.path.join(inbox_dir, NEW_COMMENTS_FILE)
    with open(new_path, "w") as f:
        f.write(render_new_comments(data, pending, round_no))

    summary_path = os.path.join(inbox_dir, REVIEW_SUMMARY_FILE)
    with open(summary_path, "w") as f:
        f.write(render_new_comments(
            data, select_pending(data, our_logins=our_logins, cursor=None),
            "{} (all, no cursor)".format(round_no)))

    pending_path = os.path.join(inbox_dir, PENDING_COMMENTS_FILE)
    with open(pending_path, "w") as f:
        json.dump({"round": round_no, "pending": pending}, f, indent=2, sort_keys=True)
        f.write("\n")

    seen_ids = [c["id"] for c in (data.get("comments") or [])
                if isinstance(c, dict) and "id" in c]
    return {
        "pending_count": len(pending),
        "pending_path": pending_path,
        "new_comments_path": new_path,
        "max_comment_id": max(seen_ids) if seen_ids else None,
        "head_sha": (data.get("pr_view") or {}).get("headRefOid", ""),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize fetched PR review data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--inbox", required=True, help="round_N directory of raw JSON")
    p_sum.add_argument("--state", default=None, help="pr_state.json to advance cursor")
    p_sum.add_argument("--our-login", action="append", default=[], dest="our_logins")
    p_sum.add_argument("--round", default=None)
    p_sum.add_argument("--no-cursor-update", action="store_true")

    p_count = sub.add_parser("count", help="print pending count; exit 28 when zero")
    p_count.add_argument("--inbox", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "summarize":
        cursor = None
        state_path = None
        if args.state:
            from llm.auto_code_gen.pr_flow import pr_state as state_mod
            state_path = args.state
            cursor = state_mod.load_state(state_path).get("cursor")
            if not args.our_logins:
                args.our_logins = [state_mod.load_state(state_path).get("fork_owner", "")]

        data = load_round(args.inbox)
        pending = select_pending(data, cursor=cursor, our_logins=args.our_logins)
        round_no = args.round or os.path.basename(os.path.abspath(args.inbox))

        with open(os.path.join(args.inbox, NEW_COMMENTS_FILE), "w") as f:
            f.write(render_new_comments(data, pending, round_no))
        with open(os.path.join(args.inbox, REVIEW_SUMMARY_FILE), "w") as f:
            f.write(render_new_comments(
                data, select_pending(data, our_logins=args.our_logins), round_no))
        with open(os.path.join(args.inbox, PENDING_COMMENTS_FILE), "w") as f:
            json.dump({"round": round_no, "pending": pending}, f, indent=2, sort_keys=True)
            f.write("\n")

        seen = [c["id"] for c in (data.get("comments") or []) if "id" in c]
        max_id = max(seen) if seen else (cursor or {}).get("max_comment_id", 0)
        if state_path and not args.no_cursor_update:
            from llm.auto_code_gen.pr_flow import pr_state as state_mod
            new_cursor = dict((state_mod.load_state(state_path).get("cursor") or {}))
            new_cursor["max_comment_id"] = max_id
            new_cursor["last_seen_at"] = max(
                [c.get("updated_at") or c.get("created_at") or "" for c in data["comments"]]
                or [new_cursor.get("last_seen_at", "")])
            state_mod.update(state_path, cursor=new_cursor)

        print("pending comments: {}".format(len(pending)))
        return 0 if pending else 28

    if args.command == "count":
        pending_path = os.path.join(args.inbox, PENDING_COMMENTS_FILE)
        payload = _load_json(pending_path, {"pending": []})
        count = len(payload.get("pending") or [])
        print(count)
        return 0 if count else 28

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
