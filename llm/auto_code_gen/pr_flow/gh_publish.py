"""Render and (on explicit --execute) post the human-visible PR text.

Two kinds of text leave this machine during the review loop:

  * one top-level PR comment per round, summarizing what changed and the numbers
    that re-verified the change (perf, lm_eval, comment closure);
  * one reply per inline review comment, taken from the REPLY: line the fix step
    wrote in comment_resolution.md.

Both are rendered to files first so the operator can read them before anything
is posted, and posting only happens when --execute is passed by the step 4
script after the operator typed 'yes'. Nothing here is imported by the agent.

    python -m llm.auto_code_gen.pr_flow.gh_publish render-comment \\
        --state <pr_state.json> --out <file> [--round N] [--iteration V]
    python -m llm.auto_code_gen.pr_flow.gh_publish replies \\
        --replies <thread_replies.json> --state <pr_state.json> [--execute]
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import json
import os
import subprocess

from llm.auto_code_gen.pr_flow import gates, pr_state

MAX_BODY_CHARS = 60000


def _owner_repo(state):
    repo = state.get("upstream_repo") or ""
    if "/" not in repo:
        raise SystemExit(f"state.upstream_repo is not owner/repo: {repo}")
    return repo.split("/", 1)


def _gate_for(state, round_no=None, iteration=None):
    """Newest gate record matching round/iteration, newest overall otherwise."""
    records = list(state.get("gates") or [])
    if round_no is not None:
        records = [
            g for g in records if str(g.get("round")) == str(round_no)
        ] or records
    if iteration is not None:
        exact = [g for g in records if str(g.get("iteration")) == str(iteration)]
        records = exact or records
    return records[-1] if records else None


def _fmt_pct(value):
    if isinstance(value, (int, float)):
        return f"{value:+.2f}%"
    return "n/a"


def render_round_comment(state, round_no=None, iteration=None):
    """Markdown body for the top-level 'round N update' PR comment."""
    gate = _gate_for(state, round_no, iteration)
    if not gate:
        raise SystemExit("no gate record in state -- nothing verified, nothing to post")

    perf = gate.get("perf") or {}
    lm_eval = gate.get("lm_eval") or {}
    closure = gate.get("closure") or {}
    verdict = gates.read_verdict(gates.latest_verdict(state["output_dir"])) or {}

    lines = []
    lines.append(
        "### Changes since the previous push (review round {})".format(
            gate.get("round")
        )
    )
    lines.append("")
    lines.append(
        "Addressed review feedback in verification iteration V{}.".format(
            gate.get("iteration")
        )
    )
    lines.append("")
    lines.append("| check | result |")
    lines.append("| --- | --- |")
    lines.append(
        "| throughput vs. baseline | {} |".format(
            "**{}**".format(_fmt_pct(perf.get("actual_pct")))
            if perf.get("pass")
            else _fmt_pct(perf.get("actual_pct"))
        )
    )
    lines.append(
        "| correctness (gsm8k) | {} |".format(
            "PASS"
            if lm_eval.get("pass")
            else "FAIL ({})".format(
                lm_eval.get("verdict") or lm_eval.get("reason") or "-"
            )
        )
    )
    lines.append(
        "| review comments addressed | {} of {} |".format(
            int(closure.get("fixed") or 0) + int(closure.get("explained") or 0),
            closure.get("total") or 0,
        )
    )
    lines.append("")

    resolution = verdict.get("notes") or []
    if resolution:
        lines.append("Notes:")
        for note in resolution[-6:]:
            lines.append(f"- {note}")
        lines.append("")

    body = "\n".join(lines)
    return body[:MAX_BODY_CHARS]


def render_thread_replies(replies_path):
    """[(comment_id, body)] from the thread_replies.json written by the runner."""
    try:
        with open(replies_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {replies_path}: {exc}")
    out = []
    for entry in payload.get("replies") or []:
        comment_id = entry.get("comment_id")
        body = (entry.get("body") or "").strip()
        if comment_id is None or not body:
            continue
        out.append((int(comment_id), body[:MAX_BODY_CHARS]))
    return out


def _gh(*args, **kwargs):
    execute = kwargs.pop("execute", True)
    if kwargs:
        raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
    if not execute:
        print("  (would run) gh " + " ".join(args))
        return 0
    result = subprocess.run(["gh"] + list(args))
    return result.returncode


def post_round_comment(state, body, execute=True):
    owner, repo = _owner_repo(state)
    number = state.get("pr_number")
    if not number:
        raise SystemExit("state.pr_number is unset -- cannot comment")
    return _gh(
        "pr",
        "comment",
        str(number),
        "--repo",
        f"{owner}/{repo}",
        "--body",
        body,
        execute=execute,
    )


def post_reply(state, comment_id, body, execute=True):
    owner, repo = _owner_repo(state)
    if not state.get("pr_number"):
        raise SystemExit("state.pr_number is unset -- cannot reply")
    path = "repos/{}/{}/pulls/{}/comments/{}/replies".format(
        owner, repo, state.get("pr_number"), comment_id
    )
    return _gh("api", "--method", "POST", path, "-f", "body=" + body, execute=execute)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render/post PR text for the review loop"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("render-comment", help="write the round comment to a file")
    p.add_argument("--state", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--round", default=None)
    p.add_argument("--iteration", default=None)

    p = sub.add_parser("comment", help="post a rendered round comment")
    p.add_argument("--state", required=True)
    p.add_argument("--body-file", required=True)
    p.add_argument(
        "--execute",
        action="store_true",
        help="actually post; without it only the gh command is printed",
    )

    p = sub.add_parser("replies", help="post per-thread replies (or print them)")
    p.add_argument("--replies", required=True, help="thread_replies.json from a V dir")
    p.add_argument("--state", required=True)
    p.add_argument(
        "--execute",
        action="store_true",
        help="actually post; without it only the gh commands are printed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "render-comment":
        state = pr_state.load_state(args.state)
        body = render_round_comment(state, args.round, args.iteration)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(body.rstrip() + "\n")
        print(args.out)
        return 0

    if args.command == "comment":
        state = pr_state.load_state(args.state)
        with open(args.body_file) as f:
            return post_round_comment(state, f.read(), execute=args.execute)

    if args.command == "replies":
        state = pr_state.load_state(args.state)
        if not state.get("pr_number"):
            raise SystemExit("state.pr_number is unset -- cannot reply")
        entries = render_thread_replies(args.replies)
        if not entries:
            print("no thread replies to post")
            return 0
        failures = 0
        for comment_id, body in entries:
            print(f"  reply -> comment {comment_id}")
            if post_reply(state, comment_id, body, execute=args.execute) != 0:
                failures += 1
                if not args.execute:
                    return 29
        if failures:
            print(f"{failures} reply/replies failed")
            return 29
        print(
            "{} reply/replies {}".format(
                len(entries), "posted" if args.execute else "rendered (not posted)"
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
