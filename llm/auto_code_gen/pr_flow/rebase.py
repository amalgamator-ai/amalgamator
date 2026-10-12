"""Rebase a PR branch onto a new upstream base: planning, conflicts, resolution.

A rebase is the one stage where history is rewritten, so it is split in three
and each part keeps the flow's invariant:

    step R1  pr_flow_step_r1_rebase.sh            human, local git, starts/
    (R1b/R1c)                                     continues/aborts the rebase
    R1b      run_rebase_conflict_fix.py           AI, edits conflicted files in
                                                 the paused worktree, no git
    R2       run_rebase_runtime_iters.py          AI, repairs what the rebase
                                                 broke, then re-measures both
                                                 gates on the new base

This module never runs a command that writes: it reads the rebase state, lists
what conflicts, and parses the agent's resolution report.

Resolution report format (written by the agent, parsed by parse_resolution):

    ## File: vllm/model_executor/x.py
    STATUS: resolved | unresolved
    NOTES: <what the upstream side changed and how the patch was re-applied>
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import json
import os
import re
import subprocess

REBASE_DIR = "rebase"
CONFLICTS_FILE = "CONFLICTS.md"
CONFLICT_LIST_FILE = "conflict_files.json"
RESOLUTION_FILE = "rebase_resolution.md"
REBASE_SUMMARY_FILE = "rebase_summary.txt"
PLAN_FILE = "rebase_plan.json"

MARKER_HEAD = re.compile(r"^(<{7})(?:\s|=)")
MARKER_TAIL = re.compile(r"^(={7}|>{7})(?:\s|$)")
MAX_HUNK_CHARS = 6000

RESOLUTION_STATUSES = ("resolved", "unresolved")


def rebase_dir(output_dir, stage_no=None):
    path = os.path.join(pr_flow_dir(output_dir), REBASE_DIR)
    return path if stage_no is None else os.path.join(path, f"R{stage_no}")


def pr_flow_dir(output_dir):
    from llm.auto_code_gen.pr_flow.pr_state import pr_flow_dir as _pf

    return _pf(output_dir)


def _git(repo_dir, *args):
    result = subprocess.run(
        ["git", "-C", repo_dir] + list(args), capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else ""


def _git_dir(repo_dir):
    git_dir = _git(repo_dir, "rev-parse", "--git-dir").strip() or ".git"
    return git_dir if os.path.isabs(git_dir) else os.path.join(repo_dir, git_dir)


def in_progress(repo_dir):
    """True while a rebase is paused waiting for conflict resolution."""
    return any(
        os.path.isdir(os.path.join(_git_dir(repo_dir), kind))
        for kind in ("rebase-merge", "rebase-apply")
    )


def _orig_head(repo_dir):
    """The branch tip git saved when it started the rebase ('' when none)."""
    for kind in ("rebase-merge", "rebase-apply"):
        path = os.path.join(_git_dir(repo_dir), kind, "orig-head")
        if os.path.isfile(path):
            try:
                with open(path) as handle:
                    return handle.read().strip()
            except OSError:
                return ""
    return ""


def pr_head(state, repo_dir):
    """Tip of the PR branch -- while paused, HEAD is only partly replayed.

    'base..HEAD' during a pause lists upstream commits as well, which would
    make the plan describe someone else's work. Prefer the pre-rebase head
    recorded by the human script, then the orig-head git keeps for itself.
    """
    if in_progress(repo_dir):
        for candidate in ((state or {}).get("pre_rebase_head"), _orig_head(repo_dir)):
            ref = (candidate or "").strip()
            if (
                ref
                and _git(
                    repo_dir, "rev-parse", "--verify", "--quiet", ref + "^{commit}"
                ).strip()
            ):
                return _git(repo_dir, "rev-parse", ref).strip()
    return _git(repo_dir, "rev-parse", "HEAD").strip()


def current_step(repo_dir):
    """Where the rebase stands -- for status output and the agent brief."""
    head = _git(repo_dir, "rev-parse", "--short", "HEAD").strip()
    if not in_progress(repo_dir):
        return f"not rebasing (HEAD {head})"
    replaying = (
        _git(repo_dir, "log", "-1", "--format=%h %s", "REBASE_HEAD").strip() or "?"
    )
    return f"PAUSED replaying '{replaying}' on top of {head} -- conflicts outstanding"


def drift(state, repo_dir=None):
    """What the rebase actually moved under the PR, once it is finished.

    'old base' is recovered as merge-base(pre_rebase_head, target) -- the tip the
    branch had before the rebase is pinned by the pr-flow/pre-rebase-* tag, so
    this keeps working after state.base_commit has been re-stamped.

    upstream_only_files is the set the rebased PR must leave byte-identical to
    the new base: files upstream changed that this PR never owned. Editing one
    of those means upstream work was overwritten or reverted.
    """
    repo_dir = repo_dir or state["repo_dir"]
    target = target_ref(state)
    target_sha = _git(repo_dir, "rev-parse", target).strip()
    pre = (state.get("pre_rebase_head") or "").strip()
    head = _git(repo_dir, "rev-parse", "HEAD").strip()
    old_base = (
        _git(repo_dir, "merge-base", pre or head, target_sha).strip()
        if target_sha
        else ""
    )

    # The PR's own files are what it adds ON TOP of its base. Once the branch sits
    # on the new base that is diff(target, HEAD); diffing against the OLD base
    # instead would count every file upstream moved as "ours".
    on_new_base = (
        bool(target_sha)
        and _git(repo_dir, "merge-base", target_sha, head).strip() == target_sha
    )
    pr_base = target_sha if on_new_base else (old_base or target_sha or head)
    pr_files = sorted(
        set(_git(repo_dir, "diff", "--name-only", pr_base, head).splitlines())
    )
    upstream_files = (
        sorted(
            set(
                _git(repo_dir, "diff", "--name-only", old_base, target_sha).splitlines()
            )
        )
        if old_base and target_sha
        else []
    )

    return {
        "old_base": old_base,
        "target": target,
        "target_sha": target_sha,
        "head": head,
        "pre_rebase_head": pre,
        "on_new_base": on_new_base,
        "pr_files": pr_files,
        "upstream_files": upstream_files,
        "drift_files": sorted(set(pr_files) & set(upstream_files)),
        "upstream_only_files": sorted(set(upstream_files) - set(pr_files)),
    }


def target_ref(state):
    return state.get("rebase_target") or "{}/{}".format(
        state.get("upstream_remote") or "upstream",
        state.get("upstream_branch") or "main",
    )


def plan(state, repo_dir=None):
    """Describe the rebase before it happens: base, target, drift, replay set."""
    repo_dir = repo_dir or state["repo_dir"]
    base = state.get("base_commit") or ""
    target = target_ref(state)
    target_sha = _git(repo_dir, "rev-parse", target).strip()
    head = pr_head(state, repo_dir)
    paused = in_progress(repo_dir)

    ahead = behind = ""
    if base and target_sha:
        counts = _git(
            repo_dir, "rev-list", "--left-right", "--count", f"{base}...{target}"
        ).split()
        if len(counts) == 2:
            behind = counts[1]  # what upstream gained since our base
        ahead = _git(repo_dir, "rev-list", "--count", f"{base}..{head}").strip()

    commits = [
        line
        for line in _git(repo_dir, "log", "--oneline", f"{base}..{head}").splitlines()
        if line.strip()
    ]

    touched = sorted(
        set(_git(repo_dir, "diff", "--name-only", base, head).splitlines())
    )
    upstream_moved = []
    if base and target_sha:
        upstream_moved = sorted(
            set(_git(repo_dir, "diff", "--name-only", base, target).splitlines())
        )

    overlap = sorted(set(touched) & set(upstream_moved))
    return {
        "base_commit": base,
        "head": head,
        "paused": paused,
        "target": target,
        "target_sha": target_sha,
        "upstream_commits_since_base": behind,
        "pr_commits_ahead_of_base": ahead,
        "pr_commits": commits,
        "pr_files": touched,
        "upstream_files_changed": len(upstream_moved),
        "overlap_files": overlap,
        "likely_conflict": bool(overlap),
    }


def render_plan(data):
    lines = [
        "Rebase plan: {} -> {}".format(
            (data.get("base_commit") or "?")[:12], data.get("target", "")
        )
    ]
    if data.get("paused"):
        lines.append(
            "  (a rebase is PAUSED -- the numbers below describe the pre-rebase tip)"
        )
    lines.append(
        "  target sha            : {}".format((data.get("target_sha") or "?")[:12])
    )
    lines.append(
        "  upstream is ahead by  : {} commit(s) since this PR's base".format(
            data.get("upstream_commits_since_base") or "?"
        )
    )
    lines.append(
        "  commits to replay     : {} ({})".format(
            data.get("pr_commits_ahead_of_base") or "?",
            "; ".join(data.get("pr_commits") or []) or "none",
        )
    )
    lines.append(
        "  files this PR touches : {}".format(
            ", ".join(data.get("pr_files") or []) or "none"
        )
    )
    lines.append(
        "  upstream changed      : {} file(s) since the base".format(
            data.get("upstream_files_changed")
        )
    )
    if data.get("overlap_files"):
        lines.append(
            "  OVERLAP (conflicts likely): {}".format(", ".join(data["overlap_files"]))
        )
    else:
        lines.append("  overlap: none -- the rebase is likely to be clean")
    return "\n".join(lines)


def unmerged_files(repo_dir):
    """Index entries git left in conflict -- cleared by 'git add', not by editing.

    Only the human script may stage, so while a rebase is paused these stay
    unmerged even after the worktree text is correct. Never treat this alone as
    'the AI has not finished'.
    """
    out = _git(repo_dir, "diff", "--name-only", "--diff-filter=U")
    return sorted(set(line.strip() for line in out.splitlines() if line.strip()))


def conflicted_files(repo_dir):
    """Files that still need attention while a rebase is paused."""
    return sorted(set(unmerged_files(repo_dir)) | set(scan_markers(repo_dir)))


def scan_markers(repo_dir):
    """Worktree files still holding conflict markers."""
    found = []
    candidates = [
        line[3:].strip().strip('"')
        for line in _git(repo_dir, "status", "--porcelain").splitlines()
        if line.strip()
    ]
    for rel in candidates:
        path = os.path.join(repo_dir, rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        head = next(
            (line for line in text.splitlines() if MARKER_HEAD.match(line)), None
        )
        tail = next(
            (line for line in text.splitlines() if MARKER_TAIL.match(line)), None
        )
        if head is not None and tail is not None:
            found.append(rel)
    return sorted(found)


def render_conflicts(repo_dir, files, plan_data, round_no=None):
    """The bounded brief the conflict-fix agent reads."""
    lines = ["# Rebase conflicts -- resolve these in the worktree", ""]
    lines.append(
        "Rebase: {} -> {}".format(
            (plan_data.get("base_commit") or "?")[:12], plan_data.get("target", "")
        )
    )
    lines.append("New base: {}".format(plan_data.get("target_sha", "")))
    lines.append("Stage: {}".format(round_no or "R1"))
    lines.append("")
    lines.append(render_plan(plan_data))
    lines.append("")
    lines.append("=" * 72)
    lines.append(
        "FILES IN CONFLICT ({}): {}".format(
            len(files), ", ".join(files) or "(none -- rebase is clean)"
        )
    )
    lines.append("=" * 72)
    for rel in files:
        lines.append("")
        lines.append(f"## File: {rel}")
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, errors="replace") as f:
                text = f.read()
        except OSError as exc:
            lines.append(f"(unreadable: {exc})")
            continue
        hunk = _conflict_hunks(text)
        if hunk:
            lines.append(hunk[:MAX_HUNK_CHARS])
    lines.append("")
    return "\n".join(lines) + "\n"


def _conflict_hunks(text):
    """Extract each <<<<<<< ... >>>>>>> region with a little context."""
    out = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if MARKER_HEAD.match(lines[index]):
            start = max(0, index - 4)
            end = index
            while end < len(lines) and not MARKER_TAIL.match(lines[end]):
                end += 1
            out.append("\n".join(lines[start : min(end + 5, len(lines))]))
            index = end + 1
        else:
            index += 1
    return "\n\n---\n\n".join(out)


def parse_resolution(path):
    """Parse the per-file resolution the conflict-fix agent must write."""
    summary = {
        "total": 0,
        "resolved": 0,
        "unresolved": 0,
        "files": [],
        "unresolved_files": [],
        "entries": [],
        "pass": False,
        "reason": "",
    }
    if not os.path.isfile(path):
        summary["reason"] = f"{os.path.basename(path)} not written"
        return summary

    current = None
    for line in open(path).read().splitlines():
        header = re.match(r"^##\s*File\s*:?\s*(\S+)?\s*$", line.strip())
        if header:
            current = {
                "file": (header.group(1) or "").strip(),
                "status": "unresolved",
                "notes": "",
            }
            summary["entries"].append(current)
            continue
        if current is None:
            continue
        status = re.match(r"^STATUS:\s*(\w+)", line.strip())
        if status:
            value = status.group(1).lower()
            if value == "fixed":  # tolerate the review-loop wording
                value = "resolved"
            current["status"] = value if value in RESOLUTION_STATUSES else "unresolved"
            continue
        notes = re.match(r"^NOTES:\s*(.*)$", line.strip())
        if notes:
            current["notes"] = notes.group(1).strip()

    for entry in summary["entries"]:
        summary[entry["status"]] += 1
        summary["files"].append(entry["file"])
        if entry["status"] == "unresolved":
            summary["unresolved_files"].append(entry["file"])
    summary["total"] = len(summary["entries"])
    summary["pass"] = summary["total"] > 0 and summary["unresolved"] == 0
    if not summary["total"]:
        summary["reason"] = "no '## File:' blocks found"
    elif summary["unresolved"]:
        summary["reason"] = "{} unresolved file(s): {}".format(
            summary["unresolved"], summary["unresolved_files"]
        )
    return summary


def closure_from_rebase(summary):
    """Map a rebase resolution onto the flow's closure-gate shape."""
    return {
        "pass": bool(summary["pass"]),
        "total": summary["total"],
        "fixed": summary["resolved"],
        "explained": 0,
        "unresolved": summary["unresolved"],
        "unresolved_ids": summary["unresolved_files"],
        "reason": summary["reason"],
        "kind": "rebase-conflicts",
    }


def record(output_dir, repo_dir, state, stage_no=None):
    """Snapshot the current rebase facts into the stage directory for auditing."""
    directory = rebase_dir(output_dir, stage_no or 1)
    os.makedirs(directory, exist_ok=True)
    files = conflicted_files(repo_dir)
    plan_data = plan(state, repo_dir)
    with open(os.path.join(directory, CONFLICT_LIST_FILE), "w") as f:
        json.dump(
            {"files": files, "in_progress": in_progress(repo_dir)},
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    with open(os.path.join(directory, PLAN_FILE), "w") as f:
        json.dump(plan_data, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(directory, CONFLICTS_FILE), "w") as f:
        f.write(render_conflicts(repo_dir, files, plan_data, f"R{stage_no or 1}"))
    with open(os.path.join(directory, REBASE_SUMMARY_FILE), "w") as f:
        f.write(render_plan(plan_data) + "\n\n" + current_step(repo_dir) + "\n")
    return {
        "directory": directory,
        "conflict_files": files,
        "in_progress": in_progress(repo_dir),
        "plan": plan_data,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a PR rebase (read-only)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--state", required=True)

    p = sub.add_parser("status")
    p.add_argument("--state", required=True)

    p = sub.add_parser("snapshot")
    p.add_argument("--state", required=True)
    p.add_argument("--stage", default="1")

    p = sub.add_parser("check-resolved")
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--resolution", required=True)

    return parser.parse_args()


def main():
    from llm.auto_code_gen.pr_flow import pr_state

    args = parse_args()

    if args.command == "check-resolved":
        summary = parse_resolution(args.resolution)
        markers = scan_markers(args.repo_dir)
        for name in summary["unresolved_files"]:
            print(f"UNRESOLVED: {name}")
        for name in markers:
            print(f"CONFLICT MARKERS LEFT: {name}")
        if summary["pass"] and not markers:
            print("rebase conflicts resolved in {} file(s)".format(summary["total"]))
            return 0
        if not summary["pass"] and not summary["reason"]:
            print("resolution report incomplete")
        return 30

    state = pr_state.load_state(args.state)
    repo_dir = state["repo_dir"]

    if args.command == "plan":
        print(render_plan(plan(state, repo_dir)))
    elif args.command == "status":
        print(current_step(repo_dir))
    elif args.command == "snapshot":
        data = record(state["output_dir"], repo_dir, state, int(args.stage))
        print(data["directory"])
        print("conflicts: {}".format(", ".join(data["conflict_files"]) or "none"))
        print("in progress: {}".format(data["in_progress"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
