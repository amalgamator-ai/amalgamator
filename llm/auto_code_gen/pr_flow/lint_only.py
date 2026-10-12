# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Record a provably-cosmetic iteration without any cluster work.

Formatting-only fixes (ruff format, ruff --fix, import ordering, lint
suppressions) change no runtime behavior, so they do not need a benchmark,
lm_eval, NSYS run, or recompile. But they DO change the tree bytes, which
breaks the fingerprint chain: step 4 refuses to push a tree no verdict
measured. This module closes that gap with proof instead of measurement:

    classify:  run the repo's own Python fixers (ruff check --fix, ruff
               format) on the PREVIOUS tree's files; the iteration is
               lint-only iff its result is byte-identical to the NEW tree.
    record:    write a genuine V[N] with a PASS verdict whose perf/lm_eval
               blocks are CARRIED from the last cluster-measured verdict,
               plus a fresh measured_content.json at the new HEAD.

A lint-only iteration therefore never creates a performance number, never
touches the cluster, and keeps every step-4 guardrail (verdict, clean-tree,
measured-content, DCO) intact.

CLI:
    python -m llm.auto_code_gen.pr_flow.lint_only --config <cfg> [--reason "..."]
"""

import os
import subprocess
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


from llm.auto_code_gen.pr_flow import gates, pr_state

# The fixers the upstream pre-commit action runs on Python files
# (.pre-commit-config.yaml: ruff-check --fix, ruff-format). Anything else in
# the diff means the change is not provably cosmetic.
DEFAULT_RUFF_BIN = os.environ.get("LINT_ONLY_RUFF_BIN", "ruff")


def _ruff_fix_in_place(ruff_bin, worktree, rel):
    """Run the repo's Python fixers on one file. Returns True on success."""
    path = os.path.join(worktree, rel)
    try:
        for args in (["check", "--fix", "--quiet", path], ["format", "--quiet", path]):
            result = subprocess.run([ruff_bin] + args, capture_output=True, text=True)
            if result.returncode not in (0, 1):
                return False
    except OSError:
        return False  # ruff binary missing
    return True


def classify_lint_only(repo_dir, previous_head, files, ruff_bin=DEFAULT_RUFF_BIN):
    """Is (previous_head -> HEAD) in `files` reproducible by the fixers alone?

    Runs the fixers on a temporary worktree of `previous_head`, so they see the
    repo's own ruff config (rule selection and per-file ignores differ per
    repo -- defaults would misclassify). Returns (verdict, detail). verdict is
    one of:
      "lint_only"     -- every file's fixed old content equals HEAD's content
      "not_lint_only" -- some diff is not formatter output
      "error"         -- could not run the proof (missing ruff, bad git state)
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lint-only-") as tmp:
        worktree = os.path.join(tmp, "wt")
        add = subprocess.run(
            [
                "git",
                "-C",
                repo_dir,
                "worktree",
                "add",
                "--detach",
                "--quiet",
                worktree,
                previous_head,
            ],
            capture_output=True,
            text=True,
        )
        if add.returncode != 0:
            return (
                "error",
                f"could not check out {previous_head[:12]}: {add.stderr.strip()}",
            )
        try:
            problems = []
            for rel in sorted(files):
                path = os.path.join(worktree, rel)
                if not os.path.isfile(path):
                    return "error", f"{rel} does not exist at {previous_head[:12]}"
                old = subprocess.run(
                    ["git", "-C", repo_dir, "show", f"{previous_head}:{rel}"],
                    capture_output=True,
                    text=True,
                )
                new = subprocess.run(
                    ["git", "-C", repo_dir, "show", f"HEAD:{rel}"],
                    capture_output=True,
                    text=True,
                )
                if old.returncode != 0 or new.returncode != 0:
                    return (
                        "error",
                        f"could not read {rel} at {previous_head[:12]} or HEAD",
                    )
                if old.stdout == new.stdout:
                    continue  # unchanged file (rename-only, etc.)
                if not _ruff_fix_in_place(ruff_bin, worktree, rel):
                    return "error", f"ruff failed on {rel} ({ruff_bin} not on PATH?)"
                with open(path, encoding="utf-8") as f:
                    fixed_content = f.read()
                if fixed_content != new.stdout:
                    problems.append(rel)
        finally:
            subprocess.run(
                ["git", "-C", repo_dir, "worktree", "remove", "--force", worktree],
                capture_output=True,
            )
    if problems:
        return (
            "not_lint_only",
            "formatter output differs from the new tree in: {}".format(
                ", ".join(problems)
            ),
        )
    return (
        "lint_only",
        f"{len(files)} file(s) reproduce exactly under ruff check --fix + ruff format",
    )


def _last_measured_verdict(output_dir):
    """Newest verdict whose perf block holds a real cluster measurement."""
    root = os.path.join(output_dir, "runtime")
    import re

    found = []
    try:
        for name in os.listdir(root):
            match = re.match(r"^V(\d+)$", name)
            path = os.path.join(root, name, gates.VERDICT_FILE)
            if match and os.path.isfile(path):
                found.append((int(match.group(1)), path))
    except OSError:
        return None, None, None
    for number, path in sorted(found, reverse=True):
        verdict = gates.read_verdict(path)
        if not verdict:
            continue
        if (verdict.get("perf") or {}).get("measured"):
            return number, verdict, path
    return None, None, None


def _is_ci_wiring_addition(repo_dir, prev_head, head, rel):
    """Does the (prev_head -> head) diff for `rel` only ADD test wiring?

    Accepts .buildkite pipeline yamls whose diff solely wires existing test
    files under tests/ into jobs -- added path/dependency lines, or a pytest
    command line edited only by inserting test paths. Wiring changes nothing
    at runtime. Anything else (a job dropped, a timeout changed, an existing
    arg removed) makes this False.
    """
    if not (rel.startswith(".buildkite/") and rel.endswith((".yaml", ".yml"))):
        return False
    diff = subprocess.run(
        ["git", "-C", repo_dir, "diff", "-U0", prev_head, head, "--", rel],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    additions, removals = [], []
    for line in diff.splitlines():
        if line.startswith(("+++ ", "--- ", "diff ", "index ", "@@ ")):
            continue
        if line.startswith("+"):
            additions.append(line[1:])
        elif line.startswith(("-", "\\")):
            removals.append(line[1:])
    if not additions:
        return False

    def _is_test_path(token):
        """A token naming an existing file under tests/.

        Commands run from tests/, so a bare ``models/x/test_y.py`` arg and a
        ``tests/models/x/test_y.py`` dependency entry both count.
        """
        if os.path.isfile(os.path.join(repo_dir, "tests", token)):
            return True
        return token.startswith("tests/") and os.path.isfile(
            os.path.join(repo_dir, token)
        )

    def _line_only_wires_tests(line):
        """Every path-like token in the added line is an existing test file,
        and there is at least one."""
        tokens = [t.strip("'\",") for t in line.split() if ".py" in t]
        return bool(tokens) and all(_is_test_path(t) for t in tokens)

    if not removals:
        return all(_line_only_wires_tests(line) for line in additions)

    # At most one line may be replaced (by inserting test paths into it);
    # further lines may only be added wholesale. The replaced line must
    # survive intact as a prefix+suffix of its replacement (the insertion
    # may sit anywhere, e.g. mid-command).
    if len(removals) != 1:
        return False
    removed = removals[0].strip()
    for added in (line.strip() for line in additions):
        if len(added) <= len(removed):
            continue
        for split in range(len(removed) + 1):
            prefix, suffix = removed[:split], removed[split:]
            if added.startswith(prefix) and added.endswith(suffix):
                inserted = added[split : len(added) - len(suffix)]
                if _line_only_wires_tests(inserted):
                    additions = [line for line in additions if line.strip() != added]
                    return all(_line_only_wires_tests(line) for line in additions)
    return False


def _pr_files(state, repo_dir):
    """Files the PR owns: everything that differs from the recorded base."""
    base = state.get("base_commit") or ""
    if not base:
        raise SystemExit("state.base_commit is unset -- step 0 has not run")
    diff = subprocess.run(
        ["git", "-C", repo_dir, "diff", "--name-only", base, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(line for line in diff.splitlines() if line.strip())


def record_lint_only_iteration(config, reason, ruff_bin=None):
    """Write V[N] carrying the last measured verdict. No cluster work."""
    ruff_bin = ruff_bin or DEFAULT_RUFF_BIN

    state_path = pr_state.state_path(config.output_dir)
    state = pr_state.load_state(state_path)
    repo_dir = state["repo_dir"]

    head = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    prev_head = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD~1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    changed = subprocess.run(
        ["git", "-C", repo_dir, "diff", "--name-only", prev_head, head],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    changed = [f for f in changed if f.strip()]
    if not changed:
        raise SystemExit("HEAD~1..HEAD is empty -- nothing to record")

    ci_wiring, cosmetic = [], []
    for rel in changed:
        if rel.endswith(".py"):
            cosmetic.append(rel)
        elif _is_ci_wiring_addition(repo_dir, prev_head, head, rel):
            ci_wiring.append(rel)
        else:
            raise SystemExit(
                f"not lint-only: {rel} is neither Python nor a additions-only CI "
                "wiring edit"
            )
    changed = cosmetic + ci_wiring

    verdict_kind, detail = classify_lint_only(repo_dir, prev_head, cosmetic, ruff_bin)
    if verdict_kind != "lint_only":
        raise SystemExit(
            f"REFUSED ({verdict_kind}): {detail} -- run a full cluster iteration instead"
        )

    carried_no, carried, carried_path = _last_measured_verdict(config.output_dir)
    if carried is None:
        raise SystemExit(
            "no cluster-measured verdict to carry -- run a full iteration first"
        )
    if carried.get("overall") != "PASS":
        raise SystemExit(
            f"carried verdict V{carried_no} is not PASS -- a cosmetic fix cannot repair a "
            "failing gate"
        )

    import re as _re

    runtime_root = os.path.join(config.output_dir, "runtime")
    existing = [
        int(m.group(1))
        for m in (
            _re.match(r"^V(\d+)$", n)
            for n in (os.listdir(runtime_root) if os.path.isdir(runtime_root) else [])
        )
        if m
    ]
    iteration = (max(existing) + 1) if existing else 1
    iter_dir = os.path.join(runtime_root, f"V{iteration}")
    os.makedirs(iter_dir, exist_ok=True)

    perf = dict(carried["perf"])
    perf["measured"] = False
    perf["reason"] = (
        f"carried from V{carried_no} (lint-only iteration, provably cosmetic; "
        "no cluster work)"
    )
    lm_eval = dict(carried.get("lm_eval") or {})
    lm_eval["measured"] = False
    lm_eval["reason"] = f"carried from V{carried_no} (lint-only iteration)"
    closure = {
        "pass": True,
        "total": len(changed),
        "fixed": len(changed),
        "explained": 0,
        "unresolved": 0,
        "unresolved_ids": [],
        "reason": "",
    }

    notes = [
        reason or "lint-only iteration",
        detail,
        f"carried from V{carried_no} ({carried_path})",
    ]
    if ci_wiring:
        notes.append(
            "CI wiring additions (no runtime change): {}".format(", ".join(ci_wiring))
        )
    verdict = gates.build_verdict(
        state.get("round") or 0,
        iteration,
        head,
        perf,
        lm_eval,
        closure,
        nsys_run=False,
        notes=notes,
        mode="lint-only",
    )
    verdict["base_commit"] = state.get("base_commit")
    verdict["lint_only"] = True
    verdict["lint_only_proof"] = {
        "files": cosmetic,
        "ci_wiring_files": ci_wiring,
        "verified_against": prev_head,
        "fixers": ["ruff check --fix", "ruff format"],
        "carried_from": f"V{carried_no}",
    }
    gates.write_verdict(os.path.join(iter_dir, gates.VERDICT_FILE), verdict)

    all_pr_files = _pr_files(state, repo_dir)
    gates.write_measured_content(
        iter_dir, head, state.get("base_commit"), all_pr_files, repo_dir
    )

    with open(os.path.join(iter_dir, "summary.txt"), "w") as f:
        f.write(f"Lint-only iteration V{iteration}\n")
        f.write(f"head: {head}\nprevious: {prev_head}\n\n")
        f.write(f"Proof: {verdict_kind}\n{detail}\n\n")
        if ci_wiring:
            f.write(
                "CI wiring additions (no runtime change): {}\n\n".format(
                    ", ".join(ci_wiring)
                )
            )
        f.write(
            f"Perf/lm_eval carried from V{carried_no} -- no cluster work was done.\n"
        )

    pr_state.record_gate(
        state_path,
        {
            "stage": "lint-only",
            "reason": reason or "lint-only iteration",
            "iteration": iteration,
            "pr_head_sha": head,
            "base_commit": state.get("base_commit"),
            "perf": perf,
            "lm_eval": lm_eval,
            "closure": closure,
            "overall": "PASS",
            "accepted": True,
        },
    )
    pr_state.update(state_path, head_sha=head, iteration=iteration)

    print(
        f"V{iteration} recorded (lint-only, verdict carried from V{carried_no}): {iter_dir}"
    )
    print(
        "  perf actual {:+.2f}% (carried), required {:+.2f}%".format(
            perf.get("actual_pct", 0), perf.get("required_pct", 0)
        )
    )
    return iteration


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Record a provably-cosmetic (lint-only) runtime iteration "
        "without cluster work. The perf/lm_eval verdict is carried "
        "from the last cluster-measured iteration."
    )
    parser.add_argument("--config", required=True, help="code-gen JSON config")
    parser.add_argument(
        "--reason", default="lint-only iteration", help="label recorded in the verdict"
    )
    parser.add_argument(
        "--ruff-bin",
        default=None,
        help="ruff executable for the idempotence proof " "(default: ruff on PATH)",
    )
    args = parser.parse_args()

    config = pr_state._load_config_quietly(args.config)
    record_lint_only_iteration(config, args.reason, ruff_bin=args.ruff_bin)


if __name__ == "__main__":
    sys.exit(main())
