"""Runtime iterations driven by upstream PR review comments.

Same measurement discipline as the code-gen runtime loop, reordered so a review
round is fix-first:

    1. agent reads the fetched comments and edits the PR worktree
    2. the tree is rsynced to the PVC and measured (bench ON vs shared OFF)
    3. perf gate, then lm_eval gate, then review-closure gate
    4. NSYS only when the perf gate regressed (an engagement question, not an
       accuracy question)

Exit codes:
    0   round fully succeeded -- run scripts/pr_flow_step_4_update_pr.sh
    28  no pending comments in this round's inbox
    50  iterations exhausted without a green verdict -- see escalation.md
    51  environment/precondition problem

Usage:
    python -m llm.auto_code_gen.pr_flow.run_pr_review_iters --config <code_gen.json> \
        --agent-config <agent.json> [--round 1] [--max-iterations 4] [--dry-run]
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
import shutil
import subprocess
import sys
import time

from common.claude_utils import PipelineStep
from common.utils import setup_logging
from llm.auto_code_gen.agents import agent_run, agent_session
from llm.auto_code_gen.code_gen_configs import load_config_and_use_case
from llm.auto_code_gen.code_gen_prompts import RUNTIME_DIR
from llm.auto_code_gen.pr_flow import checks, gates, pr_state
from llm.auto_code_gen.pr_flow.prompts import (
    COMMENT_RESOLUTION_FILE,
    SUMMARY_DRAFT_FILE,
    gen_PRCommentFixPrompt,
)
from llm.auto_code_gen.pr_flow.review_comments import (
    NEW_COMMENTS_FILE,
    PENDING_COMMENTS_FILE,
)
from llm.auto_code_gen.run_code_gen import (
    _check_run_succeeded,
    _cleanup_stale_cluster_resources,
    _compute_perf_comparison,
    _find_latest_runtime_iteration,
    _has_shared_baseline,
    _link_baseline_results,
    _read_expected_improvement_pct,
    _read_feature_disable_env_var,
    _read_status,
    _recompute_lm_eval_verdict,
    _retire_shared_baseline,
    _run_cluster_job,
    _run_nsys_step,
    _save_shared_baseline,
    _shared_baseline_dir,
    _sync_tree_to_pvc,
)

EXIT_SUCCESS = 0
EXIT_NOTHING = 28
EXIT_EXHAUSTED = 50
EXIT_PRECONDITION = 51

LMEVAL_COPY_FILES = (
    "lm_eval_enabled_exit_code.txt",
    "lm_eval_enabled_stdout.log",
    "lm_eval_enabled_stderr.log",
    "lm_eval_output",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix PR review comments and re-verify perf + correctness."
    )
    parser.add_argument("--config", required=True, help="code-gen JSON config")
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="review round (default: state.round + 1)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="fix/verify iterations allowed for this round (default 4)",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="override the V-number to continue (default: next free)",
    )
    parser.add_argument(
        "--adopt-existing-baselines",
        metavar="BASE_COMMIT",
        default=None,
        help="stamp the already-measured shared OFF baselines as belonging "
        "to BASE_COMMIT instead of re-measuring them",
    )
    parser.add_argument(
        "--allow-git-network",
        action="store_true",
        help="do not block the git network transports for this process "
        "(off by default: the agent must not reach GitHub)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the gates, run nothing",
    )
    from llm.auto_code_gen.agents import add_agent_argument

    add_agent_argument(parser)
    return parser.parse_args()


def _git(repo_dir, *args):
    result = subprocess.run(
        ["git", "-C", repo_dir] + list(args), capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else ""


def _git_ok(repo_dir, *args):
    return (
        subprocess.run(
            ["git", "-C", repo_dir] + list(args), capture_output=True, text=True
        ).returncode
        == 0
    )


def _write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text if text.endswith("\n") or not text else text + "\n")


def _changed_files(repo_dir):
    out = []
    for line in _git(repo_dir, "status", "--porcelain").splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ")[-1]
        out.append(path.strip().strip('"'))
    return out


def _base_commit(config, state, repo_dir):
    recorded = state.get("base_commit") or ""
    if recorded:
        return recorded
    upstream = state.get("upstream_remote") or "upstream"
    state.get("upstream_branch") or "main"
    merge_base = _git(repo_dir, "merge-base", f"{upstream}/*", "HEAD").strip()
    if not merge_base:
        merge_base = _git(repo_dir, "rev-parse", "HEAD").strip()
        print("  NOTE: no upstream ref to diff against; using HEAD as the PR base")
    return merge_base


def _patch_scope(config, output_dir):
    latest = checks._latest_runtime_patch(output_dir)
    return checks._patch_files(latest) if latest else []


def _repo_dir(config, state):
    """The clone the PR lives in, with a cross-check between state and config.

    The scripts were told which clone to use in step 0; the config carries its
    own source dir. If those ever disagree, fixing comments in the wrong tree
    would be silently wasted -- so refuse instead.
    """
    from_state = state.get("repo_dir") or ""
    from_config = config.source_code_dir or ""
    if from_state:
        from_state = os.path.abspath(from_state)
    if from_config:
        from_config = os.path.abspath(from_config)
    if from_state and from_config and from_state != from_config:
        raise checks.CheckError(
            checks.EXIT_STATE,
            f"state repo_dir ({from_state}) is not the config's source_code_dir ({from_config}) -- re-run "
            "pr_flow_step_0_prepare.sh against the clone the PR lives in",
        )
    return from_state or from_config


def _gate_report(expected_pct, recorded_pct, lm_eval_floor, state):
    lines = ["Current accepted result for this PR:"]
    lines.append(
        f"  throughput delta recorded : {recorded_pct:+.2f}%"
        if recorded_pct is not None
        else "  throughput delta recorded : unknown"
    )
    lines.append(
        f"  perf gate floor           : {expected_pct * 0.3:+.2f}% (0.3 x expected)"
        if expected_pct
        else "  perf gate floor           : unknown"
    )
    lines.append(
        f"  regression floor          : {gates.REGRESSION_RETENTION * 100:.0f}% of the recorded delta must survive"
    )
    lines.append(
        f"  lm_eval floor             : delta >= {gates.LM_EVAL_MIN_DELTA:+.2f}, absolute >= {lm_eval_floor:.3f}"
    )
    prior = len(state.get("gates") or [])
    lines.append(f"  previous gate records     : {prior}")
    return "\n".join(lines)


BASELINE_ARM_ARTIFACTS = {
    "bench_off": ("baseline_exit_code.txt", "baseline_bench_result.json"),
    "lm_eval_off": ("lm_eval_disabled_exit_code.txt", "lm_eval_output_disabled"),
}


def _baseline_run_problem(run_dir, kind, exit_code, status):
    """Return why the OFF arm produced nothing usable, or None when it did.

    A baseline that was never measured must not be stamped as measured. The
    rebase path trusts that stamp to decide whether the shared OFF arms still
    apply to the new base, so a false stamp quietly compares a new ON run
    against numbers from the previous base -- an invented perf delta that still
    reads as PASS. Server-start failures look exactly like this.
    """
    if exit_code != 0:
        return "cluster job exited {} (status {})".format(exit_code, status or "none")
    if status and status != "COMPLETED":
        return f"cluster job status is {status}"
    exit_file, artifact = BASELINE_ARM_ARTIFACTS.get(kind, ("", ""))
    if exit_file:
        try:
            with open(os.path.join(run_dir, exit_file)) as f:
                code = f.read().strip()
        except OSError:
            return f"missing {exit_file}"
        if code != "0":
            return "{} says {!r}".format(exit_file, code or "empty")
    if artifact and not os.path.exists(os.path.join(run_dir, artifact)):
        return f"missing {artifact}"
    if kind == "bench_off":
        try:
            with open(os.path.join(run_dir, artifact)) as f:
                result = json.load(f)
        except (OSError, ValueError) as exc:
            return f"unreadable {artifact}: {exc}"
        throughput = result.get("output_throughput")
        if not isinstance(throughput, (int, float)) or throughput <= 0:
            return f"{artifact} has no usable output_throughput"
    return None


def _feature_fingerprint(config, state, disable_var, output_dir):
    """Token for what the OFF arm actually disables, to decide baseline reuse."""
    return gates.toggle_fingerprint(
        _repo_dir(config, state), disable_var, _patch_scope(config, output_dir)
    )


def _ensure_baseline(
    config,
    output_dir,
    state,
    iter_dir,
    kind,
    execution_command,
    disable_var,
    base_commit,
    timings,
):
    """Return the directory holding the OFF arm, measuring it when necessary."""
    fingerprint = _feature_fingerprint(config, state, disable_var, output_dir)
    if (
        _has_shared_baseline(output_dir, kind)
        and gates.baseline_is_valid(state, kind, base_commit)
        and gates.baseline_toggle_is_valid(state, kind, fingerprint)
    ):
        return _shared_baseline_dir(output_dir, kind)

    print(
        f"  Shared '{kind}' baseline is missing, predates base {base_commit[:12]}, or the toggle now "
        "disables something different -- re-measuring"
    )
    stage = "baseline" if kind == "bench_off" else "lm_eval"
    run_dir = os.path.join(
        iter_dir, "baseline_run" if kind == "bench_off" else "lm_eval_off_run"
    )
    args = dict(
        bench_command=execution_command,
        disable_env_var=disable_var,
        run_lm_eval=(kind != "bench_off"),
        stage=stage,
    )
    if kind == "lm_eval_off":
        args["skip_lm_eval_baseline"] = True
        args["force_feature_off"] = True
    start = time.time()
    exit_code, status = _run_cluster_job("run-iteration", config, run_dir, **args)
    timings.append((f"{kind}_V{os.path.basename(iter_dir)}", time.time() - start))
    problem = _baseline_run_problem(run_dir, kind, exit_code, status)
    if problem:
        raise checks.CheckError(
            EXIT_PRECONDITION,
            "the '{}' arm produced no usable baseline: {} -- nothing is saved and "
            "no base is stamped, so the older baseline stays unusable rather than "
            "silently standing in for this base (see {})".format(
                kind, problem, os.path.join(run_dir, "run_status.txt")
            ),
        )
    _retire_shared_baseline(
        output_dir,
        kind,
        reason="re-measured for base {}".format((base_commit or "unknown")[:12]),
    )
    _save_shared_baseline(output_dir, run_dir, kind)
    # Merge into the on-disk baselines, not the caller's in-memory copy: an
    # earlier _ensure_baseline call in this same process may have stamped
    # another kind, and a merge from the stale dict would clobber it back to
    # its pre-measurement value, forcing every later iteration to re-measure.
    disk_baselines = pr_state.load_state(state["_path"]).get("baselines") or {}
    pr_state.update(
        state["_path"],
        baselines=dict(
            disk_baselines,
            **{
                f"{kind}_base_commit": base_commit,
                f"{kind}_toggle_fingerprint": fingerprint,
            },
        ),
    )
    state = pr_state.load_state(state["_path"])
    return _shared_baseline_dir(output_dir, kind)


def _prepare_nsys_baseline(output_dir, state, base_commit, fingerprint=None):
    """Retire a shared NSYS OFF trace that belongs to a different base commit.

    ``_run_nsys_step`` reuses ``nsys_off`` on an existence check alone -- it has no
    notion of a base commit -- so after a rebase it profiles the new tree against
    the previous base's OFF trace and hands the repair agent a diff dominated by
    upstream drift rather than the PR's own kernel delta. Base awareness lives
    here, where the base is known: retiring first makes the diagnostic measure a
    matching OFF trace, exactly as ``_ensure_baseline`` does for the other arms.
    """
    if not _has_shared_baseline(output_dir, "nsys_off"):
        return False
    if gates.baseline_is_valid(
        state, "nsys_off", base_commit
    ) and gates.baseline_toggle_is_valid(state, "nsys_off", fingerprint):
        return False
    _retire_shared_baseline(
        output_dir,
        "nsys_off",
        reason="re-measured for base {} / current toggle".format(
            (base_commit or "unknown")[:12]
        ),
    )
    return True


def _stamp_nsys_baseline(output_dir, state, base_commit, fingerprint=None):
    """Record which base the shared NSYS OFF trace on disk was measured against."""
    if not base_commit or not _has_shared_baseline(output_dir, "nsys_off"):
        return
    stamps = {"nsys_off_base_commit": base_commit}
    if fingerprint is not None:
        stamps["nsys_off_toggle_fingerprint"] = fingerprint
    pr_state.update(
        state["_path"], baselines=dict(state.get("baselines") or {}, **stamps)
    )
    state["baselines"] = pr_state.load_state(state["_path"]).get("baselines")


async def run_round(config, claude_config, use_case, state, round_no, args):
    output_dir = config.output_dir
    repo_dir = _repo_dir(config, state)
    results_dir = os.path.join(output_dir, RUNTIME_DIR)
    execution_command = config.target_run_command
    expected_pct = _read_expected_improvement_pct(output_dir)
    recorded_pct = gates.recorded_perf_pct(state, output_dir)
    disable_var = _read_feature_disable_env_var(output_dir) or ""
    base_commit = _base_commit(config, state, repo_dir)
    scope = _patch_scope(config, output_dir)
    context = use_case.create_context_str(claude_config, config)

    inbox = pr_state.inbox_dir(output_dir, round_no)
    new_comments_path = os.path.join(inbox, NEW_COMMENTS_FILE)
    checks.check_inbox(inbox)

    print("=" * 78)
    print(f"PR REVIEW ROUND {round_no} -- fix + verify iterations")
    print("=" * 78)
    print(
        "  PR            : {} (#{})".format(
            state.get("pr_url") or "-", state.get("pr_number")
        )
    )
    print(
        "  PR head       : {}".format(
            _git(repo_dir, "rev-parse", "--short", "HEAD").strip()
        )
    )
    print(f"  PR base       : {base_commit[:12]}")
    print(f"  Inbox         : {inbox}")
    print(
        "  Feature toggle: {}".format(
            disable_var or "UNKNOWN (ON/OFF gate would be blind)"
        )
    )
    print(
        "  Perf recorded : {}".format(
            f"{recorded_pct:+.2f}%" if recorded_pct is not None else "unknown"
        )
    )
    print("=" * 78 + "\n")

    if args.dry_run:
        print(
            _gate_report(expected_pct, recorded_pct, gates.LM_EVAL_MIN_ABSOLUTE, state)
        )
        print(
            "\n[dry-run] planned iterations: {} (max {})".format(
                (
                    args.start_iteration
                    or _find_latest_runtime_iteration(output_dir) + 1
                ),
                args.max_iterations,
            )
        )
        print("[dry-run] nothing executed.")
        return EXIT_SUCCESS, []

    # Deleting leftover jobs/pods is a real cluster mutation: never on dry-run.
    _cleanup_stale_cluster_resources(config.remote_namespace)

    start_iteration = args.start_iteration or (
        _find_latest_runtime_iteration(output_dir) + 1
    )
    timings = []
    succeeded = False
    notes = []

    async with agent_session(claude_config) as client:
        for iteration in range(start_iteration, start_iteration + args.max_iterations):
            iter_dir = os.path.join(results_dir, f"V{iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            print(f"\n--- PR review iteration V{iteration} (round {round_no}) ---")
            _write(
                os.path.join(iter_dir, "started.txt"),
                "STARTED V{} at {} (review round {})\n".format(
                    iteration, time.strftime("%Y-%m-%d %H:%M:%S"), round_no
                ),
            )

            pr_head = _git(repo_dir, "rev-parse", "HEAD").strip()
            _write(os.path.join(iter_dir, "pr_head_at_start.txt"), pr_head + "\n")
            _write(
                os.path.join(iter_dir, "pr_base.patch"),
                _git(repo_dir, "diff", base_commit, "HEAD"),
            )

            prev_summary = (
                os.path.join(RUNTIME_DIR, f"V{iteration - 1}", "summary.txt")
                if iteration > 1
                else ""
            )
            prompt = gen_PRCommentFixPrompt(
                context=context,
                repo_dir=repo_dir,
                output_dir=output_dir,
                iteration_dir=iter_dir,
                iteration=iteration,
                round_no=round_no,
                pr_number=state.get("pr_number"),
                pr_url=state.get("pr_url") or "-",
                new_comments_file=new_comments_path,
                prev_summary_file=prev_summary or "(none)",
                prev_verdict_file=gates.latest_verdict(output_dir) or "(none)",
                gate_report=_gate_report(
                    expected_pct, recorded_pct, gates.LM_EVAL_MIN_ABSOLUTE, state
                ),
                feature_toggle=disable_var or "UNKNOWN",
                allowed_files=scope,
            )
            start = time.time()
            await agent_run(
                claude_config,
                [
                    PipelineStep(
                        name=f"fix_review_comments_V{iteration}",
                        prompt=prompt.prompt(),
                        output_files=[
                            os.path.join(
                                os.path.relpath(iter_dir, output_dir),
                                COMMENT_RESOLUTION_FILE,
                            )
                        ],
                    ),
                ],
                client=client,
            )
            timings.append((f"fix_V{iteration}", time.time() - start))

            resolution_path = os.path.join(iter_dir, COMMENT_RESOLUTION_FILE)
            summary_path = os.path.join(iter_dir, SUMMARY_DRAFT_FILE)
            if not os.path.isfile(resolution_path) or not os.path.isfile(summary_path):
                notes.append(
                    f"V{iteration}: fix step did not write {COMMENT_RESOLUTION_FILE} / {SUMMARY_DRAFT_FILE}"
                )
                print("  FIX step incomplete -- wrote no resolution/summary")
                continue

            changed = _changed_files(repo_dir)
            unexpected = sorted(set(changed) - set(scope))
            if unexpected:
                print(
                    "  Reverting out-of-scope edits: {}".format(", ".join(unexpected))
                )
                for path in unexpected:
                    _git_ok(repo_dir, "checkout", "--", path)
                _git_ok(repo_dir, "clean", "-fd", "--")
                changed = _changed_files(repo_dir)
                notes.append(f"V{iteration}: reverted out-of-scope files {unexpected}")

            _write(
                os.path.join(iter_dir, "patch_draft_R0.patch"),
                _git(repo_dir, "diff", "HEAD"),
            )
            _write(
                os.path.join(iter_dir, "pr_candidate.patch"),
                _git(repo_dir, "diff", base_commit),
            )
            with open(os.path.join(iter_dir, "changed_files.txt"), "w") as f:
                f.write("\n".join(changed) + ("\n" if changed else ""))

            closure = gates.parse_comment_resolution(resolution_path)
            _write_thread_replies(iter_dir, closure)

            if not changed:
                previous = gates.read_verdict(gates.latest_verdict(output_dir)) or {}
                perf = dict(
                    previous.get("perf")
                    or {"pass": False, "reason": "no prior verdict"}
                )
                lm_eval = dict(
                    previous.get("lm_eval")
                    or {
                        "pass": False,
                        "reason": "no prior verdict",
                        "verdict": "INHERITED-MISSING",
                    }
                )
                perf["inherited"] = True
                lm_eval["inherited"] = True
                nsys_run = False
                notes.append(
                    f"V{iteration}: no code change (all comments explained) -- gates inherited"
                )
            else:
                if not _sync_tree_to_pvc(config, repo_dir):
                    _write(os.path.join(iter_dir, "apply_status.txt"), "FAILED\n")
                    notes.append(f"V{iteration}: rsync to PVC failed")
                    continue
                _write(os.path.join(iter_dir, "apply_status.txt"), "SUCCESS\n")

                # Fingerprint the bytes about to be measured. The human commits the
                # fixes afterwards -- a different commit with identical content --
                # and step 4 proves the push matches what the gates saw.
                gates.write_measured_content(
                    iter_dir,
                    pr_head,
                    base_commit,
                    sorted(set(scope) | set(changed)),
                    repo_dir,
                )

                bench_off_dir = _ensure_baseline(
                    config,
                    output_dir,
                    state,
                    iter_dir,
                    "bench_off",
                    execution_command,
                    disable_var,
                    base_commit,
                    timings,
                )
                _link_baseline_results(bench_off_dir, iter_dir)

                start = time.time()
                _run_cluster_job(
                    "run-iteration",
                    config,
                    iter_dir,
                    bench_command=execution_command,
                    disable_env_var=disable_var,
                    run_lm_eval=False,
                    stage="bench",
                )
                timings.append((f"bench_on_V{iteration}", time.time() - start))
                _compute_perf_comparison(iter_dir, output_dir)

                if not _check_run_succeeded(iter_dir, iteration):
                    notes.append(
                        f"V{iteration}: bench ON did not complete (status {_read_status(iter_dir)})"
                    )
                    continue

                perf = gates.evaluate_perf(iter_dir, expected_pct, recorded_pct)
                nsys_run = False
                if not perf["pass"]:
                    print(
                        "  Perf gate FAILED: {} -- running NSYS (regression diagnosis)".format(
                            perf["reason"]
                        )
                    )
                    toggle_fp = _feature_fingerprint(
                        config, state, disable_var, output_dir
                    )
                    _prepare_nsys_baseline(output_dir, state, base_commit, toggle_fp)
                    await _run_nsys_step(
                        config,
                        output_dir,
                        iter_dir,
                        execution_command,
                        disable_var,
                        iteration,
                        context,
                        claude_config,
                        client,
                        timings,
                    )
                    _stamp_nsys_baseline(output_dir, state, base_commit, toggle_fp)
                    nsys_run = True
                    lm_eval = {
                        "pass": False,
                        "verdict": "SKIPPED",
                        "reason": "perf gate failed first",
                    }
                else:
                    lm_off_dir = _ensure_baseline(
                        config,
                        output_dir,
                        state,
                        iter_dir,
                        "lm_eval_off",
                        execution_command,
                        disable_var,
                        base_commit,
                        timings,
                    )
                    start = time.time()
                    lm_dir = os.path.join(iter_dir, "lm_eval_run")
                    _run_cluster_job(
                        "run-iteration",
                        config,
                        lm_dir,
                        bench_command=execution_command,
                        disable_env_var=disable_var,
                        run_lm_eval=True,
                        skip_lm_eval_baseline=True,
                        stage="lm_eval",
                    )
                    timings.append((f"lm_eval_on_V{iteration}", time.time() - start))
                    for name in LMEVAL_COPY_FILES:
                        src = os.path.join(lm_dir, name)
                        dst = os.path.join(iter_dir, name)
                        if os.path.isfile(src) and not os.path.exists(dst):
                            shutil.copy2(src, dst)
                        elif os.path.isdir(src) and not os.path.exists(dst):
                            shutil.copytree(src, dst)
                    _recompute_lm_eval_verdict(iter_dir, lm_off_dir)
                    lm_eval = gates.evaluate_lm_eval(iter_dir)

            verdict = gates.build_verdict(
                round_no,
                iteration,
                pr_head,
                perf,
                lm_eval,
                closure,
                nsys_run,
                notes=notes,
            )
            gates.write_verdict(os.path.join(iter_dir, gates.VERDICT_FILE), verdict)
            pr_state.record_gate(
                state["_path"],
                {
                    "round": round_no,
                    "iteration": iteration,
                    "pr_head_sha": pr_head,
                    "perf": perf,
                    "lm_eval": lm_eval,
                    "closure": {
                        k: closure[k]
                        for k in (
                            "pass",
                            "total",
                            "fixed",
                            "explained",
                            "unresolved",
                            "unresolved_ids",
                            "reason",
                        )
                    },
                    "overall": verdict["overall"],
                    "accepted": verdict["overall"] == "PASS",
                },
            )

            print(
                "\n  V{} verdict: {}  (perf {}, lm_eval {}, comments fixed {}/{})".format(
                    iteration,
                    verdict["overall"],
                    "PASS" if perf["pass"] else "FAIL",
                    "PASS" if lm_eval["pass"] else "FAIL",
                    closure["fixed"] + closure["explained"],
                    closure["total"],
                )
            )

            if verdict["overall"] == "PASS":
                shutil.copy2(summary_path, os.path.join(iter_dir, "summary.txt"))
                pr_state.update(state["_path"], round=round_no, iteration=iteration)
                succeeded = True
                from llm.auto_code_gen.pr_flow.pr_descs import refresh_after_pass

                await refresh_after_pass(config, claude_config, use_case, client=client)
                print(
                    f"\n  ROUND {round_no} PASSED. Review the diff, then commit and push:"
                )
                print(
                    "    bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_1_commit.sh "
                    f"--config {args_config_path(state)} \\"
                )
                print(
                    "        --message-file {}".format(
                        os.path.join(iter_dir, "followup_commit_msg.txt")
                    )
                )
                print(
                    "    bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_4_update_pr.sh "
                    f"--config {args_config_path(state)}"
                )
                break

    if not succeeded:
        escalation = os.path.join(
            pr_state.pr_flow_dir(output_dir), f"escalation_round_{round_no}.md"
        )
        _write(escalation, _escalation_text(round_no, state, notes))
        print(
            f"\nRound {round_no} did not reach a green verdict after {args.max_iterations} iteration(s)."
        )
        print(f"  Human decision required -- see {escalation}")
        return EXIT_EXHAUSTED, timings
    return EXIT_SUCCESS, timings


def _write_thread_replies(iter_dir, closure):
    """Emit thread_replies.json for the human push script (one reply per thread)."""
    replies = [
        {
            "comment_id": entry["comment_id"],
            "status": entry["status"],
            "body": entry["reply"],
        }
        for entry in closure["entries"]
        if entry["reply"]
    ]
    _write(
        os.path.join(iter_dir, "thread_replies.json"),
        json.dumps({"replies": replies}, indent=2, sort_keys=True),
    )


def _escalation_text(round_no, state, notes):
    lines = [f"# Review round {round_no} -- human decision required", ""]
    lines.append("PR: {} (#{})".format(state.get("pr_url"), state.get("pr_number")))
    lines.append("")
    lines.append("## Iteration notes (newest last)")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Gate history")
    for gate in (state.get("gates") or [])[-6:]:
        lines.append(
            "- V{} round {}: overall {} | perf {} | lm_eval {} | comments {}".format(
                gate.get("iteration"),
                gate.get("round"),
                gate.get("overall"),
                json.dumps(gate.get("perf"), sort_keys=True),
                json.dumps(gate.get("lm_eval"), sort_keys=True),
                json.dumps(gate.get("closure"), sort_keys=True),
            )
        )
    lines.append("")
    lines.append("Nothing was pushed. The worktree holds the last attempted fixes;")
    lines.append(
        "`git -C <repo> checkout -- .` discards them and restores the PR head."
    )
    return "\n".join(lines)


def args_config_path(state):
    return state.get("config_path") or "<config.json>"


def _resolve_round(state, output_dir):
    """Which review round to run.

    Step 3 writes inbox/round_N and records N in state, so the round that is
    pending work is state.round itself -- not state.round + 1. Only when the
    recorded round has no pending comments do we assume the operator fetched a
    newer round out of band and move on.
    """
    current = int(state.get("round") or 0)
    if current <= 0:
        return 1
    pending = os.path.join(
        pr_state.inbox_dir(output_dir, current), PENDING_COMMENTS_FILE
    )
    try:
        with open(pending) as f:
            return current if json.load(f).get("pending") else current + 1
    except OSError, ValueError:
        return current + 1


def main():
    args = parse_args()
    config, use_case = load_config_and_use_case(args.config)
    claude_config = config.make_agent_config(args.agent_config)
    claude_config.cwd = config.source_code_dir

    setup_logging("pr_review_iters")

    state_path = pr_state.state_path(config.output_dir)
    if not os.path.isfile(state_path):
        print(f"ERROR: {state_path} missing -- run scripts/pr_flow_step_0_prepare.sh")
        return EXIT_PRECONDITION
    state = pr_state.load_state(state_path)
    state["_path"] = state_path

    if not args.allow_git_network:
        os.environ["GIT_SSH_COMMAND"] = "/bin/false"
        os.environ["GIT_TERMINAL_PROMPT"] = "0"
        for token in ("GH_TOKEN", "GITHUB_TOKEN"):
            os.environ.pop(token, None)
        print(
            "  Network guard: git transports disabled, GitHub tokens cleared for "
            "this process and every agent/cluster child"
        )

    if args.adopt_existing_baselines:
        base = args.adopt_existing_baselines
        pr_state.update(
            state_path,
            baselines={
                "bench_off_base_commit": base,
                "lm_eval_off_base_commit": base,
                "nsys_off_base_commit": base,
            },
        )
        state = pr_state.load_state(state_path)
        state["_path"] = state_path
        print(f"  Adopted existing OFF baselines as measured on {base[:12]}")

    if not state.get("pr_number"):
        print(
            "ERROR: pr_state.json has no pr_number -- the PR is not published yet "
            "(scripts/pr_flow_step_2_push.sh)"
        )
        return EXIT_PRECONDITION

    try:
        # A paused rebase means the worktree is mid-history-rewrite; fixing on top
        # of that would measure a tree that does not exist yet.
        print("  " + checks.check_rebase_absent(state))
    except checks.CheckError as exc:
        print(f"REFUSED ({exc.code}): {exc}")
        return exc.code

    round_no = args.round or _resolve_round(state, config.output_dir)
    if not state.get("local_branch"):
        print(
            "ERROR: pr_state.json has no local_branch -- re-run scripts/pr_flow_step_0_prepare.sh"
        )
        return EXIT_PRECONDITION

    try:
        code, _ = asyncio.run(
            run_round(config, claude_config, use_case, state, round_no, args)
        )
    except checks.CheckError as exc:
        print(f"NOTHING TO DO ({exc.code}): {exc}")
        return exc.code
    return code


if __name__ == "__main__":
    sys.exit(main())
