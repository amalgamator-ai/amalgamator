# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fix + verify runtime iterations after a rebase -- repair, then re-prove.

A rebase that no longer conflicts textually can still be broken: upstream renames
the symbol the patch calls, a registration moves, a signature gains a keyword-only
argument, or the fast path quietly stops being taken. So this stage is a *normal*
runtime iteration, not a measurement: the agent may fix code, and only then is the
tree measured against the perf and correctness gates.

    1. agent audits the resolution for semantic damage, diagnoses whatever the
       previous measurement reported, and repairs the rebased tree
    2. files upstream owns and this PR does not are restored -- upstream work is
       never "fixed" by editing it
    3. cheap local syntax check -- a marker or a bad edit costs seconds, not a
       cluster run
    4. rsync to the PVC; the shared OFF arms are re-measured for the NEW base
    5. bench ON + perf gate; NSYS only when perf regressed; then lm_eval
    6. verify_verdict.json (mode 'rebase') + one gate record per iteration

Exit codes:
    0   gates green on the new base -- commit (if it changed) and publish
    30  the rebase is not finished, or nothing was rebased
    50  iterations exhausted without green gates -- human decision required
    51  precondition problem (no state, repo mismatch)

Usage:
    python -m llm.auto_code_gen.pr_flow.run_rebase_runtime_iters --config <code_gen.json> \
        [--reason 'rebased onto upstream/main'] [--max-iterations 3] [--dry-run]
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import asyncio
import os
import shutil
import sys
import time

from common.claude_utils import PipelineStep
from common.utils import setup_logging
from llm.auto_code_gen.agents import agent_run, agent_session
from llm.auto_code_gen.code_gen_configs import load_config_and_use_case
from llm.auto_code_gen.code_gen_prompts import (
    PR_COMMANDS_DIR,
    PR_PR_DESC_FILE,
    RUNTIME_DIR,
)
from llm.auto_code_gen.pr_flow import checks, gates, pr_state, rebase
from llm.auto_code_gen.pr_flow.prompts import (
    FOLLOWUP_COMMIT_MSG_FILE,
    REBASE_FIX_REPORT_FILE,
    REBASE_RESOLUTION_FILE,
    REBASE_SUMMARY_DRAFT_FILE,
    SUMMARY_DRAFT_FILE,
    gen_RebaseRuntimeFixPrompt,
)
from llm.auto_code_gen.pr_flow.run_pr_review_iters import (
    LMEVAL_COPY_FILES,
    _changed_files,
    _ensure_baseline,
    _feature_fingerprint,
    _gate_report,
    _git,
    _git_ok,
    _patch_scope,
    _prepare_nsys_baseline,
    _repo_dir,
    _stamp_nsys_baseline,
    _write,
)
from llm.auto_code_gen.run_code_gen import (
    _check_run_succeeded,
    _cleanup_stale_cluster_resources,
    _compute_perf_comparison,
    _find_latest_runtime_iteration,
    _link_baseline_results,
    _read_expected_improvement_pct,
    _read_feature_disable_env_var,
    _read_status,
    _recompute_lm_eval_verdict,
    _run_cluster_job,
    _run_nsys_step,
    _sync_tree_to_pvc,
)

EXIT_SUCCESS = 0
EXIT_REBASE = 30
EXIT_EXHAUSTED = 50
EXIT_PRECONDITION = 51

UPSTREAM_DRIFT_FILE = "upstream_drift.patch"
REBASE_REPORT_FILE = "rebase_report.txt"
SYNTAX_FILE = "syntax_check.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair the rebased tree and re-prove perf + correctness."
    )
    parser.add_argument("--config", required=True, help="code-gen JSON config")
    parser.add_argument(
        "--reason", default="post-rebase", help="label recorded in the verdict"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="fix/verify iterations allowed (default 3)",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="override the V-number to continue (default: next free)",
    )
    parser.add_argument(
        "--allow-git-network",
        action="store_true",
        help="do not block git network transports for this process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the gates, run nothing",
    )
    from llm.auto_code_gen.agents import add_agent_argument

    add_agent_argument(parser)
    return parser.parse_args()


def check_rebase_done(config, state, repo_dir):
    """The rebase must be over, and it must actually have happened."""
    checks.check_rebase_absent(state)

    published = state.get("remote_head_sha") or state.get("head_sha") or ""
    head = _git(repo_dir, "rev-parse", "HEAD").strip()
    pre = state.get("pre_rebase_head") or ""
    if pre and head == pre:
        raise checks.CheckError(
            EXIT_REBASE,
            f"HEAD {head[:12]} is still the pre-rebase head -- nothing was " "rebased",
        )
    if published and head == published and pre and published == pre:
        raise checks.CheckError(
            EXIT_REBASE,
            f"HEAD {head[:12]} is still what the fork holds -- nothing was " "rebased",
        )

    target_sha = _git(repo_dir, "rev-parse", rebase.target_ref(state)).strip()
    merge_base = _git(repo_dir, "merge-base", target_sha, "HEAD").strip()
    if target_sha and merge_base and merge_base != target_sha:
        raise checks.CheckError(
            EXIT_REBASE,
            f"HEAD is not on top of {rebase.target_ref(state)} ({target_sha[:12]}) -- finish the rebase first",
        )
    return target_sha


def _latest_rebase_stage(output_dir):
    root = rebase.rebase_dir(output_dir)
    if not os.path.isdir(root):
        return ""
    stages = sorted(
        int(n[1:]) for n in os.listdir(root) if n.startswith("R") and n[1:].isdigit()
    )
    return rebase.rebase_dir(output_dir, stages[-1]) if stages else ""


def _write_rebase_evidence(iter_dir, repo_dir, state, stage_dir, drift_data):
    """Everything the fix step needs about the rebase, as files in the iteration."""
    report = os.path.join(iter_dir, REBASE_REPORT_FILE)
    lines = ["Rebase evidence for this iteration", ""]
    lines.append(
        "new base        : {} ({})".format(
            drift_data["target"], drift_data["target_sha"][:12]
        )
    )
    lines.append("old base        : {}".format((drift_data["old_base"] or "?")[:12]))
    lines.append(
        "pre-rebase head : {}".format((drift_data["pre_rebase_head"] or "?")[:12])
    )
    lines.append("current head    : {}".format(drift_data["head"][:12]))
    lines.append("")
    lines.append(
        "files this PR owns ({}): {}".format(
            len(drift_data["pr_files"]), ", ".join(drift_data["pr_files"]) or "none"
        )
    )
    lines.append(
        "upstream also changed these ({}): {}".format(
            len(drift_data["drift_files"]),
            ", ".join(drift_data["drift_files"]) or "none",
        )
    )
    lines.append("")
    lines.append(
        "Files upstream owns and this PR does NOT (must stay identical to"
        " the new base, {}): {}".format(
            len(drift_data["upstream_only_files"]),
            ", ".join(drift_data["upstream_only_files"]) or "none",
        )
    )
    if stage_dir:
        lines.append("")
        lines.append(
            f"conflict brief   : {os.path.join(stage_dir, rebase.CONFLICTS_FILE)}"
        )
        lines.append(
            f"resolution report: {os.path.join(stage_dir, REBASE_RESOLUTION_FILE)}"
        )
        lines.append(
            f"resolver summary : {os.path.join(stage_dir, REBASE_SUMMARY_DRAFT_FILE)}"
        )
    _write(report, "\n".join(lines))

    # What upstream did, restricted to the files this PR cares about.
    drift_patch = os.path.join(iter_dir, UPSTREAM_DRIFT_FILE)
    if drift_data["old_base"] and drift_data["drift_files"]:
        patch = _git(
            repo_dir,
            "diff",
            drift_data["old_base"],
            drift_data["target_sha"],
            "--",
            *drift_data["drift_files"],
        )
    else:
        patch = ""
    _write(drift_patch, patch or "(upstream changed nothing this PR touches)")
    return report, drift_patch


def syntax_problems(repo_dir, files):
    """Compile every changed .py file -- catches leftovers without a cluster run."""
    problems = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        path = os.path.join(repo_dir, rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as handle:
                source = handle.read()
            compile(source, path, "exec")
        except SyntaxError as exc:
            problems.append(f"{rel}:{exc.lineno}: {exc.msg}")
        except OSError as exc:
            problems.append(f"{rel}: unreadable: {exc}")
    markers = [m for m in rebase.scan_markers(repo_dir) if m in files]
    for marker in markers:
        problems.append(f"{marker}: conflict markers still present")
    return problems


def restore_upstream_owned_files(repo_dir, state, drift_data, iter_dir):
    """Undo any edit to a file upstream changed and this PR does not own.

    Those files are already exactly what upstream committed. If the fix step
    touched one, the only correct action is to put it back -- otherwise the PR
    silently reverts upstream work, which no gate will ever catch.
    """
    restored = []
    for rel in drift_data["upstream_only_files"]:
        if not _git_ok(repo_dir, "diff", "--quiet", "HEAD", "--", rel):
            _git_ok(repo_dir, "checkout", "--", rel)
            restored.append(rel)
    if restored:
        _write(
            os.path.join(iter_dir, "reverted_upstream_owned.txt"),
            "\n".join(restored) + "\n",
        )
        print(
            "  Restored file(s) upstream owns and this PR must not edit: {}".format(
                ", ".join(restored)
            )
        )
    return restored


def _perf_for_failed_run(iter_dir, status):
    return {
        "pass": False,
        "measured": False,
        "actual_pct": None,
        "gate_floor_pct": None,
        "regression_floor_pct": None,
        "required_pct": None,
        "recorded_pct": None,
        "reason": f"bench ON did not complete (status {status}) -- the rebased tree "
        "probably fails to start; read the run logs",
    }


def _skipped(reason):
    return {"pass": False, "measured": False, "verdict": "SKIPPED", "reason": reason}


async def run_iterations(config, claude_config, use_case, state, args):
    output_dir = config.output_dir
    repo_dir = _repo_dir(config, state)
    execution_command = config.target_run_command
    expected_pct = _read_expected_improvement_pct(output_dir)
    recorded_pct = gates.recorded_perf_pct(state, output_dir)
    disable_var = _read_feature_disable_env_var(output_dir) or ""
    scope = _patch_scope(config, output_dir)
    context = use_case.create_context_str(claude_config, config)
    pr_desc = os.path.join(output_dir, PR_COMMANDS_DIR, PR_PR_DESC_FILE)
    stage_dir = _latest_rebase_stage(output_dir)

    drift_data = rebase.drift(state, repo_dir)
    base_commit = drift_data["target_sha"]
    results_dir = os.path.join(output_dir, RUNTIME_DIR)

    print("=" * 78)
    print(f"REBASE RUNTIME ITERATIONS -- {args.reason}")
    print("=" * 78)
    print(f"  new base      : {base_commit[:12]}")
    print("  old base      : {}".format((drift_data["old_base"] or "?")[:12]))
    print("  rebased head  : {}".format(drift_data["head"][:12]))
    print(
        "  PR            : {} (#{})".format(
            state.get("pr_url") or "-", state.get("pr_number") or "not published"
        )
    )
    print("  drift files   : {}".format(", ".join(drift_data["drift_files"]) or "none"))
    print(f"  repair scope  : {len(scope)} file(s) the PR owns")
    print("  Feature toggle: {}".format(disable_var or "UNKNOWN"))
    print(
        "  Perf recorded : {}".format(
            f"{recorded_pct:+.2f}%" if recorded_pct is not None else "unknown"
        )
    )
    stale = gates.verify_rebase(base_commit, state)
    if stale["stale_baselines"]:
        print(
            "  Baselines     : {} will be RE-MEASURED on the new base".format(
                ", ".join(stale["stale_baselines"])
            )
        )
    print("=" * 78 + "\n")

    if args.dry_run:
        print(
            _gate_report(expected_pct, recorded_pct, gates.LM_EVAL_MIN_ABSOLUTE, state)
        )
        print(
            "\n[dry-run] each iteration: fix -> syntax check -> rsync -> OFF arms "
            "(re-measured) -> bench ON -> [NSYS if regressed] -> lm_eval"
        )
        print(
            "[dry-run] planned iterations: {} (max {})".format(
                args.start_iteration
                or (_find_latest_runtime_iteration(output_dir) + 1),
                args.max_iterations,
            )
        )
        print("[dry-run] nothing executed.")
        return EXIT_SUCCESS, []

    _cleanup_stale_cluster_resources(config.remote_namespace)
    start_iteration = args.start_iteration or (
        _find_latest_runtime_iteration(output_dir) + 1
    )
    timings = []
    notes = []

    async with agent_session(claude_config) as client:
        for iteration in range(start_iteration, start_iteration + args.max_iterations):
            iter_dir = os.path.join(results_dir, f"V{iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            pr_head = _git(repo_dir, "rev-parse", "HEAD").strip()
            print(
                f"\n--- rebase iteration V{iteration} (attempt {iteration - start_iteration + 1}/{args.max_iterations}) ---"
            )
            _write(
                os.path.join(iter_dir, "started.txt"),
                "STARTED V{} at {} ({})\n".format(
                    iteration, time.strftime("%Y-%m-%d %H:%M:%S"), args.reason
                ),
            )
            _write(os.path.join(iter_dir, "pr_head_at_start.txt"), pr_head + "\n")
            _write(
                os.path.join(iter_dir, "pr_base.patch"),
                _git(repo_dir, "diff", base_commit, "HEAD"),
            )
            report_path, drift_patch = _write_rebase_evidence(
                iter_dir, repo_dir, state, stage_dir, drift_data
            )

            fix_report = os.path.join(iter_dir, REBASE_FIX_REPORT_FILE)
            summary_path = os.path.join(iter_dir, SUMMARY_DRAFT_FILE)
            prompt = gen_RebaseRuntimeFixPrompt(
                context=context,
                repo_dir=repo_dir,
                output_dir=output_dir,
                iteration_dir=iter_dir,
                iteration=iteration,
                pr_number=state.get("pr_number"),
                pr_url=state.get("pr_url") or "-",
                rebase_report=report_path,
                drift_file=drift_patch,
                conflicts_file=(
                    os.path.join(stage_dir, rebase.CONFLICTS_FILE) if stage_dir else ""
                ),
                resolution_file=(
                    os.path.join(stage_dir, REBASE_RESOLUTION_FILE) if stage_dir else ""
                ),
                pr_description_file=pr_desc,
                prev_summary_file=(
                    os.path.join(RUNTIME_DIR, f"V{iteration - 1}", "summary.txt")
                    if iteration > 1
                    else ""
                ),
                prev_verdict_file=gates.latest_verdict(output_dir) or "",
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
                        name=f"repair_after_rebase_V{iteration}",
                        prompt=prompt.prompt(),
                        output_files=[fix_report],
                    ),
                ],
                client=client,
            )
            timings.append((f"repair_V{iteration}", time.time() - start))

            if not os.path.isfile(fix_report):
                note = f"repair step wrote no {REBASE_FIX_REPORT_FILE}"
                notes.append(f"V{iteration}: {note}")
                print(f"  REPAIR step incomplete -- {note}")
                # Record a FAIL: the next attempt and the escalation read verdicts.
                _finish_iteration(
                    state,
                    args,
                    iter_dir,
                    iteration,
                    base_commit,
                    pr_head,
                    _skipped(note),
                    _skipped(note),
                    rebase.closure_from_rebase(rebase.parse_resolution(fix_report)),
                    False,
                    notes,
                )
                continue

            restore_upstream_owned_files(repo_dir, state, drift_data, iter_dir)
            changed = _changed_files(repo_dir)
            _write(
                os.path.join(iter_dir, "patch_draft_R0.patch"),
                _git(repo_dir, "diff", "HEAD"),
            )
            _write(
                os.path.join(iter_dir, "pr_candidate.patch"),
                _git(repo_dir, "diff", base_commit, "HEAD"),
            )
            with open(os.path.join(iter_dir, "changed_files.txt"), "w") as f:
                f.write("\n".join(changed) + ("\n" if changed else ""))

            closure = rebase.closure_from_rebase(rebase.parse_resolution(fix_report))
            problems = syntax_problems(repo_dir, changed)
            _write(
                os.path.join(iter_dir, SYNTAX_FILE),
                "\n".join(problems) if problems else "all changed files compile\n",
            )
            if problems:
                notes.append(
                    "V{}: repaired files do not compile: {}".format(
                        iteration, "; ".join(problems[:5])
                    )
                )
                print("  Repaired tree does not compile -- not spending a cluster run")
                _finish_iteration(
                    state,
                    args,
                    iter_dir,
                    iteration,
                    base_commit,
                    pr_head,
                    _skipped(
                        "repaired tree does not compile: {}".format(
                            "; ".join(problems[:3])
                        )
                    ),
                    _skipped("repaired tree does not compile"),
                    closure,
                    False,
                    notes,
                )
                continue

            print(
                "  Repair changes: {}".format(
                    ", ".join(changed) or "none (verified only)"
                )
            )
            if not _sync_tree_to_pvc(config, repo_dir):
                _write(os.path.join(iter_dir, "apply_status.txt"), "FAILED\n")
                notes.append(f"V{iteration}: rsync to PVC failed")
                continue
            _write(os.path.join(iter_dir, "apply_status.txt"), "SUCCESS\n")

            # Fingerprint the bytes about to be measured. The human commits them
            # afterwards (a different commit, identical content), and step 4 uses
            # this to prove it is pushing what the gates actually saw.
            measured = sorted(
                set(drift_data["pr_files"])
                | set(changed)
                | set(_changed_files(repo_dir))
            )
            gates.write_measured_content(
                iter_dir, pr_head, base_commit, measured, repo_dir
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
                perf = _perf_for_failed_run(iter_dir, _read_status(iter_dir))
                lm_eval = {
                    "pass": False,
                    "verdict": "SKIPPED",
                    "reason": "bench ON did not complete",
                }
                _finish_iteration(
                    state,
                    args,
                    iter_dir,
                    iteration,
                    base_commit,
                    pr_head,
                    perf,
                    lm_eval,
                    closure,
                    False,
                    notes,
                )
                continue

            perf = gates.evaluate_perf(iter_dir, expected_pct, recorded_pct)
            nsys_run = False
            if not perf["pass"]:
                print(
                    "  Perf gate FAILED: {} -- running NSYS to localise the "
                    "regression".format(perf["reason"])
                )
                toggle_fp = _feature_fingerprint(config, state, disable_var, output_dir)
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

            verdict = _finish_iteration(
                state,
                args,
                iter_dir,
                iteration,
                base_commit,
                pr_head,
                perf,
                lm_eval,
                closure,
                nsys_run,
                notes,
            )
            if verdict["overall"] == "PASS":
                if changed and os.path.isfile(summary_path):
                    shutil.copy2(summary_path, os.path.join(iter_dir, "summary.txt"))
                print(
                    f"\n  THE WIN SURVIVED THE REBASE. Verdict: {os.path.join(iter_dir, gates.VERDICT_FILE)}"
                )
                await _refresh_pr_descs(config, claude_config, use_case, client)
                needs_commit = bool(changed) and os.path.isfile(
                    os.path.join(iter_dir, FOLLOWUP_COMMIT_MSG_FILE)
                )
                _print_publish_next(args, state, needs_commit, iter_dir)
                return EXIT_SUCCESS, timings

    escalation = os.path.join(
        rebase.rebase_dir(output_dir),
        f"escalation_runtime_V{start_iteration + args.max_iterations - 1}.md",
    )
    _write(escalation, _escalation_text(state, drift_data, notes, repo_dir))
    print(f"\n  Gates are NOT green after {args.max_iterations} iteration(s).")
    print("  Do not publish. Decide: repair differently, or keep the previous base.")
    print(f"  See {escalation}")
    return EXIT_EXHAUSTED, timings


def _finish_iteration(
    state,
    args,
    iter_dir,
    iteration,
    base_commit,
    pr_head,
    perf,
    lm_eval,
    closure,
    nsys_run,
    notes,
):
    verdict = gates.build_verdict(
        state.get("round") or 0,
        iteration,
        pr_head,
        perf,
        lm_eval,
        closure,
        nsys_run,
        notes=[args.reason, f"base {base_commit[:12]}"],
        mode="rebase",
    )
    verdict["base_commit"] = base_commit
    verdict["rebase_target"] = rebase.target_ref(state)
    gates.write_verdict(os.path.join(iter_dir, gates.VERDICT_FILE), verdict)
    pr_state.record_gate(
        state["_path"],
        {
            "stage": "rebase",
            "reason": args.reason,
            "iteration": iteration,
            "pr_head_sha": pr_head,
            "base_commit": base_commit,
            "perf": perf,
            "lm_eval": lm_eval,
            "closure": {
                key: closure[key]
                for key in (
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
    pr_state.update(
        state["_path"], base_commit=base_commit, head_sha=pr_head, iteration=iteration
    )
    print(
        "\n  V{} on the new base: {} (perf {}, lm_eval {}, files resolved {}/{})".format(
            iteration,
            verdict["overall"],
            "PASS" if perf["pass"] else "FAIL",
            "PASS" if lm_eval["pass"] else "FAIL",
            closure["fixed"],
            closure["total"],
        )
    )
    return verdict


async def _refresh_pr_descs(config, claude_config, use_case, client):
    """Delegate to pr_descs.refresh_after_pass (kept as a thin local name)."""
    from llm.auto_code_gen.pr_flow.pr_descs import refresh_after_pass

    await refresh_after_pass(config, claude_config, use_case, client=client)


def _print_publish_next(args, state, needs_commit, iter_dir):
    config_path = args.config
    print("  Next:")
    if needs_commit:
        print(
            "    bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_1_commit.sh "
            f"--config {config_path} \\"
        )
        print(f"         --message-file {iter_dir}/{FOLLOWUP_COMMIT_MSG_FILE}")
    if state.get("pr_number"):
        print(
            "    bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_4_update_pr.sh "
            f"--config {config_path} --force-with-lease"
        )
    else:
        print(
            "    bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_2_push.sh "
            f"--config {config_path} --remote-branch perf/<short-name>"
        )


def _escalation_text(state, drift_data, notes, repo_dir):
    lines = ["# Rebase runtime iterations -- human decision required", ""]
    lines.append(
        "PR: {} (#{})".format(
            state.get("pr_url") or "not published yet", state.get("pr_number") or "-"
        )
    )
    lines.append(f"Repo: {repo_dir}")
    lines.append(
        "New base: {} ({})".format(
            drift_data["target"], (drift_data["target_sha"] or "")[:12]
        )
    )
    lines.append("Old base: {}".format((drift_data["old_base"] or "?")[:12]))
    lines.append(
        "Safety tag: pr-flow/pre-rebase-{} holds the old head".format(
            (drift_data["pre_rebase_head"] or "?")[:7]
        )
    )
    lines.append("")
    lines.append("## Iterations")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("The rebased tree never reached green gates. Options:")
    lines.append("  * re-run this module -- the failures above are the input;")
    lines.append("  * go back to the pre-rebase base and let the PR merge on it:  ")
    lines.append(
        "      git -C {} reset --hard {}".format(
            repo_dir, drift_data["pre_rebase_head"]
        )
    )
    lines.append("  * resolve the original conflicts differently (stage R1 again).")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    config, use_case = load_config_and_use_case(args.config)
    claude_config = config.make_agent_config(args.agent_config)
    claude_config.cwd = config.source_code_dir
    setup_logging("rebase_runtime_iters")

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
        print("  Network guard: git transports disabled, GitHub tokens cleared")

    try:
        repo_dir = _repo_dir(config, state)
        check_rebase_done(config, state, repo_dir)
        code, _ = asyncio.run(
            run_iterations(config, claude_config, use_case, state, args)
        )
    except checks.CheckError as exc:
        print(f"REFUSED ({exc.code}): {exc}")
        return exc.code
    return code


if __name__ == "__main__":
    sys.exit(main())
