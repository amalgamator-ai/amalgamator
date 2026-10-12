# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rebase stage: planning, conflict snapshots, resolution parsing, guardrails.

Every test builds throwaway repositories under tmp_path -- nothing here touches
a real clone, a remote, or the network.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from llm.auto_code_gen.pr_flow import checks, gates, pr_state, rebase


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def upstream_and_pr(tmp_path):
    """An upstream whose next commit conflicts, plus a PR clone on top of it."""
    if not shutil.which("git"):
        pytest.skip("git not available")
    up = tmp_path / "up"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _git(up, "config", "user.email", "dev@example.com")
    _git(up, "config", "user.name", "Dev")
    (up / "kernel.py").write_text("def gemm(a, b):\n    return old(a, b)\n")
    (up / "cfg.py").write_text("VALUE = 1\n")
    _git(up, "add", "-A")
    _git(up, "commit", "-qm", "base commit")
    base = _git(up, "rev-parse", "HEAD").strip()

    pr = tmp_path / "pr"
    subprocess.run(
        ["git", "clone", "-q", str(up), str(pr)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(pr, "remote", "rename", "origin", "upstream")
    _git(pr, "remote", "add", "fork", "git@github.com:someone/vllm.git")
    _git(pr, "config", "user.email", "dev@example.com")
    _git(pr, "config", "user.name", "Dev")
    _git(pr, "checkout", "-q", "-b", "perf/branch", base)
    (pr / "kernel.py").write_text("def gemm(a, b):\n    return fused(a, b)\n")
    (pr / "cfg.py").write_text('VALUE = 2\nTOGGLE = "VLLM_DISABLE_FOO"\n')
    _git(pr, "add", "-A")
    _git(pr, "commit", "-qm", "perf: fused gemm path")
    pr_head = _git(pr, "rev-parse", "HEAD").strip()

    (up / "kernel.py").write_text(
        "def gemm(a, b, *, stream=None):\n    return new(a, b)\n"
    )
    (up / "cfg.py").write_text("VALUE = 5\n")
    (up / "service.py").write_text("UPSTREAM_CONSTANT = 42\n")  # the PR never owns this
    _git(up, "add", "-A")
    _git(up, "commit", "-aqm", "upstream: rework kernel signature")
    target = _git(up, "rev-parse", "HEAD").strip()
    _git(pr, "fetch", "-q", "upstream")

    state = {
        "repo_dir": str(pr),
        "output_dir": str(tmp_path / "out"),
        "local_branch": "perf/branch",
        "base_commit": base,
        "head_sha": pr_head,
        "remote_head_sha": pr_head,
        "pre_rebase_head": "",
        "rebase_target": "upstream/main",
        "upstream_remote": "upstream",
        "upstream_branch": "main",
    }
    return pr, state, {"base": base, "pr_head": pr_head, "target": target}


def _start_rebase(pr, state):
    _git(pr, "tag", "pr-flow/pre-rebase")
    state["pre_rebase_head"] = _git(pr, "rev-parse", "HEAD").strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(pr),
            "rebase",
            "--onto",
            "upstream/main",
            state["base_commit"],
            "perf/branch",
        ],
        capture_output=True,
        text=True,
    )
    return rebase.in_progress(str(pr))


def test_plan_describes_the_drift_and_the_overlap(upstream_and_pr):
    pr, state, shas = upstream_and_pr
    data = rebase.plan(state)
    assert data["base_commit"] == shas["base"]
    assert data["target_sha"] == shas["target"]
    assert data["upstream_commits_since_base"] == "1"
    assert data["pr_commits_ahead_of_base"] == "1"
    assert "perf: fused gemm path" in "; ".join(data["pr_commits"])
    assert data["pr_files"] == ["cfg.py", "kernel.py"]
    assert data["overlap_files"] == ["cfg.py", "kernel.py"]
    assert data["likely_conflict"] is True
    assert data["paused"] is False
    rendered = rebase.render_plan(data)
    assert "OVERLAP" in rendered and "PAUSED" not in rendered


def test_plan_of_a_non_overlapping_rebase_reports_no_risk(upstream_and_pr):
    """A PR that touches nothing upstream moved is predicted to be clean."""
    pr, state, shas = upstream_and_pr
    _git(pr, "reset", "-q", "--hard", shas["base"])
    (pr / "fast_path.py").write_text("def fused(a, b):\n    return a + b\n")
    _git(pr, "add", "-A")
    _git(pr, "commit", "-qm", "perf: add fused path module")
    state["head_sha"] = _git(pr, "rev-parse", "HEAD").strip()
    data = rebase.plan(state)
    assert data["pr_files"] == ["fast_path.py"]
    assert data["overlap_files"] == []
    assert data["likely_conflict"] is False
    assert "overlap: none" in rebase.render_plan(data)
    # and the rebase really is clean
    assert _start_rebase(pr, state) is False


def test_plan_while_paused_still_describes_the_pr_not_upstream(upstream_and_pr):
    """The regression: base..HEAD during a pause lists upstream's commits."""
    pr, state, shas = upstream_and_pr
    assert _start_rebase(pr, state) is True
    data = rebase.plan(state)
    assert data["paused"] is True
    assert data["head"] == shas["pr_head"]
    assert "upstream: rework kernel signature" not in "; ".join(data["pr_commits"])
    assert "perf: fused gemm path" in "; ".join(data["pr_commits"])
    assert "PAUSED" in rebase.render_plan(data)


def test_conflict_snapshot_lists_files_and_marked_hunks(upstream_and_pr, tmp_path):
    pr, state, _ = upstream_and_pr
    _start_rebase(pr, state)
    recorded = rebase.record(state["output_dir"], str(pr), state, 1)
    stage = tmp_path / "out" / "pr_p3" / "rebase" / "R1"
    assert recorded["conflict_files"] == ["cfg.py", "kernel.py"]
    assert recorded["in_progress"] is True
    brief = (stage / rebase.CONFLICTS_FILE).read_text()
    assert "FILES IN CONFLICT (2)" in brief
    assert "## File: kernel.py" in brief
    assert "<<<<<<<" in brief and ">>>>>>>" in brief
    assert json.loads((stage / rebase.PLAN_FILE).read_text())["paused"] is True
    assert json.loads((stage / rebase.CONFLICT_LIST_FILE).read_text())["files"] == [
        "cfg.py",
        "kernel.py",
    ]


def test_unmerged_and_markers_are_different_things(upstream_and_pr):
    """Resolved text with an unstaged index is the state the AI hands back."""
    pr, state, _ = upstream_and_pr
    repo = str(pr)
    _start_rebase(pr, state)
    assert rebase.unmerged_files(repo) == ["cfg.py", "kernel.py"]
    assert rebase.scan_markers(repo) == ["cfg.py", "kernel.py"]

    (pr / "kernel.py").write_text(
        "def gemm(a, b, *, stream=None):\n    return fused(a, b)\n"
    )
    (pr / "cfg.py").write_text('VALUE = 5\nTOGGLE = "VLLM_DISABLE_FOO"\n')
    assert rebase.scan_markers(repo) == []
    assert rebase.unmerged_files(repo) == ["cfg.py", "kernel.py"]
    assert rebase.conflicted_files(repo) == ["cfg.py", "kernel.py"]


def test_parse_resolution_formats(tmp_path):
    report = tmp_path / rebase.RESOLUTION_FILE
    assert rebase.parse_resolution(str(report))["reason"].endswith("not written")

    report.write_text("nothing here\n")
    empty = rebase.parse_resolution(str(report))
    assert empty["total"] == 0 and empty["pass"] is False
    assert "no '## File:' blocks" in empty["reason"]

    report.write_text(
        "## File: cfg.py\nSTATUS: resolved\nNOTES: took upstream VALUE\n"
        "## File: kernel.py\nSTATUS: fixed\nNOTES: kept the new signature\n"
    )
    ok = rebase.parse_resolution(str(report))
    assert ok["total"] == 2 and ok["resolved"] == 2 and ok["pass"] is True

    report.write_text(
        "## File: cfg.py\nSTATUS: resolved\nNOTES: fine\n"
        "## File: kernel.py\nSTATUS: unresolved\n"
    )
    bad = rebase.parse_resolution(str(report))
    assert bad["unresolved"] == 1 and bad["pass"] is False
    assert bad["unresolved_files"] == ["kernel.py"]
    assert "1 unresolved file(s)" in bad["reason"]


def test_closure_from_rebase_matches_the_gate_shape():
    summary = rebase.parse_resolution("/nonexistent")
    closure = rebase.closure_from_rebase(summary)
    assert closure["pass"] is False and closure["kind"] == "rebase-conflicts"
    assert closure["unresolved_ids"] == []


def test_rebase_plan_guardrail_refuses_bad_targets(upstream_and_pr):
    pr, state, shas = upstream_and_pr
    _rejected(
        checks.check_rebase_plan,
        dict(state, rebase_target="upstream/nope"),
        code=checks.EXIT_REMOTE,
        match="not present locally",
    )
    _rejected(
        checks.check_rebase_plan,
        dict(state, rebase_target="upstream/main", base_commit=shas["target"]),
        code=checks.EXIT_REBASE,
        match="already the PR base",
    )
    assert "Rebase plan" in checks.check_rebase_plan(state)


def test_rebase_plan_guardrail_refuses_an_uncommitted_patch(upstream_and_pr):
    pr, state, shas = upstream_and_pr
    _git(pr, "reset", "-q", "--hard", shas["base"])
    _rejected(
        checks.check_rebase_plan,
        state,
        code=checks.EXIT_TREE,
        match="did step 1 commit",
    )


def test_paused_and_resume_guardrails(upstream_and_pr):
    pr, state, _ = upstream_and_pr
    _rejected(
        checks.check_rebase_paused,
        state,
        code=checks.EXIT_REBASE,
        match="no rebase is paused",
    )
    _rejected(
        checks.check_rebase_resume,
        state,
        code=checks.EXIT_REBASE,
        match="nothing to continue",
    )
    _start_rebase(pr, state)
    assert "2 conflict(s)" in checks.check_rebase_paused(state)
    # markers left in the worktree always block a continue
    _rejected(
        checks.check_rebase_resume,
        state,
        code=checks.EXIT_REBASE,
        match="conflict markers remain",
    )


def test_resolved_guardrail_needs_markers_gone_and_every_file_covered(
    upstream_and_pr, tmp_path
):
    pr, state, _ = upstream_and_pr
    _start_rebase(pr, state)
    report = tmp_path / "rebase_resolution.md"
    report.write_text("## File: cfg.py\nSTATUS: resolved\nNOTES: ok\n")
    _rejected(
        checks.check_rebase_resolved,
        state,
        str(report),
        code=checks.EXIT_REBASE,
        match="conflict markers remain",
    )

    (pr / "kernel.py").write_text(
        "def gemm(a, b, *, stream=None):\n    return fused(a, b)\n"
    )
    (pr / "cfg.py").write_text('VALUE = 5\nTOGGLE = "VLLM_DISABLE_FOO"\n')
    # markers gone, but kernel.py is unmerged and unreported
    _rejected(
        checks.check_rebase_resolved,
        state,
        str(report),
        code=checks.EXIT_REBASE,
        match="not covered by the resolution report",
    )

    report.write_text(
        "## File: cfg.py\nSTATUS: resolved\nNOTES: ok\n"
        "## File: kernel.py\nSTATUS: resolved\nNOTES: ok\n"
    )
    outcome = checks.check_rebase_resolved(state, str(report))
    assert "2 conflict(s) resolved" in outcome and "awaiting" in outcome


def _rejected(fn, *args, code=None, match="", **kwargs):
    with pytest.raises(checks.CheckError) as excinfo:
        fn(*args, **kwargs)
    if code is not None:
        assert excinfo.value.code == code, excinfo.value
    if match:
        assert match in str(excinfo.value)
    return excinfo.value


def test_clean_tree_accepts_rewritten_history_only_when_told(upstream_and_pr):
    pr, state, shas = upstream_and_pr
    _start_rebase(pr, state)
    (pr / "kernel.py").write_text(
        "def gemm(a, b, *, stream=None):\n    return fused(a, b)\n"
    )
    (pr / "cfg.py").write_text('VALUE = 5\nTOGGLE = "VLLM_DISABLE_FOO"\n')
    _git(pr, "add", "-A")
    subprocess.run(
        ["git", "-C", str(pr), "-c", "core.editor=true", "rebase", "--continue"],
        check=True,
        capture_output=True,
        text=True,
    )

    rebased = _git(pr, "rev-parse", "HEAD").strip()
    published = state["remote_head_sha"]
    assert not _is_ancestor(pr, published, rebased)
    _rejected(
        checks.check_clean_tree,
        dict(state, head_sha=rebased),
        expect_ahead=True,
        code=checks.EXIT_TREE,
        match="force-with-lease",
    )
    outcome = checks.check_clean_tree(
        dict(state, head_sha=rebased), expect_ahead=True, allow_rewritten=True
    )
    assert "REWRITTEN" in outcome and published[:12] in outcome


def _is_ancestor(pr, sha, head):
    return (
        subprocess.run(
            ["git", "-C", str(pr), "merge-base", "--is-ancestor", sha, head],
            capture_output=True,
        ).returncode
        == 0
    )


def test_rebase_verifies_against_stale_baselines():
    stale = {
        "baselines": {
            "bench_off_base_commit": "oldbase",
            "lm_eval_off_base_commit": "oldbase",
            "nsys_off_base_commit": "oldbase",
        }
    }
    result = gates.verify_rebase("newbase", stale)
    assert result["mode"] == "rebase" and result["overall"] == "PENDING"
    assert sorted(result["stale_baselines"]) == ["bench_off", "lm_eval_off", "nsys_off"]
    assert "re-measured" in result["reason"]
    assert gates.verify_rebase("newbase", {})["stale_baselines"] == [
        "bench_off",
        "lm_eval_off",
        "nsys_off",
    ]
    fresh = {
        "baselines": {
            k + "_base_commit": "newbase"
            for k in ("bench_off", "lm_eval_off", "nsys_off")
        }
    }
    assert gates.verify_rebase("newbase", fresh)["stale_baselines"] == []


def test_state_carries_the_rebase_fields(tmp_path):
    output_dir = str(tmp_path / "out")
    path = pr_state.state_path(output_dir)
    pr_state.save_state(
        path,
        pr_state.new_state(
            output_dir=output_dir,
            repo_dir=str(tmp_path / "repo"),
            fork_url="git@github.com:someone/vllm.git",
            fork_owner="someone",
        ),
    )
    state = pr_state.load_state(path)
    assert state["rebase_target"] == "upstream/main"
    assert state["pre_rebase_head"] == "" and state["remote_head_sha"] == ""
    for field, value in (
        ("pre_rebase_head", "abc"),
        ("rebase_target", "upstream/dev"),
        ("remote_head_sha", "def"),
    ):
        pr_state.update(path, **{field: value})
        assert pr_state.load_state(path)[field] == value
    # the published head is not the local head: they only agree until a rebase
    assert pr_state.load_state(path)["head_sha"] != "def"


def test_conflict_fix_prompt_carries_the_rules(tmp_path):
    from llm.auto_code_gen.pr_flow.prompts import (
        REBASE_RESOLUTION_FILE,
        REBASE_SUMMARY_DRAFT_FILE,
        gen_RebaseConflictFixPrompt,
    )

    text = gen_RebaseConflictFixPrompt(
        context="TRACE CONTEXT",
        repo_dir="/repo",
        output_dir="/out",
        stage_dir="/out/pr_p3/rebase/R1",
        attempt=2,
        conflicts_file="/out/pr_p3/rebase/R1/CONFLICTS.md",
        prev_attempt_file="",
        pr_description_file="/out/pr-generate/pr_desc.txt",
        feature_toggle="VLLM_DISABLE_FOO",
    ).prompt()
    assert "TRACE CONTEXT" in text
    assert REBASE_RESOLUTION_FILE in text and REBASE_SUMMARY_DRAFT_FILE in text
    assert "STATUS: resolved | unresolved" in text
    assert "VLLM_DISABLE_FOO" in text
    assert "(first attempt)" in text
    for forbidden in ("git commit", "git push", "git rebase --continue"):
        assert forbidden not in text


def test_stage_default_follows_the_human_script(tmp_path):
    """R<n> created by the step script is the stage the resolver must fill in."""
    from llm.auto_code_gen.pr_flow import run_rebase_conflict_fix as fixer

    rebase.rebase_dir(str(tmp_path))
    assert fixer.existing_stages(str(tmp_path)) == []
    assert fixer.default_stage(str(tmp_path)) == 1

    stage1 = tmp_path / "pr_p3" / "rebase" / "R1"
    stage1.mkdir(parents=True)
    assert fixer.existing_stages(str(tmp_path)) == []  # empty dir is not a stage
    assert fixer.default_stage(str(tmp_path)) == 1

    (stage1 / rebase.CONFLICTS_FILE).write_text("# conflicts\n")
    assert fixer.existing_stages(str(tmp_path)) == [1]
    assert fixer.default_stage(str(tmp_path)) == 1  # R1 has no report yet

    (stage1 / rebase.RESOLUTION_FILE).write_text("## File: a.py\nSTATUS: resolved\n")
    assert fixer.default_stage(str(tmp_path)) == 2  # R1 finished -> new stage


def test_rebase_runners_agree_on_the_exit_code_contract():
    from llm.auto_code_gen.pr_flow import (
        run_rebase_conflict_fix,
        run_rebase_runtime_iters,
    )

    for module in (run_rebase_conflict_fix, run_rebase_runtime_iters):
        assert module.EXIT_REBASE == checks.EXIT_REBASE == 30
        assert module.EXIT_EXHAUSTED == 50
        assert module.EXIT_PRECONDITION == 51


def test_rebase_runner_cli_defaults(monkeypatch):
    from llm.auto_code_gen.pr_flow import (
        run_rebase_conflict_fix,
        run_rebase_runtime_iters,
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--config", "c.json"])
    fix = run_rebase_conflict_fix.parse_args()
    assert fix.config == "c.json" and fix.max_attempts == 3 and fix.stage is None
    assert fix.dry_run is False and fix.allow_git_network is False

    monkeypatch.setattr(sys, "argv", ["prog", "--config", "c.json"])
    verify = run_rebase_runtime_iters.parse_args()
    assert verify.max_iterations == 3 and verify.reason == "post-rebase"
    assert verify.dry_run is False and verify.allow_git_network is False


# --------------------------------------------------------------------------- R2 --


def _finish_rebase(pr, state):
    """Resolve every conflict the way the resolver would, then replay the rest."""
    (pr / "kernel.py").write_text(
        "def gemm(a, b, *, stream=None):\n    return fused(a, b)\n"
    )
    (pr / "cfg.py").write_text('VALUE = 5\nTOGGLE = "VLLM_DISABLE_FOO"\n')
    _git(pr, "add", "-A")
    subprocess.run(
        ["git", "-C", str(pr), "-c", "core.editor=true", "rebase", "--continue"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = _git(pr, "rev-parse", "HEAD").strip()
    state["base_commit"] = _git(pr, "rev-parse", "upstream/main").strip()
    state["head_sha"] = head
    return head


def test_drift_separates_pr_files_from_upstreams(upstream_and_pr):
    """diff(old_base, HEAD) would count upstream's own files as ours."""
    pr, state, shas = upstream_and_pr
    _start_rebase(pr, state)
    head = _finish_rebase(pr, state)
    drift = rebase.drift(state)
    assert drift["head"] == head
    assert drift["target_sha"] == shas["target"]
    assert drift["old_base"] == shas["base"]
    assert drift["pr_files"] == ["cfg.py", "kernel.py"]
    assert drift["drift_files"] == ["cfg.py", "kernel.py"]
    assert drift["upstream_only_files"] == ["service.py"]


def test_drift_before_any_rebase_has_no_pre_head(upstream_and_pr):
    pr, state, shas = upstream_and_pr
    drift = rebase.drift(state)  # pre_rebase_head is empty here
    assert drift["old_base"] == shas["base"]
    assert drift["upstream_only_files"] == ["service.py"]


def test_the_repair_may_not_edit_files_upstream_owns(upstream_and_pr, tmp_path):
    """The one rebase error no gate would ever catch is reverted automatically."""
    from llm.auto_code_gen.pr_flow import run_rebase_runtime_iters as fixver

    pr, state, _ = upstream_and_pr
    _start_rebase(pr, state)
    _finish_rebase(pr, state)
    drift = rebase.drift(state)
    iter_dir = str(tmp_path / "V1")
    assert "service.py" in drift["upstream_only_files"]

    (pr / "service.py").write_text("UPSTREAM_CONSTANT = 0  # 'fixed' by the agent\n")
    (pr / "cfg.py").write_text('VALUE = 5\nTOGGLE = "X"\nNEW = 1\n')  # legit edit
    restored = fixver.restore_upstream_owned_files(str(pr), state, drift, iter_dir)

    assert restored == ["service.py"]
    assert (pr / "service.py").read_text() == "UPSTREAM_CONSTANT = 42\n"
    assert "NEW = 1" in (pr / "cfg.py").read_text()  # legitimate edit survives
    assert (
        tmp_path / "V1" / "reverted_upstream_owned.txt"
    ).read_text().strip() == "service.py"
    # a second pass has nothing left to do
    assert fixver.restore_upstream_owned_files(str(pr), state, drift, iter_dir) == []


def test_syntax_problems_catches_broken_repairs_and_markers(upstream_and_pr):
    from llm.auto_code_gen.pr_flow import run_rebase_runtime_iters as fixver

    pr, state, _ = upstream_and_pr
    _start_rebase(pr, state)
    _finish_rebase(pr, state)
    assert fixver.syntax_problems(str(pr), ["cfg.py", "kernel.py"]) == []

    (pr / "kernel.py").write_text("def gemm(a, b):\n    return fused(\n")
    problems = fixver.syntax_problems(str(pr), ["kernel.py"])
    assert len(problems) == 1 and problems[0].startswith("kernel.py:")

    (pr / "kernel.py").write_text(
        "def gemm(a, b):\n<<<<<<< HEAD\n    return 1\n"
        "=======\n    return 2\n>>>>>>> 1234\n"
    )
    problems = fixver.syntax_problems(str(pr), ["kernel.py"])
    assert any("conflict markers" in problem for problem in problems)


def test_rebase_evidence_is_written_for_the_fix_step(upstream_and_pr, tmp_path):
    from llm.auto_code_gen.pr_flow import run_rebase_runtime_iters as fixver

    pr, state, shas = upstream_and_pr
    _start_rebase(pr, state)
    rebase.record(state["output_dir"], str(pr), state, 1)
    _finish_rebase(pr, state)
    drift = rebase.drift(state)
    iter_dir = str(tmp_path / "V1")
    stage_dir = str(tmp_path / "pr_p3" / "rebase" / "R1")

    report, drift_patch = fixver._write_rebase_evidence(
        iter_dir, str(pr), state, stage_dir, drift
    )
    with open(report) as _rf:
        report_text = _rf.read()
    assert "new base" in report_text and shas["target"][:12] in report_text
    assert "old base" in report_text
    assert "must stay identical to the new base" in report_text
    assert "service.py" in report_text  # upstream-owned, listed
    with open(drift_patch) as _df:
        drift_text = _df.read()
    assert "VALUE = 5" in drift_text and "service.py" not in drift_patch


def test_repair_prompt_holds_the_rebase_rules(tmp_path):
    from llm.auto_code_gen.pr_flow.prompts import (
        REBASE_FIX_REPORT_FILE,
        gen_RebaseRuntimeFixPrompt,
    )

    text = gen_RebaseRuntimeFixPrompt(
        context="TRACE",
        repo_dir="/repo",
        output_dir="/out",
        iteration_dir="/out/runtime/V4",
        iteration=4,
        pr_number=11,
        pr_url="u",
        rebase_report="/V4/rebase_report.txt",
        drift_file="/V4/upstream_drift.patch",
        conflicts_file="/R1/CONFLICTS.md",
        resolution_file="/R1/rebase_resolution.md",
        pr_description_file="/pr_desc.txt",
        prev_summary_file="/V3/summary.txt",
        prev_verdict_file="/V3/verify_verdict.json",
        gate_report="GATES",
        feature_toggle="VLLM_DISABLE_X",
        allowed_files=["a.py"],
    ).prompt()
    assert REBASE_FIX_REPORT_FILE in text
    assert "byte-identical" in text and "VLLM_DISABLE_X" in text
    assert "STATUS: resolved | unresolved" in text
    assert "followup_commit_msg.txt" in text
    assert "/R1/CONFLICTS.md" in text and "GATES" in text
    assert "Do NOT run any git command that writes" in text
    assert "add, commit, amend, checkout, reset, stash, rebase, push, tag" in text
    assert "No gh, no curl, no pip install" in text


def test_clean_rebase_still_reports_no_conflict_files(upstream_and_pr):
    """A PR that does not overlap replays cleanly, and drift still classifies."""
    pr, state, shas = upstream_and_pr
    _git(pr, "reset", "-q", "--hard", shas["base"])
    (pr / "fast_path.py").write_text("def fused(a, b):\n    return a + b\n")
    _git(pr, "add", "-A")
    _git(pr, "commit", "-qm", "perf: add fused path module")
    assert _start_rebase(pr, state) is False  # clean replay, never paused
    drift = rebase.drift(state)
    assert drift["pr_files"] == ["fast_path.py"]
    assert drift["drift_files"] == []
    assert sorted(drift["upstream_only_files"]) == ["cfg.py", "kernel.py", "service.py"]


# ----------------------------------------------------- measured content vs HEAD --


def _passing_verdict(out_dir, pr_head):
    from llm.auto_code_gen.pr_flow import gates as _gates

    closure = {
        "pass": True,
        "total": 1,
        "fixed": 1,
        "explained": 0,
        "unresolved": 0,
        "unresolved_ids": [],
        "reason": "",
    }
    verdict = _gates.build_verdict(
        1, 1, pr_head, {"pass": True}, {"pass": True}, closure, False
    )
    directory = os.path.join(str(out_dir), "runtime", "V1")
    os.makedirs(directory, exist_ok=True)
    _gates.write_verdict(os.path.join(directory, _gates.VERDICT_FILE), verdict)
    return directory


def test_a_committed_fix_still_matches_its_measurement(tmp_path):
    """The harness measures the worktree; the human commits it afterwards.

    That produces a different commit with identical bytes, so step 4 must accept
    it -- otherwise every review round dead-ends at the push.
    """
    from llm.auto_code_gen.pr_flow import gates as _gates

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "perf/x")
    _git(repo, "config", "user.email", "d@e.f")
    _git(repo, "config", "user.name", "D")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD").strip()

    (repo / "a.py").write_text("x = 2\n")  # the agent's repair
    out = tmp_path / "out"
    iter_dir = _passing_verdict(out, base)
    _gates.write_measured_content(iter_dir, base, base, ["a.py"], str(repo))
    state = {"repo_dir": str(repo), "output_dir": str(out), "local_branch": "perf/x"}
    assert "verdict PASS" in checks.check_verdict(state)

    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fix: the measured change")
    outcome = checks.check_verdict(state)
    assert "carries exactly the measured content" in outcome

    (repo / "a.py").write_text("x = 999\n")  # changed after measuring
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "unrelated edit")
    _rejected(
        checks.check_verdict,
        state,
        code=checks.EXIT_VERDICT,
        match="content differs from what was measured",
    )


def test_no_fingerprint_means_the_strict_rule_still_applies(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "perf/x")
    _git(repo, "config", "user.email", "d@e.f")
    _git(repo, "config", "user.name", "D")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    head = _git(repo, "rev-parse", "HEAD").strip()

    out = tmp_path / "out"
    _passing_verdict(out, head)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "extra commit")
    state = {"repo_dir": str(repo), "output_dir": str(out), "local_branch": "perf/x"}
    _rejected(
        checks.check_verdict,
        state,
        code=checks.EXIT_VERDICT,
        match="nothing recorded what was measured",
    )


def test_fingerprints_distinguish_content(tmp_path):
    from llm.auto_code_gen.pr_flow import gates as _gates

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("same\n")
    (repo / "b.py").write_text("gone\n")
    worktree = _gates.worktree_fingerprint(str(repo), ["a.py", "b.py"])
    assert worktree == _gates.worktree_fingerprint(str(repo), ["b.py", "a.py"])
    assert worktree["a.py"] != worktree["b.py"]
    assert _gates.worktree_fingerprint(str(repo), ["nope.py"])["nope.py"] == "MISSING"
    (repo / "a.py").write_text("changed\n")
    assert _gates.worktree_fingerprint(str(repo), ["a.py"])["a.py"] != worktree["a.py"]
