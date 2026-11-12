# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for cherry_pick_and_resolve — mechanical + finalize paths."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.claude_utils import BugFixClaudeConfig
from auto_bug_fix.dossier import Dossier
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    cherry_pick_and_resolve,
    port_prerequisite_chain,
)
from auto_bug_fix.tracker import Tracker

BASE = """\
void foo(void)
{
    int a = 1;
    int b = 2;
    int c = 3;
    int d = 4;
    return;
}
"""

FIXED = BASE.replace(
    "    int c = 3;",
    "    int c = 30; /* fixed */\n\t\tint extra_verification_counter = compute_checksum(a, b);",
)

REINDENTED = BASE.replace("    int", "\tint").replace("    return;", "\treturn;")


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd, ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _build_repo(tmp_path, target_content):
    """main: base -> fix; target: diverged; detached checkout of target."""
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")

    (repo / "base.c").write_text(BASE)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base_sha = _sha(rp)

    (repo / "base.c").write_text(FIXED)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix c")
    fix_sha = _sha(rp)

    _git(rp, "branch", "target", base_sha)
    _git(rp, "checkout", "-q", "target")
    (repo / "base.c").write_text(target_content)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "target divergence")
    # The pipeline runs on a detached worktree of the target branch.
    _git(rp, "checkout", "-q", "--detach", "target")
    return rp, fix_sha, repo / "base.c"


def _make_config(rp, fix_sha):
    return BugFixConfig(
        repo_path=rp,
        build_dir=rp,
        source_branch="main",
        target_branch="target",
        source_fix_commit=fix_sha,
        bug_description="test fix",
        issue_id="TEST-1",
    )


def _make_claude_config(rp):
    return BugFixClaudeConfig(
        model="test-model",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=rp,
    )


def _make_state():
    state = PipelineState()
    state.dossier = Dossier(
        issue_id="TEST-1",
        source_branch="main",
        target_branch="target",
        fix_commit="x",
        bug_description="test",
    )
    return state


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_mechanical_path_resolves_without_llm(mock_claude, tmp_path):
    rp, fix_sha, path = _build_repo(tmp_path, REINDENTED)
    config = _make_config(rp, fix_sha)
    state = _make_state()

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )

    assert result == "resolved"
    content = path.read_text()
    assert "int c = 30; /* fixed */" in content
    assert "<<<<<<<" not in content
    # A commit concluded the cherry-pick.
    message = subprocess.run(
        ["git", "log", "-1", "--format=%B"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout
    assert "Conflict summary" in message
    assert "Mechanically re-applied" in message
    # No resolution prompt was needed; only the 3a.1 check ran.
    prompts = [call.args[1][0] for call in mock_claude.call_args_list]
    assert not any("Narrow Resolution Agent" in p for p in prompts)
    assert any("Post-Resolution Verification Agent" in p for p in prompts)
    assert any("mechanical" in s.strategy for s in state.dossier.strategies)


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_llm_path_gets_snapback_and_amend(mock_claude, tmp_path):
    # Target rewrote the conflict line's content — mechanical rungs must fail.
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    state = _make_state()

    resolved_with_drift = diverged.replace(
        "    int c = 99;",
        # The "LLM" keeps the target line but retypes the new upstream line
        # with spaces instead of the upstream's tabs.
        "    int c = 99;\n        int extra_verification_counter = compute_checksum(a, b);",
    )

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            path.write_text(resolved_with_drift)
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )

    assert result == "resolved"
    committed = subprocess.run(
        ["git", "show", "HEAD:base.c"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout
    assert "int c = 99;" in committed
    # Snap-back restored the upstream bytes (tabs) for the drifted line and
    # the amend folded it into the port commit.
    assert "\t\tint extra_verification_counter = compute_checksum(a, b);" in committed
    assert "        int extra_verification_counter" not in committed
    prompts = [call.args[1][0] for call in mock_claude.call_args_list]
    assert any("Narrow Resolution Agent" in p for p in prompts)
    assert any("Post-Resolution Verification Agent" in p for p in prompts)


PREREQ_BASE = """\
/* helpers */
void old_helper_fn(int a) {}
void caller1(void) { old_helper_fn(1); }
void caller2(void) { old_helper_fn(2); }
void caller3(void) { old_helper_fn(3); }
/* padding one */
/* padding two */
/* padding three */
int version = 1;
"""


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_evidence_references_ported_prerequisites_not_bare_target(
    mock_claude, tmp_path
):
    """A prerequisite port renamed a symbol; the fix uses the new name.

    Evidence must reference base_head (target + ported prereq), not the bare
    target branch — otherwise the prereq's rename is mined backwards and the
    fix's correct call is mechanically reverted to the old symbol.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")

    (repo / "base.c").write_text(PREREQ_BASE)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base_sha = _sha(rp)

    # Prerequisite: rename old_helper_fn -> new_helper_fn (4 occurrences).
    (repo / "base.c").write_text(PREREQ_BASE.replace("old_helper_fn", "new_helper_fn"))
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "prereq: rename helper")
    prereq_sha = _sha(rp)

    # Fix: new caller using the new name, plus a version bump that conflicts.
    fixed = (
        (repo / "base.c")
        .read_text()
        .replace(
            "void caller3(void) { new_helper_fn(3); }",
            "void caller3(void) { new_helper_fn(3); }\nvoid caller4(void) { new_helper_fn(4); }",
        )
        .replace("int version = 1;", "int version = 10;")
    )
    (repo / "base.c").write_text(fixed)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix: add caller4")
    fix_sha = _sha(rp)

    # Target diverged only in the version line.
    _git(rp, "branch", "target", base_sha)
    _git(rp, "checkout", "-q", "--detach", "target")
    (repo / "base.c").write_text(
        PREREQ_BASE.replace("int version = 1;", "int version = 2;")
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "target divergence")
    # Update the target ref to include the divergence, then simulate the
    # pipeline's prerequisite port on top of it (detached, like a worktree).
    _git(rp, "branch", "-f", "target", "HEAD")
    _git(rp, "cherry-pick", prereq_sha)

    config = _make_config(rp, fix_sha)
    state = _make_state()

    resolved = (
        (repo / "base.c")
        .read_text()
        .replace(
            "void caller3(void) { new_helper_fn(3); }",
            "void caller3(void) { new_helper_fn(3); }\nvoid caller4(void) { new_helper_fn(4); }",
        )
        .replace("int version = 2;", "int version = 10;")
    )

    captured_prompts: list[str] = []

    async def fake_claude_run(claude_config, prompts, tracker=None):
        captured_prompts.append(prompts[0])
        if "Narrow Resolution Agent" in prompts[0]:
            (repo / "base.c").write_text(resolved)
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    # Capture base_head (HEAD before cherry-pick = target + prereq)
    base_head = _sha(rp)

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )
    assert result == "resolved"
    committed = subprocess.run(
        ["git", "show", "HEAD:base.c"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout
    assert "new_helper_fn(4);" in committed
    assert "old_helper_fn" not in committed

    # The resolution prompt must use base_head (not bare "target") for grep.
    resolution_prompts = [p for p in captured_prompts if "Narrow Resolution Agent" in p]
    assert resolution_prompts, "expected at least one resolution prompt"
    for rp_text in resolution_prompts:
        assert (
            base_head in rp_text
        ), "resolution prompt must use base_head for grep commands"
        assert f"git grep <symbol> {base_head}" in rp_text

    # The verification/blind-review prompts must also use base_head.
    verifier_prompts = [
        p
        for p in captured_prompts
        if "Post-Resolution Verification Agent" in p or "Independent Review Agent" in p
    ]
    for vp_text in verifier_prompts:
        assert (
            base_head in vp_text
        ), "verifier prompt must use base_head, not bare target branch"


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_committed_conflict_markers_are_flagged(mock_claude, tmp_path):
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    state = _make_state()

    broken = diverged.replace(
        "    int c = 99;",
        "<<<<<<< HEAD\n    int c = 99;\n=======\n    int c = 30; /* fixed */\n>>>>>>> theirs",
    )

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            path.write_text(broken)
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )
    assert result == "resolved"
    assert any("Conflict markers" in i.label for i in state.dossier.items)
    check_prompts = [
        c.args[1][0]
        for c in mock_claude.call_args_list
        if "Post-Resolution Verification" in c.args[1][0]
    ]
    assert check_prompts and "CRITICAL: conflict markers" in check_prompts[0]


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_three_way_violations_reach_the_verifier(mock_claude, tmp_path):
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    state = _make_state()

    # The "LLM" resolves the conflict but also modernizes an unrelated target
    # line the fix never touches — a three-way discipline violation.
    resolved = diverged.replace(
        "    int c = 99;",
        "    int c = 30; /* fixed */\n\t\tint extra_verification_counter = compute_checksum(a, b);",
    ).replace("    int a = 1;", "    int a = 100; /* modernized for no reason */")

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            path.write_text(resolved)
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )
    assert result == "resolved"
    check_prompts = [
        c.args[1][0]
        for c in mock_claude.call_args_list
        if "Post-Resolution Verification" in c.args[1][0]
    ]
    assert check_prompts
    report = check_prompts[0]
    assert "Three-way discipline check" in report
    assert "int a = 1;" in report  # the removed target line is called out
    assert "not remove" in report


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_verifier_edits_survive_post_snapback(mock_claude, tmp_path):
    """3a.1's deliberate corrections must not be snapped back to source bytes."""
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    state = _make_state()

    resolved = diverged.replace(
        "    int c = 99;",
        "    int c = 99;\n\t\tint extra_verification_counter = compute_checksum(a, b);",
    )
    # The verifier deliberately re-indents that line to a novel form that
    # normalize-matches the upstream source — snap-back must leave it alone.
    verifier_form = "      int extra_verification_counter = compute_checksum(a, b);"

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            path.write_text(resolved)
            _git(rp, "add", "base.c")
        elif "Post-Resolution Verification" in prompts[0]:
            path.write_text(
                path.read_text().replace(
                    "\t\tint extra_verification_counter = compute_checksum(a, b);",
                    verifier_form,
                )
            )
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )
    assert result == "resolved"
    committed = subprocess.run(
        ["git", "show", "HEAD:base.c"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout
    assert verifier_form in committed


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_flags_disable_mechanical_and_snapback(mock_claude, tmp_path):
    rp, fix_sha, path = _build_repo(tmp_path, REINDENTED)
    config = _make_config(rp, fix_sha)
    config.mechanical_resolution = False
    config.snapback = False
    config.evidence_checks = False
    state = _make_state()

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            path.write_text(
                REINDENTED.replace("\tint c = 3;", "\tint c = 30; /* fixed */")
            )
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )
    assert result == "resolved"
    prompts = [call.args[1][0] for call in mock_claude.call_args_list]
    assert any("Narrow Resolution Agent" in p for p in prompts)
    assert not any("mechanical" in s.strategy for s in state.dossier.strategies)


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_post_resolution_check_false_skips_3a1(mock_claude, tmp_path):
    rp, fix_sha, path = _build_repo(tmp_path, REINDENTED)
    config = _make_config(rp, fix_sha)
    config.post_resolution_check = False
    state = _make_state()

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            path.write_text(
                REINDENTED.replace("\tint c = 3;", "\tint c = 30; /* fixed */")
            )
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )
    assert result == "resolved"
    prompts = [call.args[1][0] for call in mock_claude.call_args_list]
    assert not any("Post-Resolution Verification Agent" in p for p in prompts)


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_blind_review_false_skips_3a2(mock_claude, tmp_path):
    rp, fix_sha, path = _build_repo(tmp_path, REINDENTED)
    config = _make_config(rp, fix_sha)
    config.blind_review = False
    state = _make_state()

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            path.write_text(
                REINDENTED.replace("\tint c = 3;", "\tint c = 30; /* fixed */")
            )
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )
    assert result == "resolved"
    prompts = [call.args[1][0] for call in mock_claude.call_args_list]
    assert not any("Independent Review Agent" in p for p in prompts)


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_evidence_false_blind_review_true_still_runs_3a2(mock_claude, tmp_path):
    rp, fix_sha, path = _build_repo(tmp_path, REINDENTED)
    config = _make_config(rp, fix_sha)
    config.evidence_checks = False
    config.blind_review = True
    state = _make_state()

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            path.write_text(
                REINDENTED.replace("\tint c = 3;", "\tint c = 30; /* fixed */")
            )
            _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        None,
        commit=fix_sha,
        label="test port",
    )
    assert result == "resolved"
    prompts = [call.args[1][0] for call in mock_claude.call_args_list]
    assert any("Independent Review Agent" in p for p in prompts)


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_resolution_uses_one_prompt_and_keeps_blind_review(mock_claude, tmp_path):
    """Fused mode resolves in one session; 3a.1 is skipped, 3a.2 still runs."""
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    config.fused_resolution = True
    config.build_command = "true"
    config.test_command = "true"
    state = _make_state()
    state.triage_assessment = {
        "regeneration_commands": [],
        "files_to_skip": [],
        "lint_commands": ["hack/verify-gofmt.sh"],
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = Tracker(
        issue_id="TEST-1", config_dict={}, model="test-model", output_dir=str(run_dir)
    )

    seen_prompts: list[str] = []

    async def fake_claude_run(claude_config, prompts, tracker=None):
        seen_prompts.append(prompts[0])
        path.write_text(
            diverged.replace("    int c = 99;", "    int c = 30; /* fixed */")
        )
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "backport")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        tracker,
        commit=fix_sha,
        label="test port",
    )

    assert result == "resolved"
    assert len(seen_prompts) == 2  # resolution session + triage-blind review
    prompt = seen_prompts[0]
    assert "Backport Resolution Agent" in prompt
    assert "A. RESOLVE" in prompt and "F. COMMIT" in prompt
    assert "true" in prompt  # build/test commands are embedded
    assert "hack/verify-gofmt.sh" in prompt  # triage-declared checks are listed
    assert "in-place mode" in prompt  # and the prompt asks for the fixer
    assert "Stay on the detached HEAD" in prompt  # never move the target branch
    assert "fused_resolution_report.md" in prompt
    assert not any("Post-Resolution Verification Agent" in p for p in seen_prompts)
    # The independent read survives fused mode: 3a.2 runs on the clean diff.
    assert any("Independent Review Agent" in p for p in seen_prompts)
    assert state.fused_resolution_ran is True


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_resolution_falls_back_when_conflicts_remain(mock_claude, tmp_path):
    """A fused session that leaves conflicts hands the work to the old loop."""
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    config.fused_resolution = True
    state = _make_state()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = Tracker(
        issue_id="TEST-1", config_dict={}, model="test-model", output_dir=str(run_dir)
    )

    fused_calls = 0

    async def fake_claude_run(claude_config, prompts, tracker=None):
        nonlocal fused_calls
        if "Backport Resolution Agent" in prompts[0]:
            fused_calls += 1
            state.dossier.add("Fused attempt", "discarded", "test")
            return []  # deliberately resolve nothing
        path.write_text(
            diverged.replace("    int c = 99;", "    int c = 30; /* fixed */")
        )
        _git(rp, "add", "-A")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        tracker,
        commit=fix_sha,
        label="test port",
    )

    assert result == "resolved"
    assert fused_calls == 1
    assert state.fused_resolution_ran is False
    # Evidence from the discarded attempt must not claim work that was redone.
    assert not any(i.source == "Phase 2+3a fused session" for i in state.dossier.items)
    # The legacy loop had to finish the job and its reviews still ran.
    prompts = [call.args[1][0] for call in mock_claude.call_args_list]
    assert any("Narrow Resolution Agent" in p for p in prompts)
    assert any("Post-Resolution Verification Agent" in p for p in prompts)


@patch("auto_bug_fix.run_bug_fix.cherry_pick_and_resolve")
def test_prerequisite_chain_disables_fused_resolution(mock_cp, tmp_path):
    """Prerequisite ports must not inherit the main fix's fused session."""
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, _ = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    config.fused_resolution = True
    config.fused_pre_red = False
    state = _make_state()
    state.triage_assessment = {
        "prerequisites": [
            {"symbol": "s", "portable": True, "introduced_by": fix_sha},
        ]
    }
    seen: list[bool] = []

    def fake(config, *args, **kwargs):
        seen.append(config.fused_resolution)
        return "resolved"

    mock_cp.side_effect = fake
    port_prerequisite_chain(config, _make_claude_config(rp), state, None)

    assert seen == [False]
    assert config.fused_resolution is True


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_commit_scratch_is_stripped(mock_claude, tmp_path):
    """Build caches force-added by the session never reach the candidate."""
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    config.fused_resolution = True
    config.blind_review = False  # keep the test on the session's own commit
    state = _make_state()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = Tracker(
        issue_id="TEST-1", config_dict={}, model="test-model", output_dir=str(run_dir)
    )

    async def fake_claude_run(claude_config, prompts, tracker=None):
        path.write_text(
            diverged.replace("    int c = 99;", "    int c = 30; /* fixed */")
        )
        cache = Path(rp) / ".gocache" / "x"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("cache")
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "fused backport")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        tracker,
        commit=fix_sha,
        label="test port",
    )

    assert result == "resolved"
    files = subprocess.run(
        ["git", "diff", "--name-only", "HEAD^", "HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "base.c" in files
    assert not any(f.startswith(".gocache") for f in files)


def test_fused_semantic_check_accepts_a_related_result(tmp_path):
    from auto_bug_fix.run_bug_fix import _fused_resolution_unmatched

    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    base_head = _sha(rp)
    path.write_text(diverged.replace("    int c = 99;", "    int c = 30; /* fixed */"))
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "backport")

    assert (
        _fused_resolution_unmatched(
            _make_config(rp, fix_sha), _make_state(), fix_sha, base_head
        )
        is False
    )


def test_fused_semantic_check_rejects_an_unrelated_result(tmp_path):
    from auto_bug_fix.run_bug_fix import _fused_resolution_unmatched

    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    base_head = _sha(rp)
    path.write_text(diverged.replace("    int d = 4;", "    int d = 40;"))
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "unrelated rewrite")

    assert (
        _fused_resolution_unmatched(
            _make_config(rp, fix_sha), _make_state(), fix_sha, base_head
        )
        is True
    )


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_mechanical_resolution_runs_before_the_fused_session(
    mock_claude, tmp_path, monkeypatch
):
    from auto_bug_fix import run_bug_fix as rbf

    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    config.fused_resolution = True
    config.mechanical_resolution = True
    config.blind_review = False
    state = _make_state()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = Tracker(
        issue_id="TEST-1", config_dict={}, model="test-model", output_dir=str(run_dir)
    )
    events: list[str] = []

    def fake_mechanical(repo, commit, files):
        events.append("mechanical")
        return {}

    async def fake_claude_run(claude_config, prompts, tracker=None):
        events.append("claude")
        path.write_text(
            diverged.replace("    int c = 99;", "    int c = 30; /* fixed */")
        )
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "backport")
        return []

    monkeypatch.setattr(rbf, "resolve_mechanically", fake_mechanical)
    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        tracker,
        commit=fix_sha,
        label="test port",
    )

    assert result == "resolved"
    assert events[0] == "mechanical"
    assert "claude" in events


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_fused_fallback_cleans_untracked_session_files(mock_claude, tmp_path):
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    config.fused_resolution = True
    config.blind_review = False
    state = _make_state()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = Tracker(
        issue_id="TEST-1", config_dict={}, model="test-model", output_dir=str(run_dir)
    )
    calls = {"n": 0}

    async def fake_claude_run(claude_config, prompts, tracker=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # Failed fused session: new untracked file, conflicts untouched.
            (Path(rp) / "stray.txt").write_text("scratch\n")
            return []
        path.write_text(diverged.replace("    int c = 99;", "    int c = 30;"))
        _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    result = cherry_pick_and_resolve(
        config,
        _make_claude_config(rp),
        state,
        tracker,
        commit=fix_sha,
        label="test port",
    )

    assert result == "resolved"
    assert not (Path(rp) / "stray.txt").exists()


def test_fused_semantic_check_fails_open_with_warning(tmp_path, caplog, monkeypatch):
    from auto_bug_fix import run_bug_fix as rbf

    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, _ = _build_repo(tmp_path, diverged)
    monkeypatch.setattr(
        rbf, "run_range_diff", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )

    with caplog.at_level(logging.WARNING):
        result = rbf._fused_resolution_unmatched(
            _make_config(rp, fix_sha), _make_state(), fix_sha, _sha(rp)
        )

    assert result is False
    assert "Fused semantic check failed" in caplog.text
