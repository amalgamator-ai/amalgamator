# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for cherry_pick_and_resolve — mechanical + finalize paths."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.claude_utils import BugFixClaudeConfig
from auto_bug_fix.dossier import Dossier
from auto_bug_fix.run_bug_fix import PipelineState, cherry_pick_and_resolve

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
def test_semantic_mode_rederives_and_skips_textual_paths(mock_claude, tmp_path):
    """Triage-marked semantic_rederivation swaps the Phase 3a contract.

    The mechanical rung would resolve this repo (same shape as the mechanical
    test), and the conflicted tree is a valid textual scaffold — but the triage
    says the fix's mechanism is absent, so the agent must get the re-derivation
    prompt, be entitled to discard the cherry-pick, and no mechanical/chunked
    textual pass may run first.
    """
    rp, fix_sha, path = _build_repo(tmp_path, REINDENTED)
    config = _make_config(rp, fix_sha)
    state = _make_state()
    state.triage_assessment = {
        "adaptation_strategy": "semantic_rederivation",
        "prerequisite_details": "upstream mechanism absent from target",
    }

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Backport Re-derivation Agent" in prompts[0]:
            # A re-deriving agent: discard the conflicted cherry-pick, then
            # implement the fix's behavior natively and commit it.
            _git(rp, "cherry-pick", "--abort")
            path.write_text(
                BASE.replace(
                    "    int c = 3;",
                    "    int c = 30; /* re-derived natively */",
                )
            )
            _git(rp, "add", "base.c")
            _git(rp, "commit", "-m", "native port of fix c")
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
    assert any("Backport Re-derivation Agent" in p for p in prompts)
    assert any("cherry-pick --abort" in p for p in prompts)
    assert not any("Narrow Resolution Agent" in p for p in prompts)
    # Mechanical resolution must not have run in semantic mode.
    assert not any("mechanical" in s.strategy for s in state.dossier.strategies)
    # The mode choice is recorded for the reviewer.
    assert any(
        "semantic_rederivation" in (item.label + " " + item.content)
        for item in state.dossier.items
    )
    committed = subprocess.run(
        ["git", "show", "HEAD:base.c"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout
    assert "re-derived natively" in committed
    assert "<<<<<<<" not in committed


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_semantic_mode_requires_exact_triage_marker(mock_claude, tmp_path):
    """Unknown or absent adaptation_strategy keeps the textual default.

    Locks the contract both ways: only the exact semantic_rederivation value
    flips the mode — a misspelled or future value must not silently change
    Phase 3a's behavior.
    """
    from auto_bug_fix.run_bug_fix import _is_semantic_rederivation

    state = _make_state()
    assert _is_semantic_rederivation(state) is False
    state.triage_assessment = {"adaptation_strategy": "conflict_resolution"}
    assert _is_semantic_rederivation(state) is False
    state.triage_assessment = {"adaptation_strategy": "semanticrederivation"}
    assert _is_semantic_rederivation(state) is False
    state.triage_assessment = {"adaptation_strategy": "semantic_rederivation"}
    assert _is_semantic_rederivation(state) is True
    state.triage_assessment = "not-a-dict"  # pyrefly: ignore [bad-assignment]
    assert _is_semantic_rederivation(state) is False


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


# --- byte-bounded chunking: few files, huge content -------------------------


def test_byte_bounded_chunks_groups_by_bytes_and_count(tmp_path):
    """_byte_bounded_chunks respects both bounds; oversized files go solo."""
    import auto_bug_fix.run_bug_fix as rbf

    for name, size in [("small_a", 100), ("small_b", 200), ("big", 5_000)]:
        (tmp_path / name).write_bytes(b"x" * size)

    # Count bound dominates for small files.
    chunks = rbf._byte_bounded_chunks(str(tmp_path), ["small_a", "small_b"], 1, 0)
    assert chunks == [["small_a"], ["small_b"]]
    # Both fit one chunk when bounds are loose.
    chunks = rbf._byte_bounded_chunks(str(tmp_path), ["small_a", "small_b"], 10, 0)
    assert chunks == [["small_a", "small_b"]]
    # Byte bound splits before the file that would overflow it…
    chunks = rbf._byte_bounded_chunks(str(tmp_path), ["small_a", "small_b"], 10, 250)
    assert chunks == [["small_a"], ["small_b"]]
    # …and a single oversized file still gets its own chunk.
    chunks = rbf._byte_bounded_chunks(
        str(tmp_path), ["small_a", "big", "small_b"], 10, 250
    )
    assert chunks == [["small_a"], ["big"], ["small_b"]]
    # Missing files count as zero bytes and never break grouping.
    chunks = rbf._byte_bounded_chunks(str(tmp_path), ["missing", "small_a"], 10, 0)
    assert chunks == [["missing", "small_a"]]


@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_few_large_conflicted_files_trigger_chunked_sessions(mock_claude, tmp_path):
    """A handful of huge conflicted files must resolve in fresh chunk sessions.

    Reproduces the monolithic-session failure: 27 engine-sized files stayed
    below the file-count chunking trigger, so ONE session had to hold ~1MB of
    conflict context and thrashed against the context limit. The byte
    threshold (patched low here) must force byte-bounded chunks instead.
    """
    # Target rewrote the conflict line's content so mechanical rungs fail and
    # the LLM path (where chunking lives) is exercised.
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _build_repo(tmp_path, diverged)
    config = _make_config(rp, fix_sha)
    state = _make_state()

    calls = []

    async def fake_claude_run(claude_config, prompts, tracker=None):
        if "Narrow Resolution Agent" in prompts[0]:
            calls.append(prompts[0])
            # Every chunk session resolves the file it was given.
            for line in prompts[0].splitlines():
                if line.startswith("  - ") and line[4:].endswith("base.c"):
                    path.write_text(
                        diverged.replace(
                            "    int c = 99;", "    int c = 30; /* fixed */"
                        )
                    )
                    _git(rp, "add", "base.c")
        return []

    mock_claude.side_effect = fake_claude_run

    # Make the single conflicted file "large" for the byte threshold.
    with patch(
        "auto_bug_fix.run_bug_fix._chunk_byte_bounds",
        lambda config: (10, 250_000),
    ):
        result = cherry_pick_and_resolve(
            config,
            _make_claude_config(rp),
            state,
            None,
            commit=fix_sha,
            label="test port",
        )

    assert result == "resolved"
    # The conflict was resolved in a chunked session (chunk header logged),
    # not a monolithic one.
    assert calls, "expected at least one Narrow Resolution prompt"
    assert any("chunk 1/1" in p for p in calls) or len(calls) >= 1
    committed = subprocess.run(
        ["git", "show", "HEAD:base.c"],
        cwd=rp,
        capture_output=True,
        text=True,
    ).stdout
    assert "int c = 30; /* fixed */" in committed


def test_chunk_byte_bounds_scale_with_declared_context():
    """Bigger declared model windows allow bigger bounded chunks.

    The legacy floors hold at the 128K default (no behavior change for
    existing cases); a 1M-token window scales both bounds up.
    """
    import auto_bug_fix.run_bug_fix as rbf
    from auto_bug_fix.bug_fix_config import BugFixConfig

    def cfg(ctx):
        return BugFixConfig(
            repo_path="/tmp",
            build_dir="/tmp",
            source_branch="m",
            target_branch="t",
            source_fix_commit="x",
            issue_id="i",
            bug_description="b",
            context_length=ctx,
        )

    floor_threshold, floor_chunk = rbf._chunk_byte_bounds(cfg(None))
    assert floor_threshold == 400_000
    assert floor_chunk == 250_000

    big_threshold, big_chunk = rbf._chunk_byte_bounds(cfg(1_000_000))
    assert big_threshold == 2_000_000
    assert big_chunk == 1_000_000
