# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_bug_fix.bug_fix_prompts — prompt class rendering."""

from auto_bug_fix.bug_fix_prompts import (
    BuildErrorRecoveryPrompt,
    NarrowResolutionAgentPrompt,
    PostResolutionCheckPrompt,
    SemanticRederivationAgentPrompt,
    SemanticTriagePrompt,
    TestPortAgentPrompt,
    TriageBlindReviewPrompt,
)


def test_narrow_resolution_prompt_lists_files():
    conflicted = ["lib/url.c", "lib/parser.c"]
    prompt = NarrowResolutionAgentPrompt(
        context="<context>test</context>",
        fix_diff="--- a/lib/url.c\n+++ b/lib/url.c",
        conflicted_files=conflicted,
        allowed_modules=["lib/"],
        target_branch="origin/release-1.0",
        baseline_ref="abc123deadbeef",
    )
    text = prompt.prompt()
    for f in conflicted:
        assert f in text, f"{f} should appear in the prompt output"
    assert "Three-way discipline" in text
    assert "REQUIRED for the fix" in text
    # Soft-convention adaptation must not be encouraged anywhere.
    assert "do not rename parameters or variables to match" in text
    # Adaptation section must use baseline_ref for grep commands, not bare target_branch.
    assert "git grep <symbol> abc123deadbeef" in text
    assert "git grep <symbol> origin/release-1.0" not in text
    assert "origin/release-1.0" in text  # still mentioned for readability


def test_narrow_resolution_prompt_followup_section_optional():
    kwargs = dict(
        context="<context>test</context>",
        fix_diff="diff",
        conflicted_files=["lib/url.c"],
        allowed_modules=["lib/"],
        target_branch="origin/release-1.0",
        baseline_ref="baseline_sha_123",
    )
    without = NarrowResolutionAgentPrompt(**kwargs).prompt()
    assert "follow-up fixes" not in without
    assert "keep the upstream form" in without  # adaptation rule 7
    assert "git log --all" in without  # ref-scope guard (rule 9)
    # Grep commands use baseline_ref
    assert "baseline_sha_123" in without

    with_followups = NarrowResolutionAgentPrompt(
        **kwargs,
        followup_fixes="  - abc1234 fix the fix",
    ).prompt()
    assert "follow-up fixes" in with_followups
    assert "abc1234 fix the fix" in with_followups


def test_narrow_resolution_prompt_never_sanctions_abort():
    """The textual mode must not invite the agent to discard the cherry-pick."""
    text = NarrowResolutionAgentPrompt(
        context="<context>test</context>",
        fix_diff="diff",
        conflicted_files=["lib/url.c"],
        allowed_modules=["lib/"],
        target_branch="origin/release-1.0",
        baseline_ref="abc123",
    ).prompt()
    assert "cherry-pick --abort" not in text


def test_semantic_rederivation_prompt_contract():
    kwargs = dict(
        context="<context>test</context>",
        fix_diff="--- a/engine.c\n+++ b/engine.c",
        conflicted_files=["engine.c", "engine.h"],
        allowed_modules=["engine.c"],
        target_branch="origin/eol-branch",
        baseline_ref="abc123deadbeef",
    )
    text = SemanticRederivationAgentPrompt(**kwargs).prompt()
    # The mode contract: behavior over text, with an explicit escape hatch.
    assert "Backport Re-derivation Agent" in text
    assert "SECURITY-RELEVANT BEHAVIOR" in text
    assert "cherry-pick --abort" in text
    assert "Never weaken a test" in text
    # Conflicted files and priority files are listed.
    assert "engine.h" in text
    assert "engine.c" in text
    # No triage assessment and no follow-ups: sections omitted.
    assert "Semantic triage assessment" not in text
    assert "follow-up fixes" not in text

    with_triage = SemanticRederivationAgentPrompt(
        **kwargs,
        triage_assessment='{"adaptation_strategy": "semantic_rederivation"}',
        followup_fixes="  - abc1234 fix the fix",
    ).prompt()
    assert "Semantic triage assessment" in with_triage
    assert "semantic_rederivation" in with_triage
    assert "follow-up fixes" in with_triage


def test_semantic_triage_prompt_offers_adaptation_strategy():
    """The triage schema must ask for the strategy so Phase 3a can select on it."""
    text = SemanticTriagePrompt(
        context="<context>test</context>",
        fix_diff="diff",
        source_branch="main",
        target_branch="origin/eol-branch",
        seed_files=[],
    ).prompt()
    assert '"adaptation_strategy"' in text
    assert "conflict_resolution | semantic_rederivation" in text
    assert "Decide the adaptation strategy" in text


def test_semantic_triage_prompt_preserves_working_method():
    """Triage must budget exploration and keep notes that survive compaction.

    A mega-fix triage session otherwise surveys symbols one by one, hits the
    context limit, compacts, forgets, and restarts the survey forever.
    """
    text = SemanticTriagePrompt(
        context="<context>test</context>",
        fix_diff="diff",
        source_branch="main",
        target_branch="origin/eol-branch",
        seed_files=[],
        notes_file="runs/semantic_triage_notes.txt",
    ).prompt()
    assert "Budget your exploration" in text
    assert "Persist findings as you go" in text
    assert "runs/semantic_triage_notes.txt" in text
    # The default notes path renders when none is passed.
    default = SemanticTriagePrompt(
        context="<context>test</context>",
        fix_diff="diff",
        source_branch="main",
        target_branch="origin/eol-branch",
        seed_files=[],
    ).prompt()
    assert "semantic_triage_notes.txt" in default


def test_narrow_resolution_prompt_baseline_ref_fallback():
    """When baseline_ref is empty, adaptation section uses target_branch."""
    prompt = NarrowResolutionAgentPrompt(
        context="<context>test</context>",
        fix_diff="diff",
        conflicted_files=["lib/url.c"],
        allowed_modules=["lib/"],
        target_branch="origin/release-1.0",
    )
    text = prompt.prompt()
    assert "git grep <symbol> origin/release-1.0" in text


def test_post_resolution_prompt_evidence_section_optional():
    kwargs = dict(
        context="<context>test</context>",
        fix_diff="diff",
        current_diff="applied diff",
        target_branch="origin/release-1.0",
        baseline_ref="def456baseline",
    )
    without = PostResolutionCheckPrompt(**kwargs).prompt()
    assert "Deterministic evidence" not in without
    assert "Decision rule" in without
    assert "Default to the upstream form" in without
    assert "three-way discipline" in without.lower()
    assert "dead code" in without
    # The soft-convention guidance that caused over-adaptation is gone.
    assert "Match file-local conventions" not in without
    assert "Never rename parameters" in without
    # The declaration-style exception covers layout only — never names/types.
    assert "Parameter names and types are CONTENT" in without
    # Grep commands use baseline_ref, not bare target_branch.
    assert "git grep <symbol> def456baseline" in without
    assert "git grep <symbol> origin/release-1.0" not in without
    assert "git show def456baseline:<file>" in without
    assert "origin/release-1.0" in without  # still mentioned for readability

    with_evidence = PostResolutionCheckPrompt(
        **kwargs,
        evidence_report="- `sample_token`: NOT on target",
    ).prompt()
    assert "Deterministic evidence" in with_evidence
    assert "sample_token" in with_evidence


def test_post_resolution_prompt_baseline_ref_fallback():
    """When baseline_ref is empty, template uses target_branch."""
    prompt = PostResolutionCheckPrompt(
        context="<context>test</context>",
        fix_diff="diff",
        current_diff="applied diff",
        target_branch="origin/release-1.0",
    )
    text = prompt.prompt()
    assert "git grep <symbol> origin/release-1.0" in text


def test_build_error_recovery_prompt_shows_iteration():
    prompt = BuildErrorRecoveryPrompt(
        context="<context>test</context>",
        fix_diff="some diff",
        current_diff="current diff",
        build_errors="error: undefined reference",
        test_errors="FAIL: test_foo",
        allowed_modules=["lib/"],
        allowed_seed=["lib/url.c"],
        target_branch="origin/release-1.0",
        build_command="make",
        test_command="make test",
        iteration=2,
        max_retries=3,
    )
    text = prompt.prompt()
    assert "2/3" in text
    assert "undefined reference" in text


def test_semantic_triage_prompt_contains_files():
    prompt = SemanticTriagePrompt(
        context="<context>test</context>",
        fix_diff="10 files changed",
        source_branch="origin/main",
        target_branch="origin/release-1.0",
        seed_files=["pkg/foo.go", "pkg/bar.go"],
    )
    text = prompt.prompt()
    assert "pkg/foo.go" in text
    assert "origin/release-1.0" in text
    assert "origin/main" in text


def test_test_port_prompt_lists_test_files():
    prompt = TestPortAgentPrompt(
        context="<context>test</context>",
        source_fix_commit="abc123",
        target_branch="origin/release-1.0",
        test_files=["pkg/foo_test.go", "pkg/bar_test.go"],
        output_manifest_file="/tmp/manifest.txt",
        diff_command="git show abc123 --",
    )
    text = prompt.prompt()
    assert "pkg/foo_test.go" in text
    assert "abc123" in text
    assert "Do NOT modify any production code" in text


def test_test_port_prompt_uses_diff_command_for_merges():
    prompt = TestPortAgentPrompt(
        context="<context>test</context>",
        source_fix_commit="merge_abc",
        target_branch="origin/release-1.0",
        test_files=["pkg/foo_test.go"],
        output_manifest_file="/tmp/manifest.txt",
        diff_command="git diff merge_abc^1..merge_abc --",
    )
    text = prompt.prompt()
    assert "git diff merge_abc^1..merge_abc --" in text
    assert "git show merge_abc --" not in text


def test_triage_blind_review_prompt_baseline_ref():
    prompt = TriageBlindReviewPrompt(
        context="<context>test</context>",
        fix_diff="upstream diff",
        current_diff="applied diff",
        target_branch="origin/release-1.0",
        baseline_ref="blind_baseline_sha",
    )
    text = prompt.prompt()
    assert "git grep '<symbol>' blind_baseline_sha" in text
    assert "git grep '<symbol>' origin/release-1.0" not in text
    assert "origin/release-1.0" in text  # mentioned for readability
    assert "Independent Review Agent" in text


def test_triage_blind_review_prompt_baseline_ref_fallback():
    """When baseline_ref is empty, uses target_branch."""
    prompt = TriageBlindReviewPrompt(
        context="<context>test</context>",
        fix_diff="upstream diff",
        current_diff="applied diff",
        target_branch="origin/release-1.0",
    )
    text = prompt.prompt()
    assert "git grep '<symbol>' origin/release-1.0" in text


def test_resolution_prompts_carry_cross_session_notes():
    """Both Phase 3a modes must read and append shared resolution notes.

    Chunk sessions (and compactions within one session) lose conversation
    state; the notes file is the durable cross-session record of adaptation
    decisions. Off when no notes path is given (backwards-compatible).
    """
    kwargs = dict(
        context="<context>test</context>",
        fix_diff="diff",
        conflicted_files=["engine.c"],
        allowed_modules=["engine.c"],
        target_branch="origin/eol-branch",
        baseline_ref="abc123",
    )
    narrow = NarrowResolutionAgentPrompt(**kwargs).prompt()
    semantic = SemanticRederivationAgentPrompt(**kwargs).prompt()
    assert "Resolution notes" not in narrow
    assert "Resolution notes" not in semantic

    narrow = NarrowResolutionAgentPrompt(
        **kwargs, notes_file="runs/resolution_notes.txt"
    ).prompt()
    assert "Resolution notes (cross-session state)" in narrow
    assert "Read `runs/resolution_notes.txt` FIRST" in narrow
    assert "APPEND your own decisions" in narrow

    semantic = SemanticRederivationAgentPrompt(
        **kwargs, notes_file="runs/resolution_notes.txt"
    ).prompt()
    assert "Resolution notes (cross-session state)" in semantic
    assert "Read `runs/resolution_notes.txt` FIRST" in semantic
    assert "APPEND your own decisions" in semantic


def test_test_port_prompt_requires_harness_adaptation():
    """Ported tests must run clean on the target's test harness.

    An --EXTENSIONS-- section that double-loads an already-built extension
    pollutes every run with startup warnings; a RED-triage agent reading
    that output cannot establish RED or GREEN and the whole gate chain
    degrades. The port agent must adapt such sections away.
    """
    from auto_bug_fix.bug_fix_prompts import TestPortAgentPrompt

    text = TestPortAgentPrompt(
        context="<context>test</context>",
        source_fix_commit="abc123",
        target_branch="origin/eol-branch",
        test_files=["ext/foo/tests/bar.phpt"],
        output_manifest_file="runs/manifest.txt",
        diff_command="git diff abc123 -- ext/foo/tests/bar.phpt",
    ).prompt()
    assert "Adapt the tests to the TARGET's test harness" in text
    assert "double-loads an already-built extension" in text


def test_red_triage_prompt_requires_in_place_replacement():
    """The RED gate re-runs the case's test command verbatim.

    A replacement test written to a new file the command does not reference
    can never turn the gate RED, however correct it is — the 8.0 run
    escalated and later stopped honestly on exactly this.
    """
    from auto_bug_fix.bug_fix_prompts import RedTriagePrompt

    text = RedTriagePrompt(
        context="<context>test</context>",
        bug_description="b",
        test_output="PASSED",
        ported_test_files=["ext/zend_test/tests/observer_error_01.phpt"],
        triage_assessment="{}",
    ).prompt()
    assert "REWRITE the test files the test command already names, in place" in text
    assert "a brand-new test file the command does not reference" in text
