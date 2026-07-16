"""Tests for auto_bug_fix.bug_fix_prompts — prompt class rendering."""
from auto_bug_fix.bug_fix_prompts import (
    NarrowResolutionAgentPrompt,
    PostResolutionCheckPrompt,
    BuildErrorRecoveryPrompt,
    SemanticTriagePrompt,
    TestPortAgentPrompt,
)


def test_narrow_resolution_prompt_lists_files():
    conflicted = ["lib/url.c", "lib/parser.c"]
    prompt = NarrowResolutionAgentPrompt(
        context="<context>test</context>",
        fix_diff="--- a/lib/url.c\n+++ b/lib/url.c",
        conflicted_files=conflicted,
        allowed_modules=["lib/"],
    )
    text = prompt.prompt()
    for f in conflicted:
        assert f in text, f"{f} should appear in the prompt output"
    assert "Three-way discipline" in text
    assert "REQUIRED for the fix" in text
    # Soft-convention adaptation must not be encouraged anywhere.
    assert "do not rename parameters or variables to match" in text


def test_narrow_resolution_prompt_followup_section_optional():
    kwargs = dict(
        context="<context>test</context>",
        fix_diff="diff",
        conflicted_files=["lib/url.c"],
        allowed_modules=["lib/"],
        target_branch="origin/release-1.0",
    )
    without = NarrowResolutionAgentPrompt(**kwargs).prompt()
    assert "follow-up fixes" not in without
    assert "keep the upstream form" in without  # adaptation rule 7
    assert "git log --all" in without           # ref-scope guard (rule 8)

    with_followups = NarrowResolutionAgentPrompt(
        **kwargs, followup_fixes="  - abc1234 fix the fix",
    ).prompt()
    assert "follow-up fixes" in with_followups
    assert "abc1234 fix the fix" in with_followups


def test_post_resolution_prompt_evidence_section_optional():
    kwargs = dict(
        context="<context>test</context>",
        fix_diff="diff",
        current_diff="applied diff",
        target_branch="origin/release-1.0",
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

    with_evidence = PostResolutionCheckPrompt(
        **kwargs, evidence_report="- `sample_token`: NOT on target",
    ).prompt()
    assert "Deterministic evidence" in with_evidence
    assert "sample_token" in with_evidence


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
