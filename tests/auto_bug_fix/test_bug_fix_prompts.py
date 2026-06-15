"""Tests for auto_bug_fix.bug_fix_prompts — prompt class rendering."""
from auto_bug_fix.bug_fix_prompts import (
    NarrowResolutionAgentPrompt,
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
    )
    text = prompt.prompt()
    assert "pkg/foo_test.go" in text
    assert "abc123" in text
    assert "Do NOT modify any production code" in text
