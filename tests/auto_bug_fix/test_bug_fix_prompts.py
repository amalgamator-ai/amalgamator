import pytest
from auto_bug_fix.bug_fix_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_TestPortPrompt,
    TEST_PORT_MANIFEST_FILE,
)


class TestCreateContextStr:
    def test_contains_branch_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "gcc-14-branch" in ctx
        assert "gcc-13-branch" in ctx
        assert "abc1234" in ctx

    def test_contains_bug_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "CVE-2024-1234" in ctx

    def test_contains_build_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "make -j$(nproc)" in ctx
        assert "make check -j$(nproc)" in ctx
        assert "/path/to/gcc/build" in ctx

    def test_no_gpu_language(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "gpu_type" not in ctx
        assert "batch_size" not in ctx
        assert "precision" not in ctx

    def test_contains_cwd(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "/tmp/test_cwd" in ctx


class TestCodeTracePrompt:
    def test_contains_source_branch(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "gcc-14-branch" in p.prompt()

    def test_references_fix_commit(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "source_fix_commit" in p.prompt()

    def test_no_cuda_language(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        text = p.prompt()
        assert "cuda" not in text.lower()
        assert "prefill" not in text.lower()
        assert "decode" not in text.lower()

    def test_output_file_named_after_branch(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "gcc-14-branch" in p.output_file


class TestTestPortPrompt:
    def test_references_git_show(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert "git show" in p.prompt()

    def test_contains_commit_sha(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert "abc1234" in p.prompt()

    def test_references_manifest_file(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert TEST_PORT_MANIFEST_FILE in p.prompt()

    def test_output_manifest_is_constant(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert p.output_manifest_file == TEST_PORT_MANIFEST_FILE
