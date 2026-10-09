"""Tests for the external resolver context (hybrid synthmerge→amalgamator)."""
import os
import tempfile

import pytest

from auto_bug_fix.bug_fix_prompts import (
    NarrowResolutionAgentPrompt,
    PostResolutionCheckPrompt,
    SemanticTriagePrompt,
    TriageBlindReviewPrompt,
)
from auto_bug_fix.config_loader import ConfigError, parse_bug_fix_config
from auto_bug_fix.external_context import ExternalContext, load_external_context

VALID_MD = """# External Resolver Context

Intro commentary that is passed through verbatim.

## Metadata
- resolver: synthmerge 0.3.15
- endpoint: Turnstile DeepSeek V4 Flash
- conflicts: 26 across 11 files

## Suspect findings
Findings from the deterministic checks.

- fs/xfs/libxfs/xfs_defer.h:68: symbol 'dop_inodes' declared but has zero references
- mm/memory-failure.c:1224: symbol 'num_poisoned_pages_dec' used but never defined

## Resolved files
Prior resolver output.

### fs/xfs/libxfs/xfs_attr.c
```diff
diff --git a/fs/xfs/libxfs/xfs_attr.c b/fs/xfs/libxfs/xfs_attr.c
+hunk
```

### fs/xfs/libxfs/xfs_defer.c
```diff
diff --git a/fs/xfs/libxfs/xfs_defer.c b/fs/xfs/libxfs/xfs_defer.c
+hunk
```
"""


def _write_md(content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        return f.name


BASE_CONFIG = {
    "issue_id": "CVE-0000-0896",
    "bug_description": "test",
    "repository": {"source_path": "/tmp/repo", "workdir": "/tmp/wd"},
    "branches": {"source": "main", "target": "main"},
    "fix": {"commit": "abc123"},
    "build": {},
}


class TestLoadExternalContext:
    def test_none_path_is_inactive(self):
        ctx = load_external_context(None)
        assert not ctx.active
        assert ctx.render_triage_section() == ""
        assert ctx.render_resolution_section() == ""
        assert ctx.render_verification_section() == ""

    def test_missing_file_is_inactive_not_error(self):
        ctx = load_external_context("/nonexistent/path.md")
        assert not ctx.active

    def test_empty_file_is_inactive(self):
        path = _write_md("   \n")
        try:
            assert not load_external_context(path).active
        finally:
            os.unlink(path)

    def test_valid_file_parses(self):
        path = _write_md(VALID_MD)
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.active
        assert ctx.metadata["resolver"] == "synthmerge 0.3.15"
        assert len(ctx.findings) == 2
        assert "dop_inodes" in ctx.findings[0]
        assert set(ctx.resolved_files) == {
            "fs/xfs/libxfs/xfs_attr.c",
            "fs/xfs/libxfs/xfs_defer.c",
        }
        assert "+hunk" in ctx.resolved_files["fs/xfs/libxfs/xfs_attr.c"]

    def test_findings_only_file(self):
        path = _write_md("## Suspect findings\n\n- a.c:1: bad symbol\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.active
        assert ctx.findings == ["a.c:1: bad symbol"]
        assert ctx.resolved_files == {}

    def test_malformed_file_degrades_not_raises(self):
        path = _write_md("# nothing recognizable\njust prose\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.active  # raw text is still usable context
        assert ctx.parse_errors  # but the operator is warned

    def test_default_context_is_empty_strategy(self):
        ctx = ExternalContext()
        assert not ctx.active
        assert ctx.render_triage_section() == ""
        assert ctx.render_resolution_section() == ""
        assert ctx.render_verification_section() == ""


class TestSectionRenderers:
    def test_resolution_section_contract(self):
        path = _write_md(VALID_MD)
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        s = ctx.render_resolution_section()
        assert "verbatim" in s
        assert "dop_inodes" in s
        assert "RESOLVED FILES" in s

    def test_verification_section_never_reduces_scope(self):
        path = _write_md(VALID_MD)
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        s = ctx.render_verification_section()
        assert "Verification scope is NOT reduced" in s
        assert "dop_inodes" in s

    def test_review_section_scoped_to_findings(self):
        path = _write_md(VALID_MD)
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        s = ctx.render_review_section()
        # findings + metadata are present...
        assert "dop_inodes" in s
        assert "synthmerge 0.3.15" in s
        # ...but the trusted resolutions are withheld (independence)
        assert "RESOLVED FILES" not in s
        assert "xfs_attr.c" not in s
        # and both mandates are stated
        assert "wholesale revert" in s
        assert "dead or undefined" in s

    def test_no_findings_renders_all_clear(self):
        path = _write_md("## Metadata\n- resolver: synthmerge\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert "all deterministic checks passed" in ctx.render_resolution_section()


class TestPromptInjection:
    def _ctx(self):
        path = _write_md(VALID_MD)
        try:
            return load_external_context(path)
        finally:
            os.unlink(path)

    def test_triage_prompt_includes_section(self):
        ctx = self._ctx()
        p = SemanticTriagePrompt(
            context="CTX", fix_diff="D", source_branch="s",
            target_branch="t", seed_files=["a"],
            external_context_section=ctx.render_triage_section(),
        )
        rendered = p.prompt()
        assert "external_resolver_context" in rendered
        assert "synthmerge 0.3.15" in rendered

    def test_triage_prompt_without_section_unchanged(self):
        p = SemanticTriagePrompt(
            context="CTX", fix_diff="D", source_branch="s",
            target_branch="t", seed_files=["a"],
        )
        assert "external_resolver_context" not in p.prompt()
        assert "{external_context_section}" not in p.prompt()

    def test_resolution_prompt_includes_section(self):
        ctx = self._ctx()
        p = NarrowResolutionAgentPrompt(
            context="CTX", fix_diff="D", conflicted_files=["a"],
            allowed_modules=["m"],
            external_context_section=ctx.render_resolution_section(),
        )
        rendered = p.prompt()
        assert "external_resolver_context" in rendered
        assert "SUSPECT FINDINGS" in rendered
        assert "RESOLVED FILES" in rendered

    def test_verification_prompt_includes_section(self):
        ctx = self._ctx()
        p = PostResolutionCheckPrompt(
            context="CTX", fix_diff="D", current_diff="CD", target_branch="t",
            external_context_section=ctx.render_verification_section(),
        )
        assert "Verification scope is NOT reduced" in p.prompt()

    def test_review_prompt_includes_findings_not_resolutions(self):
        ctx = self._ctx()
        p = TriageBlindReviewPrompt(
            context="CTX", fix_diff="D", current_diff="CD", target_branch="t",
            baseline_ref="b",
            external_context_section=ctx.render_review_section(),
        )
        rendered = p.prompt()
        assert "external_resolver_context" in rendered
        assert "SUSPECT FINDINGS" in rendered
        assert "RESOLVED FILES" not in rendered

    def test_review_prompt_restoration_rule_conditions_on_upstream(self):
        p = TriageBlindReviewPrompt(
            context="CTX", fix_diff="D", current_diff="CD", target_branch="t",
        )
        rendered = p.prompt()
        # the old flawed rule must be gone...
        assert "exists on the baseline but was removed from the port, the removal was wrong" not in rendered
        # ...replaced by the upstream-conditioned rule
        assert "the upstream fix itself removes the symbol" in rendered
        assert "{external_context_section}" not in rendered


class TestConfigLoader:
    def test_absent_key_leaves_field_empty(self):
        config, _, _ = parse_bug_fix_config(dict(BASE_CONFIG), config_dir="/tmp")
        assert config.external_context_file == ""
        assert config.external_context is None

    def test_relative_path_resolved_against_config_dir(self):
        with tempfile.TemporaryDirectory() as d:
            md = os.path.join(d, "ctx.md")
            open(md, "w").write(VALID_MD)
            cfg = dict(BASE_CONFIG, external_context_file="ctx.md")
            config, _, _ = parse_bug_fix_config(cfg, config_dir=d)
            assert config.external_context_file == md

    def test_missing_file_raises_config_error(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(BASE_CONFIG, external_context_file="nope.md")
            with pytest.raises(ConfigError, match="not found"):
                parse_bug_fix_config(cfg, config_dir=d)

    def test_non_string_raises_config_error(self):
        cfg = dict(BASE_CONFIG, external_context_file=42)
        with pytest.raises(ConfigError, match="non-empty path"):
            parse_bug_fix_config(cfg, config_dir="/tmp")
