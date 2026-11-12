# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

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
- checker: check-backport.py
- findings-count: 2

## Suspect findings
Findings from the deterministic checks.

- [advisory][dead_symbol][inferred/low] fs/xfs/libxfs/xfs_defer.h:68: symbol 'dop_inodes' declared but has zero references
- [strong][undefined_symbol][observed/medium] mm/memory-failure.c:1224: symbol 'num_poisoned_pages_dec' used but never defined

## Resolved files
Prior resolver output.

### fs/xfs/libxfs/xfs_attr.c
```diff
diff --git a/fs/xfs/libxfs/xfs_attr.c b/fs/xfs/libxfs/xfs_attr.c
+unique-patch-marker-attr
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
        assert ctx.finding_details[0].severity == "advisory"
        assert ctx.finding_details[0].category == "dead_symbol"
        assert ctx.finding_details[1].severity == "strong"
        assert ctx.finding_details[1].category == "undefined_symbol"
        assert ctx.finding_details[1].confidence == "medium"
        assert set(ctx.resolved_files) == {
            "fs/xfs/libxfs/xfs_attr.c",
            "fs/xfs/libxfs/xfs_defer.c",
        }
        assert (
            "unique-patch-marker-attr" in ctx.resolved_files["fs/xfs/libxfs/xfs_attr.c"]
        )

    def test_findings_only_file_with_declared_count(self):
        path = _write_md(
            "## Metadata\n- checker: check-backport.py\n- findings-count: 1\n\n"
            "## Suspect findings\n\n- a.c:1: bad symbol\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.active
        assert ctx.findings == ["a.c:1: bad symbol"]
        assert ctx.resolved_files == {}
        assert ctx.findings_status == "parsed"

    def test_unparseable_file_is_inactive_not_raising(self):
        path = _write_md("# nothing recognizable\njust prose\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        # Nothing would be rendered, so activating would only add a false
        # "a prior resolver ran" claim.  The operator is still warned.
        assert not ctx.active
        assert ctx.parse_errors
        assert ctx.render_triage_section() == ""
        assert ctx.render_resolution_section() == ""

    def test_default_context_is_empty_strategy(self):
        ctx = ExternalContext()
        assert not ctx.active
        assert ctx.render_triage_section() == ""
        assert ctx.render_resolution_section() == ""
        assert ctx.render_verification_section() == ""


class TestParserRobustness:
    def test_bullets_inside_fences_are_not_findings(self):
        path = _write_md(
            "## Metadata\n- resolver: synthmerge\n- checker: check-backport.py\n"
            "- findings-count: 1\n\n"
            "## Suspect findings\n"
            "- [strong][undefined_symbol][observed/medium] a.c:1: real finding\n"
            "```diff\n"
            "- [advisory][dead_symbol][inferred/low] phantom from a diff\n"
            "```\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert len(ctx.findings) == 1
        assert "real finding" in ctx.findings[0]
        assert not any("phantom" in f for f in ctx.findings)

    def test_fence_info_string_with_hyphen_is_parsed(self):
        path = _write_md(
            "## Resolved files\n\n### a.c\n```diff-foo\nresolved content\n```\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.resolved_files["a.c"] == "resolved content"

    def test_unclosed_fence_is_reported(self):
        path = _write_md("## Resolved files\n\n### a.c\n```diff\nresolved content\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert "a.c" not in ctx.resolved_files
        assert any("unclosed fenced block" in e for e in ctx.parse_errors)

    def test_headings_inside_fenced_content_are_not_structure(self):
        path = _write_md(
            "## Metadata\n- resolver: synthmerge\n- checker: check-backport.py\n\n"
            "## Resolved files\n\n"
            "### docs/README.md\n"
            "```markdown\n"
            "# Title\n"
            "## Not a section\n"
            "### not/a/file.c\n"
            "text\n"
            "```\n\n"
            "### real.c\n"
            "```diff\n"
            "+real\n"
            "```\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert set(ctx.resolved_files) == {"docs/README.md", "real.c"}
        content = ctx.resolved_files["docs/README.md"]
        assert "## Not a section" in content
        assert "### not/a/file.c" in content
        assert ctx.resolved_files["real.c"] == "+real"
        assert not any("unknown sections" in e for e in ctx.parse_errors)

    def test_nested_fence_in_resolved_content_is_preserved(self):
        path = _write_md(
            "## Resolved files\n\n"
            "### docs/README.md\n"
            "````markdown\n"
            "Before\n"
            "```python\n"
            "print('hi')\n"
            "```\n"
            "After\n"
            "````\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        content = ctx.resolved_files["docs/README.md"]
        assert "```python" in content
        assert "print('hi')" in content
        assert content.rstrip().endswith("After")

    def test_missing_fence_is_reported(self):
        path = _write_md("## Resolved files\n\n### a.c\nno fenced content\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert "a.c" not in ctx.resolved_files
        assert any("missing fenced block" in e for e in ctx.parse_errors)

    def test_multiple_fenced_blocks_are_ambiguous_not_merged(self):
        path = _write_md(
            "## Resolved files\n\n"
            "### a.c\n"
            "Example:\n"
            "```diff\n"
            "+example\n"
            "```\n"
            "Real patch:\n"
            "```diff\n"
            "+real\n"
            "```\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert "a.c" not in ctx.resolved_files
        assert any("multiple fenced blocks" in e for e in ctx.parse_errors)

    def test_trailing_commentary_after_block_is_ignored(self):
        path = _write_md(
            "## Resolved files\n\n"
            "### a.c\n"
            "```diff\n"
            "+real\n"
            "```\n"
            "Note: commentary after the block is not part of the patch.\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.resolved_files["a.c"] == "+real"
        assert not any("unclosed fenced block" in e for e in ctx.parse_errors)

    def test_fenced_metadata_is_not_provenance(self):
        path = _write_md(
            "## Metadata\n- resolver: synthmerge\n"
            "```text\n"
            "- checker: fake-checker\n"
            "- findings-count: 0\n"
            "```\n\n"
            "## Suspect findings\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert "checker" not in ctx.metadata
        assert "findings-count" not in ctx.metadata
        assert ctx.findings_status == "unavailable"

    def test_markdown_heading_variants_are_recognized(self):
        path = _write_md(
            "  ## Metadata ##\n"
            "- resolver: synthmerge\n"
            "- checker: check-backport.py\n"
            "- findings-count: 0\n\n"
            "  ## Suspect findings ##\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.active
        assert "metadata" in ctx.found_sections
        assert "suspect findings" in ctx.found_sections
        assert ctx.findings_status == "parsed"

    def test_duplicate_sections_are_reported(self):
        path = _write_md(
            "## Metadata\n- resolver: first\n\n"
            "## Metadata\n- resolver: second\n\n"
            "## Suspect findings\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.metadata["resolver"] == "second"
        assert any("duplicate '## Metadata'" in e for e in ctx.parse_errors)

    def test_indented_detail_bullets_are_not_findings(self):
        path = _write_md(
            "## Metadata\n- checker: check-backport.py\n- findings-count: 1\n\n"
            "## Suspect findings\n"
            "- [strong][dead_symbol][observed/high] a.c:1: real finding\n"
            "  - detail: a nested note\n"
            "    - deeper detail\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings == [
            "[strong][dead_symbol][observed/high] a.c:1: real finding"
        ]
        assert ctx.findings_status == "parsed"

    def test_unclosed_fence_in_findings_is_reported_and_unavailable(self):
        path = _write_md(
            "## Metadata\n- checker: check-backport.py\n- findings-count: 1\n\n"
            "## Suspect findings\n"
            "- real finding\n"
            "```diff\n"
            "- phantom\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings == ["real finding"]
        assert ctx.findings_status == "unavailable"
        assert any(
            "unclosed fenced block in '## Suspect findings'" in e
            for e in ctx.parse_errors
        )


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
        assert "STRONG FINDINGS" in s
        assert "ADVISORY FINDINGS" in s
        # the actual patch bytes reach the consumer, not just a filename
        assert "unique-patch-marker-attr" in s
        assert "diff --git a/fs/xfs/libxfs/xfs_attr.c" in s

    def test_resolution_section_filters_patches_to_requested_files(self):
        path = _write_md(VALID_MD)
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        s = ctx.render_resolution_section(files=["fs/xfs/libxfs/xfs_defer.c"])
        assert "unique-patch-marker-attr" not in s
        assert "xfs_defer.c" in s
        assert "xfs_attr.c" not in s

    def test_resolution_section_reports_no_matching_patch(self):
        path = _write_md(VALID_MD)
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        s = ctx.render_resolution_section(files=["not/touched.c"])
        assert "no prior-resolver patch applies" in s
        assert "unique-patch-marker-attr" not in s

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

    def test_explicit_empty_findings_means_zero_findings(self):
        path = _write_md(
            "## Metadata\n- resolver: synthmerge\n"
            "- checker: check-backport.py\n- findings-count: 0\n\n"
            "## Suspect findings\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings_status == "parsed"
        rendered = ctx.render_resolution_section()
        assert "none reported" in rendered
        assert "all deterministic checks passed" not in rendered

    def test_declared_count_mismatch_is_unavailable_not_clean(self):
        path = _write_md(
            "## Metadata\n- resolver: synthmerge\n"
            "- checker: check-backport.py\n- findings-count: 3\n\n"
            "## Suspect findings\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings_status == "unavailable"
        rendered = ctx.render_resolution_section()
        assert "unavailable" in rendered
        assert "none reported" not in rendered
        # the declared count is still visible, so the discrepancy is obvious
        assert "findings-count: 3" in rendered
        assert any("does not match" in e for e in ctx.parse_errors)

    def test_checker_without_findings_count_is_unavailable(self):
        path = _write_md(
            "## Metadata\n- resolver: synthmerge\n- checker: check-backport.py\n\n"
            "## Suspect findings\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings_status == "unavailable"
        assert any("without findings-count" in e for e in ctx.parse_errors)

    def test_findings_without_declared_count_are_unavailable(self):
        path = _write_md("## Suspect findings\n\n- a.c:1: bad symbol\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.active
        assert ctx.findings == ["a.c:1: bad symbol"]
        assert ctx.findings_status == "unavailable"
        assert any("without findings-count" in e for e in ctx.parse_errors)
        assert "none reported" not in ctx.render_resolution_section()

    def test_duplicate_findings_sections_make_result_unavailable(self):
        path = _write_md(
            "## Metadata\n- checker: check-backport.py\n- findings-count: 1\n\n"
            "## Suspect findings\n"
            "- a.c:1: first block finding\n\n"
            "## Suspect findings\n"
            "- b.c:1: second block finding\n\n"
            "## Resolved files\n\n"
            "### a.c\n"
            "```diff\n"
            "+prior\n"
            "```\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings_status == "unavailable"
        assert ctx.findings == ["b.c:1: second block finding"]
        assert any("duplicate recognized section" in e for e in ctx.parse_errors)
        resolution = ctx.render_resolution_section()
        assert "NO TRUSTED REGIONS" in resolution
        assert "NOT named in the SUSPECT FINDINGS" not in resolution

    def test_unavailable_findings_disable_verbatim_trust(self):
        path = _write_md(
            "## Metadata\n- resolver: synthmerge\n"
            "- checker: check-backport.py\n- findings-count: 2\n\n"
            "## Suspect findings\n"
            "- a.c:1: only one finding parsed\n\n"
            "## Resolved files\n\n"
            "### a.c\n"
            "```diff\n"
            "+prior\n"
            "```\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings_status == "unavailable"
        resolution = ctx.render_resolution_section()
        assert "NO TRUSTED REGIONS" in resolution
        assert "NOT named in the SUSPECT FINDINGS" not in resolution
        assert "UNVERIFIED" in resolution
        triage = ctx.render_triage_section()
        assert "likely low-risk" not in triage
        assert "no file can be treated as cleared" in triage

    def test_findings_section_without_provenance_is_unavailable(self):
        path = _write_md(
            "## Metadata\n- resolver: synthmerge\n\n"
            "## Suspect findings\nlooks fine to me\n"
        )
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings_status == "unavailable"
        assert "unavailable" in ctx.render_resolution_section()

    def test_missing_findings_section_is_unavailable(self):
        path = _write_md("## Metadata\n- resolver: synthmerge\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        assert ctx.findings_status == "unavailable"
        rendered = ctx.render_resolution_section()
        assert "unavailable" in rendered
        assert "all deterministic checks passed" not in rendered

    def test_arbitrary_prose_never_claims_checks_passed(self):
        path = _write_md("# External Resolver Context\n\nlooks fine to me\n")
        try:
            ctx = load_external_context(path)
        finally:
            os.unlink(path)
        # No recognized section: the feature stays off, so no prompt gets a
        # prior-resolver claim or an "unavailable" finding block.
        assert not ctx.active
        assert ctx.findings_status == "unavailable"
        assert ctx.render_resolution_section() == ""
        assert ctx.render_review_section() == ""


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
            context="CTX",
            fix_diff="D",
            source_branch="s",
            target_branch="t",
            seed_files=["a"],
            external_context_section=ctx.render_triage_section(),
        )
        rendered = p.prompt()
        assert "external_resolver_context" in rendered
        assert "synthmerge 0.3.15" in rendered

    def test_triage_prompt_without_section_unchanged(self):
        p = SemanticTriagePrompt(
            context="CTX",
            fix_diff="D",
            source_branch="s",
            target_branch="t",
            seed_files=["a"],
        )
        assert "external_resolver_context" not in p.prompt()
        assert "{external_context_section}" not in p.prompt()

    def test_resolution_prompt_includes_section(self):
        ctx = self._ctx()
        p = NarrowResolutionAgentPrompt(
            context="CTX",
            fix_diff="D",
            conflicted_files=["a"],
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
            context="CTX",
            fix_diff="D",
            current_diff="CD",
            target_branch="t",
            external_context_section=ctx.render_verification_section(),
        )
        assert "Verification scope is NOT reduced" in p.prompt()

    def test_review_prompt_includes_findings_not_resolutions(self):
        ctx = self._ctx()
        p = TriageBlindReviewPrompt(
            context="CTX",
            fix_diff="D",
            current_diff="CD",
            target_branch="t",
            baseline_ref="b",
            external_context_section=ctx.render_review_section(),
        )
        rendered = p.prompt()
        assert "external_resolver_context" in rendered
        assert "SUSPECT FINDINGS" in rendered
        assert "RESOLVED FILES" not in rendered

    def test_review_prompt_restoration_rule_conditions_on_upstream(self):
        p = TriageBlindReviewPrompt(
            context="CTX",
            fix_diff="D",
            current_diff="CD",
            target_branch="t",
        )
        rendered = p.prompt()
        # the old flawed rule must be gone...
        assert (
            "exists on the baseline but was removed from the port, the removal was wrong"
            not in rendered
        )
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
            with open(md, "w") as fh:
                fh.write(VALID_MD)
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

    def test_null_explicit_value_raises_config_error(self):
        cfg = dict(BASE_CONFIG, external_context_file=None)
        with pytest.raises(ConfigError, match="non-empty path"):
            parse_bug_fix_config(cfg, config_dir="/tmp")
