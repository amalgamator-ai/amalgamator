# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic human-oriented dossier view."""

from __future__ import annotations

import json

from auto_bug_fix.dossier import (
    Dossier,
    format_dossier,
    format_dossier_human,
    parse_dossier_markdown,
)

PATCH = """diff --git a/foo.c b/foo.c
--- a/foo.c
+++ b/foo.c
@@ -1,2 +1,3 @@
 line
-old
+new
+added
diff --git a/new.c b/new.c
new file mode 100644
--- /dev/null
+++ b/new.c
@@ -0,0 +1,2 @@
+a
+b
"""


def _dossier() -> Dossier:
    dossier = Dossier(
        issue_id="RHEL-1",
        source_branch="origin/master",
        target_branch="abc123",
        fix_commit="fix123",
        bug_description="CVE-2026-0001",
        cherry_pick_path="resolved",
    )
    dossier.add(
        "Semantic triage",
        json.dumps(
            {
                "estimated_difficulty": "HARD",
                "recommendation": "PROCEED_WITH_CAUTION",
                "mode": "commit_series_deterministic",
                "commit_type": "bug-fix",
                "files_to_skip": ["vendor/generated.c"],
                "test_files": ["tests/test_a.c"],
                "clusters": [
                    {
                        "id": "core",
                        "commits": ["a", "b"],
                        "intent": "core fix",
                        "kind": "fix",
                    }
                ],
                "prerequisites": [
                    {
                        "symbol": "foo",
                        "portable": False,
                        "reason": "target already provides foo",
                    }
                ],
            }
        ),
        "Phase 0.5",
    )
    dossier.add(
        "Triage result",
        json.dumps({"forward_patch_id": "not_found"}),
        "Phase 0",
    )
    dossier.add("Test port", "Ported tests/test_a.c", "Phase 0.5c")
    dossier.add(
        "Baseline (clean target branch)",
        "Build PASS; tests skipped in series mode",
        "Phase 1",
    )
    dossier.add(
        "Cluster core",
        json.dumps(
            {
                "cluster": "core",
                "actions": [
                    {
                        "symbol": "foo",
                        "action": "translate",
                        "verify": "grep -q foo file.c",
                    }
                ],
                "risks": ["foo may have a different signature"],
            }
        ),
        "Phase 2.5",
    )
    dossier.add(
        "Fused resolution session report",
        "## A. Resolve\nTook the fix bytes.\n\n## B. Complete\nNothing else.",
        "Phase 2+3a",
    )
    dossier.add("Semantic equivalence", "unmatched", "Phase 4.5")
    dossier.add(
        "Range-diff (upstream vs backport)",
        "range-diff text",
        "Phase 4.5",
    )
    dossier.add("Candidate patch", PATCH, "Phase 5")
    dossier.add("Allowlist conformance", "ok", "Phase 4")
    dossier.add_strategy("default", "clean")
    dossier.add_strategy("patience", "conflict (1 files)")
    dossier.trailers = {
        "AutoBugFix-Phase": "fused",
        "AutoBugFix-Model": "test-model",
    }
    return dossier


def test_human_dossier_is_review_oriented():
    text = format_dossier_human(
        _dossier(),
        outcome="success",
        gates={
            "verify": "passed",
            "completeness": "complete",
            "semantic_equivalence": "unmatched",
            "fused_relatedness": "advisory",
        },
    )

    assert "# Bug-Fix Porting Review: RHEL-1" in text
    assert "| Outcome | **success** |" in text
    assert "| Build / test | passed |" in text
    assert "Semantic equivalence is `unmatched`" in text
    assert "Fused relatedness is `advisory`" in text
    assert "`vendor/generated.c`" in text
    assert "2 files, +4/−1" in text
    assert "| `foo.c` | modified | 2 | 1 |" in text
    assert "| `new.c` | added | 2 | 0 |" in text
    assert "| default | clean | 1 |" in text
    assert "| patience | conflict (1 files) | 1 |" in text
    assert "#### A. Resolve" in text
    assert "diff --git a/foo.c" not in text
    assert "[`dossier.md`](dossier.md)" in text
    assert "candidate.patch" in text


def test_parse_dossier_markdown_roundtrip():
    original = _dossier()
    parsed = parse_dossier_markdown(format_dossier(original))

    assert parsed.issue_id == original.issue_id
    assert parsed.bug_description == original.bug_description
    assert parsed.source_branch == original.source_branch
    assert parsed.target_branch == original.target_branch
    assert parsed.fix_commit == original.fix_commit
    assert parsed.cherry_pick_path == original.cherry_pick_path
    assert [item.label for item in parsed.items] == [
        item.label for item in original.items
    ]
    assert [(s.strategy, s.outcome) for s in parsed.strategies] == [
        (s.strategy, s.outcome) for s in original.strategies
    ]
    assert parsed.trailers == original.trailers


def test_human_dossier_handles_malformed_and_empty_evidence():
    dossier = Dossier(issue_id="RHEL-2")
    dossier.add("Semantic triage", "{not json", "Phase 0.5")
    text = format_dossier_human(dossier)

    assert "# Bug-Fix Porting Review: RHEL-2" in text
    assert "| Outcome | **not recorded** |" in text
    assert "_Not recorded._" in text
    assert "not json" in text


def test_human_dossier_collapses_repeated_records():
    dossier = Dossier(issue_id="RHEL-3")
    for _ in range(4):
        dossier.add("Repeated note", "Same content.", "Phase 9")
    dossier.add("Repeated note", "Different content.", "Phase 9")
    text = format_dossier_human(dossier)

    assert text.count("Same content.") == 1
    assert "(×4 identical records)" in text
    assert "Different content." in text
    assert "| Repeated note | Phase 9 | 5 |" in text


def test_human_dossier_renders_python_literal_gates_as_table():
    dossier = Dossier(issue_id="RHEL-4")
    dossier.add(
        "Triage result",
        "{'forward_patch_id': 'not_found', 'ancestry': 'seed_files_only'}",
        "Phase 0",
    )
    text = format_dossier_human(dossier)

    assert "| forward_patch_id | not_found |" in text
    assert "| ancestry | seed_files_only |" in text
    assert "{'forward_patch_id'" not in text
