# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_bug_fix.dossier — trailers, dossier construction, and formatting."""

from auto_bug_fix.dossier import Dossier, build_trailers, format_dossier


def test_build_trailers_basic():
    trailers = build_trailers(phase="3a", model="claude-opus-4-7")
    assert trailers["AutoBugFix-Phase"] == "3a"
    assert trailers["AutoBugFix-Model"] == "claude-opus-4-7"


def test_build_trailers_coauthor_canonical_casing():
    trailers = build_trailers(phase="3a", model="claude-opus-4-7")
    assert "Co-authored-by" in trailers
    assert "Co-Authored-By" not in trailers
    assert trailers["Co-authored-by"] == "amalgamator <amalgamator@redhat.com>"


def test_build_trailers_optional_fields():
    trailers = build_trailers(
        phase="1",
        model="claude-opus-4-7",
        resolved_hunks="3",
    )
    assert trailers["AutoBugFix-Resolved-Hunks"] == "3"


def test_dossier_add_items():
    d = Dossier(
        issue_id="CVE-2024-9999",
        source_branch="main",
        target_branch="release-1.0",
        fix_commit="abc123",
        bug_description="Buffer overflow in parser",
    )
    d.add(label="Triage", content="all clear", source="Phase 0")
    d.add(label="Bisect", content="sha=deadbeef", source="Phase 1")
    assert len(d.items) == 2
    assert d.items[0].label == "Triage"
    assert d.items[1].content == "sha=deadbeef"


def test_format_dossier():
    d = Dossier(
        issue_id="CVE-2024-9999",
        source_branch="main",
        target_branch="release-1.0",
        fix_commit="abc123",
        bug_description="Buffer overflow in parser",
    )
    d.add(label="Triage Result", content="proceed", source="Phase 0")
    d.add_strategy(strategy="default", outcome="conflict", commentary="hunks overlap")
    d.add_strategy(strategy="patience", outcome="clean")

    output = format_dossier(d)
    assert "CVE-2024-9999" in output
    assert "**Source branch**: main" in output
    assert "**Target branch**: release-1.0" in output
    assert "Buffer overflow in parser" in output
    assert "Triage Result" in output
    assert "proceed" in output
    assert "Strategies Attempted" in output
    assert "default" in output
    assert "patience" in output
