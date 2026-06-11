"""Tests for auto_bug_fix.dossier — trailers, dossier construction, and formatting."""
from auto_bug_fix.dossier import Dossier, build_trailers, format_dossier


def test_build_trailers_basic():
    trailers = build_trailers(phase="3a", model="claude-opus-4-7")
    assert trailers["AutoBugFix-Phase"] == "3a"
    assert trailers["AutoBugFix-Model"] == "claude-opus-4-7"


def test_build_trailers_optional_fields():
    trailers = build_trailers(
        phase="1",
        model="claude-opus-4-7",
        bisect_sha="deadbeef",
    )
    assert "AutoBugFix-Triage-Bisect-SHA" in trailers
    assert trailers["AutoBugFix-Triage-Bisect-SHA"] == "deadbeef"


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


def test_dossier_vulnerability_sections():
    """Test the new bug-specific analysis sections."""
    d = Dossier(
        issue_id="CVE-2024-1234",
        source_branch="main",
        target_branch="v2.0",
        fix_commit="abc123",
        bug_description="CVE-2024-1234: Quadratic runtime DoS (CWE-407)",
    )

    d.set_vulnerability_analysis(
        "The vulnerability allows attackers to cause quadratic runtime "
        "through crafted attribute names that hash-collide."
    )
    d.set_impact_assessment(
        "Target branch IS vulnerable. Vulnerable code introduced in commit deadbeef."
    )
    d.set_fix_explanation(
        "The fix adds early collision detection before the O(n²) scan."
    )

    output = format_dossier(d)

    # Check executive summary sections appear before technical details
    assert "## Vulnerability Analysis" in output
    assert "quadratic runtime" in output
    assert "## Impact Assessment" in output
    assert "Target branch IS vulnerable" in output
    assert "## How the Fix Works" in output
    assert "early collision detection" in output
