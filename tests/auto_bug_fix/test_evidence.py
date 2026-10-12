"""Tests for auto_bug_fix.evidence — deterministic adaptation evidence."""

from __future__ import annotations

import subprocess
from typing import cast

import pytest

from auto_bug_fix.evidence import (
    LineFinding,
    TokenFinding,
    added_lines_by_file,
    annotate_dead_config,
    apply_substitutions,
    audit_added_tokens,
    audit_line_provenance,
    conditional_guards,
    dead_config_symbol,
    extract_tokens,
    find_pure_rename_restorations,
    mine_substitutions,
    port_line_changes,
    render_evidence,
    run_file_checks,
    scan_followup_fixes,
    target_config_state,
    token_exists_on_ref,
    validate_triage_prerequisites,
)

SAMPLE_DIFF = """\
diff --git a/lib/a.c b/lib/a.c
--- a/lib/a.c
+++ b/lib/a.c
@@ -1,3 +1,4 @@
 context line
+added_one();
-removed();
+added_two();
++++increment;
diff --git a/lib/gone.c b/lib/gone.c
--- a/lib/gone.c
+++ /dev/null
@@ -1 +0,0 @@
-dead();
diff --git a/lib/new.c b/lib/new.c
--- /dev/null
+++ b/lib/new.c
@@ -0,0 +1 @@
+fresh_file_line();
"""


def test_added_lines_by_file_groups_and_skips_deletions():
    added = added_lines_by_file(SAMPLE_DIFF)
    assert added == {
        # "++++increment;" is added CONTENT starting with "+++", not a file
        # header — it only counts as a header directly after a "---" line.
        "lib/a.c": ["added_one();", "added_two();", "+++increment;"],
        "lib/new.c": ["fresh_file_line();"],
    }


def test_extract_tokens_filters_stopwords_and_short_names():
    tokens = extract_tokens(["if (legacy_release_rcu(ptr) != NULL) return do_it(x);"])
    assert "legacy_release_rcu" in tokens
    assert "do_it" in tokens
    assert "if" not in tokens  # keyword
    assert "NULL" not in tokens  # stopword
    assert "ptr" in tokens
    assert "x" not in tokens  # too short


def test_apply_substitutions_only_touches_changed_lines():
    preimage = "keep_this();\nlegacy_release(z);\n"
    current = "keep_this();\nlegacy_release(z);\nlegacy_release(d);\n"
    new_text, applied = apply_substitutions(
        current, preimage, {"legacy_release": "wrapped_release"}
    )
    # The pre-existing legacy_release(z) line is target code — untouched.
    assert "legacy_release(z);" in new_text
    assert "wrapped_release(d);" in new_text
    assert applied == {"legacy_release": 1}


def test_apply_substitutions_word_bounded():
    new_text, applied = apply_substitutions(
        "my_legacy_release_wrapper(d);\n", "", {"legacy_release": "wrapped_release"}
    )
    assert new_text == "my_legacy_release_wrapper(d);\n"
    assert not applied


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


@pytest.fixture
def evidence_repo(tmp_path):
    """main: base -> pre-fix -> fix; target diverged with legacy_release->wrapped_release."""
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")

    (repo / "lib.c").write_text("void shared_helper(void) {}\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")

    # Upstream pre-fix state: uses legacy_release and old_only_sym. The unchanged
    # sync_state() line keeps the rename and the removal in separate hunks.
    (repo / "lib.c").write_text(
        "void shared_helper(void) {}\n"
        "void cleanup(void)\n"
        "{\n"
        "\tlegacy_release(alpha);\n"
        "\tlegacy_release(beta);\n"
        "\tlegacy_release(gamma);\n"
        "\tsync_state();\n"
        "\told_only_sym();\n"
        "}\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "pre-fix upstream state")
    pre_fix = _sha(rp)

    # Target branch diverged: renamed legacy_release -> wrapped_release, dropped old_only_sym.
    _git(rp, "branch", "target", pre_fix)
    _git(rp, "checkout", "-q", "target")
    (repo / "lib.c").write_text(
        "void shared_helper(void) {}\n"
        "/* delta tuning notes */\n"
        "void cleanup(void)\n"
        "{\n"
        "\twrapped_release(alpha);\n"
        "\twrapped_release(beta);\n"
        "\twrapped_release(gamma);\n"
        "\tsync_state();\n"
        "}\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "downstream: wrapped_release API")
    _git(rp, "checkout", "-q", "main")

    # The fix commit: one more legacy_release call plus a brand-new function.
    (repo / "lib.c").write_text(
        "void shared_helper(void) {}\n"
        "void brand_new_func(void) {}\n"
        "void cleanup(void)\n"
        "{\n"
        "\tlegacy_release(alpha);\n"
        "\tlegacy_release(beta);\n"
        "\tlegacy_release(gamma);\n"
        "\tlegacy_release(delta);\n"
        "\tsync_state();\n"
        "\tbrand_new_func();\n"
        "\told_only_sym();\n"
        "}\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix: free delta too")
    fix = _sha(rp)
    return rp, fix


def test_token_exists_on_ref(evidence_repo):
    rp, _fix = evidence_repo
    assert token_exists_on_ref(rp, "target", "wrapped_release")
    assert not token_exists_on_ref(rp, "target", "legacy_release")


def test_audit_classifies_upstream_only_and_unknown(evidence_repo):
    rp, fix = evidence_repo
    added = {
        "lib.c": [
            "\tlegacy_release(delta);",
            "\told_only_sym();",
            "\tbrand_new_func();",
            "\ttotally_made_up();",
        ]
    }
    findings = {f.token: f.status for f in audit_added_tokens(rp, "target", fix, added)}
    assert findings.get("legacy_release") == "upstream_only"
    assert findings.get("old_only_sym") == "upstream_only"
    assert "brand_new_func" not in findings  # introduced by the fix itself
    assert findings.get("totally_made_up") == "unknown"
    # Arguments exist on the target version of the file — never flagged.
    assert "delta" not in findings or findings["delta"] == "upstream_only"


def test_mine_substitutions_finds_consistent_rename(evidence_repo):
    rp, fix = evidence_repo
    subs = mine_substitutions(rp, f"{fix}^1", "target", ["lib.c"])
    assert subs == {"legacy_release": "wrapped_release"}


def test_mine_substitutions_respects_min_count(evidence_repo):
    rp, fix = evidence_repo
    subs = mine_substitutions(rp, f"{fix}^1", "target", ["lib.c"], min_count=4)
    assert subs == {}


def test_scan_followup_fixes(evidence_repo):
    rp, fix = evidence_repo
    repo_git = ["git", "-C", rp]
    subprocess.run(
        repo_git
        + [
            "commit",
            "--allow-empty",
            "-m",
            f'fix the fix\n\nFixes: {fix[:12]} ("fix: free delta too")',
        ],
        check=True,
        capture_output=True,
    )
    followups = scan_followup_fixes(rp, fix, "main")
    assert len(followups) == 1
    assert "fix the fix" in followups[0]
    # A commit with no follow-ups reports none.
    assert scan_followup_fixes(rp, "HEAD", "main") == []


def test_run_file_checks_reports_failures(evidence_repo):
    rp, _fix = evidence_repo
    assert run_file_checks(rp, ["lib.c"], "test -e {file}", timeout=30) == []
    failures = run_file_checks(rp, ["lib.c"], "test ! -e {file}", timeout=30)
    assert len(failures) == 1 and failures[0].startswith("lib.c")
    bad = run_file_checks(rp, ["lib.c"], "true", timeout=30)
    assert "placeholder" in bad[0]


def test_port_line_changes_tracks_new_file_linenos():
    diff = (
        "diff --git a/f.c b/f.c\n"
        "--- a/f.c\n"
        "+++ b/f.c\n"
        "@@ -10,3 +10,4 @@\n"
        " context_one();\n"
        "+added_one();\n"
        "-removed_one();\n"
        "+added_two();\n"
        " context_two();\n"
    )
    changes = port_line_changes(diff)
    assert changes == {
        "f.c": {
            "added": [(11, "added_one();"), (12, "added_two();")],
            "removed": ["removed_one();"],
        }
    }


def test_conditional_guards_and_dead_symbol():
    text = (
        "top();\n"
        "#ifdef CONFIG_ALPHA\n"
        "in_alpha();\n"
        "#else\n"
        "not_alpha();\n"
        "#endif\n"
        "after();\n"
        "#if defined(CONFIG_BETA) && defined(CONFIG_GAMMA)\n"
        "complex_cond();\n"
        "#endif\n"
    )
    guards = conditional_guards(text)
    assert guards[0] == []
    assert guards[2] == [("CONFIG_ALPHA", True)]
    assert guards[4] == [("CONFIG_ALPHA", False)]
    assert guards[6] == []
    assert guards[8] == [(None, True)]  # complex condition — unmodeled

    disabled = {"CONFIG_ALPHA": False}.get
    enabled = {"CONFIG_ALPHA": True}.get
    assert dead_config_symbol(guards[2], disabled) == "CONFIG_ALPHA"
    assert dead_config_symbol(guards[2], enabled) is None
    # The #else branch is dead when the symbol IS enabled.
    assert dead_config_symbol(guards[4], enabled) == "CONFIG_ALPHA"
    assert dead_config_symbol(guards[4], disabled) is None
    # Unmodeled conditions are conservatively live.
    assert dead_config_symbol(guards[8], disabled) is None


def test_target_config_state_from_tree_fragments(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs" / "flavor").mkdir(parents=True)
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "configs" / "flavor" / "CONFIG_FEATURE_OFF").write_text(
        "# CONFIG_FEATURE_OFF is not set\n"
    )
    (repo / "configs" / "flavor" / "CONFIG_FEATURE_ON").write_text(
        "CONFIG_FEATURE_ON=y\n"
    )
    (repo / "configs" / "flavor" / "CONFIG_FEATURE_MOD").write_text(
        "CONFIG_FEATURE_MOD=m\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "configs")

    assert target_config_state(rp, "HEAD", "CONFIG_FEATURE_OFF") is False
    assert target_config_state(rp, "HEAD", "CONFIG_FEATURE_ON") is True
    # "=m" does not define the plain macro — #ifdef reasoning would be wrong,
    # so tristate-module symbols are unknown.
    assert target_config_state(rp, "HEAD", "CONFIG_FEATURE_MOD") is None
    assert target_config_state(rp, "HEAD", "CONFIG_UNKNOWN") is None


def test_annotate_dead_config_marks_fully_dead_tokens():
    finding = TokenFinding("ghost_call", "upstream_only", ["f.c"])
    live = TokenFinding("live_call", "upstream_only", ["f.c"])
    port_changes = {
        "f.c": {"added": [(3, "\tghost_call();"), (5, "\tlive_call();")], "removed": []}
    }
    current = (
        "top();\n"
        "#ifdef CONFIG_GONE\n"
        "\tghost_call();\n"
        "#endif\n"
        "\tlive_call();\n"
    )
    annotate_dead_config(
        [finding, live],
        port_changes,
        {"f.c": current},
        {"CONFIG_GONE": False}.get,
    )
    assert finding.dead_config == "CONFIG_GONE"
    assert live.dead_config is None

    # An unknown (possibly hallucinated) token is wrong even in dead code —
    # it must never flip to the keep-verbatim directive.
    hallucinated = TokenFinding("ghost_call", "unknown", ["f.c"])
    annotate_dead_config(
        [hallucinated], port_changes, {"f.c": current}, {"CONFIG_GONE": False}.get
    )
    assert hallucinated.dead_config is None


def test_audit_line_provenance_classifies_violations(evidence_repo):
    rp, fix = evidence_repo
    port_diff = (
        "diff --git a/lib.c b/lib.c\n"
        "--- a/lib.c\n"
        "+++ b/lib.c\n"
        "@@ -4,3 +4,7 @@\n"
        " \twrapped_release(gamma);\n"
        "+\tlegacy_release(delta);\n"
        "+  legacy_release(delta);\n"
        "+\tlegacy_release_wrapper(delta);\n"
        "+\tcompletely_unrelated_novelty(omega, sigma);\n"
        "-\tsync_state();\n"
    )
    findings = audit_line_provenance(rp, "target", fix, port_diff)
    kinds = {(f.kind, f.line.strip()) for f in findings}
    # byte-identical to the fix's added line -> no finding for the first add
    assert ("ws_drift", "legacy_release(delta);") in kinds
    assert ("adapted_line", "legacy_release_wrapper(delta);") in kinds
    assert ("novel_line", "completely_unrelated_novelty(omega, sigma);") in kinds
    assert ("removed_target_line", "sync_state();") in kinds
    assert len(findings) == 4


def test_audit_line_provenance_accepts_disciplined_port(evidence_repo):
    rp, fix = evidence_repo
    port_diff = (
        "diff --git a/lib.c b/lib.c\n"
        "--- a/lib.c\n"
        "+++ b/lib.c\n"
        "@@ -4,2 +4,3 @@\n"
        " \twrapped_release(gamma);\n"
        "+\tlegacy_release(delta);\n"
    )
    assert audit_line_provenance(rp, "target", fix, port_diff) == []


def test_audit_line_provenance_exempts_auto_substitutions(evidence_repo):
    rp, fix = evidence_repo
    port_diff = (
        "diff --git a/lib.c b/lib.c\n"
        "--- a/lib.c\n"
        "+++ b/lib.c\n"
        "@@ -4,2 +4,3 @@\n"
        " \twrapped_release(gamma);\n"
        "+\twrapped_release(delta);\n"
    )
    # Without context the substituted line looks like a violation.
    naked = audit_line_provenance(rp, "target", fix, port_diff)
    assert {f.kind for f in naked} == {"adapted_line"}
    # With the pipeline's own substitution declared, it is explained.
    explained = audit_line_provenance(
        rp,
        "target",
        fix,
        port_diff,
        substitutions={"legacy_release": "wrapped_release"},
    )
    assert explained == []


def test_audit_line_provenance_ignores_removed_marker_lines(evidence_repo):
    rp, fix = evidence_repo
    port_diff = (
        "diff --git a/lib.c b/lib.c\n"
        "--- a/lib.c\n"
        "+++ b/lib.c\n"
        "@@ -4,5 +4,2 @@\n"
        " \twrapped_release(gamma);\n"
        "-<<<<<<< HEAD\n"
        "-||||||| parent of abcdef (some subject line here)\n"
        "-=======\n"
        "->>>>>>> abcdef (some subject line here)\n"
        "+\tlegacy_release(delta);\n"
    )
    # Removing markers is cleanup, never a violation.
    findings = audit_line_provenance(rp, "target", fix, port_diff)
    assert findings == []


def test_audit_line_provenance_flags_committed_markers(evidence_repo):
    rp, fix = evidence_repo
    port_diff = (
        "diff --git a/lib.c b/lib.c\n"
        "--- a/lib.c\n"
        "+++ b/lib.c\n"
        "@@ -4,1 +4,4 @@\n"
        " \twrapped_release(gamma);\n"
        "+>>>>>>> abcdef (some subject line here)\n"
        "+=======\n"
        "+\tlegacy_release(delta);\n"
    )
    findings = audit_line_provenance(rp, "target", fix, port_diff)
    kinds = [f.kind for f in findings]
    # Both the unambiguous marker and the accompanying ======= are flagged,
    # and marker findings come first (never crowded out by the cap).
    assert kinds[:2] == ["conflict_marker", "conflict_marker"]

    # A lone seven-equals added line is legitimate content, not a marker.
    lone = (
        "diff --git a/lib.c b/lib.c\n"
        "--- a/lib.c\n"
        "+++ b/lib.c\n"
        "@@ -4,1 +4,3 @@\n"
        " \twrapped_release(gamma);\n"
        "+=======\n"
        "+\tlegacy_release(delta);\n"
    )
    findings = audit_line_provenance(rp, "target", fix, lone)
    assert all(f.kind != "conflict_marker" for f in findings)


def test_audit_line_provenance_flags_stale_fix_replaced_lines(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "f.c").write_text("\tlegacy_invocation(argument);\n\tkeep_this();\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)
    (repo / "f.c").write_text("\tmodern_invocation(argument);\n\tkeep_this();\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix: modernize the call")
    fix = _sha(rp)

    # The port added the fix's new line but never removed the old one — an
    # incompletely applied hunk, invisible to added/removed auditing alone.
    port_diff = (
        "diff --git a/f.c b/f.c\n"
        "--- a/f.c\n"
        "+++ b/f.c\n"
        "@@ -1,2 +1,3 @@\n"
        " \tlegacy_invocation(argument);\n"
        "+\tmodern_invocation(argument);\n"
        " \tkeep_this();\n"
    )
    current = {
        "f.c": "\tlegacy_invocation(argument);\n\tmodern_invocation(argument);\n\tkeep_this();\n"
    }
    findings = audit_line_provenance(rp, base, fix, port_diff, current_texts=current)
    stale = [f for f in findings if f.kind == "stale_line"]
    assert len(stale) == 1
    assert stale[0].line.strip() == "legacy_invocation(argument);"

    # A port that fully applied the hunk raises nothing.
    done = {"f.c": "\tmodern_invocation(argument);\n\tkeep_this();\n"}
    port_diff_done = port_diff.replace(
        " \tlegacy_invocation(argument);\n", "-\tlegacy_invocation(argument);\n"
    )
    findings = audit_line_provenance(rp, base, fix, port_diff_done, current_texts=done)
    assert [f for f in findings if f.kind == "stale_line"] == []


def test_stale_line_not_flagged_for_faithful_line_moves(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "f.c").write_text("\tprepare_things();\n\tstep_alpha_operation();\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)
    # The fix MOVES the line earlier: removes and re-adds the same bytes.
    (repo / "f.c").write_text("\tstep_alpha_operation();\n\tprepare_things();\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix: do alpha before preparing")
    fix = _sha(rp)

    port_diff = (
        "diff --git a/f.c b/f.c\n"
        "--- a/f.c\n"
        "+++ b/f.c\n"
        "@@ -1,2 +1,2 @@\n"
        "+\tstep_alpha_operation();\n"
        " \tprepare_things();\n"
        "-\tstep_alpha_operation();\n"
    )
    current = {"f.c": "\tstep_alpha_operation();\n\tprepare_things();\n"}
    findings = audit_line_provenance(rp, base, fix, port_diff, current_texts=current)
    assert [f for f in findings if f.kind == "stale_line"] == []


def test_find_pure_rename_restorations(evidence_repo):
    rp, fix = evidence_repo
    # The port token-renamed the fix's added line toward target naming:
    # `beta` exists in the target file, `delta` (the fix's token) exists on
    # the target tree — a deterministic restoration.
    port_diff = (
        "diff --git a/lib.c b/lib.c\n"
        "--- a/lib.c\n"
        "+++ b/lib.c\n"
        "@@ -5,2 +5,3 @@\n"
        " \twrapped_release(gamma);\n"
        "+\tlegacy_release(beta);\n"
    )
    result = find_pure_rename_restorations(rp, "target", fix, port_diff)
    assert result == {
        "lib.c": [(6, "\tlegacy_release(beta);", "\tlegacy_release(delta);")]
    }

    # A rename whose fix-side token does NOT exist in the target file is
    # never restored (that would introduce a missing symbol): brand_new_func
    # is introduced by the fix itself and absent from the target's lib.c.
    port_diff_bad = port_diff.replace("\tlegacy_release(beta);", "\tsync_state();")
    assert find_pure_rename_restorations(rp, "target", fix, port_diff_bad) == {}

    # When the fix's line is ALREADY ported elsewhere in the file, restoring
    # a look-alike would manufacture a duplicate statement — skip it.
    port_diff_dup = port_diff.replace(
        "+\tlegacy_release(beta);\n",
        "+\tlegacy_release(delta);\n+\tlegacy_release(beta);\n",
    )
    assert find_pure_rename_restorations(rp, "target", fix, port_diff_dup) == {}


def test_port_line_changes_tracks_whole_file_deletion():
    diff = (
        "diff --git a/gone.c b/gone.c\n"
        "--- a/gone.c\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-first_line();\n"
        "-second_line();\n"
    )
    changes = port_line_changes(diff)
    assert changes == {
        "gone.c": {"added": [], "removed": ["first_line();", "second_line();"]}
    }


def test_render_evidence_sections():
    from auto_bug_fix.evidence import TokenFinding

    report = render_evidence(
        [
            TokenFinding("legacy_release", "upstream_only", ["lib.c"]),
            TokenFinding("ghost_sym", "unknown", ["lib.c"]),
        ],
        {"old_flat_member": "new_nested_member"},
        {"callback_alpha": "callback_alpha2"},
        ["abc1234 fix the fix"],
        ["lib.c: exit 1"],
    )
    assert "legacy_release" in report and "MUST be adapted" in report
    assert "ghost_sym" in report and "hallucinated" in report
    assert "old_flat_member" in report and "auto-applied" in report.lower()
    assert "callback_alpha" in report
    assert "abc1234" in report
    assert "exit 1" in report
    assert render_evidence([], {}, {}, [], []) == ""


def test_render_evidence_dead_config_and_line_findings():
    dead = TokenFinding(
        "ghost_sym", "upstream_only", ["f.c"], dead_config="CONFIG_GONE"
    )
    report = render_evidence(
        [dead],
        {},
        {},
        [],
        [],
        line_findings=[
            LineFinding(
                "f.c",
                "removed_target_line",
                "\told_target_call();",
                detail="removed 1x, the fix removes 0x",
            ),
            LineFinding(
                "f.c", "adapted_line", "\tnew_variant();", detail="original_from_fix();"
            ),
        ],
        conflict_marker_files=["g.c"],
    )
    assert "compiled out" in report and "CONFIG_GONE" in report
    assert "MUST be adapted" not in report  # dead finding switches directive
    assert "not remove" in report
    assert "old_target_call();" in report
    assert "reproduce the fix's bytes" in report
    assert "CRITICAL: conflict markers" in report and "g.c" in report


# ---------------------------------------------------------------------------
# validate_triage_prerequisites
# ---------------------------------------------------------------------------


class TestValidateTriagePrerequisites:
    """Cover degenerate symbols, nonexistence vs non-nonexistence reasons."""

    @pytest.mark.parametrize("symbol", ["", "   ", "()", "Foo.", ".foo"])
    def test_degenerate_symbols_produce_no_crash(self, evidence_repo, symbol):
        rp, _ = evidence_repo
        prereqs = [{"symbol": symbol, "portable": False, "reason": "does not exist"}]
        corrections = validate_triage_prerequisites(rp, "target", prereqs)
        assert corrections == []

    def test_non_dict_entry_is_skipped(self, evidence_repo):
        rp, _ = evidence_repo
        prereqs = [
            "just a string",
            42,
            None,
            {
                "symbol": "shared_helper",
                "portable": False,
                "reason": "does not exist on target",
            },
        ]
        corrections = validate_triage_prerequisites(
            rp, "target", cast(list[dict], prereqs)
        )
        assert any(c.symbol == "shared_helper" for c in corrections)

    def test_nonexistence_reason_gets_corrected(self, evidence_repo):
        rp, _ = evidence_repo
        prereqs = [
            {
                "symbol": "shared_helper",
                "portable": False,
                "reason": "shared_helper does not exist on the target branch",
            },
        ]
        corrections = validate_triage_prerequisites(rp, "target", prereqs)
        assert len(corrections) == 1
        assert "exists on the target branch" in corrections[0].correction
        assert "pipeline note" not in corrections[0].correction

    def test_signature_difference_reason_is_appended(self, evidence_repo):
        rp, _ = evidence_repo
        prereqs = [
            {
                "symbol": "shared_helper",
                "portable": False,
                "reason": "exists but the signature differs on target",
            },
        ]
        corrections = validate_triage_prerequisites(rp, "target", prereqs)
        assert len(corrections) == 1
        assert corrections[0].correction.startswith(
            "exists but the signature differs on target"
        )
        assert "pipeline note" in corrections[0].correction
        assert "word-grep" in corrections[0].correction

    def test_truly_missing_symbol_produces_no_correction(self, evidence_repo):
        rp, _ = evidence_repo
        prereqs = [
            {
                "symbol": "totally_nonexistent_xyz",
                "portable": False,
                "reason": "does not exist on target",
            },
        ]
        corrections = validate_triage_prerequisites(rp, "target", prereqs)
        assert corrections == []

    def test_portable_true_prereqs_are_skipped(self, evidence_repo):
        rp, _ = evidence_repo
        prereqs = [
            {"symbol": "shared_helper", "portable": True, "reason": "does not exist"},
        ]
        corrections = validate_triage_prerequisites(rp, "target", prereqs)
        assert corrections == []


# ---------------------------------------------------------------------------
# Macro-generated token (## token-pasting) false-positive guard
# ---------------------------------------------------------------------------


@pytest.fixture
def macro_repo(tmp_path):
    """Repo where the target generates helper_foo via ## but has no literal call."""
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")

    # Base: a generator macro, no call sites yet.
    (repo / "helpers.h").write_text(
        "#define MAKE_HELPER(name) void helper_##name(void) {}\n"
        "MAKE_HELPER(foo);\n"
        "MAKE_HELPER(bar);\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)

    # Upstream pre-fix: adds a literal call site for helper_foo.
    (repo / "caller.c").write_text("void use(void) { helper_foo(); }\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "upstream: first call site")

    # Upstream fix: adds another call site.
    (repo / "caller.c").write_text(
        "void use(void) { helper_foo(); }\n" "void use2(void) { helper_foo(); }\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix: second call site")
    fix = _sha(rp)

    # Target branch diverged from base — has the generator but no literal calls.
    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "target")
    (repo / "helpers.h").write_text(
        "#define MAKE_HELPER(name) void helper_##name(void) {}\n"
        "MAKE_HELPER(foo);\n"
        "MAKE_HELPER(bar);\n"
        "MAKE_HELPER(baz);\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "target: add baz helper")
    _git(rp, "checkout", "-q", "main")

    return rp, fix


def test_macro_generated_token_not_unconditional_must(macro_repo):
    """A token generated via ## on the target must not get 'MUST be adapted'."""
    rp, fix = macro_repo
    added = {"caller.c": ["\tvoid use2(void) { helper_foo(); }"]}
    findings = audit_added_tokens(rp, "target", fix, added)
    foo_findings = [f for f in findings if f.token == "helper_foo"]
    assert len(foo_findings) == 1
    f = foo_findings[0]
    assert f.status == "upstream_only"
    assert f.generated_hint is not None
    assert "##" in f.generated_hint or "MAKE_HELPER" in f.generated_hint

    report = render_evidence(findings, {}, {}, [], [])
    assert "MUST be adapted" not in report
    assert "macro-generated" in report or "token-pasting" in report


def test_non_macro_token_still_gets_must(evidence_repo):
    """A regular upstream_only token without ## generators still gets MUST."""
    rp, fix = evidence_repo
    added = {"lib.c": ["\tlegacy_release(delta);"]}
    findings = audit_added_tokens(rp, "target", fix, added)
    legacy = [f for f in findings if f.token == "legacy_release"]
    assert len(legacy) == 1
    assert legacy[0].generated_hint is None

    report = render_evidence(findings, {}, {}, [], [])
    assert "MUST be adapted" in report
