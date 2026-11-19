# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the dist-git shape that rejected the opus_ymir run.

The opus_ymir run (expat 2.2.5 on CentOS Stream 8) delivered tests and source
changes inside spec patch files.  The fused pre-RED result was rejected as a
"non-test path" change, and the fused resolution was rejected by the line
overlap check.  Both were false negatives; these tests pin the accepting path.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.claude_utils import BugFixClaudeConfig
from auto_bug_fix.dossier import Dossier
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    _run_fused_pre_red,
    _run_fused_resolution,
    _shares_content_with_fix,
    _validate_fused_pre_red,
)
from auto_bug_fix.tracker import Tracker

TESTS_PATCH = """\
diff --git a/expat/tests/runtests.c b/expat/tests/runtests.c
new file mode 100644
--- /dev/null
+++ b/expat/tests/runtests.c
@@ -0,0 +1,2 @@
+int test_alloc_tracker(void) { return 0; }
+/* CVE-2025-59375 test port */
"""

SPEC = "Name: expat\nVersion: 2.2.5\n"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


def _config(rp, fix):
    return BugFixConfig(
        repo_path=rp,
        build_dir=rp,
        source_branch="main",
        target_branch="target",
        source_fix_commit=fix,
        bug_description="expat CVE-2025-59375",
        issue_id="expat-cve-2025-59375-c8s",
        blind_review=True,
    )


def _claude(rp):
    return BugFixClaudeConfig(
        model="test-model",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=rp,
    )


def _state():
    state = PipelineState()
    state.seed = [
        "expat/lib/xmlparse.c",
        "expat/lib/expat.h",
        "expat/lib/internal.h",
        "expat/tests/alloc_tests.c",
        "expat/tests/nsalloc_tests.c",
        "expat/tests/basic_tests.c",
    ]
    state.dossier = Dossier(
        issue_id="expat-cve-2025-59375-c8s",
        source_branch="main",
        target_branch="target",
        fix_commit="x",
        bug_description="expat CVE-2025-59375",
    )
    return state


def _repo(tmp_path):
    """A dist-git target: a spec and no source tree, like opus_ymir."""
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "expat.spec").write_text(SPEC)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "dist-git base")
    base = _sha(rp)

    for path, text in {
        "expat/lib/xmlparse.c": (
            "void track_alloc(size_t bytes) { (void)bytes; }\n"
            "void check_amplification(size_t bytes) { (void)bytes; }\n"
        ),
        "expat/lib/expat.h": "int XML_SetAllocTracker(void);\n",
        "expat/lib/internal.h": "void track_alloc(size_t bytes);\n",
        "expat/tests/alloc_tests.c": "int test_alloc(void) { return 0; }\n",
        "expat/tests/nsalloc_tests.c": "int test_nsalloc(void) { return 0; }\n",
        "expat/tests/basic_tests.c": "int test_basic(void) { return 0; }\n",
    }.items():
        path_obj = repo / path
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(text)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "upstream fix")
    fix = _sha(rp)

    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "--detach", "target")
    return rp, base, fix


def _write_result(run_dir: Path) -> None:
    (run_dir / "pre_red_result.json").write_text(
        json.dumps(
            {
                "triage": {
                    "commit_type": "bug-fix",
                    "files_to_skip": [],
                    "regeneration_commands": [],
                    "lint_commands": [],
                    "verification_commands": [],
                    "estimated_difficulty": "HARD",
                    "recommendation": "PROCEED_WITH_CAUTION",
                    "priority_files": ["expat/lib/xmlparse.c"],
                    "reasoning": "dist-git spec patch",
                },
                "test_files": [
                    "expat/tests/alloc_tests.c",
                    "expat/tests/nsalloc_tests.c",
                    "expat/tests/basic_tests.c",
                ],
                "tests_ported": [
                    "expat-2.2.5-CVE-2025-59375-tests.patch (expat/tests/runtests.c)"
                ],
                "prerequisites": [],
                "build": {"command": "centpkg prep", "exit_code": 0},
            }
        )
    )


def _tracker(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return run_dir, Tracker(
        issue_id="expat-cve-2025-59375-c8s",
        config_dict={},
        model="test-model",
        output_dir=str(run_dir),
    )


def _write_build_tree(rp):
    """What centpkg sources/prep/local leaves in the dist-git worktree."""
    tests = Path(rp) / "libexpat-R_2_2_5" / "expat" / "tests"
    tests.mkdir(parents=True)
    (tests / "runtests.c").write_text("int main(void) { return 0; }\n")
    (tests / "runtests.log").write_text("PASS\n")
    rpm_dir = Path(rp) / "aarch64"
    rpm_dir.mkdir()
    (rpm_dir / "expat-2.2.5-18.el8.aarch64.rpm").write_text("rpm\n")
    (Path(rp) / ".build-2.2.5-18.el8.log").write_text("build log\n")


def test_ymir_pre_red_accepts_spec_plus_tests_patch(tmp_path):
    rp, base, fix = _repo(tmp_path)
    run_dir, tracker = _tracker(tmp_path)

    async def fake(claude_config, prompts, tracker=None):
        (Path(rp) / "expat-2.2.5-CVE-2025-59375-tests.patch").write_text(TESTS_PATCH)
        (Path(rp) / "expat.spec").write_text(
            SPEC + "Patch20: expat-2.2.5-CVE-2025-59375-tests.patch\n%patch20 -p1\n"
        )
        _git(rp, "add", "-A")
        _git(rp, "commit", "-m", "test-port: expat-cve-2025-59375-c8s")
        _write_build_tree(rp)
        _write_result(run_dir)
        return []

    state = _state()
    with patch("auto_bug_fix.run_bug_fix.claude_run", side_effect=fake):
        outcome = _run_fused_pre_red(_config(rp, fix), _claude(rp), state, tracker)

    assert outcome is not None, "fused pre-RED was rejected; legacy fallback redid it"
    assert state.ported_test_files == ["expat/tests/runtests.c"]
    assert any(
        item.label == "Packaging-carried pre-RED change" for item in state.dossier.items
    )
    assert _validate_fused_pre_red(_config(rp, fix), base) == []
    committed = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert not any(
        path.startswith(("libexpat-R_2_2_5/", "aarch64/")) for path in committed
    )


def test_ymir_fused_resolution_accepts_a_patch_carried_fix(tmp_path):
    rp, base, fix = _repo(tmp_path)
    run_dir, tracker = _tracker(tmp_path)
    seen: list[str] = []

    async def fake(claude_config, prompts, tracker=None):
        seen.append(prompts[0])
        if "Backport Resolution Agent" in prompts[0]:
            diff = subprocess.run(
                ["git", "diff", f"{fix}^", fix],
                cwd=rp,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            (Path(rp) / "expat-2.2.5-CVE-2025-59375.patch").write_text(diff)
            (Path(rp) / "expat.spec").write_text(
                SPEC + "Patch21: expat-2.2.5-CVE-2025-59375.patch\n%patch21 -p1\n"
            )
            _write_build_tree(rp)
            _git(rp, "add", "-A")
            _git(rp, "commit", "-m", "backport CVE-2025-59375")
        return []

    state = _state()
    config = _config(rp, fix)
    assert _shares_content_with_fix(config) is False  # before the session
    with (
        patch("auto_bug_fix.run_bug_fix.claude_run", side_effect=fake),
        tracker.phase("test"),
    ):
        result = _run_fused_resolution(
            config,
            _claude(rp),
            state,
            tracker,
            commit=fix,
            label="expat-cve-2025-59375-c8s",
            base_head=base,
            conflicted_files=["expat/lib/xmlparse.c"],
            files_to_skip=[],
            conflict_summary_parts=[],
        )

    assert result == "resolved", "fused resolution fell back to the per-chunk resolver"
    assert state.fused_resolution_ran is True
    assert _shares_content_with_fix(config) is True
    assert any("Independent Review Agent" in p for p in seen)
    assert not any("Narrow Resolution Agent" in p for p in seen)
    gates = tracker._run.phases[-1].gate_decisions
    assert gates.get("semantic_equivalence") != "fused_unmatched"
    assert gates.get("resolution_carrier") == "packaging"
    changed = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert sorted(changed) == [
        "expat-2.2.5-CVE-2025-59375.patch",
        "expat.spec",
    ]
    log = subprocess.run(
        ["git", "log", "--format=%s", f"{base}..HEAD"],
        cwd=rp,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "untrack build output" in log
