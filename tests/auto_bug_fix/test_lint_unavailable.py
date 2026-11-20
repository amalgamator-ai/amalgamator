# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A lint tool missing from the image must not fail verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import PipelineState, _run_build_and_test


def _config(tmp_path):
    return BugFixConfig(
        repo_path=str(tmp_path),
        build_dir=str(tmp_path),
        source_branch="main",
        target_branch="main",
        source_fix_commit="abc123",
        bug_description="x",
        issue_id="TEST-1",
        build_command="",
        test_command="",
    )


def _state():
    state = PipelineState()
    state.allowed_modules = []
    state.seed = []
    state.triage_assessment = {"lint_commands": ["cppcheck --quiet ."]}
    return state


def test_missing_lint_tool_is_skipped_not_failed(tmp_path):
    state = _state()
    with patch(
        "auto_bug_fix.run_bug_fix.run_test_suite",
        return_value=(127, "/bin/sh: cppcheck: command not found", ""),
    ):
        passed, *_ = _run_build_and_test(_config(tmp_path), state)

    assert passed is True


def test_lint_failure_with_the_tool_present_is_advisory(tmp_path):
    state = _state()
    with patch(
        "auto_bug_fix.run_bug_fix.run_test_suite",
        return_value=(1, "xmlparse.c:10: style error", ""),
    ):
        passed, *_ = _run_build_and_test(_config(tmp_path), state)

    assert passed is True


def test_lint_failure_is_recorded_in_the_dossier(tmp_path):
    state = _state()
    state.dossier = MagicMock()
    with patch(
        "auto_bug_fix.run_bug_fix.run_test_suite",
        return_value=(1, "xmlparse.c:10: style error", ""),
    ):
        passed, *_ = _run_build_and_test(_config(tmp_path), state)

    assert passed is True
    state.dossier.add.assert_called_once()
    args, _kwargs = state.dossier.add.call_args
    assert args[0] == "Lint advisory (not a build/test failure)"
    assert "xmlparse.c:10: style error" in args[1]
