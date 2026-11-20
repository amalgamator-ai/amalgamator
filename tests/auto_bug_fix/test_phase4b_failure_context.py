# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 4b receives the build/test failure that escalated the run."""

from __future__ import annotations

from auto_bug_fix.run_bug_fix import PipelineState, _previous_failures_section


def test_no_failures_renders_nothing():
    assert _previous_failures_section(PipelineState()) == ""


def test_failure_renders_commands_and_output():
    state = PipelineState()
    state.verify_failures.append(
        {
            "phase": "Phase 4 verify",
            "build_command": "make -C lib",
            "test_command": "make -C tests check",
            "build_output": "xmlparse.c:10: error: boom",
            "test_output": "",
        }
    )

    text = _previous_failures_section(state)

    assert "<previous_failures>" in text
    assert "make -C lib" in text
    assert "make -C tests check" in text
    assert "xmlparse.c:10: error: boom" in text
    assert "<test_output>" not in text


def test_only_the_last_three_failures_are_rendered():
    state = PipelineState()
    for index in range(5):
        state.verify_failures.append(
            {"phase": f"attempt {index}", "build_output": f"error {index}"}
        )

    text = _previous_failures_section(state)

    assert "attempt 0" not in text
    assert "attempt 1" not in text
    assert "attempt 4" in text


def test_identical_failures_are_rendered_once():
    state = PipelineState()
    entry = {
        "phase": "Phase 4 verify",
        "build_command": "make -C lib",
        "test_command": "make -C tests check",
        "build_output": "undefined reference to `expat_malloc'",
        "test_output": "",
    }
    state.verify_failures.extend([entry, dict(entry)])

    text = _previous_failures_section(state)

    assert text.count("## Phase 4 verify") == 1


def test_distinct_failures_under_the_same_phase_are_both_kept():
    state = PipelineState()
    state.verify_failures.extend(
        [
            {"phase": "Phase 4 verify", "build_output": "first error"},
            {"phase": "Phase 4 verify", "build_output": "second error"},
        ]
    )

    text = _previous_failures_section(state)

    assert text.count("## Phase 4 verify") == 2
    assert "first error" in text
    assert "second error" in text


def test_deduplication_keeps_the_latest_copy():
    state = PipelineState()
    state.verify_failures.extend(
        [
            {"phase": "Phase 4 verify", "build_output": "old failure"},
            {"phase": "Phase 4a attempt 1", "build_output": "recovery failure"},
            {"phase": "Phase 4 verify", "build_output": "old failure"},
        ]
    )

    text = _previous_failures_section(state)

    assert text.count("old failure") == 1
    assert "recovery failure" in text


def test_lint_failure_renders_in_its_own_slot():
    state = PipelineState()
    state.verify_failures.append(
        {
            "phase": "Phase 4 verify",
            "build_command": "make",
            "test_command": "make check",
            "build_output": "",
            "test_output": "",
            "lint_output": "unused import",
        }
    )

    text = _previous_failures_section(state)

    assert "<lint_output>" in text
    assert "unused import" in text
    assert "<test_output>" not in text
