# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The iteration limit is an environment override, not an import hazard."""

from __future__ import annotations

from common import agent_runner


def test_absent_limit_is_unlimited(monkeypatch):
    monkeypatch.delenv("AMALGAMATOR_MAX_AGENT_ITERATIONS", raising=False)

    assert agent_runner._iteration_limit() == 9999


def test_limit_reads_the_environment(monkeypatch):
    monkeypatch.setenv("AMALGAMATOR_MAX_AGENT_ITERATIONS", "42")

    assert agent_runner._iteration_limit() == 42


def test_malformed_limit_falls_back_instead_of_raising(monkeypatch, caplog):
    monkeypatch.setenv("AMALGAMATOR_MAX_AGENT_ITERATIONS", "seven")

    assert agent_runner._iteration_limit() == 9999
    assert "not an integer" in caplog.text
