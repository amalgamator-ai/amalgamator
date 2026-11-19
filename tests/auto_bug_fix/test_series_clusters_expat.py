# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure the real expat 2.2.5 series when a local clone is available.

Skipped by default. Point AMALGAMATOR_EXPAT_REPO at a clone containing the
PR #1034 commits (base `abf81a11`, merge `ac7a27e1`) to run it:

    AMALGAMATOR_EXPAT_REPO=... pytest tests/auto_bug_fix/test_series_clusters_expat.py
"""

from __future__ import annotations

import os
import subprocess

import pytest

from auto_bug_fix.series_clusters import build_partition

REPO = os.environ.get("AMALGAMATOR_EXPAT_REPO", "")

pytestmark = pytest.mark.skipif(
    not REPO or not os.path.isdir(REPO),
    reason="set AMALGAMATOR_EXPAT_REPO to a clone with the expat series",
)


def _series() -> list[str]:
    out = subprocess.run(
        ["git", "rev-list", "--reverse", "ac7a27e1^1..ac7a27e1^2"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def test_expat_series_splits_into_four_clusters():
    series = _series()
    assert len(series) == 19

    raw = [
        {"id": "enablers", "commits": [s[:8] for s in series[0:7]]},
        {"id": "tracker", "commits": [s[:8] for s in series[8:13]]},
        {"id": "tests", "commits": [s[:8] for s in series[13:15]]},
    ]
    dropped = [
        {"commit": sha[:8], "reason": "unrelated or docs"}
        for sha in (series[7], series[15], series[16], series[17], series[18])
    ]

    partition = build_partition(
        raw,
        series,
        REPO,
        public_interface_files=["expat/lib/expat.h", "expat/lib/libexpat.def"],
        dropped=dropped,
    )

    assert not partition.fell_back
    assert [c.size.changed_lines for c in partition.clusters] == [115, 375, 57, 266]
    assert [c.mode for c in partition.clusters] == [
        "mechanical",
        "semantic",
        "mechanical",
        "semantic",
    ]
    assert partition.splits
