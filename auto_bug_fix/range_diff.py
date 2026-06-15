"""git range-diff driver for Phase 4.5 semantic-equivalence checking."""
from __future__ import annotations

import re
from typing import Literal

from auto_bug_fix.git_tools import git_range_diff as _git_range_diff


def run_range_diff(repo_path: str, upstream_range: str, ported_range: str) -> str:
    """Run git range-diff between upstream and ported commit ranges."""
    return _git_range_diff(repo_path, upstream_range, ported_range)


def parse_equivalence(range_diff_output: str) -> Literal["identical", "modified", "unmatched"]:
    """Classify range-diff output as identical, modified, or unmatched."""
    if not range_diff_output.strip():
        return "unmatched"

    has_identical = bool(re.search(r"^\s*\d+:\s+\w+\s+=\s+\d+:\s+\w+", range_diff_output, re.MULTILINE))
    has_modified = bool(re.search(r"^\s*\d+:\s+\w+\s+!\s+\d+:\s+\w+", range_diff_output, re.MULTILINE))
    has_unmatched = bool(re.search(r"^\s*-:\s+---+\s+>\s+\d+:", range_diff_output, re.MULTILINE)) or \
                    bool(re.search(r"^\s*\d+:\s+\w+\s+<\s+-:", range_diff_output, re.MULTILINE))

    if has_unmatched:
        return "unmatched"
    if has_modified:
        return "modified"
    if has_identical:
        return "identical"
    return "unmatched"
