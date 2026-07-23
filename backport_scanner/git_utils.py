"""
backport_scanner/git_utils.py

Re-exports from triage.py for backward compatibility.
Detection logic lives in triage.py (shared with auto_bug_fix Phase 0).
"""

from .triage import (
    BranchCheckResult, FixStatus,
    check_branch, check_ref,
    get_remote_branches, get_tags,
)

__all__ = [
    "BranchCheckResult", "FixStatus",
    "check_branch", "check_ref",
    "get_remote_branches", "get_tags",
]
