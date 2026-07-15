"""
branch_scanner/git_utils.py

Re-exports from triage.py for backward compatibility.
Detection logic lives in triage.py (shared with auto_bug_fix Phase 0).
"""

from .triage import BranchCheckResult, FixStatus, check_branch, get_remote_branches

__all__ = ["BranchCheckResult", "FixStatus", "check_branch", "get_remote_branches"]
