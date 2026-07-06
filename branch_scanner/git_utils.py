"""
branch_scanner/git_utils.py

Thin helpers for branch discovery.
Detection logic has moved to triage.py (shared with auto_bug_fix Phase 0).
"""

from __future__ import annotations

from pathlib import Path

# Re-export core types so existing code that imports from git_utils still works
from .triage import BranchCheckResult, FixStatus, check_branch, _get_remote_branches


def get_remote_branches(repo_path: Path, pattern: str | None = None) -> list[str]:
    """List remote branches, optionally filtered by a glob pattern."""
    return _get_remote_branches(repo_path, pattern)
