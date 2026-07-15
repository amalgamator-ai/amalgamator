"""Branch scanner — identify which release branches need a fix backported."""

from .scanner import ScanConfig, ScanReport, run_scan
from .git_utils import FixStatus, BranchCheckResult

__all__ = ["ScanConfig", "ScanReport", "run_scan", "FixStatus", "BranchCheckResult"]
