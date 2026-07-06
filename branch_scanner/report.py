"""
Report generation — JSON and human-readable table output.
"""

import json
from pathlib import Path

from .git_utils import FixStatus
from .scanner import ScanReport

STATUS_ICON = {
    FixStatus.EXACT_MATCH:    "✓  ALREADY FIXED (exact commit)",
    FixStatus.EQUIVALENT:     "✓  ALREADY FIXED (equivalent patch)",
    FixStatus.LIKELY_PRESENT: "~  LIKELY FIXED  (file diff)",
    FixStatus.NEEDS_FIX:      "✗  NEEDS FIX",
    FixStatus.AMBIGUOUS:      "?  AMBIGUOUS     (recommend LLM check)",
    FixStatus.ERROR:          "!  ERROR",
}

STATUS_SYMBOL = {
    FixStatus.EXACT_MATCH:    "ALREADY_FIXED",
    FixStatus.EQUIVALENT:     "ALREADY_FIXED",
    FixStatus.LIKELY_PRESENT: "LIKELY_FIXED",
    FixStatus.NEEDS_FIX:      "NEEDS_FIX",
    FixStatus.AMBIGUOUS:      "AMBIGUOUS",
    FixStatus.ERROR:          "ERROR",
}


def print_report(report: ScanReport) -> None:
    """Print a human-readable report to stdout."""
    print()
    print("=" * 70)
    print("  BRANCH SCANNER REPORT")
    print("=" * 70)
    print(f"  Repo:        {report.repo}")
    print(f"  Fix commit:  {report.fix_commit}")
    if report.fix_description:
        print(f"  Description: {report.fix_description}")
    print(f"  Branches scanned: {len(report.results)}")
    print("=" * 70)
    print()

    col_w = max((len(r.branch) for r in report.results), default=20) + 2
    print(f"  {'BRANCH':<{col_w}}  STATUS")
    print(f"  {'-'*col_w}  {'-'*40}")

    for result in sorted(report.results, key=lambda r: r.branch):
        icon = STATUS_ICON.get(result.status, result.status.value)
        print(f"  {result.branch:<{col_w}}  {icon}")
        if result.detail:
            print(f"  {'':<{col_w}}  → {result.detail}")

    print()
    print("-" * 70)
    print(f"  NEEDS FIX    : {len(report.needs_fix)} branch(es)")
    print(f"  ALREADY FIXED: {len(report.already_fixed)} branch(es)")
    if report.ambiguous:
        print(f"  AMBIGUOUS    : {len(report.ambiguous)} branch(es) — review manually")
    if report.errors:
        print(f"  ERRORS       : {len(report.errors)} branch(es)")
    print("=" * 70)

    if report.needs_fix:
        print()
        print("  Branches that NEED the fix:")
        for r in report.needs_fix:
            print(f"    - {r.branch}")
    print()


def save_json(report: ScanReport, output_path: Path) -> None:
    """Save full report as JSON."""
    data = {
        "repo": report.repo,
        "fix_commit": report.fix_commit,
        "fix_description": report.fix_description,
        "summary": {
            "total": len(report.results),
            "needs_fix": len(report.needs_fix),
            "already_fixed": len(report.already_fixed),
            "ambiguous": len(report.ambiguous),
            "errors": len(report.errors),
        },
        "branches": [
            {
                "branch": r.branch,
                "status": STATUS_SYMBOL.get(r.status, r.status.value),
                "raw_status": r.status.value,
                "method": r.method,
                "detail": r.detail,
                "changed_files": r.changed_files,
            }
            for r in report.results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"  JSON report saved to: {output_path}")
