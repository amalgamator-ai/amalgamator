"""
Report generation — JSON and human-readable table output.
"""

import json
from pathlib import Path

from .scanner import ScanReport
from .triage import FixStatus

STATUS_ICON = {
    FixStatus.EXACT_MATCH: "✓  ALREADY FIXED (exact commit)",
    FixStatus.EQUIVALENT: "✓  ALREADY FIXED (equivalent patch)",
    FixStatus.NOT_AFFECTED: "—  NOT AFFECTED  (predates the bug)",
    FixStatus.LIKELY_PRESENT: "~  LIKELY FIXED  (file diff)",
    FixStatus.NEEDS_FIX: "✗  NEEDS FIX",
    FixStatus.AMBIGUOUS: "?  AMBIGUOUS     (recommend LLM check)",
    FixStatus.ERROR: "!  ERROR",
}

STATUS_SYMBOL = {
    FixStatus.EXACT_MATCH: "ALREADY_FIXED",
    FixStatus.EQUIVALENT: "ALREADY_FIXED",
    FixStatus.NOT_AFFECTED: "NOT_AFFECTED",
    FixStatus.LIKELY_PRESENT: "LIKELY_FIXED",
    FixStatus.NEEDS_FIX: "NEEDS_FIX",
    FixStatus.AMBIGUOUS: "AMBIGUOUS",
    FixStatus.ERROR: "ERROR",
}


def _ref_label(result) -> str:
    """Format a ref with its type prefix for display."""
    if result.ref_type == "tag":
        return f"[tag] {result.branch}"
    return result.branch


def print_report(report: ScanReport) -> None:
    """Print a human-readable report to stdout."""
    branch_results = [r for r in report.results if r.ref_type == "branch"]
    tag_results = [r for r in report.results if r.ref_type == "tag"]

    print()
    print("=" * 70)
    print("  BACKPORT SCANNER REPORT")
    print("=" * 70)
    print(f"  Repo:        {report.repo}")
    print(f"  Fix commit:  {report.fix_commit}")
    if report.fix_description:
        print(f"  Description: {report.fix_description}")
    parts = []
    if branch_results:
        parts.append(f"{len(branch_results)} branch(es)")
    if tag_results:
        parts.append(f"{len(tag_results)} tag(s)")
    print(f"  Refs scanned: {' + '.join(parts) if parts else len(report.results)}")
    print("=" * 70)
    print()

    col_w = max((len(_ref_label(r)) for r in report.results), default=20) + 2
    print(f"  {'REF':<{col_w}}  STATUS")
    print(f"  {'-'*col_w}  {'-'*40}")

    for result in sorted(report.results, key=lambda r: (r.ref_type, r.branch)):
        icon = STATUS_ICON.get(result.status, result.status.value)
        label = _ref_label(result)
        print(f"  {label:<{col_w}}  {icon}")
        if result.detail:
            print(f"  {'':<{col_w}}  → {result.detail}")

    print()
    print("-" * 70)
    print(f"  NEEDS FIX    : {len(report.needs_fix)} ref(s)")
    print(f"  ALREADY FIXED: {len(report.already_fixed)} ref(s)")
    if report.not_affected:
        print(f"  NOT AFFECTED : {len(report.not_affected)} ref(s) — predates the bug")
    if report.ambiguous:
        print(f"  AMBIGUOUS    : {len(report.ambiguous)} ref(s) — review manually")
    if report.errors:
        print(f"  ERRORS       : {len(report.errors)} ref(s)")
    print("=" * 70)

    if report.needs_fix:
        print()
        print("  Refs that NEED the fix:")
        for r in report.needs_fix:
            print(f"    - {_ref_label(r)}")
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
        "refs": [
            {
                "name": r.branch,
                "ref_type": r.ref_type,
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
