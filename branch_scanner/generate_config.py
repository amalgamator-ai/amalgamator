"""
generate_config.py — bridge between branch scanner and auto_bug_fix YAML configs.

Three modes (Alessandro's three cases from design meeting):

  generate  — scan branches → produce one auto_bug_fix YAML per NEEDS_FIX branch
  add       — scan branches → append newly-found NEEDS_FIX branches to existing YAML
  verify    — check branches listed in existing YAML still need the fix

Usage:
    python -m branch_scanner.generate_config --config scan.yaml --mode generate --output-dir ./out/
    python -m branch_scanner.generate_config --config scan.yaml --mode add     --existing-yaml fix.yaml
    python -m branch_scanner.generate_config --config scan.yaml --mode verify  --existing-yaml fix.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from auto_bug_fix.git_tools import git_show_files
from .triage import FixStatus, check_branch
from .run_scanner import load_config
from .scanner import ScanConfig, run_scan

log = logging.getLogger(__name__)


# ── YAML template ─────────────────────────────────────────────────────────────

def _make_bug_fix_yaml(
    repo_path: Path,
    fix_commit: str,
    fix_description: str,
    target_branch: str,
    changed_files: list[str],
) -> dict:
    """Produce an auto_bug_fix-compatible YAML dict for one target branch."""
    short = fix_commit[:10]
    safe_branch = target_branch.replace("/", "-").replace(".", "-")
    return {
        "issue_id": f"scan-{safe_branch}-{short}",
        "bug_description": fix_description or f"Backport fix {short} to {target_branch}",
        "repository": {
            "source_path": str(repo_path),
            "workdir": f"/tmp/bugfix-{safe_branch}-{short}",
        },
        "branches": {
            "source": "origin/main",
            "target": f"origin/{target_branch}",
        },
        "fix": {
            "commit": fix_commit,
        },
        "build": {
            "command": "",       # fill in manually — every project differs
            "test_command": "",  # fill in manually — every project differs
        },
        "model": "claude-opus-4-6",
        "_changed_files": changed_files,  # informational, not used by pipeline
    }


def _write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Remove private keys before writing
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    with open(path, "w") as f:
        yaml.dump(clean, f, default_flow_style=False, sort_keys=False)
    log.info("  Written: %s", path)


# ── Mode: generate ─────────────────────────────────────────────────────────────

def mode_generate(config: ScanConfig, output_dir: Path) -> None:
    """Scan branches and generate one YAML per NEEDS_FIX branch."""
    log.info("Mode: generate — scanning branches and writing YAMLs to %s", output_dir)

    report = run_scan(config)
    changed_files = git_show_files(str(config.repo_path), config.fix_commit)

    if not report.needs_fix:
        log.info("No branches need the fix — nothing to generate.")
        return

    for result in report.needs_fix:
        data = _make_bug_fix_yaml(
            repo_path=config.repo_path,
            fix_commit=config.fix_commit,
            fix_description=config.fix_description,
            target_branch=result.branch,
            changed_files=changed_files,
        )
        safe = result.branch.replace("/", "-").replace(".", "-")
        out_path = output_dir / f"{safe}.yaml"
        _write_yaml(data, out_path)

    print(f"\nGenerated {len(report.needs_fix)} YAML config(s) in {output_dir}/")
    print("Next steps:")
    print("  1. Fill in build.command and build.test_command in each YAML")
    print("  2. Run: python -m auto_bug_fix.run_bug_fix --config <yaml>")
    _print_summary(report)


# ── Mode: add ──────────────────────────────────────────────────────────────────

def mode_add(config: ScanConfig, existing_yaml: Path) -> None:
    """Scan branches and append new NEEDS_FIX branches to an existing YAML."""
    log.info("Mode: add — scanning for new affected branches")

    if not existing_yaml.exists():
        log.error("Existing YAML not found: %s", existing_yaml)
        sys.exit(1)

    with open(existing_yaml) as f:
        existing = yaml.safe_load(f)

    # Collect branches already in the existing YAML
    existing_target = existing.get("branches", {}).get("target", "")
    existing_branches: set[str] = set()
    if isinstance(existing_target, str):
        # single target — normalize (strip "origin/")
        existing_branches.add(existing_target.removeprefix("origin/"))
    elif isinstance(existing_target, list):
        existing_branches = {b.removeprefix("origin/") for b in existing_target}

    log.info("Existing YAML already covers: %s", sorted(existing_branches))

    report = run_scan(config)
    new_branches = [
        r for r in report.needs_fix
        if r.branch not in existing_branches
    ]

    if not new_branches:
        print("\nNo new branches to add — existing YAML is up to date.")
        _print_summary(report)
        return

    # Append new branches as additional target entries (list form)
    all_targets = sorted(existing_branches | {r.branch for r in new_branches})
    existing["branches"]["target"] = [f"origin/{b}" for b in all_targets]

    with open(existing_yaml, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    print(f"\nAdded {len(new_branches)} new branch(es) to {existing_yaml}:")
    for r in new_branches:
        print(f"  + {r.branch}")
    _print_summary(report)


# ── Mode: verify ───────────────────────────────────────────────────────────────

def mode_verify(config: ScanConfig, existing_yaml: Path) -> None:
    """Check each branch in an existing YAML to confirm it still needs the fix."""
    log.info("Mode: verify — checking branches in %s", existing_yaml)

    if not existing_yaml.exists():
        log.error("Existing YAML not found: %s", existing_yaml)
        sys.exit(1)

    with open(existing_yaml) as f:
        existing = yaml.safe_load(f)

    existing_target = existing.get("branches", {}).get("target", "")
    if isinstance(existing_target, str):
        branches_to_check = [existing_target.removeprefix("origin/")]
    elif isinstance(existing_target, list):
        branches_to_check = [b.removeprefix("origin/") for b in existing_target]
    else:
        log.error("Could not parse branches.target from YAML")
        sys.exit(1)

    fix_commit = existing.get("fix", {}).get("commit") or config.fix_commit
    log.info("Verifying %d branch(es) for commit %s", len(branches_to_check), fix_commit)

    still_needs: list[str] = []
    already_fixed: list[str] = []
    ambiguous: list[str] = []

    for branch in branches_to_check:
        result = check_branch(config.repo_path, branch, fix_commit)
        if result.status == FixStatus.NEEDS_FIX:
            still_needs.append(branch)
        elif result.status in (FixStatus.EXACT_MATCH, FixStatus.EQUIVALENT, FixStatus.LIKELY_PRESENT):
            already_fixed.append(branch)
        else:
            ambiguous.append(branch)
        log.info("  %s → %s", branch, result.status.value)

    print("\n" + "=" * 60)
    print("  VERIFY REPORT")
    print("=" * 60)
    print(f"  Config:      {existing_yaml}")
    print(f"  Fix commit:  {fix_commit}")
    print(f"  Checked:     {len(branches_to_check)} branch(es)")
    print()

    if still_needs:
        print(f"  ✗ Still needs fix ({len(still_needs)}):")
        for b in still_needs:
            print(f"      - {b}")
    if already_fixed:
        print(f"  ✓ Already fixed ({len(already_fixed)}) — safe to remove from YAML:")
        for b in already_fixed:
            print(f"      - {b}")
    if ambiguous:
        print(f"  ? Ambiguous ({len(ambiguous)}) — review manually:")
        for b in ambiguous:
            print(f"      - {b}")

    print("=" * 60)

    if already_fixed:
        print("\nTip: Remove already-fixed branches from the YAML to avoid redundant runs.")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _print_summary(report) -> None:
    print()
    print(f"  Scanned {len(report.results)} branch(es) total:")
    print(f"    NEEDS FIX    : {len(report.needs_fix)}")
    print(f"    ALREADY FIXED: {len(report.already_fixed)}")
    if report.ambiguous:
        print(f"    AMBIGUOUS    : {len(report.ambiguous)} (run with llm.enabled: true to resolve)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate or update auto_bug_fix YAML configs from branch scan results."
    )
    parser.add_argument("--config", "-c", required=True, type=Path,
                        help="Path to scan config YAML (same format as run_scanner)")
    parser.add_argument("--mode", "-m", required=True,
                        choices=["generate", "add", "verify"],
                        help="generate=create YAMLs from scratch, add=append branches, verify=check existing")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("./generated"),
                        help="Directory to write generated YAMLs (generate mode only)")
    parser.add_argument("--existing-yaml", "-e", type=Path,
                        help="Existing auto_bug_fix YAML to update (add/verify modes)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    if args.mode == "generate":
        mode_generate(config, args.output_dir)
    elif args.mode == "add":
        if not args.existing_yaml:
            parser.error("--existing-yaml is required for 'add' mode")
        mode_add(config, args.existing_yaml)
    elif args.mode == "verify":
        if not args.existing_yaml:
            parser.error("--existing-yaml is required for 'verify' mode")
        mode_verify(config, args.existing_yaml)


if __name__ == "__main__":
    main()
