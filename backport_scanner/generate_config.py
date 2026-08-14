"""
generate_config.py — bridge between backport scanner and auto_bug_fix YAML configs.

Three modes (Alessandro's three cases from design meeting):

  generate  — scan branches/tags → produce one auto_bug_fix YAML per NEEDS_FIX ref
  add       — scan → append newly-found NEEDS_FIX refs to existing YAML
  verify    — check refs listed in existing YAML still need the fix

Usage:
    python -m backport_scanner.generate_config --config scan.yaml --mode generate --output-dir ./out/
    python -m backport_scanner.generate_config --config scan.yaml --mode add     --existing-yaml fix.yaml
    python -m backport_scanner.generate_config --config scan.yaml --mode verify  --existing-yaml fix.yaml
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from auto_bug_fix.git_tools import git_show_files
from .triage import FixStatus, check_ref
from .run_scanner import load_config
from .scanner import ScanConfig, run_scan
from .report import print_report

log = logging.getLogger(__name__)


# ── Default branch discovery ───────────────────────────────────────────────────

def _get_default_branch(repo_path: Path) -> str:
    """
    Discover the default branch of the remote (e.g. main, master, develop).
    Uses `git symbolic-ref refs/remotes/origin/HEAD` with a fallback to 'main'.
    """
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "--short"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Returns e.g. "origin/main" — strip the "origin/" prefix
            return result.stdout.strip().removeprefix("origin/")
    except Exception:
        pass
    log.warning("Could not detect default branch, falling back to 'main'")
    return "main"


# ── Command discovery ─────────────────────────────────────────────────────────

# Files the LLM reads to understand how to build and test the project
_DISCOVERY_FILES = [
    "README.md", "README.rst", "README",
    "Makefile", "GNUmakefile",
    "go.mod",
    "pom.xml",
    "pyproject.toml", "setup.py", "tox.ini",
    "package.json",
    "Cargo.toml",
]
_DISCOVERY_CI_GLOB = ".github/workflows/*.yml"
_DISCOVERY_MAX_BYTES = 4000   # per file, to keep prompt manageable


def _discover_commands(repo_path: Path, model: str) -> tuple[str, str]:
    """
    Use an LLM to discover the build and test commands for a repository.

    Reads common project files (README, Makefile, CI workflows, build manifests)
    and asks Claude to identify the correct commands.

    Returns:
        (build_command, test_command) — empty strings if discovery fails.
    """
    from common.claude_utils import make_anthropic_client

    # Gather file contents
    sections: list[str] = []

    for fname in _DISCOVERY_FILES:
        fpath = repo_path / fname
        if fpath.exists():
            try:
                content = fpath.read_text(errors="replace")[:_DISCOVERY_MAX_BYTES]
                sections.append(f"--- {fname} ---\n{content}")
            except Exception:
                pass

    # Also read CI workflow files (up to 2)
    import glob as _glob
    ci_files = sorted(_glob.glob(str(repo_path / _DISCOVERY_CI_GLOB)))[:2]
    for ci_path in ci_files:
        try:
            content = Path(ci_path).read_text(errors="replace")[:_DISCOVERY_MAX_BYTES]
            rel = Path(ci_path).relative_to(repo_path)
            sections.append(f"--- {rel} ---\n{content}")
        except Exception:
            pass

    if not sections:
        log.warning("No project files found for command discovery in %s", repo_path)
        return "", ""

    context = "\n\n".join(sections)
    prompt = f"""You are a build system expert. Based on the project files below, identify the standard commands to build and test this project.

PROJECT FILES:
{context}

Respond with EXACTLY two lines, no explanation:
BUILD: <the build command>
TEST: <the test command>

If a command is not applicable or unclear, use an empty string for that line.
Examples of valid responses:
BUILD: go build ./...
TEST: go test ./...

BUILD: mvn install -DskipTests
TEST: mvn test

BUILD: pip install -e .
TEST: python -m pytest"""

    try:
        client = make_anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        log.info("Command discovery response: %s", text)

        build_cmd = ""
        test_cmd = ""
        for line in text.splitlines():
            if line.upper().startswith("BUILD:"):
                build_cmd = line.split(":", 1)[1].strip()
            elif line.upper().startswith("TEST:"):
                test_cmd = line.split(":", 1)[1].strip()
        return build_cmd, test_cmd

    except Exception as e:
        log.warning("Command discovery failed: %s", e)
        return "", ""


# ── YAML template ─────────────────────────────────────────────────────────────

def _parse_target_entry(entry) -> tuple[str, str, dict]:
    """Parse a target entry from branches.targets.

    Entries can be:
      - a plain string: "origin/release-1.33" or "refs/tags/v6.1.120"
      - a mapping with per-target overrides: {"branch": "origin/release-1.33", "issue_id": "..."}

    Returns (bare_name, ref_type, overrides_dict).
    """
    overrides: dict = {}
    if isinstance(entry, dict):
        raw = str(entry.get("branch", ""))
        overrides = {k: v for k, v in entry.items() if k != "branch"}
    else:
        raw = str(entry)

    if raw.startswith("refs/tags/"):
        return raw.removeprefix("refs/tags/"), "tag", overrides
    if raw.startswith("origin/"):
        return raw.removeprefix("origin/"), "branch", overrides
    return raw, "tag", overrides


def _format_target_ref(name: str, ref_type: str) -> str:
    """Format a bare ref name for YAML output."""
    if ref_type == "tag":
        return f"refs/tags/{name}"
    return f"origin/{name}"


def _safe_name(target_name: str, ref_type: str) -> str:
    """Compute a collision-resistant safe name with a prefix for tags.

    Appends a short hash of the original name so that refs whose names
    collapse under the / → - and . → - substitution (e.g. a/b.c vs a-b.c)
    still produce distinct filenames and workdirs.
    """
    import hashlib
    prefix = "tag-" if ref_type == "tag" else ""
    safe = target_name.replace("/", "-").replace(".", "-")
    short_hash = hashlib.sha256(target_name.encode()).hexdigest()[:6]
    return f"{prefix}{safe}-{short_hash}"


def _make_bug_fix_yaml(
    repo_path: Path,
    fix_commit: str,
    fix_description: str,
    target_name: str,
    changed_files: list[str],
    build_command: str = "",
    test_command: str = "",
    default_branch: str = "main",
    ref_type: str = "branch",
) -> dict:
    """Produce an auto_bug_fix-compatible YAML dict for one target ref."""
    short = fix_commit[:10]
    safe = _safe_name(target_name, ref_type)
    target_ref = _format_target_ref(target_name, ref_type)
    return {
        "issue_id": f"scan-{safe}-{short}",
        "bug_description": fix_description or f"Backport fix {short} to {target_name}",
        "repository": {
            "source_path": str(repo_path),
            "workdir": f"/tmp/bugfix-{safe}-{short}",
        },
        "branches": {
            "source": f"origin/{default_branch}",
            "target": target_ref,
        },
        "fix": {
            "commit": fix_commit,
        },
        "build": {
            "command": build_command,
            "test_command": test_command,
        },
        "model": "claude-opus-4-6",
        "changed_files": changed_files,
        "ref_type": ref_type,
    }


def _write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
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

    default_branch = _get_default_branch(config.repo_path)
    log.info("Default branch detected as: %s", default_branch)

    # Optionally discover build/test commands once for the whole repo
    build_cmd, test_cmd = "", ""
    if config.discover_commands:
        log.info("Discovering build/test commands via LLM...")
        build_cmd, test_cmd = _discover_commands(config.repo_path, config.llm_model)
        if build_cmd or test_cmd:
            log.info("Discovered — build: %r  test: %r", build_cmd, test_cmd)
        else:
            log.warning("Command discovery returned empty — leaving fields blank")

    for result in report.needs_fix:
        data = _make_bug_fix_yaml(
            repo_path=config.repo_path,
            fix_commit=config.fix_commit,
            fix_description=config.fix_description,
            target_name=result.branch,
            changed_files=changed_files,
            build_command=build_cmd,
            test_command=test_cmd,
            default_branch=default_branch,
            ref_type=result.ref_type,
        )
        out_path = output_dir / f"{_safe_name(result.branch, result.ref_type)}.yaml"
        _write_yaml(data, out_path)

    print(f"\nGenerated {len(report.needs_fix)} YAML config(s) in {output_dir}/")
    if build_cmd or test_cmd:
        print(f"  build.command:    {build_cmd or '(empty)'}")
        print(f"  build.test_command: {test_cmd or '(empty)'}")
        print("  Review these commands before running the pipeline.")
    else:
        print("  build.command and build.test_command are empty — fill in manually.")
    print("  Run: python -m auto_bug_fix.run_bug_fix --config <yaml>")
    print_report(report)


# ── Mode: add ──────────────────────────────────────────────────────────────────

def mode_add(config: ScanConfig, existing_yaml: Path) -> None:
    """Scan branches and append new NEEDS_FIX branches to an existing YAML."""
    log.info("Mode: add — scanning for new affected branches")

    if not existing_yaml.exists():
        log.error("Existing YAML not found: %s", existing_yaml)
        sys.exit(1)

    with open(existing_yaml) as f:
        existing = yaml.safe_load(f)

    # Collect refs already in the existing YAML (supports both singular
    # "target" and plural "targets" forms, and both string and mapping
    # entries).  Key by (ref_type, name) so a branch and tag with the
    # same bare name don't collide.
    branches_section = existing.get("branches", {})
    existing_target = branches_section.get("targets") or branches_section.get("target", "")
    # (ref_type, name) -> original entry (preserves per-target overrides)
    existing_refs: dict[tuple[str, str], object] = {}
    if isinstance(existing_target, str):
        name, rtype, _ = _parse_target_entry(existing_target)
        existing_refs[(rtype, name)] = existing_target
    elif isinstance(existing_target, list):
        for entry in existing_target:
            name, rtype, _ = _parse_target_entry(entry)
            existing_refs[(rtype, name)] = entry

    existing_names = {name for _, name in existing_refs}
    log.info("Existing YAML already covers: %s", sorted(existing_names))

    report = run_scan(config)
    new_branches = [
        r for r in report.needs_fix
        if (r.ref_type, r.branch) not in existing_refs
    ]

    if not new_branches:
        print("\nNo new branches to add — existing YAML is up to date.")
        print_report(report)
        return

    for r in new_branches:
        existing_refs[(r.ref_type, r.branch)] = _format_target_ref(r.branch, r.ref_type)
    all_keys = sorted(existing_refs.keys(), key=lambda k: k[1])
    existing["branches"]["targets"] = [existing_refs[k] for k in all_keys]
    existing["branches"].pop("target", None)

    with open(existing_yaml, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    print(f"\nAdded {len(new_branches)} new branch(es) to {existing_yaml}:")
    for r in new_branches:
        print(f"  + {r.branch}")
    print_report(report)


# ── Mode: verify ───────────────────────────────────────────────────────────────

_ALREADY_FIXED_STATUSES = frozenset({
    FixStatus.EXACT_MATCH, FixStatus.EQUIVALENT, FixStatus.LIKELY_PRESENT,
})


def mode_verify(config: ScanConfig, existing_yaml: Path) -> None:
    """Check each ref in an existing YAML to confirm it still needs the fix."""
    log.info("Mode: verify — checking refs in %s", existing_yaml)

    if not existing_yaml.exists():
        log.error("Existing YAML not found: %s", existing_yaml)
        sys.exit(1)

    with open(existing_yaml) as f:
        existing = yaml.safe_load(f)

    branches_section = existing.get("branches", {})
    existing_target = branches_section.get("targets") or branches_section.get("target", "")
    ref_type_map: dict[str, str] = {}
    if isinstance(existing_target, str):
        name, rtype, _ = _parse_target_entry(existing_target)
        ref_type_map[name] = rtype
    elif isinstance(existing_target, list):
        for entry in existing_target:
            name, rtype, _ = _parse_target_entry(entry)
            ref_type_map[name] = rtype
    else:
        log.error("Could not parse branches.target/targets from YAML")
        sys.exit(1)

    refs_to_check = list(ref_type_map.keys())
    fix_commit = existing.get("fix", {}).get("commit") or config.fix_commit
    log.info("Verifying %d ref(s) for commit %s", len(refs_to_check), fix_commit)

    still_needs: list[str] = []
    already_fixed: list[str] = []
    not_affected: list[str] = []
    ambiguous: list[str] = []

    for name in refs_to_check:
        result = check_ref(
            config.repo_path, name, fix_commit,
            vuln_commit=config.vuln_commit,
            ref_type=ref_type_map[name],
        )
        if result.status == FixStatus.NEEDS_FIX:
            still_needs.append(name)
        elif result.status in _ALREADY_FIXED_STATUSES:
            already_fixed.append(name)
        elif result.status == FixStatus.NOT_AFFECTED:
            not_affected.append(name)
        else:
            ambiguous.append(name)
        log.info("  %s → %s", name, result.status.value)

    print("\n" + "=" * 60)
    print("  VERIFY REPORT")
    print("=" * 60)
    print(f"  Config:      {existing_yaml}")
    print(f"  Fix commit:  {fix_commit}")
    print(f"  Checked:     {len(refs_to_check)} ref(s)")
    print()

    if still_needs:
        print(f"  ✗ Still needs fix ({len(still_needs)}):")
        for b in still_needs:
            print(f"      - {b}")
    if already_fixed:
        print(f"  ✓ Already fixed ({len(already_fixed)}) — safe to remove from YAML:")
        for b in already_fixed:
            print(f"      - {b}")
    if not_affected:
        print(f"  ~ Not affected ({len(not_affected)}) — ref predates the bug, may have been incorrectly included:")
        for b in not_affected:
            print(f"      - {b}")
    if ambiguous:
        print(f"  ? Ambiguous ({len(ambiguous)}) — review manually:")
        for b in ambiguous:
            print(f"      - {b}")

    print("=" * 60)

    if already_fixed:
        print("\nTip: Remove already-fixed refs from the YAML to avoid redundant runs.")
    if not_affected:
        print("\nTip: NOT_AFFECTED refs may have been incorrectly added — review and remove if so.")


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
