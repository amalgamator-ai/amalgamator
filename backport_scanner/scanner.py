"""
Main scanner orchestrator — scans branches AND tags.

For each ref (branch or tag) in scope:
  1. Run git-based checks (exact, cherry, file diff)
  2. Optionally run LLM semantic check for AMBIGUOUS results
  3. Collect results into a ScanReport
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from common.model_registry import ModelProfile

from .triage import (
    BranchCheckResult, FixStatus, check_ref,
    get_remote_branches, get_tags, _run, _resolve_ref,
)

log = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    repo_path: Path
    fix_commit: str
    branches: list[str] = field(default_factory=list)
    branch_pattern: str | None = None
    tags: list[str] = field(default_factory=list)
    tag_pattern: str | None = None
    llm_model: str = "claude-opus-4-6"
    fix_description: str = ""
    vuln_commit: str | None = None
    discover_commands: bool = False
    profile: ModelProfile | None = field(default=None, repr=False)
    model_ref: str | None = None
    models_path: Path | None = None

    def __post_init__(self):
        if self.profile is not None:
            self.llm_model = self.profile.model
            if self.model_ref is None:
                self.model_ref = self.profile.id


@dataclass
class ScanReport:
    repo: str
    fix_commit: str
    fix_description: str
    results: list[BranchCheckResult] = field(default_factory=list)

    @property
    def needs_fix(self) -> list[BranchCheckResult]:
        return [r for r in self.results if r.status == FixStatus.NEEDS_FIX]

    @property
    def already_fixed(self) -> list[BranchCheckResult]:
        return [r for r in self.results if r.status in (
            FixStatus.EXACT_MATCH, FixStatus.EQUIVALENT, FixStatus.LIKELY_PRESENT
        )]

    @property
    def not_affected(self) -> list[BranchCheckResult]:
        return [r for r in self.results if r.status == FixStatus.NOT_AFFECTED]

    @property
    def ambiguous(self) -> list[BranchCheckResult]:
        return [r for r in self.results if r.status == FixStatus.AMBIGUOUS]

    @property
    def errors(self) -> list[BranchCheckResult]:
        return [r for r in self.results if r.status == FixStatus.ERROR]


def resolve_branches(config: ScanConfig) -> list[str]:
    """Determine which branches to scan."""
    if config.branches:
        log.info("Using explicit branch list (%d branches)", len(config.branches))
        return config.branches

    if config.branch_pattern:
        log.info("Auto-discovering branches with pattern: %s", config.branch_pattern)
        discovered = get_remote_branches(config.repo_path, pattern=config.branch_pattern)
        log.info("Discovered %d branches: %s", len(discovered), discovered)
        return discovered

    return []


def resolve_tags(config: ScanConfig) -> list[str]:
    """Determine which tags to scan."""
    if config.tags:
        log.info("Using explicit tag list (%d tags)", len(config.tags))
        return config.tags

    if config.tag_pattern:
        log.info("Auto-discovering tags with pattern: %s", config.tag_pattern)
        discovered = get_tags(config.repo_path, pattern=config.tag_pattern)
        log.info("Discovered %d tags: %s", len(discovered), discovered)
        return discovered

    return []




def _llm_check(result: BranchCheckResult, config: ScanConfig) -> BranchCheckResult:
    """
    Use an LLM to resolve AMBIGUOUS cases.
    Reads changed files on both source commit and target ref,
    asks the LLM whether the fix is present.
    """
    try:
        from common.llm_client import completion as llm_completion

        ref = _resolve_ref(result.branch, result.ref_type)
        ref_label = f"tag '{result.branch}'" if result.ref_type == "tag" else f"branch '{result.branch}'"
        ref_type_upper = "TAG" if result.ref_type == "tag" else "BRANCH"

        _, fix_diff, _ = _run(
            ["git", "show", "--unified=5", config.fix_commit, "--", *result.changed_files],
            cwd=config.repo_path,
        )
        fix_diff = fix_diff[:8000]

        ref_content = ""
        for f in result.changed_files[:3]:
            _, diff_lines, _ = _run(
                ["git", "show", "--unified=0", config.fix_commit, "--", f],
                cwd=config.repo_path,
            )

            search_terms = []
            for line in diff_lines.splitlines():
                stripped = line[1:].strip()
                if (line.startswith("+") or line.startswith("-")) and len(stripped) > 8:
                    search_terms.append(stripped[:80])
                if len(search_terms) >= 5:
                    break

            _, full_file, _ = _run(
                ["git", "show", f"{ref}:{f}"],
                cwd=config.repo_path,
            )

            if search_terms and full_file:
                lines = full_file.splitlines()
                relevant_lines: list[tuple[int, str]] = []
                for i, line in enumerate(lines):
                    for term in search_terms:
                        if term and term in line:
                            start = max(0, i - 10)
                            end = min(len(lines), i + 15)
                            for j in range(start, end):
                                relevant_lines.append((j + 1, lines[j]))
                            break

                seen: set[int] = set()
                unique_lines = []
                for lineno, content in relevant_lines:
                    if lineno not in seen:
                        seen.add(lineno)
                        unique_lines.append(f"{lineno:5d}: {content}")

                section = "\n".join(unique_lines[:150])
            else:
                section = full_file[:6000]

            ref_content += f"\n\n--- {f} on {result.branch} (relevant sections) ---\n{section}"

        prompt = f"""You are a code review expert. I need to determine if a bug fix has already been applied to a {ref_label}.

FIX COMMIT DIFF (what the fix adds/removes):
{fix_diff}

CURRENT STATE OF CHANGED FILES ON {ref_type_upper} '{result.branch}':
{ref_content}

Question: Is the fix already applied on {ref_label}?
Answer with exactly one of:
- FIXED: The fix or its equivalent is already present
- NEEDS_FIX: The fix is NOT present and needs to be applied
- UNCERTAIN: Cannot determine from the code alone

Provide a one-sentence explanation after your answer."""

        result_obj = llm_completion(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            profile=config.profile,
        )
        answer = result_obj.text.strip()
        log.info("LLM answer for %s: %s", result.branch, answer[:120])

        first_token = answer.split()[0].rstrip(":").upper() if answer.split() else ""
        if first_token == "NEEDS_FIX":
            new_status = FixStatus.NEEDS_FIX
        elif first_token == "FIXED":
            new_status = FixStatus.LIKELY_PRESENT
        else:
            new_status = FixStatus.AMBIGUOUS

        return BranchCheckResult(
            branch=result.branch,
            status=new_status,
            method="LLM semantic check",
            detail=answer,
            changed_files=result.changed_files,
            ref_type=result.ref_type,
        )

    except Exception as e:
        log.warning("LLM check failed for %s: %s", result.branch, e)
        return result


def run_scan(config: ScanConfig) -> ScanReport:
    """
    Main entry point. Scans all branches and tags, returns a ScanReport.
    """
    branches = resolve_branches(config)
    tags = resolve_tags(config)

    if not branches and not tags:
        log.error("No branches or tags found to scan.")
        return ScanReport(
            repo=str(config.repo_path),
            fix_commit=config.fix_commit,
            fix_description=config.fix_description,
        )

    report = ScanReport(
        repo=str(config.repo_path),
        fix_commit=config.fix_commit,
        fix_description=config.fix_description,
    )

    refs_to_scan = [(b, "branch") for b in branches] + [(t, "tag") for t in tags]

    for name, ref_type in refs_to_scan:
        log.info("Checking %s: %s", ref_type, name)
        result = check_ref(
            config.repo_path, name, config.fix_commit,
            vuln_commit=config.vuln_commit, ref_type=ref_type,
        )

        if result.status == FixStatus.AMBIGUOUS:
            log.info("Running LLM check for ambiguous result on %s %s", ref_type, name)
            result = _llm_check(result, config)

        log.info("  %s → %s (%s)", name, result.status.value, result.method)
        report.results.append(result)

    return report
