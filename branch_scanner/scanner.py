"""
Main branch scanner orchestrator.

For each branch in scope:
  1. Run git-based checks (exact, cherry, file diff)
  2. Optionally run LLM semantic check for AMBIGUOUS results
  3. Collect results into a ScanReport
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .triage import BranchCheckResult, FixStatus, check_branch
from .git_utils import get_remote_branches

log = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    repo_path: Path
    fix_commit: str
    branches: list[str] = field(default_factory=list)
    branch_pattern: str | None = None
    use_llm: bool = False          # enable LLM for AMBIGUOUS cases
    llm_model: str = "claude-opus-4-6"
    fix_description: str = ""
    vuln_commit: str | None = None  # commit that introduced the bug (enables NOT_AFFECTED detection)


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

    log.info("Auto-discovering branches with pattern: %s", config.branch_pattern)
    discovered = get_remote_branches(config.repo_path, pattern=config.branch_pattern)
    log.info("Discovered %d branches: %s", len(discovered), discovered)
    return discovered


def _make_anthropic_client():
    """
    Create an Anthropic client, using Vertex AI if CLAUDE_CODE_USE_VERTEX is set,
    otherwise fall back to a direct API key.
    """
    import os
    if os.environ.get("CLAUDE_CODE_USE_VERTEX"):
        from anthropic import AnthropicVertex
        project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "")
        region = os.environ.get("CLOUD_ML_REGION", "us-east5")
        return AnthropicVertex(project_id=project, region=region)
    else:
        import anthropic
        return anthropic.Anthropic()


def _llm_check(result: BranchCheckResult, config: ScanConfig) -> BranchCheckResult:
    """
    Use an LLM to resolve AMBIGUOUS cases.
    Reads changed files on both source commit and target branch,
    asks the LLM whether the fix is present.
    """
    try:
        import subprocess

        client = _make_anthropic_client()

        # Get the fix diff
        fix_diff = subprocess.run(
            ["git", "show", "--unified=5", config.fix_commit, "--", *result.changed_files],
            cwd=config.repo_path, capture_output=True, text=True
        ).stdout[:8000]

        # Extract a focused patch-context window from each changed file on the branch.
        # Instead of the raw file (can be huge), we use `git diff` to surface the
        # lines that the original commit touched, then `git show` those hunks on the
        # target branch via `grep -n` around the changed symbols.
        branch_content = ""
        for f in result.changed_files[:3]:
            # Pull the lines changed by the fix so we know what to look for
            diff_lines = subprocess.run(
                ["git", "show", "--unified=0", config.fix_commit, "--", f],
                cwd=config.repo_path, capture_output=True, text=True
            ).stdout

            # Extract added/removed lines to use as search anchors
            search_terms = []
            for line in diff_lines.splitlines():
                stripped = line[1:].strip()
                if (line.startswith("+") or line.startswith("-")) and len(stripped) > 8:
                    search_terms.append(stripped[:80])
                if len(search_terms) >= 5:
                    break

            # Get the file on the target branch and extract relevant sections
            full_file = subprocess.run(
                ["git", "show", f"origin/{result.branch}:{f}"],
                cwd=config.repo_path, capture_output=True, text=True
            ).stdout

            if search_terms and full_file:
                # Find lines in the file that are near the changed code
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

                # Deduplicate while preserving order
                seen: set[int] = set()
                unique_lines = []
                for lineno, content in relevant_lines:
                    if lineno not in seen:
                        seen.add(lineno)
                        unique_lines.append(f"{lineno:5d}: {content}")

                section = "\n".join(unique_lines[:150])  # cap at 150 lines
            else:
                section = full_file[:6000]

            branch_content += f"\n\n--- {f} on {result.branch} (relevant sections) ---\n{section}"

        prompt = f"""You are a code review expert. I need to determine if a bug fix has already been applied to a branch.

FIX COMMIT DIFF (what the fix adds/removes):
{fix_diff}

CURRENT STATE OF CHANGED FILES ON BRANCH '{result.branch}':
{branch_content}

Question: Is the fix already applied on branch '{result.branch}'?
Answer with exactly one of:
- FIXED: The fix or its equivalent is already present
- NEEDS_FIX: The fix is NOT present and needs to be applied
- UNCERTAIN: Cannot determine from the code alone

Provide a one-sentence explanation after your answer."""

        response = client.messages.create(
            model=config.llm_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()
        log.info("LLM answer for %s: %s", result.branch, answer[:120])

        if "NEEDS_FIX" in answer.upper():
            new_status = FixStatus.NEEDS_FIX
        elif "FIXED" in answer.upper():
            new_status = FixStatus.LIKELY_PRESENT
        else:
            new_status = FixStatus.AMBIGUOUS

        return BranchCheckResult(
            branch=result.branch,
            status=new_status,
            method="LLM semantic check",
            detail=answer,
            changed_files=result.changed_files,
        )

    except Exception as e:
        log.warning("LLM check failed for %s: %s", result.branch, e)
        return result


def run_scan(config: ScanConfig) -> ScanReport:
    """
    Main entry point. Scans all branches and returns a ScanReport.
    """
    branches = resolve_branches(config)
    if not branches:
        log.error("No branches found to scan.")
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

    for branch in branches:
        log.info("Checking branch: %s", branch)
        result = check_branch(
            config.repo_path, branch, config.fix_commit,
            vuln_commit=config.vuln_commit,
        )

        # Optionally resolve ambiguous cases with LLM
        if result.status == FixStatus.AMBIGUOUS and config.use_llm:
            log.info("Running LLM check for ambiguous result on %s", branch)
            result = _llm_check(result, config)

        log.info("  %s → %s (%s)", branch, result.status.value, result.method)
        report.results.append(result)

    return report
