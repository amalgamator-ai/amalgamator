"""External resolver context (hybrid synthmerge→amalgamator pipeline).

A case configuration may point at a Markdown file that carries context
produced by an *external* conflict resolver (e.g. synthmerge) plus the
findings of deterministic post-resolution checks (e.g. dead-symbol and
undefined-symbol scans).  When present and non-empty, that context is
injected into selected pipeline prompts:

* Phase 0.5 (semantic triage) — informs the complexity assessment and
  the priority-file selection;
* Phase 3a (narrow resolution) — the agent is told to reproduce the
  prior resolver's resolutions verbatim for regions the checks cleared
  ("trusted"), and to re-derive only the flagged ("suspect") regions;
* Phase 3a.1 (post-resolution verification) — the findings are added as
  evidence; the verification scope is NEVER reduced: everything is
  verified, including the regions marked trusted.

The Markdown format is intentionally simple and human-writable::

    # External Resolver Context

    ## Metadata
    - resolver: synthmerge 0.3.15
    - endpoint: Turnstile DeepSeek V4 Flash
    - command: synthmerge -c ... --vibe --continue
    - conflicts: 26 across 11 files
    - retries: 7 relocation retries, 1 extra-conflict-lines escalation

    ## Suspect findings
    Findings from the deterministic checks that triggered this escalation.

    - fs/xfs/libxfs/xfs_defer.h:68: symbol 'dop_inodes' declared but has zero references in tree
    - mm/memory-failure.c:1224: symbol 'num_poisoned_pages_dec' used but never defined

    ## Resolved files
    Final per-file patches from the external resolver. Files listed here
    passed all deterministic checks unless also named above.

    ### fs/xfs/libxfs/xfs_attr.c
    ```diff
    (unified diff or resolved content for the file)
    ```

Only the existence, non-emptiness and the two structured sections
("Suspect findings" bullets, "Resolved files" per-file fenced blocks)
are machine-parsed; everything else is passed through verbatim to the
prompts.  An absent, empty, or unparseable file disables the feature
silently — the pipeline then behaves exactly as before.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FINDING_RE = re.compile(r"^\s*-\s+(.+?)\s*$", re.MULTILINE)
_FILE_HEADER_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_FINDING_TAG_RE = re.compile(
    r"^\[(strong|advisory)\]\[([a-z_]+)\]\[(observed|inferred)/(high|medium|low)\]\s+(.*)$"
)


def _without_fenced_blocks(text: str) -> str:
    """Drop fenced code blocks before scanning for bullet findings.

    Finding prose legitimately contains diffs and examples; a ``-`` line
    inside a fence is content, not a finding.  An unclosed fence drops the
    remainder (there is no reliable way to tell content from findings after
    it).
    """
    out: list[str] = []
    pos = 0
    while True:
        start = text.find("```", pos)
        if start < 0:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:start])
        end = text.find("```", start + 3)
        if end < 0:
            return "".join(out)
        pos = end + 3


@dataclass
class FindingDetail:
    """One parsed finding, with its class/severity/confidence/evidence.

    Untagged (human-written) bullets are treated as advisory inferences:
    the safe default when the producer did not state a confidence.
    """

    severity: str = "advisory"
    category: str = "unclassified"
    confidence: str = "low"
    evidence: str = "inferred"
    message: str = ""
    raw: str = ""


@dataclass
class ExternalContext:
    """Parsed external-resolver context, or the empty (no-op) variant.

    This is the strategy object handed to the prompt renderers: the
    ``None``-like empty instance renders empty sections for every phase,
    so prompts and pipeline code never need ``if external_context`` checks
    of their own beyond passing the rendered strings through.
    """

    source_path: str = ""
    raw: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    finding_details: list[FindingDetail] = field(default_factory=list)
    # "parsed" when the context carried a Suspect findings section (zero
    # findings then means the checks ran and reported nothing);
    # "unavailable" when no usable check result was present at all.
    findings_status: str = "unavailable"
    # file path -> extracted fenced block (resolved content/patch)
    resolved_files: dict[str, str] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        """True when any usable content was loaded."""
        return bool(self.raw.strip())

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    @property
    def has_resolved_files(self) -> bool:
        return bool(self.resolved_files)

    # ------------------------------------------------------------------
    # Section renderers (one per pipeline phase).  Each returns "" when
    # there is nothing to inject — the conditional-section strategy the
    # prompt templates already use for triage/evidence sections.
    # ------------------------------------------------------------------

    def render_triage_section(self) -> str:
        """Phase 0.5 — semantic triage context."""
        if not self.active:
            return ""
        summary = self._metadata_block()
        return (
            "<external_resolver_context>\n"
            "A prior automated resolver already attempted this backport. "
            "Its report follows. Use it to refine your assessment: files "
            "it resolved without check findings are likely low-risk; files "
            "named in the findings need the most attention in priority "
            "selection.\n"
            f"{summary}"
            f"{self._findings_block()}"
            "</external_resolver_context>\n\n"
        )

    def render_resolution_section(self, files: list[str] | None = None) -> str:
        """Phase 3a — narrow resolution context.

        The trusted/suspect contract lives here: reproduce trusted
        resolutions verbatim, re-derive only suspect regions.

        *files* restricts the embedded prior-resolution patches to the
        files the current resolution request is about (a chunk or the
        whole conflicted set).  When omitted, every resolved file is
        embedded.
        """
        if not self.active:
            return ""
        parts = [
            "<external_resolver_context>",
            "A prior automated resolver (see metadata) already resolved "
            "this cherry-pick. Its output was then checked by "
            "deterministic post-resolution scans (git grep based; no "
            "build). Treat its work as a starting point, not as ground "
            "truth:",
            "",
            "1. For conflicts in files listed under RESOLVED FILES and "
            "NOT named in the SUSPECT FINDINGS: reproduce the prior "
            "resolution verbatim — do not re-derive, improve, or reformat "
            "it. Splice, never retype. The exact prior patch bytes for "
            "each such file are embedded below; use them as the source.",
            "2. For conflicts in files (or at lines) named in the SUSPECT "
            "FINDINGS: verify the finding first, then resolve from the fix "
            "diff using it as a hint of what may have gone wrong (e.g. a "
            "symbol it left undefined or dead). Findings are heuristics, "
            "not unconditional prohibitions: if the code shows the finding "
            "is a false positive, keep the prior resolution and say why.",
            "3. The prior resolver's decisions about which upstream "
            "changes to keep are NOT authoritative: where its output "
            "conflicts with the three-way discipline above, the "
            "discipline wins.",
            "",
        ]
        parts.append(self._metadata_block())
        parts.append(self._findings_block())
        parts.append(self._resolved_files_block(files))
        parts.append("</external_resolver_context>\n\n")
        return "".join(parts)

    def render_review_section(self) -> str:
        """Phase 3a.2 — triage-blind independent review context.

        Scoped injection: findings + metadata only. The trusted resolutions
        are deliberately withheld so the reviewer stays independent of the
        resolution agent's choices — it sees the mechanical evidence, not the
        prior resolver's code. Two mandates are added:

        1. Symbols flagged dead/undefined by the deterministic scans must not
           be reintroduced by review corrections.
        2. A wholesale revert of the port is out of mandate and necessarily
           wrong here: it would restore the baseline state the checks
           flagged (or undo a port that passed them).
        """
        if not self.active:
            return ""
        return (
            "<external_resolver_context>\n"
            "A prior automated resolver produced the initial resolution of "
            "this backport, and deterministic post-resolution scans (git "
            "grep based, no LLM) ran over its output; the resolution agent "
            "then re-derived the flagged regions. You have NOT been given "
            "the prior resolver's resolutions — review independently. The "
            "scan findings below are automated heuristics, not ground "
            "truth: strong findings are high-confidence objective breakage "
            "classes, advisory findings are correlation signals. Verify "
            "each one against the code before acting; a finding may be a "
            "false positive (macros, split-line definitions, generated "
            "files, Kbuild composites).\n"
            f"{self._metadata_block()}"
            f"{self._findings_block()}"
            "Review constraints:\n"
            "- Do NOT reintroduce a symbol flagged as dead or undefined "
            "unless you can show the finding is a false positive; explain "
            "why in your final message.\n"
            "- Your mandate is symbol-adaptation errors, not port-scope "
            "decisions. A wholesale revert of the port (restoring files to "
            "the baseline) is out of mandate: if you believe the port's "
            "overall direction is wrong, report that in your final message "
            "and change nothing.\n"
            "</external_resolver_context>\n\n"
        )

    def render_verification_section(self) -> str:
        """Phase 3a.1 — post-resolution verification context.

        The findings are injected as evidence, but the verification scope
        is never reduced: trusted regions are verified too.
        """
        if not self.active:
            return ""
        return (
            "<external_resolver_context>\n"
            "A prior automated resolver produced an initial resolution of "
            "this backport (metadata below) which was partially re-derived "
            "by the resolution agent. Deterministic post-resolution scans "
            "produced the findings below — treat them as evidence on par "
            "with the other evidence in this prompt, and verify each one "
            "is now resolved. Verification scope is NOT reduced by the "
            "prior resolver's work: verify every hunk, including regions "
            "it originally produced.\n"
            f"{self._metadata_block()}"
            f"{self._findings_block()}"
            "</external_resolver_context>\n\n"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _metadata_block(self) -> str:
        if not self.metadata:
            return ""
        lines = "".join(f"- {k}: {v}\n" for k, v in self.metadata.items())
        return f"Resolver metadata:\n{lines}\n"

    def _findings_block(self) -> str:
        if self.findings_status != "parsed":
            return (
                "SUSPECT FINDINGS: unavailable — the external context did "
                "not contain a usable check result. Do not assume the "
                "deterministic checks passed.\n\n"
            )
        if not self.findings:
            return (
                "SUSPECT FINDINGS: none reported — the deterministic checks "
                "ran and reported zero findings.\n\n"
            )
        strong = [d for d in self.finding_details if d.severity == "strong"]
        advisory = [d for d in self.finding_details if d.severity != "strong"]
        lines = []
        if strong:
            lines.append(
                "STRONG FINDINGS (high-confidence objective breakage classes; "
                "still verify each against the tree before acting):"
            )
            lines.extend(f"- [{d.category}] {d.message}" for d in strong)
            lines.append("")
        if advisory:
            lines.append(
                "ADVISORY FINDINGS (heuristic correlation signals — evidence "
                "to verify, not defects by themselves):"
            )
            lines.extend(f"- [{d.category}] {d.message}" for d in advisory)
            lines.append("")
        if not lines:
            lines.append(
                "SUSPECT FINDINGS: none reported — the deterministic checks "
                "ran and reported zero findings."
            )
        return "SUSPECT FINDINGS (deterministic checks):\n" + "\n".join(lines) + "\n"

    def _resolved_files_block(self, files: list[str] | None = None) -> str:
        if not self.resolved_files:
            return ""
        selected = (
            [p for p in files if p in self.resolved_files]
            if files is not None
            else sorted(self.resolved_files)
        )
        if not selected:
            return (
                "RESOLVED FILES: no prior-resolver patch applies to the "
                "conflicted files in this request.\n\n"
            )
        chunks = [
            "RESOLVED FILES (prior resolver output, verbatim unless named "
            "in the findings). Each fenced block is the exact saved patch "
            "for that file — use its added lines as the source bytes and "
            "splice them into the conflicted file (do not run git apply on "
            "a conflicted worktree):"
        ]
        for path in sorted(selected):
            patch = self.resolved_files[path].rstrip("\n")
            chunks.append(f"### {path}\n```diff\n{patch}\n```")
        return "\n\n".join(chunks) + "\n\n"


def _split_sections(raw: str) -> dict[str, str]:
    """Split the Markdown into its ``##`` sections."""
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(raw))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections[m.group(1).strip().lower()] = raw[start:end].strip()
    return sections


def _parse_metadata(section: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in section.splitlines():
        m = re.match(r"^-\s*(.+?):\s*(.+?)\s*$", line)
        if m:
            metadata[m.group(1).strip().lower()] = m.group(2).strip()
    return metadata


def _parse_findings(section: str) -> list[str]:
    return [m.group(1) for m in _FINDING_RE.finditer(_without_fenced_blocks(section))]


def _parse_finding_details(raw_findings: list[str]) -> list[FindingDetail]:
    details = []
    for raw in raw_findings:
        m = _FINDING_TAG_RE.match(raw)
        if m:
            details.append(
                FindingDetail(
                    severity=m.group(1),
                    category=m.group(2),
                    evidence=m.group(3),
                    confidence=m.group(4),
                    message=m.group(5),
                    raw=raw,
                )
            )
        else:
            details.append(
                FindingDetail(
                    severity="advisory",
                    category="unclassified",
                    confidence="low",
                    evidence="inferred",
                    message=raw,
                    raw=raw,
                )
            )
    return details


def _parse_resolved_files(section: str) -> tuple[dict[str, str], list[str]]:
    """Parse ``### <path>`` headers each followed by a fenced block.

    Returns the parsed files and any parse errors.  An unclosed fence is
    reported explicitly instead of silently dropping the file.
    """
    resolved: dict[str, str] = {}
    errors: list[str] = []
    headers = list(_FILE_HEADER_RE.finditer(section))
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(section)
        body = section[start:end]
        fence = _FENCE_RE.search(body)
        path = h.group(1).strip()
        if fence and path:
            resolved[path] = fence.group(1).rstrip("\n")
        elif path and "```" in body:
            errors.append(f"unclosed fenced block for {path!r} in '## Resolved files'")
    return resolved, errors


def load_external_context(path: str | None) -> ExternalContext:
    """Load and parse an external-context Markdown file.

    Returns an empty (inactive) :class:`ExternalContext` when *path* is
    falsy, the file is missing, or it is empty.  Parse issues never raise:
    they are recorded in ``parse_errors`` and the affected section is
    skipped, so a malformed context file can only degrade to "no context",
    never break a run.
    """
    if not path:
        return ExternalContext()
    expanded = os.path.expanduser(os.path.expandvars(path))
    try:
        with open(expanded, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError as e:
        log.warning("external context file unreadable (%s): %s", expanded, e)
        return ExternalContext(source_path=expanded)

    ctx = ExternalContext(source_path=expanded, raw=raw)
    if not raw.strip():
        return ctx

    sections = _split_sections(raw)
    unknown = set(sections) - {"metadata", "suspect findings", "resolved files"}
    if unknown:
        ctx.parse_errors.append(
            "unknown sections ignored: " + ", ".join(sorted(unknown))
        )

    if "metadata" in sections:
        ctx.metadata = _parse_metadata(sections["metadata"])
    if "suspect findings" in sections:
        ctx.findings = _parse_findings(sections["suspect findings"])
        ctx.finding_details = _parse_finding_details(ctx.findings)
        # A heading alone is not proof the checks ran.  Zero findings are
        # only "parsed zero" when the section carries check provenance
        # (the emitter always writes `checker:` / `findings-count:`) or
        # explicit findings.
        has_provenance = any(k in ctx.metadata for k in ("checker", "findings-count"))
        if ctx.findings or has_provenance:
            ctx.findings_status = "parsed"
        else:
            ctx.parse_errors.append(
                "findings section present but no findings and no check "
                "provenance — check result unavailable"
            )
    else:
        ctx.parse_errors.append(
            "missing '## Suspect findings' section — check result unavailable"
        )
    if "resolved files" in sections:
        ctx.resolved_files, resolved_errors = _parse_resolved_files(
            sections["resolved files"]
        )
        ctx.parse_errors.extend(resolved_errors)

    # A context with neither findings nor resolutions is still useful as
    # raw context, but note it so operators can spot a mis-formatted file.
    if not ctx.findings and not ctx.resolved_files and not ctx.metadata:
        ctx.parse_errors.append(
            "no recognizable sections (expected '## Metadata', "
            "'## Suspect findings', '## Resolved files')"
        )
    for err in ctx.parse_errors:
        log.warning("external context: %s", err)
    return ctx
