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
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)


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

    def render_resolution_section(self) -> str:
        """Phase 3a — narrow resolution context.

        The trusted/suspect contract lives here: reproduce trusted
        resolutions verbatim, re-derive only suspect regions.
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
            "it. Splice, never retype.",
            "2. For conflicts in files (or at lines) named in the SUSPECT "
            "FINDINGS: disregard the prior resolution there and resolve "
            "from the fix diff, using the finding as a hint of what went "
            "wrong (e.g. a symbol it left undefined or dead).",
            "3. The prior resolver's decisions about which upstream "
            "changes to keep are NOT authoritative: where its output "
            "conflicts with the three-way discipline above, the "
            "discipline wins.",
            "",
        ]
        parts.append(self._metadata_block())
        parts.append(self._findings_block())
        parts.append(self._resolved_files_block())
        parts.append("</external_resolver_context>\n\n")
        return "".join(parts)

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
        if not self.findings:
            return "SUSPECT FINDINGS: none — all deterministic checks passed.\n\n"
        lines = "".join(f"- {f}\n" for f in self.findings)
        return f"SUSPECT FINDINGS (deterministic checks):\n{lines}\n"

    def _resolved_files_block(self) -> str:
        if not self.resolved_files:
            return ""
        lines = "\n".join(f"- {p}" for p in sorted(self.resolved_files))
        return (
            "RESOLVED FILES (prior resolver output, verbatim unless named "
            f"in the findings):\n{lines}\n\n"
        )


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
    return [m.group(1) for m in _FINDING_RE.finditer(section)]


def _parse_resolved_files(section: str) -> dict[str, str]:
    """Parse ``### <path>`` headers each followed by a fenced block."""
    resolved: dict[str, str] = {}
    headers = list(_FILE_HEADER_RE.finditer(section))
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(section)
        body = section[start:end]
        fence = _FENCE_RE.search(body)
        path = h.group(1).strip()
        if fence and path:
            resolved[path] = fence.group(1).rstrip("\n")
    return resolved


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
    if "resolved files" in sections:
        ctx.resolved_files = _parse_resolved_files(sections["resolved files"])

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
