# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""External resolver context (hybrid backport pipeline).

A case configuration may point at a Markdown file that carries context
produced by an *external* conflict resolver (the reference implementation
is synthmerge) plus the findings of deterministic post-resolution checks
(e.g. dead-symbol and undefined-symbol scans).  When present and non-empty,
that context is injected into selected pipeline prompts:

* Phase 0.5 (semantic triage) — informs the complexity assessment and
  the priority-file selection;
* Phase 3a (narrow resolution) — when a usable check result is present, the
  agent is told to reproduce the prior resolver's resolutions verbatim for
  regions the checks cleared ("trusted"), and to re-derive only the flagged
  ("suspect") regions.  If the result is unavailable, there are no trusted
  regions and the agent re-derives from the fix diff;
* Phase 3a.1 (post-resolution verification) — the findings are added as
  evidence; the verification scope is NEVER reduced: everything is
  verified, including the regions marked trusted.

A findings list is only usable when its declared ``findings-count`` matches
the number of parsed bullets.  A missing count (including hand-authored
findings), a mismatch, an invalid count, a duplicated recognized section, or
``checker:`` provenance without a count makes the check result unavailable
rather than a clean zero-findings pass.

The Markdown format is intentionally simple and human-writable::

    # External Resolver Context

    ## Metadata
    - format: external-context/1
    - resolver: synthmerge 0.3.15
    - endpoint: Turnstile DeepSeek V4 Flash
    - command: synthmerge -c ... --vibe --continue
    - conflicts: 26 across 11 files
    - retries: 7 relocation retries, 1 extra-conflict-lines escalation
    - checker: check-backport.py
    - findings-count: 2

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

Only the three structured sections (``## Metadata``, ``## Suspect findings``
and ``## Resolved files``) are machine-parsed and rendered.  Free-form
commentary and unknown ``##`` sections are *not* injected into prompts;
unknown section names are logged so a mis-formatted emitter is visible.
An absent or empty file disables the feature silently, as does a non-empty
file that contains none of the three recognized sections — the pipeline
then behaves exactly as before.  Parse issues never raise: partially
malformed files degrade to the sections that did parse.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_HEADING_RES = {
    2: re.compile(r"^[ ]{0,3}##(?!#)[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$"),
    3: re.compile(r"^[ ]{0,3}###(?!#)[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$"),
}
# Findings are top-level bullets.  Indented bullets are detail lines under a
# finding, not separate findings.
_FINDING_RE = re.compile(r"^-[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
_FENCE_CLOSE_RE = re.compile(r"^[ ]{0,3}(?P<marker>`{3,}|~{3,})[ \t]*$")
_FINDING_TAG_RE = re.compile(
    r"^\[(strong|advisory)\]\[([a-z_]+)\]\[(observed|inferred)/(high|medium|low)\]\s+(.*)$"
)

# The Markdown contract is versioned so the producer (the hybrid adapter,
# or a hand-written file) and this parser can evolve independently.  Absent
# is accepted as the unversioned legacy format; an unsupported major version
# makes the check result unavailable instead of trusting it under v1 rules.
SUPPORTED_FORMAT_MAJOR = 1
_FORMAT_RE = re.compile(r"^external-context/(\d+)(?:\.(\d+))?$")


@dataclass(frozen=True)
class FencedBlock:
    """A fenced Markdown block and the offsets that delimit it."""

    start: int
    content_start: int
    content_end: int
    end: int
    marker: str
    closed: bool


def _iter_lines(text: str) -> Iterator[tuple[int, str, str]]:
    """Yield ``(offset, line_without_eol, raw_line)`` for every line."""
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        yield offset, raw_line.rstrip("\r\n"), raw_line
        offset += len(raw_line)


def _opening_fence(line: str) -> str | None:
    """Return the opening fence run (e.g. ``` or ````), or None."""
    m = _FENCE_OPEN_RE.match(line)
    return m.group("marker") if m else None


def _is_closing_fence(line: str, marker: str) -> bool:
    """True when *line* closes a block opened with *marker*."""
    m = _FENCE_CLOSE_RE.match(line)
    if not m:
        return False
    run = m.group("marker")
    return run[0] == marker[0] and len(run) >= len(marker)


def _iter_fenced_blocks(text: str) -> Iterator[FencedBlock]:
    """Walk fenced blocks using Markdown's fence-length semantics.

    A closing fence has the same character and at least the opening run's
    length, so a longer outer fence safely contains shorter inner fences.
    An unclosed fence is yielded with ``closed=False`` and consumes the
    remainder; callers report it instead of silently dropping content.
    """
    lines = list(_iter_lines(text))
    i = 0
    while i < len(lines):
        offset, line, raw_line = lines[i]
        marker = _opening_fence(line)
        if marker is None:
            i += 1
            continue
        content_start = offset + len(raw_line)
        j = i + 1
        while j < len(lines):
            close_offset, close_line, close_raw = lines[j]
            if _is_closing_fence(close_line, marker):
                yield FencedBlock(
                    start=offset,
                    content_start=content_start,
                    content_end=close_offset,
                    end=close_offset + len(close_raw),
                    marker=marker,
                    closed=True,
                )
                i = j + 1
                break
            j += 1
        else:
            yield FencedBlock(
                start=offset,
                content_start=content_start,
                content_end=len(text),
                end=len(text),
                marker=marker,
                closed=False,
            )
            return


def _strip_fenced_blocks(text: str) -> tuple[str, list[FencedBlock]]:
    """Return *text* without fenced blocks plus the blocks that were found."""
    blocks = list(_iter_fenced_blocks(text))
    out: list[str] = []
    pos = 0
    for block in blocks:
        out.append(text[pos : block.start])
        pos = block.end
    out.append(text[pos:])
    return "".join(out), blocks


def _iter_headings(text: str, level: int) -> Iterator[tuple[str, int, int]]:
    """Yield ``(title, heading_start, content_start)`` outside fenced blocks.

    Markdown headings inside a fenced diff/content block are data, not
    structure.  Up to three leading spaces and an optional closing ``##``
    sequence are accepted, matching normal Markdown rendering.
    """
    pattern = _HEADING_RES[level]
    blocks = list(_iter_fenced_blocks(text))
    block_index = 0
    for offset, line, raw_line in _iter_lines(text):
        while block_index < len(blocks) and offset >= blocks[block_index].end:
            block_index += 1
        if (
            block_index < len(blocks)
            and blocks[block_index].start <= offset < blocks[block_index].end
        ):
            continue
        m = pattern.match(line)
        if m:
            yield m.group(1).strip(), offset, offset + len(raw_line)


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
    # Retained for diagnostics only; the raw file text is never rendered.
    raw: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    finding_details: list[FindingDetail] = field(default_factory=list)
    # "parsed" when the context carried a Suspect findings section whose
    # declared count matches the parsed bullets (zero then means the checks
    # ran and reported nothing); "unavailable" when no usable, complete
    # check result was present.
    findings_status: str = "unavailable"
    # file path -> extracted fenced block (resolved content/patch)
    resolved_files: dict[str, str] = field(default_factory=dict)
    # Lower-cased names of the recognized structured sections that were
    # present.  Free-form prose and unknown sections do not activate the
    # strategy object (they are not injected, so activating on them would
    # only add a false "a prior resolver ran" claim).
    found_sections: set[str] = field(default_factory=set)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        """True when the file carried at least one recognized section."""
        return bool(self.raw.strip()) and bool(self.found_sections)

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
        if self.findings_status == "parsed":
            guidance = (
                "Use it to refine your assessment: files it resolved without "
                "check findings are likely low-risk; files named in the "
                "findings need the most attention in priority selection.\n"
            )
        else:
            guidance = (
                "The deterministic check result is unavailable, so no file "
                "can be treated as cleared: consider every prior-resolver "
                "output unverified and give the affected files attention.\n"
            )
        return (
            "<external_resolver_context>\n"
            "A prior automated resolver already attempted this backport. "
            "Its report follows. "
            f"{guidance}"
            f"{self._metadata_block()}"
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
        if self.findings_status == "parsed":
            intro = (
                "A prior automated resolver (see metadata) already resolved "
                "this cherry-pick. Its output was then checked by "
                "deterministic post-resolution scans (git grep based; no "
                "build). Treat its work as a starting point, not as ground "
                "truth:"
            )
        else:
            intro = (
                "A prior automated resolver (see metadata) already resolved "
                "this cherry-pick, but no usable deterministic check result "
                "is available for its output. Treat its work as unverified, "
                "not as ground truth:"
            )
        parts = ["<external_resolver_context>", intro, ""]
        if self.findings_status == "parsed":
            parts.extend(
                [
                    "1. For conflicts in files listed under RESOLVED FILES "
                    "and NOT named in the SUSPECT FINDINGS: reproduce the "
                    "prior resolution verbatim — do not re-derive, improve, "
                    "or reformat it. Splice, never retype. The exact prior "
                    "patch bytes for each such file are embedded below; use "
                    "them as the source.",
                    "2. For conflicts in files (or at lines) named in the "
                    "SUSPECT FINDINGS: verify the finding first, then "
                    "resolve from the fix diff using it as a hint of what "
                    "may have gone wrong (e.g. a symbol it left undefined "
                    "or dead). Findings are heuristics, not unconditional "
                    "prohibitions: if the code shows the finding is a false "
                    "positive, keep the prior resolution and say why.",
                ]
            )
        else:
            parts.extend(
                [
                    "1. NO TRUSTED REGIONS: the deterministic check result "
                    "is unavailable, so do not reproduce the prior "
                    "resolution verbatim. Resolve every conflict from the "
                    "fix diff using the three-way discipline in your "
                    "instructions; use the prior output below only as a hint "
                    "about what was attempted.",
                    "2. The prior resolver's output is unverified: re-derive "
                    "each file rather than splicing its bytes.",
                ]
            )
        parts.extend(
            [
                "3. The prior resolver's decisions about which upstream "
                "changes to keep are NOT authoritative: where its output "
                "conflicts with the three-way discipline in your "
                "instructions, the discipline wins.",
                "",
            ]
        )
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
        if self.findings_status == "parsed":
            context = (
                "A prior automated resolver produced the initial resolution "
                "of this backport, and deterministic post-resolution scans "
                "(git grep based, no LLM) ran over its output; the resolution "
                "agent then re-derived the flagged regions. You have NOT been "
                "given the prior resolver's resolutions — review "
                "independently. The scan findings below are automated "
                "heuristics, not ground truth: strong findings are "
                "high-confidence objective breakage classes, advisory "
                "findings are correlation signals. Verify each one against "
                "the code before acting; a finding may be a false positive "
                "(macros, split-line definitions, generated files, Kbuild "
                "composites).\n"
            )
        else:
            context = (
                "A prior automated resolver produced the initial resolution "
                "of this backport, but the deterministic check result is "
                "unavailable. You have NOT been given the prior resolver's "
                "resolutions — review independently, and do not assume the "
                "checks passed: the absence of usable findings is not "
                "evidence of cleanliness.\n"
            )
        return (
            "<external_resolver_context>\n"
            f"{context}"
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
        if self.findings_status == "parsed":
            context = (
                "A prior automated resolver produced an initial resolution "
                "of this backport (metadata below) which was partially "
                "re-derived by the resolution agent. Deterministic "
                "post-resolution scans produced the findings below — treat "
                "them as evidence on par with the other evidence in this "
                "prompt, and verify each one is now resolved. Verification "
                "scope is NOT reduced by the prior resolver's work: verify "
                "every hunk, including regions it originally produced.\n"
            )
        else:
            context = (
                "A prior automated resolver produced an initial resolution "
                "of this backport (metadata below), but the deterministic "
                "check result is unavailable. Do not assume the checks "
                "passed. Verification scope is NOT reduced by the prior "
                "resolver's work: verify every hunk.\n"
            )
        return (
            "<external_resolver_context>\n"
            f"{context}"
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
        if self.findings_status == "parsed":
            intro = (
                "RESOLVED FILES (prior resolver output, verbatim unless "
                "named in the findings). Each fenced block is the exact "
                "saved patch for that file — use its added lines as the "
                "source bytes and splice them into the conflicted file (do "
                "not run git apply on a conflicted worktree):"
            )
        else:
            intro = (
                "RESOLVED FILES (prior resolver output, UNVERIFIED — the "
                "check result is unavailable, so there are no trusted "
                "regions). These blocks are context from the prior attempt "
                "only; resolve from the fix diff and do not splice their "
                "bytes verbatim:"
            )
        chunks = [intro]
        for path in sorted(selected):
            patch = self.resolved_files[path].rstrip("\n")
            chunks.append(f"### {path}\n```diff\n{patch}\n```")
        return "\n\n".join(chunks) + "\n\n"


def _split_sections(raw: str) -> tuple[dict[str, str], list[str], set[str]]:
    """Split the Markdown into its top-level ``##`` sections.

    Headings inside fenced blocks (e.g. a resolved file that itself contains
    Markdown headings) are content, not section boundaries.  Duplicate
    section names are reported and returned; callers must treat the
    duplicated section as ambiguous rather than silently keeping the later
    block.
    """
    sections: dict[str, str] = {}
    errors: list[str] = []
    duplicates: set[str] = set()
    headings = list(_iter_headings(raw, 2))
    for i, (title, _, content_start) in enumerate(headings):
        end = headings[i + 1][1] if i + 1 < len(headings) else len(raw)
        key = title.lower()
        if key in sections:
            duplicates.add(key)
            errors.append(
                f"duplicate '## {title}' section — later content overrides earlier"
            )
        sections[key] = raw[content_start:end].strip()
    return sections, errors, duplicates


def _parse_metadata(section: str) -> tuple[dict[str, str], list[str]]:
    visible, blocks = _strip_fenced_blocks(section)
    errors = [
        "unclosed fenced block in '## Metadata'" for block in blocks if not block.closed
    ]
    metadata: dict[str, str] = {}
    for line in visible.splitlines():
        m = re.match(r"^-\s*(.+?):\s*(.+?)\s*$", line)
        if m:
            metadata[m.group(1).strip().lower()] = m.group(2).strip()
    return metadata, errors


def _parse_findings(section: str) -> tuple[list[str], list[str]]:
    visible, blocks = _strip_fenced_blocks(section)
    errors = [
        "unclosed fenced block in '## Suspect findings'"
        for block in blocks
        if not block.closed
    ]
    findings = [m.group(1) for m in _FINDING_RE.finditer(visible)]
    return findings, errors


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


def _extract_fenced_block(text: str) -> tuple[str | None, str | None]:
    """Extract the single fenced block from a resolved file's body.

    Returns ``(content, problem)``.  The body may carry prose before or after
    the block; that prose is ignored rather than merged into the patch.  More
    than one top-level block is ambiguous (an example plus the real patch),
    so it is reported instead of being concatenated.  An unclosed fence is
    also reported, never silently truncated.
    """
    blocks = list(_iter_fenced_blocks(text))
    if not blocks:
        return None, None
    if any(not block.closed for block in blocks):
        return None, "unclosed fenced block"
    if len(blocks) > 1:
        return None, "multiple fenced blocks (ambiguous file body)"
    block = blocks[0]
    return text[block.content_start : block.content_end], None


def _parse_resolved_files(section: str) -> tuple[dict[str, str], list[str]]:
    """Parse ``### <path>`` headers each followed by a fenced block.

    Headers inside fenced content are ignored.  Returns the parsed files and
    any parse errors; a missing or unclosed fence is reported explicitly
    instead of silently dropping or truncating the file.
    """
    resolved: dict[str, str] = {}
    errors: list[str] = []
    headers = list(_iter_headings(section, 3))
    for i, (path, _, content_start) in enumerate(headers):
        end = headers[i + 1][1] if i + 1 < len(headers) else len(section)
        body = section[content_start:end]
        content, problem = _extract_fenced_block(body)
        if content is not None:
            resolved[path] = content.rstrip("\n")
        elif problem:
            errors.append(f"{problem} for {path!r} in '## Resolved files'")
        else:
            errors.append(f"missing fenced block for {path!r} in '## Resolved files'")
    return resolved, errors


def _format_problem(metadata: dict[str, str]) -> str | None:
    """Return a problem for an unrecognized or unsupported format marker.

    The marker is optional for backward compatibility with unversioned
    (``v0``) files.  A present-but-unsupported major version makes the check
    result unavailable: a file written for a future contract must never be
    trusted under this parser's section and tag semantics.
    """
    value = metadata.get("format")
    if value is None:
        return None
    match = _FORMAT_RE.match(value)
    if not match:
        return (
            f"unrecognized format {value!r} — expected "
            f"external-context/{SUPPORTED_FORMAT_MAJOR}"
        )
    major = int(match.group(1))
    if major != SUPPORTED_FORMAT_MAJOR:
        return (
            f"unsupported external-context format {value!r} — this parser "
            f"supports external-context/{SUPPORTED_FORMAT_MAJOR}"
        )
    return None


def load_external_context(path: str | None) -> ExternalContext:
    """Load and parse an external-context Markdown file.

    Returns an empty (inactive) :class:`ExternalContext` when *path* is
    falsy, the file is missing, or it is empty.  A non-empty file with none
    of the three recognized sections is also inactive.  Parse issues never
    raise: they are recorded in ``parse_errors`` and the affected section is
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

    sections, section_errors, duplicate_sections = _split_sections(raw)
    ctx.parse_errors.extend(section_errors)
    known_sections = {"metadata", "suspect findings", "resolved files"}
    ambiguous_sections = duplicate_sections & known_sections
    ctx.found_sections = set(sections) & known_sections
    unknown = set(sections) - known_sections
    if unknown:
        ctx.parse_errors.append(
            "unknown sections ignored: " + ", ".join(sorted(unknown))
        )

    if "metadata" in sections:
        ctx.metadata, metadata_errors = _parse_metadata(sections["metadata"])
        ctx.parse_errors.extend(metadata_errors)

    format_problem = _format_problem(ctx.metadata)
    if format_problem:
        # Record once, with the consequence attached: the findings-status
        # branch below reuses this exact string instead of appending a
        # near-duplicate, and _check_problems can recognize that the
        # unavailable status already has a specific explanation.
        format_problem = f"{format_problem} — check result unavailable"
        ctx.parse_errors.append(format_problem)

    if "suspect findings" in sections:
        ctx.findings, finding_errors = _parse_findings(sections["suspect findings"])
        ctx.finding_details = _parse_finding_details(ctx.findings)
        # A heading alone is not proof the checks ran.  The declared count is
        # the completeness check: a mismatch (or checker provenance without a
        # count) means the list may be truncated or the section rejected, so
        # the result must not be rendered as a clean "zero findings" pass.
        problem: str | None = None
        if finding_errors:
            problem = "; ".join(finding_errors) + " — check result unavailable"
        elif ambiguous_sections:
            problem = (
                "duplicate recognized section(s) make the check result "
                "ambiguous: "
                + ", ".join(sorted(ambiguous_sections))
                + " — check result unavailable"
            )
        elif format_problem:
            # Already recorded when Metadata was parsed; setting it here
            # only marks the findings result unavailable.  Appending again
            # would print the same problem twice in producer CI output.
            problem = format_problem
        else:
            declared_count = ctx.metadata.get("findings-count")
            checker = ctx.metadata.get("checker")
            if declared_count is not None:
                try:
                    declared = int(declared_count)
                except ValueError:
                    problem = (
                        f"findings-count {declared_count!r} is not an integer "
                        "— check result unavailable"
                    )
                else:
                    if declared < 0:
                        problem = (
                            f"findings-count {declared} is negative "
                            "— check result unavailable"
                        )
                    elif declared != len(ctx.findings):
                        problem = (
                            f"findings-count {declared} does not match "
                            f"{len(ctx.findings)} parsed findings "
                            "— check result unavailable"
                        )
            elif checker:
                problem = (
                    "checker provenance present without findings-count — "
                    "cannot confirm the findings list is complete; check "
                    "result unavailable"
                )
            elif ctx.findings:
                problem = (
                    "findings present without findings-count — cannot "
                    "confirm the list is complete; check result unavailable"
                )
            else:
                problem = (
                    "findings section present but no findings and no check "
                    "provenance — check result unavailable"
                )
        if problem is None:
            ctx.findings_status = "parsed"
        elif problem not in ctx.parse_errors:
            ctx.parse_errors.append(problem)
    else:
        ctx.parse_errors.append(
            "missing '## Suspect findings' section — check result unavailable"
        )
    if "resolved files" in sections:
        ctx.resolved_files, resolved_errors = _parse_resolved_files(
            sections["resolved files"]
        )
        ctx.parse_errors.extend(resolved_errors)

    # A file with no recognized section contributes nothing to any prompt,
    # so it stays inactive.  Note it so operators can spot a mis-formatted
    # file instead of silently losing the context.
    if not ctx.found_sections:
        ctx.parse_errors.append(
            "no recognizable sections (expected '## Metadata', "
            "'## Suspect findings', '## Resolved files')"
        )
    for err in ctx.parse_errors:
        log.warning("external context: %s", err)
    return ctx


def _check_problems(ctx: ExternalContext, *, require_format: bool = False) -> list[str]:
    """Return the reasons a producer-side check should reject *ctx*.

    Unknown sections are explicitly allowed by the contract (they are
    ignored, not injected), so that notice alone does not fail the check.
    Every other parse issue is a producer bug: the consumer would silently
    drop content or distrust the whole file.

    ``require_format`` is a producer-only policy: absent markers remain
    valid for the consumer (legacy v0), but an emitter's CI can require
    every artifact to declare its contract.
    """
    problems: list[str] = []
    if require_format and not ctx.metadata.get("format"):
        problems.append(
            f"missing format marker — expected external-context/{SUPPORTED_FORMAT_MAJOR}"
        )
    if not ctx.active and not any(
        "no recognizable sections" in err for err in ctx.parse_errors
    ):
        problems.append("no recognized sections — the consumer would ignore this file")
    if ctx.findings_status != "parsed" and not any(
        "unavailable" in err for err in ctx.parse_errors
    ):
        # Every unavailable-status path records a specific reason; only
        # synthesize the generic message when it did not.
        problems.append(
            "check result unavailable — the consumer would not trust any file"
        )
    for err in ctx.parse_errors:
        if not err.startswith("unknown sections ignored:"):
            problems.append(err)
    return list(dict.fromkeys(problems))


def main(argv: list[str] | None = None) -> int:
    """Validate an external-context Markdown file (producer CI entry point).

    Exit codes: 0 = usable, 2 = unusable/structurally invalid,
    1 = the file does not exist.  Producers (for example the hybrid
    adapter's ``check-backport.py``) should run this over every emitted
    context file so the consumer's parser and the emitter cannot drift.
    """
    parser = argparse.ArgumentParser(
        prog="python -m auto_bug_fix.external_context",
        description=(
            "Validate an external-resolver-context file: it must carry "
            "recognized sections and a usable, complete findings list."
        ),
    )
    parser.add_argument("path", help="Markdown file to validate")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the file (the default; accepted for explicitness)",
    )
    parser.add_argument(
        "--require-format",
        action="store_true",
        help=(
            "reject files without a format marker (producer CI policy; "
            "the consumer still accepts unversioned v0 files)"
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="emit a machine-readable report"
    )
    args = parser.parse_args(argv)

    expanded = os.path.expanduser(os.path.expandvars(args.path))
    if not os.path.isfile(expanded):
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 1

    ctx = load_external_context(args.path)
    problems = _check_problems(ctx, require_format=args.require_format)
    report = {
        "path": ctx.source_path or expanded,
        "ok": not problems,
        "active": ctx.active,
        "format": ctx.metadata.get("format", ""),
        "findings_status": ctx.findings_status,
        "findings": len(ctx.findings),
        "resolved_files": len(ctx.resolved_files),
        "sections": sorted(ctx.found_sections),
        "parse_errors": list(ctx.parse_errors),
        "problems": problems,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "OK" if report["ok"] else "INVALID"
        print(f"{report['path']}: {status}")
        print(f"  format: {report['format'] or '(unversioned)'}")
        print(
            f"  findings: {report['findings']} "
            f"({report['findings_status']}), "
            f"resolved files: {report['resolved_files']}"
        )
        for problem in problems:
            print(f"  - {problem}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
