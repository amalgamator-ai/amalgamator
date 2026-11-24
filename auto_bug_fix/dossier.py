# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Auditable dossier builder for human review."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from auto_bug_fix.git_tools import COAUTHOR_TRAILERS


@dataclass
class DossierItem:
    label: str
    content: str
    source: str


@dataclass
class StrategyAttempt:
    strategy: str
    outcome: str
    commentary: str = ""


@dataclass
class Dossier:
    issue_id: str = ""
    source_branch: str = ""
    target_branch: str = ""
    fix_commit: str = ""
    bug_description: str = ""
    items: list[DossierItem] = field(default_factory=list)
    strategies: list[StrategyAttempt] = field(default_factory=list)
    cherry_pick_path: str = ""
    trailers: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def add(self, label: str, content: str, source: str) -> None:
        """Append a labelled evidence item to the dossier."""
        self.items.append(DossierItem(label=label, content=content, source=source))

    def add_strategy(self, strategy: str, outcome: str, commentary: str = "") -> None:
        """Record a cherry-pick strategy attempt and its outcome."""
        self.strategies.append(
            StrategyAttempt(
                strategy=strategy,
                outcome=outcome,
                commentary=commentary,
            )
        )


def build_trailers(
    phase: str,
    model: str,
    resolved_hunks: str | None = None,
    strategies_tried: str | None = None,
) -> dict[str, str]:
    """Build git commit trailer dict for the ported fix commit."""
    trailers = {
        "AutoBugFix-Phase": phase,
        "AutoBugFix-Model": model,
        **COAUTHOR_TRAILERS,
    }
    if resolved_hunks:
        trailers["AutoBugFix-Resolved-Hunks"] = resolved_hunks
    if strategies_tried:
        trailers["AutoBugFix-Strategies-Tried"] = strategies_tried
    return trailers


def _fence_for(content: str) -> str:
    """A code fence longer than any backtick run inside *content*.

    A fixed ``` fence truncates the item when the payload itself contains a
    fenced block (a README or docs change, for example).
    """
    longest = max((len(run) for run in re.findall(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def format_dossier(dossier: Dossier) -> str:
    """Render the dossier as a human-readable Markdown string."""
    lines = [
        f"# Bug-Fix Porting Dossier: {dossier.issue_id}",
        "",
        f"**{dossier.bug_description}**",
        "",
        f"- **Source branch**: {dossier.source_branch}",
        f"- **Target branch**: {dossier.target_branch}",
        f"- **Fix commit**: {dossier.fix_commit}",
        f"- **Cherry-pick path**: {dossier.cherry_pick_path}",
        f"- **Generated**: {dossier.created_at}",
        "",
        "---",
        "",
    ]

    for item in dossier.items:
        fence = _fence_for(item.content)
        lines.append(f"## {item.label}")
        lines.append("")
        lines.append(f"**Source**: {item.source}")
        lines.append("")
        lines.append(fence)
        lines.append(item.content)
        lines.append(fence)
        lines.append("")

    if dossier.strategies:
        lines.append("## Strategies Attempted")
        lines.append("")
        for s in dossier.strategies:
            lines.append(f"- **{s.strategy}**")
            lines.append(f"  - Outcome (deterministic): {s.outcome}")
            if s.commentary:
                lines.append(f"  - Commentary (LLM): {s.commentary}")
        lines.append("")

    if dossier.trailers:
        lines.append("## Commit Trailers")
        lines.append("")
        lines.append("```")
        for k, v in dossier.trailers.items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


@dataclass
class _PatchFile:
    path: str
    added: int = 0
    removed: int = 0
    status: str = "modified"
    binary: bool = False


def _first_item(dossier: Dossier, label: str) -> DossierItem | None:
    return next((item for item in dossier.items if item.label == label), None)


def _parse_json_item(item: DossierItem | None) -> dict | None:
    """Parse a JSON (or Python-literal) dict from a dossier item."""
    if item is None:
        return None
    try:
        value = json.loads(item.content)
    except (TypeError, ValueError):
        try:
            value = ast.literal_eval(item.content)
        except (ValueError, SyntaxError):
            return None
    return value if isinstance(value, dict) else None


def _dedupe_items(items: list[DossierItem]) -> list[tuple[DossierItem, int]]:
    """Collapse byte-identical records, preserving first-seen order."""
    grouped: OrderedDict[tuple[str, str, str], int] = OrderedDict()
    first: dict[tuple[str, str, str], DossierItem] = {}
    for item in items:
        key = (item.label, item.source, item.content)
        grouped[key] = grouped.get(key, 0) + 1
        first.setdefault(key, item)
    return [(first[key], count) for key, count in grouped.items()]


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _code(value: str) -> str:
    return f"`{_cell(value)}`" if value else "not recorded"


def _demote_headings(text: str, levels: int = 2) -> str:
    def repl(match: re.Match[str]) -> str:
        hashes = match.group(1)
        return "#" * min(6, len(hashes) + levels) + " "

    return re.sub(r"^(#{1,6})\s", repl, text, flags=re.MULTILINE)


def _patch_stats(text: str) -> list[_PatchFile]:
    files: OrderedDict[str, _PatchFile] = OrderedDict()
    current: _PatchFile | None = None
    in_hunk = False
    # ``str.splitlines()`` also splits on form feeds and other control
    # characters that occur inside patch payloads (GNU licence texts, man
    # pages), which halves a diff line and stops the count early. Split on
    # newlines only.
    for line in text.split("\n"):
        if line.startswith("diff --git "):
            match = re.search(r" b/(.+)$", line)
            path = match.group(1) if match else line
            if path not in files:
                files[path] = _PatchFile(path=path)
            current = files[path]
            in_hunk = False
        elif current is not None and line.startswith("new file mode"):
            current.status = "added"
        elif current is not None and line.startswith("deleted file mode"):
            current.status = "deleted"
        elif current is not None and (
            line.startswith("GIT binary patch") or line.startswith("Binary files ")
        ):
            current.binary = True
            current.status = "binary"
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and current is not None:
            if line == "-- ":
                in_hunk = False
            elif line == "":
                # A stripped context line or a control-character artifact; it
                # does not end the hunk.
                continue
            elif line.startswith("+"):
                current.added += 1
            elif line.startswith("-"):
                current.removed += 1
            elif not line.startswith((" ", "\\")):
                in_hunk = False
    return list(files.values())


def _format_list(values: object, empty: str = "none") -> str:
    if not isinstance(values, list) or not values:
        return empty
    return ", ".join(f"`{_cell(value)}`" for value in values)


def _render_triage(data: dict | None) -> list[str]:
    if not data:
        return ["### Triage", "", "_Not recorded._"]
    lines = ["### Triage", ""]
    for key, label in (
        ("estimated_difficulty", "Difficulty"),
        ("recommendation", "Recommendation"),
        ("mode", "Mode"),
        ("commit_type", "Commit type"),
    ):
        if data.get(key):
            lines.append(f"- **{label}:** {_cell(data[key])}")
    lines.append(
        f"- **Files intentionally skipped:** "
        f"{_format_list(data.get('files_to_skip'))}"
    )
    lines.append(f"- **Test files ported:** {_format_list(data.get('test_files'))}")

    clusters = data.get("clusters")
    if isinstance(clusters, list) and clusters:
        lines += [
            "",
            "| Cluster | Commits | Intent | Kind |",
            "|---|---:|---|---|",
        ]
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            commits = cluster.get("commits") or []
            lines.append(
                "| `{}` | {} | {} | {} |".format(
                    _cell(cluster.get("id", "?")),
                    len(commits) if isinstance(commits, list) else 0,
                    _cell(str(cluster.get("intent", ""))[:160]),
                    _cell(cluster.get("kind", "")),
                )
            )

    prerequisites = data.get("prerequisites")
    if isinstance(prerequisites, list) and prerequisites:
        lines += [
            "",
            "<details><summary>Prerequisites</summary>",
            "",
            "| Symbol | Portable | Reason |",
            "|---|---|---|",
        ]
        for prereq in prerequisites:
            if not isinstance(prereq, dict):
                continue
            lines.append(
                "| `{}` | {} | {} |".format(
                    _cell(prereq.get("symbol", "")),
                    _cell(prereq.get("portable", "")),
                    _cell(str(prereq.get("reason", ""))[:220]),
                )
            )
        lines += ["", "</details>"]
    return lines


def _render_cluster(item: DossierItem) -> list[str]:
    data = _parse_json_item(item)
    if not data:
        return _details(item.label, item.source, item.content)
    cluster = data.get("cluster", item.label)
    lines = [f"### Cluster `{_cell(cluster)}`", ""]
    actions = data.get("actions")
    if isinstance(actions, list) and actions:
        lines += [
            "| Symbol | Action | Verify |",
            "|---|---|---|",
        ]
        for action in actions:
            if not isinstance(action, dict):
                continue
            lines.append(
                "| `{}` | {} | `{}` |".format(
                    _cell(action.get("symbol", "")),
                    _cell(action.get("action", "")),
                    _cell(action.get("verify", "")),
                )
            )
    risks = data.get("risks")
    if isinstance(risks, list) and risks:
        lines += ["", "**Risks:**"]
        for risk in risks:
            lines.append(f"- {_cell(risk)}")
    if not actions and not risks:
        lines += ["_No actions or risks recorded._"]
    return lines


def _render_section(label: str, item: DossierItem) -> list[str]:
    """Render a short evidence item as a section, prettifying bare paths."""
    content = item.content.strip()
    if content and "\n" not in content and " " not in content:
        content = f"`{content}`"
    return [
        f"### {label}",
        "",
        f"> Source: {_cell(item.source)}",
        "",
        content,
    ]


def _render_patch(item: DossierItem | None, candidate_patch: str) -> list[str]:
    if item is None:
        return ["### Patch", "", "_Not recorded._"]
    files = _patch_stats(item.content)
    added = sum(file.added for file in files)
    removed = sum(file.removed for file in files)
    counts = f"+{added}/−{removed}"
    if any(file.binary for file in files):
        counts += f", {sum(1 for file in files if file.binary)} binary"
    lines = [
        "### Patch",
        "",
        (
            f"- **Patch file:** [{candidate_patch}]({candidate_patch})"
            if candidate_patch
            else "- **Patch file:** not recorded"
        ),
        f"- **Files changed:** {len(files)}",
        f"- **Line counts:** {counts} (summed across commits)",
        "",
    ]
    if files:
        lines += [
            "| File | Status | + | − |",
            "|---|---|---:|---:|",
        ]
        for file in files:
            if file.binary:
                lines.append(f"| `{_cell(file.path)}` | binary | — | — |")
                continue
            lines.append(
                f"| `{_cell(file.path)}` | {file.status} | "
                f"{file.added} | {file.removed} |"
            )
    else:
        lines.append("_Could not parse the patch content._")
    return lines


def _details(
    label: str,
    source: str,
    content: str,
    *,
    demote: bool = False,
    code: bool = False,
) -> list[str]:
    body = _demote_headings(content, levels=3) if demote else content
    lines = [
        "<details>",
        f"<summary>{_cell(label)} — {_cell(source)}</summary>",
        "",
    ]
    if code:
        lines += ["```text", body.strip(), "```"]
    else:
        lines.append(body.strip())
    lines += ["", "</details>"]
    return lines


def _generic(item: DossierItem) -> list[str]:
    content = item.content
    if len(content) > 6000:
        content = content[:6000] + "\n\n… (truncated; see `dossier.md`)"
    return _details(item.label, item.source, content, demote=False)


def _review_focus(
    dossier: Dossier,
    triage: dict | None,
    gates: dict[str, str],
    outcome: str,
) -> list[str]:
    focus: list[str] = []
    if outcome and outcome != "success":
        focus.append(
            f"Run outcome is `{outcome}`; inspect the failure evidence before "
            "considering the patch."
        )
    equivalence = gates.get("semantic_equivalence")
    # ``modified`` is the normal outcome of a port onto a distant baseline;
    # flagging it would make the section noise on almost every adapted backport.
    if equivalence and equivalence not in {"identical", "modified"}:
        focus.append(
            f"Semantic equivalence is `{equivalence}`; review the range-diff "
            "and the resolution reports."
        )
    relatedness = gates.get("fused_relatedness")
    if relatedness in {"advisory", "unrelated"}:
        focus.append(
            f"Fused relatedness is `{relatedness}`; check the resolved hunks "
            "against the upstream fix."
        )
    skipped = (triage or {}).get("files_to_skip")
    if isinstance(skipped, list) and skipped:
        focus.append(
            "Intentionally skipped files: "
            + ", ".join(f"`{_cell(path)}`" for path in skipped)
            + ". Confirm they are generated or not needed."
        )
    if any(strategy.outcome.startswith("conflict") for strategy in dossier.strategies):
        focus.append(
            "Conflicts were resolved by the agent; inspect the resolution "
            "reports for the exact adaptations."
        )
    advisories = [
        f"`{name}={value}`"
        for name, value in gates.items()
        if value in {"advisory", "unrelated"}
    ]
    if advisories:
        focus.append("Advisory gate(s): " + ", ".join(advisories) + ".")
    if not focus:
        focus.append("No automatic red flags; review the patch and the evidence below.")
    return focus


def format_dossier_human(
    dossier: Dossier,
    *,
    outcome: str = "",
    gates: dict[str, str] | None = None,
    dossier_name: str = "dossier.md",
    candidate_patch: str = "candidate.patch",
) -> str:
    """Render a deterministic, review-oriented view of a dossier.

    This is not an LLM summary. It selects and structures the same evidence as
    ``dossier.md``: a short briefing, the patch surface, the triage and cluster
    decisions, the checks, and an index back to the full audit record.
    """
    gates = gates or {}
    triage = _parse_json_item(_first_item(dossier, "Semantic triage"))
    patch_item = _first_item(dossier, "Candidate patch")
    patch_files = _patch_stats(patch_item.content) if patch_item else []
    added = sum(file.added for file in patch_files)
    removed = sum(file.removed for file in patch_files)
    binary = sum(1 for file in patch_files if file.binary)
    counts = f"+{added}/−{removed}"
    if binary:
        counts += f", {binary} binary"

    lines = [
        f"# Bug-Fix Porting Review: {dossier.issue_id}",
        "",
        f"> Deterministic review view of [`{dossier_name}`]({dossier_name}). "
        "The full dossier remains the audit record.",
        "",
    ]
    if dossier.bug_description:
        lines += [f"**{dossier.bug_description}**", ""]
    lines += [
        f"- **Source branch:** {_code(dossier.source_branch)}",
        f"- **Target branch:** {_code(dossier.target_branch)}",
        f"- **Fix commit:** {_code(dossier.fix_commit)}",
        f"- **Cherry-pick path:** {dossier.cherry_pick_path or 'not recorded'}",
        f"- **Generated:** {dossier.created_at}",
        "",
        "## At a glance",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Outcome | **{_cell(outcome or 'not recorded')}** |",
        f"| Build / test | {_cell(gates.get('verify', 'not recorded'))} |",
        f"| Completeness | {_cell(gates.get('completeness', 'not recorded'))} |",
        f"| Semantic equivalence | "
        f"{_cell(gates.get('semantic_equivalence', 'not recorded'))} |",
        f"| Relatedness | {_cell(gates.get('fused_relatedness', 'not recorded'))} |",
        f"| Patch | {len(patch_files)} files, {counts} |",
        "",
        "## Review focus",
        "",
    ]
    for item in _review_focus(dossier, triage, gates, outcome):
        lines.append(f"- {item}")

    lines += [
        "",
        "## Review checklist",
        "",
        "- [ ] Compare each changed file with the upstream fix.",
        "- [ ] Confirm ported tests were not weakened or skipped.",
        "- [ ] Check that intentionally skipped files are generated or unneeded.",
        "- [ ] Inspect the range-diff and resolution reports when flagged.",
        "- [ ] Check the commit trailers and model provenance.",
    ]

    triage_item = _first_item(dossier, "Semantic triage")
    if triage_item:
        lines += [""]
        if triage:
            lines += _render_triage(triage)
        else:
            lines += [
                "### Triage",
                "",
                "_Not recorded._",
                "",
                *_details(
                    "Semantic triage (raw)",
                    triage_item.source,
                    triage_item.content,
                ),
            ]
    triage_result = _first_item(dossier, "Triage result")
    if triage_result:
        data = _parse_json_item(triage_result)
        if data:
            lines += [
                "",
                "### Deterministic triage gates",
                "",
                "| Gate | Value |",
                "|---|---|",
            ]
            for key, value in data.items():
                lines.append(f"| {_cell(key)} | {_cell(value)} |")
        else:
            lines += [
                "",
                *_details("Triage result", triage_result.source, triage_result.content),
            ]

    for label in ("Test port", "Baseline (clean target branch)"):
        item = _first_item(dossier, label)
        if item:
            lines += [""] + _render_section(label, item)

    lines += [""] + _render_patch(patch_item, candidate_patch)

    cluster_items = [
        item for item in dossier.items if item.label.startswith("Cluster ")
    ]
    if cluster_items:
        lines += ["", "## Cluster plans"]
        for item in cluster_items:
            lines += [""] + _render_cluster(item)

    report_items = [
        item
        for item in dossier.items
        if item.label.startswith("Fused resolution session report")
    ]
    if report_items:
        lines += ["", "## Resolution reports"]
        for item in report_items:
            lines += [""] + _details(item.label, item.source, item.content, demote=True)

    for label in (
        "Semantic equivalence",
        "Fused relatedness",
        "Allowlist conformance",
        "Fused resolution",
        "Evidence scope (fused resolution)",
        "Final status",
        "Final results",
    ):
        item = _first_item(dossier, label)
        if item:
            lines += [""] + _render_section(label, item)

    range_diff = _first_item(dossier, "Range-diff (upstream vs backport)")
    if range_diff:
        lines += [
            "",
            "<details>",
            "<summary>Range-diff (upstream vs backport)</summary>",
            "",
            "```text",
            range_diff.content.strip(),
            "```",
            "",
            "</details>",
        ]

    handled = {
        id(item)
        for item in (
            triage_item,
            triage_result,
            patch_item,
            range_diff,
            *cluster_items,
            *report_items,
        )
        if item is not None
    }
    for label in (
        "Test port",
        "Baseline (clean target branch)",
        "Semantic equivalence",
        "Fused relatedness",
        "Allowlist conformance",
        "Fused resolution",
        "Evidence scope (fused resolution)",
        "Final status",
        "Final results",
    ):
        item = _first_item(dossier, label)
        if item is not None:
            handled.add(id(item))
    remaining = [item for item in dossier.items if id(item) not in handled]
    if remaining:
        lines += ["", "## Other evidence"]
        for item, count in _dedupe_items(remaining):
            if count > 1:
                lines += [""] + _details(
                    f"{item.label} (×{count} identical records)",
                    item.source,
                    item.content,
                )
            else:
                lines += [""] + _generic(item)

    if dossier.strategies:
        counts = Counter(
            (strategy.strategy, strategy.outcome) for strategy in dossier.strategies
        )
        lines += [
            "",
            "## Strategies",
            "",
            "| Strategy | Outcome | Count |",
            "|---|---|---:|",
        ]
        for (strategy, strategy_outcome), count in sorted(counts.items()):
            lines.append(f"| {_cell(strategy)} | {_cell(strategy_outcome)} | {count} |")

    if dossier.trailers:
        lines += ["", "## Commit trailers", "", "| Trailer | Value |", "|---|---|"]
        for key, value in dossier.trailers.items():
            lines.append(f"| {_cell(key)} | {_cell(value)} |")

    index_counts: OrderedDict[tuple[str, str], int] = OrderedDict()
    for item in dossier.items:
        key = (item.label, item.source)
        index_counts[key] = index_counts.get(key, 0) + 1
    repeated = any(count > 1 for count in index_counts.values())
    lines += [
        "",
        "## Evidence index",
        "",
        f"Full audit record: [`{dossier_name}`]({dossier_name}).",
        "",
    ]
    if repeated:
        lines += ["| Section | Source | Records |", "|---|---|---:|"]
        for (label, source), count in index_counts.items():
            lines.append(f"| {_cell(label)} | {_cell(source)} | {count} |")
    else:
        lines += ["| Section | Source |", "|---|---|"]
        for label, source in index_counts:
            lines.append(f"| {_cell(label)} | {_cell(source)} |")

    return "\n".join(lines) + "\n"


_HEADER_RE = re.compile(r"^- \*\*(.+?)\*\*: (.*)$")
_FENCE_RE = re.compile(r"^(`{3,})\s*$")


def parse_dossier_markdown(text: str) -> Dossier:
    """Reconstruct a :class:`Dossier` from a generated ``dossier.md``.

    Used to backfill ``dossier_human.md`` for runs that predate the human
    view. It understands the generator's own layout; it is not a general
    Markdown parser.
    """
    lines = text.split("\n")
    dossier = Dossier()

    separator = next(
        (i for i, line in enumerate(lines) if line.strip() == "---"), len(lines)
    )
    for line in lines[:separator]:
        if line.startswith("# Bug-Fix Porting Dossier: "):
            dossier.issue_id = line.split(": ", 1)[1].strip()
        elif line.startswith("**") and line.endswith("**"):
            # Only the header block, so a later bold-only line cannot be
            # mistaken for the description.
            description = line.strip("*").strip()
            if description:
                dossier.bug_description = description
        match = _HEADER_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if key == "Source branch":
            dossier.source_branch = value
        elif key == "Target branch":
            dossier.target_branch = value
        elif key == "Fix commit":
            dossier.fix_commit = value
        elif key == "Cherry-pick path":
            dossier.cherry_pick_path = value
        elif key == "Generated":
            dossier.created_at = value

    index = separator + 1

    while index < len(lines):
        if not lines[index].startswith("## "):
            index += 1
            continue
        label = lines[index][3:].strip()
        index += 1

        if label == "Strategies Attempted":
            strategy: StrategyAttempt | None = None
            while index < len(lines) and not lines[index].startswith("## "):
                line = lines[index]
                match = re.match(r"^- \*\*(.+?)\*\*$", line)
                if match:
                    if strategy is not None:
                        dossier.strategies.append(strategy)
                    strategy = StrategyAttempt(strategy=match.group(1), outcome="")
                elif strategy is not None and "Outcome (deterministic):" in line:
                    strategy.outcome = line.split("Outcome (deterministic):", 1)[
                        1
                    ].strip()
                elif strategy is not None and "Commentary (LLM):" in line:
                    strategy.commentary = line.split("Commentary (LLM):", 1)[1].strip()
                index += 1
            if strategy is not None:
                dossier.strategies.append(strategy)
            continue

        if label == "Commit Trailers":
            while index < len(lines) and lines[index].strip() != "```":
                index += 1
            index += 1
            while index < len(lines) and lines[index].strip() != "```":
                if ": " in lines[index]:
                    key, value = lines[index].split(": ", 1)
                    dossier.trailers[key] = value
                index += 1
            index += 1
            continue

        source = ""
        while index < len(lines) and not lines[index].startswith("**Source**: "):
            if lines[index].startswith("## "):
                break
            index += 1
        if index < len(lines) and lines[index].startswith("**Source**: "):
            source = lines[index][len("**Source**: ") :].strip()
            index += 1

        fence = ""
        while index < len(lines) and not _FENCE_RE.match(lines[index].strip()):
            if lines[index].startswith("## "):
                break
            index += 1
        if index < len(lines):
            match = _FENCE_RE.match(lines[index].strip())
            if match:
                fence = match.group(1)
        if not fence:
            continue
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index].strip() != fence:
            content.append(lines[index])
            index += 1
        index += 1
        dossier.add(label, "\n".join(content), source)

    return dossier
