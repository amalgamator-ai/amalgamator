"""Auditable dossier builder for human review."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


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
    issue_id: str
    source_branch: str
    target_branch: str
    fix_commit: str
    bug_description: str
    items: list[DossierItem] = field(default_factory=list)
    strategies: list[StrategyAttempt] = field(default_factory=list)
    cherry_pick_path: str = ""
    trailers: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, label: str, content: str, source: str) -> None:
        """Append a labelled evidence item to the dossier."""
        self.items.append(DossierItem(label=label, content=content, source=source))

    def add_strategy(self, strategy: str, outcome: str, commentary: str = "") -> None:
        """Record a cherry-pick strategy attempt and its outcome."""
        self.strategies.append(StrategyAttempt(
            strategy=strategy, outcome=outcome, commentary=commentary,
        ))


def build_trailers(
    phase: str,
    model: str,
    resolved_hunks: str | None = None,
    strategies_tried: str | None = None,
    bisect_sha: str | None = None,
) -> dict[str, str]:
    """Build git commit trailer dict for the ported fix commit."""
    trailers = {
        "AutoBugFix-Phase": phase,
        "AutoBugFix-Model": model,
    }
    if resolved_hunks:
        trailers["AutoBugFix-Resolved-Hunks"] = resolved_hunks
    if strategies_tried:
        trailers["AutoBugFix-Strategies-Tried"] = strategies_tried
    if bisect_sha:
        trailers["AutoBugFix-Triage-Bisect-SHA"] = bisect_sha
    return trailers


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
        lines.append(f"## {item.label}")
        lines.append("")
        lines.append(f"**Source**: {item.source}")
        lines.append("")
        lines.append("```")
        lines.append(item.content)
        lines.append("```")
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
