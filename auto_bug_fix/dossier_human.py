# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Backfill ``dossier_human.md`` from an existing ``dossier.md``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_bug_fix.dossier import format_dossier_human, parse_dossier_markdown


def _tracker_info(path: Path) -> tuple[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gates: dict[str, str] = {}
    for phase in data.get("phases") or []:
        gates.update(phase.get("gate_decisions") or {})
    return data.get("outcome", ""), gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a deterministic dossier_human.md review view from an "
            "existing dossier.md. No LLM is involved."
        )
    )
    parser.add_argument("dossier", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: dossier_human.md beside the dossier).",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        help="Optional tracker JSON; supplies outcome and gate values.",
    )
    args = parser.parse_args(argv)

    dossier = parse_dossier_markdown(args.dossier.read_text(encoding="utf-8"))
    outcome = ""
    gates: dict[str, str] = {}
    tracker = args.tracker
    if tracker is None and dossier.issue_id:
        candidates = sorted(args.dossier.parent.glob(f"{dossier.issue_id}_*.json"))
        if candidates:
            tracker = candidates[-1]
    if tracker is not None and tracker.exists():
        outcome, gates = _tracker_info(tracker)

    output = args.output or args.dossier.with_name("dossier_human.md")
    output.write_text(
        format_dossier_human(
            dossier,
            outcome=outcome,
            gates=gates,
            dossier_name=args.dossier.name,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
