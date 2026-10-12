# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Render structured agent events as a human-readable transcript."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .runner import _format_human_event


def render(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    count = 0
    with os.fdopen(fd, "w") as output:
        output.write("AUTO_CODE_GEN HUMAN-READABLE AGENT EVENT LOG\n")
        output.write(f"Source: {input_path}\n")
        for line_number, line in enumerate(input_path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                output.write(f"\n[Invalid JSON event at line {line_number}]\n{line}\n")
                continue
            output.write(_format_human_event(event))
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="agent_events.jsonl path")
    parser.add_argument("--output", type=Path, help="human-readable output path")
    args = parser.parse_args()
    output = args.output or args.input.with_name("agent_events.log")
    count = render(args.input, output)
    print(f"Rendered {count} events to {output}")


if __name__ == "__main__":
    main()
