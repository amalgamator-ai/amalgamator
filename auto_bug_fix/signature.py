"""Capture and normalize test-run output to create comparable failure signatures."""
from __future__ import annotations

import re
import subprocess
from typing import Literal


def capture_output(cmd: list[str], cwd: str) -> tuple[int, str]:
    """Run a shell command and return (exit_code, combined_stdout_stderr)."""
    result = subprocess.run(
        " ".join(cmd),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )
    return (result.returncode, result.stdout)


def normalize_signature(raw: str) -> str:
    """Strip volatile tokens (addresses, PIDs, timestamps, line numbers) to create a comparable signature."""
    s = raw

    # Pointer addresses
    s = re.sub(r"0x[0-9a-fA-F]{4,16}", "0xADDR", s)

    # Standalone hex sequences (8+ hex chars not preceded by a word char)
    s = re.sub(r"(?<!\w)[0-9a-fA-F]{8,}", "HEX", s)

    # PID-like patterns
    s = re.sub(r"pid=\d+", "pid=PID", s)
    s = re.sub(r"PID \d+", "PID PID", s)
    s = re.sub(r"process \d+", "process PID", s)

    # ISO 8601 timestamps
    s = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "TIMESTAMP", s)

    # Unix timestamps (10-digit integers)
    s = re.sub(r"\b\d{10}\b", "TIMESTAMP", s)

    # Line numbers in stack frames
    s = re.sub(r":(\d+):", ":LINE:", s)
    s = re.sub(r"line \d+", "line LINE", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def compare_signatures(
    s_target: str, s_parent: str
) -> Literal["match", "partial", "mismatch"]:
    """Compare two failure signatures after normalization using Jaccard similarity."""
    norm_target = normalize_signature(s_target)
    norm_parent = normalize_signature(s_parent)

    if norm_target == norm_parent:
        return "match"

    target_lines = set(re.split(r"\. |\n", norm_target))
    parent_lines = set(re.split(r"\. |\n", norm_parent))

    union = target_lines | parent_lines
    if not union:
        return "match"

    intersection = target_lines & parent_lines
    jaccard = len(intersection) / len(union)

    if jaccard > 0.8:
        return "partial"

    return "mismatch"
