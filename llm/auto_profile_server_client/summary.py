# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate per-step and final summary text files."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from llm.auto_profile_server_client.models import StepSummary

log = logging.getLogger(__name__)


class SummaryWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[StepSummary] = []

    def write_step_summary(self, summary: StepSummary) -> Path:
        self.steps.append(summary)
        filename = f"summary_step_{len(self.steps):02d}_{summary.step_name}.txt"
        path = self.output_dir / filename

        lines = [
            f"Step: {summary.step_name}",
            f"Status: {summary.status}",
            f"Duration: {summary.duration_sec:.1f}s",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if summary.details:
            lines.append("Details:")
            for key, val in summary.details.items():
                if isinstance(val, (dict, list)):
                    lines.append(f"  {key}: {json.dumps(val, indent=4, default=str)}")
                else:
                    lines.append(f"  {key}: {val}")
            lines.append("")

        if summary.issues:
            lines.append("Issues encountered:")
            for issue in summary.issues:
                lines.append(f"  - {issue}")
            lines.append("")

        if summary.fixes:
            lines.append("Fixes applied:")
            for fix in summary.fixes:
                lines.append(f"  - {fix}")
            lines.append("")

        path.write_text("\n".join(lines))
        log.info("Wrote step summary: %s", path)
        return path

    def write_final_summary(self) -> Path:
        path = self.output_dir / "summary_final.txt"

        lines = [
            "=" * 72,
            "FINAL BENCHMARK & PROFILING SUMMARY",
            "=" * 72,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total steps: {len(self.steps)}",
            "",
        ]

        total_duration = sum(s.duration_sec for s in self.steps)
        lines.append(f"Total duration: {total_duration:.1f}s")
        lines.append("")

        all_issues = []
        all_fixes = []

        for i, step in enumerate(self.steps, 1):
            lines.append(
                f"--- Step {i}: {step.step_name} [{step.status}] ({step.duration_sec:.1f}s) ---"
            )
            if step.details:
                for key, val in step.details.items():
                    if isinstance(val, (dict, list)):
                        lines.append(f"  {key}: {json.dumps(val, default=str)}")
                    else:
                        lines.append(f"  {key}: {val}")
            if step.issues:
                for issue in step.issues:
                    lines.append(f"  ISSUE: {issue}")
                    all_issues.append(f"[{step.step_name}] {issue}")
            if step.fixes:
                for fix in step.fixes:
                    lines.append(f"  FIX: {fix}")
                    all_fixes.append(f"[{step.step_name}] {fix}")
            lines.append("")

        if all_issues:
            lines.append("=" * 72)
            lines.append("ALL ISSUES:")
            for issue in all_issues:
                lines.append(f"  - {issue}")
            lines.append("")

        if all_fixes:
            lines.append("ALL FIXES:")
            for fix in all_fixes:
                lines.append(f"  - {fix}")
            lines.append("")

        lines.append("=" * 72)
        path.write_text("\n".join(lines))
        log.info("Wrote final summary: %s", path)
        return path
