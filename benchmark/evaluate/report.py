"""
Leaderboard report generator.

Prints a human-readable leaderboard table and saves:
  - leaderboard.json   (machine-readable, for dashboard integration)
  - leaderboard.md     (Markdown table, for README / GitHub / Slack)
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import AggregatedMetrics


_TYPES = ["TYPE-I", "TYPE-II", "TYPE-III", "TYPE-IV", "TYPE-V"]


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_cost(v: float) -> str:
    if v < 0.001:
        return "<$0.001"
    return f"${v:.3f}"


def print_leaderboard(aggregated: list[AggregatedMetrics], eval_cfgs: dict) -> None:
    print()
    print("=" * 80)
    print("  BACKPORT BENCHMARK LEADERBOARD")
    print("=" * 80)

    for agg in sorted(aggregated, key=lambda a: -a.semantic_equiv_rate):
        label = eval_cfgs.get(agg.evaluator, {}).get("label", agg.evaluator)
        desc  = eval_cfgs.get(agg.evaluator, {}).get("description", "")
        print(f"\n  {label}")
        print(f"  {desc}")
        print(f"  {'-' * 60}")
        print(f"  Cases evaluated : {agg.n_total}  (errors: {agg.n_error})")
        print(f"  Semantic equiv  : {_pct(agg.semantic_equiv_rate)}  ← primary metric")
        print(f"  Exact match     : {_pct(agg.exact_match_rate)}")
        print(f"  Uncertain       : {_pct(agg.uncertain_rate)}")
        print(f"  Error rate      : {_pct(agg.error_rate)}")
        print(f"  Avg latency     : {agg.mean_latency_s:.1f}s")
        print(f"  Avg cost/case   : {_fmt_cost(agg.mean_cost_usd)}")
        print(f"  Total cost      : {_fmt_cost(agg.total_cost_usd)}")

        if agg.by_type:
            print()
            print(f"  {'Type':<10}  {'Semantic Equiv':>16}  {'Exact':>8}  {'Errors':>8}  {'Avg Cost':>10}  {'Avg Lat':>9}")
            print(f"  {'-'*10}  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*9}")
            for t in _TYPES:
                if t not in agg.by_type:
                    continue
                bt = agg.by_type[t]
                print(
                    f"  {t:<10}  {_pct(bt['semantic_equiv_rate']):>16}  "
                    f"{_pct(bt['exact_match_rate']):>8}  "
                    f"{_pct(bt['error_rate']):>8}  "
                    f"{_fmt_cost(bt['mean_cost_usd']):>10}  "
                    f"{bt['mean_latency_s']:>8.1f}s"
                )

    print()
    print("=" * 80)


def save_leaderboard(aggregated: list[AggregatedMetrics], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    # JSON
    data = [a.to_dict() for a in aggregated]
    (output_root / "leaderboard.json").write_text(json.dumps(data, indent=2))

    # Markdown
    md_lines = [
        "# Backport Benchmark Leaderboard",
        "",
        "| Evaluator | Semantic Equiv ↑ | Exact Match | Error Rate | Avg Cost | Avg Latency |",
        "|-----------|-----------------|-------------|------------|----------|-------------|",
    ]
    for agg in sorted(aggregated, key=lambda a: -a.semantic_equiv_rate):
        md_lines.append(
            f"| **{agg.evaluator}** | {_pct(agg.semantic_equiv_rate)} | "
            f"{_pct(agg.exact_match_rate)} | {_pct(agg.error_rate)} | "
            f"{_fmt_cost(agg.mean_cost_usd)} | {agg.mean_latency_s:.1f}s |"
        )

    md_lines += [
        "",
        "## By Complexity Type",
        "",
    ]
    for agg in aggregated:
        if not agg.by_type:
            continue
        md_lines += [f"### {agg.evaluator}", ""]
        md_lines.append("| Type | Semantic Equiv | Exact Match | Error Rate | Avg Cost | Avg Latency |")
        md_lines.append("|------|---------------|-------------|------------|----------|-------------|")
        for t in _TYPES:
            if t not in agg.by_type:
                continue
            bt = agg.by_type[t]
            md_lines.append(
                f"| {t} | {_pct(bt['semantic_equiv_rate'])} | "
                f"{_pct(bt['exact_match_rate'])} | {_pct(bt['error_rate'])} | "
                f"{_fmt_cost(bt['mean_cost_usd'])} | {bt['mean_latency_s']:.1f}s |"
            )
        md_lines.append("")

    (output_root / "leaderboard.md").write_text("\n".join(md_lines))
    print(f"\nLeaderboard saved to {output_root}/leaderboard.{{json,md}}")
