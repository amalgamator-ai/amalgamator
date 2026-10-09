"""Compare two benchmark runs to check performance stability."""

from __future__ import annotations

import logging

from llm.auto_profile_server_client.models import BenchmarkResult, StabilityResult

log = logging.getLogger(__name__)


def check_stability(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    threshold_pct: float = 5.0,
) -> StabilityResult:
    """Compare TPOT and TTFT between two runs.

    Stable if both metrics differ by less than threshold_pct.
    """
    tpot_a, tpot_b = result_a.tpot_ms, result_b.tpot_ms
    ttft_a, ttft_b = result_a.ttft_ms, result_b.ttft_ms

    tpot_diff = _relative_diff_pct(tpot_a, tpot_b)
    ttft_diff = _relative_diff_pct(ttft_a, ttft_b)

    tpot_stable = tpot_diff < threshold_pct
    ttft_stable = ttft_diff < threshold_pct
    stable = tpot_stable and ttft_stable

    parts = []
    if not tpot_stable:
        parts.append(f"TPOT diff {tpot_diff:.1f}% > {threshold_pct}% "
                     f"(a={tpot_a:.2f}ms, b={tpot_b:.2f}ms)")
    if not ttft_stable:
        parts.append(f"TTFT diff {ttft_diff:.1f}% > {threshold_pct}% "
                     f"(a={ttft_a:.2f}ms, b={ttft_b:.2f}ms)")

    if stable:
        msg = (f"Stable: TPOT diff={tpot_diff:.1f}%, TTFT diff={ttft_diff:.1f}% "
               f"(threshold={threshold_pct}%)")
    else:
        msg = f"Unstable: {'; '.join(parts)}"

    log.info(msg)

    return StabilityResult(
        stable=stable,
        tpot_diff_pct=tpot_diff,
        ttft_diff_pct=ttft_diff,
        result_a=result_a,
        result_b=result_b,
        message=msg,
    )


def _relative_diff_pct(a: float, b: float) -> float:
    avg = (a + b) / 2
    if avg == 0:
        return 0.0
    return abs(a - b) / avg * 100


def average_results(a: BenchmarkResult, b: BenchmarkResult) -> BenchmarkResult:
    """Return the average of two benchmark results."""
    return BenchmarkResult(
        ttft_ms=(a.ttft_ms + b.ttft_ms) / 2,
        tpot_ms=(a.tpot_ms + b.tpot_ms) / 2,
        throughput_tok_per_sec=(a.throughput_tok_per_sec + b.throughput_tok_per_sec) / 2,
        ttft_p50_ms=(a.ttft_p50_ms + b.ttft_p50_ms) / 2,
        ttft_p90_ms=(a.ttft_p90_ms + b.ttft_p90_ms) / 2,
        ttft_p99_ms=(a.ttft_p99_ms + b.ttft_p99_ms) / 2,
        tpot_p50_ms=(a.tpot_p50_ms + b.tpot_p50_ms) / 2,
        tpot_p90_ms=(a.tpot_p90_ms + b.tpot_p90_ms) / 2,
        tpot_p99_ms=(a.tpot_p99_ms + b.tpot_p99_ms) / 2,
        e2e_latency_ms=(a.e2e_latency_ms + b.e2e_latency_ms) / 2,
        num_prompts=a.num_prompts,
        total_time_sec=(a.total_time_sec + b.total_time_sec) / 2,
    )
