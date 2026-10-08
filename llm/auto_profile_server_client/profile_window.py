"""Analyze throughput metrics to find the pure-decode midpoint for profiling."""

from __future__ import annotations

import logging
from llm.auto_profile_server_client.models import MetricsSample, ProfileWindow

log = logging.getLogger(__name__)


def find_decode_window(
    metrics_a: list[MetricsSample],
    metrics_b: list[MetricsSample],
    total_requests: int,
    output_len: int,
) -> ProfileWindow:
    """Analyze two runs' metrics to find the pure-decode phase midpoint.

    The algorithm:
    1. Merge both runs' time-series (use run_a as primary for timing)
    2. Find when active requests stabilize (all prefills done)
    3. Find when active requests start dropping (decode finishing)
    4. The stable plateau between these two points is the pure-decode phase
    5. Pick the midpoint

    For time-based profiling (vLLM/SGLang): returns start_sec, duration_sec
    relative to the benchmark start.
    For iteration-based profiling (TRT): returns start_iter, end_iter.
    """
    if not metrics_a:
        log.warning("No metrics from run A, using heuristic profile window")
        return _heuristic_window(total_requests, output_len)

    samples = metrics_a

    t0 = samples[0].timestamp
    times = [s.timestamp - t0 for s in samples]
    running = [s.requests_running for s in samples]
    tokens = [s.generation_tokens_total for s in samples]

    decode_start = _find_prefill_end(times, running, total_requests)
    decode_end = _find_decode_end(times, running, decode_start)

    if decode_end <= decode_start:
        log.warning("Could not detect decode phase boundaries, using heuristic")
        return _heuristic_window(total_requests, output_len)

    midpoint = (decode_start + decode_end) / 2
    window_duration = min(10.0, (decode_end - decode_start) * 0.6)
    profile_start = midpoint - window_duration / 2

    if profile_start < 0:
        profile_start = decode_start

    total_duration = times[-1] if times else 60
    mid_iter = int((midpoint / total_duration) * total_requests * output_len) if total_duration > 0 else 0
    start_iter = max(0, mid_iter - 25)
    end_iter = start_iter + 50

    window = ProfileWindow(
        decode_start_sec=decode_start,
        decode_end_sec=decode_end,
        midpoint_sec=midpoint,
        profile_start_sec=profile_start,
        profile_duration_sec=window_duration,
        start_iter=start_iter,
        end_iter=end_iter,
        method="metrics",
    )
    log.info("Profile window: decode=[%.1f, %.1f]s, midpoint=%.1fs, "
             "profile=[%.1f, +%.1f]s, iters=[%d, %d]",
             decode_start, decode_end, midpoint,
             profile_start, window_duration,
             start_iter, end_iter)
    return window


def _find_prefill_end(
    times: list[float],
    running: list[float],
    total_requests: int,
) -> float:
    """Find when all requests have started (prefill phase complete).

    This is when active_requests reaches its peak and stabilizes.
    """
    if not running:
        return 0.0

    max_running = max(running)
    threshold = max_running * 0.95

    for i, r in enumerate(running):
        if r >= threshold:
            return times[i]

    return times[0]


def _find_decode_end(
    times: list[float],
    running: list[float],
    decode_start: float,
) -> float:
    """Find when requests start completing (decode phase ending).

    This is when active_requests starts dropping significantly after the peak.
    """
    if not running:
        return 0.0

    start_idx = 0
    for i, t in enumerate(times):
        if t >= decode_start:
            start_idx = i
            break

    peak = max(running[start_idx:]) if start_idx < len(running) else max(running)
    drop_threshold = peak * 0.8

    for i in range(start_idx, len(running)):
        if running[i] < drop_threshold:
            return times[i]

    return times[-1]


def _heuristic_window(total_requests: int, output_len: int) -> ProfileWindow:
    """Fallback: estimate the profile window without metrics.

    Use the same approach as the old bash code: skip warmup iterations,
    go to the middle of the run, capture 50 iterations.
    """
    num_warmups = 0
    num_waves = 4
    start_iter = (num_warmups * output_len) + ((num_waves // 2) * output_len) + (output_len // 2)
    end_iter = start_iter + 50

    return ProfileWindow(
        start_iter=start_iter,
        end_iter=end_iter,
        method="heuristic",
    )
