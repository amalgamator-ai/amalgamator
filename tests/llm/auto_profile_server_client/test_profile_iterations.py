"""Profiler window arithmetic for iteration-armed captures.

``delay_iterations`` is counted in engine steps. A run of N prompts at
concurrency C takes about N*(output_len+1)/C steps, so the delay must shrink as
concurrency grows. Getting this wrong is silent: the profiler waits for a step
that never arrives, the active window never opens, and the run finishes with no
trace file at all.
"""

from llm.auto_profile_server_client.benchmark_runner import BenchmarkRunner


def _compute(num_prompts, output_len, concurrency, active=50):
    return BenchmarkRunner._compute_profile_iterations(
        num_prompts, output_len, concurrency, active)


def test_single_request_run_targets_mid_decode():
    # 8 prompts x 1024 tokens at c1 = 8200 engine steps.
    delay, active = _compute(num_prompts=8, output_len=1024, concurrency=1)
    total_steps = 8 * 1025
    assert 0 < delay < total_steps
    assert delay == total_steps // 2


def test_delay_shrinks_with_concurrency():
    # 128 prompts x 128 tokens: 16,512 request-tokens, but only ~258 engine
    # steps at c64. Asking for a delay above 258 is what killed the capture.
    delay, active = _compute(num_prompts=128, output_len=128, concurrency=64)
    assert delay == (128 * 129 // 64) // 2
    assert delay < 258


def test_active_window_fits_in_the_run():
    for conc in (1, 8, 64, 256):
        delay, active = _compute(
            num_prompts=128, output_len=128, concurrency=conc)
        total_steps = max(1, (128 * 129) // conc)
        assert active >= 1
        assert delay + active <= total_steps


def test_concurrency_above_num_prompts_is_clamped():
    delay, active = _compute(num_prompts=4, output_len=64, concurrency=256)
    assert delay == 0
    assert active == 1
