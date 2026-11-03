# [ROCm] Prefer AITER unified attention for small-batch decode

## Purpose

On CDNA, `ROCM_ATTN` can only use the ROCm custom paged-attention kernel for
`head_size` 64 and 128 (`use_rocm_custom_paged_attention()` in
`vllm/platforms/rocm.py`). Models with `head_dim=256` — e.g.
`Qwen/Qwen3.8-27B-FP8`, `Qwen3.5` family — therefore always fall back to the
generic Triton `kernel_paged_attention_2d`, at every batch size, even though
`ROCM_AITER_UNIFIED_ATTN` supports `head_size >= 32` and is much faster at low
concurrency.

`_get_backend_priorities()` puts `ROCM_ATTN` first unconditionally, so the AITER
backend is never auto-selected. This PR prefers `ROCM_AITER_UNIFIED_ATTN` over
`ROCM_ATTN` when the run is configured for a small batch, and keeps the old
order everywhere else.

## Example: Qwen3.8-27B-FP8 on MI300X, 8192 in / 1024 out, bs=1

Measured with `vllm bench serve` (random dataset, `--ignore-eos`), ROCm 10
nightly image, TP=1, 3 repetitions per arm, identical tree; the only difference
is the backend selection:

| arm | mean TPOT | output throughput |
|---|---|---|
| ROCM_ATTN (current default) | 27.53 / 27.55 / 27.52 ms | 35.14 / 35.23 / 35.27 tok/s |
| ROCM_AITER_UNIFIED_ATTN (this PR) | 12.11 / 12.11 / 12.11 ms | 76.02 / 76.62 / 76.67 tok/s |

−56% TPOT, +117% output throughput; run-to-run spread < 0.15%.

## Re-verified on current main (`e0d00e4b8`)

The table above was taken on `2b009fc6d`. Re-measured on current main with the
same flag on both arms (`--max-num-seqs 8`, so the only difference is the
selector), two repetitions per arm, pinned image digest, tree mounted:

| arm | mean TPOT | output throughput |
|---|---|---|
| ROCM_ATTN | 14.771 / 14.775 ms | 65.88 / 66.29 tok/s |
| ROCM_AITER_UNIFIED_ATTN | 5.271 / 5.283 ms | 173.60 / 178.23 tok/s |

−64.3% TPOT, 2.80× output throughput. The same model reproduces the effect on
the newer base, and running both arms with the identical `max_num_seqs` shows
the difference comes from the backend selection rather than from the flag.

Torch-profiler trace of the same workload (50 decode steps): the decode
attention kernel is `kernel_paged_attention_2d` at 1.06 ms/call, 59.1% of GPU
time. SGLang's AITER backend on the same model runs `paged_attention_ll4mi`
at ~0.02 ms/call.

## Why the switch has to be batch-gated

`ROCM_AITER_UNIFIED_ATTN` is not faster at every batch size. Same model, same
method, 8192/512 and 4096/512:

| workload | ROCM_ATTN | AITER unified |
|---|---|---|
| 4096/512 c1 | 18.8 ms / 49.8 tok/s | 11.3 ms / 78.5 tok/s |
| 4096/512 c8 | 20.9 ms / 325.6 tok/s | 15.2 ms / 436.6 tok/s |
| 4096/512 c64 | 59.8 ms / 821.9 tok/s | 87.7 ms / 610.4 tok/s |
| 8192/512 c8 | 31.6 ms / 183.0 tok/s | 23.8 ms / 222.4 tok/s |
| 8192/512 c64 | 99.4 ms / 453.4 tok/s | 142.8 ms / 358.9 tok/s |

So the selector prefers AITER only when the engine is configured for
`max_num_seqs <= 8` (the measured safe point; 8 and 64 were measured, nothing
in between, so the threshold does not extrapolate).

## Validation protocol (runtime gate, 3 fresh servers per arm)

- Throughput (temperature > 0): +117.63% (77.24 vs 35.49 tok/s), TPOT −56.3%,
  E2EL −54.1%, P99 TPOT −56.3%, run-to-run spread 0.4% (patched) / 0.1% (baseline).
- Deterministic pair (temperature = 0): +118.0% (78.04 vs 35.79 tok/s), TPOT −56.6%.
- Prefill sweep (OSL 128): TPOT +1.8% (ISL 16), +0.1% (128), −12.8% (1024),
  −24.2% (2048); TTFT −8.5% / −0.1% / +5.3% / +4.7%.
- `lm_eval` gsm8k, 5-shot, full 1319-sample split, greedy, feature active on both
  arms (`max_num_seqs=8`): flexible-extract 0.7081 vs 0.7165 (−0.0084, 0.48 SE),
  strict-match 0.6839 vs 0.6869 (−0.0030, 0.17 SE). No metric reaches a 0.01
  regression; flexible-extract sits 0.0034 outside a strict ±0.005 band, which is
  expected for a different attention kernel and is within one standard error.

### Trade-offs worth stating

- **TTFT at long prompts**: 1073 vs 956 ms at 8K (+12%), i.e. the AITER unified
  prefill path is marginally slower than the existing chunked-prefill path. End-to-end
  latency is still −54%. If that matters more than decode throughput for a workload,
  the kill switch restores the old behavior; a follow-up could select per phase.
- **Very short contexts** (ISL 16): TPOT +1.8% in favor of `ROCM_ATTN`. The win is
  long-context small-batch decode, which is the target regime.

## What changed

`vllm/platforms/rocm.py`, `_get_backend_priorities()` only:

- when `VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH` is not `1`, AITER is available,
  and the current vLLM config reports `scheduler_config.max_num_seqs <= 8`,
  `ROCM_AITER_UNIFIED_ATTN` is appended first and the later AITER append is
  skipped (including the RDNA4 insert path, to avoid a duplicate entry);
- every other path is untouched: `ROCM_ATTN` stays next in the list as the
  fallback, MLA/sparse early returns are unchanged, and when no vLLM config is
  available the ordering is exactly today's;
- startup log lines: `... is ENABLED (max_num_seqs=N <= 8)` and
  `... is DISABLED (env override)`.

## Tests

`tests/platforms/test_rocm_aiter_small_batch_priority.py` — 20 cases covering:
small/large batch, kill switch, AITER unavailable, missing vllm config,
`use_kv_connector`, RDNA4 insert path, and priority ordering/duplicates.

Manual: `VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH=1` reproduces baseline numerics
(same lm_eval score) and the baseline throughput above.

## Limitations / follow-ups

- The gate reads the *configured* `max_num_seqs`. A server configured for high
  concurrency but serving one request keeps the old backend. Per-step selection
  would need a dynamic decision that the backend interface does not expose
  today; this is a deliberate, conservative first step.
- When a KV connector excludes `ROCM_ATTN` and AITER MHA ops are also enabled,
  small-batch runs now pick unified attention ahead of `ROCM_AITER_FA`. That is
  the intended preference, but it is a selection change beyond the
  `ROCM_ATTN` comparison.
- Independent of this change: `PR #51406` relaxes the `is_cuda()` guard so the
  fused QK-norm+RoPE+gate Triton kernel also runs on ROCm; the two are
  complementary and were measured on the same host.
