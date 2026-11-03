
### Problem

On CDNA, `ROCM_ATTN` implements paged attention with a custom kernel for exactly two head sizes, 64 and 128. For any other head size it silently falls back to a generic Triton paged-attention kernel. That kernel is slow: in a traced Qwen3.5-35B run (`head_dim=256`, batch 1) it cost about 1 ms per attention layer per step, and it is what every model with a larger head size runs, at every batch size.

The AITER backends do cover those head sizes and are several times faster, but they are never selected. `_get_backend_priorities()` puts `ROCM_ATTN` first and `ROCM_ATTN.is_backend_compatible()` always returns true — it owns that Triton fallback, so it looks usable for every head size — so the selector stops at the first entry and never reaches the AITER ones.

The fallback is therefore not just a slow path; it is a path that shadows faster backends that are available and would be selected if they were tried first.

### Change

When the model's head size has no custom `ROCM_ATTN` kernel, move the AITER entries ahead of it: `ROCM_AITER_FA`, then `ROCM_AITER_UNIFIED_ATTN`, then `ROCM_ATTN`. For head sizes 64 and 128 the order is unchanged, so nothing changes for models that already use the custom kernel.

The function reorders the entries it already produced rather than appending new ones, so the rest of the list keeps its relative order and no backend can appear twice.

### Results

`vllm bench serve`, random dataset, `--ignore-eos`, single MI300X (gfx942), ROCm 10 image pinned by digest, 8192 in. Each arm is a separate git worktree; two runs per cell. Numbers are from vLLM `76c6f3f2a9` (the 2026-11-02 nightly), source-mounted into the image, so the patch is the only difference between arms.

| model | cell | backend, before → after | before | after | change |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-FP8 | c1, 1024 out | `ROCM_ATTN` → `ROCM_AITER_FA` | 14.810 / 14.810 ms, 66.1 tok/s | 4.644 / 4.611 ms, 203 tok/s | −68.8% |
| Qwen3.5-35B-A3B-FP8 | c8, 1024 out | `ROCM_ATTN` → `ROCM_AITER_FA` | 18.188 / 18.047 ms, 402 tok/s | 7.866 / 7.666 ms, 865 tok/s | −57.1% |
| Qwen3.5-35B-A3B-FP8 | c64, 512 out, 128 prompts | `ROCM_ATTN` → `ROCM_AITER_FA` | 49.24 / 47.85 ms, 1022 / 1049 tok/s | 34.72 / 33.84 ms, 1372 / 1423 tok/s | −29.4% |
| Qwen3.8-27B-FP8 | c1, 1024 out | `ROCM_ATTN` → `ROCM_AITER_FA` | 27.46 / 27.42 ms, 35.4 tok/s | 11.16 / 11.18 ms, 83.4 tok/s | −59.4% |

c1 and c8 use `--max-num-seqs 8`; the c64 cell uses `--max-num-seqs 64`. All deltas are TPOT.

The backend column is the selector's actual decision, read from the startup log rather than inferred: the stock tree (checked with Qwen3.5) logs `Overriding with ROCM_ATTN`, and the patched tree logs `Overriding with ROCM_AITER_FA` for both models at `head_size=256`. The stock Qwen3.8 arm was not booted separately — same tree, same head size, same priority list. On the stock tree `ROCM_ATTN` is what puts these models on the Triton fallback. Selection does not depend on `max_num_seqs`, so c1, c8 and c64 all pick the same backend.

`ROCM_AITER_FA` comes before unified because it measured faster on every cell tried. On Qwen3.5-35B — the 4.6 ms row above — the same harness and base gave 4.55 ms with FA vs 5.02 ms with unified at c1, and 8.15 vs 10.55 ms at c8 (bf16 KV); with fp8 KV the two were level (4.42 vs 4.47 ms at c1). The 11.18 ms row is Qwen3.8-27B, a dense model, and is not part of this comparison.

### Constraints

- **Only for head sizes without a custom kernel.** 64 and 128 keep the old order; the change is a no-op there.
- **Only on MI300 (gfx942)**, the architecture it was measured on. Every other ROCm GPU returns before touching the order.
- **`VLLM_DISABLE_AITER_BACKEND_PRIORITY=1` restores the old order** everywhere, including gfx942.
- **Degrades to today's behaviour** when AITER is not available, when MHA is disabled, or on the KV-connector path where `ROCM_ATTN` is excluded — the reorder only moves entries the list already contains.

### Evolution

The scoping is deliberately conservative: it is where we have numbers today, not a claim that the ordering is wrong elsewhere. There is no MI300-specific mechanism in the change, and we expect to relax it to all ROCm GPUs once we have measurements on MI350 and the other parts. The kill switch stays available for anyone who wants the old order back in the meantime.

Longer term this static ordering could become a measurement instead of a rule: the fast path is the one with a real kernel for the model's head size, and the fallback should always sort last, whatever the backend list looks like in future.

### Accuracy

gsm8k, 5-shot, full split (1329 samples), two runs per arm. Flexible-extract: `ROCM_ATTN` 0.7938 / 0.7870, AITER FA 0.7832 / 0.7945 (means 0.7904 vs 0.7889, a 0.0015 difference against a 0.011 within-arm spread). Strict-match: 0.7748 / 0.7627 vs 0.7650 / 0.7794. No measurable difference either way.

### Tests

`tests/platforms/test_rocm_aiter_backend_priority.py`, 12 cases: head sizes without a custom kernel prefer AITER, head sizes 64/128 keep the old order, other architectures keep the old order too, the MHA-off fallback to unified, the kill switch, AITER being unavailable, the KV-connector path where `ROCM_ATTN` is excluded, and no backend appearing twice.

Amalgamator was used for AI assistance: https://github.com/amalgamator-ai/amalgamator
