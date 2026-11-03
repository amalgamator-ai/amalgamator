# Qwen3.5-35B-A3B on one MI300X — measured levers on top of P1 (working notes)

Cell for all rows unless noted: ISL 8192, `vllm bench serve --ignore-eos`, TP=1,
nightly-rocm100 image, TPOT in ms / output tok/s.

## Baseline and P1

| arm | TPOT | tok/s | note |
|---|---|---|---|
| stock, default mns=256 (gate off) | 15.08 | 64.6 | ROCM_ATTN (Triton paged attn, head_dim 256 not eligible for AITER custom paged) |
| P1, max-num-seqs 8 | 5.21 | 174.9 | ROCM_AITER_UNIFIED_ATTN selected — reproduces the Qwen3.8-27B result |
| SGLang (AITER attn, fp8 KV, page 16) | 4.55 | 210.8 | reference |

## Levers, measured on top of P1

| lever | c1 | c8 | c64 |
|---|---|---|---|
| fp8 KV (`--kv-cache-dtype fp8_e4m3`) | 4.59 / 201.5 | 8.28 / 657.9 | — |
| MTP spec (`num_speculative_tokens=3`) | 3.64 / 248.1 | 9.85 / 627.4 | **360.16 / 155.2** |
| fp8 KV + MTP | **2.99 / 298.0** | — | — |
| AITER tuned FMOE csv | 5.16 / 179.6 | — | — (neutral) |
| no lever (P1 only) | 5.21 / 174.9 | 10.79 / 554.9 | 49.70 / 1004.5 |

c8/c64 cells use OSL 512 (32/128 prompts); c1 cells OSL 1024 (8 prompts).

**Read:** the three levers are independent and additive. At c1 vLLM is 1.52x
faster than SGLang with fp8 KV + MTP. At c8 SGLang still leads by ~19%. At c64
MTP is catastrophic.

## The MTP collapse

Per-step cost (TPOT x mean acceptance length, acceptance 2.4-2.9 measured):

| c | MTP step | no-spec step | ratio |
|---|---|---|---|
| 1 | ~10.6 ms | 5.21 ms | 2.0x |
| 8 | ~28.6 ms | 10.79 ms | 2.7x |
| 16 | ~97 ms | — | — |
| 32 | ~307 ms | — | — |
| 64 | ~1044 ms | 49.7 ms | 21x |

- Acceptance is NOT the problem: mean acceptance length 2.41-2.91 at c16/c32.
- Scheduling is NOT the problem: `vllm:num_requests_running` = 62-63 of 64 with
  0 waiting and KV cache at 23% — upstream issue #55533 (Mamba speculative block
  accounting starving the scheduler) does not apply on a 192 GiB pool.
- So the collapse is in the decode code path and scales ~O(batch^1.7).
- Server log at mns=64: "Attention backend(s) ROCM_ATTN cannot update draft
  decode metadata in place; falling back to rebuilding attention metadata eagerly."
  (mns=64 disables the P1 gate, so the arm runs on ROCM_ATTN, not AITER.)

Open: kernel-level attribution — torch traces of c64 with and without MTP are
being captured (`mtpprof_q355.sh`) to feed a single-trace + cross-commit analysis.
