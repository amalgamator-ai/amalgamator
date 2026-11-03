# Qwen3.8-27B on one MI300X — what is left after the attention fix

Overnight run 2026-11-02 → 11-03. Every number below is `vllm bench serve`,
random dataset, `--ignore-eos`, 8192 in / 1024 out, ROCm 10 image pinned by
digest (`vllm/vllm-openai-rocm@sha256:4c92e167…`, vLLM `76c6f3f2a9`), each arm a
separate source-mounted worktree, one server at a time on the single GPU.
Raw JSON in `~/qwen38/results/` on the box; collect with
`~/qwen38/collect_q38.py`.

## Where the model stands

c1 (= the decode cell PR 1 fixed):

| config | TPOT | output tok/s | vs base |
|---|---|---|---|
| stock vLLM | 27.46 / 27.42 ms | 35.4 | — |
| + PR 1 (attention ordering) | 11.16 / 11.18 ms | 83.4 | −59.4% |
| + fp8 KV cache | 11.06 / 11.05 ms | 84.2 | −1.0% |
| + MTP spec decode, n=1 | 6.59 / 7.06 ms | 130.9 | −39% |
| + MTP, n=2 | 5.67 / 6.77 ms | 142.7 | −44% |
| + MTP, n=3 | 4.76 / 6.75 ms | 155.4 | −48% |
| + MTP, n=1, fp8 KV | 7.11 / 7.07 ms | 126.4 | −36% |
| P3 fused-QK kernel (code) | 11.13 / 11.12 ms | 83.8 | 0% |

c8 (8192 in / 1024 out, concurrency 8):

| config | TPOT | output tok/s | vs base |
|---|---|---|---|
| + PR 1 | 15.93 / 15.98 ms | 395 | — |
| + MTP, n=1 | 13.58 / 13.43 ms | 485 | −14.8% |

Measurement caveat: in every MTP arm the first run after boot is faster than
the second (n=1 6.59/7.06, n=3 4.76/6.75). The per-run acceptance rate moves
with it (38.6%–86.6%), so the spread is mostly acceptance, not a stable
draft-depth ordering. Treat these as ranges; the non-MTP arms are stable to
±0.02 ms.

## Why the previous plan stops applying

The 2026-10-31 cross-trace for this model put 16,593 µs/step (97.5% of the gap
to SGLang) in the Triton paged-attention fallback. PR 1 removes exactly that.

## Fresh trace on the post-PR-1 tree

Torch trace through `auto_profile_server_client`, tree source-mounted. Kernel
census is also the engagement check for PR 1: 752 calls of
`paged_attention_ll4mi_QKV_mfma16` plus 752 reduce-kernel calls — 47 profiled
steps × 16 full-attention layers — and **zero** `kernel_paged_attention_2d`.

| component | before PR 1 | after PR 1 |
|---|---|---|
| paged attention | 16,593 µs/step | 571 µs/step |
| share of the step | ~97% | ~5% |

The step is no longer attention, and the rest of it is weight traffic. Measured
from the safetensors headers: 29.47 GB of language-model weights (+0.48 GB MTP
layer; 0.92 GB vision tower, not read at decode).

| config | effective weight bandwidth |
|---|---|
| PR 1 at c1 (11.17 ms/step) | 2.64 TB/s |
| MTP n=1 (6.82 ms/step, 1.75 tokens) | 4.33 TB/s |
| MI300X peak | 5.3 TB/s |

So the decode step is **weight-bandwidth bound**, and MTP is what recovers the
headroom: it amortises one weight read over ~1.8 accepted tokens. Any
compute-side change can only touch the ~5% of the step that is not the GEMM
path.

The amalgamator's single-trace pass
(`llm/auto_analyze/results/results_analyze_qwen38_post_p1/vllm/`) agrees and
adds the structure:

- 64 layers in a `[3× GDN linear attention, 1× full attention]` pattern (48
  GDN, 16 full attention), `head_dim=256`, 4 KV heads, dense MLP 17408.
- Server log confirms `ROCM_AITER_FA` (PR 1 engaged), `FULL_AND_PIECEWISE` CUDA
  graphs, AITER FlyDSL GDN, `AiterFp8BlockScaledMMKernel`, and the
  `norm_quant` / `act_quant` compile fusions already on.
- Both remaining code targets in that plan are confirmed present and disabled:
  fused QK-norm+RoPE+gate (`use_fused_qk_norm_rope_gate = False` on ROCm) and
  the fused sigmoid output gate (`use_fused_output_gate = False`). Together
  they are ~207 µs of an 11.17 ms step.
- GDN decode already runs as two fused AITER Triton kernels, so the linear
  layers are not a soft target either.

## Code change tested: fused QK-norm+RoPE+gate on ROCm (plan proposal P3)

`qwen3_next.py` gates vLLM's own Triton kernel on CUDA/XPU; widening the guard
by one term lets Qwen3.5/3.8 take it on ROCm. Patch:
`.campaign/p3_fused_qk_norm_rocm.patch`.

| arm | TPOT | output tok/s |
|---|---|---|
| base | 11.117 / 11.131 ms | 83.7 / 83.6 |
| P3 | 11.134 / 11.118 ms | 83.6 / 83.7 |

No measurable change. The greedy outputs of the two arms diverge after ~7
tokens, which looked like a kernel bug, but **the same server asked the same
greedy prompt twice returns different text from the same token** (verified on
the unpatched tree). Decode on this stack is nondeterministic — fp8 GEMM
reduction order / AITER GDN atomics — so a text comparison cannot attribute
anything to the patch, and accuracy gates for this model must be aggregate
(gsm8k), not exact-match.

P3 stands as perf-neutral, numerics untested, and is not proposed for a PR.

## Recommended configuration

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

No code change is needed: vLLM carries `Qwen3_5MTP`, and the checkpoint ships
the draft layer (`mtp.*`, 22 tensors). n=1 is the safe choice (best worst-case,
and at 4.33 TB/s already near the roofline); n=2/n=3 improve the best case but
not the worst.

## What is not worth a PR

- **P2 (GDN output flatten, 212 µs projected)**, **P3 (133 µs)**, **P4 (74 µs)**:
  all three sit in the ~5% of the step that is not weight streaming, and P3
  measures as zero. On an 11.17 ms step they are 1–1.9%; on a 6.8 ms MTP step
  they would be 1–3%, still at the edge of what this GPU resolves when
  acceptance varies run to run.
- Deeper spec decode beyond n=2 is the only lever with real headroom left, and
  it is configuration, not code.

## Open items

- The MTP first-run/second-run asymmetry (acceptance 86.6% → 74.7% on
  identical back-to-back runs) is unexplained. It is worth one controlled run
  with more reps before quoting a single MTP number in any PR or doc.
- No SGLang re-measurement on this image tonight; the comparison still cites
  the 2026-10-31 SGLang figure (~76 tok/s), against which vLLM+PR 1 (83.4) and
  vLLM+MTP (131) are both ahead.
- Accuracy (gsm8k) for the MTP configuration was not run; it is the one gate
  missing before recommending MTP in writing.
