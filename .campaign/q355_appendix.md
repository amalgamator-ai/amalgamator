================================================================================
APPENDIX A: MEASURED CONSTRAINTS FOR THIS HOST (binding, added by the operator)
================================================================================
Same MI300X, same ROCm 10 images, TP=1, `vllm bench serve` (random dataset,
--ignore-eos). Four-arm matrix: framework x spec-decode, six cells each.

A.1 MATRIX (TPOT ms / output tok/s) — this model, no other patches applied
    cell                | vllm no-spec | vllm +MTP | sglang no-spec | sglang +EAGLE
    8192/1024 c1        | 15.08 / 64.6 | 8.32/114.5|  4.55 / 210.8  |  4.99 / 188.5
    8192/128  c1        | 14.42 / 59.0 | 11.39/70.9|  4.54 / 163.2  |  5.78 / 126.9
    4096/512  c1        |  9.76 / 95.0 |  6.23/141.7| 4.49 / 200.6  |  4.49 / 198.8
    4096/512  c8        | 12.01 / 577  | 11.70/537 |  6.94 / 903    |  7.91 / 823
    4096/512  c64       | 25.87 / 2014 | 77.04/742 | 21.98 / 2181   | 31.23 / 1777
    1024/512  c64       | 16.90 / 3292 |    -      | 15.89 / 3466   |    -
    Reading: SGLang wins everywhere; the win is largest at decode-heavy low
    concurrency (2.1x-3.3x at c1) and shrinks to ~1.05x at short-prompt c64.
    vLLM MTP spec helps c1 (15.08 -> 8.32) but collapses at c64 (25.87 -> 77.04).
    SGLang EAGLE slightly HURTS at c1 (4.55 -> 4.99). Spec decode is not the lever.

A.2 BACKEND SWEEP (vllm, no spec, 8192 in)
    backend                 | c1             | c8             | c64
    default (ROCM_ATTN)     | 15.06 / 64.3   | 17.58 / 380.5  | 39.05 / 1226
    ROCM_AITER_UNIFIED_ATTN |  5.27 / 165.8  | 10.45 / 543.0  | 47.45 / 1090

A.3 THE ATTENTION-SELECTION GAP IS ALREADY FIXED — DO NOT PROPOSE IT AGAIN
    The `_get_backend_priorities()` small-batch gate (prefer
    ROCM_AITER_UNIFIED_ATTN when max_num_seqs <= 8) is written, unit-tested and
    runtime-verified on Qwen3.8-27B, and it reproduces on THIS model at 2.86x
    (15.06 -> 5.27 ms at c1). It is applied to the baseline tree for this
    campaign; do not re-derive, re-implement or re-measure it, and do not
    propose an unconditional priority reorder (A.2 shows c64 regresses by 21%).

A.4 WHAT IS LEFT (the actual target for this campaign)
    With the selector fixed, the remaining gap is 5.27 vs 4.55 ms at the target
    cell = ~16% (1.16x). That residual is NOT attention: it is the MoE / GDN /
    quantization path. Ranked hypotheses, from this model's shape
    (256 experts, top-8, shared expert, 30 GDN + 10 full-attention layers):
      H1 Fused shared-expert path: vLLM gates per-layer fused shared expert
         selection (see upstream #61457); check whether it is active for this
         FP8 MoE on ROCm, and whether SGLang's shared-expert handling is fused.
      H2 Fused gated RMSNorm for QUANTIZED models: `+rms_norm_gated` is default-on
         only for unquantized models in `vllm/platforms/rocm.py`. Measured on
         Qwen3.8-27B this fusion saves ~211 us/step of kernel time but was
         invisible end-to-end there (0.2%); this model has 30 GDN layers and a
         much larger MLP, so re-measure rather than assuming.
      H3 MoE grouped-GEMM / top-k on ROCm: expert routing, grouped-gemm kernel
         choice, and the aiter FMOE tuned-config path.
      H4 GDN decode kernels: vLLM has had recent ROCm GDN work (#60645/#61855/
         #64419/#64425 merge dates 2026-10-14 .. 2026-10-29), so check whether
         the fast paths are actually engaged for this model before proposing
         anything new.

A.5 IMPLEMENTATION CONSTRAINTS
    - Pure Python unless a compiled change is unavoidable; the image can build
      vLLM in-container (clang-23, ~10 min for 72 objects) if it truly is.
    - Mandatory: env-var kill switch + startup ENABLED/DISABLED log lines, so the
      harness can drive a real baseline arm.
    - Default (kill switch set) must be identical to today's behaviour.
    - No regression at c64 or at ISL 16-128: those were measured and must be
      re-measured after the change.
    - The runtime gate bench is 8192 in / 1024 out, max-concurrency 1,
      --max-num-seqs 1.
