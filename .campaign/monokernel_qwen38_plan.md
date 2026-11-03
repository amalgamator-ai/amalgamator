# Can vLLM's AMD MonoKernel run Qwen3.8-27B on a single MI300X?

Checked against upstream `a0d7356b98` (2026-11-03) and the open specialized-decode
PRs. Short answer: **not as-is** — the shipped MonoKernels are model-specific,
refuse TP1, and none targets a dense FP8 model. A realistic port is a
specialized-decode package for the Qwen3.5 layer, roughly a 3–4 month
kernel-engineering project, with a much cheaper FFN-only milestone on the way.

## What exists upstream

`--monokernel` (`KernelConfig.monokernel`) is only a switch: model code opts in.
Nothing consumes it for Qwen3.5, so on our model the flag is a silent no-op.

Shipped / in-flight implementations, all model-owned under
`vllm/models/<model>/<vendor>/` per RFC #60904:

| model | arch | TP | hardware | notes |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash (#62713 brought it to MI325X) | MLA + MoE + mHC + Engram | 2/4/8 | gfx950, gfx942 | 2 persistent launches/layer; `ffn_only` fallback per layer |
| DeepSeek-V4-Pro | MLA + MoE | 2/4/8 | gfx950 | |
| MiniMax-M3, GLM-5.x, Muse Spark, Kimi-K3 | MoE (EAGLE3/DSpark verify) | 2/4/8 | gfx950 | |
| Qwen3.8-Flash-Next, open PR #63032 | GDN + HC + MoE, Quark MXFP4 | **TP2 only** | **gfx950 only** | 4 launches/layer; 23–37% TPOT at 1–8 requests |

Shared contract (RFC #60904): opt-in switch, default-deny refusals in three
stages, one `plan()` per step, Model Runner V2 only, prefill/mixed always stock,
models using `@support_torch_compile` out of scope, speculative decoding refused
by the current specs.

## Why it does not drop into our deployment

1. **Model-specific kernels.** The MonoKernel is a hand-written FlyDSL program
   for one layer layout (seam → attention → FFN). Qwen3.5 is a different layer:
   dense FFN (5120 → 17408), 48 GDN linear-attention + 16 full-attention layers,
   head_dim 256, FP8 W8A8 with 128×128 weight blocks, no MoE, no
   hyper-connections. None of the existing stages matches.
2. **TP1 is refused.** `vllm/models/deepseek_v41/amd/mono_decode.py` raises
   `--monokernel needs tensor parallel size 2, 4 or 8`; Qwen3.8-Flash-Next
   refuses anything but TP2. The kernels' cross-rank mailbox/all-reduce is built
   into the execution model. A single-GPU variant has to strip those stages
   (they are simpler without peers, but it is new code, not a flag).
3. **gfx942 support exists, but only on the DeepSeek path.** `CDNA_VERSIONS =
   (3, 4)` and `mono/common/gfx942.py` (FNUZ FP8, unscaled-MFMA emulation) make
   DSV4.1 work on MI325X/MI300X. Qwen3.8-Flash-Next refuses any platform other
   than unpartitioned gfx950. So the CDNA3 plumbing is proven but not wired to a
   Qwen-shaped kernel.
4. **Runner.** These decodes require Model Runner V2
   (`VLLM_USE_V2_MODEL_RUNNER=1`); our measured stack ran the default runner.
5. **MTP conflicts.** Our best c1 number uses MTP (6.8 ms vs 11.2 ms). The
   current Qwen3.8-Flash-Next spec refuses speculative decoding; Muse Spark's
   does serve verify steps. Mono + MTP is possible but is extra scope.

## What is reusable

- **Framework**: `specialized_decode/runtime/` (guards, plan, hooks,
  model_state), the tagged-mailbox ABI, `FailStop`, the refusal model.
- **Kernel primitives**: `mono/common/` (device math, shuffles, buffer ops) and
  the gfx942 FP8 shims from the DeepSeek-V4.1 package.
- **GDN design**: PR #63032's `mono/stages/gdn.py` + `in_proj.py` is the only
  public example of a MonoKernel running GDN; its weight formats (MXFP4) and
  HC seams do not match, but the structure does.
- **GEMV stage / seam**: from the DeepSeek package, for the FP8 GEMVs.

## Why the upside is real on MI300X

Qwen3.8-27B decode is weight-bandwidth bound and runs at ~50% of peak:

| config | ms/step | effective weight BW |
|---|---|---|
| PR1 at c1 | 11.17 | 2.64 TB/s |
| PR1 + MTP n=1 | 6.82 | 4.33 TB/s |
| MI300X peak | — | 5.3 TB/s |

Dense MLP is ~17.1 GB of the ~29.5 GB language weights (58%) and is identical
in all 64 layers. The RFC's thesis is exactly this regime: at low concurrency
decode is bound by launch overhead and weight reads, and a MonoKernel removes
most of both (GLM-5.2 2.3×, Qwen3.8-Flash-Next 23–37%). A Qwen3.8-27B mono at
~4 TB/s would be ~7.4 ms/step (−34%); at MTP-level 4.33 TB/s, ~6.8 ms.

Note that MTP already reaches 4.33 TB/s, so mono's incremental gain on top of
MTP is the open question; the non-MTP path is where most of the headroom is.

## Plan of action

### Phase 0 — Prove the ceiling (1 week, no kernel code)
- Boot Qwen3.8-27B with `VLLM_USE_V2_MODEL_RUNNER=1` on MI300X; confirm parity
  and that `--monokernel` is a no-op today.
- Trace the decode step at c1 and c8 (with and without MTP): kernel count,
  per-kernel time, inter-kernel gaps, achieved BW per GEMM.
- Microbenchmark pure weight streaming at our shapes (FP8, M=1..8) plus a
  throwaway fused-FFN Triton/FlyDSL probe to measure achievable BW.
- **Gate:** projected gain ≥ 25% at c1, else stop.

### Phase 1 — Skeleton (2–3 weeks)
- New package `vllm/models/qwen3_5/amd/specialized_decode/` per RFC #60904,
  starting from the Qwen3.8-Flash-Next runtime skeleton; TP1, gfx942, dense,
  FP8; refusal list covering our constraints.
- `plan()` for 1–16-row decode, vLLM forward otherwise; hooks behind
  `--monokernel`; weights read in place.

### Phase 2 — Dense FFN MonoKernel (3–4 weeks) ← first measurable milestone
- One persistent launch per layer: residual + norm + FP8 quant → gate/up GEMV
  → silu → down GEMV → residual.
- Numerics vs vLLM's modules with real weights; per-layer timing.
- **Gate:** ≥ 20% layer-time improvement; this alone covers 58% of the bytes.

### Phase 3 — GDN and attention (4–6 weeks)
- Port the GDN block from PR #63032 (in_proj_qkvz/ba, conv update, gated delta
  rule, RMSNormGated) to FP8/gfx942 shapes, reusing AITER's FlyDSL GDN device
  code where linkable.
- Full-attention layers first stay on AITER FA between mono launches (the GLM
  layer-mode pattern); a FlyDSL decode-attention kernel is a later step.

### Phase 4 — Graphs, MTP, end-to-end (2–3 weeks)
- FULL_AND_PIECEWISE graph replay, 1–16 rows; then a verify-step mode for MTP
  (Muse Spark pattern) or ship non-MTP first.
- c1/c8 benches, gsm8k/GPQA parity, no c8 regression.

### Phase 5 — Upstream (1–2 weeks)
- Kernel/routing tests, Buildkite MI300 step (the test-tethering gate), recipe
  docs, RFC review.

**Effort:** ~3–4 months with 1–2 engineers who know FlyDSL/AITER. Unlike the
TP8 kernels, TP1 removes peer memory and all-reduces, but the GDN FP8 device
port is the main risk; the gfx942 codegen path is proven only for DeepSeek's
shapes.

## Cheaper alternatives if the port is not funded

1. **Adopt the existing path**: Qwen3.8-Flash-Next (#63032) on MI355X TP2 is a
   different checkpoint and hardware, but if the deployment can move, the work
   is already written.
2. **FFN-only mini-mono** (Phase 2 alone): less risk, partial gain, and it is
   the natural stepping stone.
3. **Non-mono levers already measured**: MTP is the dominant one (+57% at c1,
   131–155 tok/s), fp8 KV ~1%, both independent of this effort.
