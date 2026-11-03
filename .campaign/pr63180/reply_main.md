Thanks — the tuned-config question is the real difference between my runs and yours, so let me put the exact setup here.

## What was injected

Both arms get the same file; the only per-arm difference is the kill switch:

```bash
# docker run, both arms
-e AITER_CONFIG_FMOE=/rocm10/tuned/combined_a8w4_5120x640_v1.csv
# OFF arm only
-e VLLM_DISABLE_FUSED_SHARED_EXPERTS=1
```

Without this, the in-image CSVs have no `5120/640` rows for any shape and **both** arms fall back to heuristics — which is exactly the line you posted. We hit it too on the first cycle: `no tuned FlyDSL config for ('gfx950', 256, …, 5120, 640, 385, 7, …)` and the `384, 6` variant, both attached.

## What is in the file

35 rows, the standard `tuned_fmoe.csv` 14-field schema, all `gfx950` / `cu_num=256` / `model_dim=5120` / `inter_dim=640`:

| shape | rows | token tiers | stage-2 kernels |
|---|---|---|---|
| 385 / 7 (fold) | 15 | 1 … 16384 | `flydsl_moe2_afp8_wfp4_bf16_*_atomic*`, **no** `_moe2_layout_` |
| 384 / 6 (stock) | 15 | 1 … 16384 | `flydsl_moe2_layout_*` (stock wrapper accepts these) |
| 128 / 3 (draft) | 5 | 64 … 8192 | `flydsl_moe2_layout_*` |

## Which lookup — your first one, on purpose

The **fused-MoE key**, not the FHMOE key. The fold appends a slot to the routed grouped GEMM (`E 384→385`, `topk 6→7`) and keeps the shared expert's native FP8 block-32 weights, so nothing about it needs the FHMOE entry point (shared_expert_id / hidden_pad / intermediate_pad / gate_mode). That is also why the missing FHMOE support in `gemm_moe_tune.py` doesn't block this PR — we never need that namespace.

## It is tuned, not heuristic

The 385/7 rows come from AITER's own tuner, run inside the pinned image:

```bash
TUNE_ONLY=flydsl AITER_PERFTEST_CUDA_EVENT=1 \
python3 .../ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i <fused input> -o <out>
```

`TUNE_ONLY` matters: it defaults to `flydslv2`, which picked `_moe2_layout_` stage-2 kernels for every fused tier, and the heterogeneous gate (`_use_fhmoe_wrappers`, the one behind `_probe_fhmoe_default_config_capability`) rejects layout kernels. That is the non-reproducible state you're seeing. Re-running with `TUNE_ONLY=flydsl` produced the v1-family rows in the attached CSV, merged with the untouched v2 routed rows by `build_combined_a8w4.py`.

## Evidence it engaged at runtime

```
census …(on):  fse_arm=1 fse_on=1 mono=1 canary=0 miss385=0 miss384=0
census …(off): fse_arm=0 fse_on=0 mono=1 canary=0 miss385=0 miss384=0
```

`miss385=0` means every 385/7 lookup resolved to a tuned row, not the fallback.

Attached: the CSV, the builder, the census lines, and our own pre-tuning miss lines for comparison.
