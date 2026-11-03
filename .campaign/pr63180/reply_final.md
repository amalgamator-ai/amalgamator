@fxmarty-amd thanks for the detailed review, this is really helpful! 

The tuned-config is a good question, here is the exact setup.

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

Inline below: the tuned CSV (the file `AITER_CONFIG_FMOE` pointed at), the script that builds it, the per-arm census lines, and our own pre-tuning miss lines for comparison.

<details><summary>tuned CSV — combined_a8w4_5120x640_v1.csv</summary>

```csv
gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,xbf16,flat,tflops,bw,nt
gfx950,256,64,5120,640,128,3,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,121.7534,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8,0.0%,65.8864,flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_reduce_sbm32,0.0%,187.6398,0,0,0,20.12,6711.13,0
gfx950,256,128,5120,640,128,3,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,122.1677,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8,0.0%,65.6675,flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32,0.0%,187.8352,0,0,0,40.19,6709.38,0
gfx950,256,256,5120,640,128,3,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,122.5035,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8,0.0%,66.9811,flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32,0.0%,189.4846,0,0,0,79.69,6661.35,0
gfx950,256,512,5120,640,128,3,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,124.2275,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8,0.0%,68.0588,flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32,0.0%,192.2863,0,0,0,157.05,6584.74,0
gfx950,256,8192,5120,640,128,3,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,282.7622,flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w2_bnt0_gui_xcd4_fp8,0.0%,216.6919,flydsl_moe2_layout_afp8_wfp4_bf16_t64x256x128_reduce_sbm64,0.0%,499.4541,0,0,0,967.42,2771.27,0
gfx950,256,1,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,68.9269,flydsl_moe1_afp8_wfp4_bf16_t64x256x256_bnt0_gui_fp8,0.0%,36.9215,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_sbm64,0.0%,105.8484,0,0,0,1.11,35663.17,0
gfx950,256,2,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,68.6286,flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_bnt0_gui_xcd4_kw4_fp8,0.0%,37.1888,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32,0.0%,105.8174,0,0,0,2.23,35673.76,0
gfx950,256,4,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,68.4805,flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_bnt0_gui_xcd4_kw2_fp8,0.0%,40.6676,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32,0.0%,109.1481,0,0,0,4.32,34585.44,0
gfx950,256,8,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,77.8295,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8,0.0%,45.489,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32,0.0%,123.3185,0,0,0,7.65,30611.76,0
gfx950,256,16,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,103.5248,flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_fp8,0.0%,57.5664,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32,0.0%,161.0912,0,0,0,11.72,23434.67,0
gfx950,256,32,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,150.4213,flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w3_gui_kw2_fp8,0.0%,82.6902,flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_nt_sbm32,0.0%,233.1115,0,0,0,16.19,16195.53,0
gfx950,256,64,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,242.4107,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_xcd4_fp8,0.0%,129.5717,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32,0.0%,371.9824,0,0,0,20.3,10150.63,0
gfx950,256,128,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,242.8348,flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw2_fp8,0.0%,129.4343,flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_nt_sbm32,0.0%,372.2691,0,0,0,40.56,10145.46,0
gfx950,256,256,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,242.2668,flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w3_gui_kw2_fp8,0.0%,133.9484,flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_reduce_nt_sbm32,0.0%,376.2152,0,0,0,80.27,10044.27,0
gfx950,256,512,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,243.0262,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8,0.0%,139.6795,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_reduce_nt_sbm32,0.0%,382.7057,0,0,0,157.82,9884.2,0
gfx950,256,1024,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,246.1659,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8,0.0%,137.3112,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32,0.0%,383.4771,0,0,0,315.0,9884.82,0
gfx950,256,2048,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,251.6875,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8,0.0%,154.3164,flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32,0.0%,406.0039,0,0,0,595.05,9375.11,0
gfx950,256,4096,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,276.9364,flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w4_gui_fp8,0.0%,230.1515,flydsl_moe2_layout_afp8_wfp4_bf16_t64x256x128_reduce_nt_sbm64,0.0%,507.0879,0,0,0,952.86,7568.29,0
gfx950,256,8192,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,128,0,399.413,flydsl_moe1_afp8_wfp4_bf16_t128x128x256_w3_gui_fp8,0.0%,418.6694,flydsl_moe2_layout_afp8_wfp4_bf16_t64x256x128_reduce_sbm128,0.0%,818.0824,0,0,0,1181.26,4768.1,0
gfx950,256,16384,5120,640,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,128,0,739.1669,flydsl_moe1_afp8_wfp4_bf16_t128x256x256_w4_bnt0_gui_fp8,0.0%,758.1686,flydsl_moe2_layout_afp8_wfp4_bf16_t64x256x128_reduce_sbm128,0.0%,1497.3355,0,0,0,1290.78,2689.13,0
gfx950,256,1,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,74.7099,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8,4.9%,49.078,flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2_xcd4_persist,0.0%,123.7879,0,0,0,1.11,30574.23,0
gfx950,256,2,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,74.5063,flydsl_moe1_afp8_wfp4_bf16_t16x128x256_gui_fp8,5.3%,50.0008,flydsl_moe2_afp8_wfp4_bf16_t16x128x128_atomic_bnt2,0.0%,124.5071,0,0,0,2.21,30397.74,0
gfx950,256,4,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,76.8972,flydsl_moe1_afp8_wfp4_bf16_t16x256x256_w3_bnt0_gui_fp8,5.4%,54.3485,flydsl_moe2_afp8_wfp4_bf16_t16x128x128_atomic_persist,0.0%,131.2457,0,0,0,4.19,28837.25,0
gfx950,256,8,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,86.2633,flydsl_moe1_afp8_wfp4_bf16_t16x128x256_w3_gui_fp8,5.5%,60.1795,flydsl_moe2_afp8_wfp4_bf16_t16x256x128_atomic_bnt2_persist,0.0%,146.4428,0,0,0,7.52,25845.09,0
gfx950,256,16,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,108.9455,flydsl_moe1_afp8_wfp4_bf16_t16x256x256_w2_gui_fp8,5.2%,71.0833,flydsl_moe2_afp8_wfp4_bf16_t16x256x128_atomic_persist,0.0%,180.0288,0,0,0,12.23,21024.13,0
gfx950,256,32,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,149.2079,flydsl_moe1_afp8_wfp4_bf16_t16x128x256_w4_gui_fp8,5.5%,89.8102,flydsl_moe2_afp8_wfp4_bf16_t16x128x256_atomic_persist,0.0%,239.0181,0,0,0,18.43,15836.44,0
gfx950,256,64,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,196.2011,flydsl_moe1_afp8_wfp4_bf16_t16x128x256_w3_gui_fp8,5.1%,124.6782,flydsl_moe2_afp8_wfp4_bf16_t16x128x256_atomic_bnt2_persist,0.0%,320.8793,0,0,0,27.45,11797.85,0
gfx950,256,128,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,234.5869,flydsl_moe1_afp8_wfp4_bf16_t16x256x256_w2_gui_fp8,5.2%,147.0927,flydsl_moe2_afp8_wfp4_bf16_t16x128x128_atomic_persist,0.0%,381.6796,0,0,0,46.15,9921.07,0
gfx950,256,256,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,248.0571,flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw2_fp8,5.2%,157.4275,flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_persist,0.0%,405.4846,0,0,0,86.89,9343.48,0
gfx950,256,512,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,260.614,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8,5.2%,162.6524,flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic,0.0%,423.2664,0,0,0,166.48,8960.24,0
gfx950,256,1024,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,269.8977,flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8,5.2%,172.7943,flydsl_moe2_afp8_wfp4_bf16_t32x256x256_atomic_bnt2,0.0%,442.692,0,0,0,318.34,8584.82,0
gfx950,256,2048,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,306.1946,flydsl_moe1_afp8_wfp4_bf16_t64x256x256_w3_gui_fp8,5.2%,207.1245,flydsl_moe2_afp8_wfp4_bf16_t64x256x256_atomic_persist,0.0%,513.3191,0,0,0,549.09,7434.29,0
gfx950,256,4096,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,128,0,429.8848,flydsl_moe1_afp8_wfp4_bf16_t128x256x256_w2_gui_fp8,5.2%,321.2094,flydsl_moe2_afp8_wfp4_bf16_t64x128x128_atomic_sbm128,0.0%,751.0942,0,0,0,750.52,5122.68,0
gfx950,256,8192,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,611.638,flydsl_moe1_afp8_wfp4_bf16_t64x256x256_w2_bnt0_gui_xcd4_fp8,5.2%,481.5438,flydsl_moe2_afp8_wfp4_bf16_t64x256x256_reduce_bnt2_persist,0.0%,1093.1818,0,0,0,1031.33,3577.2,0
gfx950,256,16384,5120,640,385,7,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,965.5527,flydsl_moe1_afp8_wfp4_bf16_t64x256x256_w4_bnt0_gui_xcd4_fp8,5.2%,841.2768,flydsl_moe2_afp8_wfp4_bf16_t64x256x128_reduce,0.0%,1806.8295,0,0,0,1247.96,2233.95,0
```

</details>

<details><summary>pre-tuning miss lines (our run, before the CSV existed)</summary>

```
no tuned FlyDSL config for ('gfx950', 256, 1024, 5120, 640, 384, 6, <ActivationType.Silu: 0>, 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False), using heuristic FlyDSL fallback (kn1='flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui', kn2='flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2')
no tuned FlyDSL config for ('gfx950', 256, 1024, 5120, 640, 385, 7, <ActivationType.Silu: 0>, 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False), using heuristic FlyDSL fallback (kn1='flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui', kn2='flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2')
no tuned FlyDSL config for ('gfx950', 256, 128, 5120, 640, 384, 6, <ActivationType.Silu: 0>, 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False), using heuristic FlyDSL fallback (kn1='flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui', kn2='flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2')
no tuned FlyDSL config for ('gfx950', 256, 128, 5120, 640, 385, 7, <ActivationType.Silu: 0>, 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False), using heuristic FlyDSL fallback (kn1='flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui', kn2='flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2')
```

</details>

<details><summary>per-arm census</summary>

```
  arm env: <none>
census r10w2_c5r9_tp4_on (on): fse_arm=1 fse_on=1 mono=1 canary=0 miss385=0 miss384=0
census r10w2_c5r9_tp4_on (on): fse_arm=1 fse_on=1 mono=1 canary=0 miss385=0 miss384=0
== r10w2_c5r9_tp4_on engaged/ENABLED lines:
      1 10-28 21:40:53 [rocm.py:1196] prefill fusion above mono: defaulting VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 for the heterogeneous native-FP8 fused shared experts (the AITER router/kernel coordination flag); the FP8->MXFP4 requant fusion stays refused under --monokernel. Set VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO=1 to keep the stock behavior.
      1 10-28 21:41:15 [model.py:499] prefill fusion above mono engaged: the heterogeneous fused shared experts (native-FP8) run above --monokernel decode (prefill-shaped batches of 16..2048 tokens only; mono decode keeps the stock 384/6 kernels on decode-only steps; disable with VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO=1)
  arm env: VLLM_DISABLE_FUSED_SHARED_EXPERTS=1
census r10w2_c5r9_tp4_off (off): fse_arm=0 fse_on=0 mono=1 canary=0 miss385=0 miss384=0
census r10w2_c5r9_tp4_off (off): fse_arm=0 fse_on=0 mono=1 canary=0 miss385=0 miss384=0
== r10w2_c5r9_tp4_off engaged/ENABLED lines:
  arm env: <none>
== r10w2_c5r9_tp2_on engaged/ENABLED lines:
```

</details>

<details><summary>build_combined_a8w4.py — how the CSV was assembled</summary>

```python
#!/usr/bin/env python3
"""Build the combined a8w4 tuned_fmoe CSV for the crit5 long-ISL A/B.

Union of the campaign's fused-shape tuning (hetero_tuned_fused_v1.csv) and
the raw hetero tuning sweep (hetero_tuned_raw.csv), restricted to the exact
signatures this server looks up at runtime (verified against the cycle-1
"no tuned FlyDSL config" miss lines):

  ('gfx950', 256, M, 5120, 640, 385, 7, Silu, bf16, fp8e4m3, fp4x2, per_1x32, 1, 0)  # ON arm (fold)
  ('gfx950', 256, M, 5120, 640, 384, 6, Silu, bf16, fp8e4m3, fp4x2, per_1x32, 1, 0)  # OFF arm (stock)
  ('gfx950', 256, M, 5120, 640, 128, 3, Silu, bf16, fp8e4m3, fp4x2, per_1x32, 1, 0)  # draft MoE (both arms)

Best row (lowest total `us`) per index key — aiter's merge would otherwise
raise on duplicate shape entries. Injected identically into both arms via
AITER_CONFIG_FMOE, so each arm runs its best tuned configs (fair A/B).
"""
import csv
from collections import defaultdict

T = "/data/amatveev/rocm10/tuned"
FILES = [f"{T}/hetero_tuned_fused_v1.csv", f"{T}/hetero_tuned_raw.csv"]
WANT = {("385", "7"), ("384", "6"), ("128", "3")}
OUT = f"{T}/combined_a8w4_5120x640_v1.csv"

def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]

hdr = None
best = {}
for path in FILES:
    h, rows = load(path)
    if hdr is None:
        hdr = h
    assert h == hdr, f"header mismatch in {path}"
    H = {n: i for i, n in enumerate(hdr)}
    is_fused_v1 = path.endswith("hetero_tuned_fused_v1.csv")
    kept = 0
    for row in rows:
        if not (row[H["gfx"]] == "gfx950"
                and row[H["cu_num"]] == "256"
                and row[H["model_dim"]] == "5120"
                and row[H["inter_dim"]] == "640"
                and row[H["q_dtype_a"]] == "torch.float8_e4m3fn"
                and row[H["q_dtype_w"]] == "torch.float4_e2m1fn_x2"
                and row[H["q_type"]] == "QuantType.per_1x32"
                and row[H["act_type"]] == "ActivationType.Silu"
                and (row[H["expert"]], row[H["topk"]]) in WANT):
            continue
        # The 385/7 fold shape goes through AITER's heterogeneous wrapper
        # (_use_fhmoe_wrappers), which only accepts the two-stage FlyDSL
        # family: kn1 'flydsl_moe1_*' and kn2 'flydsl_moe2_*' (NOT the
        # flydslv2 'flydsl_moe2_layout_*' family — the TUNE_ONLY default,
        # fhmoe-incompatible) with run_1stage=0. hetero_tuned_raw.csv's
        # 385/7 rows are layout-family (the first injection attempt armed
        # the fold OFF via the init probe), so the fused shape takes rows
        # ONLY from hetero_tuned_fused_v1.csv (tuned through the fhmoe
        # path). The 384/6 stock and 128/3 draft shapes run the plain
        # fused_moe path, which accepts any family — raw's layout rows are
        # fine there (and are the tuned best).
        if (row[H["expert"]], row[H["topk"]]) == ("385", "7"):
            if not is_fused_v1:
                continue
            kn1, kn2 = row[H["kernelName1"]], row[H["kernelName2"]]
            assert kn1.startswith("flydsl_moe1_"), f"bad kn1 {kn1}"
            assert kn2.startswith("flydsl_moe2_") and not kn2.startswith(
                "flydsl_moe2_layout_"
            ), f"bad kn2 {kn2}"
            assert row[H["run_1stage"]] == "0", "run_1stage must be 0"
        k = tuple(row[H[c]] for c in ("gfx", "cu_num", "token", "model_dim",
                                      "inter_dim", "expert", "topk", "act_type",
                                      "dtype", "q_dtype_a", "q_dtype_w", "q_type",
                                      "use_g1u1", "doweight_stage1"))
        us = float(row[H["us"]] or 1e9)
        if k not in best or us < float(best[k][H["us"]] or 1e9):
            best[k] = row
        kept += 1
    print(f"{path}: {kept} candidate rows")

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for k in sorted(best, key=lambda k: (int(k[5]), int(k[6]), int(k[2]))):
        w.writerow(best[k])
print(f"wrote {OUT}: {len(best)} rows")

cov = defaultdict(list)
for k in best:
    cov[(k[5], k[6])].append(int(k[2]))
for shape in sorted(cov):
    ms = sorted(cov[shape])
    print(f"  shape E/topk={shape[0]}/{shape[1]}: {len(ms)} Ms: {ms}")
    # the two measured prefill Ms must be tuned for the A/B shapes; the aux
    # draft-MoE shape (128/3) only has a partial grid — its uncovered Ms stay
    # heuristic in BOTH arms (identical), which is fine.
    if shape in {("384", "6"), ("385", "7")}:
        for need in (8192, 16384):
            assert need in ms, f"MISSING M={need} for shape {shape}"
assert len(best) == len(set(best)), "duplicate keys"
print("coverage OK (8192+16384 present for 384/6 and 385/7)")
```

</details>

