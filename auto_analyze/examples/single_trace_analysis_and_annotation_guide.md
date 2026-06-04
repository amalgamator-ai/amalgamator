# Single-Trace Analysis and High-Level Trace Annotation

## Motivation

When profiling LLM inference frameworks (vLLM, SGLang, TensorRT-LLM), a raw GPU
trace in Perfetto shows hundreds of low-level CUDA kernels with cryptic names
like `cutlass_sm100_gemm_f16_64x128x32` or `triton_red_fused_native_layer_norm_3`. Understanding which transformer operation
each kernel belongs to — attention projection, MoE routing, allreduce — requires
manually tracing the source code call chain for every kernel, which is extremely
time-consuming.

The single-trace analysis pipeline automates this entirely. It:

1. **Analyzes the framework source code** to identify the high-level operations
   in each transformer block type (attention projections, MoE routing, allreduce, etc.)
2. **Extracts all GPU operations** from the trace with timestamps, streams, and
   launch parameters
3. **Correlates every low-level kernel** to its high-level transformer block
   operation through source code deep-dive analysis
4. **Produces an annotated trace** (`single_trace_transformer_block.json`) that
   you can open in [Perfetto](https://ui.perfetto.dev) where every kernel is
   labeled with its high-level operation name, source code references, and call chain

This gives you an instant, unified view of high-level to low-level operation
mapping across all CUDA streams — no manual source code tracing needed.

Below is an example annotated trace from Kimi-K2.5-NVFP4 running on vLLM with
8xB200 GPUs (BS=1 decode). Each kernel is labeled with its high-level operation
(e.g., `Attn: flashinfer_mla_decode_attn`, `MoE: moe_expert_gemm1_gate_up_silu`),
and clicking any operation reveals its source code references and call chain.
In this specific example, we can see the selected `moe_gate_router_gemm` operation
with its `high_level_op`, `source_code` references, `call_chain`, and `kernel_name`
details in the bottom panel:

<p align="center">
  <img src="cross_commit_cmp_example_kimi/kimi_annotated_trace.png" alt="Annotated trace in Perfetto" width="600">
</p>

An example of single trace analysis results is here:
[vllm_kimi_v0.16.0/analyze](https://github.com/neuralmagic/ai_auto_analyze/tree/main/auto_analyze/examples/cross_commit_cmp_example_kimi/vllm_kimi_v0.16.0/analyze)

Here is a snippet from the [v0.16.0 median transformer block](https://github.com/neuralmagic/ai_auto_analyze/blob/main/auto_analyze/examples/cross_commit_cmp_example_kimi/vllm_kimi_v0.16.0/analyze/median_block.txt) showing the high-level to
low-level correlation (Kimi-K2.5 MoE decoder layer, 29 operations across 2 CUDA streams):

<sub>

| # | High-Level Operation | Kernel Name | Source Code Ref | Stream | Dur (ns) |
|--:|----------------------|-------------|-----------------|-------:|---:|
| 3 | Attn: qkv_a_proj_gemm | `nvjet_tst_16x64_64x16_4x1_v_bz_TNN` | mla.py:132, deepseek_v2.py:759 | 23 | 18,112 |
| 4 | Attn: q_a_layernorm_elementwise | `triton_poi_fused_1` | mla.py:137, layernorm.py:93 | 23 | 1,600 |
| 5 | Attn: q_a_layernorm_reduce | `triton_red_fused_2` | mla.py:137, layernorm.py:93 | 23 | 2,080 |
| 6 | Attn: q_b_proj_gemm | `nvjet_tst_16x64_64x16_4x1_v_bz_TNN` | mla.py:138, deepseek_v2.py:778 | 23 | 5,280 |
| 7 | Attn: rope_kva_layernorm_fused | `triton_poi_fused_add_clone_copy_...` | mla.py:149-159 | 23 | 2,240 |
| 8 | Attn: kv_cache_store | `concat_and_cache_mla_kernel` | mla_attention.py:524-531 | 23 | 2,303 |
| 9 | Attn: q_absorption_bmm | `nvjet_tst_64x8_64x16_4x1_v_bz_NNT` | mla_attention.py:601-615 | 23 | 2,815 |
| 10 | Attn: q_latent_pe_concat_fp8_prep | `triton_poi_fused__to_copy_cat_...` | mla_attention.py:620-624 | 23 | 1,792 |
| 11 | Attn: decode_mla_attention | `fmhaSm100fKernel_QkvE4m3O...` | mla_attention.py:635 | 23 | 9,184 |
| 12 | Attn: v_up_proj_bmm | `nvjet_tst_8x64_64x16_4x1_v_bz_TNN` | mla_attention.py:647 | 23 | 3,552 |
| 13 | Attn: o_proj_gemm | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` | mla.py:176, deepseek_v2.py:801 | 23 | 5,248 |
| 14 | Comm: fused_allreduce_post_attn | `allreduce_fusion_kernel` | allreduce_rms_fusion.py:85-150 | 23 | 10,016 |
| 15 | Comm: allreduce_completion_memcpy | `memcpy32_post` | allreduce_rms_fusion.py:85-150 | 23 | 1,056 |
| 16 | MoE: moe_gate_router_gemm | `nvjet_tst_8x64_64x16_4x1_v_bz_TNN` | deepseek_v2.py:251-257 | 23 | 10,880 |
| 17 | ShExp: gate_up_act_fp4_quant | `cvt_fp16_to_fp4` | nvfp4_utils.py:214 | 5518 | 2,721 |
| | *...29 ops total* | | | | *123,232* |

</sub>

Each operation also includes a detailed **source code explanation** column (not
shown above due to space constraints) describing what the kernel computes and how
it fits in the transformer block. For example:

> **qkv_a_proj_gemm** — MergedColumnParallelLinear (replicated, disable_tp=True)
> computing fused Q/KV low-rank projection. BF16 GEMM: [B,7168] x [7168,2112] ->
> [B,2112]. Output split into q_c=[B,1536] and kv_lora=[B,576]. Uses CUTLASS
> nvjet kernel.

> **rope_kva_layernorm_fused** — Torch.compile fused kernel combining: KV split
> (kv_c=[B,512], k_pe=[B,64]), RMSNorm(512) on kv_c, Q reshape, k_pe unsqueeze,
> and YaRN-scaled RoPE application. All from mla.py:149-159.

The single-trace analysis is also the foundation for **cross-trace comparison**,
where two single-trace results are compared to identify kernel-level performance
differences. See:
- [Cross-Commit Performance Comparison](cross_commit_comparison_guide.md) — compare different versions of the same framework
- [Cross-Framework Performance Comparison](cross_framework_comparison_guide.md) — compare different frameworks running the same model

## End-to-End Walkthrough

This walkthrough uses vLLM with the Kimi-K2.5 model on B200 GPUs as a concrete
example. The same process applies to any framework and model.

### Prerequisites

- A framework source code directory (clean git repo, no uncommitted changes)
- Access to the target model on HuggingFace

### Step 1: Generate a PyTorch Trace

First, capture a profiling trace of the workload you want to analyze. See
[vllm_generate_trace_example.md](vllm_generate_trace_example.md) for detailed instructions
on server/client trace generation.

The key outputs from this step are:
- A **trace file** (`.json.gz` PyTorch Chrome trace, or `.nsys-rep` / `.sqlite` NSYS trace)
- A **run log** containing the full execution output, including the run command at the top

### Step 2: Set Up Parameters

Define the parameters that describe your run. These must match the exact
configuration used when generating the trace.

```bash
# --- Model and hardware ---
MODEL="nvidia/Kimi-K2.5-NVFP4"
GPU_TYPE="B200"

# --- Execution parameters (must match the trace) ---
BATCH_RANGE="1"          # Batch size used
PREFILL_RANGE="4"        # Input / prefill sequence length
OUTPUT_RANGE="1024"      # Output / decode sequence length

# --- Input files ---
TRACE_FILE="/path/to/trace/rank0.pt.trace.json.gz"
RUN_LOG="/path/to/run_log.txt"

# --- Framework source code (must be a clean git repo) ---
SOURCE_CODE="/path/to/vllm"
COMMIT_ID="8f89381fc6b2d54591a7a560e20ee5211ce1ac33"  # or "HEAD"

# --- Output paths ---
ANALYZE_OUTPUT_DIR="/path/to/analysis_output"
OUTPUT_CONFIG_FILE="/path/to/single_trace_config_kimi_vllm"
```

**Notes:**
- `TRACE_FILE`: For PyTorch traces with tensor parallelism, each GPU produces a
  separate trace file. Pick any single rank (e.g., rank 0) — they all execute the
  same operations.
- `RUN_LOG`: Should be the full execution log including the run command and its
  parameters at the top. The framework name and run command are auto-extracted from it.
- `SOURCE_CODE`: Must be a clean git repo with no modified or uncommitted files.
  The analysis automatically creates a separate branch for the specified commit ID.
- `COMMIT_ID`: The exact git commit of the framework used during the trace capture.
  Use `"HEAD"` if the source code is already at the correct commit.

### Step 3: Create the Analysis Config

Use the helper script to generate the config JSON:

```bash
python -m auto_analyze.create_single_trace_config \
    --model $MODEL \
    --gpu-type $GPU_TYPE \
    --batch-size-range $BATCH_RANGE \
    --prefill-size-range $PREFILL_RANGE \
    --output-size-range $OUTPUT_RANGE \
    --trace-file $TRACE_FILE \
    --run-log-file $RUN_LOG \
    --clean-source-code-path $SOURCE_CODE \
    --commit-id $COMMIT_ID \
    --analyze-output-dir $ANALYZE_OUTPUT_DIR \
    --output-config-file $OUTPUT_CONFIG_FILE
```

This will:
- Infer the framework name from the run log (e.g., `vllm`)
- Extract the run command from the run log
- Verify the commit ID exists in the source code repo
- Write the config JSON to `$OUTPUT_CONFIG_FILE.json`

**Optional parameters** for more targeted analysis:

```bash
    --high-level-focus "Focus on pure decode execution and low-latency batch sizes" \
    --perf-analysis-focus "Pay special attention to torch.compile fusion opportunities" \
    --trace-gpu-focus "0" \
    --max-gpu-ops 1000
```

### Step 4: Run the Analysis

```bash
python -m auto_analyze.run_single_trace --config $OUTPUT_CONFIG_FILE.json
```

The analysis pipeline executes these steps automatically:

1. **High-level operations** — Reads the framework source code and identifies the
   sequence of logical operations in each transformer block type (attention
   projections, MoE routing, allreduce, etc.)

2. **GPU operations extraction** — Parses the trace file and extracts all GPU
   kernel events with timestamps, streams, and launch parameters.

3. **Operation correlation** — Correlates every low-level GPU kernel to its
   high-level transformer block operation through source code deep-dive analysis.
   Selects the median transformer block as the representative block.

4. **Annotated trace generation** — Produces a Chrome trace JSON for the median
   transformer block with every kernel labeled with its high-level operation,
   source code references, and call chain.

### Step 5: View in Perfetto

Open the annotated trace in Perfetto:

1. Go to [https://ui.perfetto.dev](https://ui.perfetto.dev)
2. Open `$ANALYZE_OUTPUT_DIR/single_trace_transformer_block.json`
3. Click any kernel to see its high-level operation, source code location, and call chain

A human-readable summary is also available at
`$ANALYZE_OUTPUT_DIR/single_trace_transformer_block.txt`.

## Output Files

After the analysis completes, the output directory contains:

| File | Description |
|------|-------------|
| `single_trace_transformer_block.json` | Annotated Chrome trace JSON — open in Perfetto |
| `single_trace_transformer_block.txt` | Human-readable summary of the annotated trace |
| `transformer_block_high_level_ops.txt` | High-level operation sequence from source code analysis |
| `gpu_ops.txt` | Raw GPU operations extracted from the trace |
| `gpu_ops_to_blocks.txt` | Full correlation of GPU ops to transformer blocks |
| `median_block.txt` | The selected median transformer block |
| `run_params.txt` | Run parameters for downstream tools |
| `run_originals/` | Copies of the original trace file, run command, and run log |

An example of these output files is available at:
[vllm_kimi_v0.16.0/analyze](https://github.com/neuralmagic/ai_auto_analyze/tree/main/auto_analyze/examples/cross_commit_cmp_example_kimi/vllm_kimi_v0.16.0/analyze)

## Optional: Enable Performance Analysis

By default, the config generator disables performance analysis to keep the run
focused on trace annotation. To also generate a detailed performance analysis
with improvement proposals, add:

```bash
python -m auto_analyze.create_single_trace_config \
    ... \
    --enable-single-trace-perf-analysis
```

This adds a `perf_analysis_single_trace.txt` output with bottleneck identification,
improvement proposals with code snippets, and impact estimates.

## Next Step: Cross-Trace Comparison

Once you have single-trace analysis results for two traces, you can compare them
to identify kernel-level performance differences:

- **Same framework, different commits** — See [Cross-Commit Performance Comparison](cross_commit_comparison_guide.md)
- **Different frameworks, same model** — See [Cross-Framework Performance Comparison](cross_framework_comparison_guide.md)
