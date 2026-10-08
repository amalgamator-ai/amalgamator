# Generating PyTorch Trace Files in vLLM

This guide demonstrates how to generate PyTorch profiling trace files for large language models using vLLM, illustrated with the `deepseek-ai/DeepSeek-R1-0528` model.

## Overview

PyTorch profiling helps analyze model performance and identify bottlenecks during inference. This workflow involves:
1. Starting a vLLM server with profiling enabled
2. Running a benchmark client to generate load
3. Capturing profile traces during steady-state operation

## Prerequisites

- Multi-GPU setup (example uses 8 GPUs with tensor parallelism)
- vLLM installed and configured
- Access to the target model from HuggingFace

## Step 1: Start the vLLM Server

Launch the vLLM server with profiling configuration. Note that prefix caching is disabled to capture worst-case execution patterns.

```bash
export MODEL=deepseek-ai/DeepSeek-R1-0528  # Your HuggingFace model ID
export VLLM_TORCH_PROFILER_DIR=$HOME/profiles  # Directory for profile output
export PORT=8123  # Server port (adjust if needed)

vllm serve $MODEL \
  --port $PORT \
  --no-enable-prefix-caching \
  --max-num-seqs 128 \
  --max-model-len 16384 \
  --tensor-parallel-size 8
```

**Configuration notes:**
- `--no-enable-prefix-caching`: Disables prefix caching to represent worst-case scenario
- `--tensor-parallel-size 8`: Distributes model across 8 GPUs
- `--max-num-seqs 128`: Maximum concurrent sequences
- `--max-model-len 16384`: Maximum sequence length

## Step 2: Run the Benchmark Client

In a separate terminal, execute the benchmark client to generate inference load. This example demonstrates pure-decode workload with batch size 16.

```bash
export MODEL=deepseek-ai/DeepSeek-R1-0528
export VLLM_TORCH_PROFILER_DIR=$HOME/profiles
export PORT=8123

export CONCURRENCY=16    # Batch size (concurrent requests)
export INPUT_LEN=4       # Prompt length in tokens
export OUTPUT_LEN=1024   # Number of output tokens per request
export NUM_PROMPTS=64    # Total number of queries to process

vllm bench serve \
    --port $PORT \
    --model $MODEL \
    --dataset-name random \
    --max-concurrency $CONCURRENCY \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS \
    --seed $(date +%s) \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 90,95,99 \
    --ignore-eos \
    --trust-remote-code
```

**Important flags:**
- `--ignore-eos`: Ensures all requests process the full output length, critical for consistent profiling
- `--max-concurrency`: Controls the batch size for profiling
- `--random-input-len` / `--random-output-len`: Specify prompt and output token counts

## Step 3: Capture the Profile

Wait for the warm-up phase to complete and the system to reach steady state at the target batch size. Monitor the server logs to confirm the current batch size matches your configuration.

### Start Profiling

Once the system is saturated:

```bash
export PORT=8123
export BASE_URL="http://0.0.0.0:$PORT"

curl -X POST ${BASE_URL}/start_profile
```

### Stop Profiling

Wait approximately 5 seconds to collect sufficient profiling data, then stop:

```bash
export PORT=8123
export BASE_URL="http://0.0.0.0:$PORT"

curl -X POST ${BASE_URL}/stop_profile
```

## Output Files

Profile trace files will be saved to the directory specified by `VLLM_TORCH_PROFILER_DIR`.

**Note:** In tensor-parallel configurations, vLLM generates one trace file per GPU. Since all GPUs execute identical operations in this setup, analyzing a single GPU's trace file is sufficient for most performance analysis tasks.

## Workload Customization

Adjust these parameters based on your profiling needs:

| Parameter | Purpose | Example Values |
|-----------|---------|----------------|
| `CONCURRENCY` | Control batch size | 1, 8, 16, 32, 64 |
| `INPUT_LEN` | Prompt length | 4, 128, 512, 2048 |
| `OUTPUT_LEN` | Decode length | 256, 512, 1024, 2048 |
| `NUM_PROMPTS` | Total queries | 32, 64, 128 |