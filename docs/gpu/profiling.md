# Profiling

Profiling stands up inference servers, drives load, and captures GPU traces that
later stages analyze. Two complementary packages exist:

| Package | Role |
|---------|------|
| `llm/auto_profile` | Shell/Docker orchestration (`run_profile_core.sh`), result parsing into analysis configs |
| `llm/auto_profile_server_client` | Python client with Docker and Kubernetes Job backends, metrics, stability analysis |

Model inventory → run config generation:

```bash
uv run python -m llm.auto_config --help
```

## Example campaign

Checked-in example: DeepSeek-R1-NVFP4 on B200, vLLM vs SGLang:

`llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json`

Shared infra JSON lives beside it under `llm/auto_profile/test_configs/`
(`infra_config.json`, `gpu_configs.json`, `docker_images.json`, etc.).

## End-to-end: profile → parse → analyze

```bash
# All three steps
./run_all.sh ./llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json

# Or skip stages you already completed
./run_all.sh ./llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json \
  --skip-profile --skip-pdf --skip-jiras
```

Individual steps:

```bash
./run_step1_profile.sh <run_config.json>
./run_step2_parse.sh <run_config.json>
./run_step3_analyze.sh <run_config.json>   # optional --skip-pdf / --skip-trace / --skip-jiras
```

Core profiler (when not using the wrappers):

```bash
./llm/auto_profile/run_profile_core.sh <run_config.json>
```

## Capturing traces without the full campaign

For vLLM PyTorch profiling traces, see
[vLLM Trace Generation](../../llm/auto_analyze/examples/vllm_generate_trace_example.md).

Supported analysis inputs later include `.nsys-rep`, `.sqlite`, `.json`, and
`.json.gz`. Converting `.nsys-rep` requires the `nsys` CLI.

## After profiling

Parsing emits configs and paths that [trace analysis](analysis.md) consumes.
For Python server/client orchestration details, start at
`llm/auto_profile_server_client/` (`pipeline.py`, `run_config.py`, `frameworks/`).

## Requirements reminder

Profiling needs GPU hardware (or cluster jobs), Docker/K8s as configured, and
framework images/sources. It does not replace Claude for the analysis step —
analysis still needs Claude credentials when you run auto_analyze.
