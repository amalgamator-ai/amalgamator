# AMD Profiling Support — Engineering Proposal

**llm use-case · October 9, 2026 · target: vLLM vs SGLang comparison on dell-mi355x-3 with `deepseek-ai/DeepSeek-V4.1-Flash`**

Status: host verified · images verified (registry API) · torch profiler verified on this ROCm stack · recommendation: port `auto_profile`, ~100 lines

---

## §0 Recommendation in short

- **Docker is the right vehicle, and it is fully supported.** Both frameworks publish official ROCm images that match this machine almost exactly: `vllm/vllm-openai-rocm:v0.31.0` is built on `rocm/dev-ubuntu-22.04:7.2.3-complete` — the same ROCm version installed on this host — with gfx950 (MI355X) in its kernel build list, and vLLM even ships a model-specific `deepseekv41-flash-0909` tag plus a dedicated `vllm/models/deepseek_v41/amd/` implementation (ROCM AITER MLA attention). SGLang's ROCm images live in the daily-built `lmsysorg/sglang-rocm` repo — `v0.5.21-rocm724-mi35x-20261009` is the MI350-series build its DeepSeek-V4.1 cookbook points to. Note: V4.1-Flash hasn't shipped in an sglang release yet; the daily preview images carry it, so pin the full date-suffixed tag.
- **Trace generation: replace nsys with the in-framework PyTorch profiler.** Verified on this host: torch on ROCm emits Chrome-trace JSON with GPU kernel events, and `auto_analyze` already auto-detects and parses exactly that format (the `PYTORCH` trace type). The analysis side needs no changes.
- **Port target: `auto_profile_server_client`** (the python client–server pipeline) — **this port is now implemented** (2026-10-09). It is the primary AMD path: vendor detection + ROCm docker flags + `rocm-smi` memory polling in `docker_runner.py`, the JSON-string `--profiler-config` fix in the vLLM adapter, a fix for the outer-docker flow (the inner run config was never written into the container), and the campaign config `llm/auto_profile_server_client/test_configs/deepseek_v41_flash_mi355x.json`. ~100 net lines, all vendor-gated — NVIDIA hosts take the byte-identical old docker args. The shell `auto_profile` port (behind `/profile`) drops to an optional follow-up; its delta is ~100 lines across 7 files if the in-process benchmark view is ever needed.
- **The model is large but fits well.** DeepSeek-V4.1-Flash is a 763 GB FP8 checkpoint (48 safetensors shards, ungated). It must live on `/data` — the HF hub cache moves to `/data/amatveev/hf` (6.6 TB free; root fs has only 141 GB). TP=8 puts ~95 GB of weights on each 288 GB GPU, leaving ~190 GB per GPU for KV cache.
- **First gate is a no-code-change bring-up (Phase 0)**: install docker, pull the images, smoke-test GPUs in-container, validate the real model end-to-end in both frameworks. The `deepseek_v41` architecture is new on ROCm, but the risk is well contained: vLLM has a purpose-built image tag and a dedicated AMD implementation of the model, and SGLang's cookbook carries a verified MI355X launch recipe for exactly this model. The first real run is still the gate before any porting starts.

---

## §1 The machine and the goal

### Host facts (probed on this machine)

| What | Value |
|---|---|
| GPUs | 8× AMD Instinct MI355X — gfx950 / CDNA4, 288 GB HBM3E each (2.3 TB total) |
| Host ROCm | 7.2.3 — **exactly matches the vLLM ROCm image base** (`rocm/dev-ubuntu-22.04:7.2.3-complete`) |
| CPU / RAM | 256 cores / 3 TB |
| Disks | `/` 141 GB free · `/data` 6.6 TB free (nvme) |
| Docker | Not installed yet (dnf available; user in wheel / video / render groups) |
| ROCm profiler / telemetry tools | rocprof, rocprofv2, rocprofv3, rocprof-sys, rocprof-compute in `/opt/rocm/bin`; `amd-smi`, `rocm-smi`, `rocminfo` on PATH |
| HF cache | Moving to `/data/amatveev/hf` via `HF_HUB_CACHE` (default `~/.cache/huggingface/hub` on `/` holds only Qwen3-0.6B) |
| Local checkouts | `~/code/vllm` (main; vllm 0.31.1rc1.dev204 on a working ROCm torch 2.13 build) · `~/code/sglang` (main; sglang not installed) |

### Model facts (HF API, checked 2026-10-09)

| deepseek-ai/DeepSeek-V4.1-Flash | Value |
|---|---|
| Checkpoint | 763.2 GB of safetensors across 48 shards |
| Format | FP8 (blockwise) · architecture `DeepseekV41ForCausalLM` · multimodal (image-text-to-text) · MIT |
| Access | Ungated — no `HF_TOKEN` needed |
| Version floor | vLLM ≥ 0.30.0 (first release registering `deepseek_v41`) · SGLang ≥ 0.5.21 |
| Placement | TP=8 → ~95 GB weights/GPU, ~190 GB KV headroom per GPU (TP=4 also fits at ~191 GB/GPU but is tight at long context) |

### The goal, in amalgamator terms

Reproduce the shape of the existing `deepseek_r1_nvfp4` campaign (the B200 example chain) on MI355X: `pure_decode` profiles (isl 4 / osl 1024, concurrency 1) plus the kv-cache-fill decode variant for both frameworks, with per-iteration traces, then feed `/parse → /analyze → cross-trace → codegen` unchanged. The deliverable is the same one the NVIDIA campaign produces: TPOT comparison tables plus per-framework kernel timelines that `auto_analyze` can diff.

---

## §2 How the current pipelines work

### auto_profile — shell pipeline, in-process benchmarking

```
run config JSON ──► parse_run_config.py ──► run_profile_core.sh (framework × profile × concurrency)
                                            ├─► vllm_run_docker.sh ─► vllm_bench.sh (vllm bench throughput)
                                            └─► sgl_run_docker.sh  ─► sgl_bench.sh (one_batch, in-process)
                                                     └─► results/<fw>/<model>/test-*/ (bench json · logs · traces)
                                                             └─► /parse (test_results/ + TPOT tables) ─► /analyze ─► /codegen
```

- **Config layer.** A campaign directory under `llm/auto_profile/test_configs/` holds `infra_config.json` (pointing at `docker_images.json`, `gpu_configs.json`, `exec_modes.json`) and a run config that selects model, GPU set, modes, frameworks, concurrencies, and `enable_trace`. Per-mode env/flag files live at `modes/<name>/{vllm,sgl}`. `parse_run_config.py` validates it and emits bash associative arrays (`PROFILE_GPU_IDS`, `INPUT_LENS`, `VLLM_MODES`, `SGL_MODES`, …) plus the docker image variables.
- **Execution layer.** `run_profile_core.sh` loops over framework × profile × concurrency and calls `vllm_run_docker.sh` / `sgl_run_docker.sh`, which start one container per run via `utils.sh::_run_docker` — today with `--runtime nvidia --gpus=all`, HF-cache and repo mounts, port mapping — and execute the bench script inside.
- **vLLM bench.** `vllm_bench.sh` runs `vllm bench throughput --async-engine` with a random dataset (isl/osl from the mode), `--max-num-seqs` = concurrency, JSON results into `results/<fw>/<model>/test-<name>/bench-*.json`. The profiling variant wraps the identical command in `nsys profile -t cuda,nvtx -c cudaProfilerApi …` plus `--profile --profiler-config '{"profiler":"cuda","delay_iterations":N,"max_iterations":50}'`, where N (`calc_start_iter`: warmups + half the waves + half the output length) lands the capture window at the decode midpoint. Afterwards `nsys export --type=sqlite` produces the trace the analysis step consumes.
- **SGLang bench.** `sgl_bench.sh` runs `python -m sglang.bench_one_batch` — in-process, batch-size = concurrency — with `--result-filename` JSON output. The profiling variant adds `--profile --profile-activities CUDA_PROFILER --profile-stage=decode --profile-start-step N --profile-steps 50` under nsys: the `CUDA_PROFILER` activity makes the bench call `cudaProfilerStart/Stop` so nsys's `-c cudaProfilerApi` captures exactly that window.
- **Metadata.** `write_run_metadata` records GPU name/memory via `nvidia-smi` into each result dir.
- **Downstream.** `/parse` (`run_profile_summary.py` driving a Claude agent) restructures results into `test_results/<test-id>/` and builds TPOT comparison tables; `/analyze` consumes traces — nsys `.sqlite` (*NSYS*) or torch `.json` (*PYTORCH*, auto-detected) — for single-trace and cross-trace analysis, Chrome-trace view, PDF, and JIRAs; `/codegen` ports kernels between frameworks from there.

### auto_profile_server_client — python pipeline, client–server benchmarking

- **Config.** `run_config.py` defines per-framework entries: docker image, server/client args, `enable_trace`, and `profiler_type` — `"nsys"` (default) or `"torch"`, **both already implemented**.
- **Orchestration.** `pipeline.py` runs either through `DockerRunner` (outer python starts a container that re-invokes the pipeline inside) or directly when already inside a container. `docker_runner.py` builds the `docker run` args (NVIDIA runtime today), auto-mounts host nsys when present, and waits for GPU memory release via `nvidia-smi` between runs. `server.py` starts `vllm serve` / `sglang.launch_server` with tee'd logs and health polling.
- **Benchmark + profiling.** `benchmark_runner.py` starts the server, warms it up, runs the client (`vllm bench serve` / `sglang.bench_serving`), then optionally profiles with one of two strategies:
  - *Iteration-based (vLLM):* server gets `--profiler-config` (cuda for nsys; torch with `torch_profiler_dir=/tmp/vllm_traces` for torch), the client is armed, and `delay_iterations`/`max_iterations` bound the window. `_finalize_torch_trace` copies `/tmp/*.trace.json(.gz)` out of the container; `_finalize_nsys_trace` SIGINTs nsys and exports sqlite.
  - *Metrics/time-based (SGLang):* a dry-run client measures TTFT/TPOT, the runner computes a fire delay, starts the real client, sleeps the delay, and POSTs `/start_profile {"num_steps": N, "activities": ["CUDA_PROFILER"]}` — or `["CPU","GPU"]` for the torch path.

### The key architectural fact

The stack already contains a complete, vendor-neutral torch-profiler path: `server_client` implements `profiler_type: "torch"` end-to-end, and `auto_analyze`'s prompts auto-detect the resulting Chrome-trace JSON (`PYTORCH` type) and extract GPU kernel events from it. The NVIDIA-only parts are confined to container runtime flags, `nvidia-smi` calls, and the nsys mount/export helpers. The AMD port is therefore mostly "swap container flags and use the torch path that already exists" — not new machinery.

---

## §3 What breaks on AMD, and the fix

A sweep of the `llm/` tree finds ~145 NVIDIA-specific touchpoints. Condensed to the ones that matter for this machine:

| File / function | NVIDIA assumption today | AMD replacement | Size |
|---|---|---|---|
| `auto_profile/utils.sh :: _run_docker` | `--runtime nvidia --gpus=all` | `--device /dev/kfd --device /dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined` (keep `--ipc=host`, shm-size, ulimits; GPU selection still via `CUDA_VISIBLE_DEVICES`) | small |
| `auto_profile/utils.sh :: write_run_metadata` | `nvidia-smi --query-gpu` | `rocm-smi` (`--showproductname --showmeminfo vram` or `--json`) | trivial |
| `utils.sh` nsys helpers, `NSYS_DEFAULT_FLAGS`, export step | nsys profile/export | gone on AMD — replaced by the in-framework torch profiler (§5) | core change |
| `auto_profile/vllm/vllm_bench.sh` profile branch | nsys wrap + `"profiler":"cuda"` | `"profiler":"torch"` + `torch_profiler_dir`; collect `.json(.gz)`; no export step (§6 has the diff) | small |
| `auto_profile/sgl/sgl_bench.sh` | module `sglang.bench_one_batch`; `--profile-activities CUDA_PROFILER` under nsys | module renamed to `sglang.benchmark.one_batch` in current sglang; drop the `CUDA_PROFILER` activity so the default `CPU,GPU` activities run the torch profiler; set `SGLANG_TORCH_PROFILER_DIR` | small |
| `vllm_run_docker.sh`, `sgl_run_docker.sh` | mount host nsys read-only into container | skip the mount on AMD (guard by vendor) | trivial |
| `auto_profile/config.sh` | `B200/H200` gpu types; HF cache wherever the env points | add `mi355x` type; `HF_HUB_CACHE` must point at `/data` | trivial |
| `run_profile_summary.py` (parse step) | expects `trace-*.nsys-rep` / `.sqlite` | accept `trace-*.json` / `.json.gz` as PYTORCH traces; gunzip if needed | small |
| `server_client/docker_runner.py` | NVIDIA runtime flags; `nvidia-smi` in `wait_for_gpu_memory_release` | same ROCm flags; `rocm-smi` | ~20 lines |
| `server_client/cluster_runner.py` | NVIDIA-B200 node selectors | out of scope — no K8s on this host | deferred |
| trt framework | NVIDIA-only | simply not enabled in the AMD campaign config | n/a |

Everything else in the chain — parse restructuring, analysis prompts, codegen, PR flow — is trace-format-driven and already vendor-neutral.

---

## §4 Benchmarking choices on AMD

### vLLM

| Option | Status (verified) | Verdict |
|---|---|---|
| `vllm/vllm-openai-rocm` — tags `v0.31.0`, `deepseekv41-flash-0909`, `nightly` | Official image. Built from `rocm/dev-ubuntu-22.04:7.2.3-complete` — the exact host ROCm — with `PYTORCH_ROCM_ARCH` including gfx950; ships ROCm triton, Aiter, amdsmi. v0.31.0 has `deepseek_v41` (first appeared in v0.30.0). | **use this** — pin `v0.31.0`; the `deepseekv41-flash-0909` tag is a model-specific known-good pairing to try if the release tag has issues |
| `rocm/vllm`, `rocm/vllm-dev` (AMD's images) | Deprecated since 2026-01-20 in favor of the official vLLM images; newest tags top out at vllm 0.29.0 (even on ROCm 10.1) | **reject** — too old for `deepseek_v41` (needs ≥ 0.30.0) |
| `rocm/vllm-ci` | Daily builds, commit-hash tags only | Debugging regressions; bad for a reproducible campaign |
| Native build (`~/code/vllm` already builds: vllm 0.31.1rc1.dev204 on ROCm torch 2.13) | Proven on this host | Right vehicle later for `auto_code_gen` porting work (needs an editable repo); heavier for benchmarking and drifts from the pinned comparison |
| Build own image from `docker/Dockerfile.rocm_base` | Full control | Only if custom kernels are needed; slow first build |

The benchmark tool itself — `vllm bench throughput --async-engine` (in-process, what `auto_profile` uses) and `vllm bench serve` (client–server, what `server_client` uses) — is pure python over the engine and runs on ROCm unchanged. AMD's "vLLM V1 on Instinct" optimization guide recommends `VLLM_ROCM_USE_AITER=1` (master switch for the AITER kernel family, with sub-flags for MLA/MoE/linear/RMSNorm/FP8 paths) and `--attention-backend ROCM_AITER_MLA` for DeepSeek-class MLA models (1.2–1.5× over TRITON_MLA per AMD) — that env goes into the campaign's vLLM mode file.

### SGLang

| Option | Status (verified) | Verdict |
|---|---|---|
| `lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261009` | Official daily-built ROCm repo (26 GB). ROCm 7.2.4 (host is 7.2.3), built for MI350-series from `docker/rocm.Dockerfile` (default `GPU_ARCH=gfx950`, ships AITER). SGLang's DeepSeek-V4.1 cookbook: V4.1-Flash hasn't shipped in a release yet — use these preview images, ROCm 7.2 recommended. | **use this** — pin the full date-suffixed tag for reproducibility |
| `lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-<date>` | ROCm 7.2.0 variant — the exact stack the cookbook recipe was validated on | Alternate pin |
| Other `sglang-rocm` variants (`rocm10`, `mi30x`, `mi45x`) / `rocm/sgl-dev` mirror | Other ROCm stacks and GPU classes; AMD-mirrored dev images | Fallbacks / debugging |
| Native source build (`setup_rocm.py` + `pip install -e "python[all_hip]"`) | Documented; checkout exists at `~/code/sglang` | Later, for codegen iterations |

One compat detail that matters: in current sglang the in-process benchmark module is `sglang.benchmark.one_batch` — the `sglang.bench_one_batch` path used by `auto_profile/sgl/sgl_bench.sh` no longer exists at v0.5.21. The port must use the new module path (it fails fast, not silently, so this is easy to catch).

The SGLang cookbook also carries a **verified MI355X launch recipe for DeepSeek-V4.1-Flash** (`docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1`): env `SGLANG_USE_AITER=1`, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, `TORCH_BLAS_PREFER_HIPBLASLT=1`, `GPU_MAX_HW_QUEUES=2`, `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`, and flags including `--fp8-gemm-backend aiter` — the cookbook notes that overriding the aiter GEMM backend falls back to Triton and "costs most of the model's bs=1 throughput", which matters directly for a bs=1 decode campaign. These go into the campaign's SGLang mode file (§6); the recipe's DSpark speculative decoding is left off for the base comparison so both frameworks run plain decode.

### So — can we use docker? Yes. Exact commands:

```bash
# pull + GPU smoke test (flags are the union of vLLM's and SGLang's official ROCm docs)
docker pull vllm/vllm-openai-rocm:v0.31.0
docker pull lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261009   # date-suffixed; rebuilt daily — pin the tag

docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size 32g \
  -v /data/amatveev/hf:/app/hf_hub_cache -e HF_HUB_CACHE=/app/hf_hub_cache \
  vllm/vllm-openai-rocm:v0.31.0 rocm-smi
```

`--network=host --privileged` (SGLang's doc alias) are only required for RDMA, which a single-node comparison doesn't use. GPU selection works with `CUDA_VISIBLE_DEVICES` on ROCm exactly as the pipelines already do it. The user is already in the `video` and `render` groups, so device access inside the container needs no extra setup.

---

## §5 How to generate profile traces on AMD

### Option A — the in-framework PyTorch profiler (recommended)

- **vLLM:** `ProfilerConfig` supports `profiler ∈ {torch, cuda, proton}`. Switching the existing `"cuda"` value to `"torch"` adds `torch_profiler_dir`; `delay_iterations`/`max_iterations` keep their exact semantics, so the decode-midpoint window math in `calc_start_iter` is unchanged. Activities default to CPU+CUDA, which on ROCm routes through kineto/roctracer. Gzip is on by default (`.json.gz` output). The JSON-string form of `--profiler-config` (as `auto_profile` already passes it) is the documented form, and the serving API also accepts `delay_iterations`/`max_iterations` in the `/start_profile` body. Output lands as `rank0.<id>.pt.trace.json.gz` per worker. One caveat: trace flushing takes minutes at scale (vLLM's docs clock ~10 min for ~100 requests of a 70B model on an H100) — let `/stop_profile` or process exit run to completion before collecting.
- **SGLang:** `one_batch --profile` with the default `--profile-activities CPU GPU` starts `torch.profiler`; `--profile-stage decode --profile-start-step N --profile-steps 50` bounds the window (defaults: start = output_len/2, steps = 1); the trace is written via `export_chrome_trace` into `SGLANG_TORCH_PROFILER_DIR` (default `/tmp`). The `CUDA_PROFILER` activity — what `auto_profile` passes today — is only the external-nsys hook and is simply dropped. On the serving path, POST `/start_profile` with `activities ["CPU","GPU"]` does the same (already implemented in `server_client`). The serving API's activity set is `["CPU","GPU","MEM","RPD"]` — `RPD` is a ROCm-native rocprofiler-based path that additionally writes a rocpd SQLite (interesting for Phase 3). Traces are named `{profile_id}-TP-{tp}-…-EP-{ep}.trace.json.gz` with an optional `merge_profiles` flag, and `SGLANG_PROFILE_WITH_STACK=False` works around a known PyTorch stack-capture replay bug.
- **Verified on this host:** a torch-profiler run on this exact ROCm stack (torch 2.13 ROCm build, hip 7.2.53211) produced a Chrome trace with GPU kernel events — category `kernel` (e.g. `__amd_rocclr_fillBufferAligned.kd`, 7.68 µs) alongside `cpu_op` and runtime events.
- **auto_analyze already consumes it:** `single_trace_prompts.py` auto-detects the trace type from the extension — `.sqlite` → NSYS, `.json` → PYTORCH — and instructs the agent to parse the Chrome trace and extract kernel events (short_name, start_ns, duration_ns, stream, category). Zero analysis-side changes.

### Option B — rocprofiler-SDK (`rocprofv3`; on this host's PATH)

The legacy `rocprof`/`rocprofv2` CLIs are deprecated (end-of-support anticipated by end of Q2 2026); the current tool is `rocprofv3`. It traces HIP/HSA/ROCTx/RCCL activity and kernel dispatches system-wide — including dynamic attach to a running server (`rocprofv3 -p <pid> --sys-trace`), which nsys never offered — and writes the rocpd SQLite format, convertible with `rocpd convert --output-format pftrace` to Perfetto-viewable traces (also csv/otf2/json). Valuable as a cross-check and for counter-level analysis later, but it has no equivalent of the engines' in-process arming (`delay_iterations`, `--profile-start-step`), so precisely windowing the decode midpoint is harder, and its output is a new trace format for `auto_analyze`. Defer to Phase 3.

### Option C — rocprof-compute (the Omniperf successor)

Hardware-counter Speed-of-Light / memory-chart analysis for deep optimization dives, with live attach (`rocprof-compute profile --attach-pid <pid>`). It replays kernels for counter collection and serializes dispatches while profiling, so it is a per-kernel "why is this slow" tool for offline use after timelines identify hot kernels — not for the comparison campaign itself.

**Why A wins:** it is a drop-in replacement for both frameworks' existing windowing machinery and for the analysis pipeline's existing format support. The only code that changes is the wrapper that arms the profiler and the wrapper that collects the output file.

---

## §6 The plan

### Phase 0 — bring-up, no code changes (~half a day, mostly model download)

1. Install docker (needs sudo; user is in wheel):

   ```bash
   sudo dnf install -y docker-ce          # after adding the docker-ce repo for RHEL 10
   sudo mkdir -p /data/docker
   # /etc/docker/daemon.json → {"data-root": "/data/docker"}
   # (root fs has only 141 GB; ROCm images are ~50 GB each)
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER          # re-login afterwards
   ```

2. Pull both images (§4) and smoke-test the GPUs in-container: `rocm-smi` shows all 8 devices, then a small torch matmul per device. Pin clocks for reproducible numbers: `sudo amd-smi set --perf-level STABLE_STD` (revert with `AUTO` afterwards).
3. Smoke benchmark through the raw docker path with the already-cached Qwen3-0.6B: `vllm bench throughput` in the vLLM image and `python -m sglang.benchmark.one_batch` in the SGLang image, each with `--profile` to produce a Chrome trace. This proves bench + trace end-to-end before touching the repo.
4. Point the HF hub cache at `/data/amatveev/hf` — `export HF_HUB_CACHE=/data/amatveev/hf` (persisted in `~/.bashrc`; both pipelines mount `${HF_HUB_CACHE}` into the container automatically) — and download the model: `hf download deepseek-ai/DeepSeek-V4.1-Flash` (763 GB, ungated; the CLI is `hf` from huggingface_hub 2.x, the older name is `huggingface-cli`).
5. **Validation gate:** short TP8 run + short profile of the real model in each framework — for vLLM with `VLLM_ROCM_USE_AITER=1` (AMD's recommended master switch; `--attention-backend ROCM_AITER_MLA` for DeepSeek-class MLA), for SGLang with the cookbook env from §4. The JSON-string form of `--profiler-config` is the documented one; confirm it on the image's `vllm bench throughput --help` (two minutes). If the Flash variant hits missing ROCm kernels: try the `deepseekv41-flash-0909` tag, then nightlies, before writing any code.

### Phase 1 — AMD port of auto_profile_server_client — **DONE** (2026-10-09)

Implemented in the working tree (uncommitted), with unit tests in `tests/llm/auto_profile_server_client/`:

- `docker_runner.py`: `_gpu_vendor()` (nvidia-smi → nvidia, rocm-smi → amd, else unknown) gates `_base_docker_args` — AMD gets `--device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined`, NVIDIA keeps the exact previous `--runtime nvidia --gpus=all`. `wait_for_gpu_memory_release` now reads `rocm-smi --showmeminfo vram --json` on AMD (it previously burned its whole 120 s timeout when nvidia-smi was absent) and skips immediately when no management CLI exists.
- `frameworks/vllm.py`: `profiling_server_extra_flags` now emits the documented single-JSON-string form `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm_traces","delay_iterations":N,"max_iterations":M,"ignore_frontend":true}'` (the old dotted form `--profiler-config.profiler …` is not a valid vLLM flag — argparse rejects it, so vLLM profiling was broken on both vendors; the `nsys` type keeps `profiler: cuda`). Verified against `vllm/config/profiler.py` in the local checkout: `torch_profiler_dir` is *required* when profiler is `torch` and *rejected* otherwise, and `/start_profile` only exists when `--profiler-config` is set — the adapter's arm-then-POST flow matches.
- `pipeline.py`: `_run_via_docker` previously told the inner container to read `/tmp/run_config_<fw>.json` but never wrote it (the host→docker mode could not run at all). It now serializes the run config into the mounted `base_dir` (`/app/.server_client_inner_<fw>.json`), invokes the inner pipeline with `python3`, and gives each framework its own subtree of the host-visible output dir.
- Campaign config: `llm/auto_profile_server_client/test_configs/deepseek_v41_flash_mi355x.json` — both frameworks, `pure_decode` (isl 4 / osl 1024), concurrency `[1]`, `enable_trace` + `profiler_type: "torch"`, the §4 recipes baked into `server_args.extra_env`/`extra_flags`, `num_gpus: 4` (TP=2 cannot hold the 763 GB checkpoint: 381.6 GB/GPU > 288 GB; TP=4 → 190.8 GB/GPU is the minimum).
- Run: `python3 -m llm.auto_profile_server_client.pipeline --config llm/auto_profile_server_client/test_configs/deepseek_v41_flash_mi355x.json` from the host (needs the Phase 0 docker/images/model prerequisites). Output lands in `llm/auto_profile_server_client/results/deepseek_v41_flash_mi355x/<fw>/{benchmark,profiling}/` with Chrome-trace JSON per framework.
- Not done (deliberately): no GPUVendor class hierarchy, no config-schema changes, no per-adapter vendor branching, `cluster_runner.py` untouched. `/parse` does not yet understand this output layout (the server_client → run_profile_summary integration remains future work, ~half a day).

### Phase 2 — optional: port the shell auto_profile (~100 lines, one day)

All changes gate behind a vendor switch so NVIDIA campaigns are untouched:

`utils.sh` — vendor detection + docker flags:

```bash
detect_gpu_vendor() {                      # nvidia | amd | unknown
  command -v nvidia-smi >/dev/null 2>&1 && echo nvidia && return
  command -v rocm-smi   >/dev/null 2>&1 && echo amd     && return
  echo unknown
}

if [ "${GPU_VENDOR}" = "amd" ]; then
  base_flags="--device /dev/kfd --device /dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
else
  base_flags="--runtime nvidia --gpus=all"
fi
# ... rest of _run_docker unchanged (ipc=host, shm-size, ulimits, mounts, CUDA_VISIBLE_DEVICES)
```

`vllm_bench.sh` — profile branch: cuda→torch, drop nsys:

```bash
# NVIDIA (today)
nsys profile ${NSYS_DEFAULT_FLAGS} -o ${trace_file_prefix} \
  ${bench_cmd} --profile \
  --profiler-config "{\"profiler\":\"cuda\",\"delay_iterations\":${start_iter},\"max_iterations\":${NUM_TRACE_ITERS}}"
nsys export --type=sqlite -o ${trace_file_prefix}.sqlite ${trace_file_prefix}.nsys-rep

# AMD (proposed) — window math (calc_start_iter) unchanged
${bench_cmd} --profile \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/tmp/vllm_torch_trace\",\"delay_iterations\":${start_iter},\"max_iterations\":${NUM_TRACE_ITERS}}"
cp /tmp/vllm_torch_trace/*.json* ${result_dir}/    # → trace-<id>.json[.gz]
```

`sgl_bench.sh` — module rename + torch profiler:

```bash
# NVIDIA (today)
nsys profile ${NSYS_DEFAULT_FLAGS} -o ${trace_file_prefix} \
  python -m sglang.bench_one_batch ... \
    --profile --profile-activities CUDA_PROFILER \
    --profile-stage=decode --profile-start-step ${start_step} --profile-steps ${NUM_TRACE_ITERS}

# AMD (proposed) — default CPU,GPU activities = torch profiler; no CUDA_PROFILER, no nsys
SGLANG_TORCH_PROFILER_DIR=/tmp/sgl_torch_trace \
python -m sglang.benchmark.one_batch ... \
  --profile --profile-stage=decode \
  --profile-start-step ${start_step} --profile-steps ${NUM_TRACE_ITERS}
cp /tmp/sgl_torch_trace/*.json ${result_dir}/
```

Remaining Phase 1 items:

- `utils.sh :: write_run_metadata` → `rocm-smi`; `vllm_run_docker.sh`/`sgl_run_docker.sh` → skip the nsys mount when vendor is amd.
- `config.sh` → add the `mi355x` GPU type; `HF_HUB_CACHE` stays `/data/amatveev/hf` (host path → container `/app/hf_hub_cache`, same mount convention as today). Worth hardening `utils.sh` with a default (`HF_HUB_CACHE=${HF_HUB_CACHE:-/data/amatveev/hf}`) since the shell pipeline currently mounts `${HF_HUB_CACHE}` with no fallback.
- `run_profile_summary.py` → accept `trace-*.json`/`.json.gz` in the test-results layout (gunzip in place if needed).
- New campaign `llm/auto_profile/test_configs/deepseek_v41_flash_mi355x/`: `docker_images.json` (the two ROCm images), `gpu_configs.json` (`gpu_ids_8` = 0–7), `exec_modes.json` (reuse `pure_decode` isl 4 / osl 1024 and the 32k kv-cache-fill mode), run config with both frameworks, `enable_trace`, concurrency `[1]` initially (mirror of the NVIDIA campaign; extend to a sweep later). Mode files are seeded from the verified recipes rather than left empty: SGLang gets the cookbook env (`SGLANG_USE_AITER=1`, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, `TORCH_BLAS_PREFER_HIPBLASLT=1`, `GPU_MAX_HW_QUEUES=2`, `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`) plus `--fp8-gemm-backend aiter` where the bench accepts it, without DSpark speculative decoding; vLLM gets `VLLM_ROCM_USE_AITER=1`.
- Run: `/profile` with the new run config → `/parse` → `/analyze` (single + cross-trace) → the same vLLM-vs-SGLang comparison the B200 campaign produces.

### Phase 3 — optional: rocprofiler-SDK system-level tracing

`rocprofv3` (attach to a running server with `-p <pid>`, rocpd SQLite output, `rocpd convert --output-format pftrace` for Perfetto) and `rocprof-compute` for hardware-counter deep dives, integrated as a new trace source for `auto_analyze` — new format, new parsing. Also note SGLang's serving API already exposes the ROCm-native `RPD` activity (§5), which would be the cheapest way to get rocprofiler data into the existing pipeline. Defer until the basic comparison loop is running and there is a concrete question the torch traces can't answer.

---

## §7 Risks and open questions

| Risk | Assessment / mitigation | Severity |
|---|---|---|
| DeepSeek-V4.1-Flash maturity on ROCm | Well-contained: vLLM ships a model-specific `deepseekv41-flash-0909` image and a dedicated `vllm/models/deepseek_v41/amd/` implementation (ROCM AITER MLA attention); SGLang's cookbook has a verified MI355X recipe — but V4.1-Flash hasn't shipped in an sglang release, so it rides on daily preview images (pin the date-suffixed tag). Phase 0's real-model run is the gate. | high impact, likely OK |
| Exact `--profiler-config` CLI form on the image's vLLM | Resolved: vLLM registers `--profiler-config` as a single flag taking one JSON string (verified in the local checkout's `vllm/engine/arg_utils.py` + `vllm/config/profiler.py`); the server_client adapter now emits that form for both `torch` and `nsys` types. One `--help` check on the image during Phase 0 still confirms the v0.31.0 build matches. | resolved |
| gzip'd torch traces in the parse step | vLLM gzips by default (`.json.gz`); the parse step must gunzip or accept. One small change, listed in Phase 1. | small |
| SGLang module rename | `bench_one_batch` → `benchmark.one_batch`; the old path errors immediately on v0.5.21, so it fails fast rather than silently. | small |
| Torch trace size | Stack capture is on by default in vLLM's profiler config; traces can get large. The 50-iteration window bounds it; disable stacks if files become unwieldy. | tuning |
| Docker install + disk layout | One-time admin step (sudo in wheel). Image data-root must be `/data`; the 763 GB model also lives on `/data`. | trivial |
| Multimodal architecture | Image-text-to-text model; the campaign's random *text* datasets should be unaffected, but watch the processor load path during Phase 0 validation. | low |
| Keeping NVIDIA campaigns working | Every change is vendor-gated; the default path is unchanged when `nvidia-smi` is present. | design constraint |

### What stays untouched

`auto_analyze` (PYTORCH trace support already exists), `auto_code_gen` + `pr_flow` + `pr_review` (repo-based, vendor-neutral), the parse prompt logic (only the file-pattern acceptance widens), the trt pipeline (not enabled in the AMD campaign), and `/chrome-trace` (Chrome-trace JSON is already its native input).

---

## Verification provenance

- Host hardware, ROCm version, profiler tools, disks, groups, local checkouts — probed directly on dell-mi355x-3.
- Image tags, dates, and deprecations — Docker Hub registry API, 2026-10-09.
- vLLM ROCm image base and gfx950 support — `docker/Dockerfile.rocm_base` in the local vllm checkout (base `rocm/dev-ubuntu-22.04:7.2.3-complete`, arch list includes gfx950).
- `deepseek_v41` version floor — vllm `registry.py` across tags (first in v0.30.0).
- Torch profiler on ROCm — empirical run on this host's stack: Chrome trace with GPU kernel events.
- vLLM / SGLang profiler plumbing — read from the local checkouts (vllm main; sglang v0.5.21): `ProfilerConfig`, throughput.py profile handling, one_batch profiling args and defaults.
- SGLang FP8-on-ROCm claim and docker flags — SGLang and vLLM official AMD documentation.
- Model facts — Hugging Face model API, 2026-10-09.
- SGLang DeepSeek-V4.1 cookbook (verified MI355X recipe), vLLM profiling docs, AMD's "vLLM V1 on Instinct" optimization guide, and ROCprofiler-SDK docs — fetched from docs.sglang.io, docs.vllm.ai, and rocm.docs.amd.com by the background research pass (its WebSearch backend was down; every claim cites primary docs).
- SGLang ROCm image repo (`lmsysorg/sglang-rocm`, date-suffixed daily tags) — re-verified directly against the Docker Hub API after the research pass flagged it; `lmsysorg/sglang` itself carries only CUDA/CANN tags.
- amd-smi / rocm-smi / rocminfo / rocprofv3 on host PATH — probed directly.
