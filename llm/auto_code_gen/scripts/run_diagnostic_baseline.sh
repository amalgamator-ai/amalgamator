#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# run_diagnostic_baseline.sh — compile + nsys profile baseline (feature OFF) + SQLite export.
#
# Expected env vars:
#   OUT              — per-job output directory
#   VLLM_DIR         — vllm source tree on the workspace PVC
#   HF_HOME          — HuggingFace cache directory
#   BENCH_CMD        — benchmark command (the workload that drives the capture)
#   SERVER_CMD       — (optional) server launch command. When set, BENCH_CMD is a
#                      client and the engine runs in the server process, so the
#                      server is what gets profiled (with the feature forced off).
#   DISABLE_ENV_VAR  — name of env var that disables the feature
#   DELAY_ITERATIONS — warmup iterations before nsys tracing starts

export UV_CACHE_DIR=/tmp/uv-cache
export TORCH_CUDA_ARCH_LIST="10.0"
export FLASHINFER_JIT_DIR=/tmp/flashinfer
export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer
export VLLM_CACHE_ROOT=/tmp/vllm-cache
export SKIP_CMAKE=1

mkdir -p "$OUT"
touch "$OUT/build.log" "$OUT/runner.log"
echo "[cluster_runner] diagnostic-baseline out=$OUT HF_HOME=$HF_HOME" | tee -a $OUT/runner.log

BUILD_FAIL_STATUS="DIAGNOSTIC_BUILD_FAILED"
STATUS_FILE="$OUT/diagnostic_status.txt"

# === INCLUDE: venv_compile.sh ===


MAX_TRACE_ITERATIONS=50
# vLLM opens the capture window itself: skip DELAY_ITERATIONS iterations (torch
# compile, Triton autotune, cold caches) and record MAX_TRACE_ITERATIONS after.
PROFILER_FLAGS="--profiler-config.profiler cuda --profiler-config.delay_iterations ${DELAY_ITERATIONS} --profiler-config.max_iterations ${MAX_TRACE_ITERATIONS}"
NSYS_FLAGS="-t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --trace-fork-before-exec=true"

echo "Running BASELINE NSYS profile..." > $OUT/step1_nsys_profiles.txt

if [ -n "${SERVER_CMD:-}" ]; then
    # Server mode. `vllm bench serve` is a client: the engine, and therefore
    # every kernel worth measuring, lives in the server process, and
    # --profiler-config is a server flag in this vLLM (dotted form). Wrapping
    # the client instead both fails its argument parser ("unrecognized
    # arguments: --profiler-config ...") and, if it did not, would profile a
    # process that launches no kernels at all.
    SERVER_PORT="$(echo "$SERVER_CMD" | sed -n 's/.*--port[ =]\([0-9][0-9]*\).*/\1/p')"
    SERVER_PORT="${SERVER_PORT:-8000}"
    echo "[cluster_runner] warmup skipped in server mode (delay_iterations covers it)" \
        | tee -a $OUT/runner.log

    NSYS_CMD="env ${DISABLE_ENV_VAR}=1 nsys profile $NSYS_FLAGS -o $OUT/trace_baseline bash -c \"exec $SERVER_CMD $PROFILER_FLAGS\""
    echo "[cluster_runner] nsys profile baseline (server mode, port $SERVER_PORT)..." | tee -a $OUT/runner.log
    echo "[cluster_runner] cmd: $NSYS_CMD" | tee -a $OUT/runner.log
    echo "cmd: $NSYS_CMD" >> $OUT/step1_nsys_profiles.txt

    set +e
    setsid bash -c "$NSYS_CMD" > $OUT/server_baseline.log 2>&1 &
    PROFILE_PID=$!
    SERVER_READY=0
    for _ in $(seq 1 900); do
        if curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
            SERVER_READY=1
            break
        fi
        if ! kill -0 "$PROFILE_PID" 2>/dev/null; then
            echo "[cluster_runner] profiling server exited before readiness" | tee -a $OUT/runner.log
            tail -50 $OUT/server_baseline.log | tee -a $OUT/step1_nsys_profiles.txt
            break
        fi
        sleep 2
    done
    if [ "$SERVER_READY" = "1" ]; then
        # The CUDA profiler is armed over HTTP: delay_iterations counts engine
        # steps *after* /start_profile, and nsys only opens its cudaProfilerApi
        # capture range when the engine calls cudaProfilerStart. Without this
        # POST the workload runs, nothing is ever recorded and nsys writes an
        # empty (or missing) report.
        echo "[cluster_runner] arming baseline profiler (POST /start_profile)..." | tee -a $OUT/runner.log
        curl -sf -X POST "http://127.0.0.1:${SERVER_PORT}/start_profile" \
            > $OUT/start_profile.txt 2>&1
        echo "start_profile exit=$?" >> $OUT/step1_nsys_profiles.txt
        echo "[cluster_runner] profiling server ready; running bench workload..." \
            | tee -a $OUT/runner.log
        eval "$BENCH_CMD" > $OUT/bench_baseline_stdout.log 2> $OUT/bench_baseline_stderr.log
        echo "bench exit=$?" >> $OUT/step1_nsys_profiles.txt
        # Let the profiler stop and nsys close its capture range before dying.
        sleep 20
    fi
    kill -TERM -- "-$PROFILE_PID" 2>/dev/null || kill -TERM "$PROFILE_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "$PROFILE_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$PROFILE_PID" 2>/dev/null || true
    wait "$PROFILE_PID" 2>/dev/null || true
    if grep -q "Starting profiler after delay" $OUT/server_baseline.log 2>/dev/null; then
        echo "capture window opened (engine started profiler after delay)" \
            >> $OUT/step1_nsys_profiles.txt
    else
        echo "[cluster_runner] WARNING: engine never opened the capture window" \
            | tee -a $OUT/runner.log
        echo "capture window NEVER opened (no 'Starting profiler after delay')" \
            >> $OUT/step1_nsys_profiles.txt
    fi
    if [ "$SERVER_READY" != "1" ]; then
        echo "DIAGNOSTIC_NSYS_BASELINE_FAILED" > $OUT/diagnostic_status.txt
        echo "[cluster_runner] nsys baseline FAILED (server never became ready)" \
            | tee -a $OUT/runner.log
        exit 1
    fi
else
    # Offline mode: the bench command owns the engine, so it is profiled directly.
    WARMUP_CMD="$(echo "$BENCH_CMD" | sed 's|bench_result.json|warmup_bench_result.json|')"
    echo "[cluster_runner] baseline warmup bench (results discarded)..." | tee -a $OUT/runner.log
    set +e
    eval "${DISABLE_ENV_VAR}=1 $WARMUP_CMD" > $OUT/warmup_baseline_stdout.log 2> $OUT/warmup_baseline_stderr.log
    WARMUP_EXIT=$?
    set -e
    if [ "$WARMUP_EXIT" -ne 0 ]; then
        echo "[cluster_runner] warmup FAILED (exit $WARMUP_EXIT) — continuing anyway" | tee -a $OUT/runner.log
    else
        echo "[cluster_runner] warmup OK" | tee -a $OUT/runner.log
    fi

    NSYS_CMD="env ${DISABLE_ENV_VAR}=1 nsys profile $NSYS_FLAGS -o $OUT/trace_baseline $BENCH_CMD --profile --profiler-config '{\"profiler\":\"cuda\",\"delay_iterations\":${DELAY_ITERATIONS},\"max_iterations\":${MAX_TRACE_ITERATIONS}}'"
    echo "[cluster_runner] nsys profile baseline (offline mode)..." | tee -a $OUT/runner.log
    echo "[cluster_runner] cmd: $NSYS_CMD" | tee -a $OUT/runner.log
    echo "cmd: $NSYS_CMD" >> $OUT/step1_nsys_profiles.txt
    set +e
    env ${DISABLE_ENV_VAR}=1 nsys profile \
        -t cuda,nvtx \
        -c cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        --trace-fork-before-exec=true \
        -o $OUT/trace_baseline \
        $BENCH_CMD \
        --profile \
        --profiler-config '{"profiler":"cuda","delay_iterations":'${DELAY_ITERATIONS}',"max_iterations":'${MAX_TRACE_ITERATIONS}'}' \
        2>&1 | tee -a $OUT/step1_nsys_profiles.txt
    NSYS_BASELINE_EXIT=${PIPESTATUS[0]}
fi

if [ ! -f "$OUT/trace_baseline.nsys-rep" ]; then
    echo "[cluster_runner] no trace_baseline.nsys-rep produced" | tee -a $OUT/runner.log
    NSYS_BASELINE_EXIT=1
fi
set -e
echo "baseline nsys exit=${NSYS_BASELINE_EXIT:-0}" | tee -a $OUT/step1_nsys_profiles.txt
if [ "${NSYS_BASELINE_EXIT:-0}" -ne 0 ]; then
    echo "DIAGNOSTIC_NSYS_BASELINE_FAILED" > $OUT/diagnostic_status.txt
    echo "[cluster_runner] nsys baseline FAILED" | tee -a $OUT/runner.log
    exit 1
fi

# ── export baseline trace to SQLite ───────────────────────────────────
echo "[cluster_runner] nsys export baseline to SQLite..." | tee -a $OUT/runner.log
nsys export --type=sqlite \
    --output=$OUT/trace_baseline.sqlite \
    $OUT/trace_baseline.nsys-rep
echo "[cluster_runner] baseline SQLite export done." | tee -a $OUT/runner.log

echo "NSYS_DONE" > $OUT/diagnostic_status.txt
echo "[cluster_runner] diagnostic baseline profiling complete." | tee -a $OUT/runner.log
