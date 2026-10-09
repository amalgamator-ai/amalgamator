#!/bin/bash
set -euo pipefail

# run_iteration.sh — compile + bench patched + bench baseline + perf comparison + lm_eval.
#
# Expected env vars:
#   OUT              — per-job output directory
#   VLLM_DIR         — vllm source tree on the workspace PVC
#   HF_HOME          — HuggingFace cache directory
#   BENCH_CMD        — benchmark command (must write --output-json $OUT/bench_result.json)
#   SERVER_CMD       — command that launches the API server used by BENCH_CMD
#   DISABLE_ENV_VAR  — name of env var that disables the feature (e.g. VLLM_FUSED_A_GEMM_DISABLE)
#   RUN_LM_EVAL      — "1" to run lm_eval after benchmarks, "0" to skip
#   LM_EVAL_CMD      — lm_eval command (ignored if RUN_LM_EVAL != "1")
#   STAGE            — (optional) "compile", "bench", "baseline", or "lm_eval"
#                      to run only that phase.  "compile" exits after build.
#                      Empty or "all" runs everything sequentially (backward
#                      compatible).
#   SKIP_CMAKE       — (optional) "1" to skip cmake build (artifacts already on PVC)

export UV_CACHE_DIR=/tmp/uv-cache
export TORCH_CUDA_ARCH_LIST="10.0"
export FLASHINFER_JIT_DIR=/tmp/flashinfer
export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer
export VLLM_CACHE_ROOT=/tmp/vllm-cache

STAGE="${STAGE:-all}"

mkdir -p "$OUT"
touch "$OUT/build.log" "$OUT/runner.log"
echo "[cluster_runner] out=$OUT vllm=$VLLM_DIR stage=$STAGE" | tee $OUT/runner.log

BUILD_FAIL_STATUS="BUILD_FAILED"
STATUS_FILE="$OUT/run_status.txt"

# ── candidate isolation: copy base tree + apply patch inside the pod ─────────
if [ -n "${PATCH_FILE:-}" ] && [ -n "${VLLM_BASE_DIR:-}" ]; then
    echo "[cluster_runner] creating candidate work copy: $VLLM_DIR" | tee -a $OUT/runner.log
    cp -r "$VLLM_BASE_DIR" "$VLLM_DIR"
    cd "$VLLM_DIR"
    set +e
    git apply "$PATCH_FILE" > $OUT/patch_apply.log 2>&1
    APPLY_EXIT=$?
    cat $OUT/patch_apply.log | tee -a $OUT/runner.log
    set -e
    if [ "$APPLY_EXIT" -ne 0 ]; then
        echo "PATCH_APPLY_FAILED" > $OUT/run_status.txt
        echo "[cluster_runner] patch apply FAILED (exit $APPLY_EXIT)" | tee -a $OUT/runner.log
        exit 1
    fi
    echo "[cluster_runner] patch applied from $PATCH_FILE" | tee -a $OUT/runner.log
fi

# === INCLUDE: venv_compile.sh ===

# ── compile-only: exit early after build ─────────────────────────
if [ "$STAGE" = "compile" ]; then
    echo "COMPLETED" > $OUT/run_status.txt
    echo "[cluster_runner] compile-only done." | tee -a $OUT/runner.log
    exit 0
fi

# ── API server lifecycle + benchmark result validation ──────────
SERVER_PID=""

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "[cluster_runner] stopping server pid=$SERVER_PID" | tee -a "$OUT/runner.log"
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

start_server() {
    local feature_mode="$1"
    local log_file="$2"
    local server_port
    server_port="$(echo "$SERVER_CMD" | sed -n 's/.*--port[ =]\([0-9][0-9]*\).*/\1/p')"
    server_port="${server_port:-8000}"

    stop_server
    : > "$log_file"
    nvidia-smi topo -m > "$OUT/topology.log" 2>&1 || true
    echo "$SERVER_CMD" > "$OUT/server_command_${feature_mode}.txt"
    echo "[cluster_runner] starting server feature=$feature_mode port=$server_port" \
        | tee -a "$OUT/runner.log"

    if [ "$feature_mode" = "OFF" ]; then
        setsid env "$DISABLE_ENV_VAR=1" bash -c "exec $SERVER_CMD" \
            > "$log_file" 2>&1 &
    else
        setsid bash -c "exec $SERVER_CMD" > "$log_file" 2>&1 &
    fi
    SERVER_PID=$!

    for _ in $(seq 1 900); do
        if curl -fsS "http://127.0.0.1:${server_port}/health" >/dev/null 2>&1; then
            echo "[cluster_runner] server ready feature=$feature_mode pid=$SERVER_PID" \
                | tee -a "$OUT/runner.log"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[cluster_runner] server exited before readiness" | tee -a "$OUT/runner.log"
            tail -100 "$log_file" | tee -a "$OUT/runner.log"
            SERVER_PID=""
            return 1
        fi
        sleep 2
    done

    echo "[cluster_runner] server readiness timed out" | tee -a "$OUT/runner.log"
    tail -100 "$log_file" | tee -a "$OUT/runner.log"
    return 1
}

validate_benchmark_result() {
    local result_file="$1"
    python3 - "$result_file" <<'PYEOF'
import json
import os
import sys

path = sys.argv[1]
if not os.path.isfile(path):
    raise SystemExit("benchmark result missing: {}".format(path))
with open(path) as f:
    result = json.load(f)
completed = result.get("completed")
if not isinstance(completed, (int, float)) or completed <= 0:
    raise SystemExit("benchmark completed={!r}; expected > 0".format(completed))
print("[cluster_runner] validated {}: completed={}".format(path, completed))
PYEOF
}

trap stop_server EXIT

# ── warmup run (discard results — stabilizes torch.compile caches + GPU thermals)
if [ "$STAGE" = "all" ] || [ "$STAGE" = "bench" ]; then
    if [ -z "${SERVER_CMD:-}" ] || ! start_server "ON" "$OUT/server_enabled.log"; then
        echo "SERVER_START_FAILED" > "$OUT/run_status.txt"
        exit 1
    fi
    WARMUP_CMD="$(echo "$BENCH_CMD" | sed 's|bench_result.json|warmup_bench_result.json|')"
    echo "[cluster_runner] bench warmup (results discarded)..." | tee -a $OUT/runner.log
    set +e
    eval "$WARMUP_CMD" > >(tee $OUT/warmup_stdout.log) 2> >(tee $OUT/warmup_stderr.log >&2)
    WARMUP_EXIT=$?
    set -e
    if [ "$WARMUP_EXIT" -ne 0 ]; then
        echo "[cluster_runner] warmup FAILED (exit $WARMUP_EXIT) — continuing anyway" | tee -a $OUT/runner.log
    else
        echo "[cluster_runner] warmup OK" | tee -a $OUT/runner.log
    fi
fi

# ── benchmark patched (feature ON) ───────────────────────────────
if [ "$STAGE" = "all" ] || [ "$STAGE" = "bench" ]; then
    echo "[cluster_runner] bench feature=ON..." | tee -a $OUT/runner.log
    echo "[cluster_runner] cmd: $BENCH_CMD" | tee -a $OUT/runner.log
    echo "$BENCH_CMD" > $OUT/run_command.txt
    set +e
    eval "$BENCH_CMD" > >(tee $OUT/run_stdout.log) 2> >(tee $OUT/run_stderr.log >&2)
    BENCH_EXIT=$?
    set -e
    echo $BENCH_EXIT > $OUT/run_exit_code.txt
    cat $OUT/run_stdout.log $OUT/run_stderr.log > $OUT/run_combined.log
    if [ "$BENCH_EXIT" -ne 0 ] || ! validate_benchmark_result "$OUT/bench_result.json" \
            2>&1 | tee -a "$OUT/runner.log"; then
        echo "BENCH_FAILED" > $OUT/run_status.txt
        echo "[cluster_runner] BENCH FAILED (exit $BENCH_EXIT)" | tee -a $OUT/runner.log
        exit 1
    fi
    echo "[cluster_runner] bench OK" | tee -a $OUT/runner.log
    stop_server
fi

# ── warmup baseline (discard results — stabilizes caches for the OFF graph)
if [ "$STAGE" = "all" ] || [ "$STAGE" = "baseline" ]; then
    if [ -z "${SERVER_CMD:-}" ] || ! start_server "OFF" "$OUT/server_disabled.log"; then
        echo "SERVER_START_FAILED" > "$OUT/run_status.txt"
        exit 1
    fi
    BASELINE_WARMUP_CMD="$(echo "$BENCH_CMD" | sed 's|bench_result.json|warmup_baseline_bench_result.json|')"
    echo "[cluster_runner] baseline warmup (results discarded)..." | tee -a $OUT/runner.log
    set +e
    eval "$DISABLE_ENV_VAR=1 $BASELINE_WARMUP_CMD" \
        > >(tee $OUT/warmup_baseline_stdout.log) 2> >(tee $OUT/warmup_baseline_stderr.log >&2)
    WARMUP_BASE_EXIT=$?
    set -e
    if [ "$WARMUP_BASE_EXIT" -ne 0 ]; then
        echo "[cluster_runner] baseline warmup FAILED (exit $WARMUP_BASE_EXIT) — continuing anyway" | tee -a $OUT/runner.log
    else
        echo "[cluster_runner] baseline warmup OK" | tee -a $OUT/runner.log
    fi
fi

# ── benchmark baseline (feature OFF) ─────────────────────────────
if [ "$STAGE" = "all" ] || [ "$STAGE" = "baseline" ]; then
    echo "[cluster_runner] bench feature=OFF..." | tee -a $OUT/runner.log
    BASELINE_CMD="$(echo "$BENCH_CMD" | sed 's|bench_result.json|baseline_bench_result.json|')"
    echo "[cluster_runner] cmd: $DISABLE_ENV_VAR=1 $BASELINE_CMD" | tee -a $OUT/runner.log
    echo "$DISABLE_ENV_VAR=1 $BASELINE_CMD" > $OUT/baseline_command.txt
    set +e
    eval "$DISABLE_ENV_VAR=1 $BASELINE_CMD" \
        > >(tee $OUT/baseline_stdout.log) 2> >(tee $OUT/baseline_stderr.log >&2)
    BASE_EXIT=$?
    set -e
    echo $BASE_EXIT > $OUT/baseline_exit_code.txt
    cat $OUT/baseline_stdout.log $OUT/baseline_stderr.log > $OUT/baseline_combined.log
    if [ "$BASE_EXIT" -ne 0 ] || ! validate_benchmark_result "$OUT/baseline_bench_result.json" \
            2>&1 | tee -a "$OUT/runner.log"; then
        echo "BASELINE_FAILED" > $OUT/run_status.txt
        echo "[cluster_runner] BASELINE FAILED (exit $BASE_EXIT)" | tee -a $OUT/runner.log
        exit 1
    fi
    echo "[cluster_runner] baseline OK" | tee -a $OUT/runner.log
    stop_server
fi

# ── perf comparison (only in sequential mode) ────────────────────
if [ "$STAGE" = "all" ]; then
    python3 - <<'PYEOF'
import json, os
out = os.environ["OUT"]
def load(f):
    try: return json.load(open(os.path.join(out, f)))
    except Exception: return {}
p = load("bench_result.json")
b = load("baseline_bench_result.json")
lines = ["Patched (feature ON) vs Baseline (feature OFF)\n"]
for key in sorted(set(list(p) + list(b))):
    pv, bv = p.get(key, "N/A"), b.get(key, "N/A")
    if isinstance(pv, (int, float)) and isinstance(bv, (int, float)) and bv != 0:
        d = (pv - bv) / abs(bv) * 100
        lines.append("  {}: patched={:.4g} baseline={:.4g} delta={:+.2f}%".format(
            key, pv, bv, d))
    else:
        lines.append("  {}: patched={} baseline={}".format(key, pv, bv))
open(os.path.join(out, "perf_comparison.txt"), "w").write("\n".join(lines) + "\n")
PYEOF
fi

# ── lm_eval (optional) ───────────────────────────────────────────
if [ "$STAGE" = "all" ] || [ "$STAGE" = "lm_eval" ]; then
    if [ "${RUN_LM_EVAL:-0}" = "1" ] && [ -n "${LM_EVAL_CMD:-}" ]; then
        if ! python3 -c "import lm_eval" 2>/dev/null; then
            echo "[cluster_runner] installing lm-eval..." | tee -a $OUT/runner.log
            uv pip install "lm-eval[api]" 2>&1 | tee -a $OUT/build.log
        fi

        # Run 1: lm_eval (feature ON unless FORCE_FEATURE_OFF=1)
        # A FORCE_FEATURE_OFF pod IS the OFF arm (shared lm_eval_off baseline),
        # so its artifacts use the lm_eval_disabled_* names and the
        # lm_eval_output_disabled dir that _save_shared_baseline and
        # _baseline_run_problem expect -- not the ON-arm slot names.
        if [ "${FORCE_FEATURE_OFF:-0}" = "1" ]; then
            echo "[cluster_runner] lm_eval feature=OFF (FORCE_FEATURE_OFF)..." | tee -a $OUT/runner.log
            LM_EVAL_RUN_CMD="$DISABLE_ENV_VAR=1 $LM_EVAL_CMD"
            LM_EVAL_RUN_CMD="${LM_EVAL_RUN_CMD//lm_eval_output/lm_eval_output_disabled}"
            LM_TAG="disabled"
        else
            echo "[cluster_runner] lm_eval feature=ON..." | tee -a $OUT/runner.log
            LM_EVAL_RUN_CMD="$LM_EVAL_CMD"
            LM_TAG="enabled"
        fi
        echo "[cluster_runner] cmd: $LM_EVAL_RUN_CMD" | tee -a $OUT/runner.log
        mkdir -p $OUT/lm_eval_output_disabled $OUT/lm_eval_output
        set +e
        eval "$LM_EVAL_RUN_CMD" \
            > >(tee $OUT/lm_eval_${LM_TAG}_stdout.log) \
            2> >(tee $OUT/lm_eval_${LM_TAG}_stderr.log >&2)
        LM_ENABLED_EXIT=$?
        set -e
        echo $LM_ENABLED_EXIT > $OUT/lm_eval_${LM_TAG}_exit_code.txt

        if [ "$LM_ENABLED_EXIT" -ne 0 ]; then
            echo "LM_EVAL_FAILED exit=$LM_ENABLED_EXIT (feature $LM_TAG run)" > $OUT/lm_eval.txt
            cat $OUT/lm_eval_${LM_TAG}_stdout.log $OUT/lm_eval_${LM_TAG}_stderr.log \
                > $OUT/lm_eval_combined.log
            echo "[cluster_runner] lm_eval FAILED feature=$LM_TAG (exit $LM_ENABLED_EXIT)" \
                | tee -a $OUT/runner.log
        elif [ "${SKIP_LM_EVAL_BASELINE:-0}" = "1" ]; then
            echo "[cluster_runner] lm_eval feature=$LM_TAG OK (skipping baseline — reusing shared)" \
                | tee -a $OUT/runner.log
        else
            echo "[cluster_runner] lm_eval feature=ON OK" | tee -a $OUT/runner.log

            # Run 2: lm_eval with feature DISABLED
            echo "[cluster_runner] lm_eval feature=OFF..." | tee -a $OUT/runner.log
            LM_EVAL_CMD_DISABLED="$(echo "$LM_EVAL_CMD" \
                | sed 's|lm_eval_output|lm_eval_output_disabled|g')"
            echo "[cluster_runner] cmd: $DISABLE_ENV_VAR=1 $LM_EVAL_CMD_DISABLED" \
                | tee -a $OUT/runner.log
            mkdir -p $OUT/lm_eval_output_disabled
            set +e
            eval "$DISABLE_ENV_VAR=1 $LM_EVAL_CMD_DISABLED" \
                > >(tee $OUT/lm_eval_disabled_stdout.log) \
                2> >(tee $OUT/lm_eval_disabled_stderr.log >&2)
            LM_DISABLED_EXIT=$?
            set -e
            echo $LM_DISABLED_EXIT > $OUT/lm_eval_disabled_exit_code.txt

            # Merge all logs for backward compat
            cat $OUT/lm_eval_enabled_stdout.log $OUT/lm_eval_enabled_stderr.log \
                $OUT/lm_eval_disabled_stdout.log $OUT/lm_eval_disabled_stderr.log \
                > $OUT/lm_eval_combined.log

            if [ "$LM_DISABLED_EXIT" -ne 0 ]; then
                echo "[cluster_runner] lm_eval FAILED feature=OFF (exit $LM_DISABLED_EXIT)" \
                    | tee -a $OUT/runner.log
            else
                echo "[cluster_runner] lm_eval feature=OFF OK" | tee -a $OUT/runner.log
            fi

            # Compare both runs and write verdict
            python3 - <<'PYEOF'
import json, glob, os

out = os.environ["OUT"]
enabled_exit = int(open(
    os.path.join(out, "lm_eval_enabled_exit_code.txt")).read().strip())
disabled_exit = int(open(
    os.path.join(out, "lm_eval_disabled_exit_code.txt")).read().strip())

def extract_scores(result_dir):
    results = sorted(glob.glob(
        os.path.join(result_dir, "**", "*.json"), recursive=True))
    if not results:
        return None, None
    r = json.load(open(results[-1])).get("results", {}).get("gsm8k", {})
    return (r.get("exact_match,flexible-extract"),
            r.get("exact_match,strict-match"))

lines = []

flex_en, strict_en = extract_scores(os.path.join(out, "lm_eval_output"))
lines.append("Feature ENABLED results: flexible-extract={}, strict-match={}".format(
    flex_en, strict_en))
lines.append("Feature ENABLED exit code: {}".format(enabled_exit))

if disabled_exit != 0:
    lines.append("Feature DISABLED results: FAILED (exit {})".format(disabled_exit))
    lines.append("Delta: N/A (disabled run failed)")
    lines.append("Verdict: INCONCLUSIVE (disabled run failed, enabled score={})".format(
        flex_en if flex_en is not None else strict_en))
else:
    flex_dis, strict_dis = extract_scores(
        os.path.join(out, "lm_eval_output_disabled"))
    lines.append("Feature DISABLED results: flexible-extract={}, strict-match={}".format(
        flex_dis, strict_dis))
    lines.append("Feature DISABLED exit code: {}".format(disabled_exit))

    deltas = []
    if flex_en is not None and flex_dis is not None:
        fd = flex_en - flex_dis
        lines.append("Delta flexible-extract: {:+.4f}".format(fd))
        deltas.append(fd)
    if strict_en is not None and strict_dis is not None:
        sd = strict_en - strict_dis
        lines.append("Delta strict-match: {:+.4f}".format(sd))
        deltas.append(sd)

    if not deltas:
        lines.append("Verdict: INCONCLUSIVE (could not extract scores)")
    elif any(d < -0.01 for d in deltas):
        lines.append("Verdict: FAIL (significant accuracy degradation detected)")
    else:
        lines.append("Verdict: PASS (scores within tolerance)")

open(os.path.join(out, "lm_eval.txt"), "w").write("\n".join(lines) + "\n")
PYEOF
        fi
    fi
fi

echo "COMPLETED" > $OUT/run_status.txt
echo "[cluster_runner] done." | tee -a $OUT/runner.log
