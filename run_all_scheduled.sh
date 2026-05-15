#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 <run_config> <datetime>"
    echo ""
    echo "Schedules run_all.sh to execute at the specified date/time."
    echo "Before running, verifies that the required GPUs have no active processes."
    echo ""
    echo "Arguments:"
    echo "  run_config  Path to run config JSON (e.g., ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json)"
    echo "  datetime    When to run, in any format accepted by 'date -d', e.g.:"
    echo "              '2026-05-01 03:00'    specific date and time"
    echo "              'tomorrow 14:30'      relative date"
    echo "              '2026-05-01T03:00:00' ISO format"
    echo ""
    echo "Example:"
    echo "  $0 ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json '2026-05-01 03:00'"
    exit 1
}

if [ $# -ne 2 ]; then
    usage
fi

RUN_CONFIG="$1"
SCHEDULED_TIME="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$RUN_CONFIG" ]; then
    echo "Error: run config not found: $RUN_CONFIG"
    exit 1
fi

test_name=$(python3 -c "from common.utils import test_name_from_run_config; print(test_name_from_run_config('$RUN_CONFIG'))")

TARGET_EPOCH=$(date -d "$SCHEDULED_TIME" +%s 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$TARGET_EPOCH" ]; then
    echo "Error: could not parse datetime: '$SCHEDULED_TIME'"
    echo "Use a format like '2026-05-01 03:00' or 'tomorrow 14:30'"
    exit 1
fi

CURRENT_EPOCH=$(date +%s)
if [ "$TARGET_EPOCH" -le "$CURRENT_EPOCH" ]; then
    echo "Error: scheduled time is in the past: $(date -d "@$TARGET_EPOCH" '+%Y-%m-%d %H:%M:%S')"
    exit 1
fi

INFRA_CONFIG="${SCRIPT_DIR}/auto_profile/test_configs/infra_config.json"

get_required_gpu_ids() {
    python3 -c "
import json, sys

with open('$INFRA_CONFIG') as f:
    infra = json.load(f)

gpu_configs = {}
gpu_configs_file = infra.get('gpu_configs_file')
if gpu_configs_file:
    with open(gpu_configs_file) as f:
        gpu_configs = json.load(f)

with open('$RUN_CONFIG') as f:
    run = json.load(f)

all_gpu_ids = set()
for profile in run.get('profiles', []):
    gpu_ids = profile.get('gpu_ids')
    if isinstance(gpu_ids, str):
        gpu_ids = gpu_configs.get(gpu_ids, [])
    if isinstance(gpu_ids, list):
        all_gpu_ids.update(gpu_ids)

print(' '.join(str(g) for g in sorted(all_gpu_ids)))
"
}

REQUIRED_GPUS=$(get_required_gpu_ids)
if [ -z "$REQUIRED_GPUS" ]; then
    echo "Error: could not determine required GPU IDs from config"
    exit 1
fi

check_gpus_free() {
    local busy_gpus
    busy_gpus=$(python3 -c "
import subprocess, sys

bus_id_to_index = {}
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=index,gpu_bus_id', '--format=csv,noheader'],
    capture_output=True, text=True
)
for line in result.stdout.strip().splitlines():
    parts = [p.strip() for p in line.split(',')]
    if len(parts) == 2:
        bus_id_to_index[parts[1]] = int(parts[0])

busy_indices = set()
result = subprocess.run(
    ['nvidia-smi', '--query-compute-apps=gpu_bus_id', '--format=csv,noheader'],
    capture_output=True, text=True
)
for line in result.stdout.strip().splitlines():
    bus_id = line.strip()
    if bus_id in bus_id_to_index:
        busy_indices.add(bus_id_to_index[bus_id])

required = {${REQUIRED_GPUS// /, }}
conflicts = required & busy_indices
if conflicts:
    print(' '.join(str(g) for g in sorted(conflicts)))
")

    if [ -n "$busy_gpus" ]; then
        echo "$busy_gpus"
        return 1
    fi
    return 0
}

WAIT_SECONDS=$((TARGET_EPOCH - CURRENT_EPOCH))
TARGET_DISPLAY=$(date -d "@$TARGET_EPOCH" '+%Y-%m-%d %H:%M:%S')

echo "=== Scheduled Run ==="
echo "Run config:     $RUN_CONFIG"
echo "Test:           $test_name"
echo "Required GPUs:  $REQUIRED_GPUS"
echo "Scheduled for:  $TARGET_DISPLAY"
echo "Waiting:        ${WAIT_SECONDS}s (~$((WAIT_SECONDS / 3600))h $((WAIT_SECONDS % 3600 / 60))m)"
echo "====================="
echo ""
echo "Waiting until $TARGET_DISPLAY ..."

sleep "$WAIT_SECONDS"

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') - Scheduled time reached. Checking GPU availability..."

MAX_RETRIES=60
RETRY_INTERVAL=60

for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    if busy=$(check_gpus_free); then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - All required GPUs are free. Starting run_all.sh"
        echo ""
        exec "${SCRIPT_DIR}/run_all.sh" "$RUN_CONFIG"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - GPUs in use: $busy (attempt $attempt/$MAX_RETRIES, retrying in ${RETRY_INTERVAL}s)"
        sleep "$RETRY_INTERVAL"
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - Error: GPUs did not become free after $MAX_RETRIES attempts ($((MAX_RETRIES * RETRY_INTERVAL / 60)) minutes). Aborting."
exit 1
