#!/usr/bin/env bash
set -euo pipefail

# Remote GPU monitor for X-MinimaxH3 benchmark.
#
# Usage:
#   ./gpu_monitor_benchmark.sh [GPU_INDEX] [INTERVAL_MS]
#
# Example:
#   ./gpu_monitor_benchmark.sh 1 200
#
# Output CSV columns:
#   timestamp,index,memory.used,memory.free,memory.total,
#   utilization.gpu,utilization.memory,temperature.gpu,power.draw
#
# Notes:
# - No log file is written on the server.
# - The local Python benchmark consumes stdout directly over SSH.
# - Ctrl+C or closing the SSH channel stops nvidia-smi.

GPU_INDEX="${1:-1}"
INTERVAL_MS="${2:-200}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found" >&2
    exit 127
fi

if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid GPU index: $GPU_INDEX" >&2
    exit 2
fi

if ! [[ "$INTERVAL_MS" =~ ^[0-9]+$ ]] || (( INTERVAL_MS < 50 )); then
    echo "ERROR: invalid interval_ms: $INTERVAL_MS (minimum 50)" >&2
    exit 2
fi

if ! nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index \
    --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "ERROR: GPU index $GPU_INDEX is not available" >&2
    exit 3
fi

exec nvidia-smi \
    -i "$GPU_INDEX" \
    --query-gpu=timestamp,index,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv,noheader,nounits \
    -lms "$INTERVAL_MS"
