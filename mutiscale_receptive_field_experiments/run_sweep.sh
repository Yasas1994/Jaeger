#!/usr/bin/env bash
# Run all multiscale + global-context experiments locally.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR" || exit 1

for cfg in multiscale_rf_baseline multiscale_rf_axial_attention multiscale_rf_bilstm multiscale_rf_mlp; do
    echo "========================================"
    echo "Starting $cfg at $(date -Iseconds)"
    echo "========================================"
    jaeger train -c "$SCRIPT_DIR/${cfg}.yaml" \
        --precision bf16 \
        --force \
        --xla \
        2>&1 | tee "$LOG_DIR/${cfg}.log"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "ERROR: $cfg failed with exit code $status" | tee -a "$LOG_DIR/${cfg}.log"
    else
        echo "DONE: $cfg at $(date -Iseconds)" | tee -a "$LOG_DIR/${cfg}.log"
    fi
done

echo "Sweep complete at $(date -Iseconds)"
