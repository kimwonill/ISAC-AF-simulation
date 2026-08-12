#!/usr/bin/env bash
set -euo pipefail

if (( $# != 10 )); then
    echo "Expected 10 worker arguments; received $#." >&2
    exit 2
fi

WORKER_ID="$1"
MC_START="$2"
MC_END="$3"
MC="$4"
FORCE_RERUN="$5"
MAX_RETRIES="$6"
SIM_DIR="$7"
CVX_DIR="$8"
MOSEK_MATLAB_DIR="$9"
LOG_DIR="${10}"
WORKER_LOG="${LOG_DIR}/worker_${WORKER_ID}_${MC_START}_${MC_END}.log"

: >"$WORKER_LOG"
for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
    force_arg="false"
    if [[ "$FORCE_RERUN" == "1" && "$attempt" -eq 1 ]]; then
        force_arg="true"
    fi
    printf '=== attempt %d/%d force=%s started %s ===\n' \
        "$attempt" "$MAX_RETRIES" "$force_arg" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$WORKER_LOG"
    if matlab -singleCompThread -softwareopengl -batch \
        "restoredefaultpath; addpath('$MOSEK_MATLAB_DIR'); addpath(genpath('$CVX_DIR')); cd('$SIM_DIR'); addpath(genpath(pwd)); run_cv_stress_rank1_experiment(${MC}, ${force_arg}, ${MC_START}:${MC_END});" \
        >>"$WORKER_LOG" 2>&1; then
        printf '=== worker completed %s ===\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$WORKER_LOG"
        exit 0
    fi
    printf '=== attempt %d failed; resuming point files ===\n' \
        "$attempt" >>"$WORKER_LOG"
    sleep 3
done

printf '=== worker failed after %d attempts ===\n' \
    "$MAX_RETRIES" >>"$WORKER_LOG"
exit 1
