#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC="${MC:-100}"
WORKERS="${WORKERS:-1}"
CV_STEP="${CV_STEP:-0.05}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
CVX_DIR="${CVX_DIR:-/home/wonill/matlab/cvx}"
MOSEK_MATLAB_DIR="${MOSEK_MATLAB_DIR:-/home/wonill/mosek/11.2/toolbox/r2019b}"
LOG_DIR="${SIM_DIR}/results/cv_stress_axis_sharded_mc${MC}"
mkdir -p "$LOG_DIR"
STATUS_FILE="${LOG_DIR}/status.txt"

finish_run() {
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
        printf 'COMPLETED %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_FILE"
    else
        printf 'FAILED exit=%d %s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_FILE"
    fi
}
trap finish_run EXIT
printf 'RUNNING mc=%s workers=%s cv_step=%s started=%s\n' \
    "$MC" "$WORKERS" "$CV_STEP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_FILE"

if (( MC < 1 || WORKERS < 1 )); then
    echo "MC and WORKERS must be positive integers." >&2
    exit 2
fi
if (( WORKERS > MC )); then
    WORKERS="$MC"
fi
CV_POINTS="$(awk -v step="$CV_STEP" 'BEGIN { printf "%d", 1 / step + 0.5 }')"

# Independent Monte Carlo realizations are split across MATLAB processes.
# -singleCompThread prevents each CVX worker from oversubscribing the CPU.
ranges=()
matlab_ranges="["
for ((worker = 1; worker <= WORKERS; worker++)); do
    mc_start=$(( (worker - 1) * MC / WORKERS + 1 ))
    mc_end=$(( worker * MC / WORKERS ))
    ranges+=("${mc_start} ${mc_end}")
    matlab_ranges+="${mc_start} ${mc_end};"
done
matlab_ranges+="]"

pids=()
worker=0
run_shard() {
    local worker_id="$1"
    local mc_start="$2"
    local mc_end="$3"
    local worker_log="${LOG_DIR}/worker_${worker_id}_${mc_start}_${mc_end}.log"
    local attempt

    : >"$worker_log"
    for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
        printf '=== attempt %d/%d started %s ===\n' \
            "$attempt" "$MAX_RETRIES" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >>"$worker_log"
        if matlab -singleCompThread -softwareopengl -batch \
            "restoredefaultpath; addpath('$MOSEK_MATLAB_DIR'); addpath(genpath('$CVX_DIR')); cd('$SIM_DIR'); addpath(genpath(pwd)); run_cv_stress_axis_experiment(${MC}, true, ${mc_start}:${mc_end}, ${CV_STEP}, false);" \
            >>"$worker_log" 2>&1; then
            printf '=== shard completed %s ===\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$worker_log"
            return 0
        fi
        printf '=== attempt %d failed; retrying ===\n' "$attempt" >>"$worker_log"
        sleep 5
    done
    printf '=== shard failed after %d attempts ===\n' "$MAX_RETRIES" >>"$worker_log"
    return 1
}

for range in "${ranges[@]}"; do
    read -r mc_start mc_end <<< "$range"
    worker=$((worker + 1))
    shard_path="${SIM_DIR}/results/cv_stress_axis_pslr_only_S2S3S4_CV${CV_POINTS}_NT4_N16_MC${MC}_shard_$(printf '%03d' "$mc_start")_$(printf '%03d' "$mc_end").mat"
    if [[ "$FORCE_RERUN" != "1" && -f "$shard_path" ]]; then
        echo "Reusing completed shard ${mc_start}:${mc_end}"
        continue
    fi
    run_shard "$worker" "$mc_start" "$mc_end" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one Figure 7 shard failed. See ${LOG_DIR}." >&2
    exit "$status"
fi

matlab -singleCompThread -softwareopengl -batch \
    "restoredefaultpath; addpath('$MOSEK_MATLAB_DIR'); addpath(genpath('$CVX_DIR')); cd('$SIM_DIR'); addpath(genpath(pwd)); merge_cv_stress_axis_shards(${MC},${matlab_ranges},${CV_STEP}); run_cv_stress_axis_experiment(${MC}, false, [], ${CV_STEP}, true);"
