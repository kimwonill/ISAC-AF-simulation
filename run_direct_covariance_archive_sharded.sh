#!/usr/bin/env bash
set -euo pipefail

# Direct-SCA only: no proposed CV, CRB, or MI optimization is run.
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC="${MC:-100}"
# The reference server has 32 physical cores. Each MATLAB/MOSEK process is
# single-threaded, so 32 workers use all physical cores without SMT overlap.
WORKERS="${WORKERS:-32}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
CVX_DIR="${CVX_DIR:-/home/wonill/matlab/cvx}"
MOSEK_MATLAB_DIR="${MOSEK_MATLAB_DIR:-/home/wonill/mosek/11.2/toolbox/r2019b}"
OUTPUT_DIR="${SIM_DIR}/results/direct_covariance_archive_MC${MC}"
LOG_DIR="${OUTPUT_DIR}/logs"
STATUS_FILE="${LOG_DIR}/status.txt"
LOCK_FILE="${OUTPUT_DIR}/run.lock"
mkdir -p "$LOG_DIR"

if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to guard this archive run." >&2
    exit 2
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another Direct covariance archive run is active: ${LOCK_FILE}" >&2
    exit 2
fi

if (( MC != 100 )); then
    echo "This archive replays the tracked MC=100 thresholds; set MC=100." >&2
    exit 2
fi
if (( WORKERS < 1 )); then
    echo "WORKERS must be a positive integer." >&2
    exit 2
fi
if (( WORKERS > MC )); then
    WORKERS="$MC"
fi

finish_run() {
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
        printf 'COMPLETED %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_FILE"
    else
        printf 'FAILED exit=%d %s\n' "$rc" \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_FILE"
    fi
}
trap finish_run EXIT
printf 'RUNNING mc=%s workers=%s started=%s\n' \
    "$MC" "$WORKERS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_FILE"

force_matlab=false
if [[ "$FORCE_RERUN" == "1" ]]; then
    force_matlab=true
fi

run_worker() {
    local worker_id="$1"
    local mc_start="$2"
    local mc_end="$3"
    local worker_log="${LOG_DIR}/worker_${worker_id}_${mc_start}_${mc_end}.log"
    local attempt
    local attempt_force

    : >"$worker_log"
    for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
        attempt_force=false
        if [[ "$force_matlab" == "true" && "$attempt" -eq 1 ]]; then
            attempt_force=true
        fi
        printf '=== attempt %d/%d started %s ===\n' \
            "$attempt" "$MAX_RETRIES" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >>"$worker_log"
        if matlab -singleCompThread -softwareopengl -batch \
            "restoredefaultpath; addpath('$MOSEK_MATLAB_DIR'); addpath(genpath('$CVX_DIR')); cd('$SIM_DIR'); addpath(genpath(pwd)); run_direct_covariance_archive(${MC}, ${attempt_force}, ${mc_start}:${mc_end});" \
            >>"$worker_log" 2>&1; then
            printf '=== worker completed %s ===\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$worker_log"
            return 0
        fi
        printf '=== attempt %d failed; point-level resume will retry ===\n' \
            "$attempt" >>"$worker_log"
        sleep 5
    done
    printf '=== worker failed after %d attempts ===\n' \
        "$MAX_RETRIES" >>"$worker_log"
    return 1
}

pids=()
worker=0
for ((worker_id = 1; worker_id <= WORKERS; worker_id++)); do
    mc_start=$(( (worker_id - 1) * MC / WORKERS + 1 ))
    mc_end=$(( worker_id * MC / WORKERS ))
    worker=$((worker + 1))
    run_worker "$worker" "$mc_start" "$mc_end" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one Direct-SCA archive worker failed. See ${LOG_DIR}." >&2
    exit "$status"
fi

matlab -singleCompThread -softwareopengl -batch \
    "restoredefaultpath; cd('$SIM_DIR'); addpath(genpath(pwd)); finalize_direct_covariance_archive(${MC});" \
    >"${LOG_DIR}/finalize.log" 2>&1
echo "Direct covariance archive completed: ${OUTPUT_DIR}"
