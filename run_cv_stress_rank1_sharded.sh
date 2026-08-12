#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC="${MC:-100}"
WORKERS="${WORKERS:-32}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
CVX_DIR="${CVX_DIR:-/home/wonill/matlab/cvx}"
MOSEK_MATLAB_DIR="${MOSEK_MATLAB_DIR:-/home/wonill/mosek/11.2/toolbox/r2019b}"
RUN_DIR="${SIM_DIR}/results/cv_stress_rank1_MC${MC}"
LOG_DIR="${RUN_DIR}/logs"
STATUS_FILE="${RUN_DIR}/status.txt"
LOCK_FILE="${RUN_DIR}/run.lock"
mkdir -p "$LOG_DIR"

if ! [[ "$MC" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[1-9][0-9]*$ && \
        "$MAX_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "MC, WORKERS, and MAX_RETRIES must be positive integers." >&2
    exit 2
fi
if [[ "$FORCE_RERUN" != "0" && "$FORCE_RERUN" != "1" ]]; then
    echo "FORCE_RERUN must be 0 or 1." >&2
    exit 2
fi
if (( WORKERS > MC )); then
    WORKERS="$MC"
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A stress rank-one run already holds ${LOCK_FILE}." >&2
    exit 1
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${RANDOM}"
export STRESS_WORKERS="$WORKERS"
export STRESS_RUN_ID="$RUN_ID"
export STRESS_REQUIRE_RUN_ID="$FORCE_RERUN"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

pids=()
worker_groups=()
finish_run() {
    local rc=$?
    trap - EXIT INT TERM
    for pgid in "${worker_groups[@]:-}"; do
        if [[ -n "$pgid" ]]; then
            kill -- "-${pgid}" 2>/dev/null || true
        fi
    done
    for _ in {1..10}; do
        local any_alive=0
        for pgid in "${worker_groups[@]:-}"; do
            if [[ -n "$pgid" ]] && kill -0 -- "-${pgid}" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        if [[ "$any_alive" -eq 0 ]]; then
            break
        fi
        sleep 1
    done
    for pgid in "${worker_groups[@]:-}"; do
        if [[ -n "$pgid" ]] && kill -0 -- "-${pgid}" 2>/dev/null; then
            kill -KILL -- "-${pgid}" 2>/dev/null || true
        fi
    done
    for pid in "${pids[@]:-}"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ "$rc" -eq 0 ]]; then
        printf 'COMPLETED run_id=%s workers=%s %s\n' \
            "$RUN_ID" "$WORKERS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$STATUS_FILE"
    else
        printf 'FAILED run_id=%s workers=%s exit=%d %s\n' \
            "$RUN_ID" "$WORKERS" "$rc" \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_FILE"
    fi
    exit "$rc"
}
trap finish_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'RUNNING run_id=%s mc=%s workers=%s started=%s\n' \
    "$RUN_ID" "$MC" "$WORKERS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$STATUS_FILE"

ranges=()
for ((worker = 1; worker <= WORKERS; worker++)); do
    mc_start=$(( (worker - 1) * MC / WORKERS + 1 ))
    mc_end=$(( worker * MC / WORKERS ))
    ranges+=("${mc_start} ${mc_end}")
done

worker=0
for range in "${ranges[@]}"; do
    read -r mc_start mc_end <<<"$range"
    worker=$((worker + 1))
    setsid bash "${SIM_DIR}/run_cv_stress_rank1_worker.sh" \
        "$worker" "$mc_start" "$mc_end" "$MC" "$FORCE_RERUN" \
        "$MAX_RETRIES" "$SIM_DIR" "$CVX_DIR" "$MOSEK_MATLAB_DIR" \
        "$LOG_DIR" &
    worker_pid="$!"
    pids+=("$worker_pid")
    worker_groups+=("$worker_pid")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
pids=()
worker_groups=()
if [[ "$status" -ne 0 ]]; then
    echo "At least one stress worker failed. See ${LOG_DIR}." >&2
    exit "$status"
fi

setsid matlab -singleCompThread -softwareopengl -batch \
    "restoredefaultpath; addpath('$MOSEK_MATLAB_DIR'); addpath(genpath('$CVX_DIR')); cd('$SIM_DIR'); addpath(genpath(pwd)); finalize_cv_stress_rank1_results(${MC});" &
finalizer_pid="$!"
pids+=("$finalizer_pid")
worker_groups+=("$finalizer_pid")
wait "$finalizer_pid"
pids=()
worker_groups=()
