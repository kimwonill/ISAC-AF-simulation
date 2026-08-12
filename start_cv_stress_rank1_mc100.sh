#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SIM_DIR}/results/cv_stress_rank1_MC100"
PID_FILE="${RUN_DIR}/master.pid"
MASTER_LOG="${RUN_DIR}/master.log"
START_LOCK="${RUN_DIR}/start.lock"
mkdir -p "$RUN_DIR"

exec 8>"$START_LOCK"
if ! flock -n 8; then
    echo "Another stress launcher is starting." >&2
    exit 1
fi
exec 7>"${RUN_DIR}/run.lock"
if ! flock -n 7; then
    echo "A stress rank-one run is already active." >&2
    exit 1
fi
flock -u 7

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(tr -dc '0-9' <"$PID_FILE")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Stress rank-one MC=100 is already running with PID ${old_pid}."
        exit 1
    fi
fi

nohup setsid env MC=100 WORKERS="${WORKERS:-32}" \
    FORCE_RERUN="${FORCE_RERUN:-0}" MAX_RETRIES="${MAX_RETRIES:-3}" \
    CVX_DIR="${CVX_DIR:-/home/wonill/matlab/cvx}" \
    MOSEK_MATLAB_DIR="${MOSEK_MATLAB_DIR:-/home/wonill/mosek/11.2/toolbox/r2019b}" \
    bash "${SIM_DIR}/run_cv_stress_rank1_sharded.sh" \
    >"$MASTER_LOG" 2>&1 < /dev/null &
job_pid=$!
printf '%s\n' "$job_pid" >"$PID_FILE"

echo "Started detached stress rank-one MC=100 run."
echo "PID: ${job_pid}"
echo "Workers: ${WORKERS:-32}"
echo "Master log: ${MASTER_LOG}"
echo "Status: ${RUN_DIR}/status.txt"
