#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SIM_DIR}/results/cv_stress_axis_sharded_mc100"
PID_FILE="${LOG_DIR}/master.pid"
MASTER_LOG="${LOG_DIR}/master.log"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(tr -dc '0-9' <"$PID_FILE")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Figure 7 MC=100 is already running with PID ${old_pid}."
        exit 1
    fi
fi

nohup setsid env MC=100 WORKERS="${WORKERS:-1}" CV_STEP="${CV_STEP:-0.05}" \
    TIME_BUDGET="${TIME_BUDGET:-3}" FORCE_RERUN=1 MAX_RETRIES=3 \
    bash "${SIM_DIR}/run_cv_stress_axis_sharded.sh" \
    >"$MASTER_LOG" 2>&1 < /dev/null &
job_pid=$!
printf '%s\n' "$job_pid" >"$PID_FILE"

echo "Started detached Figure 7 MC=100 job."
echo "PID: ${job_pid}"
echo "Master log: ${MASTER_LOG}"
echo "Status: ${LOG_DIR}/status.txt"
