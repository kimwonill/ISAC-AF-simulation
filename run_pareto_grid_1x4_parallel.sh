#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC="${1:-100}"
LOG_DIR="${SIM_DIR}/results/pareto_grid_1x4_pslr/logs_mc${MC}"
mkdir -p "$LOG_DIR"

pids=()
for index in 1 2 3 4; do
    matlab -singleCompThread -softwareopengl -batch \
        "restoredefaultpath; addpath(genpath('/home/wonill/matlab/cvx')); cd('$SIM_DIR'); addpath(genpath(pwd)); plot_pareto_grid_1x4(${MC}, ${index});" \
        >"${LOG_DIR}/config_${index}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [[ "$status" -ne 0 ]]; then
    echo "At least one Figure 4 configuration failed. See ${LOG_DIR}." >&2
    exit "$status"
fi

matlab -singleCompThread -softwareopengl -batch \
    "restoredefaultpath; addpath(genpath('/home/wonill/matlab/cvx')); cd('$SIM_DIR'); addpath(genpath(pwd)); plot_pareto_grid_1x4(${MC});"

echo "Figure 4 parallel regeneration complete: MC=${MC}"
