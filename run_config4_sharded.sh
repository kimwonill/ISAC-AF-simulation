#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC=100
LOG_DIR="${SIM_DIR}/results/pareto_grid_1x4_pslr/sharded_mc${MC}"
mkdir -p "$LOG_DIR"

ranges=("1 13" "14 25" "26 38" "39 50" "51 63" "64 75" "76 88" "89 100")
pids=()
worker=0
for range in "${ranges[@]}"; do
    read -r mc_start mc_end <<< "$range"
    worker=$((worker + 1))
    matlab -singleCompThread -softwareopengl -batch \
        "restoredefaultpath; addpath(genpath('/home/wonill/matlab/cvx')); cd('$SIM_DIR'); addpath(genpath(pwd)); plot_pareto_grid_1x4(${MC}, 4, ${mc_start}:${mc_end});" \
        >"${LOG_DIR}/worker_${worker}_${mc_start}_${mc_end}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one config 4 shard failed. See ${LOG_DIR}." >&2
    exit "$status"
fi

matlab -singleCompThread -softwareopengl -batch \
    "restoredefaultpath; addpath(genpath('/home/wonill/matlab/cvx')); cd('$SIM_DIR'); addpath(genpath(pwd)); merge_pareto_grid_1x4_shards(${MC}, 4); plot_pareto_grid_1x4(${MC}, 4);"
