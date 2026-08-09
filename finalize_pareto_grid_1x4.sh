#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC="${1:-100}"
shift || true
LOG_PATH="${SIM_DIR}/results/pareto_grid_1x4_pslr/parallel_finalize_mc${MC}.log"

{
    echo "Waiting for Figure 4 workers: $*"
    for pid in "$@"; do
        pidwait "$pid" || true
    done
    echo "Workers finished; exporting combined Figure 4."
    matlab -singleCompThread -softwareopengl -batch \
        "restoredefaultpath; addpath(genpath('/home/wonill/matlab/cvx')); cd('$SIM_DIR'); addpath(genpath(pwd)); plot_pareto_grid_1x4(${MC});"
    echo "Combined Figure 4 export complete."
} >>"$LOG_PATH" 2>&1
