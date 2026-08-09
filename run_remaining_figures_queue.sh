#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SIM_DIR}/results/queued_figures_mc100"
mkdir -p "$LOG_DIR"

{
    echo "Waiting for Figure 4 finalization supervisor."
    pidwait 2207950 || true
    echo "Figure 4 finalization ended. Starting Figure 7."
    matlab -singleCompThread -softwareopengl -batch \
        "restoredefaultpath; addpath(genpath('/home/wonill/matlab/cvx')); cd('$SIM_DIR'); addpath(genpath(pwd)); run_cv_stress_axis_experiment(100, true);"
    echo "Figure 7 complete. Starting Figure 8."
    matlab -singleCompThread -softwareopengl -batch \
        "restoredefaultpath; addpath(genpath('/home/wonill/matlab/cvx')); cd('$SIM_DIR'); addpath(genpath(pwd)); run_ml_learning_advantage_experiment(100, 100, true);"
    echo "Figure 8 complete. Starting Figure 10."
    matlab -singleCompThread -softwareopengl -batch \
        "restoredefaultpath; addpath(genpath('/home/wonill/matlab/cvx')); cd('$SIM_DIR'); addpath(genpath(pwd)); plot_pdes_pareto_sweep(100, true);"
    echo "Figures 7, 8, and 10 complete."
} >"${LOG_DIR}/queue.log" 2>&1
