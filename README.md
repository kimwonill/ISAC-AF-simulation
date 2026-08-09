# ISAC Ambiguity-Function Simulation

MATLAB simulation code for ambiguity-function-aware MIMO-OFDM integrated
sensing and communications (ISAC) beamforming. The repository compares the
proposed coefficient-of-variation (CV) constrained SDP with direct PSLR SCA,
direct ISLR, communications-only, CRB-inspired, MI-inspired, and selected ML
baselines.

For a file-by-file description of the experiments, algorithms, generated
figures, result files, cache naming, and parallel runners, see
[EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md).

## Requirements

- MATLAB
- [CVX](http://cvxr.com/cvx/)
- MOSEK for the paper-scale Figure 7 configuration
- Bash/Linux only for the optional parallel and detached runner scripts

The most recent Figure 7 run used MATLAB R2025b, CVX, MOSEK 11.2.2, and one
MOSEK thread per MATLAB worker. Paths to CVX and MOSEK in the shell scripts
are machine-specific and should be overridden before running on another
machine.

## Quick start

From MATLAB, change to this directory and add it to the path:

```matlab
cd('/path/to/ISAC-AF-simulation');
addpath(genpath(pwd));
```

Common entry points are:

```matlab
main;                                      % baseline Pareto experiment
plot_pareto_grid_1x4(100);                 % four-configuration Pareto grid
run_cv_stress_axis_experiment(100, false); % Figure 7, reuse cache if present
run_ml_learning_advantage_experiment;      % ML learning comparison
plot_pdes_pareto_sweep(100, false);        % illumination-floor sweep
```

Generated numerical data live under `results/`; rendered PNG/PDF figures
live under `figures/`. Files containing `_shard_`, worker logs, PID files,
and status files are intermediate parallel-run artifacts rather than the
canonical scientific results.
