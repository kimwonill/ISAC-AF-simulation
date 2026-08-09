# Experiments, Algorithms, and Result Files

This document explains what each simulation entry point does, where it saves
its data, and which files are final results versus temporary caches. It is
intended for readers who did not develop the repository and need to inspect,
reproduce, or extend the experiments safely.

## 1. Repository conventions

The repository uses the following layout:

| Location | Meaning |
| --- | --- |
| Repository root (`*.m`) | Simulation entry points, algorithms, solvers, metrics, and plotting utilities |
| `results/` | MATLAB source data (`.mat`), cached Monte Carlo shards, and diagnostic previews |
| `figures/` | Final rendered figures (`.png` and `.pdf`) |
| Root-level `.fig`, `.png`, and `.mat` files | Legacy/editable outputs retained for compatibility with older scripts |
| `results/*/worker_*.log` | Per-worker execution logs; useful for debugging but not scientific source data |
| `results/*/status.txt`, `*.pid` | Run-control metadata; not scientific source data |

Important naming conventions:

- `MC100` means 100 Monte Carlo channel realizations.
- `NT4_N16` means four transmit antennas and 16 OFDM subcarriers.
- `CV10` means a CV grid with step 0.1, not `CV = 10`.
- `_grid` variables normally retain values for every operating point and
  Monte Carlo realization. Means and confidence intervals should be derived
  from these arrays rather than from a rendered figure.
- `_lin` denotes a linear-scale metric. A `_dB` variable is already converted
  to decibels.
- `_shard_001_025` denotes an intermediate result containing MC indices
  1 through 25. Shards must be merged before they are treated as a complete
  result.

Most experiment functions accept a `force_rerun` argument. When it is
`false`, an existing `.mat` cache is loaded and only the plots/summary are
regenerated. When it is `true`, the numerical experiment is recomputed and
the matching cache is overwritten.

## 2. Software requirements and reproducibility

The MATLAB code requires CVX. Paper-scale Figure 7 runs are configured for
MOSEK 11.2.2 with one MOSEK thread per MATLAB process. Parallelism is achieved
by launching multiple independent single-threaded MATLAB workers, with each
worker receiving a disjoint range of Monte Carlo indices.

The shell runners contain paths that were valid on the original server:

```text
/home/wonill/matlab/cvx
/home/wonill/mosek/11.2/toolbox/r2019b
```

On another machine, override `CVX_DIR` and `MOSEK_MATLAB_DIR` where supported,
or edit the MATLAB path setup in the relevant runner. Do not assume that the
machine's saved CVX default solver matches the solver used to produce a cache.
Interior-point iteration counts are particularly solver- and version-dependent.

For reproducible comparisons, keep the following fixed across methods:

1. Channel seed and initialization.
2. MATLAB, CVX, and solver versions.
3. Solver thread count.
4. Cold-start/warm-start policy.
5. Feasibility definition and wall-clock budget.
6. Which execution supplies runtime and iteration/solve-count measurements.

## 3. Primary manuscript experiments

### 3.1 Baseline Pareto experiment

| Item | Description |
| --- | --- |
| Entry point | `main.m` |
| Purpose | Sweeps `CV_max`, runs the proposed AO/CV-SDP algorithm, and compares sum rate against PSLR and ISLR. It can also run direct and surrogate baselines according to `setup_params.m`. |
| Canonical data | `results.mat` in the repository root |
| Main outputs | `pareto_curve.png/.fig`, `pareto_curve_islr.png/.fig`, `comparison_time.png/.fig`, plus manuscript copies produced by the script |
| Follow-up | `append_cv1_results.m` computes only the missing `CV_max = 1.0` proposed samples and appends them to `results.mat`. |

`plot_pareto_results.m` reloads `results.mat`, completes any compatible cached
reference curves, writes figure-source data under `results/`, and exports the
PSLR/ISLR Pareto figures.

### 3.2 Four-configuration PSLR Pareto grid (Figure 4 workflow)

| Item | Description |
| --- | --- |
| Entry point | `plot_pareto_grid_1x4.m` |
| Purpose | Runs or loads four `(N_T, N)` configurations and plots a one-row, four-panel PSLR-only Pareto comparison. Despite its name, this file is both an experiment driver and a plotter. |
| Configurations | `(4,16)`, `(4,32)`, `(8,16)`, `(8,32)` |
| Canonical data | `results/pareto_grid_1x4_pslr/pareto_pslr_NT*_N*_MC*.mat` |
| Final figures | `figures/Pareto_Frontier_Grid_1x4.png` and `.pdf` |
| Compatibility wrapper | `plot_pareto_grid_2x4.m` now calls the 1x4 implementation. Older 2x4 caches remain under `results/pareto_grid_2x4/`. |

Each canonical Pareto cache contains `params`, `CV_max_list`, proposed
`sumrate_grid`/`pslr_lin_grid`, direct-SCA grids, communications-only grids,
CRB-inspired grids, MI-inspired grids, and completion masks. Completion masks
allow interrupted runs to resume without discarding completed MC samples.

Parallel helpers for this workflow are described in Section 7.

### 3.3 CV stress-axis diagnostic (Figure 7 workflow)

| Item | Description |
| --- | --- |
| Entry point | `run_cv_stress_axis_experiment.m` |
| Purpose | Compares CV-SDP and direct PSLR SCA over `CV_max = 0:0.1:1` for three stressed scenarios. |
| Scenarios | S2 higher illumination, S3 more targets, and S4 joint QoS/illumination stress |
| Canonical MC=100 data | `results/cv_stress_axis_pslr_only_S2S3S4_CV10_NT4_N16_MC100.mat` |
| Final figures | `figures/CV_Stress_Axis_Diagnostic.png/.pdf` and `figures/CV_Stress_Axis_Diagnostic_OneColumn_1x3.png/.pdf` |
| Parallel runner | `run_cv_stress_axis_sharded.sh` |
| Detached launcher | `start_cv_stress_axis_mc100.sh` |

The canonical MC=100 file currently contains:

- `CV_grid`, `num_mc`, and the `scenarios` structure.
- `prop_success` and `direct_success` solver-success flags.
- `prop_status` and `direct_status` CVX status strings.
- `prop_time` and `direct_time` measured runtimes.
- `prop_cvx_solver_iters` and `direct_cvx_solver_iters`.
- `time_budget_seconds` and `budget_policy` used for post-processing.

The time budget is a post-processing feasibility threshold. It does not stop
an optimization after that number of seconds. A run is counted as budget
feasible only when it reports a solved status and its measured runtime is no
larger than the scenario budget.

#### Important provenance note for the current Figure 7 cache

The current MC=100 cache was generated with the code path that measured
runtime in a quiet timed run and obtained IPM counts from a separate
solver-log replay. It does **not** yet store convex-subproblem counts,
per-solve IPM histories, solver/CVX/MATLAB version strings, Git commit, or a
complete provenance structure. Therefore, do not interpret this cache as if
it already follows a timed-run-only solve-count protocol. If Figure 7 is
regenerated under a revised protocol, both this documentation and the saved
schema should be updated together.

IPM counts from older MC=15/16 files are not directly comparable with the
current MOSEK 11.2.2 cache. Tests on the same channel seed showed that changing
the solver/backend can change the recorded total by several-fold.

The following are intermediate rather than canonical results:

```text
results/cv_stress_axis_pslr_only_*_shard_*.mat
results/cv_stress_axis_sharded_mc*/worker_*.log
results/cv_stress_axis_sharded_mc*/master.log
results/cv_stress_axis_sharded_mc*/status.txt
results/cv_stress_axis_sharded_mc*/master.pid
```

### 3.4 ML learning-advantage experiment (Figure 8 workflow)

| Item | Description |
| --- | --- |
| Entry point | `run_ml_learning_advantage_experiment.m` |
| Purpose | Uses a fixed label-free training budget to compare how easily a CV constraint and a direct PSLR constraint can be learned/enforced. |
| Canonical data | `results/ml_learning_advantage_pslr_only_results.mat` |
| Final figures | `figures/ML_CV_Advantage_Integrated_Result.png/.pdf` |

The result file contains the CV sweep, training/validation MC counts, options,
trained CV/direct policies, sweep metrics, learning traces, and total elapsed
time. Passing `force_rerun = false` reuses a cache only when its stored
training and validation sample counts match the requested counts.

### 3.5 Illumination-floor experiments (Figure 10 workflow)

| Entry point | Purpose | Canonical data | Final figure |
| --- | --- | --- | --- |
| `plot_pdes_beampattern.m` | Sweeps `P_des = n P_max/N` and plots optimized directional beam power versus azimuth. | `results/beamgain_pdes_sweep_results.mat` | `figures/beamgain_pdes_sweep.png/.pdf` |
| `plot_pdes_pareto_sweep.m` | Sweeps both `P_des` and `CV_max` and plots sum-rate/PSLR frontiers. | `results/pdes_pareto_sweep_results.mat` | `figures/pdes_pareto_sweep.png/.pdf` |

`standardize_pdes_figures_from_fig.m` reopens the saved `.fig` files and
re-exports them with the current paper style without rerunning CVX.

## 4. Other numerical experiments

Every `run_*.m` file is listed here. Some are manuscript experiments; others
are diagnostics or algorithm routines called by a higher-level experiment.

| File | Role |
| --- | --- |
| `run_computational_burden_experiment.m` | Cold-start CV-SDP versus direct-SCA comparison of runtime, convex-subproblem solves, and accumulated solver iterations. Results follow `results/computational_burden_<init>_NT4_N16_MC*.mat`. |
| `run_cv_stress_axis_experiment.m` | Figure 7 stress-axis experiment described above. |
| `run_feasibility_stress_experiment.m` | Diagnostic, not a primary manuscript figure. Tests algorithmic feasibility under stricter QoS, illumination, and degree-of-freedom scenarios. Saves `results/feasibility_stress_MC*.mat`. |
| `run_integrated_stress_interior_preview.m` | Lightweight solver-log profiling used by the legacy integrated-stress preview. Saves `results/integrated_stress_interior_preview_MC*.mat`. |
| `run_ml_constraint_landscape_experiment.m` | Perturbs directional-power profiles to compare the smooth CV landscape against direct peak-sidelobe active-index switching. Saves `results/ml_constraint_landscape_results.mat`. |
| `run_ml_constraint_surrogate_experiment.m` | Trains the same random-feature regressor on CV and direct peak-sidelobe targets to compare surrogate learnability. Saves `results/ml_constraint_surrogate_results.mat`. |
| `run_ml_experiments.m` | Legacy label-free CEM baselines layered on `results.mat`. Saves `results/ml_experiment_results.mat`. |
| `run_ml_inference_experiments.m` | Offline expert-label training followed by frozen online inference and feasibility projection. Saves model and inference result caches under `results/`. |
| `run_ml_learning_advantage_experiment.m` | Primary label-free CV-versus-direct learning comparison described above. |
| `run_ml_raw_policy_learning_experiment.m` | Controlled raw-policy experiment without a handcrafted feasibility projection. Saves `results/ml_raw_policy_learning_results.mat`. |
| `run_ml_reward_inference_experiments.m` | Label-free reward-trained policy followed by frozen inference. Saves policy and result caches under `results/`. |
| `run_ml_runtime_tightness_experiment.m` | Compares ML/RL-style runtime as the sensing constraint becomes tighter. Saves `results/ml_runtime_tightness_results.mat`. |
| `run_runtime_coldstart_experiment.m` | Independently initializes every channel/CV point and compares cold-start runtimes. Saves `results/runtime_coldstart_results.mat`. |
| `run_sdr_covariance_evd_learning_experiment.m` | Supervised proof of concept that predicts SDR covariance matrices and extracts rank-one beamformers by principal EVD. Saves `results/ml_sdr_covariance_evd_results.mat`. |
| `run_proposed.m` | Core proposed AO algorithm. Alternates the CV-constrained SDP and subcarrier-allocation update. `result.iters` is the number of convex SDP solves performed by this routine. |
| `run_direct_sca.m` | Core direct-PSLR baseline. Alternates outer allocation updates and inner SCA beamforming solves. `result.inner_iters` is the total number of convex subproblems solved. |
| `run_direct_islr_exact.m` | Thin wrapper expressing the exact direct-ISLR/CV-SOC equivalence while reusing `run_proposed.m`. |
| `run_surrogate_baseline.m` | AO wrapper for CRB- and MI-inspired non-AF surrogate baselines. |
| `run_ml_policy_search.m` | Cross-entropy policy-search engine used by several ML experiments; not a stand-alone figure driver. |

Additional experiment-like entry points:

| File | Purpose and result |
| --- | --- |
| `check_islr_cv_equivalence.m` | Numerically verifies exact ISLR/CV equivalence and saves `results/islr_cv_equivalence_NT4_N16_MC*.mat`. |
| `generate_supervised_cv_labels.m` | Creates cold-start CV-SDP labels for optional neural-network experiments; saves `results/supervised_cv_labels_NT4_N16_train*_test*.mat`. |
| `explore_optimized_pslr_distribution.m` | Studies PSLR distributions induced by optimized beamformers; saves `results/optimized_pslr_distribution_source_data.mat`. |
| `explore_pslr_cv_distribution.m` | Exploratory fixed-CV profile sampler, not a paper-generation script; saves `results/pslr_cv_distribution_source_data.mat`. |

## 5. Core algorithms, solvers, and metrics

| File | Description |
| --- | --- |
| `setup_params.m` | Central default system, channel, optimization, and Monte Carlo parameters. Inspect this first when reproducing a result. |
| `generate_channel.m` | Generates a random frequency-selective MISO-OFDM channel from the configured model. |
| `compute_steering.m` | Builds array steering vectors at the configured target angles. |
| `solve_sdp.m` | One CV-constrained convex beamforming SDP solve used by the proposed AO method. |
| `solve_direct_sca_sdp.m` | One convexified direct-PSLR SCA beamforming subproblem. |
| `solve_surrogate_sdp.m` | Convex subproblem for CRB-/MI-inspired surrogate baselines. |
| `update_alpha.m` | Updates the subcarrier-to-user allocation after a beamforming solve. |
| `init_alpha.m` | Initial subcarrier allocation. Some experiments define a stricter QoS-safe initializer locally. |
| `init_covariance_flat.m` | Flat covariance initialization used by direct SCA. |
| `init_covariance_mrt.m` | MRT-based covariance initialization used by selected comparisons. |
| `compute_directional_power.m` | Computes directional power across subcarriers for one look direction. |
| `directional_power_grid.m` | Computes directional power for all configured look directions. |
| `compute_pslr.m` | Evaluates peak sidelobe ratio from a directional-power sequence. |
| `compute_islr.m` | Evaluates integrated sidelobe ratio. |
| `compute_rank_stats.m` | Summarizes numerical covariance rank after SDR. |
| `direct_thresholds_from_cv.m` | Maps a CV threshold to the direct PSLR/ISLR thresholds used for matched comparisons. |
| `cv_from_islr_threshold.m` | Inverse ISLR-to-CV map used by the exact equivalence wrapper. |
| `parse_cvx_solver_iterations.m` | Parses solver logs for interior-point iteration totals. Counts are solver/version dependent. |

## 6. Plot and export utilities

Some files named `plot_*` only render cached data; others also run an
experiment when their cache is absent. Read the header and cache check before
calling a plotting file on a new machine.

| File or group | Description |
| --- | --- |
| `plot_cv_theory_bounds.m` | Validates and plots theoretical CV-to-PSLR/ISLR bounds, using `results.mat` when available. |
| `plot_computational_burden_results.m` | Renders the computational-burden result and writes a compact summary cache. |
| `plot_integrated_stress_ladder_preview.m` | Combines legacy feasibility/runtime/IPM caches into an integrated stress preview. |
| `plot_runtime_coldstart_comparison.m` | Renders cold-start runtime results, optionally overlaying frozen ML inference. |
| `plot_runtime_tightness_comparison.m` | Renders runtime versus `1 - CV_max`. |
| `plot_ml_nn_pareto_nt4_n16.m` | Combines available neural-network Pareto caches for the `(N_T,N)=(4,16)` case. |
| `plot_af_simulation_example.m` | Runs or loads one deterministic optimized channel example and evaluates the closed-form equivalent sidelobe level. |
| `plot_af_surface_heatmap_comparison.m` | Runs or loads ambiguity-function surface/heatmap comparisons for proposed and reference designs. |
| `plot_af_3d_variants.m`, `plot_af_surface_heatmap_corner.m` | Alternative 3-D/corner visualizations from AF result caches. |
| `plot_af_zero_doppler_cut_comparison.m` | Zero-Doppler cut from the surface-comparison cache. |
| `plot_multitarget_worst_af_zero_doppler_*.m` | Multi-target worst-direction zero-Doppler comparisons using cached optimized cases. |
| `plot_single_channel_af_zero_doppler_comparison.m` | Single-channel zero-Doppler comparison. |
| `plot_pdes_beampattern.m`, `plot_pdes_pareto_sweep.m` | Illumination-floor experiment drivers described above. |
| `paper_palette.m`, `plot_config.m` | Shared paper colors, fonts, line widths, and layout defaults. |
| `tight_export_figure.m` | Consistent tight PNG/PDF export helper. |
| `add_panel_caption.m`, `draw_pdes_legend.m` | Shared caption and custom-legend helpers. |
| `format_time.m` | Converts elapsed seconds to a readable duration. |

## 7. Parallel and detached shell runners

| File | Purpose |
| --- | --- |
| `run_pareto_grid_1x4_parallel.sh` | Launches the four Figure 4 configurations in parallel, then exports the combined figure. |
| `run_config4_sharded.sh` | Splits only the most expensive fourth Figure 4 configuration across eight MC shards, merges them, and re-plots. |
| `merge_pareto_grid_1x4_shards.m` | Validates and merges Figure 4 shards into a canonical configuration cache. |
| `finalize_pareto_grid_1x4.sh` | Waits for supplied worker PIDs and exports the combined Figure 4. |
| `run_cv_stress_axis_sharded.sh` | Splits Figure 7 MC indices across workers, retries failed shards, merges them, and renders the final figure. Environment variables include `MC`, `WORKERS`, `FORCE_RERUN`, `MAX_RETRIES`, `CVX_DIR`, and `MOSEK_MATLAB_DIR`. |
| `merge_cv_stress_axis_shards.m` | Validates full MC coverage and merges Figure 7 shards. |
| `start_cv_stress_axis_mc100.sh` | Starts a detached Figure 7 MC=100 run, defaulting to four workers, and writes master PID/log/status files. |
| `run_remaining_figures_queue.sh` | Historical server-specific queue for Figures 7, 8, and 10. It contains a fixed PID and fixed paths and should not be used unchanged on another machine. |

The final Figure 7 MC=100 run recorded in the repository took approximately
one hour with four parallel MATLAB workers, each using one solver thread. That
run also performed a separate solver-log replay, so a timed-run-only protocol
would be expected to finish faster.

## 8. Which files should be included in a scientific commit?

A useful commit normally contains:

1. MATLAB source files and runner scripts used for the result.
2. The canonical merged `.mat` file for each reported experiment.
3. The final PNG and PDF figures.
4. This documentation and a commit message describing solver/version changes.

The following are usually omitted unless they are needed to audit a failed or
long-running job:

- Monte Carlo shard `.mat` files after a canonical merge has been verified.
- Worker logs, queue logs, PID files, and status files.
- Preview images and temporary exports.
- Legacy duplicate copies when a canonical `results/` copy exists.

Be careful with `git add -A`: this working tree can contain many intermediate
shards and logs. Review `git status --short` and add canonical files
explicitly. The repository currently contains a file named `gitignore`
without the leading dot; Git does not apply it automatically as `.gitignore`.

Before committing a Monte Carlo result, verify at minimum:

```matlab
S = load('path/to/canonical_result.mat');
whos('-file', 'path/to/canonical_result.mat')
```

Also confirm that all expected MC indices are present, numerical metric fields
contain no unintended `NaN` values, completion masks are true, the plotted
budget matches the saved budget metadata, and the final PNG/PDF timestamps are
newer than the canonical source data used to generate them.
