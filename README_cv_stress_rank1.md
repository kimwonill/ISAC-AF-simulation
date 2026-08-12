# Rank-one stress experiment

Run the detached MC=100 experiment on the 32-core server with

```bash
WORKERS=32 bash ./start_cv_stress_rank1_mc100.sh
```

The default run resumes validated point files. To deliberately recompute every
point, set `FORCE_RERUN=1`. Every MATLAB process and every MOSEK solve uses one
thread; Monte Carlo indices are split over the workers.

Progress is written to
`results/cv_stress_rank1_MC100/status.txt`, with the master and worker logs in
the same run directory. Each `(scenario, CV_max, MC)` point is atomically saved,
so a failed worker retry resumes completed points.

After all workers finish, the finalizer creates
`results/cv_stress_rank1_S1S2S3_CV10_NT4_N16_MC100.mat` and exports the square
panel `2 x 3` figure as `figures/CV_Stress_RankOne_2x3.{pdf,png}`. The figure
contains absolute runtime and total IPM iterations in the top row, and
time-budgeted, EVD-only, and EVD+GR feasibility in the bottom row.

The saved runtime is wall-clock time under the recorded worker count. Changing
`WORKERS` changes the concurrency environment and can therefore change the
absolute runtime and deadline-based feasibility.

Each method is run quietly for wall-clock timing and then deterministically
replayed without timing to parse the accumulated MOSEK IPM iterations. This
preserves the prior runtime protocol while collecting complete IPM counts.
