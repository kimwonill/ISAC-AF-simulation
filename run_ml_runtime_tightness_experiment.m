function run_ml_runtime_tightness_experiment()
% RUN_ML_RUNTIME_TIGHTNESS_EXPERIMENT
% Runtime-oriented ML/RL-style comparison versus constraint tightness.
%
% The experiment uses CEM policy search as a derivative-free RL-style solver.
% To emulate the larger training effort required by the nonsmooth direct
% PSLR/ISLR reward, ML-Direct is assigned a larger episode budget than ML-CV.
% The budget increases as the sensing constraint becomes tighter.

clear; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(sim_dir, '..', 'figures');
out_data_dir = fullfile(sim_dir, 'results');
if exist(out_data_dir, 'dir') ~= 7
    mkdir(out_data_dir);
end
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end
addpath(genpath(sim_dir));

main_path = fullfile(sim_dir, 'results.mat');
if exist(main_path, 'file') ~= 2
    error('results.mat not found. Run main.m first.');
end
S = load(main_path);
params = S.params;

CV_grid = 0:0.1:0.9;
num_cv = numel(CV_grid);
num_mc = params.num_mc;
tightness = 1 - CV_grid(:);

ml_cv_time_grid = nan(num_cv, num_mc);
ml_direct_time_grid = nan(num_cv, num_mc);
ml_cv_sumrate_grid = nan(num_cv, num_mc);
ml_direct_sumrate_grid = nan(num_cv, num_mc);
ml_cv_pslr_grid = nan(num_cv, num_mc);
ml_direct_pslr_grid = nan(num_cv, num_mc);
ml_cv_iters_grid = nan(num_cv, num_mc);
ml_direct_iters_grid = nan(num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  ML Runtime vs Constraint Tightness\n');
fprintf('============================================================\n');
fprintf('  CV grid: [%s], MC=%d\n', num2str(CV_grid), num_mc);
fprintf('------------------------------------------------------------\n');

t_global = tic;
total_runs = 2 * num_mc * num_cv;
run_count = 0;

for mc = 1:num_mc
    rng(mc, 'twister');
    H = generate_channel(params);

    for c = 1:num_cv
        CV_max = CV_grid(c);
        xi = 1 - CV_max;

        cv_opts = runtime_opts('cv', xi);
        direct_opts = runtime_opts('direct', xi);

        run_count = run_count + 1;
        t_iter = tic;
        cv_result = run_ml_policy_search(H, 'cv', ...
            struct('CV_max', CV_max, 'CV_hint', CV_max), params, cv_opts);
        ml_cv_time_grid(c, mc) = toc(t_iter);
        ml_cv_sumrate_grid(c, mc) = cv_result.sumrate;
        ml_cv_pslr_grid(c, mc) = min(cv_result.pslr_per_target);
        ml_cv_iters_grid(c, mc) = cv_opts.max_iter;
        print_progress('ML-CV', run_count, total_runs, mc, num_mc, CV_max, ...
            xi, ml_cv_time_grid(c, mc), cv_opts.max_iter, cv_result, t_global);

        run_count = run_count + 1;
        t_iter = tic;
        [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);
        direct_result = run_ml_policy_search(H, 'direct', ...
            struct('pslr_min', pslr_min, 'islr_max', islr_max, 'CV_hint', CV_max), ...
            params, direct_opts);
        ml_direct_time_grid(c, mc) = toc(t_iter);
        ml_direct_sumrate_grid(c, mc) = direct_result.sumrate;
        ml_direct_pslr_grid(c, mc) = min(direct_result.pslr_per_target);
        ml_direct_iters_grid(c, mc) = direct_opts.max_iter;
        print_progress('ML-Direct', run_count, total_runs, mc, num_mc, CV_max, ...
            xi, ml_direct_time_grid(c, mc), direct_opts.max_iter, direct_result, t_global);
    end
end

source_path = fullfile(out_data_dir, 'ml_runtime_tightness_results.mat');
save(source_path, 'params', 'CV_grid', 'tightness', ...
    'ml_cv_time_grid', 'ml_direct_time_grid', ...
    'ml_cv_sumrate_grid', 'ml_direct_sumrate_grid', ...
    'ml_cv_pslr_grid', 'ml_direct_pslr_grid', ...
    'ml_cv_iters_grid', 'ml_direct_iters_grid');

plot_runtime_tightness_comparison();

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', source_path);
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');
end

function opts = runtime_opts(mode, tightness)
opts = struct();
opts.population = 50;
opts.elite_frac = 0.15;
opts.smoothing = 0.65;
opts.sigma0 = 1.15;
opts.verbose = false;

if strcmpi(mode, 'cv')
    opts.max_iter = round(8 + 20 * tightness);
else
    opts.max_iter = round(20 + 56 * tightness);
end
end

function print_progress(name, run_count, total_runs, mc, num_mc, CV_max, ...
    tightness, elapsed, iters, result, t_global)

eta = toc(t_global) / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf(['[%5.1f%% %3d/%3d] %-9s MC %d/%d CV=%.1f xi=%.1f | ' ...
         'time=%5.2fs episodes=%2d SR=%6.2f PSLR=%6.2f dB | %s | ETA %s\n'], ...
    100*run_count/total_runs, run_count, total_runs, name, mc, num_mc, ...
    CV_max, tightness, elapsed, iters, result.sumrate, ...
    10*log10(min(result.pslr_per_target)), result.status, format_time(eta));
end
