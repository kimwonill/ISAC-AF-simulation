function run_runtime_coldstart_experiment()
% RUN_RUNTIME_COLDSTART_EXPERIMENT
% Fair runtime comparison with no cross-CV warm start.
%
% Every scheme is initialized independently at each (channel, CV) point:
%   - CV-SDP: run_proposed(H,CV,params) with init_alpha only.
%   - Direct SCA: run_direct_sca(H,pslr_min,params) with init_alpha and
%     flat covariance linearization only.
%   - ML-CV / ML-Direct: CEM policy search from a generic policy prior;
%     no previous operating point or previous learned policy is reused.

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

params = setup_params();
params.warm_start_cv = false;
CV_grid = 0:0.1:1.0;
tightness = 1 - CV_grid(:);
num_cv = numel(CV_grid);
num_mc = params.num_mc;

prop_time_grid = nan(num_cv, num_mc);
direct_time_grid = nan(num_cv, num_mc);
ml_cv_time_grid = nan(num_cv, num_mc);
ml_direct_time_grid = nan(num_cv, num_mc);

prop_sumrate_grid = nan(num_cv, num_mc);
direct_sumrate_grid = nan(num_cv, num_mc);
ml_cv_sumrate_grid = nan(num_cv, num_mc);
ml_direct_sumrate_grid = nan(num_cv, num_mc);

prop_pslr_grid = nan(num_cv, num_mc);
direct_pslr_grid = nan(num_cv, num_mc);
ml_cv_pslr_grid = nan(num_cv, num_mc);
ml_direct_pslr_grid = nan(num_cv, num_mc);

prop_status_grid = strings(num_cv, num_mc);
direct_status_grid = strings(num_cv, num_mc);
ml_cv_status_grid = strings(num_cv, num_mc);
ml_direct_status_grid = strings(num_cv, num_mc);

ml_cv_iters_grid = nan(num_cv, num_mc);
ml_direct_iters_grid = nan(num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  Cold-start Runtime Experiment (no warm starts)\n');
fprintf('============================================================\n');
fprintf('  CV grid=[%s], MC=%d\n', num2str(CV_grid), num_mc);
fprintf('------------------------------------------------------------\n');

t_global = tic;
total_runs = 4 * num_cv * num_mc;
run_count = 0;

for mc = 1:num_mc
    rng(mc, 'twister');
    H = generate_channel(params);

    for c = 1:num_cv
        CV_max = CV_grid(c);
        xi = 1 - CV_max;
        pslr_min = direct_thresholds_from_cv(CV_max, params);

        run_count = run_count + 1;
        t_iter = tic;
        prop_result = run_proposed(H, CV_max, params);
        prop_time_grid(c, mc) = toc(t_iter);
        [prop_sumrate_grid(c, mc), prop_pslr_grid(c, mc), prop_status_grid(c, mc)] = ...
            summarize_result(prop_result);
        print_progress('CV-SDP', run_count, total_runs, mc, num_mc, CV_max, xi, ...
            prop_time_grid(c, mc), prop_result.status, prop_sumrate_grid(c, mc), ...
            prop_pslr_grid(c, mc), t_global);

        run_count = run_count + 1;
        t_iter = tic;
        direct_result = run_direct_sca(H, pslr_min, params);
        direct_time_grid(c, mc) = toc(t_iter);
        [direct_sumrate_grid(c, mc), direct_pslr_grid(c, mc), direct_status_grid(c, mc)] = ...
            summarize_result(direct_result);
        print_progress('DirectSCA', run_count, total_runs, mc, num_mc, CV_max, xi, ...
            direct_time_grid(c, mc), direct_result.status, direct_sumrate_grid(c, mc), ...
            direct_pslr_grid(c, mc), t_global);

        ml_cv_opts = runtime_opts('cv', xi);
        run_count = run_count + 1;
        t_iter = tic;
        ml_cv_result = run_ml_policy_search(H, 'cv', ...
            struct('CV_max', CV_max, 'CV_hint', 0.5), params, ml_cv_opts);
        ml_cv_time_grid(c, mc) = toc(t_iter);
        ml_cv_iters_grid(c, mc) = ml_cv_opts.max_iter;
        [ml_cv_sumrate_grid(c, mc), ml_cv_pslr_grid(c, mc), ml_cv_status_grid(c, mc)] = ...
            summarize_result(ml_cv_result);
        print_progress('ML-CV', run_count, total_runs, mc, num_mc, CV_max, xi, ...
            ml_cv_time_grid(c, mc), ml_cv_result.status, ml_cv_sumrate_grid(c, mc), ...
            ml_cv_pslr_grid(c, mc), t_global);

        ml_direct_opts = runtime_opts('direct', xi);
        run_count = run_count + 1;
        t_iter = tic;
        ml_direct_result = run_ml_policy_search(H, 'direct', ...
            struct('pslr_min', pslr_min, 'islr_max', islr_max, 'CV_hint', 0.5), ...
            params, ml_direct_opts);
        ml_direct_time_grid(c, mc) = toc(t_iter);
        ml_direct_iters_grid(c, mc) = ml_direct_opts.max_iter;
        [ml_direct_sumrate_grid(c, mc), ml_direct_pslr_grid(c, mc), ml_direct_status_grid(c, mc)] = ...
            summarize_result(ml_direct_result);
        print_progress('ML-Direct', run_count, total_runs, mc, num_mc, CV_max, xi, ...
            ml_direct_time_grid(c, mc), ml_direct_result.status, ml_direct_sumrate_grid(c, mc), ...
            ml_direct_pslr_grid(c, mc), t_global);
    end
end

source_path = fullfile(out_data_dir, 'runtime_coldstart_results.mat');
save(source_path, ...
    'params', 'CV_grid', 'tightness', ...
    'prop_time_grid', 'direct_time_grid', 'ml_cv_time_grid', 'ml_direct_time_grid', ...
    'prop_sumrate_grid', 'direct_sumrate_grid', 'ml_cv_sumrate_grid', 'ml_direct_sumrate_grid', ...
    'prop_pslr_grid', 'direct_pslr_grid', 'ml_cv_pslr_grid', 'ml_direct_pslr_grid', ...
    'prop_status_grid', 'direct_status_grid', 'ml_cv_status_grid', 'ml_direct_status_grid', ...
    'ml_cv_iters_grid', 'ml_direct_iters_grid');

plot_runtime_coldstart_comparison();

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

function [sumrate, pslr_min, status] = summarize_result(result)
sumrate = result.sumrate;
if isempty(result.pslr_per_target)
    pslr_min = NaN;
else
    pslr_min = min(result.pslr_per_target);
end
status = string(result.status);
end

function print_progress(name, run_count, total_runs, mc, num_mc, CV_max, xi, ...
    elapsed, status, sumrate, pslr_min, t_global)

eta = toc(t_global) / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf(['[%5.1f%% %3d/%3d] %-9s MC %d/%d CV=%.1f xi=%.1f | ' ...
         'time=%6.2fs SR=%6.2f PSLR=%6.2f dB | %s | ETA %s\n'], ...
    100*run_count/total_runs, run_count, total_runs, name, mc, num_mc, ...
    CV_max, xi, elapsed, sumrate, 10*log10(pslr_min), status, format_time(eta));
end
