function run_computational_burden_experiment(num_mc_override, force_rerun, init_mode)
% RUN_COMPUTATIONAL_BURDEN_EXPERIMENT
% Cold-start computational-burden comparison between the proposed CV-SDP
% formulation and the direct PSLR/ISLR SCA baseline.
%
% Metrics:
%   1) wall-clock optimization time;
%   2) number of convex subproblem solves;
%   3) CVX solver iterations accumulated inside those solves.

if nargin < 1 || isempty(num_mc_override)
    num_mc_override = [];
end
if nargin < 2 || isempty(force_rerun)
    force_rerun = false;
end
if nargin < 3 || isempty(init_mode)
    init_mode = 'mrt';
end

clearvars -except num_mc_override force_rerun init_mode; close all; clc;
init_mode = lower(string(init_mode));

sim_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(sim_dir, '..', 'figures');
out_data_dir = fullfile(sim_dir, 'results');
if exist(out_data_dir, 'dir') ~= 7, mkdir(out_data_dir); end
if exist(paper_fig_dir, 'dir') ~= 7, mkdir(paper_fig_dir); end
addpath(genpath(sim_dir));

params = setup_params();
params.NT = 4;
params.N = 16;
params.P_des = 0.8 * params.P_max / params.N;
params.warm_start_cv = false;
params.stop_if_alpha_unchanged = true;
params.sdp_quiet = true;
params.collect_cvx_solver_log = false;
if ~isempty(num_mc_override)
    params.num_mc = num_mc_override;
end
params_profile = params;
params_profile.collect_cvx_solver_log = true;
if init_mode == "mrt50"
    params.mrt_warm_start_weight = 0.5;
    params_profile.mrt_warm_start_weight = 0.5;
elseif init_mode == "mrt25"
    params.mrt_warm_start_weight = 0.25;
    params_profile.mrt_warm_start_weight = 0.25;
elseif init_mode == "mrt15"
    params.mrt_warm_start_weight = 0.15;
    params_profile.mrt_warm_start_weight = 0.15;
elseif init_mode == "mrt10"
    params.mrt_warm_start_weight = 0.10;
    params_profile.mrt_warm_start_weight = 0.10;
elseif init_mode == "mrt05" || init_mode == "mrt05safe"
    params.mrt_warm_start_weight = 0.05;
    params_profile.mrt_warm_start_weight = 0.05;
elseif init_mode == "mrt75"
    params.mrt_warm_start_weight = 0.75;
    params_profile.mrt_warm_start_weight = 0.75;
end

CV_grid = 0:0.1:0.9;
num_cv = numel(CV_grid);
num_mc = params.num_mc;

source_path = fullfile(out_data_dir, sprintf( ...
    'computational_burden_%s_NT%d_N%d_MC%d.mat', char(init_mode), params.NT, params.N, num_mc));

if exist(source_path, 'file') == 2 && ~force_rerun
    fprintf('Loading cached computational-burden result: %s\n', source_path);
    plot_computational_burden_results(source_path);
    print_summary(source_path);
    return;
end

prop_time_grid = nan(num_cv, num_mc);
direct_time_grid = nan(num_cv, num_mc);
prop_iters_grid = nan(num_cv, num_mc);
direct_outer_iters_grid = nan(num_cv, num_mc);
direct_inner_iters_grid = nan(num_cv, num_mc);
prop_cvx_solver_iters_grid = nan(num_cv, num_mc);
direct_cvx_solver_iters_grid = nan(num_cv, num_mc);
prop_sumrate_grid = nan(num_cv, num_mc);
direct_sumrate_grid = nan(num_cv, num_mc);
prop_pslr_grid = nan(num_cv, num_mc);
direct_pslr_grid = nan(num_cv, num_mc);
prop_status_grid = strings(num_cv, num_mc);
direct_status_grid = strings(num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  Computational Burden Experiment: CV-SDP vs Direct SCA\n');
fprintf('============================================================\n');
fprintf('  N_T=%d, N=%d, MC=%d, CV grid=[%s]\n', ...
    params.NT, params.N, num_mc, num2str(CV_grid));
fprintf('  Initialization mode: %s\n', char(init_mode));
fprintf('  No cross-CV warm starts or previous-solution warm starts are used.\n');
fprintf('------------------------------------------------------------\n');

t_global = tic;
total_runs = 2 * num_cv * num_mc;
run_count = 0;

for mc = 1:num_mc
    rng(mc, 'twister');
    H = generate_channel(params);

    for c = 1:num_cv
        CV_max = CV_grid(c);
        [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);
        alpha0 = init_alpha(H, params);
        W0 = build_initial_covariance(H, alpha0, params, init_mode, CV_max);

        run_count = run_count + 1;
        t_iter = tic;
        prop_result = run_proposed(H, CV_max, params, alpha0);
        prop_time_grid(c, mc) = toc(t_iter);
        prop_profile = run_proposed(H, CV_max, params_profile, alpha0);
        prop_iters_grid(c, mc) = prop_result.iters;
        prop_cvx_solver_iters_grid(c, mc) = prop_profile.cvx_solver_iters;
        [prop_sumrate_grid(c, mc), prop_pslr_grid(c, mc), prop_status_grid(c, mc)] = ...
            summarize_result(prop_result);
        print_progress('CV-SDP', run_count, total_runs, mc, num_mc, CV_max, ...
            prop_time_grid(c, mc), prop_iters_grid(c, mc), prop_cvx_solver_iters_grid(c, mc), ...
            prop_sumrate_grid(c, mc), prop_pslr_grid(c, mc), prop_result.status, t_global);

        run_count = run_count + 1;
        t_iter = tic;
        direct_result = run_direct_sca(H, pslr_min, islr_max, params, alpha0, W0);
        direct_time_grid(c, mc) = toc(t_iter);
        direct_profile = run_direct_sca(H, pslr_min, islr_max, params_profile, alpha0, W0);
        direct_outer_iters_grid(c, mc) = direct_result.iters;
        direct_inner_iters_grid(c, mc) = direct_result.inner_iters;
        direct_cvx_solver_iters_grid(c, mc) = direct_profile.cvx_solver_iters;
        [direct_sumrate_grid(c, mc), direct_pslr_grid(c, mc), direct_status_grid(c, mc)] = ...
            summarize_result(direct_result);
        print_progress('DirectSCA', run_count, total_runs, mc, num_mc, CV_max, ...
            direct_time_grid(c, mc), direct_inner_iters_grid(c, mc), direct_cvx_solver_iters_grid(c, mc), ...
            direct_sumrate_grid(c, mc), direct_pslr_grid(c, mc), direct_result.status, t_global);
    end

    save(source_path, 'params', 'CV_grid', 'num_mc', ...
        'init_mode', ...
        'prop_time_grid', 'direct_time_grid', ...
        'prop_iters_grid', 'direct_outer_iters_grid', 'direct_inner_iters_grid', ...
        'prop_cvx_solver_iters_grid', 'direct_cvx_solver_iters_grid', ...
        'prop_sumrate_grid', 'direct_sumrate_grid', ...
        'prop_pslr_grid', 'direct_pslr_grid', ...
        'prop_status_grid', 'direct_status_grid');
end

function W0 = build_initial_covariance(H, alpha0, params, init_mode, CV_max)
switch init_mode
    case "mrt"
        W0 = init_covariance_mrt(H, alpha0, params);
    case {"mrt05", "mrt10", "mrt15", "mrt25", "mrt50", "mrt75"}
        W0 = init_covariance_mrt(H, alpha0, params);
    case "mrt05safe"
        if abs(CV_max) < 1e-12
            W0 = init_covariance_flat(params);
        else
            W0 = init_covariance_mrt(H, alpha0, params);
        end
    case "flat"
        W0 = init_covariance_flat(params);
    otherwise
        error('Unsupported initialization mode: %s', char(init_mode));
end
end

plot_computational_burden_results(source_path);
print_summary(source_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', source_path);
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');
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

function print_progress(name, run_count, total_runs, mc, num_mc, CV_max, ...
    elapsed, solves, solver_iters, sumrate, pslr_min, status, t_global)
eta = toc(t_global) / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf(['[%5.1f%% %3d/%3d] %-9s MC %d/%d CV=%.1f | ' ...
         'time=%6.2fs solves=%4.1f slvItr=%5.1f SR=%6.2f PSLR=%6.2f dB | %s | ETA %s\n'], ...
    100*run_count/total_runs, run_count, total_runs, name, mc, num_mc, CV_max, ...
    elapsed, solves, solver_iters, sumrate, 10*log10(pslr_min), status, format_time(eta));
end

function print_summary(source_path)
S = load(source_path);
prop_success = contains(string(S.prop_status_grid), 'Solved');
direct_success = contains(string(S.direct_status_grid), 'Solved');
prop_time_grid = S.prop_time_grid;
direct_time_grid = S.direct_time_grid;
prop_solves_grid = S.prop_iters_grid;
direct_solves_grid = S.direct_inner_iters_grid;
prop_solver_iters_grid = S.prop_cvx_solver_iters_grid;
direct_solver_iters_grid = S.direct_cvx_solver_iters_grid;
prop_time_grid(~prop_success) = NaN;
direct_time_grid(~direct_success) = NaN;
prop_solves_grid(~prop_success) = NaN;
direct_solves_grid(~direct_success) = NaN;
prop_solver_iters_grid(~prop_success) = NaN;
direct_solver_iters_grid(~direct_success) = NaN;
prop_time = prop_time_grid(:);
direct_time = direct_time_grid(:);
prop_solves = prop_solves_grid(:);
direct_solves = direct_solves_grid(:);
prop_solver_iters = prop_solver_iters_grid(:);
direct_solver_iters = direct_solver_iters_grid(:);
fprintf('Summary over all finite operating points:\n');
fprintf('  CV-SDP:     time %.3fs, convex solves %.2f, solver iters %.2f (%.2f/solve)\n', ...
    mean(prop_time, 'omitnan'), mean(prop_solves, 'omitnan'), ...
    mean(prop_solver_iters, 'omitnan'), mean(prop_solver_iters, 'omitnan') / mean(prop_solves, 'omitnan'));
fprintf('  Direct SCA: time %.3fs, convex solves %.2f, solver iters %.2f (%.2f/solve)\n', ...
    mean(direct_time, 'omitnan'), mean(direct_solves, 'omitnan'), ...
    mean(direct_solver_iters, 'omitnan'), mean(direct_solver_iters, 'omitnan') / mean(direct_solves, 'omitnan'));
fprintf('  Time ratio Direct/CV: %.2fx\n', ...
    mean(direct_time, 'omitnan') / mean(prop_time, 'omitnan'));
fprintf('  Solve ratio Direct/CV: %.2fx\n', ...
    mean(direct_solves, 'omitnan') / mean(prop_solves, 'omitnan'));
fprintf('  Solver-iteration ratio Direct/CV: %.2fx\n', ...
    mean(direct_solver_iters, 'omitnan') / mean(prop_solver_iters, 'omitnan'));
fprintf('  Success rate: CV-SDP %.1f%%, Direct SCA %.1f%%\n', ...
    100 * mean(prop_success(:)), 100 * mean(direct_success(:)));
end

function s = format_time(sec)
if ~isfinite(sec) || sec < 0
    s = 'n/a';
elseif sec < 60
    s = sprintf('%.0fs', sec);
elseif sec < 3600
    s = sprintf('%.1fmin', sec/60);
else
    s = sprintf('%.1fh', sec/3600);
end
end
