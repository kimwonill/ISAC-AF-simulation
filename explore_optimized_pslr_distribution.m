function explore_optimized_pslr_distribution()
% EXPLORE_OPTIMIZED_PSLR_DISTRIBUTION  PSLR distribution after optimization.
%
% This script studies the algorithm-induced PSLR distribution:
%   random channel H -> proposed CV-constrained optimization -> achieved PSLR.
%
% By default, it reuses simulation/results.mat produced by main.m and analyzes
% the optimized worst-case PSLR samples already stored there. Set
% rerun_optimization=true below to regenerate random channels, solve the
% optimization, and additionally save achieved per-target CV values.

clear; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(sim_dir, 'results');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end
addpath(genpath(sim_dir));

% Set true when you want the full random-channel -> optimization experiment.
% This requires CVX and can take a long time for the paper-scale parameters.
rerun_optimization = true;
num_mc_override = 30;

params = setup_params();
CV_list = params.CV_max_list(:).';
num_cv = numel(CV_list);
num_mc = num_mc_override;

if rerun_optimization
    result_data = run_optimized_distribution(params, CV_list, num_mc);
else
    result_data = load_main_results(sim_dir);
end

stats = summarize_optimized_pslr(result_data);
analytic_ref = load_analytic_reference(out_dir);
plot_optimized_distribution(stats, result_data, out_dir, analytic_ref);
print_summary(stats);
print_analytic_comparison(stats, analytic_ref);

save(fullfile(out_dir, 'optimized_pslr_distribution_source_data.mat'), ...
    'params', 'CV_list', 'num_mc', 'result_data', 'stats', ...
    'analytic_ref', 'rerun_optimization');

fprintf('Saved: %s\n', fullfile(out_dir, 'optimized_pslr_distribution.png'));
end

function result_data = load_main_results(sim_dir)
results_path = fullfile(sim_dir, 'results.mat');
if exist(results_path, 'file') ~= 2
    error(['results.mat not found. Run main.m first, or set ' ...
           'rerun_optimization=true in this script.']);
end

S = load(results_path, 'CV_max_list', 'sumrate_grid', ...
    'pslr_lin_grid', 'islr_lin_grid', 'params');

result_data.CV_list = S.CV_max_list(:).';
result_data.params = S.params;
result_data.sumrate_grid = S.sumrate_grid;
result_data.pslr_worst_grid = S.pslr_lin_grid;
result_data.islr_worst_grid = S.islr_lin_grid;
result_data.achieved_cv_worst_grid = nan(size(S.pslr_lin_grid));
result_data.pslr_per_target = [];
result_data.achieved_cv_per_target = [];
result_data.status_grid = strings(size(S.pslr_lin_grid));
result_data.source = 'results.mat from main.m';
end

function result_data = run_optimized_distribution(params, CV_list, num_mc)
if exist('cvx_begin', 'file') ~= 2
    error('CVX is required for rerun_optimization=true.');
end

num_cv = numel(CV_list);
sumrate_grid = nan(num_cv, num_mc);
pslr_worst_grid = nan(num_cv, num_mc);
islr_worst_grid = nan(num_cv, num_mc);
achieved_cv_worst_grid = nan(num_cv, num_mc);
pslr_per_target = nan(num_cv, num_mc, params.L);
islr_per_target = nan(num_cv, num_mc, params.L);
achieved_cv_per_target = nan(num_cv, num_mc, params.L);
status_grid = strings(num_cv, num_mc);
channel_seed = nan(1, num_mc);

A = compute_steering(params);
t_global = tic;
attempt_count = 0;
success_count = 0;
max_attempts = max(3*num_mc, num_mc + 20);

fprintf('============================================================\n');
fprintf('  Optimized PSLR Distribution Experiment\n');
fprintf('============================================================\n');
fprintf('  K=%d, L=%d, NT=%d, N=%d, target successful MC=%d\n', ...
    params.K, params.L, params.NT, params.N, num_mc);
fprintf('------------------------------------------------------------\n');

while success_count < num_mc && attempt_count < max_attempts
    attempt_count = attempt_count + 1;
    rng(attempt_count, 'twister');
    H = generate_channel(params);
    alpha_warm = [];
    accept_channel = true;
    reject_status = "";

    sumrate_col = nan(num_cv, 1);
    pslr_worst_col = nan(num_cv, 1);
    islr_worst_col = nan(num_cv, 1);
    achieved_cv_worst_col = nan(num_cv, 1);
    pslr_per_target_col = nan(num_cv, params.L);
    islr_per_target_col = nan(num_cv, params.L);
    achieved_cv_per_target_col = nan(num_cv, params.L);
    status_col = strings(num_cv, 1);

    for c = 1:num_cv
        CV_max = CV_list(c);
        t_iter = tic;

        if isfield(params, 'warm_start_cv') && params.warm_start_cv
            result = run_proposed(H, CV_max, params, alpha_warm);
        else
            result = run_proposed(H, CV_max, params);
        end

        status_col(c) = string(result.status);
        if ~isnan(result.sumrate)
            alpha_warm = result.alpha;
            sumrate_col(c) = result.sumrate;
            pslr_per_target_col(c, :) = result.pslr_per_target;
            islr_per_target_col(c, :) = result.islr_per_target;
            pslr_worst_col(c) = min(result.pslr_per_target);
            islr_worst_col(c) = max(result.islr_per_target);

            cv_targets = nan(params.L, 1);
            for l = 1:params.L
                P = compute_directional_power(result.W, A(:, l));
                cv_targets(l) = directional_power_cv(P);
            end
            achieved_cv_per_target_col(c, :) = cv_targets;
            achieved_cv_worst_col(c) = max(cv_targets);
        else
            alpha_warm = [];
            accept_channel = false;
            reject_status = string(result.status);
        end

        elapsed = toc(t_global);
        accepted_iters = success_count*num_cv + c;
        target_iters = num_mc*num_cv;
        eta = elapsed / max(accepted_iters, 1) * ...
            max(target_iters - accepted_iters, 0);
        fprintf(['[valid %2d/%2d attempt %2d CV=%.1f] ' ...
                 'SR=%6.2f PSLR=%6.2f dB CV_ach=%5.3f | %s | %5.1fs ETA %s\n'], ...
            success_count, num_mc, attempt_count, CV_max, sumrate_col(c), ...
            10*log10(pslr_worst_col(c)), achieved_cv_worst_col(c), ...
            result.status, toc(t_iter), format_time(eta));

        if ~accept_channel
            break;
        end
    end

    if accept_channel
        success_count = success_count + 1;
        sumrate_grid(:, success_count) = sumrate_col;
        pslr_worst_grid(:, success_count) = pslr_worst_col;
        islr_worst_grid(:, success_count) = islr_worst_col;
        achieved_cv_worst_grid(:, success_count) = achieved_cv_worst_col;
        pslr_per_target(:, success_count, :) = pslr_per_target_col;
        islr_per_target(:, success_count, :) = islr_per_target_col;
        achieved_cv_per_target(:, success_count, :) = achieved_cv_per_target_col;
        status_grid(:, success_count) = status_col;
        channel_seed(success_count) = attempt_count;

        fprintf('  accepted seed %d as sample %d/%d\n', ...
            attempt_count, success_count, num_mc);
    else
        fprintf('  skipped seed %d: %s\n', attempt_count, reject_status);
    end
end

if success_count < num_mc
    warning('Only %d successful samples were collected after %d attempts.', ...
        success_count, attempt_count);

    sumrate_grid = sumrate_grid(:, 1:success_count);
    pslr_worst_grid = pslr_worst_grid(:, 1:success_count);
    islr_worst_grid = islr_worst_grid(:, 1:success_count);
    achieved_cv_worst_grid = achieved_cv_worst_grid(:, 1:success_count);
    pslr_per_target = pslr_per_target(:, 1:success_count, :);
    islr_per_target = islr_per_target(:, 1:success_count, :);
    achieved_cv_per_target = achieved_cv_per_target(:, 1:success_count, :);
    status_grid = status_grid(:, 1:success_count);
    channel_seed = channel_seed(1:success_count);
end

result_data.CV_list = CV_list;
result_data.params = params;
result_data.sumrate_grid = sumrate_grid;
result_data.pslr_worst_grid = pslr_worst_grid;
result_data.islr_worst_grid = islr_worst_grid;
result_data.achieved_cv_worst_grid = achieved_cv_worst_grid;
result_data.pslr_per_target = pslr_per_target;
result_data.islr_per_target = islr_per_target;
result_data.achieved_cv_per_target = achieved_cv_per_target;
result_data.status_grid = status_grid;
result_data.source = 'rerun random channels and run_proposed';
result_data.channel_seed = channel_seed;
result_data.num_channel_attempts = attempt_count;
end

function cv = directional_power_cv(P)
P = P(:);
mu = mean(P);
sigma = sqrt(mean((P - mu).^2));
cv = sigma / max(mu, eps);
end

function stats = summarize_optimized_pslr(result_data)
P = result_data.pslr_worst_grid;
stats.CV_list = result_data.CV_list(:).';
stats.num_samples = sum(isfinite(P), 2);
stats.mean_pslr = mean(P, 2, 'omitnan');
stats.var_pslr = var(P, 1, 2, 'omitnan');
stats.std_pslr = sqrt(stats.var_pslr);
stats.median_pslr = median(P, 2, 'omitnan');
stats.q05_pslr = nan(size(stats.mean_pslr));
stats.q95_pslr = nan(size(stats.mean_pslr));
for i = 1:size(P, 1)
    vals = P(i, isfinite(P(i, :)));
    if ~isempty(vals)
        q = quantile(vals, [0.05 0.95]);
        stats.q05_pslr(i) = q(1);
        stats.q95_pslr(i) = q(2);
    end
end
stats.mean_sumrate = mean(result_data.sumrate_grid, 2, 'omitnan');

if isfield(result_data, 'achieved_cv_worst_grid')
    C = result_data.achieved_cv_worst_grid;
    stats.mean_achieved_cv = mean(C, 2, 'omitnan');
    stats.var_achieved_cv = var(C, 1, 2, 'omitnan');
else
    stats.mean_achieved_cv = nan(size(stats.CV_list(:)));
    stats.var_achieved_cv = nan(size(stats.CV_list(:)));
end
end

function analytic_ref = load_analytic_reference(out_dir)
analytic_ref = struct('has_data', false, 'CV_list', [], ...
    'mean_pslr', [], 'var_pslr', []);
path = fullfile(out_dir, 'pslr_cv_distribution_source_data.mat');
if exist(path, 'file') ~= 2
    return;
end
S = load(path, 'CV_list', 'theory_mean_pslr', 'theory_var_pslr');
analytic_ref.has_data = true;
analytic_ref.CV_list = S.CV_list(:).';
analytic_ref.mean_pslr = S.theory_mean_pslr(:).';
analytic_ref.var_pslr = S.theory_var_pslr(:).';
end

function plot_optimized_distribution(stats, result_data, out_dir, analytic_ref)
CV_list = stats.CV_list;
P = result_data.pslr_worst_grid;
[cv_grid, ~] = ndgrid(CV_list(:), 1:size(P, 2));

fig = figure('Color', 'w', 'Position', [100 100 820 560]);
hold on; grid on; box on;

fill([CV_list, fliplr(CV_list)], ...
     [10*log10(stats.q95_pslr(:).'), fliplr(10*log10(stats.q05_pslr(:).'))], ...
     [0.70 0.78 0.62], 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
     'DisplayName', 'Optimized 5--95%');
scatter(cv_grid(:) + 0.006*randn(numel(cv_grid), 1), ...
    10*log10(P(:)), 38, ...
    'MarkerFaceColor', [0.10 0.10 0.10], ...
    'MarkerEdgeColor', 'w', 'LineWidth', 0.5, ...
    'DisplayName', 'Optimized samples');
plot(CV_list, 10*log10(stats.mean_pslr), '-o', ...
    'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.45 0.65], ...
    'Color', [0.20 0.45 0.65], ...
    'DisplayName', 'Optimized mean');
plot(CV_list, 10*log10(stats.median_pslr), '--s', ...
    'LineWidth', 1.8, 'MarkerSize', 7, ...
    'MarkerFaceColor', [0.32 0.50 0.48], ...
    'Color', [0.32 0.50 0.48], ...
    'DisplayName', 'Optimized median');
if analytic_ref.has_data
    [common_cv, i_opt, i_ana] = intersect(round(CV_list, 12), ...
        round(analytic_ref.CV_list, 12), 'stable'); %#ok<ASGLU>
    plot(CV_list(i_opt), 10*log10(analytic_ref.mean_pslr(i_ana)), '-x', ...
        'LineWidth', 1.9, 'MarkerSize', 8, ...
        'Color', [0.48 0.20 0.55], ...
        'DisplayName', 'Fixed-CV analytic mean');
end

xlabel('CV_{max}', 'FontSize', 15, 'Interpreter', 'tex');
ylabel('Worst-case PSLR after optimization (dB)', 'FontSize', 15);
title('Algorithm-Induced PSLR Distribution', 'FontSize', 15);
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 13);
xlim([min(CV_list)-0.03, max(CV_list)+0.03]);

exportgraphics(fig, fullfile(out_dir, 'optimized_pslr_distribution.png'), ...
    'Resolution', 300);
exportgraphics(fig, fullfile(out_dir, 'optimized_pslr_distribution.pdf'), ...
    'ContentType', 'vector');
saveas(fig, fullfile(out_dir, 'optimized_pslr_distribution.fig'));
end

function print_summary(stats)
fprintf('============================================================\n');
fprintf('  Optimized PSLR summary (%d CV points)\n', numel(stats.CV_list));
fprintf('============================================================\n');
fprintf('CVmax  samples  mean(dB)  var(linear)  std~dB  q05(dB)  q95(dB)  meanSR  meanAchCV\n');
for i = 1:numel(stats.CV_list)
    std_db_approx = 10/log(10) * stats.std_pslr(i) / stats.mean_pslr(i);
    fprintf('%.1f   %4d    %7.2f   %.4e    %6.3f  %7.2f  %7.2f  %6.2f  %7.3f\n', ...
        stats.CV_list(i), stats.num_samples(i), ...
        10*log10(stats.mean_pslr(i)), stats.var_pslr(i), std_db_approx, ...
        10*log10(stats.q05_pslr(i)), 10*log10(stats.q95_pslr(i)), ...
        stats.mean_sumrate(i), stats.mean_achieved_cv(i));
end
end

function print_analytic_comparison(stats, analytic_ref)
if ~analytic_ref.has_data
    fprintf('No analytic fixed-CV reference found.\n');
    return;
end

fprintf('============================================================\n');
fprintf('  Optimized simulation vs. fixed-CV analytic reference\n');
fprintf('============================================================\n');
fprintf('CVmax  analyticMean  optimizedMean  diff(dB)  analyticVar  optimizedVar  varRatio\n');
for i = 1:numel(stats.CV_list)
    cv = stats.CV_list(i);
    j = find(abs(analytic_ref.CV_list - cv) < 1e-12, 1);
    if isempty(j)
        continue;
    end
    analytic_mean_dB = 10*log10(analytic_ref.mean_pslr(j));
    optimized_mean_dB = 10*log10(stats.mean_pslr(i));
    analytic_var = analytic_ref.var_pslr(j);
    optimized_var = stats.var_pslr(i);
    var_ratio = optimized_var / max(analytic_var, eps);
    fprintf('%.1f    %8.2f      %8.2f    %+7.2f   %.4e   %.4e   %7.3f\n', ...
        cv, analytic_mean_dB, optimized_mean_dB, ...
        optimized_mean_dB - analytic_mean_dB, ...
        analytic_var, optimized_var, var_ratio);
end
end
