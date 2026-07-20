function run_ml_experiments()
% RUN_ML_EXPERIMENTS  Add ML baselines for CV and direct AF constraints.
%
% Two label-free ML experiments are run on the same channel/CV grid as the
% main simulation:
%   1) ML-CV: CEM policy search with CV/illumination/QoS penalties.
%   2) ML-Direct: CEM policy search with PSLR/ISLR/illumination/QoS penalties.
%
% The script does not overwrite results.mat. It saves separate source data
% and overlay figures under simulation/results and simulation/.

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

results_path = fullfile(sim_dir, 'results.mat');
if exist(results_path, 'file') ~= 2
    error('results.mat not found. Run main.m first so ML baselines share the same reference grid.');
end

S = load(results_path);
params = S.params;
CV_max_list = S.CV_max_list(:).';
num_cv = numel(CV_max_list);
num_mc = min(params.num_mc, size(S.sumrate_grid, 2));
source_path = fullfile(out_data_dir, 'ml_experiment_results.mat');

ml_opts = struct();
ml_opts.num_mc = num_mc;
ml_opts.population = 70;
ml_opts.max_iter = 24;
ml_opts.elite_frac = 0.15;
ml_opts.smoothing = 0.65;
ml_opts.sigma0 = 1.15;
ml_opts.verbose = false;

reuse_existing_ml_results = true;
if reuse_existing_ml_results && exist(source_path, 'file') == 2
    fprintf('Reusing saved ML source data: %s\n', source_path);
    plot_ml_overlays(S, source_path, sim_dir, paper_fig_dir);
    print_ml_summary(source_path);
    return;
end

ml_cv_sumrate_grid = nan(num_cv, num_mc);
ml_cv_pslr_lin_grid = nan(num_cv, num_mc);
ml_cv_islr_lin_grid = nan(num_cv, num_mc);
ml_cv_time_grid = nan(num_cv, num_mc);
ml_cv_feasible_grid = false(num_cv, num_mc);
ml_cv_status_grid = strings(num_cv, num_mc);

ml_direct_sumrate_grid = nan(num_cv, num_mc);
ml_direct_pslr_lin_grid = nan(num_cv, num_mc);
ml_direct_islr_lin_grid = nan(num_cv, num_mc);
ml_direct_time_grid = nan(num_cv, num_mc);
ml_direct_feasible_grid = false(num_cv, num_mc);
ml_direct_status_grid = strings(num_cv, num_mc);
ml_direct_pslr_min_grid = nan(num_cv, num_mc);
ml_direct_islr_max_grid = nan(num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  ML Beamforming Experiments (CEM policy search)\n');
fprintf('============================================================\n');
fprintf('  K=%d, L=%d, NT=%d, N=%d, MC=%d, CV points=%d\n', ...
    params.K, params.L, params.NT, params.N, num_mc, num_cv);
fprintf('  population=%d, iterations=%d\n', ml_opts.population, ml_opts.max_iter);
fprintf('------------------------------------------------------------\n');

t_global = tic;
total_runs = 2 * num_mc * num_cv;
run_count = 0;

for mc = 1:num_mc
    rng(mc, 'twister');
    H = generate_channel(params);

    for c = 1:num_cv
        CV_max = CV_max_list(c);

        run_count = run_count + 1;
        t_iter = tic;
        cv_constraint = struct('CV_max', CV_max, 'CV_hint', CV_max);
        cv_result = run_ml_policy_search(H, 'cv', cv_constraint, params, ml_opts);
        ml_cv_time_grid(c, mc) = toc(t_iter);
        [ml_cv_sumrate_grid(c, mc), ml_cv_pslr_lin_grid(c, mc), ...
            ml_cv_islr_lin_grid(c, mc)] = summarize_ml_result(cv_result);
        ml_cv_feasible_grid(c, mc) = cv_result.feasible;
        ml_cv_status_grid(c, mc) = string(cv_result.status);
        print_progress('ML-CV', run_count, total_runs, mc, num_mc, CV_max, ...
            ml_cv_sumrate_grid(c, mc), ml_cv_pslr_lin_grid(c, mc), ...
            ml_cv_islr_lin_grid(c, mc), cv_result.status, ...
            ml_cv_time_grid(c, mc), t_global);

        run_count = run_count + 1;
        t_iter = tic;
        direct_constraint = get_direct_constraint(S, c, mc, CV_max, params);
        direct_result = run_ml_policy_search(H, 'direct', direct_constraint, params, ml_opts);
        ml_direct_time_grid(c, mc) = toc(t_iter);
        [ml_direct_sumrate_grid(c, mc), ml_direct_pslr_lin_grid(c, mc), ...
            ml_direct_islr_lin_grid(c, mc)] = summarize_ml_result(direct_result);
        ml_direct_feasible_grid(c, mc) = direct_result.feasible;
        ml_direct_status_grid(c, mc) = string(direct_result.status);
        ml_direct_pslr_min_grid(c, mc) = direct_constraint.pslr_min;
        ml_direct_islr_max_grid(c, mc) = direct_constraint.islr_max;
        print_progress('ML-Direct', run_count, total_runs, mc, num_mc, CV_max, ...
            ml_direct_sumrate_grid(c, mc), ml_direct_pslr_lin_grid(c, mc), ...
            ml_direct_islr_lin_grid(c, mc), direct_result.status, ...
            ml_direct_time_grid(c, mc), t_global);
    end
end

save(source_path, ...
    'params', 'CV_max_list', 'ml_opts', ...
    'ml_cv_sumrate_grid', 'ml_cv_pslr_lin_grid', 'ml_cv_islr_lin_grid', ...
    'ml_cv_time_grid', 'ml_cv_feasible_grid', 'ml_cv_status_grid', ...
    'ml_direct_sumrate_grid', 'ml_direct_pslr_lin_grid', 'ml_direct_islr_lin_grid', ...
    'ml_direct_time_grid', 'ml_direct_feasible_grid', 'ml_direct_status_grid', ...
    'ml_direct_pslr_min_grid', 'ml_direct_islr_max_grid');

plot_ml_overlays(S, source_path, sim_dir, paper_fig_dir);
print_ml_summary(source_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved ML source data: %s\n', source_path);
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');
end

function constraint = get_direct_constraint(S, c, mc, CV_max, params)
has_saved_targets = isfield(S, 'direct_pslr_min_grid') && ...
    isfield(S, 'direct_islr_max_grid') && ...
    c <= size(S.direct_pslr_min_grid, 1) && mc <= size(S.direct_pslr_min_grid, 2) && ...
    isfinite(S.direct_pslr_min_grid(c, mc));

if has_saved_targets
    pslr_min = S.direct_pslr_min_grid(c, mc);
    islr_max = S.direct_islr_max_grid(c, mc);
else
    [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);
end

constraint = struct();
constraint.pslr_min = pslr_min;
constraint.islr_max = islr_max;
constraint.CV_hint = CV_max;
end

function [sumrate, pslr_min, islr_max] = summarize_ml_result(result)
sumrate = result.sumrate;
pslr_min = min(result.pslr_per_target);
islr_max = max(result.islr_per_target);
end

function print_progress(name, run_count, total_runs, mc, num_mc, CV_max, ...
    sumrate, pslr_lin, islr_lin, status, elapsed_iter, t_global)

elapsed_total = toc(t_global);
eta = elapsed_total / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf(['[%5.1f%% %3d/%3d] %-9s MC %d/%d CV=%.1f | ' ...
         'SR=%6.2f PSLR=%6.2f dB ISLR=%6.2f dB | %s | %5.1fs ETA %s\n'], ...
    100*run_count/total_runs, run_count, total_runs, name, ...
    mc, num_mc, CV_max, sumrate, 10*log10(pslr_lin), 10*log10(islr_lin), ...
    status, elapsed_iter, format_time(eta));
end

function plot_ml_overlays(S, ml_source_path, sim_dir, paper_fig_dir)
M = load(ml_source_path);

CV_max_list = M.CV_max_list(:);
params = M.params;
num_cv = numel(CV_max_list);

prop_sumrate_avg = row_mean(S.sumrate_grid, num_cv);
prop_pslr_avg_dB = 10*log10(row_mean(S.pslr_lin_grid, num_cv));
prop_islr_avg_dB = 10*log10(row_mean(S.islr_lin_grid, num_cv));
prop_time_avg = row_mean(S.proposed_time_grid, num_cv);

direct_sumrate_avg = row_mean(get_or_nan(S, 'direct_sumrate_grid'), num_cv);
direct_pslr_avg_dB = 10*log10(row_mean(get_or_nan(S, 'direct_pslr_lin_grid'), num_cv));
direct_islr_avg_dB = 10*log10(row_mean(get_or_nan(S, 'direct_islr_lin_grid'), num_cv));
direct_time_avg = row_mean(get_or_nan(S, 'direct_time_grid'), num_cv);

ml_cv_sumrate_avg = row_mean(M.ml_cv_sumrate_grid, num_cv);
ml_cv_pslr_avg_dB = 10*log10(row_mean(M.ml_cv_pslr_lin_grid, num_cv));
ml_cv_islr_avg_dB = 10*log10(row_mean(M.ml_cv_islr_lin_grid, num_cv));
ml_cv_time_avg = row_mean(M.ml_cv_time_grid, num_cv);

ml_direct_sumrate_avg = row_mean(M.ml_direct_sumrate_grid, num_cv);
ml_direct_pslr_avg_dB = 10*log10(row_mean(M.ml_direct_pslr_lin_grid, num_cv));
ml_direct_islr_avg_dB = 10*log10(row_mean(M.ml_direct_islr_lin_grid, num_cv));
ml_direct_time_avg = row_mean(M.ml_direct_time_grid, num_cv);

title_str = sprintf('(K=%d, L=%d, N_T=%d, N=%d)', ...
    params.K, params.L, params.NT, params.N);

% PSLR Pareto overlay
fig = figure('Position', [100 100 850 610], 'Color', 'w');
hold on; grid on; box on;
h1 = plot(prop_sumrate_avg, prop_pslr_avg_dB, '-o', ...
    'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.45 0.80], 'Color', [0.20 0.45 0.80], ...
    'DisplayName', 'CV-SDP');
h2 = plot(direct_sumrate_avg, direct_pslr_avg_dB, '--d', ...
    'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.85 0.25 0.20], 'Color', [0.85 0.25 0.20], ...
    'DisplayName', 'Direct SCA');
h3 = plot(ml_cv_sumrate_avg, ml_cv_pslr_avg_dB, '-.^', ...
    'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], ...
    'DisplayName', 'ML-CV (CEM)');
h4 = plot(ml_direct_sumrate_avg, ml_direct_pslr_avg_dB, ':v', ...
    'LineWidth', 2.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], ...
    'DisplayName', 'ML-Direct (CEM)');
xlabel('Sum-rate (bps/Hz)', 'FontSize', 13);
ylabel('PSLR (dB, worst-case across targets)', 'FontSize', 13);
title(['ML PSLR Pareto Overlay  ' title_str], 'FontSize', 13);
legend([h1 h2 h3 h4], 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 12);
save_figure(fig, sim_dir, paper_fig_dir, 'ml_pareto_curve', 'ML_Pareto_Frontier_Result');

% ISLR Pareto overlay
fig_islr = figure('Position', [120 120 850 610], 'Color', 'w');
hold on; grid on; box on;
h1 = plot(prop_sumrate_avg, prop_islr_avg_dB, '-s', ...
    'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.80 0.45 0.20], 'Color', [0.80 0.45 0.20], ...
    'DisplayName', 'CV-SDP');
h2 = plot(direct_sumrate_avg, direct_islr_avg_dB, '--d', ...
    'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.55 0.40], 'Color', [0.20 0.55 0.40], ...
    'DisplayName', 'Direct SCA');
h3 = plot(ml_cv_sumrate_avg, ml_cv_islr_avg_dB, '-.^', ...
    'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], ...
    'DisplayName', 'ML-CV (CEM)');
h4 = plot(ml_direct_sumrate_avg, ml_direct_islr_avg_dB, ':v', ...
    'LineWidth', 2.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], ...
    'DisplayName', 'ML-Direct (CEM)');
xlabel('Sum-rate (bps/Hz)', 'FontSize', 13);
ylabel('ISLR (dB, worst-case across targets)', 'FontSize', 13);
title(['ML ISLR Pareto Overlay  ' title_str], 'FontSize', 13);
legend([h1 h2 h3 h4], 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 12);
save_figure(fig_islr, sim_dir, paper_fig_dir, 'ml_pareto_curve_islr', 'ML_ISLR_Pareto_Frontier_Result');

% Runtime comparison
fig_time = figure('Position', [140 140 820 560], 'Color', 'w');
hold on; grid on; box on;
plot(CV_max_list, prop_time_avg, '-o', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.45 0.80], 'Color', [0.20 0.45 0.80], ...
    'DisplayName', 'CV-SDP');
plot(CV_max_list, direct_time_avg, '--d', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.85 0.25 0.20], 'Color', [0.85 0.25 0.20], ...
    'DisplayName', 'Direct SCA');
plot(CV_max_list, ml_cv_time_avg, '-.^', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], ...
    'DisplayName', 'ML-CV (CEM)');
plot(CV_max_list, ml_direct_time_avg, ':v', 'LineWidth', 2.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], ...
    'DisplayName', 'ML-Direct (CEM)');
set(gca, 'YScale', 'log', 'FontSize', 12);
xlabel('CV_{max}', 'FontSize', 13, 'Interpreter', 'tex');
ylabel('Runtime per point (s)', 'FontSize', 13);
title(['ML Runtime Comparison  ' title_str], 'FontSize', 13);
legend('Location', 'best', 'FontSize', 11);
save_figure(fig_time, sim_dir, paper_fig_dir, 'ml_runtime_comparison', 'ML_Runtime_Comparison');
end

function avg = row_mean(grid, num_rows)
avg = nan(num_rows, 1);
if isempty(grid)
    return;
end
rows = min(num_rows, size(grid, 1));
avg(1:rows) = mean(grid(1:rows, :), 2, 'omitnan');
end

function value = get_or_nan(S, name)
if isfield(S, name)
    value = S.(name);
else
    value = nan(0, 0);
end
end

function save_figure(fig, sim_dir, paper_fig_dir, stem, paper_stem)
png_path = fullfile(sim_dir, [stem '.png']);
fig_path = fullfile(sim_dir, [stem '.fig']);
pdf_path = fullfile(paper_fig_dir, [paper_stem '.pdf']);
paper_png_path = fullfile(paper_fig_dir, [paper_stem '.png']);

saveas(fig, png_path);
saveas(fig, fig_path);
try
    exportgraphics(fig, pdf_path, 'ContentType', 'vector');
    exportgraphics(fig, paper_png_path, 'Resolution', 300);
catch
    print(fig, pdf_path, '-dpdf', '-r300');
    print(fig, paper_png_path, '-dpng', '-r300');
end
end

function print_ml_summary(source_path)
M = load(source_path);
CV = M.CV_max_list(:);

ml_cv_sr = mean(M.ml_cv_sumrate_grid, 2, 'omitnan');
ml_cv_pslr = 10*log10(mean(M.ml_cv_pslr_lin_grid, 2, 'omitnan'));
ml_cv_islr = 10*log10(mean(M.ml_cv_islr_lin_grid, 2, 'omitnan'));
ml_cv_time = mean(M.ml_cv_time_grid, 2, 'omitnan');
ml_cv_feas = mean(double(M.ml_cv_feasible_grid), 2, 'omitnan');

ml_direct_sr = mean(M.ml_direct_sumrate_grid, 2, 'omitnan');
ml_direct_pslr = 10*log10(mean(M.ml_direct_pslr_lin_grid, 2, 'omitnan'));
ml_direct_islr = 10*log10(mean(M.ml_direct_islr_lin_grid, 2, 'omitnan'));
ml_direct_time = mean(M.ml_direct_time_grid, 2, 'omitnan');
ml_direct_feas = mean(double(M.ml_direct_feasible_grid), 2, 'omitnan');

fprintf('============================================================\n');
fprintf('  ML experiment summary\n');
fprintf('============================================================\n');
fprintf('CVmax | ML-CV SR PSLR ISLR time feas | ML-Direct SR PSLR ISLR time feas\n');
for i = 1:numel(CV)
    fprintf(['%.1f | %6.2f %6.2f %6.2f %6.2f %4.2f | ' ...
             '%6.2f %6.2f %6.2f %6.2f %4.2f\n'], ...
        CV(i), ml_cv_sr(i), ml_cv_pslr(i), ml_cv_islr(i), ml_cv_time(i), ml_cv_feas(i), ...
        ml_direct_sr(i), ml_direct_pslr(i), ml_direct_islr(i), ...
        ml_direct_time(i), ml_direct_feas(i));
end
end
