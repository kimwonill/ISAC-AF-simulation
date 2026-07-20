function plot_pareto_results()
% PLOT_PARETO_RESULTS  Rebuild paper Pareto figures with benchmark schemes.

out_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(out_dir, '..', 'figures');
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end

load(fullfile(out_dir, 'results.mat'));

if ~exist('comm_sumrate_grid', 'var') || isempty(comm_sumrate_grid)
    [comm_sumrate_grid, comm_pslr_lin_grid, comm_islr_lin_grid] = run_comm_only(params);
    save(fullfile(out_dir, 'results.mat'), 'comm_sumrate_grid', ...
         'comm_pslr_lin_grid', 'comm_islr_lin_grid', '-append');
end

recompute_surrogate_baselines = false;
crb_eta_target = [0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 ...
                  4.5e-3 6.7e-3 ...
                  1e-2 1.32e-2 1.73e-2 2.28e-2 ...
                  3e-2 3.98e-2 5.28e-2 6.95e-2 ...
                  0.1 0.144 0.208 0.3 1 3 10 30 100];
mi_eta_target  = [0.1 0.3 1 3 10 30 100];

needs_crb = recompute_surrogate_baselines || ~exist('crb_sumrate_grid', 'var') || ...
            ~exist('crb_eta_list', 'var') || ~same_eta_grid(crb_eta_list, crb_eta_target);
needs_mi = recompute_surrogate_baselines || ~exist('mi_sumrate_grid', 'var') || ...
           ~exist('mi_eta_list', 'var') || ~same_eta_grid(mi_eta_list, mi_eta_target);

if needs_crb
    crb_eta_list = crb_eta_target;
    [crb_sumrate_grid, crb_pslr_lin_grid, crb_islr_lin_grid] = ...
        run_surrogate_sweep('crb', crb_eta_list, params);
    save(fullfile(out_dir, 'results.mat'), 'crb_eta_list', ...
         'crb_sumrate_grid', 'crb_pslr_lin_grid', 'crb_islr_lin_grid', '-append');
end

if needs_mi
    mi_eta_list = mi_eta_target;
    [mi_sumrate_grid, mi_pslr_lin_grid, mi_islr_lin_grid] = ...
        run_surrogate_sweep('mi', mi_eta_list, params);
    save(fullfile(out_dir, 'results.mat'), 'mi_eta_list', ...
         'mi_sumrate_grid', 'mi_pslr_lin_grid', 'mi_islr_lin_grid', '-append');
end

sumrate_avg = mean(sumrate_grid, 2, 'omitnan');
pslr_avg_dB = 10*log10(mean(pslr_lin_grid, 2, 'omitnan'));
islr_avg_dB = 10*log10(mean(islr_lin_grid, 2, 'omitnan'));

direct_sumrate_avg = mean(direct_sumrate_grid, 2, 'omitnan');
direct_pslr_avg_dB = 10*log10(mean(direct_pslr_lin_grid, 2, 'omitnan'));
direct_islr_avg_dB = 10*log10(mean(direct_islr_lin_grid, 2, 'omitnan'));

direct_equiv_sumrate_avg = mean(direct_equiv_sumrate_grid, 2, 'omitnan');
direct_equiv_pslr_avg_dB = 10*log10(mean(direct_equiv_pslr_lin_grid, 2, 'omitnan'));
direct_equiv_islr_avg_dB = 10*log10(mean(direct_equiv_islr_lin_grid, 2, 'omitnan'));

comm_sumrate = mean(comm_sumrate_grid, 'omitnan');
comm_pslr_dB = 10*log10(mean(comm_pslr_lin_grid, 'omitnan'));
comm_islr_dB = 10*log10(mean(comm_islr_lin_grid, 'omitnan'));

crb_sumrate_avg = mean(crb_sumrate_grid, 2, 'omitnan');
crb_pslr_avg_dB = 10*log10(mean(crb_pslr_lin_grid, 2, 'omitnan'));
crb_islr_avg_dB = 10*log10(mean(crb_islr_lin_grid, 2, 'omitnan'));

mi_sumrate_avg = mean(mi_sumrate_grid, 2, 'omitnan');
mi_pslr_avg_dB = 10*log10(mean(mi_pslr_lin_grid, 2, 'omitnan'));
mi_islr_avg_dB = 10*log10(mean(mi_islr_lin_grid, 2, 'omitnan'));

e2e_source_path = fullfile(out_dir, 'results', 'ml_end_to_end_learning_results.mat');
e2e_available = exist(e2e_source_path, 'file') == 2;
e2e_cv_sumrate_avg = [];
e2e_cv_pslr_avg_dB = [];
e2e_cv_islr_avg_dB = [];
e2e_direct_sumrate_avg = [];
e2e_direct_pslr_avg_dB = [];
e2e_direct_islr_avg_dB = [];
if e2e_available
    E2E = load(e2e_source_path);
    e2e_cv_sumrate_avg = mean(E2E.e2e_cv_sumrate_grid, 2, 'omitnan');
    e2e_cv_pslr_avg_dB = 10*log10(mean(E2E.e2e_cv_pslr_lin_grid, 2, 'omitnan'));
    e2e_cv_islr_avg_dB = 10*log10(mean(E2E.e2e_cv_islr_lin_grid, 2, 'omitnan'));
    e2e_direct_sumrate_avg = mean(E2E.e2e_direct_sumrate_grid, 2, 'omitnan');
    e2e_direct_pslr_avg_dB = 10*log10(mean(E2E.e2e_direct_pslr_lin_grid, 2, 'omitnan'));
    e2e_direct_islr_avg_dB = 10*log10(mean(E2E.e2e_direct_islr_lin_grid, 2, 'omitnan'));
    fprintf('Loaded end-to-end NN Pareto data: %s\n', e2e_source_path);
end

pslr_max_dB = 10*log10(1 + params.N/(params.kappa - 1));
islr_min_dB = 10*log10((params.N - 1)*(params.N + 2*params.kappa - 2) / ...
                       (2*(params.N + params.kappa - 1)));

title_str = sprintf('(K=%d, L=%d, N_T=%d, N=%d)', ...
                    params.K, params.L, params.NT, params.N);
label_idx = 1:2:numel(CV_max_list);
plot_style = struct( ...
    'figure_position', [100 100 760 520], ...
    'axes_position', [0.10 0.12 0.865 0.80], ...
    'axes_font', 15, ...
    'label_font', 19, ...
    'title_font', 22, ...
    'legend_font', 16.5, ...
    'cv_label_font', 10.5);

source_data_dir = fullfile(out_dir, 'results');
if exist(source_data_dir, 'dir') ~= 7
    mkdir(source_data_dir);
end
source_data_path = fullfile(source_data_dir, 'fig4_fig5_source_data.mat');
save(source_data_path, ...
    'CV_max_list', 'params', ...
    'sumrate_grid', 'pslr_lin_grid', 'islr_lin_grid', ...
    'direct_sumrate_grid', 'direct_pslr_lin_grid', 'direct_islr_lin_grid', ...
    'direct_equiv_sumrate_grid', 'direct_equiv_pslr_lin_grid', 'direct_equiv_islr_lin_grid', ...
    'comm_sumrate_grid', 'comm_pslr_lin_grid', 'comm_islr_lin_grid', ...
    'crb_eta_list', 'crb_sumrate_grid', 'crb_pslr_lin_grid', 'crb_islr_lin_grid', ...
    'mi_eta_list', 'mi_sumrate_grid', 'mi_pslr_lin_grid', 'mi_islr_lin_grid', ...
    'sumrate_avg', 'pslr_avg_dB', 'islr_avg_dB', ...
    'direct_sumrate_avg', 'direct_pslr_avg_dB', 'direct_islr_avg_dB', ...
    'direct_equiv_sumrate_avg', 'direct_equiv_pslr_avg_dB', 'direct_equiv_islr_avg_dB', ...
    'comm_sumrate', 'comm_pslr_dB', 'comm_islr_dB', ...
    'crb_sumrate_avg', 'crb_pslr_avg_dB', 'crb_islr_avg_dB', ...
    'mi_sumrate_avg', 'mi_pslr_avg_dB', 'mi_islr_avg_dB', ...
    'e2e_available', 'e2e_source_path', ...
    'e2e_cv_sumrate_avg', 'e2e_cv_pslr_avg_dB', 'e2e_cv_islr_avg_dB', ...
    'e2e_direct_sumrate_avg', 'e2e_direct_pslr_avg_dB', 'e2e_direct_islr_avg_dB', ...
    'pslr_max_dB', 'islr_min_dB');
copyfile(fullfile(out_dir, 'results.mat'), ...
    fullfile(source_data_dir, 'fig4_fig5_full_results.mat'), 'f');
fprintf('Saved Fig. 4/5 source data: %s\n', source_data_path);

% ---------------- PSLR plot ----------------
fig = figure('Position', plot_style.figure_position, 'Color', 'w');
hold on; grid on; box on;
valid = ~isnan(sumrate_avg) & ~isnan(pslr_avg_dB);
valid_direct = ~isnan(direct_sumrate_avg) & ~isnan(direct_pslr_avg_dB);
valid_equiv = ~isnan(direct_equiv_sumrate_avg) & ~isnan(direct_equiv_pslr_avg_dB);
valid_crb = ~isnan(crb_sumrate_avg) & ~isnan(crb_pslr_avg_dB);
valid_mi = ~isnan(mi_sumrate_avg) & ~isnan(mi_pslr_avg_dB);
valid_e2e_cv = ~isnan(e2e_cv_sumrate_avg) & ~isnan(e2e_cv_pslr_avg_dB);
valid_e2e_direct = ~isnan(e2e_direct_sumrate_avg) & ~isnan(e2e_direct_pslr_avg_dB);

x_front = [direct_equiv_sumrate_avg(valid_equiv); direct_sumrate_avg(valid_direct); comm_sumrate];
y_front = [direct_equiv_pslr_avg_dB(valid_equiv); direct_pslr_avg_dB(valid_direct); comm_pslr_dB];
[x_front, y_front] = upper_envelope(x_front, y_front);

x_all = [sumrate_avg(valid); direct_sumrate_avg(valid_direct); direct_equiv_sumrate_avg(valid_equiv); ...
         crb_sumrate_avg(valid_crb); mi_sumrate_avg(valid_mi); ...
         e2e_cv_sumrate_avg(valid_e2e_cv); e2e_direct_sumrate_avg(valid_e2e_direct); ...
         comm_sumrate];
y_all = [pslr_avg_dB(valid); direct_pslr_avg_dB(valid_direct); direct_equiv_pslr_avg_dB(valid_equiv); ...
         crb_pslr_avg_dB(valid_crb); mi_pslr_avg_dB(valid_mi); ...
         e2e_cv_pslr_avg_dB(valid_e2e_cv); e2e_direct_pslr_avg_dB(valid_e2e_direct); ...
         comm_pslr_dB; pslr_max_dB];
[xlim_vals, ylim_vals] = padded_limits(x_all, y_all);
ylim_vals = [0, ylim_vals(2)];
xlim(xlim_vals); ylim(ylim_vals);
[x_shade, y_shade] = smooth_boundary(x_front, y_front, xlim_vals, 'left');
fill([x_shade; flipud(x_shade)], [y_shade; ylim_vals(1)*ones(size(y_shade))], ...
     [0.90 0.94 1.00], 'FaceAlpha', 1.0, 'EdgeColor', 'none', ...
     'DisplayName', 'Dominated/achievable region');

h_prop = plot(sumrate_avg(valid), pslr_avg_dB(valid), '-o', 'LineWidth', 2.0, ...
    'MarkerSize', 8, 'MarkerFaceColor', [0.20 0.45 0.80], 'Color', [0.20 0.45 0.80], ...
    'DisplayName', 'Proposed CV');
h_direct = plot(direct_sumrate_avg(valid_direct), direct_pslr_avg_dB(valid_direct), '--d', ...
    'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.25 0.20], ...
    'Color', [0.85 0.25 0.20], 'DisplayName', 'Direct PSLR/ISLR SCA');
h_equiv = plot(direct_equiv_sumrate_avg(valid_equiv), direct_equiv_pslr_avg_dB(valid_equiv), 'o', ...
    'LineStyle', 'none', 'LineWidth', 1.5, 'MarkerSize', 12, 'MarkerFaceColor', 'none', ...
    'Color', [0.10 0.10 0.10], 'DisplayName', 'Direct, ISLR-active');
h_crb = plot(crb_sumrate_avg(valid_crb), crb_pslr_avg_dB(valid_crb), '-.^', ...
    'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', [0.45 0.70 0.30], ...
    'Color', [0.25 0.55 0.20], 'DisplayName', 'CRB-inspired');
h_mi = plot(mi_sumrate_avg(valid_mi), mi_pslr_avg_dB(valid_mi), ':v', ...
    'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', [0.55 0.35 0.75], ...
    'Color', [0.45 0.25 0.65], 'DisplayName', 'MI-inspired');
add_cluster_box(mi_sumrate_avg(valid_mi), mi_pslr_avg_dB(valid_mi), ...
    [0.45 0.25 0.65], xlim_vals, ylim_vals, 0.80, 1.30);
h_e2e_cv = [];
h_e2e_direct = [];
if e2e_available
    h_e2e_cv = plot(e2e_cv_sumrate_avg(valid_e2e_cv), e2e_cv_pslr_avg_dB(valid_e2e_cv), '-x', ...
        'LineWidth', 2.0, 'MarkerSize', 8, 'Color', [0.00 0.45 0.55], ...
        'DisplayName', 'E2E NN-CV');
    h_e2e_direct = plot(e2e_direct_sumrate_avg(valid_e2e_direct), ...
        e2e_direct_pslr_avg_dB(valid_e2e_direct), '--x', ...
        'LineWidth', 2.0, 'MarkerSize', 8, 'Color', [0.55 0.20 0.55], ...
        'DisplayName', 'E2E NN-Direct');
end
h_comm = plot(comm_sumrate, comm_pslr_dB, 'kp', 'LineWidth', 1.6, 'MarkerSize', 13, ...
    'MarkerFaceColor', [0.95 0.75 0.10], 'DisplayName', 'Communication-only');
h_comm_rate = xline(comm_sumrate, '-.', 'LineWidth', 1.6, 'Color', [0 0 0], ...
    'DisplayName', 'Comm.-only rate');
h_global = yline(pslr_max_dB, '--', 'LineWidth', 1.7, 'Color', [0.10 0.10 0.10], ...
    'DisplayName', 'Global max PSLR');
for ii = label_idx
    if valid(ii)
        text(sumrate_avg(ii), pslr_avg_dB(ii) + 0.18, sprintf('%.1f', CV_max_list(ii)), ...
            'FontSize', plot_style.cv_label_font, 'Color', [0.45 0.10 0.10]);
    end
end
ax = gca;
set(ax, 'FontSize', plot_style.axes_font, 'Layer', 'bottom', ...
    'Units', 'normalized', 'Position', plot_style.axes_position);
xlabel(ax, 'Sum-rate (bps/Hz)', 'FontSize', plot_style.label_font);
ylabel(ax, 'PSLR (dB, worst-case across targets)', 'FontSize', plot_style.label_font);
title_handle = title(ax, ['PSLR Pareto  ' title_str], 'FontSize', plot_style.title_font);
legend_handles = [h_prop h_direct h_equiv h_crb h_mi];
if e2e_available
    legend_handles = [legend_handles h_e2e_cv h_e2e_direct];
end
legend_handles = [legend_handles h_comm h_comm_rate h_global];
legend(ax, legend_handles, ...
    'Location', 'southwest', 'NumColumns', 2, ...
    'FontSize', min(plot_style.legend_font, 14.5));
set(title_handle, 'FontSize', plot_style.title_font);
fig = ancestor(ax, 'figure');
drawnow;
pslr_png = fullfile(out_dir, 'pareto_curve.png');
safe_save(fig, pslr_png);
safe_save(fig, fullfile(out_dir, 'pareto_curve.fig'));
safe_export(fig, fullfile(paper_fig_dir, 'Pareto_Frontier_Result.pdf'), 'pdf');
copyfile(pslr_png, fullfile(paper_fig_dir, 'Pareto_Frontier_Result.png'), 'f');

% ---------------- ISLR plot ----------------
fig_islr = figure('Position', plot_style.figure_position, 'Color', 'w');
hold on; grid on; box on;
valid_crb_i = ~isnan(crb_sumrate_avg) & ~isnan(crb_islr_avg_dB);
valid_mi_i = ~isnan(mi_sumrate_avg) & ~isnan(mi_islr_avg_dB);
valid_e2e_cv_i = ~isnan(e2e_cv_sumrate_avg) & ~isnan(e2e_cv_islr_avg_dB);
valid_e2e_direct_i = ~isnan(e2e_direct_sumrate_avg) & ~isnan(e2e_direct_islr_avg_dB);

x_front_i = [direct_equiv_sumrate_avg(valid_equiv); direct_sumrate_avg(valid_direct); comm_sumrate];
y_front_i = [direct_equiv_islr_avg_dB(valid_equiv); direct_islr_avg_dB(valid_direct); comm_islr_dB];
[x_front_i, y_front_i] = lower_envelope(x_front_i, y_front_i);

x_all_i = [sumrate_avg(valid); direct_sumrate_avg(valid_direct); direct_equiv_sumrate_avg(valid_equiv); ...
           crb_sumrate_avg(valid_crb_i); mi_sumrate_avg(valid_mi_i); ...
           e2e_cv_sumrate_avg(valid_e2e_cv_i); e2e_direct_sumrate_avg(valid_e2e_direct_i); ...
           comm_sumrate];
y_all_i = [islr_avg_dB(valid); direct_islr_avg_dB(valid_direct); direct_equiv_islr_avg_dB(valid_equiv); ...
           crb_islr_avg_dB(valid_crb_i); mi_islr_avg_dB(valid_mi_i); ...
           e2e_cv_islr_avg_dB(valid_e2e_cv_i); e2e_direct_islr_avg_dB(valid_e2e_direct_i); ...
           comm_islr_dB; islr_min_dB];
[xlim_i, ylim_i] = padded_limits(x_all_i, y_all_i);
ylim_i = [ylim_i(1), 9.4];
xlim(xlim_i); ylim(ylim_i);
[x_shade_i, y_shade_i] = smooth_boundary(x_front_i, y_front_i, xlim_i, 'left');
fill([x_shade_i; flipud(x_shade_i)], [y_shade_i; ylim_i(2)*ones(size(y_shade_i))], ...
     [1.00 0.94 0.90], 'FaceAlpha', 1.0, 'EdgeColor', 'none', ...
     'DisplayName', 'Dominated/achievable region');

h_prop_i = plot(sumrate_avg(valid), islr_avg_dB(valid), '-s', 'LineWidth', 2.0, ...
    'MarkerSize', 8, 'MarkerFaceColor', [0.80 0.45 0.20], 'Color', [0.80 0.45 0.20], ...
    'DisplayName', 'Proposed CV');
h_direct_i = plot(direct_sumrate_avg(valid_direct), direct_islr_avg_dB(valid_direct), '--d', ...
    'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.20 0.55 0.40], ...
    'Color', [0.20 0.55 0.40], 'DisplayName', 'Direct PSLR/ISLR SCA');
h_equiv_i = plot(direct_equiv_sumrate_avg(valid_equiv), direct_equiv_islr_avg_dB(valid_equiv), 'o', ...
    'LineStyle', 'none', 'LineWidth', 1.5, 'MarkerSize', 12, 'MarkerFaceColor', 'none', ...
    'Color', [0.10 0.10 0.10], 'DisplayName', 'Direct, ISLR-active');
h_crb_i = plot(crb_sumrate_avg(valid_crb_i), crb_islr_avg_dB(valid_crb_i), '-.^', ...
    'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', [0.45 0.70 0.30], ...
    'Color', [0.25 0.55 0.20], 'DisplayName', 'CRB-inspired');
h_mi_i = plot(mi_sumrate_avg(valid_mi_i), mi_islr_avg_dB(valid_mi_i), ':v', ...
    'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', [0.55 0.35 0.75], ...
    'Color', [0.45 0.25 0.65], 'DisplayName', 'MI-inspired');
add_cluster_box(mi_sumrate_avg(valid_mi_i), mi_islr_avg_dB(valid_mi_i), ...
    [0.45 0.25 0.65], xlim_i, ylim_i, 0.45, 0.18);
h_e2e_cv_i = [];
h_e2e_direct_i = [];
if e2e_available
    h_e2e_cv_i = plot(e2e_cv_sumrate_avg(valid_e2e_cv_i), e2e_cv_islr_avg_dB(valid_e2e_cv_i), '-x', ...
        'LineWidth', 2.0, 'MarkerSize', 8, 'Color', [0.00 0.45 0.55], ...
        'DisplayName', 'E2E NN-CV');
    h_e2e_direct_i = plot(e2e_direct_sumrate_avg(valid_e2e_direct_i), ...
        e2e_direct_islr_avg_dB(valid_e2e_direct_i), '--x', ...
        'LineWidth', 2.0, 'MarkerSize', 8, 'Color', [0.55 0.20 0.55], ...
        'DisplayName', 'E2E NN-Direct');
end
h_comm_i = plot(comm_sumrate, comm_islr_dB, 'kp', 'LineWidth', 1.6, 'MarkerSize', 13, ...
    'MarkerFaceColor', [0.95 0.75 0.10], 'DisplayName', 'Communication-only');
h_comm_rate_i = xline(comm_sumrate, '-.', 'LineWidth', 1.6, 'Color', [0 0 0], ...
    'DisplayName', 'Comm.-only rate');
h_global_i = yline(islr_min_dB, '--', 'LineWidth', 1.7, 'Color', [0.10 0.10 0.10], ...
    'DisplayName', 'Global min ISLR');
for ii = label_idx
    if valid(ii)
        text(sumrate_avg(ii), islr_avg_dB(ii) + 0.01, sprintf('%.1f', CV_max_list(ii)), ...
            'FontSize', plot_style.cv_label_font, 'Color', [0.10 0.30 0.55]);
    end
end
ax_i = gca;
set(ax_i, 'FontSize', plot_style.axes_font, 'Layer', 'bottom', ...
    'Units', 'normalized', 'Position', plot_style.axes_position);
xlabel(ax_i, 'Sum-rate (bps/Hz)', 'FontSize', plot_style.label_font);
ylabel(ax_i, 'ISLR (dB, worst-case across targets)', 'FontSize', plot_style.label_font);
title_handle_i = title(ax_i, ['ISLR Pareto  ' title_str], 'FontSize', plot_style.title_font);
legend_handles_i = [h_prop_i h_direct_i h_equiv_i h_crb_i h_mi_i];
if e2e_available
    legend_handles_i = [legend_handles_i h_e2e_cv_i h_e2e_direct_i];
end
legend_handles_i = [legend_handles_i h_comm_i h_comm_rate_i h_global_i];
legend(ax_i, legend_handles_i, ...
    'Location', 'northwest', 'NumColumns', 2, ...
    'FontSize', min(plot_style.legend_font, 14.5));
set(title_handle_i, 'FontSize', plot_style.title_font);
fig_islr = ancestor(ax_i, 'figure');
drawnow;
islr_png = fullfile(out_dir, 'pareto_curve_islr.png');
safe_save(fig_islr, islr_png);
safe_save(fig_islr, fullfile(out_dir, 'pareto_curve_islr.fig'));
safe_export(fig_islr, fullfile(paper_fig_dir, 'ISLR_Pareto_Frontier_Result.pdf'), 'pdf');
copyfile(islr_png, fullfile(paper_fig_dir, 'ISLR_Pareto_Frontier_Result.png'), 'f');

fprintf('Communication-only: SR=%.2f, PSLR=%.2f dB, ISLR=%.2f dB\n', ...
        comm_sumrate, comm_pslr_dB, comm_islr_dB);
end

function [sumrate_grid, pslr_grid, islr_grid] = run_comm_only(params)
sumrate_grid = nan(1, params.num_mc);
pslr_grid = nan(1, params.num_mc);
islr_grid = nan(1, params.num_mc);
params_comm = params;
params_comm.P_des = 0;
params_comm.run_direct_baseline = false;
for mc = 1:params.num_mc
    rng(mc, 'twister');
    H = generate_channel(params_comm);
    result = run_proposed(H, 1e3, params_comm);
    if ~isnan(result.sumrate)
        sumrate_grid(mc) = result.sumrate;
        pslr_grid(mc) = min(result.pslr_per_target);
        islr_grid(mc) = max(result.islr_per_target);
    end
end
end

function tf = same_eta_grid(a, b)
a = a(:); b = b(:);
tf = numel(a) == numel(b) && all(abs(a - b) <= 1e-12 * max(1, abs(b)));
end

function [sumrate_grid, pslr_grid, islr_grid] = run_surrogate_sweep(mode, eta_list, params)
sumrate_grid = nan(numel(eta_list), params.num_mc);
pslr_grid = nan(numel(eta_list), params.num_mc);
islr_grid = nan(numel(eta_list), params.num_mc);
for mc = 1:params.num_mc
    rng(mc, 'twister');
    H = generate_channel(params);
    alpha_warm = [];
    for i = 1:numel(eta_list)
        result = run_surrogate_baseline(H, mode, eta_list(i), params, alpha_warm);
        if ~isnan(result.sumrate)
            alpha_warm = result.alpha;
            sumrate_grid(i, mc) = result.sumrate;
            pslr_grid(i, mc) = min(result.pslr_per_target);
            islr_grid(i, mc) = max(result.islr_per_target);
        else
            alpha_warm = [];
        end
        fprintf('%s baseline eta=%.3g: SR=%.2f, PSLR=%.2f dB, ISLR=%.2f dB\n', ...
            upper(mode), eta_list(i), sumrate_grid(i, mc), ...
            10*log10(pslr_grid(i, mc)), 10*log10(islr_grid(i, mc)));
    end
end
end

function [x_env, y_env] = upper_envelope(x, y)
valid = isfinite(x) & isfinite(y);
x = x(valid); y = y(valid);
[x, idx] = sort(x); y = y(idx);
[xu, ~, ic] = unique(round(x, 8));
yu = accumarray(ic, y, [], @max);
y_env = yu;
for i = numel(y_env)-1:-1:1
    y_env(i) = max(y_env(i), y_env(i+1));
end
x_env = xu;
end

function [x_env, y_env] = lower_envelope(x, y)
valid = isfinite(x) & isfinite(y);
x = x(valid); y = y(valid);
[x, idx] = sort(x); y = y(idx);
[xu, ~, ic] = unique(round(x, 8));
yu = accumarray(ic, y, [], @min);
y_env = yu;
for i = numel(y_env)-1:-1:1
    y_env(i) = min(y_env(i), y_env(i+1));
end
x_env = xu;
end


function [xq, yq] = smooth_boundary(x, y, xl, side)
valid = isfinite(x) & isfinite(y);
x = x(valid); y = y(valid);
[x, idx] = sort(x); y = y(idx);
[x, ia] = unique(x, 'stable'); y = y(ia);
if isempty(x)
    xq = []; yq = [];
    return;
end
if strcmpi(side, 'left') && x(1) > xl(1)
    x = [xl(1); x];
    y = [y(1); y];
end
if numel(x) >= 3
    xq = linspace(min(x), max(x), 250).';
    yq = interp1(x, y, xq, 'pchip');
else
    xq = x(:); yq = y(:);
end
end

function add_cluster_box(x, y, color, xl, yl, min_x_span, min_y_span)
valid = isfinite(x) & isfinite(y);
x = x(valid); y = y(valid);
if isempty(x)
    return;
end
if nargin < 6, min_x_span = 0.45; end
if nargin < 7, min_y_span = 0.18; end
x_span = max(max(x) - min(x), min_x_span);
y_span = max(max(y) - min(y), min_y_span);
x_pad = 0.35 * x_span;
y_pad = 0.45 * y_span;
x0 = max(min(x) - x_pad, xl(1));
x1 = min(max(x) + x_pad, xl(2));
y0 = max(min(y) - y_pad, yl(1));
y1 = min(max(y) + y_pad, yl(2));
rectangle('Position', [x0, y0, max(x1 - x0, eps), max(y1 - y0, eps)], ...
    'EdgeColor', color, 'LineStyle', '--', 'LineWidth', 1.4, ...
    'FaceColor', 'none', 'HandleVisibility', 'off');
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        exportgraphics(fig, filename, 'ContentType', 'image', 'Resolution', 600);
    else
        exportgraphics(fig, filename, 'Resolution', 600);
    end
catch
    parent_fig = ancestor(fig, 'figure');
    if ~isempty(parent_fig)
        fig = parent_fig;
    end
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-opengl', '-r300');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
end

function safe_save(fig, filename)
try
    saveas(fig, filename);
catch
    [~, ~, ext] = fileparts(filename);
    if strcmpi(ext, '.fig')
        savefig(fig, filename);
    elseif strcmpi(ext, '.png')
        print(fig, filename, '-dpng', '-r300');
    else
        saveas(fig, filename);
    end
end
end

function [xl, yl] = padded_limits(x, y)
x = x(isfinite(x)); y = y(isfinite(y));
xs = max(x) - min(x); if xs <= 0, xs = max(abs(x(1))*0.05, 1); end
ys = max(y) - min(y); if ys <= 0, ys = max(abs(y(1))*0.05, 1); end
xl = [min(x)-0.08*xs, max(x)+0.16*xs];
yl = [min(y)-0.15*ys, max(y)+0.15*ys];
end



