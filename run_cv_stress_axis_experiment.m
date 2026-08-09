function run_cv_stress_axis_experiment(num_mc_override, force_rerun, mc_indices)
% RUN_CV_STRESS_AXIS_EXPERIMENT
% Stress-axis diagnostic with CV on the vertical axis.
%
% Each panel fixes one system stress level and sweeps CV_max. The horizontal
% axes compare feasibility rate and direct/CV computational-burden ratios.

if nargin < 1 || isempty(num_mc_override)
    num_mc_override = 5;
end
if nargin < 2 || isempty(force_rerun)
    force_rerun = false;
end
if nargin < 3 || isempty(mc_indices)
    mc_indices = [];
end
is_shard = ~isempty(mc_indices);
close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
out_data_dir = fullfile(sim_dir, 'results');
fig_dir = fullfile(sim_dir, 'figures');
if exist(out_data_dir, 'dir') ~= 7, mkdir(out_data_dir); end
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
addpath(genpath(sim_dir));

CV_grid = 0:0.1:1.0;
num_cv = numel(CV_grid);
num_mc = num_mc_override;
scenarios = build_scenarios();
num_scenarios = numel(scenarios);
time_budget_seconds = feasibility_time_budgets(scenarios);
budget_policy = ['Common per-scenario wall-clock budget for both methods; ' ...
    '3 s for S2/S4 and 4 s for S3.'];

if isempty(mc_indices)
    mc_indices = 1:num_mc;
else
    mc_indices = unique(mc_indices(:).', 'stable');
    if any(mc_indices < 1) || any(mc_indices > num_mc) || ...
            any(mc_indices ~= floor(mc_indices)) || ...
            any(diff(mc_indices) ~= 1)
        error('mc_indices must be a consecutive integer range within 1:num_mc.');
    end
end

source_path = fullfile(out_data_dir, sprintf( ...
    'cv_stress_axis_pslr_only_S2S3S4_CV10_NT4_N16_MC%d.mat', num_mc));
if is_shard
    source_path = fullfile(out_data_dir, sprintf( ...
        'cv_stress_axis_pslr_only_S2S3S4_CV10_NT4_N16_MC%d_shard_%03d_%03d.mat', ...
        num_mc, mc_indices(1), mc_indices(end)));
end

if exist(source_path, 'file') == 2 && ~force_rerun
    fprintf('Using cached CV stress-axis result: %s\n', source_path);
    if is_shard
        return;
    end
    plot_cv_stress_axis_results(source_path);
    print_summary(source_path);
    return;
end

prop_success = false(num_scenarios, num_cv, num_mc);
direct_success = false(num_scenarios, num_cv, num_mc);
prop_status = strings(num_scenarios, num_cv, num_mc);
direct_status = strings(num_scenarios, num_cv, num_mc);
prop_time = nan(num_scenarios, num_cv, num_mc);
direct_time = nan(num_scenarios, num_cv, num_mc);
prop_cvx_solver_iters = nan(num_scenarios, num_cv, num_mc);
direct_cvx_solver_iters = nan(num_scenarios, num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  CV Stress-Axis Diagnostic: CV-SDP vs Direct PSLR SCA\n');
fprintf('============================================================\n');
fprintf('  MC=%d, CV grid=[%s]\n', num_mc, num2str(CV_grid));
fprintf('  Scenarios: S2 Higher Illumination, S3 More Targets, S4 Joint Stress\n');
fprintf('  N_T is fixed at 4 for all scenarios.\n');
fprintf('  Common method budgets by scenario: [%s] s\n', num2str(time_budget_seconds.'));
fprintf('  Timed runs are quiet; solver-log replay is used for total IPM iterations.\n');
fprintf('------------------------------------------------------------\n');

t_global = tic;
total_runs = num_scenarios * num_cv * numel(mc_indices) * 2;
run_count = 0;

for s = 1:num_scenarios
    params = scenario_params(scenarios(s), false);
    params_profile = scenario_params(scenarios(s), true);
    fprintf('Scenario %d/%d: %s\n', s, num_scenarios, scenarios(s).label);
    fprintf('  N_T=%d, N=%d, L=%d, Q=%.2f, P_des=%.2f Pmax/N\n', ...
        params.NT, params.N, params.L, params.Q(1), params.P_des * params.N / params.P_max);

    for mc = mc_indices
        rng(2000*s + mc, 'twister');
        H = generate_channel(params);
        alpha0 = init_alpha_qos_safe(H, params);
        W0 = init_covariance_flat(params);

        for c = 1:num_cv
            CV_max = CV_grid(c);
            pslr_min = direct_thresholds_from_cv(CV_max, params);

            run_count = run_count + 1;
            t_run = tic;
            prop = run_proposed(H, CV_max, params, alpha0);
            prop_time(s, c, mc) = toc(t_run);
            prop_profile = run_proposed(H, CV_max, params_profile, alpha0);
            prop_cvx_solver_iters(s, c, mc) = prop_profile.cvx_solver_iters;
            prop_status(s, c, mc) = string(prop.status);
            prop_success(s, c, mc) = is_solved(prop.status);
            print_progress('CV-SDP', run_count, total_runs, scenarios(s).short, mc, num_mc, ...
                CV_max, prop_time(s, c, mc), prop_cvx_solver_iters(s, c, mc), prop.status, t_global);

            run_count = run_count + 1;
            t_run = tic;
            direct = run_direct_sca(H, pslr_min, params, alpha0, W0);
            direct_time(s, c, mc) = toc(t_run);
            direct_profile = run_direct_sca(H, pslr_min, params_profile, alpha0, W0);
            direct_cvx_solver_iters(s, c, mc) = direct_profile.cvx_solver_iters;
            direct_status(s, c, mc) = string(direct.status);
            direct_success(s, c, mc) = is_solved(direct.status);
            print_progress('Direct', run_count, total_runs, scenarios(s).short, mc, num_mc, ...
                CV_max, direct_time(s, c, mc), direct_cvx_solver_iters(s, c, mc), direct.status, t_global);
        end
    end

    save(source_path, 'CV_grid', 'num_mc', 'scenarios', ...
        'time_budget_seconds', 'budget_policy', ...
        'prop_success', 'direct_success', 'prop_status', 'direct_status', ...
        'prop_time', 'direct_time', ...
        'prop_cvx_solver_iters', 'direct_cvx_solver_iters');
end

if is_shard
    fprintf('Saved Figure 7 shard: %s\n', source_path);
    return;
end

plot_cv_stress_axis_results(source_path);
print_summary(source_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', source_path);
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');
end

function scenarios = build_scenarios()
scenarios = struct([]);

scenarios(1).id = 'S2';
scenarios(1).short = 'S2-Illum';
scenarios(1).label = 'S2 Higher Illumination';
scenarios(1).NT = 4;
scenarios(1).N = 16;
scenarios(1).L = 4;
scenarios(1).theta = [-30, 0, 30, 60] * pi/180;
scenarios(1).Q = 1.00;
scenarios(1).Pdes_scale = 1.10;

scenarios(2).id = 'S3';
scenarios(2).short = 'S3-Tgts';
scenarios(2).label = 'S3 More Targets';
scenarios(2).NT = 4;
scenarios(2).N = 16;
scenarios(2).L = 6;
scenarios(2).theta = linspace(-60, 60, 6) * pi/180;
scenarios(2).Q = 1.25;
scenarios(2).Pdes_scale = 0.90;

scenarios(3).id = 'S4';
scenarios(3).short = 'S4-Joint';
scenarios(3).label = 'S4 Joint QoS + Illumination';
scenarios(3).NT = 4;
scenarios(3).N = 16;
scenarios(3).L = 4;
scenarios(3).theta = [-30, 0, 30, 60] * pi/180;
scenarios(3).Q = 2.00;
scenarios(3).Pdes_scale = 1.10;
end

function params = scenario_params(scenario, collect_solver_log)
params = setup_params();
params.NT = scenario.NT;
params.N = scenario.N;
params.L = scenario.L;
params.theta = scenario.theta;
params.Q = scenario.Q * ones(params.K, 1);
params.P_des = scenario.Pdes_scale * params.P_max / params.N;
params.num_mc = 1;
params.warm_start_cv = false;
params.stop_if_alpha_unchanged = true;
params.sdp_quiet = ~collect_solver_log;
params.collect_cvx_solver_log = collect_solver_log;
params.cvx_solver = 'mosek';
params.cvx_solver_threads = 1;
params.max_iter = 5;
params.direct_ao_max_iter = 5;
params.direct_sca_max_iter = 5;
params.direct_sca_tol = 1e-3;
end

function tf = is_solved(status)
tf = contains(string(status), 'Solved');
end

function alpha = init_alpha_qos_safe(H, params)
K = params.K;
N = params.N;
alpha = zeros(K, N);
if K > N
    alpha = init_alpha(H, params);
    return;
end

available = true(1, N);
for k = 1:K
    gains = squeeze(sum(abs(H(:, k, :)).^2, 1)).';
    gains(~available) = -Inf;
    [~, n_best] = max(gains);
    alpha(k, n_best) = 1;
    available(n_best) = false;
end

for n = find(available)
    h_norms = vecnorm(H(:, :, n), 2, 1);
    [~, k_best] = max(h_norms);
    alpha(k_best, n) = 1;
end
end

function print_progress(name, run_count, total_runs, scenario, mc, num_mc, CV_max, elapsed, ipm_iters, status, t_global)
eta = toc(t_global) / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf('[%5.1f%% %3d/%3d] %-6s %-8s MC %d/%d CV=%.1f | time=%6.2fs | IPM=%5.0f | %s | ETA %s\n', ...
    100*run_count/total_runs, run_count, total_runs, name, scenario, mc, num_mc, ...
    CV_max, elapsed, ipm_iters, status, format_time(eta));
end

function plot_cv_stress_axis_results(source_path)
S = load(source_path);
sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, 'figures');
paper_fig_dir = fullfile(sim_dir, '..', 'MyPaper', 'figures');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
if exist(paper_fig_dir, 'dir') ~= 7, mkdir(paper_fig_dir); end

cfg = plot_config();
num_scenarios = numel(S.scenarios);
[cv_feas, direct_feas, ~, ~, ...
    cv_time, direct_time, cv_ipm, direct_ipm] = summarize_grids(S);
keep = S.CV_grid(:) >= 0.1 - 1e-12;
CV = S.CV_grid(keep).';
cv_feas = cv_feas(:, keep);
direct_feas = direct_feas(:, keep);
cv_time = cv_time(:, keep);
direct_time = direct_time(:, keep);
cv_ipm = cv_ipm(:, keep);
direct_ipm = direct_ipm(:, keep);

palette = paper_palette();
cv_color = palette(1, :);
direct_color = palette(2, :);
plot_style = struct( ...
    'figure_position', [100 100 1320 1600], ...
    'axes_x', [0.085 0.395 0.705], ...
    'row_y', [0.700 0.385 0.070], ...
    'axes_width', 0.265, ...
    'axes_height', 0.245, ...
    'legend_position', [0.795 0.710 0.180 0.062], ...
    'axes_font', round(cfg.tall_panel_font_scale * cfg.axes_font), ...
    'label_font', round(cfg.tall_panel_font_scale * cfg.label_font), ...
    'title_font', round(cfg.tall_panel_font_scale * cfg.title_font), ...
    'legend_font', round(cfg.tall_panel_font_scale * cfg.legend_font), ...
    'line_width', cfg.line_width, ...
    'marker_size', round(0.85 * cfg.marker_size), ...
    'axes_line_width', cfg.axes_line_width);
% Use one shared limit per row, padded only slightly above that row's
% overall maximum so all three scenario panels remain directly comparable.
runtime_upper = nice_axis_upper([cv_time(:); direct_time(:)], 2, 2);
ipm_upper = nice_axis_upper([cv_ipm(:); direct_ipm(:)], 40, 40);
row_limits = {[-2 102], [0 runtime_upper], [0 ipm_upper]};
row_ticks = {[0 50 100], 0:2:runtime_upper, 0:40:ipm_upper};
x_limits = valid_axis_limits(CV, cfg);

fig = figure('Position', plot_style.figure_position, 'Color', 'w');
set(fig, 'PaperPositionMode', 'auto');
ax_grid = gobjects(3, num_scenarios);
title_handles = gobjects(1, num_scenarios);
for s = 1:num_scenarios
    ax_feas = axes(fig, 'Position', ...
        [plot_style.axes_x(s), plot_style.row_y(1), ...
         plot_style.axes_width, plot_style.axes_height]);
    ax_grid(1, s) = ax_feas;
    hold(ax_feas, 'on'); grid(ax_feas, 'on'); box(ax_feas, 'on');
    set(ax_feas, 'Layer', 'top', 'FontSize', plot_style.axes_font, ...
        'FontWeight', 'normal', ...
        'LineWidth', plot_style.axes_line_width, ...
        'LabelFontSizeMultiplier', 1);

    plot(ax_feas, CV, 100*cv_feas(s, :).', '-d', ...
        'Color', cv_color, 'MarkerFaceColor', cv_color, ...
        'MarkerEdgeColor', cv_color, 'LineWidth', plot_style.line_width, ...
        'MarkerSize', plot_style.marker_size, 'DisplayName', 'CV-SDP');
    plot(ax_feas, CV, 100*direct_feas(s, :).', '-d', ...
        'Color', direct_color, 'MarkerFaceColor', direct_color, ...
        'MarkerEdgeColor', direct_color, 'LineWidth', plot_style.line_width, ...
        'MarkerSize', plot_style.marker_size, 'DisplayName', 'Direct SCA');
    xlim(ax_feas, x_limits);
    ylim(ax_feas, row_limits{1});
    yticks(ax_feas, row_ticks{1});
    xticks(ax_feas, [0 0.5 1]);
    set_row_xticklabels(ax_feas, s);
    if s == 1
        ylabel(ax_feas, 'Budgeted feasibility (%)', ...
            'FontSize', plot_style.label_font, 'FontWeight', 'normal');
    else
        yticklabels(ax_feas, []);
    end

    title_handles(s) = title(ax_feas, display_label(S.scenarios(s)), ...
        'Interpreter', 'tex', ...
        'FontSize', plot_style.title_font, ...
        'FontWeight', 'normal');

    ax_runtime = axes(fig, 'Position', ...
        [plot_style.axes_x(s), plot_style.row_y(2), ...
         plot_style.axes_width, plot_style.axes_height]);
    ax_grid(2, s) = ax_runtime;
    hold(ax_runtime, 'on'); grid(ax_runtime, 'on'); box(ax_runtime, 'on');
    set(ax_runtime, 'Layer', 'top', 'FontSize', plot_style.axes_font, ...
        'FontWeight', 'normal', ...
        'LineWidth', plot_style.axes_line_width, ...
        'YScale', 'linear', ...
        'LabelFontSizeMultiplier', 1);

    plot(ax_runtime, CV, cv_time(s, :).', '-d', ...
        'Color', cv_color, 'MarkerFaceColor', cv_color, ...
        'MarkerEdgeColor', cv_color, 'LineWidth', plot_style.line_width, ...
        'MarkerSize', plot_style.marker_size, 'DisplayName', 'CV-SDP');
    plot(ax_runtime, CV, direct_time(s, :).', '-d', ...
        'Color', direct_color, 'MarkerFaceColor', direct_color, ...
        'MarkerEdgeColor', direct_color, 'LineWidth', plot_style.line_width, ...
        'MarkerSize', plot_style.marker_size, 'DisplayName', 'Direct SCA');
    xlim(ax_runtime, x_limits);
    ylim(ax_runtime, row_limits{2});
    yticks(ax_runtime, row_ticks{2});
    xticks(ax_runtime, [0 0.5 1]);
    set_row_xticklabels(ax_runtime, s);
    if s == 1
        ylabel(ax_runtime, 'Runtime (s)', ...
            'FontSize', plot_style.label_font, 'FontWeight', 'normal');
    else
        yticklabels(ax_runtime, []);
    end

    ax_ipm = axes(fig, 'Position', ...
        [plot_style.axes_x(s), plot_style.row_y(3), ...
         plot_style.axes_width, plot_style.axes_height]);
    ax_grid(3, s) = ax_ipm;
    hold(ax_ipm, 'on'); grid(ax_ipm, 'on'); box(ax_ipm, 'on');
    set(ax_ipm, 'Layer', 'top', 'FontSize', plot_style.axes_font, ...
        'FontWeight', 'normal', ...
        'LineWidth', plot_style.axes_line_width, ...
        'YScale', 'linear', ...
        'LabelFontSizeMultiplier', 1);

    plot(ax_ipm, CV, cv_ipm(s, :).', '-d', ...
        'Color', cv_color, 'MarkerFaceColor', cv_color, ...
        'MarkerEdgeColor', cv_color, 'LineWidth', plot_style.line_width, ...
        'MarkerSize', plot_style.marker_size, 'DisplayName', 'CV-SDP');
    plot(ax_ipm, CV, direct_ipm(s, :).', '-d', ...
        'Color', direct_color, 'MarkerFaceColor', direct_color, ...
        'MarkerEdgeColor', direct_color, 'LineWidth', plot_style.line_width, ...
        'MarkerSize', plot_style.marker_size, 'DisplayName', 'Direct SCA');
    xlim(ax_ipm, x_limits);
    ylim(ax_ipm, row_limits{3});
    yticks(ax_ipm, row_ticks{3});
    xticks(ax_ipm, [0 0.5 1]);
    set_row_xticklabels(ax_ipm, s);
    if s == 1
        ylabel(ax_ipm, 'Total IPM iterations', ...
            'FontSize', plot_style.label_font, 'FontWeight', 'normal');
    else
        yticklabels(ax_ipm, []);
    end
end

plot_config(fig);
for row_idx = 1:3
    for s = 1:num_scenarios
        ax = ax_grid(row_idx, s);
        try
            set(ax, 'PositionConstraint', 'innerposition');
        catch
        end
        set(ax, 'Units', 'normalized', ...
            'FontSize', plot_style.axes_font, ...
            'FontWeight', 'normal', ...
            'LineWidth', plot_style.axes_line_width, ...
            'Position', [plot_style.axes_x(s), plot_style.row_y(row_idx), ...
                         plot_style.axes_width, plot_style.axes_height]);
    end
end
for s = 1:num_scenarios
    set(title_handles(s), 'FontSize', plot_style.title_font, ...
        'FontWeight', 'normal');
end
set(ax_grid(1, 1).YLabel, 'Units', 'normalized', ...
    'Position', [-0.135 0.5 0], ...
    'FontSize', plot_style.label_font, 'FontWeight', 'normal');
set(ax_grid(2, 1).YLabel, 'Units', 'normalized', ...
    'Position', [-0.135 0.5 0], ...
    'FontSize', plot_style.label_font, 'FontWeight', 'normal');
set(ax_grid(3, 1).YLabel, 'Units', 'normalized', ...
    'Position', [-0.135 0.5 0], ...
    'FontSize', plot_style.label_font, 'FontWeight', 'normal');
draw_row_xlabels(fig, plot_style, cfg);
draw_stress_legend(fig, plot_style.legend_position, ...
    cv_color, direct_color, plot_style, cfg);

out_png = fullfile(fig_dir, 'CV_Stress_Axis_Diagnostic.png');
out_pdf = fullfile(fig_dir, 'CV_Stress_Axis_Diagnostic.pdf');
out_onecol_png = fullfile(fig_dir, 'CV_Stress_Axis_Diagnostic_OneColumn_1x3.png');
out_onecol_pdf = fullfile(fig_dir, 'CV_Stress_Axis_Diagnostic_OneColumn_1x3.pdf');
safe_export(fig, out_png, 'png');
safe_export(fig, out_pdf, 'pdf');
safe_export(fig, out_onecol_png, 'png');
safe_export(fig, out_onecol_pdf, 'pdf');
copyfile(out_onecol_png, fullfile(paper_fig_dir, 'CV_Stress_Axis_Diagnostic_OneColumn_1x3.png'));
copyfile(out_onecol_pdf, fullfile(paper_fig_dir, 'CV_Stress_Axis_Diagnostic_OneColumn_1x3.pdf'));
fprintf('Saved CV stress-axis figure: %s\n', out_pdf);
fprintf('Saved CV stress-axis figure: %s\n', out_png);
fprintf('Saved CV stress-axis figure: %s\n', out_onecol_pdf);
fprintf('Saved CV stress-axis figure: %s\n', out_onecol_png);
end

function set_row_xticklabels(ax, ~)
% Keep the complete CV scale visible in every panel.
xticks(ax, [0.1 0.4 0.7 1.0]);
xticklabels(ax, {'0.1', '0.4', '0.7', '1.0'});
end

function draw_row_xlabels(fig, style, cfg)
label_y = style.row_y - 0.062;
label_width = style.axes_x(end) + style.axes_width - style.axes_x(1);
for row_idx = 1:numel(style.row_y)
    annotation(fig, 'textbox', ...
        [style.axes_x(1), label_y(row_idx), label_width, 0.035], ...
        'String', 'CV_{max}', ...
        'Interpreter', 'tex', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'EdgeColor', 'none', ...
        'FitBoxToText', 'off', ...
        'FontName', cfg.font_name, ...
        'FontSize', style.label_font, ...
        'FontWeight', 'normal');
end
end

function draw_stress_legend(fig, position, cv_color, direct_color, style, cfg)
ax_leg = axes(fig, 'Position', position, ...
    'XLim', [0 1], 'YLim', [0 1], ...
    'Visible', 'off', ...
    'Color', 'none');
hold(ax_leg, 'on');
rectangle(ax_leg, 'Position', [0.010 0.020 0.980 0.960], ...
    'FaceColor', cfg.legend_background_color, ...
    'FaceAlpha', cfg.legend_face_alpha, ...
    'EdgeColor', cfg.legend_edge_color, ...
    'LineWidth', style.axes_line_width);

legend_font = style.legend_font;
y_pos = [0.700 0.300];
labels = {'CV-SDP', 'Direct SCA'};
colors = [cv_color; direct_color];
for idx = 1:numel(labels)
    plot(ax_leg, [0.045 0.235], [y_pos(idx) y_pos(idx)], '-', ...
        'Color', colors(idx, :), ...
        'LineWidth', cfg.line_width, ...
        'Clipping', 'off');
    x_center = 0.140;
    dx = 0.029;
    dy = 0.082;
    patch(ax_leg, x_center + [0 dx 0 -dx], ...
        y_pos(idx) + [dy 0 -dy 0], ...
        colors(idx, :), ...
        'EdgeColor', colors(idx, :), ...
        'LineWidth', cfg.secondary_line_width, ...
        'Clipping', 'off');
    text(ax_leg, 0.280, y_pos(idx), labels{idx}, ...
        'Interpreter', 'tex', ...
        'FontName', cfg.font_name, ...
        'FontSize', legend_font, ...
        'FontWeight', 'normal', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
end
try
    uistack(ax_leg, 'top');
catch
end
end

function label = display_label(scenario)
switch string(scenario.id)
    case "S2"
        label = 'S2 Higher Illum.';
    case "S3"
        label = 'S3 More Targets';
    case "S4"
        label = 'S4 Joint Stress';
    otherwise
        label = scenario.label;
end
end

function [cv_feas, direct_feas, runtime_ratio, ipm_ratio, ...
    cv_time_mean, direct_time_mean, cv_ipm_mean, direct_ipm_mean] = summarize_grids(S)
num_scenarios = numel(S.scenarios);
num_cv = numel(S.CV_grid);
cv_feas = nan(num_scenarios, num_cv);
direct_feas = nan(num_scenarios, num_cv);
runtime_ratio = nan(num_scenarios, num_cv);
ipm_ratio = nan(num_scenarios, num_cv);
cv_time_mean = nan(num_scenarios, num_cv);
direct_time_mean = nan(num_scenarios, num_cv);
cv_ipm_mean = nan(num_scenarios, num_cv);
direct_ipm_mean = nan(num_scenarios, num_cv);
time_budget = result_time_budgets(S);

for s = 1:num_scenarios
    for c = 1:num_cv
        cv_ok = squeeze(S.prop_success(s, c, :)) & ...
            squeeze(S.prop_time(s, c, :)) <= time_budget(s);
        direct_ok = squeeze(S.direct_success(s, c, :)) & ...
            squeeze(S.direct_time(s, c, :)) <= time_budget(s);
        cv_feas(s, c) = mean(cv_ok(:));
        direct_feas(s, c) = mean(direct_ok(:));

        cv_t = squeeze(S.prop_time(s, c, :));
        direct_t = squeeze(S.direct_time(s, c, :));
        cv_time_mean(s, c) = mean(cv_t(:), 'omitnan');
        direct_time_mean(s, c) = mean(direct_t(:), 'omitnan');
        runtime_ratio(s, c) = direct_time_mean(s, c) / cv_time_mean(s, c);

        cv_ipm = squeeze(S.prop_cvx_solver_iters(s, c, :));
        direct_ipm = squeeze(S.direct_cvx_solver_iters(s, c, :));
        cv_ipm_mean(s, c) = mean(cv_ipm(:), 'omitnan');
        direct_ipm_mean(s, c) = mean(direct_ipm(:), 'omitnan');
        ipm_ratio(s, c) = direct_ipm_mean(s, c) / cv_ipm_mean(s, c);
    end
end
end

function time_budget = feasibility_time_budgets(scenarios)
% Fixed latency budget, applied identically to CV-SDP and Direct SCA for
% every scenario.
time_budget = 3 * ones(numel(scenarios), 1);
if numel(scenarios) >= 2
    time_budget(2) = 4;
end
end

function time_budget = result_time_budgets(S)
if isfield(S, 'time_budget_seconds') && ...
        numel(S.time_budget_seconds) == numel(S.scenarios)
    time_budget = S.time_budget_seconds(:);
else
    time_budget = feasibility_time_budgets(S.scenarios);
end
end

function upper = nice_axis_upper(values, quantum, minimum_upper)
valid = values(isfinite(values) & values >= 0);
if isempty(valid)
    upper = minimum_upper;
else
    upper = max(minimum_upper, quantum * ceil(1.08 * max(valid) / quantum));
end
end

function upper = padded_row_upper(values, fallback_upper)
valid = values(isfinite(values) & values >= 0);
if isempty(valid)
    upper = fallback_upper;
else
    upper = 1.05 * max(valid);
end
end

function set_log_limits(ax, values, fallback_limits)
valid = values(isfinite(values) & values > 0);
if isempty(valid)
    ylim(ax, fallback_limits);
    return;
end
lower = 10^floor(log10(min(valid) * 0.8));
upper = 10^ceil(log10(max(valid) * 1.25));
if ~isfinite(lower) || ~isfinite(upper) || lower >= upper
    lower = min(valid) * 0.8;
    upper = max(valid) * 1.25;
end
if lower <= 0 || lower >= upper
    ylim(ax, fallback_limits);
else
    ylim(ax, [lower upper]);
end
end

function print_summary(source_path)
S = load(source_path);
[cv_feas, direct_feas, runtime_ratio, ipm_ratio] = summarize_grids(S);
keep = S.CV_grid >= 0.1 - 1e-12;
time_budget = result_time_budgets(S);
fprintf('\nStress-axis summary by scenario:\n');
for s = 1:numel(S.scenarios)
    fprintf('  %s | budget %.1fs | CV feas %.1f%%, Direct feas %.1f%% | runtime %.1fx | IPM %.1fx\n', ...
        S.scenarios(s).short, time_budget(s), ...
        100*mean(cv_feas(s, keep), 'omitnan'), ...
        100*mean(direct_feas(s, keep), 'omitnan'), ...
        mean(runtime_ratio(s, keep), 'omitnan'), mean(ipm_ratio(s, keep), 'omitnan'));
end
fprintf('\nBy CV target:\n');
for s = 1:numel(S.scenarios)
    fprintf('  %s\n', S.scenarios(s).short);
    for c = find(keep)
        fprintf('    CV=%.1f: feasibility CV %.0f%%, Direct %.0f%% | runtime %.1fx | IPM %.1fx\n', ...
            S.CV_grid(c), 100*cv_feas(s, c), 100*direct_feas(s, c), ...
            runtime_ratio(s, c), ipm_ratio(s, c));
    end
end
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        tmp_png = [tempname(fileparts(filename)), '.png'];
        tight_export_figure(fig, tmp_png, 'Resolution', 300, ...
            'TightLayout', false, 'TightPad', 0);
        if ~png_to_pdf(tmp_png, filename)
            tight_export_figure(fig, filename, 'ContentType', 'image', ...
                'Resolution', 450, 'TightLayout', false, 'TightPad', 0);
        end
        if exist(tmp_png, 'file') == 2
            delete(tmp_png);
        end
    else
        tight_export_figure(fig, filename, 'Resolution', 300, ...
            'TightLayout', false, 'TightPad', 0);
    end
catch
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-r300');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
end

function ok = png_to_pdf(png_file, pdf_file)
ok = false;
python_exe = fullfile(getenv('USERPROFILE'), '.cache', 'codex-runtimes', ...
    'codex-primary-runtime', 'dependencies', 'python', 'python.exe');
if exist(python_exe, 'file') ~= 2
    return;
end
py_code = sprintf(['from PIL import Image; ', ...
    'img=Image.open(r''%s'').convert(''RGB''); ', ...
    'img.save(r''%s'', ''PDF'', resolution=300.0)'], png_file, pdf_file);
cmd = sprintf('"%s" -c "%s"', python_exe, py_code);
[status, ~] = system(cmd);
ok = (status == 0) && exist(pdf_file, 'file') == 2;
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
