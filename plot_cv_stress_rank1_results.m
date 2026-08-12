function plot_cv_stress_rank1_results(source_path, output_dir, copy_to_paper)
%PLOT_CV_STRESS_RANK1_RESULTS Square-panel 2-by-3 stress-test figure.

if nargin < 1 || isempty(source_path)
    sim_dir = fileparts(mfilename('fullpath'));
    config = cv_stress_rank1_config(100);
    source_path = fullfile(sim_dir, 'results', config.result_filename);
end
if nargin < 2, output_dir = ''; end
if nargin < 3 || isempty(copy_to_paper), copy_to_paper = true; end
S = load(source_path);
validate_source(S);

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = output_dir;
if isempty(fig_dir), fig_dir = fullfile(sim_dir, 'figures'); end
paper_fig_dir = fullfile(sim_dir, '..', 'MyPaper', 'figures');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
if exist(paper_fig_dir, 'dir') ~= 7, mkdir(paper_fig_dir); end

summary = summarize_results(S);
cfg = plot_config();
palette = cfg.palette;
cv_color = palette(1, :);
direct_color = palette(2, :);
neutral = [0.12 0.12 0.12];

runtime_upper = nice_upper([summary.cv_runtime(:); ...
    summary.direct_runtime(:)], 1, 2);
ipm_upper = nice_upper([summary.cv_ipm(:); ...
    summary.direct_ipm(:)], 10, 20);

fig = figure('Position', [80 50 1320 920], 'Color', 'w');
set(fig, 'PaperPositionMode', 'auto');
axes_x = [0.055 0.375 0.695];
axes_y = [0.565 0.135];
axes_w = 0.235;
axes_h = 0.325;
num_scenarios = numel(S.scenarios);
ax_grid = gobjects(2, num_scenarios);
top_legend_handles = gobjects(1, 4);
bottom_legend_handles = gobjects(1, 6);

for s = 1:num_scenarios
    ax = axes(fig, 'Position', [axes_x(s), axes_y(1), axes_w, axes_h]);
    ax_grid(1, s) = ax;
    hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
    style_axes(ax, cfg);
    yyaxis(ax, 'left');
    h1 = plot(ax, S.CV_grid, summary.cv_runtime(s, :), '-o', ...
        'Color', cv_color, 'MarkerFaceColor', cv_color, ...
        'LineWidth', 1.8, 'MarkerSize', 5.2);
    h2 = plot(ax, S.CV_grid, summary.direct_runtime(s, :), '-d', ...
        'Color', direct_color, 'MarkerFaceColor', direct_color, ...
        'LineWidth', 1.8, 'MarkerSize', 5.2);
    ylim(ax, [0 runtime_upper]);
    ylabel(ax, 'Runtime (s)', 'FontSize', 13);

    yyaxis(ax, 'right');
    h3 = plot(ax, S.CV_grid, summary.cv_ipm(s, :), '--^', ...
        'Color', cv_color, 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.6, 'MarkerSize', 5.2);
    h4 = plot(ax, S.CV_grid, summary.direct_ipm(s, :), '--v', ...
        'Color', direct_color, 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.6, 'MarkerSize', 5.2);
    ylim(ax, [0 ipm_upper]);
    ylabel(ax, 'Total IPM iterations', 'FontSize', 13);
    set_dual_axis_colors(ax, neutral);
    finish_panel(ax, S.CV_grid, S.scenarios(s).label, ...
        sprintf('(%c)', 'a' + s - 1), false);
    if s == 1
        top_legend_handles = [h1 h2 h3 h4];
    end
end

for s = 1:num_scenarios
    ax = axes(fig, 'Position', [axes_x(s), axes_y(2), axes_w, axes_h]);
    ax_grid(2, s) = ax;
    hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
    style_axes(ax, cfg);
    yyaxis(ax, 'left');
    h1 = plot(ax, S.CV_grid, 100 * summary.cv_budget(s, :), '--o', ...
        'Color', cv_color, 'MarkerFaceColor', cv_color, ...
        'LineWidth', 1.8, 'MarkerSize', 5.2);
    h2 = plot(ax, S.CV_grid, 100 * summary.direct_budget(s, :), '--d', ...
        'Color', direct_color, 'MarkerFaceColor', direct_color, ...
        'LineWidth', 1.8, 'MarkerSize', 5.2);
    ylim(ax, [0 102]);
    yticks(ax, [0 25 50 75 100]);
    ylabel(ax, 'Budgeted feasibility (%)', 'FontSize', 13);

    yyaxis(ax, 'right');
    h3 = plot(ax, S.CV_grid, 100 * summary.cv_evd(s, :), ':^', ...
        'Color', cv_color, 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.6, 'MarkerSize', 5.2);
    h4 = plot(ax, S.CV_grid, 100 * summary.cv_gr(s, :), '-s', ...
        'Color', cv_color, 'MarkerFaceColor', cv_color, ...
        'LineWidth', 1.8, 'MarkerSize', 5.0);
    h5 = plot(ax, S.CV_grid, 100 * summary.direct_evd(s, :), ':v', ...
        'Color', direct_color, 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.6, 'MarkerSize', 5.2);
    h6 = plot(ax, S.CV_grid, 100 * summary.direct_gr(s, :), '-p', ...
        'Color', direct_color, 'MarkerFaceColor', direct_color, ...
        'LineWidth', 1.8, 'MarkerSize', 5.6);
    ylim(ax, [0 102]);
    yticks(ax, [0 25 50 75 100]);
    ylabel(ax, 'Rank-one feasibility (%)', 'FontSize', 13);
    set_dual_axis_colors(ax, neutral);
    finish_panel(ax, S.CV_grid, '', sprintf('(%c)', 'd' + s - 1), true);
    if s == 1
        bottom_legend_handles = [h1 h2 h3 h4 h5 h6];
    end
end

top_legend = legend(ax_grid(1, 1), top_legend_handles, ...
    {'CV runtime', 'Direct runtime', 'CV IPM iter.', 'Direct IPM iter.'}, ...
    'Orientation', 'horizontal', 'NumColumns', 4, ...
    'Box', 'off', 'FontSize', 12);
set(top_legend, 'Units', 'normalized', ...
    'Position', [0.245 0.925 0.510 0.035]);

bottom_legend = legend(ax_grid(2, 1), bottom_legend_handles, ...
    {'CV budget', 'Direct budget', 'CV EVD', 'CV EVD+GR', ...
     'Direct EVD', 'Direct EVD+GR'}, ...
    'Orientation', 'horizontal', 'NumColumns', 6, ...
    'Box', 'off', 'FontSize', 11);
set(bottom_legend, 'Units', 'normalized', ...
    'Position', [0.105 0.015 0.790 0.035]);

drawnow;
for row = 1:2
    for s = 1:num_scenarios
        pbaspect(ax_grid(row, s), [1 1 1]);
    end
end

base = 'CV_Stress_RankOne_2x3';
out_png = fullfile(fig_dir, [base '.png']);
out_pdf = fullfile(fig_dir, [base '.pdf']);
safe_export(fig, out_png, 'png');
safe_export(fig, out_pdf, 'pdf');
if copy_to_paper
    copyfile(out_png, fullfile(paper_fig_dir, [base '.png']));
    copyfile(out_pdf, fullfile(paper_fig_dir, [base '.pdf']));
end
fprintf('Saved square-panel 2-by-3 stress figure: %s\n', out_pdf);
end

function summary = summarize_results(S)
summary.cv_runtime = mean(S.prop_time, 3, 'omitnan');
summary.direct_runtime = mean(S.direct_time, 3, 'omitnan');
summary.cv_ipm = mean(S.prop_cvx_solver_iters, 3, 'omitnan');
summary.direct_ipm = mean(S.direct_cvx_solver_iters, 3, 'omitnan');
summary.cv_evd = mean(S.prop_initial_evd_feasible, 3);
summary.direct_evd = mean(S.direct_initial_evd_feasible, 3);
summary.cv_gr = mean(S.prop_final_rank1_feasible, 3);
summary.direct_gr = mean(S.direct_final_rank1_feasible, 3);
budget = reshape(S.time_budget_seconds, [], 1, 1);
summary.cv_budget = mean(S.prop_final_rank1_feasible & ...
    S.prop_time <= budget, 3);
summary.direct_budget = mean(S.direct_final_rank1_feasible & ...
    S.direct_time <= budget, 3);
end

function validate_source(S)
required = {'CV_grid', 'scenarios', 'time_budget_seconds', ...
    'prop_time', 'direct_time', 'prop_cvx_solver_iters', ...
    'direct_cvx_solver_iters', 'prop_initial_evd_feasible', ...
    'direct_initial_evd_feasible', 'prop_final_rank1_feasible', ...
    'direct_final_rank1_feasible', 'completed'};
for i = 1:numel(required)
    if ~isfield(S, required{i})
        error('Stress result is missing %s.', required{i});
    end
end
if ~all(S.completed(:))
    error('Stress result contains incomplete Monte Carlo points.');
end
end

function style_axes(ax, cfg)
set(ax, 'Layer', 'top', 'FontName', cfg.font_name, 'FontSize', 11.5, ...
    'FontWeight', 'normal', 'LineWidth', 0.9, ...
    'GridColor', cfg.grid_color, 'GridAlpha', cfg.grid_alpha, ...
    'GridLineStyle', ':', 'LabelFontSizeMultiplier', 1);
end

function finish_panel(ax, CV_grid, panel_title, panel_label, show_xlabel)
xlim(ax, [min(CV_grid) - 0.02, max(CV_grid) + 0.02]);
xticks(ax, [0.1 0.4 0.7 1.0]);
if show_xlabel
    xlabel(ax, 'CV_{max}', 'Interpreter', 'tex', 'FontSize', 13);
end
if ~isempty(panel_title)
    title(ax, panel_title, 'FontSize', 14, 'FontWeight', 'normal');
end
text(ax, -0.16, 1.02, panel_label, 'Units', 'normalized', ...
    'FontSize', 13, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
pbaspect(ax, [1 1 1]);
end

function set_dual_axis_colors(ax, color)
try
    ax.YAxis(1).Color = color;
    ax.YAxis(2).Color = color;
catch
end
end

function upper = nice_upper(values, quantum, minimum_upper)
values = values(isfinite(values) & values >= 0);
if isempty(values)
    upper = minimum_upper;
else
    upper = max(minimum_upper, quantum * ceil(1.08 * max(values) / quantum));
end
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        exportgraphics(fig, filename, 'ContentType', 'vector');
    else
        exportgraphics(fig, filename, 'Resolution', 300);
    end
catch
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-bestfit');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
end
