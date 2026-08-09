function plot_pareto_grid_1x4(num_mc_override, config_indices, mc_indices)
% PLOT_PARETO_GRID_1X4  Build the PSLR-only 1-by-4 Pareto figure.
%
% The four panels use the same configurations and paper styling as the
% former 2-by-4 PSLR/ISLR figure. Legacy PSLR, CRB, MI, and communication-
% only results are migrated when available, while the direct baseline is
% recomputed without any ISLR constraint.

if nargin < 1
    num_mc_override = [];
end
if nargin < 2 || isempty(config_indices)
    config_indices = 1:4;
end
if nargin < 3 || isempty(mc_indices)
    mc_indices = [];
end

sim_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(sim_dir);
sim_fig_dir = fullfile(sim_dir, 'figures');
paper_fig_dir = fullfile(project_dir, 'MyPaper', 'figures');
cache_dir = fullfile(sim_dir, 'results', 'pareto_grid_1x4_pslr');
legacy_cache_dir = fullfile(sim_dir, 'results', 'pareto_grid_2x4');
ensure_dir(sim_fig_dir);
ensure_dir(paper_fig_dir);
ensure_dir(cache_dir);

configs = struct( ...
    'NT', {4, 4, 8, 8}, ...
    'N',  {16, 32, 16, 32}, ...
    'label', {'$N_T=4,N=16$', '$N_T=4,N=32$', ...
              '$N_T=8,N=16$', '$N_T=8,N=32$'});

config_indices = unique(config_indices(:).', 'stable');
if any(config_indices < 1) || any(config_indices > numel(configs))
    error('config_indices must contain values from 1 to %d.', numel(configs));
end

case_data = cell(numel(configs), 1);
for i = config_indices
    case_data{i} = load_or_run_case(configs(i), cache_dir, ...
        legacy_cache_dir, num_mc_override, mc_indices);
end

if numel(config_indices) < numel(configs)
    fprintf('Completed configuration indices [%s]. No combined figure was exported.\n', ...
        num2str(config_indices));
    return;
end

cfg = plot_config();
font_scale = 0.65;
cfg.axes_font = max(round(font_scale * cfg.axes_font), 1);
cfg.label_font = max(round(font_scale * cfg.label_font), 1);
cfg.title_font = max(round(font_scale * cfg.title_font), 1);
cfg.legend_font = max(round(font_scale * cfg.legend_font), 1);
cfg.panel_caption_font = max(round(font_scale * cfg.panel_caption_font), 1);
cfg.curve_marker_size = max(round(font_scale * (cfg.marker_size + 3)), 1);
cfg.boundary_marker_size = max(round(font_scale * (cfg.marker_size + 15)), 1);
fig_width = 1905;
fig_height = 700;
fig = figure('Position', [80 80 fig_width fig_height], 'Color', 'w');
axes_x = [0.055 0.291 0.527 0.763];
axes_width = 0.187;
axes_y = 0.155;
axes_height = 0.590;

panel_axes = gobjects(1, numel(configs));
for i = 1:numel(configs)
    ax = axes(fig, 'Units', 'normalized', ...
        'Position', [axes_x(i) axes_y axes_width axes_height]);
    panel_axes(i) = ax;
    plot_case_panel(ax, case_data{i}, configs(i).label, cfg);
    if i > 1
        ylabel(ax, '');
    end
    pbaspect(ax, [1 1 1]);
end

plot_config(fig);
for i = 1:numel(panel_axes)
    set(panel_axes(i), 'FontSize', cfg.axes_font);
    set(panel_axes(i).XLabel, 'FontSize', cfg.label_font);
    set(panel_axes(i).YLabel, 'FontSize', cfg.label_font);
    set(panel_axes(i).Title, 'FontSize', cfg.title_font);
    line_objects = findall(panel_axes(i), 'Type', 'line');
    for j = 1:numel(line_objects)
        marker = get(line_objects(j), 'Marker');
        if strcmpi(marker, 'none')
            continue;
        elseif strcmpi(marker, 'p')
            set(line_objects(j), 'MarkerSize', cfg.boundary_marker_size);
        else
            set(line_objects(j), 'MarkerSize', cfg.curve_marker_size);
        end
    end
end
set(panel_axes(1).YLabel, ...
    'FontSize', cfg.label_font, ...
    'FontWeight', 'bold');

annotation(fig, 'textbox', [0 0.014 1 0.070], ...
    'String', 'Sum-rate (bps/Hz)', ...
    'Interpreter', 'tex', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', ...
    'FitBoxToText', 'off', ...
    'FontName', cfg.font_name, ...
    'FontSize', cfg.label_font, ...
    'FontWeight', 'bold');
draw_pareto_grid_legend(fig, [0.140 0.860 0.720 0.105], cfg);

pdf_name = 'Pareto_Frontier_Grid_1x4.pdf';
png_name = 'Pareto_Frontier_Grid_1x4.png';
saveas(fig, fullfile(sim_dir, 'pareto_grid_1x4.fig'));
sim_pdf_path = fullfile(sim_fig_dir, pdf_name);
sim_png_path = fullfile(sim_fig_dir, png_name);
safe_export(fig, sim_pdf_path, 'pdf');
safe_export(fig, sim_png_path, 'png');
copyfile(sim_pdf_path, fullfile(paper_fig_dir, pdf_name), 'f');
copyfile(sim_png_path, fullfile(paper_fig_dir, png_name), 'f');
fprintf('Saved PSLR-only 1x4 Pareto figure to:\n  %s\n  %s\n', ...
    sim_fig_dir, paper_fig_dir);
end

function S = load_or_run_case(cfg, cache_dir, legacy_cache_dir, num_mc_override, mc_indices)
params = setup_params();
params.NT = cfg.NT;
params.N = cfg.N;
params.P_des = 0.8 * params.P_max / params.N;
params.CV_max_list = [0, 0.1:0.1:0.9];
if ~isempty(num_mc_override)
    params.num_mc = num_mc_override;
end

cache_name = sprintf('pareto_pslr_NT%d_N%d_MC%d.mat', ...
    cfg.NT, cfg.N, params.num_mc);
cache_path = fullfile(cache_dir, cache_name);
if ~isempty(mc_indices)
    cache_path = fullfile(cache_dir, sprintf( ...
        'pareto_pslr_NT%d_N%d_MC%d_shard_%03d_%03d.mat', ...
        cfg.NT, cfg.N, params.num_mc, mc_indices(1), mc_indices(end)));
end
legacy_path = fullfile(legacy_cache_dir, sprintf( ...
    'pareto_grid_NT%d_N%d_MC%d.mat', cfg.NT, cfg.N, params.num_mc));

if exist(cache_path, 'file') == 2
    S = load(cache_path);
    fprintf('Loaded PSLR-only cache: %s\n', cache_path);
else
    if exist(legacy_path, 'file') == 2
        legacy = load(legacy_path);
        S = migrate_legacy_pslr(legacy, params);
        fprintf('Migrated PSLR data from: %s\n', legacy_path);
    else
        S = [];
    end
end

    S = run_reference_curves(params, cache_path, S, mc_indices);
    save(cache_path, '-struct', 'S', '-v7.3');
S = ensure_direct_pslr_curve(S, cache_path, mc_indices);
end

function S = migrate_legacy_pslr(legacy, params)
S.params = params;
S.CV_max_list = params.CV_max_list;
S.sumrate_grid = legacy.sumrate_grid;
S.pslr_lin_grid = legacy.pslr_lin_grid;
S.comm_sumrate_grid = legacy.comm_sumrate_grid;
S.comm_pslr_lin_grid = legacy.comm_pslr_lin_grid;
S.crb_eta_list = legacy.crb_eta_list;
S.crb_sumrate_grid = legacy.crb_sumrate_grid;
S.crb_pslr_lin_grid = legacy.crb_pslr_lin_grid;
S.mi_eta_list = legacy.mi_eta_list;
S.mi_sumrate_grid = legacy.mi_sumrate_grid;
S.mi_pslr_lin_grid = legacy.mi_pslr_lin_grid;
S.direct_sumrate_grid = nan(size(S.sumrate_grid));
S.direct_pslr_lin_grid = nan(size(S.pslr_lin_grid));
S.direct_completed_mc = false(1, params.num_mc);
S.direct_pslr_only_version = 1;
end

function S = run_reference_curves(params, cache_path, S, mc_indices)
[crb_eta_list, mi_eta_list] = surrogate_eta_lists(params);
num_cv = numel(params.CV_max_list);
num_mc = params.num_mc;

if isempty(S)
    S.params = params;
    S.CV_max_list = params.CV_max_list;
    S.sumrate_grid = nan(num_cv, num_mc);
    S.pslr_lin_grid = nan(num_cv, num_mc);
    S.comm_sumrate_grid = nan(1, num_mc);
    S.comm_pslr_lin_grid = nan(1, num_mc);
    S.crb_eta_list = crb_eta_list;
    S.crb_sumrate_grid = nan(numel(crb_eta_list), num_mc);
    S.crb_pslr_lin_grid = nan(numel(crb_eta_list), num_mc);
    S.mi_eta_list = mi_eta_list;
    S.mi_sumrate_grid = nan(numel(mi_eta_list), num_mc);
    S.mi_pslr_lin_grid = nan(numel(mi_eta_list), num_mc);
else
    S.params = params;
    if ~isfield(S, 'reference_completed_mc')
        S.reference_completed_mc = all(isfinite(S.comm_sumrate_grid), 1);
    end
end
if ~isfield(S, 'reference_completed_mc') || numel(S.reference_completed_mc) ~= num_mc
    S.reference_completed_mc = false(1, num_mc);
end
% Keep "attempted" separate from "all outputs finite".  Older interrupted
% runs could mark an entirely blank realization as completed, which made the
% resume path skip it forever.  A partially populated realization, however,
% may represent a genuine solver failure and must not be retried silently.
if ~isfield(S, 'reference_attempted_mc') || ...
        numel(S.reference_attempted_mc) ~= num_mc
    S.reference_attempted_mc = reference_has_any_result(S);
end

fprintf('Running PSLR reference curves: N_T=%d, N=%d, MC=%d\n', ...
    params.NT, params.N, num_mc);
if isempty(mc_indices)
    mc_indices = 1:num_mc;
end
for mc = mc_indices
    if S.reference_attempted_mc(mc)
        continue;
    end
    rng(mc, 'twister');
    H = generate_channel(params);

    alpha_warm = [];
    for c = 1:num_cv
        result = run_proposed(H, params.CV_max_list(c), params, alpha_warm);
        if isfinite(result.sumrate)
            alpha_warm = result.alpha;
            S.sumrate_grid(c, mc) = result.sumrate;
            S.pslr_lin_grid(c, mc) = min(result.pslr_per_target);
        else
            alpha_warm = [];
        end
    end

    params_comm = params;
    params_comm.P_des = 0;
    comm_result = run_proposed(H, 1e3, params_comm);
    if isfinite(comm_result.sumrate)
        S.comm_sumrate_grid(mc) = comm_result.sumrate;
        S.comm_pslr_lin_grid(mc) = min(comm_result.pslr_per_target);
    end

    S = run_surrogate_curve(S, H, params, 'crb', mc);
    S = run_surrogate_curve(S, H, params, 'mi', mc);
    S.reference_attempted_mc(mc) = true;
    S.reference_completed_mc(mc) = reference_is_complete(S, mc);
    save(cache_path, '-struct', 'S', '-v7.3');
end
end

function S = run_surrogate_curve(S, H, params, mode, mc)
eta_list = S.(sprintf('%s_eta_list', mode));
sumrate_field = sprintf('%s_sumrate_grid', mode);
pslr_field = sprintf('%s_pslr_lin_grid', mode);
alpha_warm = [];
for e = 1:numel(eta_list)
    result = run_surrogate_baseline(H, mode, eta_list(e), params, alpha_warm);
    if isfinite(result.sumrate)
        alpha_warm = result.alpha;
        S.(sumrate_field)(e, mc) = result.sumrate;
        S.(pslr_field)(e, mc) = min(result.pslr_per_target);
    else
        alpha_warm = [];
    end
end
end

function S = ensure_direct_pslr_curve(S, cache_path, mc_indices)
num_mc = S.params.num_mc;
num_cv = numel(S.params.CV_max_list);
if ~isfield(S, 'direct_sumrate_grid') || ...
        ~isequal(size(S.direct_sumrate_grid), [num_cv, num_mc])
    S.direct_sumrate_grid = nan(num_cv, num_mc);
    S.direct_pslr_lin_grid = nan(num_cv, num_mc);
end
if ~isfield(S, 'direct_completed_mc') || ...
        numel(S.direct_completed_mc) ~= num_mc
    S.direct_completed_mc = false(1, num_mc);
end
if ~isfield(S, 'direct_attempted_with_reference_mc') || ...
        numel(S.direct_attempted_with_reference_mc) ~= num_mc
    S.direct_attempted_with_reference_mc = ...
        S.direct_completed_mc & reference_has_any_result(S) & ...
        any(isfinite(S.direct_sumrate_grid), 1);
end
if ~isfield(S, 'direct_pslr_only_version') || ...
        S.direct_pslr_only_version ~= 1
    S.direct_sumrate_grid(:) = NaN;
    S.direct_pslr_lin_grid(:) = NaN;
    S.direct_completed_mc(:) = false;
    S.direct_attempted_with_reference_mc(:) = false;
    S.direct_pslr_only_version = 1;
end

if isempty(mc_indices)
    mc_indices = 1:num_mc;
end
for mc = mc_indices
    if S.direct_attempted_with_reference_mc(mc)
        continue;
    end
    rng(mc, 'twister');
    H = generate_channel(S.params);
    alpha_warm = [];
    W_warm = [];
    fprintf('Direct PSLR-only: N_T=%d, N=%d, MC=%02d/%02d\n', ...
        S.params.NT, S.params.N, mc, num_mc);
    for c = 1:num_cv
        achieved_pslr = S.pslr_lin_grid(c, mc);
        if ~isfinite(achieved_pslr)
            continue;
        end
        pslr_min = achieved_pslr * (1 - S.params.direct_pslr_target_relax);
        result = run_direct_sca(H, pslr_min, S.params, alpha_warm, W_warm);
        if isfinite(result.sumrate)
            S.direct_sumrate_grid(c, mc) = result.sumrate;
            S.direct_pslr_lin_grid(c, mc) = min(result.pslr_per_target);
            alpha_warm = result.alpha;
            W_warm = result.W;
        else
            alpha_warm = [];
            W_warm = [];
        end
        fprintf('  CV %.1f: SR %.2f, PSLR %.2f dB, %s\n', ...
            S.params.CV_max_list(c), S.direct_sumrate_grid(c, mc), ...
            10*log10(S.direct_pslr_lin_grid(c, mc)), result.status);
    end
    S.direct_attempted_with_reference_mc(mc) = true;
    S.direct_completed_mc(mc) = all(isfinite(S.direct_sumrate_grid(:, mc))) && ...
        all(isfinite(S.direct_pslr_lin_grid(:, mc)));
    save(cache_path, '-struct', 'S', '-v7.3');
end
end

function mask = reference_has_any_result(S)
fields = {'sumrate_grid', 'pslr_lin_grid', 'comm_sumrate_grid', ...
    'comm_pslr_lin_grid', 'crb_sumrate_grid', 'crb_pslr_lin_grid', ...
    'mi_sumrate_grid', 'mi_pslr_lin_grid'};
num_mc = size(S.sumrate_grid, 2);
mask = false(1, num_mc);
for i = 1:numel(fields)
    if isfield(S, fields{i})
        mask = mask | any(isfinite(S.(fields{i})), 1);
    end
end
end

function tf = reference_is_complete(S, mc)
fields = {'sumrate_grid', 'pslr_lin_grid', 'comm_sumrate_grid', ...
    'comm_pslr_lin_grid', 'crb_sumrate_grid', 'crb_pslr_lin_grid', ...
    'mi_sumrate_grid', 'mi_pslr_lin_grid'};
tf = true;
for i = 1:numel(fields)
    tf = tf && all(isfinite(S.(fields{i})(:, mc)));
end
end

function [crb_eta_list, mi_eta_list] = surrogate_eta_lists(params)
base_crb = [0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 ...
            4.5e-3 6.7e-3 1e-2 1.32e-2 1.73e-2 2.28e-2 ...
            3e-2 3.98e-2 5.28e-2 6.95e-2 0.1 0.144 0.208 0.3 1 3 10 30 100];
base_mi = [0.1 0.3 1 3 10 30 100];
if params.NT >= 8
    crb_extra = [5e-4 7e-4 1.5e-3 2e-3 2.5e-3 3.5e-3 ...
                 5e-3 5.7e-3 8e-3 1.2e-2 1.6e-2];
    mi_extra = [0.2 0.5 0.7 1.5 2 4 5 7 15 20 50];
else
    crb_extra = [];
    mi_extra = [];
end
crb_eta_list = sort(unique([base_crb crb_extra]));
mi_eta_list = sort(unique([base_mi mi_extra]));
end

function plot_case_panel(ax, S, title_text, cfg)
axes(ax);
hold(ax, 'on');
grid(ax, 'on');
box(ax, 'on');

sumrate_avg = mean(S.sumrate_grid, 2, 'omitnan');
direct_sumrate_avg = mean(S.direct_sumrate_grid, 2, 'omitnan');
crb_sumrate_avg = mean(S.crb_sumrate_grid, 2, 'omitnan');
mi_sumrate_avg = mean(S.mi_sumrate_grid, 2, 'omitnan');
comm_sumrate = mean(S.comm_sumrate_grid, 'omitnan');
y_prop = 10*log10(mean(S.pslr_lin_grid, 2, 'omitnan'));
y_direct = 10*log10(mean(S.direct_pslr_lin_grid, 2, 'omitnan'));
y_crb = 10*log10(mean(S.crb_pslr_lin_grid, 2, 'omitnan'));
y_mi = 10*log10(mean(S.mi_pslr_lin_grid, 2, 'omitnan'));
y_bound = 10*log10(1 + S.params.N/(S.params.kappa - 1));

valid_prop = isfinite(sumrate_avg) & isfinite(y_prop);
valid_direct = isfinite(direct_sumrate_avg) & isfinite(y_direct);
valid_crb = isfinite(crb_sumrate_avg) & isfinite(y_crb);
valid_mi = isfinite(mi_sumrate_avg) & isfinite(y_mi);

palette = paper_palette();
c_prop = palette(1, :);
c_direct = palette(2, :);
c_crb = palette(7, :);
c_mi = palette(4, :);
c_comm = [0.20 0.62 0.22];
c_sensing = palette(9, :);

h_prop = plot(ax, sumrate_avg(valid_prop), y_prop(valid_prop), '--o', ...
    'Color', c_prop, 'MarkerFaceColor', c_prop, ...
    'LineWidth', 1.4, 'MarkerSize', cfg.curve_marker_size, ...
    'DisplayName', 'Proposed CV');
h_direct = plot(ax, direct_sumrate_avg(valid_direct), y_direct(valid_direct), '--d', ...
    'Color', c_direct, 'MarkerFaceColor', c_direct, ...
    'LineWidth', 1.2, 'MarkerSize', cfg.curve_marker_size, ...
    'DisplayName', 'Direct SCA');
plot(ax, crb_sumrate_avg(valid_crb), y_crb(valid_crb), '--^', ...
    'Color', c_crb, 'MarkerFaceColor', c_crb, ...
    'LineWidth', 1.1, 'MarkerSize', cfg.curve_marker_size, ...
    'DisplayName', 'CRB-inspired');
plot(ax, mi_sumrate_avg(valid_mi), y_mi(valid_mi), '--v', ...
    'Color', c_mi, 'MarkerFaceColor', c_mi, ...
    'LineWidth', 1.3, 'MarkerSize', cfg.curve_marker_size, ...
    'DisplayName', 'MI-inspired');
xline(ax, comm_sumrate, '--', 'Color', c_comm, 'LineWidth', 1.4, ...
    'DisplayName', 'Communication-only');
yline(ax, y_bound, '--', 'Color', c_sensing, 'LineWidth', 1.4, ...
    'DisplayName', 'sensing-only');

pareto_x = [sumrate_avg(valid_prop); direct_sumrate_avg(valid_direct)];
pareto_y = [y_prop(valid_prop); y_direct(valid_direct)];
h_comm_star = gobjects(0);
h_sensing_star = gobjects(0);
if ~isempty(pareto_x)
    [~, comm_idx] = min(abs(pareto_x - comm_sumrate));
    [~, sensing_idx] = min(abs(pareto_y - y_bound));
    h_comm_star = plot(ax, pareto_x(comm_idx), pareto_y(comm_idx), 'p', ...
        'LineStyle', 'none', 'Color', c_comm, 'MarkerFaceColor', c_comm, ...
        'MarkerSize', cfg.boundary_marker_size, 'LineWidth', 2.0, ...
        'HandleVisibility', 'off');
    h_sensing_star = plot(ax, pareto_x(sensing_idx), pareto_y(sensing_idx), 'p', ...
        'LineStyle', 'none', 'Color', c_sensing, 'MarkerFaceColor', c_sensing, ...
        'MarkerSize', cfg.boundary_marker_size, 'LineWidth', 2.0, ...
        'HandleVisibility', 'off');
end

x_focus = [sumrate_avg(valid_prop); direct_sumrate_avg(valid_direct); comm_sumrate];
y_focus = [y_prop(valid_prop); y_direct(valid_direct); y_bound];
[xl, yl] = corner_limits(x_focus, y_focus);
xlim(ax, xl);
ylim(ax, yl);
add_enclosed_region_shade(ax, ...
    {sumrate_avg(valid_prop), direct_sumrate_avg(valid_direct)}, ...
    {y_prop(valid_prop), y_direct(valid_direct)}, comm_sumrate, y_bound);
try
    uistack(h_direct, 'top');
    uistack(h_prop, 'top');
    if isgraphics(h_comm_star), uistack(h_comm_star, 'top'); end
    if isgraphics(h_sensing_star), uistack(h_sensing_star, 'top'); end
catch
end

ylabel(ax, 'PSLR (dB)', 'FontSize', cfg.label_font - 2);
set(ax, 'FontSize', cfg.axes_font, 'Layer', 'top');
try
    ax.TitleFontSizeMultiplier = 1;
catch
end
title_handle = title(ax, title_text, 'Interpreter', 'latex', ...
    'FontSize', cfg.title_font, 'FontWeight', 'bold');
title_handle.FontSize = cfg.title_font;
end

function add_enclosed_region_shade(ax, x_cells, y_cells, comm_x, bound_y)
palette = paper_palette();
[frontier_x, frontier_y] = outer_frontier(x_cells, y_cells);
if numel(frontier_x) < 2 || ~isfinite(comm_x) || ~isfinite(bound_y)
    return;
end
valid = frontier_x <= comm_x & isfinite(frontier_x) & isfinite(frontier_y);
frontier_x = frontier_x(valid);
frontier_y = frontier_y(valid);
if numel(frontier_x) < 2
    return;
end
axis_x = xlim(ax);
axis_y = ylim(ax);
poly_x = [axis_x(1); frontier_x(:); comm_x; comm_x; axis_x(1)];
poly_y = [bound_y; frontier_y(:); frontier_y(end); axis_y(1); axis_y(1)];
region = patch(ax, poly_x, poly_y, palette(1, :), ...
    'FaceAlpha', 0.10, 'EdgeColor', 'none', 'HandleVisibility', 'off');
try
    uistack(region, 'bottom');
catch
end
end

function draw_pareto_grid_legend(fig, position, cfg)
palette = paper_palette();
colors = [palette(1, :); palette(2, :); palette(7, :); ...
          palette(4, :); palette(9, :); 0.20 0.62 0.22];
labels = {'Proposed CV', 'Direct SCA', 'CRB-inspired', ...
          'MI-inspired', 'sensing-only', 'Communication-only'};
markers = {'o', 'd', '^', 'v', 'p', 'p'};

ax_leg = axes(fig, 'Position', position, ...
    'XLim', [0 1], 'YLim', [0 1], 'Visible', 'off', 'Color', 'none');
hold(ax_leg, 'on');
rectangle(ax_leg, 'Position', [0.015 0.015 0.970 0.970], ...
    'FaceColor', 'w', 'EdgeColor', [0.15 0.15 0.15], ...
    'LineWidth', cfg.axes_line_width);

legend_font = cfg.legend_font;
legend_marker_size = max(1.15 * cfg.marker_size, 10);
legend_star_size = max(1.85 * cfg.marker_size, 16);
legend_line_width = max(0.80 * cfg.line_width, 1.5);
num_columns = 3;
row_y = [0.650 0.350];
content_left = 0.020;
content_width = 0.960;
column_width = content_width / num_columns;

for idx = 1:numel(labels)
    row_idx = 1 + floor((idx - 1) / num_columns);
    column_idx = mod(idx - 1, num_columns);
    cell_left = content_left + column_idx * column_width;
    y = row_y(row_idx);
    x_line = cell_left + [0.010 0.065];
    x_marker = mean(x_line);
    x_text = cell_left + 0.078;
    marker_size = legend_marker_size;
    if idx >= 5
        marker_size = legend_star_size;
    end
    plot(ax_leg, x_line, [y y], '--', ...
        'Color', colors(idx, :), 'LineWidth', legend_line_width, ...
        'Clipping', 'off');
    scatter(ax_leg, x_marker, y, marker_size^2, ...
        'Marker', markers{idx}, ...
        'MarkerEdgeColor', colors(idx, :), ...
        'MarkerFaceColor', colors(idx, :), ...
        'LineWidth', 1.5, 'Clipping', 'off');
    text(ax_leg, x_text, y, labels{idx}, ...
        'Interpreter', 'tex', 'FontName', cfg.font_name, ...
        'FontSize', legend_font, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end
try
    uistack(ax_leg, 'top');
catch
end
end

function [frontier_x, frontier_y] = outer_frontier(x_cells, y_cells)
num_curves = numel(x_cells);
mins = nan(num_curves, 1);
maxs = nan(num_curves, 1);
for i = 1:num_curves
    x = x_cells{i}(:);
    y = y_cells{i}(:);
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    if numel(x) >= 2
        mins(i) = min(x);
        maxs(i) = max(x);
    end
end
x_min = min(mins, [], 'omitnan');
x_max = max(maxs, [], 'omitnan');
if ~isfinite(x_min) || ~isfinite(x_max) || x_max <= x_min
    frontier_x = [];
    frontier_y = [];
    return;
end
frontier_x = linspace(x_min, x_max, 200).';
Y = nan(numel(frontier_x), num_curves);
for i = 1:num_curves
    x = x_cells{i}(:);
    y = y_cells{i}(:);
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    if numel(x) < 2
        continue;
    end
    [x, order] = sort(x);
    y = y(order);
    [x_unique, ~, ic] = unique(x);
    y_unique = accumarray(ic, y, [], @max);
    if numel(x_unique) >= 2
        in_range = frontier_x >= min(x_unique) & frontier_x <= max(x_unique);
        Y(in_range, i) = interp1(x_unique, y_unique, frontier_x(in_range), 'linear');
    end
end
frontier_y = max(Y, [], 2, 'omitnan');
valid = isfinite(frontier_y);
frontier_x = frontier_x(valid);
frontier_y = frontier_y(valid);
end

function [xl, yl] = corner_limits(x, y)
x = x(isfinite(x));
y = y(isfinite(y));
xs = max(x) - min(x);
if xs <= 0, xs = max(abs(x(1))*0.05, 1); end
ys = max(y) - min(y);
if ys <= 0, ys = max(abs(y(1))*0.05, 1); end
target_fraction = 0.40;
outside_pad = 0.24;
x_high = max(x) + outside_pad * xs;
xl = [x_high - xs / target_fraction, x_high];
y_high = max(y) + outside_pad * ys;
yl = [y_high - ys / target_fraction, y_high];
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        tight_export_figure(fig, filename, 'ContentType', 'image', ...
            'Resolution', 450, 'TightLayout', false);
    else
        tight_export_figure(fig, filename, 'Resolution', 300, ...
            'TightLayout', false);
    end
catch
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-vector');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
end

function ensure_dir(path_name)
if exist(path_name, 'dir') ~= 7
    mkdir(path_name);
end
end
