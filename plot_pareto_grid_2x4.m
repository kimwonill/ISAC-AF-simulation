function plot_pareto_grid_2x4(num_mc_override)
% PLOT_PARETO_GRID_2X4  Build a two-column 2-by-4 Pareto figure.
%
% Columns sweep representative (N_T,N) configurations. The top row shows
% PSLR Pareto curves and the bottom row shows ISLR Pareto curves.
if nargin < 1
    num_mc_override = [];
end

out_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(out_dir, 'figures');
cache_dir = fullfile(out_dir, 'results', 'pareto_grid_2x4');
if exist(paper_fig_dir, 'dir') ~= 7, mkdir(paper_fig_dir); end
if exist(cache_dir, 'dir') ~= 7, mkdir(cache_dir); end

configs = struct( ...
    'NT', {4, 4, 8, 8}, ...
    'N',  {16, 32, 16, 32}, ...
    'label', {'$N_T=4,N=16$', '$N_T=4,N=32$', '$N_T=8,N=16$', '$N_T=8,N=32$'});

case_data = cell(numel(configs), 1);
for i = 1:numel(configs)
    case_data{i} = load_or_run_case(configs(i), cache_dir, num_mc_override);
end

cfg = plot_config();
base_width = 2000;
base_height = 1250;
fig_width = 1905;
fig_height = 1205;
x_crop_scale = base_width / fig_width;
y_crop_scale = base_height / fig_height;
fig = figure('Position', [80 80 fig_width fig_height], 'Color', 'w');
axes_x = [0.055 0.291 0.527 0.763] * x_crop_scale;
axes_width = 0.187 * x_crop_scale;
axes_height = 0.350 * y_crop_scale;
top_y = 0.445 * y_crop_scale;
bottom_y = 0.080 * y_crop_scale;

panel_axes = gobjects(2, numel(configs));
for i = 1:numel(configs)
    ax = axes(fig, 'Units', 'normalized', ...
        'Position', [axes_x(i) top_y axes_width axes_height]);
    panel_axes(1, i) = ax;
    plot_case_panel(ax, case_data{i}, 'pslr', configs(i).label);
    if i > 1
        ylabel(ax, '');
    end

    ax = axes(fig, 'Units', 'normalized', ...
        'Position', [axes_x(i) bottom_y axes_width axes_height]);
    panel_axes(2, i) = ax;
    plot_case_panel(ax, case_data{i}, 'islr', '');
    if i > 1
        ylabel(ax, '');
    end
end

plot_config(fig);
for i = 1:numel(panel_axes)
    pbaspect(panel_axes(i), [1 1 1]);
end
for row_idx = 1:2
    y_label_x = -0.120;
    if row_idx == 2
        y_label_x = -0.135;
    end
    set(panel_axes(row_idx, 1).YLabel, ...
        'Units', 'normalized', ...
        'Position', [y_label_x 0.500 0], ...
        'FontSize', cfg.label_font + 2, ...
        'FontWeight', 'bold');
end

annotation(fig, 'textbox', [0 0.001 * y_crop_scale 1 0.055 * y_crop_scale], ...
    'String', 'Sum-rate (bps/Hz)', ...
    'Interpreter', 'tex', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', ...
    'FitBoxToText', 'off', ...
    'FontName', cfg.font_name, ...
    'FontSize', cfg.label_font + 2, ...
    'FontWeight', 'bold');
draw_pareto_grid_legend(fig, ...
    [0.140 * x_crop_scale 0.855 * y_crop_scale ...
     0.720 * x_crop_scale 0.105 * y_crop_scale], cfg);

safe_export(fig, fullfile(paper_fig_dir, 'Pareto_Frontier_Grid_2x4.pdf'), 'pdf');
safe_export(fig, fullfile(paper_fig_dir, 'Pareto_Frontier_Grid_2x4.png'), 'png');
saveas(fig, fullfile(out_dir, 'pareto_grid_2x4.fig'));
fprintf('Saved 2x4 Pareto figure to %s\n', paper_fig_dir);
end

function S = load_or_run_case(cfg, cache_dir, num_mc_override)
params = setup_params();
params.NT = cfg.NT;
params.N = cfg.N;
params.P_des = 0.8 * params.P_max / params.N;
params.CV_max_list = [0, 0.1:0.1:0.9];
if ~isempty(num_mc_override)
    params.num_mc = num_mc_override;
end
[desired_crb_eta_list, desired_mi_eta_list] = surrogate_eta_lists(params);

cache_name = sprintf('pareto_grid_NT%d_N%d_MC%d.mat', cfg.NT, cfg.N, params.num_mc);
cache_path = fullfile(cache_dir, cache_name);
if exist(cache_path, 'file') == 2
    S = load(cache_path);
    S = ensure_exact_islr_fields(S);
    S = ensure_surrogate_eta_grid(S, cache_path, desired_crb_eta_list, desired_mi_eta_list);
    fprintf('Loaded cached case: %s\n', cache_path);
    return;
end

CV_max_list = params.CV_max_list;
num_cv = numel(CV_max_list);
num_mc = params.num_mc;

sumrate_grid = nan(num_cv, num_mc);
pslr_lin_grid = nan(num_cv, num_mc);
islr_lin_grid = nan(num_cv, num_mc);
direct_equiv_sumrate_grid = nan(num_cv, num_mc);
direct_equiv_pslr_lin_grid = nan(num_cv, num_mc);
direct_equiv_islr_lin_grid = nan(num_cv, num_mc);
direct_sumrate_grid = nan(num_cv, num_mc);
direct_pslr_lin_grid = nan(num_cv, num_mc);
direct_islr_lin_grid = nan(num_cv, num_mc);
direct_islr_exact_sumrate_grid = nan(num_cv, num_mc);
direct_islr_exact_pslr_lin_grid = nan(num_cv, num_mc);
direct_islr_exact_islr_lin_grid = nan(num_cv, num_mc);
direct_islr_exact_target_grid = nan(num_cv, num_mc);

comm_sumrate_grid = nan(1, num_mc);
comm_pslr_lin_grid = nan(1, num_mc);
comm_islr_lin_grid = nan(1, num_mc);

crb_eta_list = desired_crb_eta_list;
mi_eta_list = desired_mi_eta_list;
crb_sumrate_grid = nan(numel(crb_eta_list), num_mc);
crb_pslr_lin_grid = nan(numel(crb_eta_list), num_mc);
crb_islr_lin_grid = nan(numel(crb_eta_list), num_mc);
mi_sumrate_grid = nan(numel(mi_eta_list), num_mc);
mi_pslr_lin_grid = nan(numel(mi_eta_list), num_mc);
mi_islr_lin_grid = nan(numel(mi_eta_list), num_mc);

fprintf('Running Pareto grid case: N_T=%d, N=%d, MC=%d\n', params.NT, params.N, num_mc);
for mc = 1:num_mc
    rng(mc, 'twister');
    H = generate_channel(params);

    alpha_warm = [];
    for c = 1:num_cv
        CV_max = CV_max_list(c);
        result = run_proposed(H, CV_max, params, alpha_warm);
        if ~isnan(result.sumrate)
            alpha_warm = result.alpha;
            pslr_min_lin = min(result.pslr_per_target);
            islr_max_lin = max(result.islr_per_target);
            sumrate_grid(c, mc) = result.sumrate;
            pslr_lin_grid(c, mc) = pslr_min_lin;
            islr_lin_grid(c, mc) = islr_max_lin;

            direct_equiv_sumrate_grid(c, mc) = result.sumrate;
            direct_equiv_pslr_lin_grid(c, mc) = pslr_min_lin;
            direct_equiv_islr_lin_grid(c, mc) = islr_max_lin;

            % ISLR has an exact CV-SOC equivalent, so its direct Pareto
            % curve should coincide with the proposed CV curve. The SCA
            % direct run below is PSLR-active and is used only for PSLR.
            [~, islr_exact_max] = direct_thresholds_from_cv(CV_max, params, false);
            direct_islr_exact_target_grid(c, mc) = islr_exact_max;
            direct_islr_exact_sumrate_grid(c, mc) = result.sumrate;
            direct_islr_exact_pslr_lin_grid(c, mc) = pslr_min_lin;
            direct_islr_exact_islr_lin_grid(c, mc) = islr_max_lin;

            pslr_min = pslr_min_lin * (1 - params.direct_pslr_target_relax);
            islr_cv = CV_max + params.direct_pslr_active_islr_cv_gap;
            [~, islr_max] = direct_thresholds_from_cv(islr_cv, params);
            direct_result = run_direct_sca(H, pslr_min, islr_max, params, result.alpha, result.W);
            if ~isnan(direct_result.sumrate)
                direct_sumrate_grid(c, mc) = direct_result.sumrate;
                direct_pslr_lin_grid(c, mc) = min(direct_result.pslr_per_target);
                direct_islr_lin_grid(c, mc) = max(direct_result.islr_per_target);
            end
        else
            alpha_warm = [];
        end
        fprintf('  MC %02d/%02d CV %.1f: SR %.2f\n', mc, num_mc, CV_max, sumrate_grid(c, mc));
    end

    params_comm = params;
    params_comm.P_des = 0;
    comm_result = run_proposed(H, 1e3, params_comm);
    if ~isnan(comm_result.sumrate)
        comm_sumrate_grid(mc) = comm_result.sumrate;
        comm_pslr_lin_grid(mc) = min(comm_result.pslr_per_target);
        comm_islr_lin_grid(mc) = max(comm_result.islr_per_target);
    end

    crb_alpha = [];
    for e = 1:numel(crb_eta_list)
        crb_result = run_surrogate_baseline(H, 'crb', crb_eta_list(e), params, crb_alpha);
        if ~isnan(crb_result.sumrate)
            crb_alpha = crb_result.alpha;
            crb_sumrate_grid(e, mc) = crb_result.sumrate;
            crb_pslr_lin_grid(e, mc) = min(crb_result.pslr_per_target);
            crb_islr_lin_grid(e, mc) = max(crb_result.islr_per_target);
        end
    end

    mi_alpha = [];
    for e = 1:numel(mi_eta_list)
        mi_result = run_surrogate_baseline(H, 'mi', mi_eta_list(e), params, mi_alpha);
        if ~isnan(mi_result.sumrate)
            mi_alpha = mi_result.alpha;
            mi_sumrate_grid(e, mc) = mi_result.sumrate;
            mi_pslr_lin_grid(e, mc) = min(mi_result.pslr_per_target);
            mi_islr_lin_grid(e, mc) = max(mi_result.islr_per_target);
        end
    end

    save(cache_path, '-v7.3');
end

S = load(cache_path);
S = ensure_exact_islr_fields(S);
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

function S = ensure_surrogate_eta_grid(S, cache_path, desired_crb_eta_list, desired_mi_eta_list)
S = append_missing_surrogate_eta(S, cache_path, 'crb', desired_crb_eta_list);
S = append_missing_surrogate_eta(S, cache_path, 'mi', desired_mi_eta_list);
end

function S = append_missing_surrogate_eta(S, cache_path, mode, desired_eta_list)
eta_field = sprintf('%s_eta_list', mode);
sumrate_field = sprintf('%s_sumrate_grid', mode);
pslr_field = sprintf('%s_pslr_lin_grid', mode);
islr_field = sprintf('%s_islr_lin_grid', mode);

old_eta = S.(eta_field)(:).';
missing_eta = desired_eta_list(~arrayfun(@(e) any(abs(old_eta - e) <= max(1e-12, 1e-8*abs(e))), desired_eta_list));
if isempty(missing_eta)
    return;
end

params = S.params;
num_mc = size(S.sumrate_grid, 2);
new_sumrate = nan(numel(missing_eta), num_mc);
new_pslr = nan(numel(missing_eta), num_mc);
new_islr = nan(numel(missing_eta), num_mc);

fprintf('Augmenting %s samples for N_T=%d, N=%d: %d eta points x %d MC\n', ...
    upper(mode), params.NT, params.N, numel(missing_eta), num_mc);
for mc = 1:num_mc
    rng(mc, 'twister');
    H = generate_channel(params);
    alpha_warm = [];
    for e = 1:numel(missing_eta)
        result = run_surrogate_baseline(H, mode, missing_eta(e), params, alpha_warm);
        if ~isnan(result.sumrate)
            alpha_warm = result.alpha;
            new_sumrate(e, mc) = result.sumrate;
            new_pslr(e, mc) = min(result.pslr_per_target);
            new_islr(e, mc) = max(result.islr_per_target);
        else
            alpha_warm = [];
        end
        fprintf('  %s MC %02d/%02d eta %.4g: SR %.2f\n', ...
            upper(mode), mc, num_mc, missing_eta(e), new_sumrate(e, mc));
    end
end

all_eta = [old_eta missing_eta];
all_sumrate = [S.(sumrate_field); new_sumrate];
all_pslr = [S.(pslr_field); new_pslr];
all_islr = [S.(islr_field); new_islr];
[all_eta, order] = sort(all_eta);

S.(eta_field) = all_eta;
S.(sumrate_field) = all_sumrate(order, :);
S.(pslr_field) = all_pslr(order, :);
S.(islr_field) = all_islr(order, :);
save(cache_path, '-struct', 'S', '-v7.3');
end

function [handles, labels] = plot_case_panel(ax, S, metric, title_text)
axes(ax);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
cfg = plot_config();

S = ensure_exact_islr_fields(S);
sumrate_avg = mean(S.sumrate_grid, 2, 'omitnan');
direct_equiv_sumrate_avg = mean(S.direct_equiv_sumrate_grid, 2, 'omitnan');
crb_sumrate_avg = mean(S.crb_sumrate_grid, 2, 'omitnan');
mi_sumrate_avg = mean(S.mi_sumrate_grid, 2, 'omitnan');
comm_sumrate = mean(S.comm_sumrate_grid, 'omitnan');

if strcmpi(metric, 'pslr')
    direct_sumrate_avg = mean(S.direct_sumrate_grid, 2, 'omitnan');
    y_prop = 10*log10(mean(S.pslr_lin_grid, 2, 'omitnan'));
    y_direct = 10*log10(mean(S.direct_pslr_lin_grid, 2, 'omitnan'));
    y_equiv = 10*log10(mean(S.direct_equiv_pslr_lin_grid, 2, 'omitnan'));
    y_crb = 10*log10(mean(S.crb_pslr_lin_grid, 2, 'omitnan'));
    y_mi = 10*log10(mean(S.mi_pslr_lin_grid, 2, 'omitnan'));
    y_bound = 10*log10(1 + S.params.N/(S.params.kappa - 1));
    y_label = 'PSLR (dB)';
    y_floor = [];
else
    direct_sumrate_avg = mean(S.direct_islr_exact_sumrate_grid, 2, 'omitnan');
    y_prop = 10*log10(mean(S.islr_lin_grid, 2, 'omitnan'));
    y_direct = 10*log10(mean(S.direct_islr_exact_islr_lin_grid, 2, 'omitnan'));
    y_equiv = 10*log10(mean(S.direct_equiv_islr_lin_grid, 2, 'omitnan'));
    y_crb = 10*log10(mean(S.crb_islr_lin_grid, 2, 'omitnan'));
    y_mi = 10*log10(mean(S.mi_islr_lin_grid, 2, 'omitnan'));
    y_bound = 10*log10((S.params.N - 1)*(S.params.N + 2*S.params.kappa - 2) / ...
                       (2*(S.params.N + S.params.kappa - 1)));
    y_label = 'ISLR (dB)';
    y_floor = [];
end

valid_prop = isfinite(sumrate_avg) & isfinite(y_prop);
valid_direct = isfinite(direct_sumrate_avg) & isfinite(y_direct);
valid_equiv = isfinite(direct_equiv_sumrate_avg) & isfinite(y_equiv);
valid_crb = isfinite(crb_sumrate_avg) & isfinite(y_crb);
valid_mi = isfinite(mi_sumrate_avg) & isfinite(y_mi);

palette = paper_palette();
c_prop = palette(1, :);
c_direct = palette(2, :);
c_crb = palette(7, :);
c_mi = palette(4, :);
c_neutral = palette(8, :);
c_comm = [0.20 0.62 0.22];
c_sensing = palette(9, :);

h1 = plot(ax, sumrate_avg(valid_prop), y_prop(valid_prop), '--o', ...
    'Color', c_prop, 'MarkerFaceColor', c_prop, ...
    'LineWidth', 1.4, 'MarkerSize', cfg.marker_size + 3, ...
    'DisplayName', 'Proposed CV');
h2 = plot(ax, direct_sumrate_avg(valid_direct), y_direct(valid_direct), '--d', ...
    'Color', c_direct, 'MarkerFaceColor', c_direct, ...
    'LineWidth', 1.2, 'MarkerSize', cfg.marker_size + 3, ...
    'DisplayName', 'Direct SCA');
h3 = plot(ax, direct_equiv_sumrate_avg(valid_equiv), y_equiv(valid_equiv), '--o', ...
    'Color', c_neutral, 'MarkerFaceColor', c_neutral, ...
    'LineWidth', 1.2, 'MarkerSize', cfg.marker_size + 4, ...
    'DisplayName', 'ISLR-active');
h4 = plot(ax, crb_sumrate_avg(valid_crb), y_crb(valid_crb), '--^', ...
    'Color', c_crb, 'MarkerFaceColor', c_crb, ...
    'LineWidth', 1.1, 'MarkerSize', cfg.marker_size + 3, ...
    'DisplayName', 'CRB-inspired');
h5 = plot(ax, mi_sumrate_avg(valid_mi), y_mi(valid_mi), '--v', ...
    'Color', c_mi, 'MarkerFaceColor', c_mi, ...
    'LineWidth', 1.3, 'MarkerSize', cfg.marker_size + 3, ...
    'DisplayName', 'MI-inspired');
h6 = xline(ax, comm_sumrate, '--', 'Color', c_comm, 'LineWidth', 1.4, ...
    'DisplayName', 'Communication-only');
h7 = yline(ax, y_bound, '--', 'Color', c_sensing, 'LineWidth', 1.4, ...
    'DisplayName', 'sensing-only');

pareto_x = [sumrate_avg(valid_prop); direct_sumrate_avg(valid_direct)];
pareto_y = [y_prop(valid_prop); y_direct(valid_direct)];
h_comm_star = gobjects(0);
h_sensing_star = gobjects(0);
if ~isempty(pareto_x)
    [~, comm_star_idx] = min(abs(pareto_x - comm_sumrate));
    [~, sensing_star_idx] = min(abs(pareto_y - y_bound));
    h_comm_star = plot(ax, pareto_x(comm_star_idx), pareto_y(comm_star_idx), 'p', ...
        'LineStyle', 'none', ...
        'Color', c_comm, ...
        'MarkerFaceColor', c_comm, ...
        'MarkerSize', cfg.marker_size + 15, ...
        'LineWidth', 2.0, ...
        'HandleVisibility', 'off');
    h_sensing_star = plot(ax, pareto_x(sensing_star_idx), pareto_y(sensing_star_idx), 'p', ...
        'LineStyle', 'none', ...
        'Color', c_sensing, ...
        'MarkerFaceColor', c_sensing, ...
        'MarkerSize', cfg.marker_size + 15, ...
        'LineWidth', 2.0, ...
        'HandleVisibility', 'off');
end

x_focus = [sumrate_avg(valid_prop); direct_sumrate_avg(valid_direct); ...
           direct_equiv_sumrate_avg(valid_equiv); comm_sumrate];
y_focus = [y_prop(valid_prop); y_direct(valid_direct); ...
           y_equiv(valid_equiv); y_bound];
[xl, yl] = corner_limits(x_focus, y_focus, metric);
if ~isempty(y_floor), yl(1) = y_floor; end
xlim(ax, xl); ylim(ax, yl);
if strcmpi(metric, 'islr')
    y_tick_min = ceil(10 * yl(1) - 1e-9) / 10;
    y_tick_max = floor(10 * yl(2) + 1e-9) / 10;
    yticks(ax, y_tick_min:0.1:y_tick_max);
    ytickformat(ax, '%.1f');
end
add_enclosed_region_shade(ax, metric, ...
    {sumrate_avg(valid_prop), direct_sumrate_avg(valid_direct)}, ...
    {y_prop(valid_prop), y_direct(valid_direct)}, ...
    comm_sumrate, y_bound);
try
    % Keep the verified Proposed-CV red curve above overlapping gray/blue
    % points, then restore the two Pareto-boundary stars to the top.
    uistack(h2, 'top');
    uistack(h1, 'top');
    if isgraphics(h_comm_star), uistack(h_comm_star, 'top'); end
    if isgraphics(h_sensing_star), uistack(h_sensing_star, 'top'); end
catch
end

ylabel(ax, y_label, 'FontSize', cfg.label_font - 2);
set(ax, 'FontSize', cfg.axes_font, 'Layer', 'top');
try
    ax.TitleFontSizeMultiplier = 1;
catch
end
if ~isempty(title_text)
    title_handle = title(ax, title_text, 'Interpreter', 'latex', ...
        'FontSize', cfg.title_font, 'FontWeight', 'bold');
    title_handle.FontSize = cfg.title_font;
end
handles = [h1 h2 h3 h4 h5 h6 h7];
labels = get(handles, 'DisplayName');
end

function add_enclosed_region_shade(ax, metric, x_cells, y_cells, comm_x, bound_y)
palette = paper_palette();
if strcmpi(metric, 'pslr')
    shade_color = palette(1, :);
else
    shade_color = palette(2, :);
end

[frontier_x, frontier_y] = outer_frontier(metric, x_cells, y_cells);
if numel(frontier_x) < 2 || ~isfinite(comm_x) || ~isfinite(bound_y)
    return;
end

frontier_x = frontier_x(:);
frontier_y = frontier_y(:);
valid = frontier_x <= comm_x & isfinite(frontier_x) & isfinite(frontier_y);
frontier_x = frontier_x(valid);
frontier_y = frontier_y(valid);
if numel(frontier_x) < 2
    return;
end

axis_x = xlim(ax);
axis_y = ylim(ax);
if strcmpi(metric, 'pslr')
    inside_y = axis_y(1);
else
    inside_y = axis_y(2);
end

left_x = axis_x(1);
poly_x = [left_x; frontier_x; comm_x; comm_x; left_x];
poly_y = [bound_y; frontier_y; frontier_y(end); inside_y; inside_y];
region = patch(ax, poly_x, poly_y, shade_color, ...
    'FaceAlpha', 0.10, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
try
    uistack(region, 'bottom');
catch
end
end

function draw_pareto_grid_legend(fig, position, cfg)
palette = paper_palette();
c_prop = palette(1, :);
c_direct = palette(2, :);
c_crb = palette(7, :);
c_mi = palette(4, :);
c_neutral = palette(8, :);
c_comm = [0.20 0.62 0.22];
c_sensing = palette(9, :);

ax_leg = axes(fig, 'Position', position, ...
    'XLim', [0 1], 'YLim', [0 1], ...
    'Visible', 'off', ...
    'Color', 'none');
hold(ax_leg, 'on');
rectangle(ax_leg, 'Position', [0.015 0.015 0.970 0.970], ...
    'FaceColor', 'w', ...
    'EdgeColor', [0.15 0.15 0.15], ...
    'LineWidth', cfg.axes_line_width);

legend_font = max(cfg.legend_font + 2, 1);
legend_marker_size = max(1.85 * cfg.marker_size, 17);
legend_star_size = max(3.00 * cfg.marker_size, 27);
legend_line_width = max(cfg.line_width, 2.2);
labels = {'Proposed CV', 'Direct SCA', 'ISLR-active', ...
          'CRB-inspired', 'MI-inspired', ...
          'sensing-only', 'Communication-only'};
line_styles = repmat({'--'}, 1, numel(labels));
markers = {'o', 'd', 'o', '^', 'v', 'p', 'p'};
colors = [c_prop; c_direct; c_neutral; c_crb; c_mi; c_sensing; c_comm];
num_columns = 4;
row_y = [0.640 0.360];
content_left = 0.014;
content_width = 0.972;
column_width = content_width / num_columns;
column_indices = [0 1 2 3 0 1 2];
row_indices = [1 1 1 1 2 2 2];

for idx = 1:numel(labels)
    row_idx = row_indices(idx);
    column_idx = column_indices(idx);
    cell_left = content_left + column_idx * column_width;
    y = row_y(row_idx);
    x_line = cell_left + [0.006 0.046];
    x_marker = mean(x_line);
    x_text = cell_left + 0.055;
    style = line_styles{idx};
    marker = markers{idx};
    marker_size = legend_marker_size;
    if idx >= 6
        marker_size = legend_star_size;
    end
    if strcmp(marker, 'none')
        plot(ax_leg, x_line, [y y], style, ...
            'Color', colors(idx, :), ...
            'LineWidth', legend_line_width, ...
            'Clipping', 'off');
    elseif strcmp(style, 'none')
        scatter(ax_leg, x_marker, y, marker_size^2, ...
            'Marker', marker, ...
            'MarkerEdgeColor', colors(idx, :), ...
            'MarkerFaceColor', colors(idx, :), ...
            'LineWidth', 1.5, ...
            'Clipping', 'off');
    else
        plot(ax_leg, x_line, [y y], style, ...
            'Color', colors(idx, :), ...
            'LineWidth', legend_line_width, ...
            'Clipping', 'off');
        scatter(ax_leg, x_marker, y, marker_size^2, ...
            'Marker', marker, ...
            'MarkerEdgeColor', colors(idx, :), ...
            'MarkerFaceColor', colors(idx, :), ...
            'LineWidth', 1.5, ...
            'Clipping', 'off');
    end
    text(ax_leg, x_text, y, labels{idx}, ...
        'Interpreter', 'tex', ...
        'FontName', cfg.font_name, ...
        'FontSize', legend_font, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
end
try
    uistack(ax_leg, 'top');
catch
end
end

function [frontier_x, frontier_y] = outer_frontier(metric, x_cells, y_cells)
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
    if strcmpi(metric, 'pslr')
        y_unique = accumarray(ic, y, [], @max);
    else
        y_unique = accumarray(ic, y, [], @min);
    end
    if numel(x_unique) >= 2
        in_range = frontier_x >= min(x_unique) & frontier_x <= max(x_unique);
        Y(in_range, i) = interp1(x_unique, y_unique, frontier_x(in_range), 'linear');
    end
end

if strcmpi(metric, 'pslr')
    frontier_y = max(Y, [], 2, 'omitnan');
else
    frontier_y = min(Y, [], 2, 'omitnan');
end
valid_frontier = isfinite(frontier_y);
frontier_x = frontier_x(valid_frontier);
frontier_y = frontier_y(valid_frontier);
end

function S = ensure_exact_islr_fields(S)
if ~isfield(S, 'direct_islr_exact_sumrate_grid')
    S.direct_islr_exact_sumrate_grid = S.sumrate_grid;
end
if ~isfield(S, 'direct_islr_exact_pslr_lin_grid')
    S.direct_islr_exact_pslr_lin_grid = S.pslr_lin_grid;
end
if ~isfield(S, 'direct_islr_exact_islr_lin_grid')
    S.direct_islr_exact_islr_lin_grid = S.islr_lin_grid;
end
if ~isfield(S, 'direct_islr_exact_target_grid')
    S.direct_islr_exact_target_grid = nan(size(S.islr_lin_grid));
    for i = 1:numel(S.params.CV_max_list)
        [~, islr_exact_max] = direct_thresholds_from_cv(S.params.CV_max_list(i), S.params, false);
        S.direct_islr_exact_target_grid(i, :) = islr_exact_max;
    end
end
end

function [xl, yl] = corner_limits(x, y, metric)
x = x(isfinite(x)); y = y(isfinite(y));
xs = max(x) - min(x); if xs <= 0, xs = max(abs(x(1))*0.05, 1); end
ys = max(y) - min(y); if ys <= 0, ys = max(abs(y(1))*0.05, 1); end
target_fraction = 0.40;
outside_pad = 0.24;

% Keep the Pareto boundary close to the right-side communication limit.
x_high = max(x) + outside_pad * xs;
xl = [x_high - xs / target_fraction, x_high];

if strcmpi(metric, 'pslr')
    % PSLR is upper-bounded, so keep the curve near the upper-right corner.
    y_high = max(y) + outside_pad * ys;
    yl = [y_high - ys / target_fraction, y_high];
else
    % ISLR is lower-bounded, so keep the curve near the lower-right corner.
    y_low = min(y) - outside_pad * ys;
    yl = [y_low, y_low + ys / target_fraction];
end
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        % Rasterize the PDF to avoid missing-font glyph boxes in LaTeX/Poppler.
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
