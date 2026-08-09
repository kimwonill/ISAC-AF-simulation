function plot_cv_theory_bounds(force_rerun)
% PLOT_CV_THEORY_BOUNDS  Analytic CV-to-PSLR curves used in Sec. III.
% This figure visualizes the closed-form PSLR bounds. If results.mat
% exists, the simulation PSLR range is shown using the per-CV min-max
% envelope.

if nargin < 1 || isempty(force_rerun)
    force_rerun = false;
end

repo_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(repo_dir, 'figures');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

addpath(repo_dir);

params = setup_params();
N = params.N;
kappa = params.kappa;
cv = linspace(0, 1.0, 501);
marker_idx = 1:50:numel(cv);

U = (kappa - 1) ./ (kappa - 1 + N/(N-1)) + ...
    (N^2*kappa / ((N-1)*(kappa - 1 + N/(N-1)))) ./ ...
    ((kappa - 1 + N/(N-1))*cv.^2 + kappa - 1);
L = (kappa - 1) ./ (N + kappa - 1) + ...
    (N*(N + 2*kappa - 2)/(N + kappa - 1)) ./ ...
    ((N + kappa - 1)*cv.^2 + kappa - 1);

[sim_cv, sim_pslr_dB, sim_cv_list, sim_pslr_min_dB, sim_pslr_max_dB] = ...
    load_or_generate_simulation_overlay(repo_dir, params, force_rerun);

cfg = plot_config();
export_resolution = cfg.export_resolution;
plot_style = struct( ...
    'figure_position', [100 100 1040 560], ...
    'axes_position', [0.112 0.300 0.818 0.520], ...
    'legend_position', [0.135 0.325 0.220 0.150], ...
    'axes_font', cfg.axes_font, ...
    'label_font', cfg.label_font, ...
    'title_font', cfg.title_font, ...
    'panel_caption_font', cfg.panel_caption_font, ...
    'legend_font', cfg.legend_font, ...
    'axes_line_width', cfg.axes_line_width);

fig = figure('Color', 'w', 'Position', plot_style.figure_position);
set(fig, 'PaperPositionMode', 'auto');

ax1 = axes(fig, 'Position', plot_style.axes_position);
hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
palette = paper_palette();
upper_color = palette(1, :);
lower_color = palette(2, :);
sim_color = palette(3, :);
has_sim_range = ~isempty(sim_cv_list);
if has_sim_range
    fill(ax1, ...
        [sim_cv_list, fliplr(sim_cv_list)], ...
        [sim_pslr_max_dB, fliplr(sim_pslr_min_dB)], ...
        sim_color, 'FaceAlpha', cfg.band_face_alpha, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
end
plot(ax1, cv, 10*log10(U), '-d', ...
     'LineWidth', cfg.line_width, 'MarkerIndices', marker_idx, ...
     'MarkerSize', cfg.marker_size, 'MarkerFaceColor', upper_color, ...
     'Color', upper_color, ...
     'DisplayName', '$\mathcal{U}(\mathrm{CV})$');
plot(ax1, cv, 10*log10(L), '-d', ...
     'LineWidth', cfg.line_width, 'MarkerIndices', marker_idx, ...
     'MarkerSize', cfg.marker_size, 'MarkerFaceColor', lower_color, ...
     'Color', lower_color, ...
     'DisplayName', '$\mathcal{L}(\mathrm{CV})$');
if ~isempty(sim_cv)
    scatter(ax1, sim_cv, sim_pslr_dB, cfg.simulation_scatter_size, 'x', ...
            'MarkerEdgeColor', sim_color, ...
            'LineWidth', cfg.marker_edge_width, ...
            'DisplayName', 'Simulation');
else
    plot(ax1, nan, nan, 'x', ...
        'MarkerSize', cfg.simulation_marker_size, ...
        'Color', sim_color, ...
        'LineWidth', cfg.marker_edge_width, ...
        'DisplayName', 'Simulation');
end
xlabel(ax1, 'CV', 'FontSize', plot_style.label_font);
ylabel(ax1, 'PSLR (dB)', 'FontSize', plot_style.label_font);
set(ax1.XLabel, 'Units', 'normalized', 'Position', [0.5 -0.125 0]);
set(ax1.YLabel, 'Units', 'normalized', 'Position', [-0.105 0.5 0]);
all_pslr_dB = [10*log10(U(:)); 10*log10(L(:)); ...
    sim_pslr_min_dB(:); sim_pslr_max_dB(:)];
set(ax1, 'FontSize', plot_style.axes_font, 'XLim', [0 1], ...
         'XTick', 0:0.5:1, ...
         'XTickLabel', {'0', '0.5', '1.0'}, ...
         'YLim', valid_axis_limits(all_pslr_dB, cfg, 'Clip', [0 Inf]), ...
         'LabelFontSizeMultiplier', 1);

drawnow;
plot_config(fig);
set(ax1.XLabel, 'Units', 'normalized', 'Position', [0.5 -0.125 0]);
set(ax1.YLabel, 'Units', 'normalized', 'Position', [-0.105 0.5 0]);
draw_pslr_legend(fig, plot_style.legend_position, ...
                 upper_color, lower_color, sim_color, palette(10, :), ...
                 plot_style, cfg);
tight_export_figure(fig, fullfile(out_dir, 'cv_pslr_theory.pdf'), ...
                    'ContentType', 'image', ...
                    'Resolution', export_resolution, ...
                    'TightPad', 0);
tight_export_figure(fig, fullfile(out_dir, 'cv_pslr_theory.png'), ...
                    'Resolution', export_resolution, ...
                    'TightPad', 0);
end

function [sim_cv, sim_pslr_dB, sim_cv_list, sim_pslr_min_dB, sim_pslr_max_dB] = ...
    load_or_generate_simulation_overlay(repo_dir, params, force_rerun)
sim_cv = [];
sim_pslr_dB = [];
sim_cv_list = [];
sim_pslr_min_dB = [];
sim_pslr_max_dB = [];

result_candidates = { ...
    fullfile(repo_dir, 'results', 'pareto_grid_1x4_pslr', ...
        'pareto_pslr_NT4_N16_MC100.mat'), ...
    fullfile(repo_dir, 'results.mat'), ...
    fullfile(repo_dir, 'results', 'fig4_fig5_full_results.mat')};

for idx = 1:numel(result_candidates)
    result_path = result_candidates{idx};
    if exist(result_path, 'file') ~= 2
        continue;
    end
    S = load(result_path, 'CV_max_list', 'pslr_lin_grid');
    if isfield(S, 'CV_max_list') && isfield(S, 'pslr_lin_grid')
        [sim_cv, sim_pslr_dB, sim_cv_list, sim_pslr_min_dB, sim_pslr_max_dB] = ...
            unpack_simulation_grid(S.CV_max_list, S.pslr_lin_grid);
        return;
    end
end

cache_dir = fullfile(repo_dir, 'results');
if exist(cache_dir, 'dir') ~= 7
    mkdir(cache_dir);
end
cache_path = fullfile(cache_dir, 'cv_pslr_theory_profile_simulation.mat');
if exist(cache_path, 'file') == 2 && ~force_rerun
    S = load(cache_path, 'CV_max_list', 'pslr_lin_grid');
else
    S = generate_profile_simulation_grid(params);
    save(cache_path, '-struct', 'S');
end

[sim_cv, sim_pslr_dB, sim_cv_list, sim_pslr_min_dB, sim_pslr_max_dB] = ...
    unpack_simulation_grid(S.CV_max_list, S.pslr_lin_grid);
end

function [sim_cv, sim_pslr_dB, sim_cv_list, sim_pslr_min_dB, sim_pslr_max_dB] = ...
    unpack_simulation_grid(CV_max_list, pslr_lin_grid)
sim_cv_list = CV_max_list(:).';
pslr_dB_grid = 10*log10(pslr_lin_grid);
sim_pslr_min_dB = min(pslr_dB_grid, [], 2, 'omitnan').';
sim_pslr_max_dB = max(pslr_dB_grid, [], 2, 'omitnan').';
valid_range = isfinite(sim_cv_list) & ...
    isfinite(sim_pslr_min_dB) & isfinite(sim_pslr_max_dB);
sim_cv_list = sim_cv_list(valid_range);
sim_pslr_min_dB = sim_pslr_min_dB(valid_range);
sim_pslr_max_dB = sim_pslr_max_dB(valid_range);
[cv_grid, ~] = ndgrid(CV_max_list(:), 1:size(pslr_dB_grid, 2));
sim_cv = cv_grid(:);
sim_pslr_dB = pslr_dB_grid(:);
valid_samples = isfinite(sim_cv) & isfinite(sim_pslr_dB);
sim_cv = sim_cv(valid_samples);
sim_pslr_dB = sim_pslr_dB(valid_samples);
end

function S = generate_profile_simulation_grid(params)
CV_max_list = 0:0.1:1.0;
num_samples_per_cv = 100;
pslr_lin_grid = nan(numel(CV_max_list), num_samples_per_cv);
rng_seed = 23;
rng(rng_seed, 'twister');

for i = 1:numel(CV_max_list)
    P = sample_exact_cv_profiles(params.N, CV_max_list(i), num_samples_per_cv);
    pslr_lin_grid(i, :) = compute_pslr_many(P, params.kappa);
end

S = struct( ...
    'CV_max_list', CV_max_list, ...
    'pslr_lin_grid', pslr_lin_grid, ...
    'num_samples_per_cv', num_samples_per_cv, ...
    'rng_seed', rng_seed, ...
    'params', params, ...
    'source', 'fixed-CV nonnegative directional-power profile simulation');
end

function P_samples = sample_exact_cv_profiles(N, target_cv, num_samples)
if target_cv <= 1e-12
    P_samples = ones(N, num_samples);
    return;
end

P_samples = zeros(N, num_samples);
num_kept = 0;
num_drawn = 0;
batch_size = max(2000, 100 * num_samples);
max_drawn = 2e6;

while num_kept < num_samples
    E = -log(max(rand(N, batch_size), realmin));
    Q = N * E ./ sum(E, 1);
    q_cv = sqrt(mean((Q - 1).^2, 1));
    valid = q_cv >= target_cv;
    Q = Q(:, valid);
    q_cv = q_cv(valid);

    if ~isempty(Q)
        alpha = target_cv ./ q_cv;
        P = 1 + (Q - 1) .* alpha;
        take = min(size(P, 2), num_samples - num_kept);
        P_samples(:, num_kept + (1:take)) = P(:, 1:take);
        num_kept = num_kept + take;
    end

    num_drawn = num_drawn + batch_size;
    if num_drawn > max_drawn
        error('Could not sample enough profiles at CV %.3f.', target_cv);
    end
end
end

function pslr = compute_pslr_many(P, kappa)
P_dft = fft(P, [], 1);
sl_max = max(abs(P_dft(2:end, :)).^2, [], 1);
sq = sum(P.^2, 1);
sum_p = sum(P, 1);
mainlobe = (kappa - 1) * sq + sum_p.^2;
denom = (kappa - 1) * sq + sl_max;
pslr = mainlobe ./ denom;
end

function draw_pslr_legend(fig, position, upper_color, lower_color, sim_color, ~, style, cfg)
ax_leg = axes(fig, 'Position', position, ...
    'XLim', [0 1], 'YLim', [0 1], ...
    'Visible', 'off', ...
    'Color', 'none');
hold(ax_leg, 'on');
rectangle(ax_leg, 'Position', [0.020 0.020 0.960 0.960], ...
    'FaceColor', cfg.legend_background_color, ...
    'FaceAlpha', cfg.legend_face_alpha, ...
    'EdgeColor', cfg.legend_edge_color, ...
    'LineWidth', style.axes_line_width);

legend_font = max(style.legend_font - 7, 1);
y_pos = [0.755 0.500 0.245];
labels = {'$\mathcal{U}(\mathrm{CV})$', '$\mathcal{L}(\mathrm{CV})$', 'Simulation'};
line_colors = [upper_color; lower_color; sim_color];
for idx = 1:numel(labels)
    if idx < 3
        plot(ax_leg, [0.070 0.225], [y_pos(idx) y_pos(idx)], '-', ...
            'Color', line_colors(idx, :), ...
            'LineWidth', cfg.secondary_line_width, ...
            'Clipping', 'off');
        x_center = 0.1475;
        dx = 0.020;
        dy = 0.047;
        patch(ax_leg, x_center + [0 dx 0 -dx], y_pos(idx) + [dy 0 -dy 0], ...
            line_colors(idx, :), ...
            'EdgeColor', line_colors(idx, :), ...
            'LineWidth', 1.2, ...
            'Clipping', 'off');
    else
        plot(ax_leg, 0.1475, y_pos(idx), 'x', ...
            'Color', sim_color, ...
            'MarkerSize', cfg.simulation_marker_size, ...
            'LineWidth', cfg.marker_edge_width);
    end
    if idx < 3
        label_y = y_pos(idx) + 0.010;
    else
        label_y = y_pos(idx) + 0.040;
    end
    text(ax_leg, 0.290, label_y, labels{idx}, ...
        'Interpreter', 'latex', ...
        'FontName', cfg.font_name, ...
        'FontSize', legend_font, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
end
try
    uistack(ax_leg, 'top');
catch
end
end
