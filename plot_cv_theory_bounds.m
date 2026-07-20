function plot_cv_theory_bounds()
% PLOT_CV_THEORY_BOUNDS  Analytic CV-to-PSLR/ISLR curves used in Sec. III.
% This figure visualizes the closed-form PSLR bounds and exact ISLR
% expression. If results.mat exists, the simulation PSLR range is shown
% using the per-CV min-max envelope.

root_dir = fileparts(fileparts(mfilename('fullpath')));
out_dir = fullfile(root_dir, 'figures');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

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

islr = ((N*(2*kappa - 1) - 2*(kappa - 1))*cv.^2 + ...
        (N - 1)*(N + 2*kappa - 2)) ./ ...
       (2*((kappa - 1)*cv.^2 + N + kappa - 1));

sim_cv = [];
sim_pslr_dB = [];
sim_cv_list = [];
sim_pslr_min_dB = [];
sim_pslr_max_dB = [];
results_path = fullfile(fileparts(mfilename('fullpath')), 'results.mat');
if exist(results_path, 'file') == 2
    S = load(results_path, 'CV_max_list', 'pslr_lin_grid');
    if isfield(S, 'CV_max_list') && isfield(S, 'pslr_lin_grid')
        sim_cv_list = S.CV_max_list(:).';
        pslr_dB_grid = 10*log10(S.pslr_lin_grid);
        sim_pslr_min_dB = min(pslr_dB_grid, [], 2, 'omitnan').';
        sim_pslr_max_dB = max(pslr_dB_grid, [], 2, 'omitnan').';
        valid_range = isfinite(sim_cv_list) & ...
            isfinite(sim_pslr_min_dB) & isfinite(sim_pslr_max_dB);
        sim_cv_list = sim_cv_list(valid_range);
        sim_pslr_min_dB = sim_pslr_min_dB(valid_range);
        sim_pslr_max_dB = sim_pslr_max_dB(valid_range);

        [cv_grid, ~] = ndgrid(S.CV_max_list(:), 1:size(S.pslr_lin_grid, 2));
        sim_cv = cv_grid(:);
        sim_pslr_dB = pslr_dB_grid(:);
        valid_sim = isfinite(sim_cv) & isfinite(sim_pslr_dB);
        sim_cv = sim_cv(valid_sim);
        sim_pslr_dB = sim_pslr_dB(valid_sim);
    end
end

export_resolution = 1200;
plot_style = struct( ...
    'figure_position', [100 100 760 520], ...
    'left_axes_position', [0.095 0.180 0.385 0.640], ...
    'right_axes_position', [0.610 0.180 0.365 0.640], ...
    'axes_font', 15, ...
    'label_font', 19, ...
    'title_font', 20, ...
    'legend_font', 16.5);

fig = figure('Color', 'w', 'Position', plot_style.figure_position);
set(fig, 'PaperPositionMode', 'auto');

ax1 = axes(fig, 'Position', plot_style.left_axes_position);
hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
palette = paper_palette();
upper_color = palette(1, :);
lower_color = palette(2, :);
sim_color = palette(3, :);
islr_color = palette(4, :);
has_sim_range = ~isempty(sim_cv_list);
if has_sim_range
    fill(ax1, ...
        [sim_cv_list, fliplr(sim_cv_list)], ...
        [sim_pslr_max_dB, fliplr(sim_pslr_min_dB)], ...
        sim_color, 'FaceAlpha', 0.16, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
end
h_upper = plot(ax1, cv, 10*log10(U), '-d', ...
               'LineWidth', 2.2, 'MarkerIndices', marker_idx, ...
               'MarkerSize', 7.0, 'MarkerFaceColor', upper_color, ...
               'Color', upper_color);
h_lower = plot(ax1, cv, 10*log10(L), '-d', ...
               'LineWidth', 2.2, 'MarkerIndices', marker_idx, ...
               'MarkerSize', 7.0, 'MarkerFaceColor', lower_color, ...
               'Color', lower_color);
if ~isempty(sim_cv)
    h_sim = scatter(ax1, sim_cv, sim_pslr_dB, 34, ...
                    'MarkerEdgeColor', palette(10, :), ...
                    'MarkerFaceColor', sim_color, ...
                    'LineWidth', 0.7);
end
if ~isempty(sim_cv)
    legend(ax1, [h_upper h_lower h_sim], ...
           {'Upper', 'Lower', 'Simulation'}, ...
           'Location', 'southwest', 'FontSize', plot_style.legend_font);
else
    legend(ax1, [h_upper h_lower], {'Upper', 'Lower'}, ...
           'Location', 'southwest', 'FontSize', plot_style.legend_font);
end
xlabel(ax1, 'CV', 'FontSize', plot_style.label_font);
ylabel(ax1, 'PSLR (dB)', 'FontSize', plot_style.label_font);
title(ax1, '(a) PSLR bounds', 'FontSize', plot_style.title_font);
set(ax1, 'FontSize', plot_style.axes_font, 'XLim', [-0.02 1.02], ...
         'XTick', 0:0.5:1, ...
         'LabelFontSizeMultiplier', 1);

ax2 = axes(fig, 'Position', plot_style.right_axes_position);
plot(ax2, cv, 10*log10(islr), '-d', ...
     'LineWidth', 2.2, 'MarkerIndices', marker_idx, ...
     'MarkerSize', 7.0, 'MarkerFaceColor', islr_color, ...
     'Color', islr_color);
grid(ax2, 'on'); box(ax2, 'on');
xlabel(ax2, 'CV', 'FontSize', plot_style.label_font);
ylabel(ax2, 'ISLR (dB)', 'FontSize', plot_style.label_font);
title(ax2, '(b) Exact ISLR', 'FontSize', plot_style.title_font);
set(ax2, 'FontSize', plot_style.axes_font, 'XLim', [-0.02 1.02], ...
         'XTick', 0:0.5:1, ...
         'LabelFontSizeMultiplier', 1);

drawnow;
exportgraphics(fig, fullfile(out_dir, 'cv_pslr_islr_theory.pdf'), ...
               'ContentType', 'image', 'Resolution', export_resolution);
exportgraphics(fig, fullfile(out_dir, 'cv_pslr_islr_theory.png'), ...
               'Resolution', export_resolution);
end
