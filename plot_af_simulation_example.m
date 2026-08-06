function plot_af_simulation_example()
% PLOT_AF_SIMULATION_EXAMPLE  Fig. 2 AF example from one simulation run.
%
% The script runs the proposed design for one deterministic channel
% realization, extracts P_n(theta) from the optimized covariance, and
% evaluates the closed-form ESL in Lemma 1.

clearvars; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
data_path = fullfile(data_dir, 'fig2_af_simulation_example.mat');

if exist(fig_dir, 'dir') ~= 7
    mkdir(fig_dir);
end
if exist(data_dir, 'dir') ~= 7
    mkdir(data_dir);
end

addpath(genpath(sim_dir));

if exist(data_path, 'file') == 2
    S = load(data_path, 'ESL', 'ESL_dB');
    ESL = S.ESL;
    ESL_dB = S.ESL_dB;
else
    if exist('cvx_begin', 'file') ~= 2
        error('CVX is required. Install CVX and run cvx_setup first.');
    end

    params = setup_params();
    CV_max = 0.5;
    rng_seed = 1;
    rng(rng_seed, 'twister');

    H = generate_channel(params);
    result = run_proposed(H, CV_max, params);
    if isempty(result.W) || isnan(result.sumrate)
        error('The proposed design failed for the Fig. 2 example: %s', result.status);
    end

    A = compute_steering(params);
    [~, target_idx] = min(abs(rad2deg(params.theta)));
    target_deg = rad2deg(params.theta(target_idx));
    P = compute_directional_power(result.W, A(:, target_idx));

    [ESL, ESL_dB] = closed_form_esl(P, params.kappa);

    save(data_path, 'A', 'CV_max', 'ESL', 'ESL_dB', 'H', 'P', ...
         'params', 'result', 'rng_seed', 'target_deg', 'target_idx');
end

plot_combined(ESL, ESL_dB, fig_dir, data_dir);

fprintf('Using Fig. 2 data: %s\n', data_path);
fprintf('Updated Fig. 2 files in: %s\n', fig_dir);
end

function [ESL, ESL_dB] = closed_form_esl(P, kappa)
N = numel(P);
P = P(:);
ESL = zeros(N, N);
sumP2 = sum(P.^2);
subcarrier_idx = (0:N-1).';

for tau = 0:N-1
    phasor = exp(-1j * 2*pi * subcarrier_idx * tau / N);
    bracket = abs(P.' * phasor)^2 + (kappa - 2) * sumP2;
    ESL(tau+1, 1) = N^2 * bracket + N^2 * sumP2;
end

for nu = 1:N-1
    cc = sum(P(nu+1:N) .* P(1:N-nu));
    ESL(:, nu+1) = N^2 * cc;
end

ESL = max(real(ESL), realmin);
ESL_dB = 10 * log10(ESL / max(ESL(:)));
end

function plot_combined(ESL, ESL_dB, fig_dir, data_dir)
N = size(ESL, 1);
tau_axis = 0:N-1;
nu_axis = 0:N-1;
ESL_plot_dB = ESL_dB;
nu_list = [0 1 2 4 6 8 12 N-1];
nu_list = unique(nu_list(nu_list <= N-1), 'stable');
tau_plot = 0:N-1;

cfg = plot_config();
style = combined_figure_style(cfg);
colors = muted_colors(numel(nu_list));
fig = figure('Color', 'w', 'Position', style.figure_position);
set(fig, 'PaperPositionMode', 'auto');

ax1 = axes(fig, 'Position', style.left_axes_position);
imagesc(ax1, tau_axis, nu_axis, max(ESL_plot_dB.', style.contour_floor_dB));
axis(ax1, 'xy');
axis(ax1, 'tight');
pbaspect(ax1, [1 1 1]);
box(ax1, 'on');
colormap(ax1, high_contrast_af_colormap(256));
clim(ax1, [style.contour_floor_dB 0]);
ylabel_handle1 = ylabel(ax1, 'Doppler index \nu', 'Interpreter', 'tex', ...
                        'FontSize', style.label_font, ...
                        'FontWeight', 'normal');
set(ylabel_handle1, 'Units', 'normalized', 'Position', [-0.140 0.5 0]);
set(ax1, 'FontSize', style.axes_font, 'TickLabelInterpreter', 'tex', ...
         'LabelFontSizeMultiplier', 1, 'Layer', 'top', ...
         'LineWidth', style.axes_line_width);
set(ax1, 'XTick', 0:5:15, 'XTickLabel', []);

cb = colorbar(ax1);
cb.Position = style.colorbar_position;
cb.Label.String = '';
cb.Title.String = '';
cb.Title.Interpreter = 'tex';
cb.Title.FontSize = style.colorbar_label_font;
cb.FontSize = style.colorbar_font;
cb.TickLabelInterpreter = 'tex';
cb.Ticks = style.colorbar_ticks;
cb.TickLabels = repmat({''}, size(style.colorbar_ticks));
try
    cb.AxisLocation = 'out';
    cb.TickDirection = 'out';
catch
end
draw_colorbar_title(fig, cb, 'AF (dB)', style);
draw_colorbar_tick_labels(fig, cb, style.colorbar_ticks, style);

ax2 = axes(fig, 'Position', style.right_axes_position);
hold(ax2, 'on');
grid(ax2, 'on');
box(ax2, 'on');
for idx = 1:numel(nu_list)
    nu = nu_list(idx);
    if nu == 0
        line_width = style.main_line_width;
    else
        line_width = style.line_width;
    end
    plot(ax2, tau_plot, ESL_dB(:, nu+1), '-', ...
         'LineWidth', line_width, ...
         'Color', colors(idx, :), ...
         'DisplayName', sprintf('\\nu=%d', nu));
end
text(ax2, style.right_ylabel_x, 0.5, 'AF (dB)', ...
    'Units', 'normalized', ...
    'Interpreter', 'tex', ...
    'FontName', cfg.font_name, ...
    'FontSize', style.label_font, ...
    'FontWeight', 'normal', ...
    'Rotation', 90, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'Clipping', 'off');
xlim(ax2, [-0.3 N-0.7]);
ylim(ax2, [-40 2]);
pbaspect(ax2, [1 1 1]);
set(ax2, 'FontSize', style.axes_font, 'TickLabelInterpreter', 'tex', ...
         'YAxisLocation', 'left', ...
         'LabelFontSizeMultiplier', 1, 'LineWidth', style.axes_line_width);
set(ax2, 'XTick', 0:5:15, 'XTickLabel', []);
draw_cut_legend(fig, style.legend_position, nu_list, colors, style);

drawnow;
plot_config(fig);
draw_manual_x_tick_labels(ax1, 0:5:15, style);
draw_manual_x_tick_labels(ax2, 0:5:15, style);
draw_axis_xlabel(ax1, 'Delay index \tau', style);
draw_axis_xlabel(ax2, 'Delay index \tau', style);
draw_panel_caption(ax1, '(a)', style);
draw_panel_caption(ax2, '(b)', style);
drawnow;
tight_export_figure(fig, fullfile(fig_dir, 'af_no_coupling_combined.pdf'), ...
               'ContentType', 'image', ...
               'Resolution', style.export_resolution, ...
               'TightPad', style.export_padding);
tight_export_figure(fig, fullfile(fig_dir, 'af_no_coupling_combined.png'), ...
               'Resolution', style.export_resolution);
tight_export_figure(fig, fullfile(data_dir, 'fig2_af_combined_preview.png'), ...
               'Resolution', style.export_resolution);
end

function style = combined_figure_style(cfg)
style = struct( ...
    'figure_position', [100 100 1040 520], ...
    'left_axes_position', [0.110 0.235 0.320 0.667], ...
    'right_axes_position', [0.648 0.235 0.320 0.667], ...
    'legend_position', [0.670 0.255 0.285 0.275], ...
    'colorbar_position', [0.440 0.235 0.018 0.667], ...
    'colorbar_ticks', -25:5:0, ...
    'contour_floor_dB', -28, ...
    'axes_font', cfg.axes_font, ...
    'label_font', cfg.label_font, ...
    'legend_font', cfg.legend_font, ...
    'colorbar_font', cfg.axes_font, ...
    'colorbar_label_font', cfg.label_font, ...
    'panel_font', cfg.panel_caption_font, ...
    'right_ylabel_x', -0.220, ...
    'colorbar_tick_label_x_offset', 0.0008, ...
    'colorbar_title_height', 0.045, ...
    'colorbar_title_y_offset', 0.018, ...
    'panel_caption_y', -0.240, ...
    'main_line_width', 2.2, ...
    'line_width', 1.8, ...
    'axes_line_width', cfg.axes_line_width, ...
    'export_padding', 14, ...
    'export_resolution', min(cfg.export_resolution, 600));
end

function cmap = high_contrast_af_colormap(num_colors)
try
    cmap = turbo(num_colors);
catch
    cmap = hot(num_colors);
end
end

function colors = muted_colors(num_colors)
colors = paper_palette(1:num_colors);
end

function draw_manual_x_tick_labels(ax, tick_values, style)
xl = xlim(ax);
for idx = 1:numel(tick_values)
    x_normalized = (tick_values(idx) - xl(1)) / (xl(2) - xl(1));
    if idx == 1
        horizontal_alignment = 'left';
    elseif idx == numel(tick_values)
        horizontal_alignment = 'right';
    else
        horizontal_alignment = 'center';
    end
    text(ax, x_normalized, 0.006, sprintf('%g', tick_values(idx)), ...
        'Units', 'normalized', ...
        'Interpreter', 'tex', ...
        'FontName', plot_config().font_name, ...
        'FontSize', style.axes_font, ...
        'HorizontalAlignment', horizontal_alignment, ...
        'VerticalAlignment', 'top', ...
        'Clipping', 'off');
end
end

function draw_axis_xlabel(ax, label_text, style)
text(ax, 0.5, -0.070, label_text, ...
    'Units', 'normalized', ...
    'Interpreter', 'tex', ...
    'FontName', plot_config().font_name, ...
    'FontSize', style.axes_font, ...
    'FontWeight', 'normal', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'top', ...
    'Clipping', 'off');
end

function draw_panel_caption(ax, caption_text, style)
text(ax, 0.5, style.panel_caption_y, caption_text, ...
    'Units', 'normalized', ...
    'Interpreter', 'tex', ...
    'FontName', plot_config().font_name, ...
    'FontSize', style.panel_font, ...
    'FontWeight', 'normal', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'Clipping', 'off');
end

function draw_colorbar_title(fig, cb, title_text, style)
p = cb.Position;
annotation(fig, 'textbox', ...
    [p(1) - 0.035, p(2) + p(4) + style.colorbar_title_y_offset, ...
     p(3) + 0.070, style.colorbar_title_height], ...
    'String', title_text, ...
    'Interpreter', 'tex', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', ...
    'FitBoxToText', 'off', ...
    'FontName', plot_config().font_name, ...
    'FontSize', style.colorbar_label_font);
end

function draw_colorbar_tick_labels(fig, cb, ticks, style)
p = cb.Position;
tick_min = style.contour_floor_dB;
tick_max = 0;
label_width = 0.040;
label_height = 0.040;
for idx = 1:numel(ticks)
    y = p(2) + (ticks(idx) - tick_min) / (tick_max - tick_min) * p(4);
    annotation(fig, 'textbox', ...
        [p(1) + p(3) + style.colorbar_tick_label_x_offset, ...
         y - 0.5 * label_height, label_width, label_height], ...
        'String', sprintf('%g', ticks(idx)), ...
        'Interpreter', 'tex', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', ...
        'EdgeColor', 'none', ...
        'FitBoxToText', 'off', ...
        'FontName', plot_config().font_name, ...
        'FontSize', style.colorbar_font);
end
end

function draw_cut_legend(fig, position, nu_list, colors, style)
ax_leg = axes(fig, 'Position', position, ...
    'XLim', [0 1], 'YLim', [0 1], ...
    'Visible', 'off', ...
    'Color', 'none');
hold(ax_leg, 'on');
rectangle(ax_leg, 'Position', [0.015 0.015 0.970 0.970], ...
    'FaceColor', 'w', ...
    'EdgeColor', [0.15 0.15 0.15], ...
    'LineWidth', style.axes_line_width);

num_cols = 2;
num_rows = ceil(numel(nu_list) / num_cols);
x_line = [0.060 0.520];
x_text = [0.235 0.695];
y_pos = linspace(0.790, 0.210, num_rows);
for idx = 1:numel(nu_list)
    row = mod(idx - 1, num_rows) + 1;
    col = floor((idx - 1) / num_rows) + 1;
    y = y_pos(row);
    plot(ax_leg, x_line(col) + [0 0.145], [y y], '-', ...
        'Color', colors(idx, :), 'LineWidth', style.line_width);
    text(ax_leg, x_text(col), y, sprintf('\\nu=%d', nu_list(idx)), ...
        'Interpreter', 'tex', ...
        'FontName', plot_config().font_name, ...
        'FontSize', style.legend_font, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
end
try
    uistack(ax_leg, 'top');
catch
end
end
