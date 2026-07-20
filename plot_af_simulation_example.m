function plot_af_simulation_example()
% PLOT_AF_SIMULATION_EXAMPLE  Fig. 2 AF example from one simulation run.
%
% The script runs the proposed design for one deterministic channel
% realization, extracts P_n(theta) from the optimized covariance, and
% evaluates the closed-form ESL in Lemma 1.

clearvars; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(sim_dir);
fig_dir = fullfile(root_dir, 'figures');
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
fprintf('Updated Fig. 2 PDFs in: %s\n', fig_dir);
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

style = combined_figure_style();
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
xlabel(ax1, 'Delay index $\tau$', 'Interpreter', 'latex', ...
       'FontSize', style.label_font);
ylabel(ax1, 'Doppler index $\nu$', 'Interpreter', 'latex', ...
       'FontSize', style.label_font);
text(ax1, 0.040, 0.940, '(a)', 'Units', 'normalized', ...
     'FontSize', style.panel_font, 'FontWeight', 'bold', ...
     'BackgroundColor', 'w', 'Margin', 1.5);
set(ax1, 'FontSize', style.axes_font, 'TickLabelInterpreter', 'latex', ...
         'LabelFontSizeMultiplier', 1, 'Layer', 'top', ...
         'LineWidth', style.axes_line_width);

cb = colorbar(ax1);
cb.Position = style.colorbar_position;
cb.Label.String = '';
cb.Title.String = 'AF (dB)';
cb.Title.Interpreter = 'latex';
cb.Title.FontSize = style.colorbar_label_font;
cb.FontSize = style.colorbar_font;

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
         'DisplayName', sprintf('$\\nu=%d$', nu));
end
xlabel(ax2, 'Delay index $\tau$', 'Interpreter', 'latex', ...
       'FontSize', style.label_font);
ylabel_handle = ylabel(ax2, 'AF (dB)', 'Interpreter', 'latex', ...
                       'FontSize', style.label_font);
set(ylabel_handle, 'Units', 'normalized', 'Position', [-0.065 0.5 0]);
xlim(ax2, [-0.3 N-0.7]);
ylim(ax2, [-40 2]);
pbaspect(ax2, [1 1 1]);
text(ax2, 0.035, 0.940, '(b)', 'Units', 'normalized', ...
     'FontSize', style.panel_font, 'FontWeight', 'bold', ...
     'BackgroundColor', 'w', 'Margin', 1.5);
legend(ax2, 'Interpreter', 'latex', 'Location', 'southeast', ...
       'NumColumns', 2, 'FontSize', style.legend_font);
set(ax2, 'FontSize', style.axes_font, 'TickLabelInterpreter', 'latex', ...
         'LabelFontSizeMultiplier', 1, 'LineWidth', style.axes_line_width);

drawnow;
exportgraphics(fig, fullfile(fig_dir, 'af_no_coupling_combined.pdf'), ...
               'ContentType', 'image', 'Resolution', style.export_resolution);
exportgraphics(fig, fullfile(data_dir, 'fig2_af_combined_preview.png'), ...
               'Resolution', style.export_resolution);
end

function style = combined_figure_style()
style = struct( ...
    'figure_position', [100 100 840 500], ...
    'left_axes_position', [0.085 0.190 0.369 0.620], ...
    'right_axes_position', [0.590 0.190 0.369 0.620], ...
    'colorbar_position', [0.466 0.190 0.018 0.620], ...
    'contour_floor_dB', -28, ...
    'axes_font', 17, ...
    'label_font', 20, ...
    'legend_font', 16.5, ...
    'colorbar_font', 15, ...
    'colorbar_label_font', 18, ...
    'panel_font', 20, ...
    'main_line_width', 2.2, ...
    'line_width', 1.8, ...
    'axes_line_width', 1.2, ...
    'export_resolution', 1200);
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
