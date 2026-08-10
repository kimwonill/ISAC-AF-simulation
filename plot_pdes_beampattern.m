%% PLOT_PDES_BEAMPATTERN  Beam-power sweep versus illumination floor
%
% Generates the Section V-D style plot:
%   P_des = n * P_max / N, n = 0, 1, 2, 3
% and plots the optimized average directional power versus azimuth angle.
%
% The optimization problem is unchanged from the proposed CV-constrained
% design; only the illumination floor is swept.

clear; close all; clc;

addpath(genpath(fileparts(mfilename('fullpath'))));
if exist('cvx_begin', 'file') ~= 2
    error('CVX is required. Install CVX and run cvx_setup first.');
end

params = setup_params();

% Match the discussion around Fig. V-D: dual targets at 0 and 30 degrees.
params.NT = 8;
params.N = 32;
params.L = 2;
params.theta = [0, 30] * pi/180;
% Isolate the illumination-floor effect for this figure.
params.Q = zeros(params.K, 1);

pdes_multipliers = [0, 1, 2, 3];
pdes_values = pdes_multipliers * params.P_max / params.N;
CV_max = params.CV_max_pdes_plot;
scan_deg = -90:0.5:90;
scan_rad = scan_deg * pi/180;

fprintf('============================================================\n');
fprintf('  P_des Beam-Power Sweep\n');
fprintf('============================================================\n');
fprintf('  Targets = [%s] deg, CV_max = %.2f\n', ...
        num2str(rad2deg(params.theta)), CV_max);
fprintf('  P_des = n P_max/N, n = [%s]\n', num2str(pdes_multipliers));
fprintf('------------------------------------------------------------\n');

rng(1, 'twister');
H = generate_channel(params);

A_scan = steering_for_angles(params, scan_rad);
beam_power = nan(numel(scan_deg), numel(pdes_values));
sumrate = nan(numel(pdes_values), 1);
status = strings(numel(pdes_values), 1);
iters = nan(numel(pdes_values), 1);
alpha_warm = [];

t_total = tic;
for idx = 1:numel(pdes_values)
    params_case = params;
    params_case.P_des = pdes_values(idx);

    t_iter = tic;
    result = run_proposed(H, CV_max, params_case, alpha_warm);
    status(idx) = string(result.status);
    iters(idx) = result.iters;

    if isnan(result.sumrate)
        alpha_warm = [];
        fprintf('  n=%d, P_des=%.4f: FAIL (%s)\n', ...
                pdes_multipliers(idx), pdes_values(idx), result.status);
        continue;
    end

    alpha_warm = result.alpha;
    sumrate(idx) = result.sumrate;
    beam_power(:, idx) = average_directional_power(result.W, A_scan);

    fprintf(['  n=%d, P_des=%.4f: SR=%.2f bps/Hz, %s, %d iters, %s, ' ...
             'elapsed %.1fs\n'], ...
            pdes_multipliers(idx), pdes_values(idx), result.sumrate, ...
            result.status, result.iters, result.stop_reason, toc(t_iter));
end

beam_power_dB = 10 * log10(max(beam_power, realmin));

out_dir = fileparts(mfilename('fullpath'));
cfg = plot_config();
export_resolution = cfg.export_resolution;
plot_style = struct( ...
    'figure_position', [100 100 1040 560], ...
    'axes_position', [0.100 0.205 0.850 0.655], ...
    'legend_position', [0.100 0.648 0.300 0.210], ...
    'axes_font', cfg.axes_font, ...
    'label_font', cfg.label_font, ...
    'title_font', cfg.panel_caption_font, ...
    'legend_font', max(cfg.legend_font - 5, 1));

fig = figure('Position', plot_style.figure_position, 'Color', 'w');
set(fig, 'PaperPositionMode', 'auto');
hold on; grid on; box on;

colors = muted_pdes_colors(numel(pdes_values));
plot_step_deg = 5;
plot_stride = max(1, round(plot_step_deg / (scan_deg(2) - scan_deg(1))));
plot_idx = 1:plot_stride:numel(scan_deg);
if plot_idx(end) ~= numel(scan_deg)
    plot_idx = [plot_idx, numel(scan_deg)];
end

for idx = 1:numel(pdes_values)
    if all(isnan(beam_power_dB(:, idx)))
        continue;
    end

    plot(scan_deg, beam_power_dB(:, idx), '-d', ...
         'LineWidth', cfg.line_width, ...
         'MarkerSize', cfg.marker_size, ...
         'MarkerIndices', plot_idx, ...
         'MarkerFaceColor', colors(idx, :), ...
         'MarkerEdgeColor', colors(idx, :), ...
         'Color', colors(idx, :), ...
         'DisplayName', pdes_legend_label(pdes_multipliers(idx)));
end

target_deg = rad2deg(params.theta);
for l = 1:numel(target_deg)
    xline(target_deg(l), '--k', 'LineWidth', cfg.secondary_line_width, ...
          'HandleVisibility', 'off');
end

ax = gca;
set(ax, 'FontSize', plot_style.axes_font, 'LabelFontSizeMultiplier', 1, ...
        'LineWidth', cfg.axes_line_width, 'Layer', 'top', ...
        'Units', 'normalized', 'Position', plot_style.axes_position);
xlabel(ax, 'Azimuth angle (deg)', 'FontSize', plot_style.label_font);
ylabel(ax, 'Directional power (dB)', 'FontSize', plot_style.label_font);
title_handle = title(ax, '', ...
                     'Interpreter', 'tex', 'FontSize', plot_style.title_font, ...
                     'FontWeight', 'normal');
set(ax.XLabel, 'Units', 'normalized', 'Position', [0.5 -0.115 0]);
set(ax.YLabel, 'Units', 'normalized', 'Position', [-0.055 0.5 0]);
xlim(valid_axis_limits(scan_deg, cfg, 'Clip', [-90 90]));
xticks(-90:30:90);

valid_beam = beam_power_dB(isfinite(beam_power_dB));
if ~isempty(valid_beam)
    ylim(valid_axis_limits(valid_beam, cfg));
end
set(ax.XLabel, 'FontSize', plot_style.label_font);
set(title_handle, 'FontSize', plot_style.title_font);
set(ax, 'LooseInset', max(get(ax, 'TightInset'), [0.015 0.015 0.015 0.015]));
plot_config(fig);
set(ax, 'FontSize', plot_style.axes_font, ...
        'LabelFontSizeMultiplier', 1, ...
        'LineWidth', cfg.axes_line_width, ...
        'Layer', 'top', ...
        'Units', 'normalized', ...
        'Position', plot_style.axes_position);
set(ax.XLabel, 'FontSize', plot_style.label_font, ...
    'Units', 'normalized', 'Position', [0.5 -0.115 0]);
set(ax.YLabel, 'String', 'Directional power (dB)', ...
    'FontSize', plot_style.label_font, ...
    'Units', 'normalized', 'Position', [-0.055 0.5 0]);
set(title_handle, 'FontSize', plot_style.title_font, 'FontWeight', 'normal');
legend_labels = arrayfun(@pdes_legend_label, pdes_multipliers, ...
    'UniformOutput', false);
draw_pdes_legend(fig, plot_style.legend_position, legend_labels, colors, ...
    repmat({'-'}, 1, numel(legend_labels)), cfg, 0);

png_path = fullfile(out_dir, 'beamgain_pdes_sweep.png');
tight_export_figure(fig, png_path, 'Resolution', export_resolution, ...
    'TightPad', 3);
saveas(fig, fullfile(out_dir, 'beamgain_pdes_sweep.fig'));

paper_fig_dir = fullfile(out_dir, 'figures');
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end
copyfile(png_path, fullfile(paper_fig_dir, 'beamgain_pdes_sweep.png'));
tight_export_figure(fig, fullfile(paper_fig_dir, 'beamgain_pdes_sweep.pdf'), ...
               'ContentType', 'image', 'Resolution', export_resolution, ...
               'TightPad', 3);

results_dir = fullfile(out_dir, 'results');
if exist(results_dir, 'dir') ~= 7
    mkdir(results_dir);
end
results_path = fullfile(results_dir, 'beamgain_pdes_sweep_results.mat');
save(results_path, ...
     'beam_power', 'beam_power_dB', 'scan_deg', 'pdes_multipliers', ...
     'pdes_values', 'CV_max', 'sumrate', 'status', 'iters', 'params');
copyfile(results_path, fullfile(out_dir, 'beamgain_pdes_sweep_results.mat'), 'f');

fprintf('------------------------------------------------------------\n');
fprintf('  Saved: beamgain_pdes_sweep.png/.fig, beamgain_pdes_sweep_results.mat\n');
fprintf('  Updated: figures/beamgain_pdes_sweep.png/.pdf\n');
fprintf('  Total elapsed: %s\n', format_time(toc(t_total)));
fprintf('============================================================\n');

function A = steering_for_angles(params, angles_rad)
phase_step = 2*pi * params.dT / params.lambda;
antenna_idx = (0:params.NT-1).';
A = exp(1j * phase_step * antenna_idx * sin(angles_rad(:).'));
end

function p = average_directional_power(W, A_scan)
num_angles = size(A_scan, 2);
N = size(W, 3);
p = zeros(num_angles, 1);

A_metric = conj(A_scan);
for m = 1:num_angles
    acc = 0;
    a_m = A_metric(:, m);
    for n = 1:N
        acc = acc + real(a_m' * W(:, :, n) * a_m);
    end
    p(m) = acc / N;
end
end

function label = pdes_legend_label(multiplier)
if multiplier == 0
    label = 'P_{des}=0';
elseif multiplier == 1
    label = 'P_{des}=P_{max}/N';
else
    label = sprintf('P_{des}=%dP_{max}/N', multiplier);
end
end

function colors = muted_pdes_colors(num_colors)
colors = paper_palette(1:num_colors);
end
