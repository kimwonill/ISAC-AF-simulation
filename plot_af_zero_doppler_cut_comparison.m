function plot_af_zero_doppler_cut_comparison(cv_target)
% PLOT_AF_ZERO_DOPPLER_CUT_COMPARISON
% Plot the zero-Doppler AF cut for the cached proposed/CRB/MI comparison.

if nargin < 1 || isempty(cv_target)
    cv_target = 0.4;
end

clearvars -except cv_target; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(sim_dir);
fig_dir = fullfile(root_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
addpath(genpath(sim_dir));

cv_tag = cv_filename_tag(cv_target);
cache_path = fullfile(data_dir, sprintf('af_surface_heatmap_comparison_%s_results.mat', cv_tag));
if exist(cache_path, 'file') ~= 2 && abs(cv_target - 0.5) < 1e-12
    cache_path = fullfile(data_dir, 'af_surface_heatmap_comparison_results.mat');
end
if exist(cache_path, 'file') ~= 2
    error('Missing AF comparison cache: %s', cache_path);
end

S = load(cache_path, 'cases');
cases = S.cases;
N = size(cases(1).ESL_dB, 1);
interp_factor = 32;
num_fine = interp_factor * N;
tau_fine = linspace(-N/2, N/2, num_fine + 1);
y_floor = -35;
colors = paper_palette(1:numel(cases));
line_styles = {'-', '--', '-.'};

fig = figure('Color', 'w', 'Position', [120 120 900 590]);
ax = axes(fig);
hold(ax, 'on');

for i = 1:numel(cases)
    zero_doppler_cut_dB = cases(i).ESL_dB(:, 1);
    [tau_samples, cut_samples_dB] = center_zero_delay_sample_grid(zero_doppler_cut_dB);
    smooth_cut_dB = interp1(tau_samples, cut_samples_dB, tau_fine, 'pchip');
    smooth_cut_dB = max(smooth_cut_dB, y_floor);
    plot(ax, tau_fine, smooth_cut_dB, ...
        'LineWidth', 2.8, ...
        'LineStyle', line_styles{1 + mod(i - 1, numel(line_styles))}, ...
        'Color', colors(i, :), ...
        'DisplayName', legend_label(cases(i)));
    plot(ax, tau_samples, max(cut_samples_dB, y_floor), 'o', ...
        'MarkerSize', 4.8, ...
        'MarkerFaceColor', colors(i, :), ...
        'MarkerEdgeColor', 'w', ...
        'LineWidth', 0.7, ...
        'HandleVisibility', 'off');
end

grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'Delay index \tau');
ylabel(ax, 'Normalized ESL at \nu=0 (dB)');
title(ax, sprintf('Zero-Doppler AF Cut, CV_{max}=%.1f', cv_target), ...
    'FontSize', 16, 'FontWeight', 'bold');
xlim(ax, [-N/2 N/2]);
ylim(ax, [y_floor 1]);
xticks(ax, -N/2:2:N/2);
set(ax, 'FontSize', 14, 'LineWidth', 1.1, 'Layer', 'top');

lgd = legend(ax, 'Location', 'southoutside', 'Orientation', 'vertical');
lgd.FontSize = 11.5;
lgd.Box = 'off';

out_png = fullfile(fig_dir, sprintf('AF_Zero_Doppler_Cut_Comparison_%s.png', cv_tag));
out_pdf = fullfile(fig_dir, sprintf('AF_Zero_Doppler_Cut_Comparison_%s.pdf', cv_tag));
tight_export_figure(fig, out_png, 'Resolution', 450);
tight_export_figure(fig, out_pdf, 'ContentType', 'vector');
fprintf('Saved zero-Doppler AF cut: %s\n', out_png);
fprintf('Saved zero-Doppler AF cut: %s\n', out_pdf);
end

function tag = cv_filename_tag(cv_target)
tag = sprintf('CV%02d', round(10 * cv_target));
end

function [tau_samples, cut_centered_dB] = center_zero_delay_sample_grid(cut_dB)
N = numel(cut_dB);
cut_centered_dB = fftshift(cut_dB(:));
tau_samples = (-N/2:N/2-1).';

% Close the periodic interval for interpolation and plotting.
tau_samples = [tau_samples; N/2];
cut_centered_dB = [cut_centered_dB; cut_centered_dB(1)];
end

function str = legend_label(c)
if isnan(c.eta)
    eta_str = '';
else
    eta_str = sprintf(', \\eta=%g', c.eta);
end
str = sprintf('%s%s, SR %.2f, shown PSLR %.2f dB', ...
    c.name, eta_str, c.sumrate, plotted_target_pslr_dB(c.ESL_dB));
end

function value = plotted_target_pslr_dB(ESL_dB)
zero_doppler_cut = ESL_dB(:, 1);
value = zero_doppler_cut(1) - max(zero_doppler_cut(2:end));
end
