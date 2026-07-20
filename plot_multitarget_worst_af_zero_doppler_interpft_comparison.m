function plot_multitarget_worst_af_zero_doppler_interpft_comparison(CV_max)
% PLOT_MULTITARGET_WORST_AF_ZERO_DOPPLER_INTERPFT_COMPARISON
% Regenerate the interpft-smoothed worst-target zero-Doppler AF cut.

if nargin < 1 || isempty(CV_max)
    CV_max = 0.5;
end

clearvars -except CV_max; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(sim_dir);
fig_dir = fullfile(root_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
addpath(genpath(sim_dir));

cv_tag = cv_filename_tag(CV_max);
cache_path = fullfile(data_dir, sprintf('multitarget_worst_af_zero_doppler_%s_results.mat', cv_tag));
if exist(cache_path, 'file') ~= 2
    error('Missing multi-target worst-case AF cache: %s', cache_path);
end

R = load(cache_path, 'cases');
cases = R.cases;
plot_interpft_zero_doppler_cut(cases, fig_dir, CV_max);
end

function plot_interpft_zero_doppler_cut(cases, fig_dir, CV_max)
N = size(cases(1).ESL_dB, 1);
interp_factor = 32;
num_fine = interp_factor * N;
tau_fine = linspace(-N/2, N/2, num_fine + 1);
y_min_seen = Inf;
colors = paper_palette(1:numel(cases));
line_styles = {'-', '--', '-.'};

fig = figure('Color', 'w', 'Position', [120 120 560 390]);
ax = axes(fig);
hold(ax, 'on');

for i = 1:numel(cases)
    zero_doppler_cut_dB = cases(i).ESL_dB(:, 1);
    [tau_samples, cut_samples_dB] = center_zero_delay_sample_grid(zero_doppler_cut_dB);
    smooth_cut_dB = interpft_centered_zero_delay(zero_doppler_cut_dB, num_fine);
    y_min_seen = min([y_min_seen; smooth_cut_dB(:); cut_samples_dB(:)]);
    plot(ax, tau_fine, smooth_cut_dB, ...
        'LineWidth', 2.0, ...
        'LineStyle', line_styles{1 + mod(i - 1, numel(line_styles))}, ...
        'Color', colors(i, :), ...
        'DisplayName', legend_label(cases(i)));
    plot(ax, tau_samples, cut_samples_dB, 'o', ...
        'MarkerSize', 3.8, ...
        'MarkerFaceColor', colors(i, :), ...
        'MarkerEdgeColor', 'w', ...
        'LineWidth', 0.7, ...
        'HandleVisibility', 'off');
end

grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'Delay index \tau');
ylabel(ax, 'ESL (dB)');
xlim(ax, [-N/2 N/2]);
ylim(ax, [floor(y_min_seen) - 0.5, 1]);
xticks(ax, -N/2:2:N/2);
set(ax, 'FontSize', 9.5, 'LineWidth', 0.9, 'Layer', 'top');

lgd = legend(ax, 'Location', 'southoutside', 'Orientation', 'horizontal');
lgd.FontSize = 8.2;
lgd.NumColumns = 3;
lgd.Box = 'off';

cv_tag = cv_filename_tag(CV_max);
out_png = fullfile(fig_dir, sprintf('AF_Multitarget_Worst_Zero_Doppler_Cut_%s.png', cv_tag));
out_pdf = fullfile(fig_dir, sprintf('AF_Multitarget_Worst_Zero_Doppler_Cut_%s.pdf', cv_tag));
exportgraphics(fig, out_png, 'Resolution', 450);
exportgraphics(fig, out_pdf, 'ContentType', 'image', 'Resolution', 450);
fprintf('Saved interpft multi-target worst-case zero-Doppler AF cut: %s\n', out_png);
fprintf('Saved interpft multi-target worst-case zero-Doppler AF cut: %s\n', out_pdf);
end

function [tau_samples, cut_centered_dB] = center_zero_delay_sample_grid(cut_dB)
N = numel(cut_dB);
tau_samples = (-N/2:N/2-1).';
cut_centered_dB = fftshift(cut_dB(:));
tau_samples = [tau_samples; N/2];
cut_centered_dB = [cut_centered_dB; cut_centered_dB(1)];
end

function interp_centered_dB = interpft_centered_zero_delay(cut_dB, num_fine)
interp_periodic_dB = real(interpft(cut_dB(:), num_fine));
interp_centered_dB = fftshift(interp_periodic_dB);
interp_centered_dB = [interp_centered_dB; interp_centered_dB(1)];
end

function str = legend_label(c)
str = sprintf('%s (%.1f dB)', c.short, 10*log10(c.pslr));
end

function tag = cv_filename_tag(CV_max)
tag = sprintf('CV%02d', round(10 * CV_max));
end
