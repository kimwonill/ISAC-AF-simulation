function plot_single_channel_af_zero_doppler_comparison(force_rerun, CV_max)
% PLOT_SINGLE_CHANNEL_AF_ZERO_DOPPLER_COMPARISON
% Optimize proposed/CRB/MI designs on one fixed channel and one plotted target.

if nargin < 1 || isempty(force_rerun)
    force_rerun = false;
end
if nargin < 2 || isempty(CV_max)
    CV_max = 0.5;
end

clearvars -except force_rerun CV_max; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(sim_dir);
fig_dir = fullfile(root_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
fig2_path = fullfile(data_dir, 'fig2_af_simulation_example.mat');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
if exist(data_dir, 'dir') ~= 7, mkdir(data_dir); end
addpath(genpath(sim_dir));

if exist(fig2_path, 'file') ~= 2
    error('Missing channel data: %s. Run plot_af_simulation_example first.', fig2_path);
end

S = load(fig2_path, 'H', 'params', 'target_deg');
params = single_target_params(S.params, S.target_deg);
cache_path = fullfile(data_dir, sprintf('single_channel_af_zero_doppler_%s_results.mat', cv_filename_tag(CV_max)));

if exist(cache_path, 'file') == 2 && ~force_rerun
    R = load(cache_path, 'cases', 'CV_max', 'target_deg');
    cases = R.cases;
    fprintf('Loaded single-channel AF comparison cache: %s\n', cache_path);
else
    cases = optimize_single_channel_cases(S.H, params, CV_max);
    target_deg = S.target_deg;
    save(cache_path, 'cases', 'CV_max', 'target_deg');
    fprintf('Saved single-channel AF comparison cache: %s\n', cache_path);
end

plot_zero_doppler_cut(cases, fig_dir, CV_max, S.target_deg);
end

function params = single_target_params(params, target_deg)
params.L = 1;
params.theta = target_deg * pi / 180;
params.sdp_quiet = true;
params.stop_if_alpha_unchanged = true;
end

function cases = optimize_single_channel_cases(H, params, CV_max)
A = compute_steering(params);

fprintf('Solving proposed CV on one channel/one target: CV_max=%.2f\n', CV_max);
proposed = run_proposed(H, CV_max, params);
if isempty(proposed.W) || isnan(proposed.sumrate)
    error('Proposed CV failed: %s', proposed.status);
end
target_sumrate = proposed.sumrate;

cases = result_to_case('Proposed CV', 'Proposed', NaN, proposed, A, params);
fprintf('Target proposed sum-rate: %.3f bps/Hz, PSLR %.3f dB\n', ...
    target_sumrate, 10 * log10(proposed.pslr_per_target(1)));

cases(2) = select_surrogate_case(H, params, A, 'crb', crb_eta_candidates(), target_sumrate);
cases(3) = select_surrogate_case(H, params, A, 'mi', mi_eta_candidates(), target_sumrate);
end

function case_out = select_surrogate_case(H, params, A, mode, eta_list, target_sumrate)
best_gap = Inf;
best_case = [];

for i = 1:numel(eta_list)
    eta = eta_list(i);
    t = tic;
    result = run_surrogate_baseline(H, mode, eta, params);
    if isempty(result.W) || isnan(result.sumrate)
        fprintf('  %-3s eta=%9.3g | failed: %s | %.1fs\n', ...
            upper(mode), eta, result.status, toc(t));
        continue;
    end

    candidate = result_to_case(sprintf('%s-based', upper(mode)), upper(mode), eta, result, A, params);
    gap = abs(candidate.sumrate - target_sumrate);
    fprintf('  %-3s eta=%9.3g | SR=%7.3f | PSLR=%5.2f dB | gap=%6.3f | %.1fs\n', ...
        upper(mode), eta, candidate.sumrate, 10*log10(candidate.pslr), gap, toc(t));

    if gap < best_gap
        best_gap = gap;
        best_case = candidate;
    end
end

if isempty(best_case)
    error('No feasible %s-based single-channel point found.', upper(mode));
end

case_out = best_case;
fprintf('Selected %-3s eta=%g: SR %.3f bps/Hz, PSLR %.3f dB, gap %.3f\n', ...
    upper(mode), case_out.eta, case_out.sumrate, 10*log10(case_out.pslr), best_gap);
end

function case_out = result_to_case(name, short, eta, result, A, params)
Pn = compute_directional_power(result.W, A(:, 1));

case_out.name = name;
case_out.short = short;
case_out.eta = eta;
case_out.sumrate = result.sumrate;
case_out.pslr = compute_pslr(Pn, params.kappa);
case_out.islr = compute_islr(Pn, params.kappa);
case_out.ESL_dB = af_from_power(Pn, params.kappa);
case_out.status = result.status;
end

function eta = crb_eta_candidates()
eta = [0, 0.001, 0.003, 0.005, 0.006, 0.007, 0.008, 0.009, ...
       0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.030, 0.100];
end

function eta = mi_eta_candidates()
eta = [0, 0.3, 1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5];
end

function ESL_dB = af_from_power(P, kappa)
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
ESL_dB = 10 * log10(ESL / ESL(1, 1));
end

function plot_zero_doppler_cut(cases, fig_dir, CV_max, target_deg)
N = size(cases(1).ESL_dB, 1);
interp_factor = 32;
num_fine = interp_factor * N;
tau_fine = linspace(-N/2, N/2, num_fine + 1);
y_floor = -35;
colors = paper_palette(1:numel(cases));
line_styles = {'-', '--', '-.'};

fig = figure('Color', 'w', 'Position', [120 120 940 600]);
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
title(ax, sprintf('Single-Channel Zero-Doppler AF Cut, CV_{max}=%.1f, \\theta=%.0f^\\circ', ...
    CV_max, target_deg), 'FontSize', 16, 'FontWeight', 'bold');
xlim(ax, [-N/2 N/2]);
ylim(ax, [y_floor 1]);
xticks(ax, -N/2:2:N/2);
set(ax, 'FontSize', 14, 'LineWidth', 1.1, 'Layer', 'top');

lgd = legend(ax, 'Location', 'southoutside', 'Orientation', 'vertical');
lgd.FontSize = 11.3;
lgd.Box = 'off';

cv_tag = cv_filename_tag(CV_max);
out_png = fullfile(fig_dir, sprintf('AF_Single_Channel_Zero_Doppler_Cut_%s.png', cv_tag));
out_pdf = fullfile(fig_dir, sprintf('AF_Single_Channel_Zero_Doppler_Cut_%s.pdf', cv_tag));
exportgraphics(fig, out_png, 'Resolution', 450);
exportgraphics(fig, out_pdf, 'ContentType', 'vector');
fprintf('Saved single-channel zero-Doppler AF cut: %s\n', out_png);
fprintf('Saved single-channel zero-Doppler AF cut: %s\n', out_pdf);
end

function [tau_samples, cut_centered_dB] = center_zero_delay_sample_grid(cut_dB)
N = numel(cut_dB);
cut_centered_dB = fftshift(cut_dB(:));
tau_samples = (-N/2:N/2-1).';
tau_samples = [tau_samples; N/2];
cut_centered_dB = [cut_centered_dB; cut_centered_dB(1)];
end

function str = legend_label(c)
if isnan(c.eta)
    eta_str = '';
else
    eta_str = sprintf(', \\eta=%g', c.eta);
end
str = sprintf('%s%s, SR %.2f, PSLR %.2f dB', ...
    c.name, eta_str, c.sumrate, 10*log10(c.pslr));
end

function tag = cv_filename_tag(CV_max)
tag = sprintf('CV%02d', round(10 * CV_max));
end
