function plot_multitarget_worst_af_zero_doppler_comparison(force_rerun, CV_max)
% PLOT_MULTITARGET_WORST_AF_ZERO_DOPPLER_COMPARISON
% Plot the zero-Doppler AF cut for the worst-PSLR target of each scheme.

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

S = load(fig2_path, 'H', 'params', 'CV_max', 'result');
params = S.params;
params.sdp_quiet = true;
params.stop_if_alpha_unchanged = true;

cache_path = fullfile(data_dir, sprintf('multitarget_worst_af_zero_doppler_%s_results.mat', ...
    cv_filename_tag(CV_max)));

if exist(cache_path, 'file') == 2 && ~force_rerun
    R = load(cache_path, 'cases', 'CV_max');
    cases = R.cases;
    fprintf('Loaded multi-target worst-case AF cache: %s\n', cache_path);
else
    cases = optimize_multitarget_cases(S, params, CV_max, data_dir);
    save(cache_path, 'cases', 'CV_max');
    fprintf('Saved multi-target worst-case AF cache: %s\n', cache_path);
end

plot_worst_zero_doppler_cut(cases, fig_dir, CV_max);
end

function cases = optimize_multitarget_cases(S, params, CV_max, data_dir)
A = compute_steering(params);

if abs(CV_max - S.CV_max) < 1e-12 && isfield(S, 'result') && ~isempty(S.result.W)
    proposed = S.result;
    fprintf('Using cached proposed multi-target result: CV_max=%.2f\n', CV_max);
else
    fprintf('Solving proposed multi-target CV case: CV_max=%.2f\n', CV_max);
    proposed = run_proposed(S.H, CV_max, params);
end
if isempty(proposed.W) || isnan(proposed.sumrate)
    error('Proposed CV failed: %s', proposed.status);
end

cases = result_to_worst_case('Proposed CV', 'Proposed', NaN, proposed, A, params);
fprintf('Proposed target SR %.3f bps/Hz, worst PSLR %.3f dB at target %d\n', ...
    proposed.sumrate, 10*log10(cases(1).pslr), cases(1).worst_target_idx);

[crb_eta, mi_eta] = selected_etas_from_cache(data_dir);
fprintf('Solving CRB multi-target result at eta=%g\n', crb_eta);
crb = run_surrogate_baseline(S.H, 'crb', crb_eta, params);
cases(2) = result_to_worst_case('CRB-based', 'CRB', crb_eta, crb, A, params);

fprintf('Solving MI multi-target result at eta=%g\n', mi_eta);
mi = run_surrogate_baseline(S.H, 'mi', mi_eta, params);
cases(3) = result_to_worst_case('MI-based', 'MI', mi_eta, mi, A, params);

for i = 1:numel(cases)
    fprintf('%s | SR %.3f | worst theta %.0f deg | PSLR %.3f dB | ISLR %.3f dB\n', ...
        cases(i).name, cases(i).sumrate, cases(i).worst_theta_deg, ...
        10*log10(cases(i).pslr), 10*log10(cases(i).islr));
end
end

function [crb_eta, mi_eta] = selected_etas_from_cache(data_dir)
crb_eta = 0.008;
mi_eta = 1e4;
cache_path = fullfile(data_dir, 'af_surface_heatmap_comparison_results.mat');
if exist(cache_path, 'file') ~= 2
    return;
end

C = load(cache_path, 'cases');
if isfield(C, 'cases') && numel(C.cases) >= 3
    if isfield(C.cases(2), 'eta') && ~isnan(C.cases(2).eta)
        crb_eta = C.cases(2).eta;
    end
    if isfield(C.cases(3), 'eta') && ~isnan(C.cases(3).eta)
        mi_eta = C.cases(3).eta;
    end
end
end

function case_out = result_to_worst_case(name, short, eta, result, A, params)
if isempty(result.W) || isnan(result.sumrate)
    error('%s optimization failed: %s', name, result.status);
end

num_targets = params.L;
pslr_per_target = zeros(num_targets, 1);
islr_per_target = zeros(num_targets, 1);
ESL_dB_per_target = cell(num_targets, 1);
Pn_per_target = cell(num_targets, 1);

for l = 1:num_targets
    Pn = compute_directional_power(result.W, A(:, l));
    pslr_per_target(l) = compute_pslr(Pn, params.kappa);
    islr_per_target(l) = compute_islr(Pn, params.kappa);
    ESL_dB_per_target{l} = af_from_power(Pn, params.kappa);
    Pn_per_target{l} = Pn;
end

[~, worst_idx] = min(pslr_per_target);

case_out.name = name;
case_out.short = short;
case_out.eta = eta;
case_out.sumrate = result.sumrate;
case_out.pslr = pslr_per_target(worst_idx);
case_out.islr = islr_per_target(worst_idx);
case_out.pslr_per_target = pslr_per_target;
case_out.islr_per_target = islr_per_target;
case_out.worst_target_idx = worst_idx;
case_out.worst_theta_deg = params.theta(worst_idx) * 180 / pi;
case_out.Pn = Pn_per_target{worst_idx};
case_out.kappa = params.kappa;
case_out.ESL_dB = ESL_dB_per_target{worst_idx};
case_out.status = result.status;
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

function plot_worst_zero_doppler_cut(cases, fig_dir, CV_max)
N = size(cases(1).ESL_dB, 1);
interp_factor = 32;
num_fine = interp_factor * N;
tau_fine = linspace(-N/2, N/2, num_fine + 1);
y_min_seen = Inf;
colors = paper_palette(1:numel(cases));
line_styles = {'-', '--', '-.'};

fig = figure('Color', 'w', 'Position', [120 120 980 620]);
ax = axes(fig);
hold(ax, 'on');

for i = 1:numel(cases)
    zero_doppler_cut_dB = cases(i).ESL_dB(:, 1);
    [tau_samples, cut_samples_dB] = center_zero_delay_sample_grid(zero_doppler_cut_dB);
    smooth_cut_dB = fractional_zero_doppler_esl(cases(i).Pn, cases(i).kappa, tau_fine);
    y_min_seen = min([y_min_seen; smooth_cut_dB(:); cut_samples_dB(:)]);
    plot(ax, tau_fine, smooth_cut_dB, ...
        'LineWidth', 2.8, ...
        'LineStyle', line_styles{1 + mod(i - 1, numel(line_styles))}, ...
        'Color', colors(i, :), ...
        'DisplayName', legend_label(cases(i)));
    plot(ax, tau_samples, cut_samples_dB, 'o', ...
        'MarkerSize', 4.8, ...
        'MarkerFaceColor', colors(i, :), ...
        'MarkerEdgeColor', 'w', ...
        'LineWidth', 0.7, ...
        'HandleVisibility', 'off');
end

grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'Delay index \tau');
ylabel(ax, 'Oversampled ESL at \nu=0 (dB)');
title(ax, sprintf('Multi-Target Worst-Case Fractional-Delay AF Cut, CV_{max}=%.1f', CV_max), ...
    'FontSize', 16, 'FontWeight', 'bold');
xlim(ax, [-N/2 N/2]);
y_lower = floor(y_min_seen) - 0.5;
ylim(ax, [y_lower 1]);
xticks(ax, -N/2:2:N/2);
set(ax, 'FontSize', 14, 'LineWidth', 1.1, 'Layer', 'top');

lgd = legend(ax, 'Location', 'southoutside', 'Orientation', 'vertical');
lgd.FontSize = 10.8;
lgd.Box = 'off';

cv_tag = cv_filename_tag(CV_max);
out_png = fullfile(fig_dir, sprintf('AF_Multitarget_Worst_Fractional_Zero_Doppler_Cut_%s.png', cv_tag));
out_pdf = fullfile(fig_dir, sprintf('AF_Multitarget_Worst_Fractional_Zero_Doppler_Cut_%s.pdf', cv_tag));
exportgraphics(fig, out_png, 'Resolution', 450);
exportgraphics(fig, out_pdf, 'ContentType', 'vector');
fprintf('Saved multi-target worst-case zero-Doppler AF cut: %s\n', out_png);
fprintf('Saved multi-target worst-case zero-Doppler AF cut: %s\n', out_pdf);
end

function [tau_samples, cut_centered_dB] = center_zero_delay_sample_grid(cut_dB)
N = numel(cut_dB);
tau_samples = (-N/2:N/2-1).';
cut_centered_dB = fftshift(cut_dB(:));
tau_samples = [tau_samples; N/2];
cut_centered_dB = [cut_centered_dB; cut_centered_dB(1)];
end

function ESL_dB = fractional_zero_doppler_esl(P, kappa, tau)
N = numel(P);
P = P(:);
n = (0:N-1).';
V = exp(-1j * 2*pi * n * tau(:).' / N);
mainlobe = N^2 * ((kappa - 1) * sum(P.^2) + sum(P)^2);
ESL = N^2 * ((kappa - 1) * sum(P.^2) + abs(P.' * V).^2);
ESL_dB = 10 * log10(max(real(ESL), realmin) / mainlobe);
end

function str = legend_label(c)
if isnan(c.eta)
    eta_str = '';
else
    eta_str = sprintf(', \\eta=%g', c.eta);
end
str = sprintf('%s%s, SR %.2f, worst \\theta %.0f^\\circ, PSLR %.2f dB', ...
    c.name, eta_str, c.sumrate, c.worst_theta_deg, 10*log10(c.pslr));
end

function tag = cv_filename_tag(CV_max)
tag = sprintf('CV%02d', round(10 * CV_max));
end
