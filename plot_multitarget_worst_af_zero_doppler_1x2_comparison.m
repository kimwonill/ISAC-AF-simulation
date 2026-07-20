function plot_multitarget_worst_af_zero_doppler_1x2_comparison(force_rerun_right)
% PLOT_MULTITARGET_WORST_AF_ZERO_DOPPLER_1X2_COMPARISON
% One-column 1x2 zero-Doppler AF cut: baseline case and a tighter NT=8 case.

if nargin < 1 || isempty(force_rerun_right)
    force_rerun_right = false;
end

clearvars -except force_rerun_right; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(sim_dir);
fig_dir = fullfile(root_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
if exist(data_dir, 'dir') ~= 7, mkdir(data_dir); end
addpath(genpath(sim_dir));

left_cases = load_left_cases(data_dir);
right_cases = load_or_build_right_cases(data_dir, force_rerun_right);

panels = struct([]);
panels(1).cases = left_cases;
panels(1).title = '(a) $N_T=4$, $N=16$, $\mathrm{CV}_{\max}=0.5$';
panels(2).cases = right_cases;
panels(2).title = '(b) $N_T=8$, $N=16$, $\mathrm{CV}_{\max}=0.1$';

plot_1x2(panels, fig_dir);
end

function cases = load_left_cases(data_dir)
cache_path = fullfile(data_dir, 'multitarget_worst_af_zero_doppler_CV05_results.mat');
if exist(cache_path, 'file') ~= 2
    error('Missing left-panel cache: %s. Run plot_multitarget_worst_af_zero_doppler_comparison first.', cache_path);
end
S = load(cache_path, 'cases');
cases = S.cases;
end

function cases = load_or_build_right_cases(data_dir, force_rerun)
cache_path = fullfile(data_dir, 'multitarget_worst_af_zero_doppler_NT8_N16_CV01_results.mat');
if exist(cache_path, 'file') == 2 && ~force_rerun
    S = load(cache_path, 'cases');
    cases = S.cases;
    fprintf('Loaded right-panel cache: %s\n', cache_path);
    return;
end

params = setup_params();
params.NT = 8;
params.N = 16;
params.sdp_quiet = true;
params.stop_if_alpha_unchanged = true;

CV_max = 0.1;
rng(81601, 'twister');
H = generate_channel(params);
cases = optimize_multitarget_cases(H, params, CV_max);
save(cache_path, 'cases', 'params', 'CV_max', 'H');
fprintf('Saved right-panel cache: %s\n', cache_path);
end

function cases = optimize_multitarget_cases(H, params, CV_max)
A = compute_steering(params);

fprintf('Solving proposed multi-target case: NT=%d, N=%d, CV=%.2f\n', ...
    params.NT, params.N, CV_max);
proposed = run_proposed(H, CV_max, params);
if isempty(proposed.W) || isnan(proposed.sumrate)
    error('Proposed CV failed: %s', proposed.status);
end
target_sumrate = proposed.sumrate;
cases = result_to_worst_case('Proposed CV', 'Proposed', NaN, proposed, A, params);
fprintf('  Proposed | SR %.3f | worst PSLR %.3f dB\n', ...
    target_sumrate, 10*log10(cases(1).pslr));

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

    candidate = result_to_worst_case(sprintf('%s-based', upper(mode)), upper(mode), eta, result, A, params);
    gap = abs(candidate.sumrate - target_sumrate);
    fprintf('  %-3s eta=%9.3g | SR=%7.3f | worst PSLR=%5.2f dB | gap=%6.3f | %.1fs\n', ...
        upper(mode), eta, candidate.sumrate, 10*log10(candidate.pslr), gap, toc(t));

    if gap < best_gap
        best_gap = gap;
        best_case = candidate;
    end
end

if isempty(best_case)
    error('No feasible %s-based point found.', upper(mode));
end

case_out = best_case;
fprintf('Selected %-3s eta=%g: SR %.3f bps/Hz, PSLR %.3f dB, gap %.3f\n', ...
    upper(mode), case_out.eta, case_out.sumrate, 10*log10(case_out.pslr), best_gap);
end

function case_out = result_to_worst_case(name, short, eta, result, A, params)
num_targets = params.L;
pslr_per_target = zeros(num_targets, 1);
islr_per_target = zeros(num_targets, 1);
ESL_dB_per_target = cell(num_targets, 1);

for l = 1:num_targets
    Pn = compute_directional_power(result.W, A(:, l));
    pslr_per_target(l) = compute_pslr(Pn, params.kappa);
    islr_per_target(l) = compute_islr(Pn, params.kappa);
    ESL_dB_per_target{l} = af_from_power(Pn, params.kappa);
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

function plot_1x2(panels, fig_dir)
N = size(panels(1).cases(1).ESL_dB, 1);
num_fine = 32 * N;
tau_fine = linspace(-N/2, N/2, num_fine + 1);
colors = paper_palette(1:3);
line_styles = {'-', '--', '-.'};

fig = figure('Color', 'w', 'Position', [120 120 760 320]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
line_handles = gobjects(3, 1);
ax_list = gobjects(numel(panels), 1);

for p = 1:numel(panels)
    ax = nexttile(tl, p);
    ax_list(p) = ax;
    hold(ax, 'on');
    y_min_seen = Inf;

    for i = 1:numel(panels(p).cases)
        c = panels(p).cases(i);
        zero_doppler_cut_dB = c.ESL_dB(:, 1);
        [tau_samples, cut_samples_dB] = center_zero_delay_sample_grid(zero_doppler_cut_dB);
        smooth_cut_dB = interpft_centered_zero_delay(zero_doppler_cut_dB, num_fine);
        y_min_seen = min([y_min_seen; smooth_cut_dB(:); cut_samples_dB(:)]);
        h = plot(ax, tau_fine, smooth_cut_dB, ...
            'LineWidth', 1.7, ...
            'LineStyle', line_styles{1 + mod(i - 1, numel(line_styles))}, ...
            'Color', colors(i, :), ...
            'DisplayName', legend_label(c));
        if p == 1
            line_handles(i) = h;
        end
        plot(ax, tau_samples, cut_samples_dB, 'o', ...
            'MarkerSize', 3.1, ...
            'MarkerFaceColor', colors(i, :), ...
            'MarkerEdgeColor', 'w', ...
            'LineWidth', 0.5, ...
            'HandleVisibility', 'off');
    end

    grid(ax, 'on'); box(ax, 'on');
    title(ax, panels(p).title, 'Interpreter', 'latex', 'FontSize', 20);
    xlabel(ax, 'Delay index \tau', 'FontSize', 19);
    xlim(ax, [-N/2 N/2]);
    ylim(ax, [floor(y_min_seen) - 0.5, 1]);
    xticks(ax, -N/2:4:N/2);
    set(ax, 'FontSize', 15, 'LineWidth', 0.8, 'Layer', 'top');
    if p == 1
        ylabel(ax, 'ESL (dB)', 'FontSize', 19);
    else
        yticklabels(ax, []);
    end
end

lgd = legend(ax_list(1), line_handles, {'Proposed', 'CRB', 'MI'}, ...
    'Location', 'northwest', 'Orientation', 'vertical');
lgd.FontSize = 16.5;
lgd.Box = 'off';

out_png = fullfile(fig_dir, 'AF_Multitarget_Worst_Zero_Doppler_Cut_1x2.png');
out_pdf = fullfile(fig_dir, 'AF_Multitarget_Worst_Zero_Doppler_Cut_1x2.pdf');
exportgraphics(fig, out_png, 'Resolution', 450);
exportgraphics(fig, out_pdf, 'ContentType', 'image', 'Resolution', 450);
fprintf('Saved 1x2 zero-Doppler AF cut: %s\n', out_png);
fprintf('Saved 1x2 zero-Doppler AF cut: %s\n', out_pdf);
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

function eta = crb_eta_candidates()
eta = [0, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2, ...
       1.4e-2, 1.8e-2, 2.3e-2, 3e-2, 5e-2, 0.1, 0.3, 1, 3, 10];
end

function eta = mi_eta_candidates()
eta = [0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6];
end
