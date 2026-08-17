function plot_multitarget_worst_af_zero_doppler_1x2_comparison(force_rerun_right)
% PLOT_MULTITARGET_WORST_AF_ZERO_DOPPLER_1X2_COMPARISON
% One-column 1x2 zero-Doppler AF cut: baseline case and a tighter NT=8 case.

if nargin < 1 || isempty(force_rerun_right)
    force_rerun_right = false;
end

clearvars -except force_rerun_right; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
if exist(data_dir, 'dir') ~= 7, mkdir(data_dir); end
addpath(genpath(sim_dir));

left_cases = load_left_cases(data_dir);
right_cases = load_or_build_right_cases(data_dir, force_rerun_right);

panels = struct([]);
panels(1).cases = left_cases;
panels(1).title = '(a) N_T=4, CV_{max}=0.5';
panels(2).cases = right_cases;
panels(2).title = '(b) N_T=8, CV_{max}=0.1';

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
    S = load(cache_path);
    if all(isfield(S.cases, {'Pn', 'kappa'}))
        cases = S.cases;
        fprintf('Loaded right-panel cache: %s\n', cache_path);
        return;
    end
    if all(isfield(S, {'H', 'params', 'CV_max'}))
        fprintf('Upgrading right-panel cache with fractional-delay inputs: %s\n', cache_path);
        cases = rebuild_selected_right_cases(S);
        H = S.H;
        params = S.params;
        CV_max = S.CV_max;
        save(cache_path, 'cases', 'params', 'CV_max', 'H');
        return;
    end
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

function cases = rebuild_selected_right_cases(S)
params = S.params;
default_params = setup_params();
default_fields = fieldnames(default_params);
for i = 1:numel(default_fields)
    name = default_fields{i};
    if ~isfield(params, name)
        params.(name) = default_params.(name);
    end
end
params.sdp_quiet = true;
params.stop_if_alpha_unchanged = true;
A = compute_steering(params);

proposed = run_proposed(S.H, S.CV_max, params);
if isempty(proposed.W) || isnan(proposed.sumrate)
    error('Proposed CV cache upgrade failed: %s', proposed.status);
end
cases = result_to_worst_case('Proposed CV', 'Proposed', NaN, ...
    proposed, A, params);

for i = 2:numel(S.cases)
    mode = lower(S.cases(i).short);
    eta = S.cases(i).eta;
    result = run_surrogate_baseline(S.H, mode, eta, params);
    if isempty(result.W) || isnan(result.sumrate)
        error('%s cache upgrade failed: %s', upper(mode), result.status);
    end
    cases(i) = result_to_worst_case(sprintf('%s-based', upper(mode)), ...
        upper(mode), eta, result, A, params);
end
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

function plot_1x2(panels, fig_dir)
cfg = plot_config();
plot_style = struct( ...
    'figure_position', [100 100 1040 560], ...
    'left_axes_position', [0.100 0.300 0.385 0.520], ...
    'right_axes_position', [0.535 0.300 0.385 0.520], ...
    'legend_position', [0.660 0.658 0.260 0.150], ...
    'axes_font', cfg.axes_font, ...
    'label_font', cfg.label_font, ...
    'panel_caption_font', cfg.panel_caption_font, ...
    'axes_line_width', cfg.axes_line_width);
N = size(panels(1).cases(1).ESL_dB, 1);
num_fine = 32 * N;
tau_fine = linspace(-N/2, N/2, num_fine + 1);
marker_idx = unique(round(linspace(1, numel(tau_fine), 11)));
colors = paper_palette(1:3);

fig = figure('Color', 'w', 'Position', plot_style.figure_position);
set(fig, 'PaperPositionMode', 'auto');
line_handles = gobjects(numel(panels), 3);
ax_list = gobjects(numel(panels), 1);

for p = 1:numel(panels)
    if p == 1
        axes_position = plot_style.left_axes_position;
    else
        axes_position = plot_style.right_axes_position;
    end
    ax = axes(fig, 'Position', axes_position);
    ax_list(p) = ax;
    hold(ax, 'on');
    y_min_seen = Inf;
    y_max_seen = -Inf;

    for i = 1:numel(panels(p).cases)
        c = panels(p).cases(i);
        zero_doppler_cut_dB = c.ESL_dB(:, 1);
        [~, cut_samples_dB] = center_zero_delay_sample_grid(zero_doppler_cut_dB);
        smooth_cut_dB = fractional_zero_doppler_esl( ...
            c.Pn, c.kappa, tau_fine);
        y_min_seen = min([y_min_seen; smooth_cut_dB(:); cut_samples_dB(:)]);
        y_max_seen = max([y_max_seen; smooth_cut_dB(:); cut_samples_dB(:)]);
        h = plot(ax, tau_fine, smooth_cut_dB, '-d', ...
            'LineWidth', cfg.line_width, ...
            'MarkerIndices', marker_idx, ...
            'MarkerSize', cfg.marker_size, ...
            'MarkerFaceColor', colors(i, :), ...
            'MarkerEdgeColor', colors(i, :), ...
            'Color', colors(i, :), ...
            'DisplayName', legend_label(c));
        line_handles(p, i) = h;
    end

    grid(ax, 'on'); box(ax, 'on');
    xlabel(ax, 'Delay index \tau', 'FontSize', plot_style.label_font);
    xlim(ax, [-N/2, N/2]);
    ylim(ax, valid_axis_limits([y_min_seen y_max_seen], cfg));
    xticks(ax, -N/2:4:N/2);
    set(ax, 'FontSize', plot_style.axes_font, ...
        'LineWidth', plot_style.axes_line_width, ...
        'Layer', 'top', ...
        'LabelFontSizeMultiplier', 1);
    if p == 1
        ylabel(ax, 'ESL (dB)', 'FontSize', plot_style.label_font);
    else
        yticklabels(ax, []);
    end
end

set(ax_list(1).XLabel, 'Units', 'normalized', 'Position', [0.5 -0.125 0]);
set(ax_list(1).YLabel, 'Units', 'normalized', 'Position', [-0.105 0.5 0]);
set(ax_list(2).XLabel, 'Units', 'normalized', 'Position', [0.5 -0.125 0]);

drawnow;
plot_config(fig);
set(ax_list(1).XLabel, 'Units', 'normalized', 'Position', [0.5 -0.125 0]);
set(ax_list(1).YLabel, 'Units', 'normalized', 'Position', [-0.105 0.5 0]);
set(ax_list(2).XLabel, 'Units', 'normalized', 'Position', [0.5 -0.125 0]);
draw_multitarget_legend(fig, plot_style.legend_position, colors, plot_style, cfg);
for p = 1:numel(panels)
    add_panel_caption(fig, ax_list(p), panels(p).title, ...
        'YOffset', 0.235, 'Height', 0.055, ...
        'FontSize', plot_style.panel_caption_font, ...
        'Interpreter', 'tex', ...
        'WidthScale', 1.20);
end

out_png = fullfile(fig_dir, 'AF_Multitarget_Worst_Zero_Doppler_Cut_1x2.png');
out_pdf = fullfile(fig_dir, 'AF_Multitarget_Worst_Zero_Doppler_Cut_1x2.pdf');
tight_export_figure(fig, out_pdf, 'ContentType', 'image', ...
    'Resolution', cfg.export_resolution, 'TightPad', 0);
tight_export_figure(fig, out_png, ...
    'Resolution', cfg.export_resolution, 'TightPad', 0);
fprintf('Saved 1x2 zero-Doppler AF cut: %s\n', out_png);
fprintf('Saved 1x2 zero-Doppler AF cut: %s\n', out_pdf);
end

function draw_multitarget_legend(fig, position, colors, style, cfg)
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

legend_font = max(cfg.legend_font - 7, 1);
y_pos = [0.755 0.500 0.245];
labels = {'Proposed', 'CRB', 'MI'};
for idx = 1:numel(labels)
    plot(ax_leg, [0.085 0.270], [y_pos(idx) y_pos(idx)], '-', ...
        'Color', colors(idx, :), ...
        'LineWidth', cfg.secondary_line_width, ...
        'Clipping', 'off');
    x_center = 0.1775;
    dx = 0.020;
    dy = 0.047;
    patch(ax_leg, x_center + [0 dx 0 -dx], ...
        y_pos(idx) + [dy 0 -dy 0], ...
        colors(idx, :), ...
        'EdgeColor', colors(idx, :), ...
        'LineWidth', 1.2, ...
        'Clipping', 'off');
    text(ax_leg, 0.335, y_pos(idx), labels{idx}, ...
        'Interpreter', 'tex', ...
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
ESL_dB = ESL_dB(:);
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
