function plot_af_surface_heatmap_comparison(force_rerun, CV_max_target, reuse_surrogate_cv)
% PLOT_AF_SURFACE_HEATMAP_COMPARISON
% Compare optimized AF surfaces for proposed, CRB-based, and MI-based designs.
%
% CRB/MI sensing weights are selected to match the proposed sum-rate as
% closely as possible on the same channel realization unless a surrogate
% cache CV is supplied as the third argument.

if nargin < 1 || isempty(force_rerun)
    force_rerun = false;
end
if nargin < 2
    CV_max_target = [];
end
if nargin < 3
    reuse_surrogate_cv = [];
end

clearvars -except force_rerun CV_max_target reuse_surrogate_cv; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
fig2_path = fullfile(data_dir, 'fig2_af_simulation_example.mat');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
if exist(data_dir, 'dir') ~= 7, mkdir(data_dir); end
addpath(genpath(sim_dir));

if exist(fig2_path, 'file') ~= 2
    error('Missing optimized AF data: %s. Run plot_af_simulation_example first.', fig2_path);
end

S = load(fig2_path, 'A', 'CV_max', 'H', 'params', 'result', 'target_idx', 'target_deg');
if isempty(CV_max_target)
    CV_max_target = S.CV_max;
end
S = set_proposed_case(S, CV_max_target);

cv_tag = cv_filename_tag(CV_max_target);
cache_path = fullfile(data_dir, sprintf('af_surface_heatmap_comparison_%s_results.mat', cv_tag));
if exist(cache_path, 'file') ~= 2 && abs(CV_max_target - 0.5) < 1e-12
    cache_path = fullfile(data_dir, 'af_surface_heatmap_comparison_results.mat');
end
if exist(cache_path, 'file') == 2 && ~force_rerun
    C = load(cache_path);
    cache_cv_ok = (isfield(C, 'CV_max_target') && abs(C.CV_max_target - CV_max_target) < 1e-12) || ...
        (~isfield(C, 'CV_max_target') && abs(CV_max_target - 0.5) < 1e-12);
    if isfield(C, 'cases') && isfield(C, 'target_sumrate') && ...
            cache_cv_ok && abs(C.target_sumrate - S.result.sumrate) < 1e-8
        cases = C.cases;
        fprintf('Loaded cached AF comparison data: %s\n', cache_path);
    else
        cases = build_comparison_cases(S, cache_path, CV_max_target, data_dir, reuse_surrogate_cv);
    end
else
    cases = build_comparison_cases(S, cache_path, CV_max_target, data_dir, reuse_surrogate_cv);
end

plot_comparison(cases, S, fig_dir, CV_max_target, cv_tag);
end

function S = set_proposed_case(S, CV_max_target)
if abs(CV_max_target - S.CV_max) < 1e-12
    return;
end

params = S.params;
params.sdp_quiet = true;
params.stop_if_alpha_unchanged = true;

fprintf('Solving proposed CV case on the Fig. 2 channel: CV_max=%.2f\n', CV_max_target);
result = run_proposed(S.H, CV_max_target, params);
if isempty(result.W) || isnan(result.sumrate)
    error('Proposed CV case failed for CV_max=%.2f: %s', CV_max_target, result.status);
end

S.CV_max = CV_max_target;
S.result = result;
end

function cases = build_comparison_cases(S, cache_path, CV_max_target, data_dir, reuse_surrogate_cv)
params = S.params;
params.sdp_quiet = true;
params.stop_if_alpha_unchanged = true;

target_sumrate = S.result.sumrate;
target_idx = S.target_idx;
kappa = params.kappa;

cases = struct([]);
cases(1).name = 'Proposed CV';
cases(1).short = 'Proposed';
cases(1).eta = NaN;
cases(1).sumrate = S.result.sumrate;
cases(1).pslr = min(S.result.pslr_per_target);
cases(1).islr = max(S.result.islr_per_target);
cases(1).ESL_dB = af_from_covariance(S.result.W, S.A(:, target_idx), kappa);

fprintf('Target proposed sum-rate: %.3f bps/Hz\n', target_sumrate);
crb_trace = [];
mi_trace = [];
surrogate_source_cache = '';
if ~isempty(reuse_surrogate_cv)
    surrogate_source_cache = surrogate_cache_path(data_dir, reuse_surrogate_cv);
    R = load(surrogate_source_cache, 'cases');
    if ~isfield(R, 'cases') || numel(R.cases) < 3
        error('Reusable surrogate cache does not contain CRB/MI cases: %s', surrogate_source_cache);
    end
    cases(2) = R.cases(2);
    cases(3) = R.cases(3);
    fprintf('Reused CRB/MI cases from cache: %s\n', surrogate_source_cache);
else
    [cases(2), crb_trace] = select_surrogate_case(S, 'crb', crb_eta_candidates(), target_sumrate);
    [cases(3), mi_trace] = select_surrogate_case(S, 'mi', mi_eta_candidates(), target_sumrate);
end

save(cache_path, 'cases', 'crb_trace', 'mi_trace', 'target_sumrate', ...
    'CV_max_target', 'surrogate_source_cache');
fprintf('Saved AF comparison cache: %s\n', cache_path);
end

function path = surrogate_cache_path(data_dir, reuse_surrogate_cv)
tag = cv_filename_tag(reuse_surrogate_cv);
path = fullfile(data_dir, sprintf('af_surface_heatmap_comparison_%s_results.mat', tag));
if exist(path, 'file') ~= 2 && abs(reuse_surrogate_cv - 0.5) < 1e-12
    path = fullfile(data_dir, 'af_surface_heatmap_comparison_results.mat');
end
if exist(path, 'file') ~= 2
    error('Missing reusable surrogate cache for CV=%.2f: %s', reuse_surrogate_cv, path);
end
end

function [case_out, trace] = select_surrogate_case(S, mode, eta_list, target_sumrate)
params = S.params;
params.sdp_quiet = true;
params.stop_if_alpha_unchanged = true;

trace = struct('eta', eta_list(:), 'sumrate', nan(numel(eta_list), 1), ...
    'pslr', nan(numel(eta_list), 1), 'islr', nan(numel(eta_list), 1), ...
    'status', strings(numel(eta_list), 1));
results = cell(numel(eta_list), 1);

alpha_warm = [];
for i = 1:numel(eta_list)
    eta = eta_list(i);
    t = tic;
    result = run_surrogate_baseline(S.H, mode, eta, params, alpha_warm);
    trace.status(i) = string(result.status);
    if ~isnan(result.sumrate)
        alpha_warm = result.alpha;
        trace.sumrate(i) = result.sumrate;
        trace.pslr(i) = min(result.pslr_per_target);
        trace.islr(i) = max(result.islr_per_target);
        results{i} = result;
        fprintf('  %-3s eta=%9.3g | SR=%7.3f | PSLR=%5.2f dB | ISLR=%5.2f dB | %.1fs\n', ...
            upper(mode), eta, result.sumrate, 10*log10(trace.pslr(i)), ...
            10*log10(trace.islr(i)), toc(t));
    else
        alpha_warm = [];
        fprintf('  %-3s eta=%9.3g | failed: %s | %.1fs\n', ...
            upper(mode), eta, result.status, toc(t));
    end
end

valid = isfinite(trace.sumrate);
if ~any(valid)
    error('No feasible %s-based AF comparison point found.', upper(mode));
end

valid_idx = find(valid);
[~, local_ix] = min(abs(trace.sumrate(valid_idx) - target_sumrate));
best_ix = valid_idx(local_ix);
best = results{best_ix};

case_out.name = sprintf('%s-based', upper(mode));
case_out.short = upper(mode);
case_out.eta = eta_list(best_ix);
case_out.sumrate = best.sumrate;
case_out.pslr = min(best.pslr_per_target);
case_out.islr = max(best.islr_per_target);
case_out.ESL_dB = af_from_covariance(best.W, S.A(:, S.target_idx), params.kappa);

fprintf('Selected %-3s eta=%g: SR %.3f bps/Hz (target %.3f, gap %.3f)\n', ...
    upper(mode), case_out.eta, case_out.sumrate, target_sumrate, ...
    abs(case_out.sumrate - target_sumrate));
end

function eta = crb_eta_candidates()
eta = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, ...
       1e-3, 3e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, ...
       1e-2, 1.2e-2, 1.4e-2, 1.6e-2, 1.8e-2, 2e-2, ...
       2.3e-2, 2.6e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10];
end

function eta = mi_eta_candidates()
eta = [0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, ...
       1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7];
end

function ESL_dB = af_from_covariance(W, a, kappa)
P = compute_directional_power(W, a);
[~, ESL_dB] = closed_form_esl_local(P, kappa);
end

function [ESL, ESL_dB] = closed_form_esl_local(P, kappa)
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

function plot_comparison(cases, ~, fig_dir, CV_max_target, cv_tag)
cfg = plot_config();
z_floor = -28;
num_cases = numel(cases);
N = size(cases(1).ESL_dB, 1);

fig = figure('Color', 'w', 'Position', [80 80 1180 1060]);
axes_positions = [
    0.330 0.575 0.345 0.340
    0.085 0.220 0.345 0.340
    0.515 0.220 0.345 0.340
];
ax_list = gobjects(num_cases, 1);
[TAU, NU] = meshgrid(0:N-1, 0:N-1);
edge_color = [0.25 0.25 0.25];

for i = 1:num_cases
    ax = axes(fig, 'Position', axes_positions(i, :));
    ax_list(i) = ax;
    hold(ax, 'on');
    Z = max(cases(i).ESL_dB, z_floor).';
    surf(ax, TAU, NU, z_floor * ones(size(Z)), Z, ...
        'FaceColor', 'texturemap', ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.70, ...
        'HandleVisibility', 'off');
    surf(ax, TAU, NU, Z, Z, ...
        'FaceColor', 'interp', ...
        'EdgeColor', edge_color, ...
        'LineWidth', 0.18, ...
        'FaceAlpha', 0.96, ...
        'HandleVisibility', 'off');
    clim(ax, [z_floor 0]);
    style_3d_panel(ax, N, z_floor, cfg);
    if i == num_cases
        ylabel(ax, '');
    end
    if i == num_cases
        cb = colorbar(ax);
        cb.Label.String = '';
        cb.Title.String = '';
        cb.FontSize = cfg.axes_font - 7;
        cb.Position = [0.915 0.235 0.018 0.620];
        try
            cb.AxisLocation = 'out';
            cb.TickDirection = 'out';
        catch
        end
    end
end

colormap(fig, turbo(256));
drawnow;
plot_config(fig);
for i = 1:num_cases
    set(ax_list(i), 'FontSize', cfg.axes_font - 6, ...
        'FontWeight', 'bold', ...
        'GridColor', [0.15 0.15 0.15], ...
        'GridAlpha', 0.15, ...
        'GridLineStyle', '-', ...
        'LabelFontSizeMultiplier', 1);
    align_3d_axis_labels(ax_list(i), cfg, i == num_cases);
end
set(cb, 'FontWeight', 'bold');
ax_cb_label = axes(fig, 'Position', [0.955 0.235 0.020 0.620], ...
    'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1]);
text(ax_cb_label, 0.5, 0.5, 'AF (dB)', ...
    'Units', 'normalized', ...
    'Interpreter', 'tex', ...
    'FontName', cfg.font_name, ...
    'FontSize', cfg.label_font - 6, ...
    'FontWeight', 'bold', ...
    'Rotation', 90, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'Clipping', 'off');
caption_offsets = [0.045 0.045 0.045];
for i = 1:num_cases
    caption_handle = add_panel_caption(fig, ax_list(i), panel_title(cases(i)), ...
        'YOffset', caption_offsets(i), 'Height', 0.050, ...
        'FontSize', cfg.panel_caption_font - 3, 'Interpreter', 'tex');
    set(caption_handle, 'FontWeight', 'bold');
end

out_png = fullfile(fig_dir, sprintf('AF_3D_Surface_Heatmap_Comparison_%s.png', cv_tag));
out_pdf = fullfile(fig_dir, sprintf('AF_3D_Surface_Heatmap_Comparison_%s.pdf', cv_tag));
% The first export applies the shared tight-layout transform. Pull the
% transformed objects inward by a visually negligible amount so 3-D tick
% labels and border strokes are not clipped at the canvas boundary.
tight_export_figure(fig, out_png, 'Resolution', 450);
inset_top_level_positions(fig, 0.008);
tight_export_figure(fig, out_png, 'Resolution', 450, 'TightLayout', false);
tight_export_figure(fig, out_pdf, 'ContentType', 'image', 'Resolution', 450, ...
    'TightLayout', false);
if abs(CV_max_target - 0.5) < 1e-12
    copyfile(out_png, fullfile(fig_dir, 'AF_3D_Surface_Heatmap_Comparison.png'), 'f');
    copyfile(out_pdf, fullfile(fig_dir, 'AF_3D_Surface_Heatmap_Comparison.pdf'), 'f');
end
fprintf('Saved AF comparison figure: %s\n', out_png);
fprintf('Saved AF comparison figure: %s\n', out_pdf);
end

function tag = cv_filename_tag(CV_max_target)
tag = sprintf('CV%02d', round(10 * CV_max_target));
end

function style_3d_panel(ax, N, z_floor, cfg)
grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'Delay index \tau', 'FontSize', cfg.label_font - 8);
ylabel(ax, 'Doppler \nu', 'FontSize', cfg.label_font - 8);
zlabel(ax, '');
xlim(ax, [0 N-1]); ylim(ax, [0 N-1]);
zlim(ax, [z_floor 1]);
xticks(ax, 0:5:N-1); yticks(ax, 0:5:N-1);
zticks(ax, -20:10:0);
set(ax, 'FontSize', cfg.axes_font - 6, ...
    'Layer', 'top', ...
    'XDir', 'reverse', ...
    'YDir', 'reverse');
pbaspect(ax, [1 1 0.58]);
view(ax, [-42 29]);
camlight(ax, 'headlight');
lighting(ax, 'gouraud');
end

function align_3d_axis_labels(ax, cfg, hide_ylabel)
xl = get(ax, 'XLabel');
yl = get(ax, 'YLabel');
set(xl, ...
    'FontSize', cfg.label_font - 8, ...
    'FontWeight', 'bold', ...
    'Rotation', 22, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle');
if hide_ylabel
    set(yl, 'String', '');
else
    set(yl, ...
        'FontSize', cfg.label_font - 8, ...
        'FontWeight', 'bold', ...
        'Rotation', -33, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
end
end

function str = panel_title(c)
if contains(c.name, 'Proposed', 'IgnoreCase', true)
    str = '(a) Proposed CV';
elseif contains(c.name, 'CRB', 'IgnoreCase', true)
    str = '(b) CRB-based';
else
    str = '(c) MI-based';
end
end

function inset_top_level_positions(fig, pad)
scale = 1 - 2 * pad;
objects = {};

axes_list = findall(fig, 'Type', 'axes');
for i = 1:numel(axes_list)
    objects{end + 1} = axes_list(i); %#ok<AGROW>
end
colorbar_list = findall(fig, 'Type', 'colorbar');
for i = 1:numel(colorbar_list)
    objects{end + 1} = colorbar_list(i); %#ok<AGROW>
end
caption_list = findall(fig, 'Type', 'textboxshape');
for i = 1:numel(caption_list)
    objects{end + 1} = caption_list(i); %#ok<AGROW>
end

positions = cell(size(objects));
old_units = cell(size(objects));
valid = false(size(objects));
for i = 1:numel(objects)
    try
        old_units{i} = get(objects{i}, 'Units');
        set(objects{i}, 'Units', 'normalized');
        positions{i} = get(objects{i}, 'Position');
        set(objects{i}, 'Units', old_units{i});
        valid(i) = numel(positions{i}) >= 4;
    catch
    end
end

for i = 1:numel(objects)
    if ~valid(i)
        continue;
    end
    try
        pos = positions{i};
        pos(1:2) = pad + scale * pos(1:2);
        pos(3:4) = scale * pos(3:4);
        set(objects{i}, 'Units', 'normalized', 'Position', pos);
        set(objects{i}, 'Units', old_units{i});
    catch
    end
end
drawnow;
end
