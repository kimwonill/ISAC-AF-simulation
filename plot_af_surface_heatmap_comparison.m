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
root_dir = fileparts(sim_dir);
fig_dir = fullfile(root_dir, 'figures');
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

function plot_comparison(cases, S, fig_dir, CV_max_target, cv_tag)
z_floor = -28;
num_cases = numel(cases);
N = size(cases(1).ESL_dB, 1);
[TAU, NU] = meshgrid(0:N-1, 0:N-1);
palette = paper_palette();
edge_color = 0.35 * palette(1, :);

fig = figure('Color', 'w', 'Position', [80 80 760 330]);
tl = tiledlayout(fig, 1, num_cases, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:num_cases
    ax = nexttile(tl, i);
    Z = max(cases(i).ESL_dB, z_floor).';
    Z = rot90(Z, 2);
    hold(ax, 'on');
    surf(ax, TAU, NU, z_floor * ones(size(Z)), Z, ...
        'FaceColor', 'texturemap', 'EdgeColor', 'none', ...
        'FaceAlpha', 0.72, 'HandleVisibility', 'off');
    surf(ax, TAU, NU, Z, Z, ...
        'FaceColor', 'interp', 'EdgeColor', edge_color, ...
        'LineWidth', 0.18, 'FaceAlpha', 0.94, ...
        'HandleVisibility', 'off');
    clim(ax, [z_floor 0]);
    style_panel(ax, N, z_floor);
    title(ax, panel_title(cases(i)), 'FontSize', 14, 'FontWeight', 'bold');
    if i == num_cases
        cb = colorbar(ax);
        cb.Label.String = 'AF (dB)';
        cb.FontSize = 10.5;
    end
end

colormap(fig, turbo(256));

out_png = fullfile(fig_dir, sprintf('AF_3D_Surface_Heatmap_Comparison_%s.png', cv_tag));
out_pdf = fullfile(fig_dir, sprintf('AF_3D_Surface_Heatmap_Comparison_%s.pdf', cv_tag));
exportgraphics(fig, out_png, 'Resolution', 450);
exportgraphics(fig, out_pdf, 'ContentType', 'image', 'Resolution', 450);
fprintf('Saved AF comparison figure: %s\n', out_png);
fprintf('Saved AF comparison figure: %s\n', out_pdf);
end

function tag = cv_filename_tag(CV_max_target)
tag = sprintf('CV%02d', round(10 * CV_max_target));
end

function style_panel(ax, N, z_floor)
grid(ax, 'on'); box(ax, 'on');
xlabel(ax, '\tau', 'FontSize', 12);
ylabel(ax, '\nu', 'FontSize', 12);
zlabel(ax, '');
xlim(ax, [0 N-1]); ylim(ax, [0 N-1]); zlim(ax, [z_floor 1]);
xticks(ax, 0:5:N-1); yticks(ax, 0:5:N-1);
set(ax, 'FontSize', 10.5, 'Layer', 'top');
pbaspect(ax, [1 1 0.62]);
view(ax, [-44 30]);
camlight(ax, 'headlight');
lighting(ax, 'gouraud');
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

function value = plotted_target_pslr_dB(ESL_dB)
zero_doppler_cut = ESL_dB(:, 1);
value = zero_doppler_cut(1) - max(zero_doppler_cut(2:end));
end

function value = plotted_target_islr_dB(ESL_dB)
ESL = 10.^(ESL_dB / 10);
mainlobe = ESL(1, 1);
islr = (sum(ESL(:)) - mainlobe) / mainlobe;
value = 10 * log10(max(islr, realmin));
end
