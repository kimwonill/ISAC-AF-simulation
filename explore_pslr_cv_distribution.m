function explore_pslr_cv_distribution()
% EXPLORE_PSLR_CV_DISTRIBUTION  Empirical PSLR distribution at fixed CV.
%
% This is an exploratory script, not a paper-generation script. It samples
% nonnegative directional-power profiles with exactly fixed CV, evaluates the
% PSLR mean/variance/quantiles, and optionally overlays existing proposed
% simulation points from results.mat.
%
% Important: PSLR is scale-invariant in P, so a total power budget only fixes
% the profile scale unless additional per-tone or beamforming feasibility
% constraints are imposed.

clear; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(sim_dir);
out_dir = fullfile(sim_dir, 'results');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end
addpath(genpath(sim_dir));

params = setup_params();
rng_seed = 11;
rng(rng_seed, 'twister');

CV_list = params.CV_max_list(:).';
num_samples = 50000;
N = params.N;
kappa = params.kappa;

fprintf('============================================================\n');
fprintf('  Fixed-CV PSLR Distribution Explorer\n');
fprintf('============================================================\n');
fprintf('  N = %d, kappa = %.3f, samples/CV = %d\n', N, kappa, num_samples);
fprintf('  CV grid = [%s]\n', num2str(CV_list));
fprintf('------------------------------------------------------------\n');

mean_pslr = nan(size(CV_list));
var_pslr = nan(size(CV_list));
std_pslr = nan(size(CV_list));
theory_mean_pslr = nan(size(CV_list));
theory_var_pslr = nan(size(CV_list));
theory_std_pslr = nan(size(CV_list));
median_pslr = nan(size(CV_list));
q05_pslr = nan(size(CV_list));
q95_pslr = nan(size(CV_list));
lower_bound = nan(size(CV_list));
upper_bound = nan(size(CV_list));
accept_rate = nan(size(CV_list));

all_sample_cv = cell(size(CV_list));
all_sample_pslr = cell(size(CV_list));

[v_grid, v_cdf] = max_sidelobe_cdf_grid(N, 1600);
cv_auto_nonnegative = 1 / sqrt(N - 1);

t_total = tic;
for i = 1:numel(CV_list)
    cv = CV_list(i);
    [P_samples, accept_rate(i)] = sample_fixed_cv_profiles(N, cv, num_samples);
    sample_pslr = compute_pslr_many(P_samples, kappa);
    sample_cv = compute_cv_many(P_samples);

    mean_pslr(i) = mean(sample_pslr);
    var_pslr(i) = var(sample_pslr, 1);
    std_pslr(i) = sqrt(var_pslr(i));
    median_pslr(i) = median(sample_pslr);
    q05_pslr(i) = quantile(sample_pslr, 0.05);
    q95_pslr(i) = quantile(sample_pslr, 0.95);
    [lower_bound(i), upper_bound(i)] = pslr_bounds_from_cv(N, kappa, cv);
    [theory_mean_pslr(i), theory_var_pslr(i)] = ...
        pslr_stats_from_v_cdf(v_grid, v_cdf, N, kappa, cv);
    theory_std_pslr(i) = sqrt(theory_var_pslr(i));

    all_sample_cv{i} = sample_cv;
    all_sample_pslr{i} = sample_pslr;

    fprintf(['  CV=%.2f | sample mean=%6.2f dB, theory mean=%6.2f dB, ' ...
             'q05/q95=[%6.2f, %6.2f] dB, accept=%.3f%s\n'], ...
        cv, 10*log10(mean_pslr(i)), ...
        10*log10(theory_mean_pslr(i)), ...
        10*log10(q05_pslr(i)), 10*log10(q95_pslr(i)), accept_rate(i), ...
        positivity_note(cv, cv_auto_nonnegative));
end

sim_overlay = load_existing_simulation_overlay(sim_dir);

source_path = fullfile(out_dir, 'pslr_cv_distribution_source_data.mat');
save(source_path, ...
    'CV_list', 'N', 'kappa', 'num_samples', 'rng_seed', 'params', ...
    'mean_pslr', 'var_pslr', 'std_pslr', 'median_pslr', ...
    'theory_mean_pslr', 'theory_var_pslr', 'theory_std_pslr', ...
    'q05_pslr', 'q95_pslr', 'lower_bound', 'upper_bound', ...
    'accept_rate', 'all_sample_cv', 'all_sample_pslr', ...
    'v_grid', 'v_cdf', 'cv_auto_nonnegative', 'sim_overlay');

plot_distribution(CV_list, mean_pslr, median_pslr, q05_pslr, q95_pslr, ...
    theory_mean_pslr, lower_bound, upper_bound, sim_overlay, sim_dir, out_dir);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', source_path);
fprintf('  Saved figures: %s\n', fullfile(out_dir, 'pslr_cv_distribution.png'));
fprintf('  Total elapsed: %s\n', format_time(toc(t_total)));
fprintf('============================================================\n');
end

function note = positivity_note(cv, cv_auto_nonnegative)
if cv <= cv_auto_nonnegative + 1e-12
    note = '';
else
    note = ' (positivity-conditioned)';
end
end

function [P_samples, accept_rate] = sample_fixed_cv_profiles(N, cv, num_samples)
if cv == 0
    P_samples = ones(N, num_samples);
    accept_rate = 1;
    return;
end

P_samples = zeros(N, num_samples);
num_kept = 0;
num_drawn = 0;
num_valid_drawn = 0;
batch_size = max(2000, 4 * num_samples);
max_drawn = max(1e7, 500 * num_samples);

while num_kept < num_samples
    Z = randn(N, batch_size);
    Z = Z - mean(Z, 1);
    z_norm = sqrt(sum(Z.^2, 1));
    Z = sqrt(N) * Z ./ z_norm;

    P = 1 + cv * Z;
    valid = all(P >= -1e-12, 1);
    num_valid_drawn = num_valid_drawn + nnz(valid);
    P = max(P(:, valid), 0);

    take = min(size(P, 2), num_samples - num_kept);
    if take > 0
        P_samples(:, num_kept + (1:take)) = P(:, 1:take);
        num_kept = num_kept + take;
    end

    num_drawn = num_drawn + batch_size;
    if num_drawn > max_drawn
        error('Sampling failed: CV=%.3f has too few nonnegative profiles.', cv);
    end
end

accept_rate = num_valid_drawn / num_drawn;
end

function pslr = compute_pslr_many(P, kappa)
N = size(P, 1); %#ok<NASGU>
P_dft = fft(P, [], 1);
sl_max = max(abs(P_dft(2:end, :)).^2, [], 1);
sq = sum(P.^2, 1);
sum_p = sum(P, 1);
mainlobe = (kappa - 1) * sq + sum_p.^2;
denom = (kappa - 1) * sq + sl_max;
pslr = mainlobe ./ denom;
end

function cv = compute_cv_many(P)
mu = mean(P, 1);
sigma = sqrt(mean(P.^2, 1) - mu.^2);
cv = sigma ./ mu;
end

function [lower_bound, upper_bound] = pslr_bounds_from_cv(N, kappa, cv)
upper_bound = (kappa - 1) / (kappa - 1 + N/(N-1)) + ...
    (N^2*kappa / ((N-1)*(kappa - 1 + N/(N-1)))) / ...
    ((kappa - 1 + N/(N-1))*cv^2 + kappa - 1);

lower_bound = (kappa - 1) / (N + kappa - 1) + ...
    (N*(N + 2*kappa - 2)/(N + kappa - 1)) / ...
    ((N + kappa - 1)*cv^2 + kappa - 1);
end

function [v_grid, F] = max_sidelobe_cdf_grid(N, num_grid)
% MAX_SIDELOBE_CDF_GRID  Exact CDF of V = max_{m~=0} |DFT(z)_m|^2/N
% for a real zero-mean vector z uniformly distributed on the sphere
% sum z_n^2 = N. The DFT is unitary in the derivation.

if mod(N, 2) == 0
    v_min = N / (N - 1);
    v_max = N;
else
    v_min = N / (N - 1);
    v_max = N / 2;
end
v_grid = linspace(v_min, v_max, num_grid);
F = zeros(size(v_grid));

if mod(N, 2) == 0
    K = N/2 - 1; % complex-conjugate DFT pairs
    beta_norm = beta(0.5, K);
    for i = 1:numel(v_grid)
        v = v_grid(i);
        h_max = min(v / N, 1);
        integrand = @(h) beta_pdf(h, 0.5, K, beta_norm) .* ...
            dirichlet_uniform_max_cdf(2*v ./ (N * (1 - h)), K);
        F(i) = integral(integrand, 0, h_max, ...
            'ArrayValued', true, 'RelTol', 1e-9, 'AbsTol', 1e-11);
    end
else
    K = (N - 1) / 2;
    F = dirichlet_uniform_max_cdf(2 * v_grid / N, K);
end

F = max(0, min(1, F));
F = cummax(F);
F(end) = 1;
end

function y = beta_pdf(h, a, b, beta_norm)
y = h.^(a - 1) .* (1 - h).^(b - 1) / beta_norm;
y(h <= 0) = 0;
y(h >= 1) = 0;
end

function F = dirichlet_uniform_max_cdf(a, K)
% CDF of max_i U_i for U ~ Dirichlet(1,...,1), i=1..K.
F = zeros(size(a));
for idx = 1:numel(a)
    ai = a(idx);
    if ai < 1 / K
        F(idx) = 0;
    elseif ai >= 1
        F(idx) = 1;
    else
        j_max = min(K, floor(1 / ai));
        acc = 0;
        for j = 0:j_max
            acc = acc + (-1)^j * nchoosek(K, j) * (1 - j * ai)^(K - 1);
        end
        F(idx) = acc;
    end
end
F = max(0, min(1, F));
end

function [mean_pslr, var_pslr] = pslr_stats_from_v_cdf(v_grid, F, N, kappa, cv)
if cv == 0
    mean_pslr = 1 + N / (kappa - 1);
    var_pslr = 0;
    return;
end

v_mid = 0.5 * (v_grid(1:end-1) + v_grid(2:end));
weights = diff(F);
weights = max(weights, 0);
weights = weights / sum(weights);

A = N + (kappa - 1) * (1 + cv^2);
B = (kappa - 1) * (1 + cv^2);
pslr = A ./ (B + cv^2 * v_mid);

mean_pslr = sum(weights .* pslr);
var_pslr = sum(weights .* (pslr - mean_pslr).^2);
end

function sim_overlay = load_existing_simulation_overlay(sim_dir)
sim_overlay = struct('has_data', false, 'cv', [], 'pslr', []);
results_path = fullfile(sim_dir, 'results.mat');
if exist(results_path, 'file') ~= 2
    return;
end

S = load(results_path, 'CV_max_list', 'pslr_lin_grid');
if ~isfield(S, 'CV_max_list') || ~isfield(S, 'pslr_lin_grid')
    return;
end

[cv_grid, ~] = ndgrid(S.CV_max_list(:), 1:size(S.pslr_lin_grid, 2));
sim_overlay.cv = cv_grid(:);
sim_overlay.pslr = S.pslr_lin_grid(:);
valid = isfinite(sim_overlay.cv) & isfinite(sim_overlay.pslr);
sim_overlay.cv = sim_overlay.cv(valid);
sim_overlay.pslr = sim_overlay.pslr(valid);
sim_overlay.has_data = ~isempty(sim_overlay.cv);
end

function plot_distribution(CV_list, mean_pslr, median_pslr, q05_pslr, q95_pslr, ...
    theory_mean_pslr, lower_bound, upper_bound, sim_overlay, sim_dir, out_dir)

fig = figure('Color', 'w', 'Position', [100 100 820 560]);
hold on; grid on; box on;

fill([CV_list, fliplr(CV_list)], ...
     [10*log10(q95_pslr), fliplr(10*log10(q05_pslr))], ...
     [0.70 0.78 0.62], 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
     'DisplayName', 'Sample 5--95%');
h_mean = plot(CV_list, 10*log10(mean_pslr), '-o', ...
    'LineWidth', 2.2, 'MarkerSize', 7.5, ...
    'MarkerFaceColor', [0.20 0.45 0.65], ...
    'Color', [0.20 0.45 0.65], ...
    'DisplayName', 'Sample mean');
h_median = plot(CV_list, 10*log10(median_pslr), '--s', ...
    'LineWidth', 1.8, 'MarkerSize', 6.5, ...
    'MarkerFaceColor', [0.32 0.50 0.48], ...
    'Color', [0.32 0.50 0.48], ...
    'DisplayName', 'Sample median');
h_theory_mean = plot(CV_list, 10*log10(theory_mean_pslr), '-x', ...
    'LineWidth', 1.8, 'MarkerSize', 8.0, ...
    'Color', [0.10 0.10 0.10], ...
    'DisplayName', 'Analytic mean');
h_upper = plot(CV_list, 10*log10(upper_bound), '-.', ...
    'LineWidth', 2.0, 'Color', [0.36 0.34 0.60], ...
    'DisplayName', 'Theoretical upper');
h_lower = plot(CV_list, 10*log10(lower_bound), ':', ...
    'LineWidth', 2.3, 'Color', [0.74 0.36 0.22], ...
    'DisplayName', 'Theoretical lower');

handles = [h_mean h_median h_theory_mean h_upper h_lower];
if sim_overlay.has_data
    jitter = 0.006 * randn(size(sim_overlay.cv));
    h_sim = scatter(sim_overlay.cv + jitter, 10*log10(sim_overlay.pslr), ...
        36, 'MarkerFaceColor', [0.10 0.10 0.10], ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.5, ...
        'DisplayName', 'Existing proposed runs');
    handles = [handles h_sim]; %#ok<AGROW>
end

xlabel('CV', 'FontSize', 15);
ylabel('PSLR (dB)', 'FontSize', 15);
title('Typical PSLR Distribution at Fixed CV', 'FontSize', 15);
legend(handles, 'Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 13);
xlim([min(CV_list)-0.03, max(CV_list)+0.03]);

png_path = fullfile(out_dir, 'pslr_cv_distribution.png');
fig_path = fullfile(out_dir, 'pslr_cv_distribution.fig');
pdf_path = fullfile(out_dir, 'pslr_cv_distribution.pdf');
tight_export_figure(fig, png_path, 'Resolution', 300);
tight_export_figure(fig, pdf_path, 'ContentType', 'vector');
saveas(fig, fig_path);
end
