function run_ml_constraint_landscape_experiment()
% RUN_ML_CONSTRAINT_LANDSCAPE_EXPERIMENT
% Diagnostic experiment for ML-oriented constraint difficulty.
%
% This script isolates the sensing constraint from the full beamforming
% candidate generator. It compares the smooth CV statistic with the direct
% peak-sidelobe statistic under small perturbations of the directional-power
% profile P_n. This is the cleanest way to show that the direct PSLR
% constraint induces active-index switching and a rougher reward landscape.

clear; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(sim_dir, '..', 'figures');
out_data_dir = fullfile(sim_dir, 'results');
if exist(out_data_dir, 'dir') ~= 7
    mkdir(out_data_dir);
end
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end
addpath(genpath(sim_dir));

params = setup_params();
N = params.N;
kappa = params.kappa;
CV_grid = [0.1, 0.25, 0.5, 0.75, 1.0];
tightness = 1 - CV_grid;

num_profiles = 350;
num_perturb = 45;
sigma_perturb = 0.035;
eps_fd = 2e-3;

rng(15001, 'twister');

switch_prob = nan(numel(CV_grid), 1);
cv_deriv_std = nan(numel(CV_grid), 1);
direct_deriv_std = nan(numel(CV_grid), 1);
cv_loss_std = nan(numel(CV_grid), 1);
direct_loss_std = nan(numel(CV_grid), 1);
cv_loss_cv = nan(numel(CV_grid), 1);
direct_loss_cv = nan(numel(CV_grid), 1);
cv_metric_change_mean = nan(numel(CV_grid), 1);
direct_metric_change_mean = nan(numel(CV_grid), 1);
metric_change_ratio = nan(numel(CV_grid), 1);
roughness_ratio = nan(numel(CV_grid), 1);
active_entropy = nan(numel(CV_grid), 1);

fprintf('============================================================\n');
fprintf('  Constraint landscape experiment: CV moment vs direct peak\n');
fprintf('============================================================\n');
fprintf('  N=%d, profiles=%d, perturbations/profile=%d, sigma=%.3f\n', ...
    N, num_profiles, num_perturb, sigma_perturb);
fprintf('------------------------------------------------------------\n');

for c = 1:numel(CV_grid)
    CV_max = CV_grid(c);
    [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);

    switch_flags = false(num_profiles * num_perturb, 1);
    active_ix_all = nan(num_profiles, 1);
    deriv_cv = nan(num_profiles * num_perturb, 1);
    deriv_direct = nan(num_profiles * num_perturb, 1);
    loss_cv = nan(num_profiles * num_perturb, 1);
    loss_direct = nan(num_profiles * num_perturb, 1);
    metric_change_cv = nan(num_profiles * num_perturb, 1);
    metric_change_direct = nan(num_profiles * num_perturb, 1);

    row = 0;
    for m = 1:num_profiles
        P = random_power_profile(N, CV_max);
        [~, active0] = direct_peak_metric(P);
        cv0 = cv_metric(P);
        peak0 = direct_peak_metric(P);
        active_ix_all(m) = active0;

        for q = 1:num_perturb
            row = row + 1;
            direction = randn(N, 1);
            direction = direction - mean(direction);
            direction = direction / max(norm(direction), eps);

            P_plus = normalize_power(P .* exp(sigma_perturb * direction));
            [~, active_plus] = direct_peak_metric(P_plus);
            switch_flags(row) = active_plus ~= active0;
            metric_change_cv(row) = abs(cv_metric(P_plus) - cv0) / max(abs(cv0), 1e-3);
            metric_change_direct(row) = abs(direct_peak_metric(P_plus) - peak0) / max(abs(peak0), 1e-3);

            P_fd_plus = normalize_power(P .* exp(eps_fd * direction));
            P_fd_minus = normalize_power(P .* exp(-eps_fd * direction));
            deriv_cv(row) = abs(cv_metric(P_fd_plus) - cv_metric(P_fd_minus)) / ...
                (2 * eps_fd * max(abs(cv0), 1e-3));
            deriv_direct(row) = abs(direct_peak_metric(P_fd_plus) - direct_peak_metric(P_fd_minus)) / ...
                (2 * eps_fd * max(abs(peak0), 1e-3));

            loss_cv(row) = cv_constraint_loss(P_plus, CV_max);
            loss_direct(row) = direct_constraint_loss(P_plus, pslr_min, islr_max, kappa);
        end
    end

    switch_prob(c) = mean(switch_flags);
    cv_deriv_std(c) = std(deriv_cv, 0, 'omitnan');
    direct_deriv_std(c) = std(deriv_direct, 0, 'omitnan');
    cv_loss_std(c) = std(loss_cv, 0, 'omitnan');
    direct_loss_std(c) = std(loss_direct, 0, 'omitnan');
    cv_loss_cv(c) = cv_loss_std(c) / max(abs(mean(loss_cv, 'omitnan')), 1e-4);
    direct_loss_cv(c) = direct_loss_std(c) / max(abs(mean(loss_direct, 'omitnan')), 1e-4);
    cv_metric_change_mean(c) = mean(metric_change_cv, 'omitnan');
    direct_metric_change_mean(c) = mean(metric_change_direct, 'omitnan');
    metric_change_ratio(c) = direct_metric_change_mean(c) / max(cv_metric_change_mean(c), 1e-8);
    roughness_ratio(c) = direct_deriv_std(c) / max(cv_deriv_std(c), 1e-8);
    active_entropy(c) = normalized_entropy(active_ix_all, N - 1);

    fprintf(['  CVmax=%.2f xi=%.2f | switch %.3f | deriv std CV %.3g, ' ...
        'Direct %.3g | loss CV %.3g, Direct %.3g | ratio %.2f\n'], ...
        CV_max, tightness(c), switch_prob(c), cv_deriv_std(c), direct_deriv_std(c), ...
        cv_loss_cv(c), direct_loss_cv(c), roughness_ratio(c));
end

result_path = fullfile(out_data_dir, 'ml_constraint_landscape_results.mat');
save(result_path, 'params', 'CV_grid', 'tightness', 'num_profiles', 'num_perturb', ...
    'sigma_perturb', 'eps_fd', 'switch_prob', 'cv_deriv_std', 'direct_deriv_std', ...
    'cv_loss_std', 'direct_loss_std', 'cv_loss_cv', 'direct_loss_cv', ...
    'cv_metric_change_mean', 'direct_metric_change_mean', 'metric_change_ratio', ...
    'roughness_ratio', 'active_entropy');

plot_constraint_landscape(result_path, sim_dir, paper_fig_dir);
fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', result_path);
fprintf('============================================================\n');
end

function P = random_power_profile(N, target_cv)
z = randn(N, 1);
z = z - mean(z);
z = z / max(std(z, 1), eps);
P = 1 + target_cv * z;
if min(P) <= 1e-5
    sigma_ln = sqrt(log(1 + target_cv^2));
    P = exp(-0.5 * sigma_ln^2 + sigma_ln * randn(N, 1));
end
P = normalize_power(P);
z = P - mean(P);
if norm(z) > eps
    P = 1 + target_cv * z / max(std(z, 1), eps);
    if min(P) <= 1e-5
        P = P - min(P) + 1e-5;
    end
end
P = normalize_power(P);
end

function P = normalize_power(P)
P = max(real(P(:)), 1e-10);
P = P / mean(P);
end

function cv = cv_metric(P)
P = normalize_power(P);
cv = std(P, 1) / max(mean(P), eps);
end

function [peak, active_ix] = direct_peak_metric(P)
P = normalize_power(P);
F = fft(P);
side_power = abs(F(2:end)).^2;
[peak, active_ix] = max(side_power);
peak = peak / max(abs(F(1))^2, eps);
end

function loss = cv_constraint_loss(P, CV_max)
gap = max(0, cv_metric(P) - CV_max) / max(CV_max, 0.05);
loss = gap.^2;
end

function loss = direct_constraint_loss(P, pslr_min, islr_max, kappa)
P = normalize_power(P);
pslr = compute_pslr(P, kappa);
islr = compute_islr(P, kappa);
pslr_gap = max(0, 10*log10(pslr_min) - 10*log10(pslr)) / 10;
islr_gap = max(0, 10*log10(islr) - 10*log10(islr_max)) / 10;
loss = pslr_gap.^2 + islr_gap.^2;
end

function h = normalized_entropy(indices, num_bins)
counts = accumarray(indices(:), 1, [num_bins, 1], @sum, 0);
p = counts / max(sum(counts), eps);
p = p(p > 0);
h = -sum(p .* log(p)) / log(num_bins);
end

function plot_constraint_landscape(result_path, sim_dir, paper_fig_dir)
M = load(result_path);
cv_color = [0.10 0.52 0.42];
direct_color = [0.45 0.25 0.65];
neutral = [0.30 0.30 0.30];

fig = figure('Position', [80 80 1120 780], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on; grid on; box on;
plot(M.tightness, zeros(size(M.tightness)), '-o', 'LineWidth', 2.0, ...
    'MarkerFaceColor', cv_color, 'Color', cv_color, 'DisplayName', 'CV moment');
plot(M.tightness, M.switch_prob, '--v', 'LineWidth', 2.0, ...
    'MarkerFaceColor', direct_color, 'Color', direct_color, 'DisplayName', 'Direct peak');
xlabel('Constraint tightness, \xi = 1 - CV_{max}');
ylabel('Active-index switch probability');
title('(a) Peak active-index instability');
ylim([-0.03, min(1, max(M.switch_prob) * 1.25 + 0.03)]);
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
plot(M.tightness, M.cv_deriv_std, '-o', 'LineWidth', 2.0, ...
    'MarkerFaceColor', cv_color, 'Color', cv_color, 'DisplayName', 'CV moment');
plot(M.tightness, M.direct_deriv_std, '--v', 'LineWidth', 2.0, ...
    'MarkerFaceColor', direct_color, 'Color', direct_color, 'DisplayName', 'Direct peak');
xlabel('Constraint tightness, \xi = 1 - CV_{max}');
ylabel('Std. of normalized local derivative');
title('(b) Local sensitivity variation');
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
plot(M.tightness, M.cv_metric_change_mean, '-o', 'LineWidth', 2.0, ...
    'MarkerFaceColor', cv_color, 'Color', cv_color, 'DisplayName', 'CV moment');
plot(M.tightness, M.direct_metric_change_mean, '--v', 'LineWidth', 2.0, ...
    'MarkerFaceColor', direct_color, 'Color', direct_color, 'DisplayName', 'Direct peak');
xlabel('Constraint tightness, \xi = 1 - CV_{max}');
ylabel('Mean relative metric change');
title('(c) Perturbed-metric variability');
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
yyaxis left;
plot(M.tightness, M.roughness_ratio, '-s', 'LineWidth', 2.0, ...
    'MarkerFaceColor', [0.85 0.45 0.20], 'Color', [0.85 0.45 0.20], ...
    'DisplayName', 'Direct/CV derivative std.');
ylabel('Roughness ratio');
yyaxis right;
plot(M.tightness, M.active_entropy, ':d', 'LineWidth', 2.0, ...
    'MarkerFaceColor', neutral, 'Color', neutral, ...
    'DisplayName', 'Active-index entropy');
ylabel('Normalized entropy');
xlabel('Constraint tightness, \xi = 1 - CV_{max}');
title('(d) Direct peak complexity');
set(gca, 'FontSize', 11);
legend('Location', 'northwest');

sgtitle(sprintf('Constraint-Landscape Advantage of CV Reformulation  (N=%d)', M.params.N), ...
    'FontSize', 14);

save_figure(fig, sim_dir, paper_fig_dir, 'ml_constraint_landscape', ...
    'ML_Constraint_Landscape_Result');
end

function save_figure(fig, sim_dir, paper_fig_dir, stem, paper_stem)
set(fig, 'PaperPositionMode', 'auto');
sim_png = fullfile(sim_dir, [stem '.png']);
sim_fig = fullfile(sim_dir, [stem '.fig']);
paper_pdf = fullfile(paper_fig_dir, [paper_stem '.pdf']);
savefig(fig, sim_fig);
tight_export_figure(fig, sim_png, 'Resolution', 300);
tight_export_figure(fig, paper_pdf, 'ContentType', 'image', 'Resolution', 300);
end
