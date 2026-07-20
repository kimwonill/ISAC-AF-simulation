function run_ml_constraint_surrogate_experiment()
% RUN_ML_CONSTRAINT_SURROGATE_EXPERIMENT
% Surrogate-learning test for the learnability of sensing constraints.
%
% A fixed random-feature regressor learns either the smooth CV statistic or
% the direct peak-sidelobe statistic from raw directional-power profiles P_n.
% Lower test error with the same samples/model size is evidence that the CV
% reformulation is easier to learn as an ML constraint surrogate.

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
num_train_pool = 3200;
num_test = 3200;
train_sizes = [50, 100, 200, 500, 1000, 2000];
feature_sizes = [8, 16, 32, 64, 128];
main_features = 32;
main_train_size = 500;
num_repeats = 8;
ridge = 1e-4;

rng(21001, 'twister');
[X_pool, y_cv_pool, y_peak_pool] = make_surrogate_dataset(N, num_train_pool);
[X_test, y_cv_test, y_peak_test] = make_surrogate_dataset(N, num_test);

fprintf('============================================================\n');
fprintf('  Constraint surrogate-learning experiment\n');
fprintf('============================================================\n');
fprintf('  N=%d, train pool=%d, test=%d, repeats=%d\n', ...
    N, num_train_pool, num_test, num_repeats);
fprintf('------------------------------------------------------------\n');

sample_result = sample_efficiency_sweep(X_pool, y_cv_pool, y_peak_pool, ...
    X_test, y_cv_test, y_peak_test, train_sizes, main_features, num_repeats, ridge);
model_result = model_size_sweep(X_pool, y_cv_pool, y_peak_pool, ...
    X_test, y_cv_test, y_peak_test, main_train_size, feature_sizes, num_repeats, ridge);
scatter_result = scatter_fit(X_pool, y_cv_pool, y_peak_pool, ...
    X_test, y_cv_test, y_peak_test, main_train_size, main_features, ridge);

result_path = fullfile(out_data_dir, 'ml_constraint_surrogate_results.mat');
save(result_path, 'params', 'N', 'num_train_pool', 'num_test', ...
    'train_sizes', 'feature_sizes', 'main_features', 'main_train_size', ...
    'num_repeats', 'ridge', 'sample_result', 'model_result', 'scatter_result');

plot_constraint_surrogate(result_path, sim_dir, paper_fig_dir);
print_surrogate_summary(result_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', result_path);
fprintf('============================================================\n');
end

function [X, y_cv, y_peak] = make_surrogate_dataset(N, num_samples)
X = zeros(num_samples, N);
y_cv = zeros(num_samples, 1);
y_peak = zeros(num_samples, 1);
for i = 1:num_samples
    temp = 0.05 + 1.55 * rand();
    z = temp * randn(N, 1);
    P = z_to_power(z);
    X(i, :) = (P - 1).';
    y_cv(i) = log1p(cv_metric(P));
    y_peak(i) = log1p(direct_peak_metric(P));
end
end

function result = sample_efficiency_sweep(X_pool, y_cv_pool, y_peak_pool, ...
    X_test, y_cv_test, y_peak_test, train_sizes, num_features, num_repeats, ridge)
num_sizes = numel(train_sizes);
cv_err = nan(num_sizes, num_repeats);
peak_err = nan(num_sizes, num_repeats);
for s = 1:num_sizes
    n_train = train_sizes(s);
    for r = 1:num_repeats
        rng(22000 + 100*s + r, 'twister');
        idx = randperm(size(X_pool, 1), n_train);
        feature = make_random_features(size(X_pool, 2), num_features);
        cv_err(s, r) = train_eval_surrogate(X_pool(idx, :), y_cv_pool(idx), ...
            X_test, y_cv_test, feature, ridge);
        peak_err(s, r) = train_eval_surrogate(X_pool(idx, :), y_peak_pool(idx), ...
            X_test, y_peak_test, feature, ridge);
    end
    fprintf('  Samples=%4d | nRMSE CV %.3f | Direct peak %.3f\n', ...
        n_train, mean(cv_err(s, :)), mean(peak_err(s, :)));
end
result.cv_err = cv_err;
result.peak_err = peak_err;
result.cv_mean = mean(cv_err, 2, 'omitnan');
result.peak_mean = mean(peak_err, 2, 'omitnan');
result.cv_std = std(cv_err, 0, 2, 'omitnan');
result.peak_std = std(peak_err, 0, 2, 'omitnan');
result.error_ratio = result.peak_mean ./ max(result.cv_mean, eps);
end

function result = model_size_sweep(X_pool, y_cv_pool, y_peak_pool, ...
    X_test, y_cv_test, y_peak_test, n_train, feature_sizes, num_repeats, ridge)
num_sizes = numel(feature_sizes);
cv_err = nan(num_sizes, num_repeats);
peak_err = nan(num_sizes, num_repeats);
for s = 1:num_sizes
    num_features = feature_sizes(s);
    for r = 1:num_repeats
        rng(23000 + 100*s + r, 'twister');
        idx = randperm(size(X_pool, 1), n_train);
        feature = make_random_features(size(X_pool, 2), num_features);
        cv_err(s, r) = train_eval_surrogate(X_pool(idx, :), y_cv_pool(idx), ...
            X_test, y_cv_test, feature, ridge);
        peak_err(s, r) = train_eval_surrogate(X_pool(idx, :), y_peak_pool(idx), ...
            X_test, y_peak_test, feature, ridge);
    end
    fprintf('  Features=%3d | nRMSE CV %.3f | Direct peak %.3f\n', ...
        num_features, mean(cv_err(s, :)), mean(peak_err(s, :)));
end
result.cv_err = cv_err;
result.peak_err = peak_err;
result.cv_mean = mean(cv_err, 2, 'omitnan');
result.peak_mean = mean(peak_err, 2, 'omitnan');
result.cv_std = std(cv_err, 0, 2, 'omitnan');
result.peak_std = std(peak_err, 0, 2, 'omitnan');
result.error_ratio = result.peak_mean ./ max(result.cv_mean, eps);
end

function result = scatter_fit(X_pool, y_cv_pool, y_peak_pool, ...
    X_test, y_cv_test, y_peak_test, n_train, num_features, ridge)
rng(24001, 'twister');
idx = randperm(size(X_pool, 1), n_train);
feature = make_random_features(size(X_pool, 2), num_features);
[~, pred_cv] = train_eval_surrogate(X_pool(idx, :), y_cv_pool(idx), ...
    X_test, y_cv_test, feature, ridge);
[~, pred_peak] = train_eval_surrogate(X_pool(idx, :), y_peak_pool(idx), ...
    X_test, y_peak_test, feature, ridge);
plot_idx = randperm(size(X_test, 1), min(700, size(X_test, 1)));
result.true_cv = y_cv_test(plot_idx);
result.pred_cv = pred_cv(plot_idx);
result.true_peak = y_peak_test(plot_idx);
result.pred_peak = pred_peak(plot_idx);
end

function feature = make_random_features(input_dim, num_features)
feature.W = randn(input_dim, num_features) / sqrt(input_dim);
feature.b = 2 * rand(1, num_features) - 1;
end

function [nrmse, pred_test] = train_eval_surrogate(X_train, y_train, X_test, y_test, feature, ridge)
mu = mean(y_train);
sig = std(y_train, 1);
if sig < 1e-10
    sig = 1;
end
y_train_n = (y_train - mu) / sig;
Phi_train = feature_map(X_train, feature);
Phi_test = feature_map(X_test, feature);
coef = (Phi_train' * Phi_train + ridge * eye(size(Phi_train, 2))) \ (Phi_train' * y_train_n);
pred_test_n = Phi_test * coef;
pred_test = pred_test_n * sig + mu;
nrmse = sqrt(mean((pred_test - y_test).^2)) / max(std(y_test, 1), eps);
end

function Phi = feature_map(X, feature)
Phi_hidden = tanh(X * feature.W + feature.b);
Phi = [ones(size(X, 1), 1), X, Phi_hidden];
end

function P = z_to_power(z)
z = z(:) - max(z);
q = exp(z);
P = numel(z) * q / sum(q);
end

function cv = cv_metric(P)
P = P(:);
cv = std(P, 1) / max(mean(P), eps);
end

function peak = direct_peak_metric(P)
P = P(:);
F = fft(P);
side_power = abs(F(2:end)).^2;
peak = max(side_power) / max(abs(F(1))^2, eps);
end

function plot_constraint_surrogate(result_path, sim_dir, paper_fig_dir)
M = load(result_path);
cv_color = [0.10 0.52 0.42];
direct_color = [0.45 0.25 0.65];
ratio_color = [0.85 0.45 0.20];

fig = figure('Position', [90 90 1120 780], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on; grid on; box on;
errorbar(M.train_sizes, M.sample_result.cv_mean, M.sample_result.cv_std, '-o', ...
    'LineWidth', 2.0, 'MarkerFaceColor', cv_color, 'Color', cv_color, ...
    'DisplayName', 'CV moment');
errorbar(M.train_sizes, M.sample_result.peak_mean, M.sample_result.peak_std, '--v', ...
    'LineWidth', 2.0, 'MarkerFaceColor', direct_color, 'Color', direct_color, ...
    'DisplayName', 'Direct peak');
set(gca, 'XScale', 'log');
xlabel('Training samples');
ylabel('Test nRMSE');
title('(a) Sample efficiency');
legend('Location', 'northeast');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
errorbar(M.feature_sizes, M.model_result.cv_mean, M.model_result.cv_std, '-o', ...
    'LineWidth', 2.0, 'MarkerFaceColor', cv_color, 'Color', cv_color, ...
    'DisplayName', 'CV moment');
errorbar(M.feature_sizes, M.model_result.peak_mean, M.model_result.peak_std, '--v', ...
    'LineWidth', 2.0, 'MarkerFaceColor', direct_color, 'Color', direct_color, ...
    'DisplayName', 'Direct peak');
set(gca, 'XScale', 'log');
xlabel('Random features');
ylabel('Test nRMSE');
title(sprintf('(b) Model-size efficiency (%d samples)', M.main_train_size));
legend('Location', 'northeast');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
plot(M.sample_result.error_ratio, '-s', 'LineWidth', 2.0, ...
    'MarkerFaceColor', ratio_color, 'Color', ratio_color);
xticks(1:numel(M.train_sizes));
xticklabels(arrayfun(@num2str, M.train_sizes, 'UniformOutput', false));
xlabel('Training samples');
ylabel('Direct/CV nRMSE ratio');
title('(c) Direct learning error multiplier');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
scatter(M.scatter_result.true_cv, M.scatter_result.pred_cv, 12, cv_color, 'filled', ...
    'MarkerFaceAlpha', 0.35, 'DisplayName', 'CV moment');
scatter(M.scatter_result.true_peak, M.scatter_result.pred_peak, 12, direct_color, 'filled', ...
    'MarkerFaceAlpha', 0.35, 'DisplayName', 'Direct peak');
lims = axis;
mn = min(lims([1 3]));
mx = max(lims([2 4]));
plot([mn mx], [mn mx], 'k:', 'LineWidth', 1.2, 'DisplayName', 'Ideal');
xlabel('True normalized metric');
ylabel('Predicted normalized metric');
title('(d) Surrogate prediction scatter');
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

sgtitle(sprintf('Constraint Surrogate Learnability of CV Reformulation  (N=%d)', M.N), ...
    'FontSize', 14);
save_figure(fig, sim_dir, paper_fig_dir, 'ml_constraint_surrogate', ...
    'ML_Constraint_Surrogate_Result');
end

function save_figure(fig, sim_dir, paper_fig_dir, stem, paper_stem)
set(fig, 'PaperPositionMode', 'auto');
sim_png = fullfile(sim_dir, [stem '.png']);
sim_fig = fullfile(sim_dir, [stem '.fig']);
paper_pdf = fullfile(paper_fig_dir, [paper_stem '.pdf']);
savefig(fig, sim_fig);
exportgraphics(fig, sim_png, 'Resolution', 300);
exportgraphics(fig, paper_pdf, 'ContentType', 'image', 'Resolution', 300);
end

function print_surrogate_summary(result_path)
M = load(result_path);
[~, ix_500] = min(abs(M.train_sizes - M.main_train_size));
fprintf('============================================================\n');
fprintf('  Constraint surrogate-learning summary\n');
fprintf('============================================================\n');
fprintf('At %d samples, nRMSE: CV %.3f, Direct %.3f, ratio %.2f\n', ...
    M.train_sizes(ix_500), M.sample_result.cv_mean(ix_500), ...
    M.sample_result.peak_mean(ix_500), M.sample_result.error_ratio(ix_500));
fprintf('Mean Direct/CV sample-efficiency error ratio: %.2f\n', ...
    mean(M.sample_result.error_ratio, 'omitnan'));
fprintf('Mean Direct/CV model-size error ratio:        %.2f\n', ...
    mean(M.model_result.error_ratio, 'omitnan'));
end
