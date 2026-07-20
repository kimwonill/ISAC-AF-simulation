function run_sdr_covariance_evd_learning_experiment(force_rerun)
% RUN_SDR_COVARIANCE_EVD_LEARNING_EXPERIMENT
% Supervised proof-of-concept for learning SDR-level beamforming covariance.
%
% The model predicts relaxed covariance matrices W_n from (H, CV_max).
% The predicted matrices are projected onto the PSD cone and then converted
% to rank-one beamforming covariances via the principal eigenmode.

if nargin < 1 || isempty(force_rerun)
    force_rerun = false;
end

clearvars -except force_rerun; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
result_dir = fullfile(sim_dir, 'results');
fig_dir = fullfile(sim_dir, '..', 'figures');
if exist(result_dir, 'dir') ~= 7, mkdir(result_dir); end
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
addpath(genpath(sim_dir));

label_path = fullfile(result_dir, 'supervised_cv_labels_NT4_N16_train80_test20.mat');
if exist(label_path, 'file') ~= 2
    fprintf('Label file not found. Generating a small CV-SDP label set first.\n');
    generate_supervised_cv_labels(24, 8, false);
    label_path = fullfile(result_dir, 'supervised_cv_labels_NT4_N16_train24_test8.mat');
end

out_path = fullfile(result_dir, 'ml_sdr_covariance_evd_results.mat');
if exist(out_path, 'file') == 2 && ~force_rerun
    fprintf('Result file already exists: %s\n', out_path);
    print_summary(out_path);
    plot_results(out_path, fig_dir);
    return;
end

S = load(label_path);
params = S.params;
params.sdp_quiet = true;
A = compute_steering(params);

valid = S.status_ok(:) & isfinite(S.sumrate_labels(:));
train_mask = valid & S.is_train(:);
test_mask = valid & ~S.is_train(:);

[X_all, input_info] = build_inputs(S.H_samples, S.cv_samples, valid);
[Y_all, output_info] = build_outputs(S.W_labels, valid);
train_idx = find(train_mask(valid));
test_idx = find(test_mask(valid));

X_train = X_all(train_idx, :);
Y_train = Y_all(train_idx, :);
X_test = X_all(test_idx, :);
Y_test = Y_all(test_idx, :);

fprintf('============================================================\n');
fprintf('  SDR covariance learning + EVD recovery experiment\n');
fprintf('============================================================\n');
fprintf('  Label source: %s\n', label_path);
fprintf('  Train samples: %d, test samples: %d\n', size(X_train, 1), size(X_test, 1));
fprintf('  Input dim: %d, covariance output dim: %d\n', size(X_train, 2), size(Y_train, 2));

feature_sizes = [0 16 32 64 128 256];
ridge = 1e-3;
num_models = numel(feature_sizes);

result = initialize_result(num_models);
result.feature_sizes = feature_sizes(:);
result.params = params;
result.label_path = label_path;
result.input_info = input_info;
result.output_info = output_info;
result.ridge = ridge;
result.num_train = size(X_train, 1);
result.num_test = size(X_test, 1);

[X_train_n, x_mu, x_sig] = normalize_train(X_train);
X_test_n = apply_normalize(X_test, x_mu, x_sig);
[Y_train_n, y_mu, y_sig] = normalize_train(Y_train);

label_metrics = evaluate_label_set(S, find(test_mask), A, params);
result.label = label_metrics;

for m = 1:num_models
    num_features = feature_sizes(m);
    rng(31000 + num_features, 'twister');
    feature = make_features(size(X_train_n, 2), num_features);
    train_tic = tic;
    model = train_ridge_model(X_train_n, Y_train_n, feature, ridge);
    result.train_time(m) = toc(train_tic);
    result.model_params(m) = count_model_params(model, feature, x_mu, x_sig, y_mu, y_sig);

    infer_tic = tic;
    Y_pred_n = feature_map(X_test_n, feature) * model.coef;
    Y_pred = Y_pred_n .* y_sig + y_mu;
    result.inference_time_per_sample(m) = toc(infer_tic) / max(size(X_test_n, 1), 1);
    result.cov_nrmse(m) = sqrt(mean((Y_pred(:) - Y_test(:)).^2)) / max(std(Y_test(:), 1), eps);

    eval_result = evaluate_predictions(Y_pred, S, find(test_mask), output_info, A, params);
    fields = fieldnames(eval_result);
    for f = 1:numel(fields)
        result.(fields{f})(m) = eval_result.(fields{f});
    end

    fprintf(['  features=%3d | params=%8.0f | nRMSE=%.3f | ' ...
             'cov-feas %.2f | EVD-feas %.2f | scaled %.2f | rate %.2f%% | top/tr %.3f\n'], ...
        num_features, result.model_params(m), result.cov_nrmse(m), ...
        result.cov_cv_feas(m), result.evd_cv_feas(m), result.scaled_evd_cv_feas(m), ...
        100 * result.scaled_evd_rate_ratio(m), result.pred_toptrace_mean(m));
end

save(out_path, 'result');
plot_results(out_path, fig_dir);
print_summary(out_path);

fprintf('------------------------------------------------------------\n');
fprintf('Saved source data: %s\n', out_path);
fprintf('Saved figure: %s\n', fullfile(fig_dir, 'ML_SDR_Covariance_EVD_Result.pdf'));
fprintf('============================================================\n');
end

function result = initialize_result(num_models)
names = {'model_params','train_time','inference_time_per_sample','cov_nrmse', ...
    'cov_cv_feas','cov_direct_feas','evd_cv_feas','evd_direct_feas', ...
    'scaled_evd_cv_feas','scaled_evd_direct_feas', ...
    'cov_rate_ratio','evd_rate_ratio','scaled_evd_rate_ratio', ...
    'cov_sumrate','evd_sumrate','scaled_evd_sumrate', ...
    'cov_pslr_dB','evd_pslr_dB','scaled_evd_pslr_dB', ...
    'cov_islr_dB','evd_islr_dB','scaled_evd_islr_dB', ...
    'pred_toptrace_mean','pred_eig2eig1_mean','evd_power_retained'};
for i = 1:numel(names)
    result.(names{i}) = nan(num_models, 1);
end
end

function [X, info] = build_inputs(H_samples, cv_samples, valid)
idx = find(valid);
num = numel(idx);
sample = H_samples(:, :, :, idx(1));
input_dim = 2 * numel(sample) + 3;
X = zeros(num, input_dim);
for i = 1:num
    H = H_samples(:, :, :, idx(i));
    hvec = [real(H(:)); imag(H(:))].';
    cv = cv_samples(idx(i));
    X(i, :) = [hvec, cv, cv^2, double(cv == 0)];
end
info.input_dim = input_dim;
info.includes = 'real(H), imag(H), CV, CV^2, 1{CV=0}';
end

function [Y, info] = build_outputs(W_labels, valid)
idx = find(valid);
W0 = W_labels(:, :, :, idx(1));
out_dim = 2 * numel(W0);
Y = zeros(numel(idx), out_dim);
for i = 1:numel(idx)
    W = W_labels(:, :, :, idx(i));
    Y(i, :) = [real(W(:)); imag(W(:))].';
end
info.NT = size(W0, 1);
info.N = size(W0, 3);
info.output_dim = out_dim;
end

function [Xn, mu, sig] = normalize_train(X)
mu = mean(X, 1);
sig = std(X, 0, 1);
sig(sig < 1e-10) = 1;
Xn = (X - mu) ./ sig;
end

function Xn = apply_normalize(X, mu, sig)
Xn = (X - mu) ./ sig;
end

function feature = make_features(input_dim, num_features)
feature.num_features = num_features;
if num_features > 0
    feature.W = randn(input_dim, num_features) / sqrt(input_dim);
    feature.b = 2 * rand(1, num_features) - 1;
else
    feature.W = zeros(input_dim, 0);
    feature.b = zeros(1, 0);
end
end

function Phi = feature_map(X, feature)
if feature.num_features > 0
    Phi_hidden = tanh(X * feature.W + feature.b);
else
    Phi_hidden = zeros(size(X, 1), 0);
end
Phi = [ones(size(X, 1), 1), X, Phi_hidden];
end

function model = train_ridge_model(X, Y, feature, ridge)
Phi = feature_map(X, feature);
G = Phi' * Phi + ridge * eye(size(Phi, 2));
model.coef = G \ (Phi' * Y);
end

function n = count_model_params(model, feature, x_mu, x_sig, y_mu, y_sig)
n = numel(model.coef) + numel(feature.W) + numel(feature.b) + ...
    numel(x_mu) + numel(x_sig) + numel(y_mu) + numel(y_sig);
end

function eval_result = evaluate_predictions(Y_pred, S, original_test_idx, output_info, A, params)
num = size(Y_pred, 1);
cov_metrics = empty_metric_arrays(num);
evd_metrics = empty_metric_arrays(num);
scaled_evd_metrics = empty_metric_arrays(num);
rank_toptrace = nan(num, 1);
rank_eig2eig1 = nan(num, 1);
power_retained = nan(num, 1);

for i = 1:num
    W_pred = vector_to_covariance(Y_pred(i, :), output_info);
    W_cov = project_covariance_tensor(W_pred, params.P_max);
    W_evd = principal_evd_covariance(W_cov, false);
    W_scaled_evd = principal_evd_covariance(W_cov, true);

    cv = S.cv_samples(original_test_idx(i));
    cov_metrics = fill_metric(cov_metrics, i, W_cov, S.H_samples(:, :, :, original_test_idx(i)), A, params, cv);
    evd_metrics = fill_metric(evd_metrics, i, W_evd, S.H_samples(:, :, :, original_test_idx(i)), A, params, cv);
    scaled_evd_metrics = fill_metric(scaled_evd_metrics, i, W_scaled_evd, S.H_samples(:, :, :, original_test_idx(i)), A, params, cv);

    rstats = compute_rank_stats(W_cov);
    rank_toptrace(i) = rstats.top_to_trace_mean;
    rank_eig2eig1(i) = rstats.eig2_over_eig1_mean;
    power_retained(i) = tensor_trace(W_evd) / max(tensor_trace(W_cov), eps);
end

label_sumrate = S.sumrate_labels(original_test_idx);
eval_result.cov_cv_feas = mean(cov_metrics.cv_feasible);
eval_result.cov_direct_feas = mean(cov_metrics.direct_feasible);
eval_result.evd_cv_feas = mean(evd_metrics.cv_feasible);
eval_result.evd_direct_feas = mean(evd_metrics.direct_feasible);
eval_result.scaled_evd_cv_feas = mean(scaled_evd_metrics.cv_feasible);
eval_result.scaled_evd_direct_feas = mean(scaled_evd_metrics.direct_feasible);
eval_result.cov_rate_ratio = mean(cov_metrics.sumrate ./ max(label_sumrate, eps), 'omitnan');
eval_result.evd_rate_ratio = mean(evd_metrics.sumrate ./ max(label_sumrate, eps), 'omitnan');
eval_result.scaled_evd_rate_ratio = mean(scaled_evd_metrics.sumrate ./ max(label_sumrate, eps), 'omitnan');
eval_result.cov_sumrate = mean(cov_metrics.sumrate, 'omitnan');
eval_result.evd_sumrate = mean(evd_metrics.sumrate, 'omitnan');
eval_result.scaled_evd_sumrate = mean(scaled_evd_metrics.sumrate, 'omitnan');
eval_result.cov_pslr_dB = 10 * log10(mean(cov_metrics.pslr, 'omitnan'));
eval_result.evd_pslr_dB = 10 * log10(mean(evd_metrics.pslr, 'omitnan'));
eval_result.scaled_evd_pslr_dB = 10 * log10(mean(scaled_evd_metrics.pslr, 'omitnan'));
eval_result.cov_islr_dB = 10 * log10(mean(cov_metrics.islr, 'omitnan'));
eval_result.evd_islr_dB = 10 * log10(mean(evd_metrics.islr, 'omitnan'));
eval_result.scaled_evd_islr_dB = 10 * log10(mean(scaled_evd_metrics.islr, 'omitnan'));
eval_result.pred_toptrace_mean = mean(rank_toptrace, 'omitnan');
eval_result.pred_eig2eig1_mean = mean(rank_eig2eig1, 'omitnan');
eval_result.evd_power_retained = mean(power_retained, 'omitnan');
end

function metrics = evaluate_label_set(S, original_test_idx, A, params)
num = numel(original_test_idx);
arr = empty_metric_arrays(num);
for i = 1:num
    ix = original_test_idx(i);
    arr = fill_metric(arr, i, S.W_labels(:, :, :, ix), S.H_samples(:, :, :, ix), A, params, S.cv_samples(ix));
end
metrics.cv_feas = mean(arr.cv_feasible);
metrics.direct_feas = mean(arr.direct_feasible);
metrics.sumrate = mean(arr.sumrate, 'omitnan');
metrics.pslr_dB = 10 * log10(mean(arr.pslr, 'omitnan'));
metrics.islr_dB = 10 * log10(mean(arr.islr, 'omitnan'));
end

function arr = empty_metric_arrays(num)
arr.sumrate = nan(num, 1);
arr.pslr = nan(num, 1);
arr.islr = nan(num, 1);
arr.cv_feasible = false(num, 1);
arr.direct_feasible = false(num, 1);
end

function arr = fill_metric(arr, i, W, H, A, params, CV_max)
metrics = evaluate_covariance_candidate(H, W, A, params);
[pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params, false);
tol = 1e-5;
arr.sumrate(i) = metrics.sumrate;
arr.pslr(i) = min(metrics.pslr_per_target);
arr.islr(i) = max(metrics.islr_per_target);
arr.cv_feasible(i) = all(metrics.user_rate >= params.Q - tol) && ...
    all(metrics.mu_p_per_target >= params.P_des - tol) && ...
    all(metrics.cv_per_target <= CV_max + 1e-4);
arr.direct_feasible(i) = all(metrics.user_rate >= params.Q - tol) && ...
    all(metrics.mu_p_per_target >= params.P_des - tol) && ...
    all(metrics.pslr_per_target >= pslr_min * (1 - 1e-4)) && ...
    all(metrics.islr_per_target <= islr_max * (1 + 1e-4));
end

function W = vector_to_covariance(y, info)
num_complex = numel(y) / 2;
z = complex(y(1:num_complex), y(num_complex+1:end));
W = reshape(z, info.NT, info.NT, info.N);
end

function Wp = project_covariance_tensor(W, Pmax)
[NT, ~, N] = size(W);
Wp = zeros(NT, NT, N);
for n = 1:N
    Wn = 0.5 * (W(:, :, n) + W(:, :, n)');
    [V, D] = eig(Wn);
    d = max(real(diag(D)), 0);
    Wp(:, :, n) = V * diag(d) * V';
    Wp(:, :, n) = 0.5 * (Wp(:, :, n) + Wp(:, :, n)');
end
tr = tensor_trace(Wp);
if tr > Pmax
    Wp = Wp * (Pmax / tr);
end
end

function Wr = principal_evd_covariance(W, preserve_subcarrier_trace)
[NT, ~, N] = size(W);
Wr = zeros(NT, NT, N);
for n = 1:N
    Wn = 0.5 * (W(:, :, n) + W(:, :, n)');
    [V, D] = eig(Wn);
    [lambda, idx] = max(real(diag(D)));
    if lambda > 0
        u = V(:, idx);
        if preserve_subcarrier_trace
            lambda = max(real(trace(Wn)), 0);
        end
        Wr(:, :, n) = lambda * (u * u');
    end
end
end

function tr = tensor_trace(W)
tr = 0;
for n = 1:size(W, 3)
    tr = tr + real(trace(W(:, :, n)));
end
end

function metrics = evaluate_covariance_candidate(H, W, A, params)
K = params.K;
N = params.N;
L = params.L;

R = zeros(K, N);
for n = 1:N
    Wn = W(:, :, n);
    for k = 1:K
        hk = H(:, k, n);
        g = max(real(hk' * Wn * hk), 0);
        R(k, n) = log2(1 + g / params.sigma2);
    end
end

alpha = update_alpha(H, W, params);
metrics.alpha = alpha;
metrics.sumrate = sum(alpha .* R, 'all');
metrics.user_rate = sum(alpha .* R, 2);
metrics.pslr_per_target = nan(L, 1);
metrics.islr_per_target = nan(L, 1);
metrics.cv_per_target = nan(L, 1);
metrics.mu_p_per_target = nan(L, 1);
for l = 1:L
    Pn = compute_directional_power(W, A(:, l));
    mu = mean(Pn);
    metrics.pslr_per_target(l) = compute_pslr(Pn, params.kappa);
    metrics.islr_per_target(l) = compute_islr(Pn, params.kappa);
    metrics.mu_p_per_target(l) = mu;
    metrics.cv_per_target(l) = sqrt(mean((Pn - mu).^2)) / max(mu, eps);
end
end

function plot_results(out_path, fig_dir)
S = load(out_path, 'result');
R = S.result;
x = max(R.model_params, 1);

fig = figure('Position', [90 90 1180 740], 'Color', 'w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(tl, 1);
hold on; grid on; box on;
plot(x, R.cov_cv_feas, '-o', 'LineWidth', 2.0, 'MarkerFaceColor', [0.20 0.45 0.75], ...
    'DisplayName', 'Relaxed covariance');
plot(x, R.evd_cv_feas, '--s', 'LineWidth', 2.0, 'MarkerFaceColor', [0.85 0.35 0.25], ...
    'DisplayName', 'EVD');
plot(x, R.scaled_evd_cv_feas, '-.^', 'LineWidth', 2.0, 'MarkerFaceColor', [0.30 0.60 0.35], ...
    'DisplayName', 'Scaled EVD');
yline(R.label.cv_feas, ':', 'LineWidth', 1.4, 'Color', [0.15 0.15 0.15], ...
    'DisplayName', 'CV-SDP label');
set(gca, 'XScale', 'log');
xlabel('Model parameters');
ylabel('CV feasibility rate');
title('(a) CV feasibility');
legend('Location', 'southeast');

nexttile(tl, 2);
hold on; grid on; box on;
plot(x, R.cov_direct_feas, '-o', 'LineWidth', 2.0, 'MarkerFaceColor', [0.20 0.45 0.75], ...
    'DisplayName', 'Relaxed covariance');
plot(x, R.evd_direct_feas, '--s', 'LineWidth', 2.0, 'MarkerFaceColor', [0.85 0.35 0.25], ...
    'DisplayName', 'EVD');
plot(x, R.scaled_evd_direct_feas, '-.^', 'LineWidth', 2.0, 'MarkerFaceColor', [0.30 0.60 0.35], ...
    'DisplayName', 'Scaled EVD');
yline(R.label.direct_feas, ':', 'LineWidth', 1.4, 'Color', [0.15 0.15 0.15], ...
    'DisplayName', 'CV-SDP label');
set(gca, 'XScale', 'log');
xlabel('Model parameters');
ylabel('Equivalent PSLR/ISLR feasibility rate');
title('(b) Direct-metric feasibility');

nexttile(tl, 3);
hold on; grid on; box on;
plot(x, 100 * R.cov_rate_ratio, '-o', 'LineWidth', 2.0, 'MarkerFaceColor', [0.20 0.45 0.75], ...
    'DisplayName', 'Relaxed covariance');
plot(x, 100 * R.evd_rate_ratio, '--s', 'LineWidth', 2.0, 'MarkerFaceColor', [0.85 0.35 0.25], ...
    'DisplayName', 'EVD');
plot(x, 100 * R.scaled_evd_rate_ratio, '-.^', 'LineWidth', 2.0, 'MarkerFaceColor', [0.30 0.60 0.35], ...
    'DisplayName', 'Scaled EVD');
set(gca, 'XScale', 'log');
xlabel('Model parameters');
ylabel('Rate retained vs. CV-SDP label (%)');
title('(c) Communication performance');

nexttile(tl, 4);
hold on; grid on; box on;
yyaxis left;
plot(x, R.cov_nrmse, '-o', 'LineWidth', 2.0, 'MarkerFaceColor', [0.25 0.55 0.35]);
ylabel('Covariance nRMSE');
yyaxis right;
plot(x, R.pred_toptrace_mean, '--s', 'LineWidth', 2.0, 'MarkerFaceColor', [0.55 0.35 0.75]);
ylabel('Mean \lambda_1 / trace');
set(gca, 'XScale', 'log');
xlabel('Model parameters');
title('(d) Prediction error and rank concentration');

safe_export(fig, fullfile(fig_dir, 'ML_SDR_Covariance_EVD_Result.pdf'), 'pdf');
safe_export(fig, fullfile(fig_dir, 'ML_SDR_Covariance_EVD_Result.png'), 'png');
end

function print_summary(out_path)
S = load(out_path, 'result');
R = S.result;
[best_feas, idx] = max(R.scaled_evd_cv_feas);
fprintf('\nSummary: SDR covariance learning + EVD\n');
fprintf('  Train/test samples: %d/%d\n', R.num_train, R.num_test);
fprintf('  Label CV/direct feasibility: %.2f / %.2f\n', R.label.cv_feas, R.label.direct_feas);
fprintf('  Best scaled-EVD CV feasibility: %.2f at %d random features\n', best_feas, R.feature_sizes(idx));
fprintf('  At best feasibility: direct-feas %.2f, rate %.2f%%, PSLR %.2f dB, ISLR %.2f dB\n', ...
    R.scaled_evd_direct_feas(idx), 100 * R.scaled_evd_rate_ratio(idx), ...
    R.scaled_evd_pslr_dB(idx), R.scaled_evd_islr_dB(idx));
fprintf('  Fastest inference/sample: %.3g ms\n', 1e3 * min(R.inference_time_per_sample));
fprintf('\nfeatures  params      nRMSE  covFeas  evdFeas  scEVDFeas  scDirect  scRate%%  topTrace\n');
for i = 1:numel(R.feature_sizes)
    fprintf('%8d  %8.0f  %7.3f  %7.2f  %7.2f  %9.2f  %8.2f  %7.1f  %8.3f\n', ...
        R.feature_sizes(i), R.model_params(i), R.cov_nrmse(i), ...
        R.cov_cv_feas(i), R.evd_cv_feas(i), R.scaled_evd_cv_feas(i), ...
        R.scaled_evd_direct_feas(i), 100 * R.scaled_evd_rate_ratio(i), ...
        R.pred_toptrace_mean(i));
end
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        exportgraphics(fig, filename, 'ContentType', 'vector');
    else
        exportgraphics(fig, filename, 'Resolution', 300);
    end
catch
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-vector');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
end
