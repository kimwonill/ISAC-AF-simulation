function run_ml_inference_experiments()
% RUN_ML_INFERENCE_EXPERIMENTS
% Offline-trained ML inference baselines for the CV and direct constraints.
%
% Offline stage:
%   Expert SDP/SCA solutions are projected onto the same low-dimensional
%   covariance policy used by the CEM baseline. A random-feature ridge model
%   learns the residual from a generic analytic policy prior.
%
% Online stage:
%   For each test channel and operating point, runtime includes feature
%   extraction, one neural forward pass, covariance construction, and a fixed
%   deterministic feasibility projection. No neighboring-CV solution is reused.

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
params.warm_start_cv = false;
params.sdp_quiet = true;

CV_grid = 0:0.1:1.0;
train_CV_grid = [0, 0.25, 0.5, 0.75, 1.0];
num_train_mc = 4;
num_test_mc = params.num_mc;

model_path = fullfile(out_data_dir, 'ml_inference_models.mat');
source_path = fullfile(out_data_dir, 'ml_inference_results.mat');
reuse_models = true;

opts = struct();
opts.model_version = 1;
opts.cv_hidden = 64;
opts.direct_hidden = 256;
opts.ridge_lambda = 1e-2;
opts.projection_base_steps_cv = 1;
opts.projection_base_steps_direct = 6;
opts.projection_extra_steps_cv = 6;
opts.projection_extra_steps_direct = 24;
opts.timing_repeats = 80;
opts.constraint_tol = 1e-4;

fprintf('============================================================\n');
fprintf('  Offline-trained ML inference experiment\n');
fprintf('============================================================\n');
fprintf('  Train MC=%d, train CV=[%s]\n', num_train_mc, num2str(train_CV_grid));
fprintf('  Test MC=%d, test CV=[%s]\n', num_test_mc, num2str(CV_grid));
fprintf('  CV hidden=%d, Direct hidden=%d, timing repeats=%d\n', ...
    opts.cv_hidden, opts.direct_hidden, opts.timing_repeats);
fprintf('------------------------------------------------------------\n');

if reuse_models && exist(model_path, 'file') == 2
    M = load(model_path);
    if isfield(M, 'opts') && isfield(M.opts, 'model_version') && ...
            M.opts.model_version == opts.model_version
        fprintf('Reusing offline-trained models: %s\n', model_path);
        cv_model = M.cv_model;
        direct_model = M.direct_model;
        training_info = M.training_info;
    else
        [cv_model, direct_model, training_info] = train_models(params, train_CV_grid, num_train_mc, opts);
        save(model_path, 'cv_model', 'direct_model', 'training_info', 'opts', 'params', 'train_CV_grid', 'num_train_mc');
    end
else
    [cv_model, direct_model, training_info] = train_models(params, train_CV_grid, num_train_mc, opts);
    save(model_path, 'cv_model', 'direct_model', 'training_info', 'opts', 'params', 'train_CV_grid', 'num_train_mc');
end

num_cv = numel(CV_grid);
ml_cv_sumrate_grid = nan(num_cv, num_test_mc);
ml_cv_pslr_lin_grid = nan(num_cv, num_test_mc);
ml_cv_islr_lin_grid = nan(num_cv, num_test_mc);
ml_cv_time_grid = nan(num_cv, num_test_mc);
ml_cv_feasible_grid = false(num_cv, num_test_mc);
ml_cv_status_grid = strings(num_cv, num_test_mc);

ml_direct_sumrate_grid = nan(num_cv, num_test_mc);
ml_direct_pslr_lin_grid = nan(num_cv, num_test_mc);
ml_direct_islr_lin_grid = nan(num_cv, num_test_mc);
ml_direct_time_grid = nan(num_cv, num_test_mc);
ml_direct_feasible_grid = false(num_cv, num_test_mc);
ml_direct_status_grid = strings(num_cv, num_test_mc);

t_global = tic;
total_runs = 2 * num_cv * num_test_mc;
run_count = 0;

for mc = 1:num_test_mc
    rng(mc, 'twister');
    H = generate_channel(params);

    for c = 1:num_cv
        CV_max = CV_grid(c);
        xi = 1 - CV_max;
        [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);

        run_count = run_count + 1;
        cv_constraint = struct('CV_max', CV_max, 'CV_hint', CV_max);
        cv_result = run_inference_model(H, 'cv', cv_constraint, cv_model, params, opts);
        ml_cv_time_grid(c, mc) = cv_result.elapsed;
        [ml_cv_sumrate_grid(c, mc), ml_cv_pslr_lin_grid(c, mc), ml_cv_islr_lin_grid(c, mc)] = ...
            summarize_result(cv_result);
        ml_cv_feasible_grid(c, mc) = cv_result.feasible;
        ml_cv_status_grid(c, mc) = string(cv_result.status);
        print_progress('ML-CV inf.', run_count, total_runs, mc, num_test_mc, CV_max, xi, ...
            cv_result.elapsed, cv_result.sumrate, min(cv_result.pslr_per_target), cv_result.status, t_global);

        run_count = run_count + 1;
        direct_constraint = struct('pslr_min', pslr_min, 'islr_max', islr_max, 'CV_hint', CV_max);
        direct_result = run_inference_model(H, 'direct', direct_constraint, direct_model, params, opts);
        ml_direct_time_grid(c, mc) = direct_result.elapsed;
        [ml_direct_sumrate_grid(c, mc), ml_direct_pslr_lin_grid(c, mc), ml_direct_islr_lin_grid(c, mc)] = ...
            summarize_result(direct_result);
        ml_direct_feasible_grid(c, mc) = direct_result.feasible;
        ml_direct_status_grid(c, mc) = string(direct_result.status);
        print_progress('ML-Dir inf.', run_count, total_runs, mc, num_test_mc, CV_max, xi, ...
            direct_result.elapsed, direct_result.sumrate, min(direct_result.pslr_per_target), ...
            direct_result.status, t_global);
    end
end

save(source_path, ...
    'params', 'CV_grid', 'train_CV_grid', 'num_train_mc', 'num_test_mc', ...
    'opts', 'training_info', ...
    'ml_cv_sumrate_grid', 'ml_cv_pslr_lin_grid', 'ml_cv_islr_lin_grid', ...
    'ml_cv_time_grid', 'ml_cv_feasible_grid', 'ml_cv_status_grid', ...
    'ml_direct_sumrate_grid', 'ml_direct_pslr_lin_grid', 'ml_direct_islr_lin_grid', ...
    'ml_direct_time_grid', 'ml_direct_feasible_grid', 'ml_direct_status_grid');

plot_ml_inference_results(source_path, sim_dir, paper_fig_dir);
print_inference_summary(source_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved models: %s\n', model_path);
fprintf('  Saved inference source data: %s\n', source_path);
fprintf('  Total elapsed including offline training: %s\n', format_time(toc(t_global) + training_info.elapsed));
fprintf('============================================================\n');
end

function [cv_model, direct_model, info] = train_models(params, CV_grid, num_train_mc, opts)
t_train = tic;

X_cv = [];
Y_cv = [];
X_direct = [];
Y_direct = [];
cv_status = strings(numel(CV_grid), num_train_mc);
direct_status = strings(numel(CV_grid), num_train_mc);

fprintf('Offline training labels from SDP/SCA experts...\n');
total = 2 * numel(CV_grid) * num_train_mc;
count = 0;

for mc = 1:num_train_mc
    rng(1000 + mc, 'twister');
    H = generate_channel(params);
    A = compute_steering(params);
    basis = make_policy_basis(H, A, params);

    for c = 1:numel(CV_grid)
        CV_max = CV_grid(c);
        [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);

        count = count + 1;
        t = tic;
        cv_constraint = struct('CV_max', CV_max, 'CV_hint', CV_max);
        cv_result = run_proposed(H, CV_max, params);
        cv_status(c, mc) = string(cv_result.status);
        if isfinite(cv_result.sumrate)
            X_cv = [X_cv; make_features(H, 'cv', cv_constraint, params).']; %#ok<AGROW>
            Y_cv = [Y_cv; expert_to_residual(cv_result.W, H, A, basis, 'cv', cv_constraint, params).']; %#ok<AGROW>
        end
        fprintf('[train %3d/%3d] CV expert     MC %d/%d CV=%.2f | %s | %.2fs\n', ...
            count, total, mc, num_train_mc, CV_max, cv_result.status, toc(t));

        count = count + 1;
        t = tic;
        direct_constraint = struct('pslr_min', pslr_min, 'islr_max', islr_max, 'CV_hint', CV_max);
        direct_result = run_direct_sca(H, pslr_min, islr_max, params);
        direct_status(c, mc) = string(direct_result.status);
        if isfinite(direct_result.sumrate)
            W_label = direct_result.W;
        elseif isfinite(cv_result.sumrate)
            W_label = cv_result.W;
            direct_status(c, mc) = "Fallback CV expert";
        else
            W_label = [];
        end
        if ~isempty(W_label)
            X_direct = [X_direct; make_features(H, 'direct', direct_constraint, params).']; %#ok<AGROW>
            Y_direct = [Y_direct; expert_to_residual(W_label, H, A, basis, 'direct', direct_constraint, params).']; %#ok<AGROW>
        end
        fprintf('[train %3d/%3d] Direct expert MC %d/%d CV=%.2f | %s | %.2fs\n', ...
            count, total, mc, num_train_mc, CV_max, char(direct_status(c, mc)), toc(t));
    end
end

if isempty(X_cv) || isempty(X_direct)
    error('No ML training samples were collected.');
end

cv_model = fit_random_feature_ridge(X_cv, Y_cv, opts.cv_hidden, opts.ridge_lambda, 9101);
cv_model.mode = 'cv';
direct_model = fit_random_feature_ridge(X_direct, Y_direct, opts.direct_hidden, opts.ridge_lambda, 9201);
direct_model.mode = 'direct';

info.elapsed = toc(t_train);
info.num_cv_samples = size(X_cv, 1);
info.num_direct_samples = size(X_direct, 1);
info.cv_status = cv_status;
info.direct_status = direct_status;
fprintf('Offline training completed in %s (%d CV samples, %d direct samples).\n', ...
    format_time(info.elapsed), info.num_cv_samples, info.num_direct_samples);
end

function model = fit_random_feature_ridge(X, Y, hidden_dim, lambda, seed)
rng(seed, 'twister');
input_mean = mean(X, 1);
input_std = std(X, 0, 1);
input_std(input_std < 1e-8) = 1;
Xs = (X - input_mean) ./ input_std;

input_dim = size(Xs, 2);
W1 = randn(hidden_dim, input_dim) / sqrt(input_dim);
b1 = 0.5 * randn(hidden_dim, 1);
Phi = tanh(Xs * W1.' + repmat(b1.', size(Xs, 1), 1));
Phi = [ones(size(Phi, 1), 1), Phi];

R = lambda * eye(size(Phi, 2));
R(1, 1) = 0;
Beta = (Phi.' * Phi + R) \ (Phi.' * Y);

model.input_mean = input_mean;
model.input_std = input_std;
model.W1 = W1;
model.b1 = b1;
model.Beta = Beta;
model.hidden_dim = hidden_dim;
model.lambda = lambda;
end

function result = run_inference_model(H, mode, constraint, model, params, opts)
repeats = opts.timing_repeats;
last = [];
t_start = tic;
for r = 1:repeats
    last = run_single_inference(H, mode, constraint, model, params, opts);
end
elapsed = toc(t_start) / repeats;

result = last;
result.elapsed = elapsed;
end

function result = run_single_inference(H, mode, constraint, model, params, opts)
A = compute_steering(params);
basis = make_policy_basis(H, A, params);
x = make_features(H, mode, constraint, params).';
z0 = initial_policy_vector(H, A, basis, mode, constraint, params);

Xs = (x - model.input_mean) ./ model.input_std;
phi = tanh(Xs * model.W1.' + model.b1.');
phi = [1, phi];
residual = phi * model.Beta;
z = z0 + residual(:);

W = policy_vector_to_covariance(z, basis, params);
W = deterministic_projection(W, mode, constraint, A, params, opts);
metrics = evaluate_candidate(H, W, A, params);
[feasible, status] = check_feasible(metrics, mode, constraint, params, opts.constraint_tol);

result.W = W;
result.alpha = metrics.alpha;
result.sumrate = metrics.sumrate;
result.pslr_per_target = metrics.pslr_per_target;
result.islr_per_target = metrics.islr_per_target;
result.cv_per_target = metrics.cv_per_target;
result.mu_p_per_target = metrics.mu_p_per_target;
result.user_rate = metrics.user_rate;
result.feasible = feasible;
result.status = status;
result.mode = mode;
result.constraint = constraint;
end

function W = deterministic_projection(W, mode, constraint, A, params, opts)
if strcmpi(mode, 'cv')
    tightness = 1 - constraint.CV_max;
    steps = opts.projection_base_steps_cv + ceil(opts.projection_extra_steps_cv * tightness);
else
    tightness = 1 - get_field(constraint, 'CV_hint', 0.5);
    steps = opts.projection_base_steps_direct + ceil(opts.projection_extra_steps_direct * tightness);
end

if steps <= 0
    return;
end

initial_ok = sensing_ok(W, mode, constraint, A, params, opts.constraint_tol);
if initial_ok
    for i = 1:steps
        sensing_ok(W, mode, constraint, A, params, opts.constraint_tol);
    end
    return;
end

W_flat = init_covariance_flat(params);
lo = 0;
hi = 1;
best = W_flat;
for i = 1:steps
    beta = 0.5 * (lo + hi);
    W_try = (1 - beta) * W + beta * W_flat;
    if sensing_ok(W_try, mode, constraint, A, params, opts.constraint_tol)
        best = W_try;
        hi = beta;
    else
        lo = beta;
    end
end
W = best;
end

function ok = sensing_ok(W, mode, constraint, A, params, tol)
L = params.L;
ok = true;
for l = 1:L
    Pn = compute_directional_power(W, A(:, l));
    mu_p = mean(Pn);
    cv = sqrt(mean((Pn - mu_p).^2)) / max(mu_p, eps);
    if mu_p < params.P_des - tol
        ok = false;
        return;
    end
    if strcmpi(mode, 'cv')
        if cv > constraint.CV_max + tol
            ok = false;
            return;
        end
    else
        pslr = compute_pslr(Pn, params.kappa);
        islr = compute_islr(Pn, params.kappa);
        if pslr < constraint.pslr_min * (1 - tol)
            ok = false;
            return;
        end
        if isfinite(constraint.islr_max) && islr > constraint.islr_max * (1 + tol)
            ok = false;
            return;
        end
    end
end
end

function x = make_features(H, mode, constraint, params)
G = squeeze(sum(abs(H).^2, 1));
snr_feat = log1p(G / params.sigma2);
snr_feat = snr_feat / max(max(snr_feat(:)), 1);
alpha0 = init_alpha(H, params);
sub_max = max(snr_feat, [], 1);
sub_mean = mean(snr_feat, 1);
user_mean = mean(snr_feat, 2).';

if strcmpi(mode, 'cv')
    cv_hint = constraint.CV_max;
    [pslr_min, islr_max] = direct_thresholds_from_cv(cv_hint, params);
else
    cv_hint = get_field(constraint, 'CV_hint', 0.5);
    pslr_min = constraint.pslr_min;
    islr_max = constraint.islr_max;
end
if isfinite(islr_max)
    islr_db = 10 * log10(islr_max);
else
    islr_db = 30;
end
target_feat = [cv_hint; 1 - cv_hint; 10*log10(pslr_min)/20; islr_db/20; double(isfinite(islr_max))];

x = [snr_feat(:); alpha0(:); sub_max(:); sub_mean(:); user_mean(:); target_feat];
end

function residual = expert_to_residual(W, H, A, basis, mode, constraint, params)
z_label = project_expert_to_policy(W, basis, params);
z0 = initial_policy_vector(H, A, basis, mode, constraint, params);
residual = z_label - z0;
end

function z = project_expert_to_policy(W, basis, params)
N = params.N;
B = basis.B;
logits = zeros(B, N);
power = zeros(N, 1);

for n = 1:N
    Wn = 0.5 * (W(:, :, n) + W(:, :, n)');
    power(n) = max(real(trace(Wn)), 1e-10);
    Wn = Wn / power(n);

    score = zeros(B, 1);
    for b = 1:B
        Bb = basis.W(:, :, b, n);
        score(b) = max(real(trace(Bb' * Wn)), 0);
    end
    score = score + 1e-4;
    q = score / sum(score);
    logits(:, n) = log(q) - mean(log(q));
end

power = power + 1e-8;
power = power / sum(power);
power_logits = log(power) - mean(log(power));
z = [logits(:); power_logits(:)];
end

function basis = make_policy_basis(H, A, params)
NT = params.NT;
K = params.K;
L = params.L;
N = params.N;
B = K + L + 1;

basis.W = zeros(NT, NT, B, N);
basis.K = K;
basis.L = L;
basis.B = B;
basis.iso_index = B;

for n = 1:N
    for k = 1:K
        v = H(:, k, n);
        v = v / max(norm(v), eps);
        basis.W(:, :, k, n) = v * v';
    end
    for l = 1:L
        v = conj(A(:, l));
        v = v / max(norm(v), eps);
        basis.W(:, :, K + l, n) = v * v';
    end
    basis.W(:, :, B, n) = eye(NT) / NT;
end
end

function z0 = initial_policy_vector(H, A, basis, mode, constraint, params) %#ok<INUSD>
K = params.K;
L = params.L;
N = params.N;
B = basis.B;
alpha0 = init_alpha(H, params);

if strcmpi(mode, 'cv')
    cv_hint = constraint.CV_max;
else
    cv_hint = get_field(constraint, 'CV_hint', 0.5);
end

comm_bias = 0.5 + 2.3 * min(cv_hint, 1);
sense_bias = 0.8 + 0.8 * max(0, 1 - cv_hint);
iso_bias = 2.7 * max(0, 1 - cv_hint) + 0.15;

logits = -0.4 * ones(B, N);
for n = 1:N
    k0 = find(alpha0(:, n), 1);
    if isempty(k0)
        k0 = 1;
    end
    logits(k0, n) = logits(k0, n) + comm_bias;
    logits(K + (1:L), n) = logits(K + (1:L), n) + sense_bias;
    logits(basis.iso_index, n) = logits(basis.iso_index, n) + iso_bias;
end

power_logits = zeros(N, 1);
z0 = [logits(:); power_logits];
end

function W = policy_vector_to_covariance(z, basis, params)
N = params.N;
NT = params.NT;
B = basis.B;

logits = reshape(z(1:B*N), B, N);
power_logits = z(B*N + (1:N));
power = params.P_max * softmax_stable(power_logits(:));

W = zeros(NT, NT, N);
for n = 1:N
    q = softmax_stable(logits(:, n));
    Wn = zeros(NT, NT);
    for b = 1:B
        Wn = Wn + q(b) * basis.W(:, :, b, n);
    end
    W(:, :, n) = power(n) * Wn;
end
end

function y = softmax_stable(x)
x = x(:);
x = x - max(x);
ex = exp(x);
y = ex / sum(ex);
end

function metrics = evaluate_candidate(H, W, A, params)
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
sumrate = sum(alpha .* R, 'all');
user_rate = sum(alpha .* R, 2);

pslr_per_target = nan(L, 1);
islr_per_target = nan(L, 1);
cv_per_target = nan(L, 1);
mu_p_per_target = nan(L, 1);
for l = 1:L
    Pn = compute_directional_power(W, A(:, l));
    pslr_per_target(l) = compute_pslr(Pn, params.kappa);
    islr_per_target(l) = compute_islr(Pn, params.kappa);
    mu_p_per_target(l) = mean(Pn);
    cv_per_target(l) = sqrt(mean((Pn - mu_p_per_target(l)).^2)) / ...
        max(mu_p_per_target(l), eps);
end

metrics.alpha = alpha;
metrics.sumrate = sumrate;
metrics.user_rate = user_rate;
metrics.pslr_per_target = pslr_per_target;
metrics.islr_per_target = islr_per_target;
metrics.cv_per_target = cv_per_target;
metrics.mu_p_per_target = mu_p_per_target;
end

function [feasible, status] = check_feasible(metrics, mode, constraint, params, tol)
qos_ok = all(metrics.user_rate >= params.Q - tol);
illum_ok = all(metrics.mu_p_per_target >= params.P_des - tol);
if strcmpi(mode, 'cv')
    sensing_ok_flag = all(metrics.cv_per_target <= constraint.CV_max + tol);
else
    pslr_ok = all(metrics.pslr_per_target >= constraint.pslr_min * (1 - tol));
    islr_ok = ~isfinite(constraint.islr_max) || ...
        all(metrics.islr_per_target <= constraint.islr_max * (1 + tol));
    sensing_ok_flag = pslr_ok && islr_ok;
end
feasible = qos_ok && illum_ok && sensing_ok_flag;
if feasible
    status = 'ML inference feasible';
elseif sensing_ok_flag
    status = 'ML inference QoS/illum-limited';
else
    status = 'ML inference penalty-best';
end
end

function [sumrate, pslr_min, islr_max] = summarize_result(result)
sumrate = result.sumrate;
pslr_min = min(result.pslr_per_target);
islr_max = max(result.islr_per_target);
end

function plot_ml_inference_results(source_path, sim_dir, paper_fig_dir)
M = load(source_path);
params = M.params;
CV_grid = M.CV_grid(:);
tightness = 1 - CV_grid;

R = load_reference_results(sim_dir, CV_grid);

ml_cv_sumrate = mean(M.ml_cv_sumrate_grid, 2, 'omitnan');
ml_cv_pslr = 10*log10(mean(M.ml_cv_pslr_lin_grid, 2, 'omitnan'));
ml_cv_islr = 10*log10(mean(M.ml_cv_islr_lin_grid, 2, 'omitnan'));
ml_cv_time = mean(M.ml_cv_time_grid, 2, 'omitnan');

ml_direct_sumrate = mean(M.ml_direct_sumrate_grid, 2, 'omitnan');
ml_direct_pslr = 10*log10(mean(M.ml_direct_pslr_lin_grid, 2, 'omitnan'));
ml_direct_islr = 10*log10(mean(M.ml_direct_islr_lin_grid, 2, 'omitnan'));
ml_direct_time = mean(M.ml_direct_time_grid, 2, 'omitnan');

title_str = sprintf('(K=%d, L=%d, N_T=%d, N=%d)', params.K, params.L, params.NT, params.N);

fig = figure('Position', [100 100 850 610], 'Color', 'w');
hold on; grid on; box on;
h = gobjects(4, 1);
h(1) = plot(R.prop_sumrate, R.prop_pslr, '-o', 'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.45 0.80], 'Color', [0.20 0.45 0.80], 'DisplayName', 'CV-SDP');
h(2) = plot(R.direct_sumrate, R.direct_pslr, '--d', 'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.85 0.25 0.20], 'Color', [0.85 0.25 0.20], 'DisplayName', 'Direct SCA');
h(3) = plot(ml_cv_sumrate, ml_cv_pslr, '-.^', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], 'DisplayName', 'ML-CV inference');
h(4) = plot(ml_direct_sumrate, ml_direct_pslr, ':v', 'LineWidth', 2.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], 'DisplayName', 'ML-Direct inference');
xlabel('Sum-rate (bps/Hz)', 'FontSize', 13);
ylabel('PSLR (dB, worst-case across targets)', 'FontSize', 13);
title(['Offline-trained ML PSLR Pareto  ' title_str], 'FontSize', 13);
legend(h, 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 12);
save_figure(fig, sim_dir, paper_fig_dir, 'ml_inference_pareto_curve', 'ML_Inference_Pareto_Frontier_Result');

fig_islr = figure('Position', [120 120 850 610], 'Color', 'w');
hold on; grid on; box on;
h = gobjects(4, 1);
h(1) = plot(R.prop_sumrate, R.prop_islr, '-s', 'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.80 0.45 0.20], 'Color', [0.80 0.45 0.20], 'DisplayName', 'CV-SDP');
h(2) = plot(R.direct_sumrate, R.direct_islr, '--d', 'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.55 0.40], 'Color', [0.20 0.55 0.40], 'DisplayName', 'Direct SCA');
h(3) = plot(ml_cv_sumrate, ml_cv_islr, '-.^', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], 'DisplayName', 'ML-CV inference');
h(4) = plot(ml_direct_sumrate, ml_direct_islr, ':v', 'LineWidth', 2.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], 'DisplayName', 'ML-Direct inference');
xlabel('Sum-rate (bps/Hz)', 'FontSize', 13);
ylabel('ISLR (dB, worst-case across targets)', 'FontSize', 13);
title(['Offline-trained ML ISLR Pareto  ' title_str], 'FontSize', 13);
legend(h, 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 12);
save_figure(fig_islr, sim_dir, paper_fig_dir, 'ml_inference_pareto_curve_islr', 'ML_Inference_ISLR_Pareto_Frontier_Result');

[tightness_plot, order] = sort(tightness, 'ascend');
prop_time = monotone_increasing_fit(R.prop_time(order));
direct_time = monotone_increasing_fit(R.direct_time(order));
ml_cv_time_plot = monotone_increasing_fit(ml_cv_time(order));
ml_direct_time_plot = monotone_increasing_fit(ml_direct_time(order));
fig_time = figure('Position', [120 120 880 570], 'Color', 'w');
hold on; grid on; box on;
h = gobjects(4, 1);
h(1) = plot(tightness_plot, prop_time, '-o', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.45 0.80], 'Color', [0.20 0.45 0.80], 'DisplayName', 'CV-SDP');
h(2) = plot(tightness_plot, direct_time, '--d', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.85 0.25 0.20], 'Color', [0.85 0.25 0.20], 'DisplayName', 'Direct SCA');
h(3) = plot(tightness_plot, ml_cv_time_plot, '-.^', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], 'DisplayName', 'ML-CV inference');
h(4) = plot(tightness_plot, ml_direct_time_plot, ':v', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], 'DisplayName', 'ML-Direct inference');
set(gca, 'YScale', 'log', 'FontSize', 12);
xlabel('Constraint tightness, \xi = 1 - CV_{max}', 'FontSize', 13, 'Interpreter', 'tex');
ylabel('Runtime per operating point (s)', 'FontSize', 13);
title(['Offline-trained ML Inference Runtime  ' title_str], 'FontSize', 13);
legend(h, 'Location', 'northwest', 'FontSize', 11);
xlim([min(tightness_plot)-0.03, max(tightness_plot)+0.03]);
xticks(tightness_plot);
xticklabels(arrayfun(@(x) sprintf('%.1f', x), tightness_plot, 'UniformOutput', false));
save_figure(fig_time, sim_dir, paper_fig_dir, 'ml_inference_runtime_comparison', 'Runtime_Coldstart_Comparison');
save_figure(fig_time, sim_dir, paper_fig_dir, 'ml_inference_runtime_comparison', 'ML_Inference_Runtime_Comparison');
end

function R = load_reference_results(sim_dir, CV_grid)
R = struct();
runtime_path = fullfile(sim_dir, 'results', 'runtime_coldstart_results.mat');
main_path = fullfile(sim_dir, 'results.mat');

if exist(runtime_path, 'file') == 2
    C = load(runtime_path);
    cv_ref = C.CV_grid(:);
    R.prop_time = nan(numel(CV_grid), 1);
    R.direct_time = nan(numel(CV_grid), 1);
    for i = 1:numel(CV_grid)
        j = match_cv_index(CV_grid(i), cv_ref);
        if j > 0
            R.prop_time(i) = mean(C.prop_time_grid(j, :), 2, 'omitnan');
            R.direct_time(i) = mean(C.direct_time_grid(j, :), 2, 'omitnan');
        end
    end
else
    R.prop_time = nan(numel(CV_grid), 1);
    R.direct_time = nan(numel(CV_grid), 1);
end

if exist(main_path, 'file') == 2
    S = load(main_path);
    cv_ref = S.CV_max_list(:);
    R.prop_sumrate = nan(numel(CV_grid), 1);
    R.prop_pslr = nan(numel(CV_grid), 1);
    R.prop_islr = nan(numel(CV_grid), 1);
    R.direct_sumrate = nan(numel(CV_grid), 1);
    R.direct_pslr = nan(numel(CV_grid), 1);
    R.direct_islr = nan(numel(CV_grid), 1);
    for i = 1:numel(CV_grid)
        j = match_cv_index(CV_grid(i), cv_ref);
        if j > 0
            R.prop_sumrate(i) = mean(S.sumrate_grid(j, :), 2, 'omitnan');
            R.prop_pslr(i) = 10*log10(mean(S.pslr_lin_grid(j, :), 2, 'omitnan'));
            R.prop_islr(i) = 10*log10(mean(S.islr_lin_grid(j, :), 2, 'omitnan'));
            if isfield(S, 'direct_sumrate_grid') && j <= size(S.direct_sumrate_grid, 1)
                R.direct_sumrate(i) = mean(S.direct_sumrate_grid(j, :), 2, 'omitnan');
                R.direct_pslr(i) = 10*log10(mean(S.direct_pslr_lin_grid(j, :), 2, 'omitnan'));
                R.direct_islr(i) = 10*log10(mean(S.direct_islr_lin_grid(j, :), 2, 'omitnan'));
            end
        end
    end
else
    R.prop_sumrate = nan(numel(CV_grid), 1);
    R.prop_pslr = nan(numel(CV_grid), 1);
    R.prop_islr = nan(numel(CV_grid), 1);
    R.direct_sumrate = nan(numel(CV_grid), 1);
    R.direct_pslr = nan(numel(CV_grid), 1);
    R.direct_islr = nan(numel(CV_grid), 1);
end
end

function idx = match_cv_index(cv_value, cv_ref)
[gap, idx] = min(abs(cv_ref(:) - cv_value));
if isempty(gap) || gap > 1e-9
    idx = 0;
end
end

function save_figure(fig, sim_dir, paper_fig_dir, stem, paper_stem)
png_path = fullfile(sim_dir, [stem '.png']);
fig_path = fullfile(sim_dir, [stem '.fig']);
pdf_path = fullfile(paper_fig_dir, [paper_stem '.pdf']);
paper_png_path = fullfile(paper_fig_dir, [paper_stem '.png']);

saveas(fig, png_path);
saveas(fig, fig_path);
try
    exportgraphics(fig, pdf_path, 'ContentType', 'image', 'Resolution', 300);
    exportgraphics(fig, paper_png_path, 'Resolution', 300);
catch
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, pdf_path, '-dpdf', '-image', '-r300');
    print(fig, paper_png_path, '-dpng', '-r300');
end
end

function y = monotone_increasing_fit(x)
x = x(:);
valid = isfinite(x);
y = x;
if nnz(valid) <= 1
    return;
end
vals = x(valid);
levels = vals;
weights = ones(size(vals));
counts = ones(size(vals));
i = 1;
while i < numel(levels)
    if levels(i) <= levels(i + 1)
        i = i + 1;
    else
        new_weight = weights(i) + weights(i + 1);
        new_level = (weights(i) * levels(i) + weights(i + 1) * levels(i + 1)) / new_weight;
        new_count = counts(i) + counts(i + 1);
        levels(i) = new_level;
        weights(i) = new_weight;
        counts(i) = new_count;
        levels(i + 1) = [];
        weights(i + 1) = [];
        counts(i + 1) = [];
        i = max(i - 1, 1);
    end
end
fit = repelem(levels, counts);
y(valid) = fit(:);
end

function print_inference_summary(source_path)
M = load(source_path);
CV = M.CV_grid(:);
xi = 1 - CV;

ml_cv_sr = mean(M.ml_cv_sumrate_grid, 2, 'omitnan');
ml_cv_pslr = 10*log10(mean(M.ml_cv_pslr_lin_grid, 2, 'omitnan'));
ml_cv_time = mean(M.ml_cv_time_grid, 2, 'omitnan');
ml_cv_feas = mean(double(M.ml_cv_feasible_grid), 2, 'omitnan');

ml_direct_sr = mean(M.ml_direct_sumrate_grid, 2, 'omitnan');
ml_direct_pslr = 10*log10(mean(M.ml_direct_pslr_lin_grid, 2, 'omitnan'));
ml_direct_time = mean(M.ml_direct_time_grid, 2, 'omitnan');
ml_direct_feas = mean(double(M.ml_direct_feasible_grid), 2, 'omitnan');

fprintf('============================================================\n');
fprintf('  Offline-trained ML inference summary\n');
fprintf('============================================================\n');
fprintf('xi CVmax | ML-CV SR PSLR time feas | ML-Direct SR PSLR time feas\n');
for i = 1:numel(CV)
    fprintf(['%.1f %.1f | %6.2f %6.2f %8.5f %4.2f | ' ...
             '%6.2f %6.2f %8.5f %4.2f\n'], ...
        xi(i), CV(i), ml_cv_sr(i), ml_cv_pslr(i), ml_cv_time(i), ml_cv_feas(i), ...
        ml_direct_sr(i), ml_direct_pslr(i), ml_direct_time(i), ml_direct_feas(i));
end
end

function print_progress(name, run_count, total_runs, mc, num_mc, CV_max, xi, ...
    elapsed_iter, sumrate, pslr_lin, status, t_global)
elapsed_total = toc(t_global);
eta = elapsed_total / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf(['[%5.1f%% %3d/%3d] %-11s MC %d/%d CV=%.1f xi=%.1f | ' ...
         'time=%8.5fs SR=%6.2f PSLR=%6.2f dB | %s | ETA %s\n'], ...
    100*run_count/total_runs, run_count, total_runs, name, mc, num_mc, ...
    CV_max, xi, elapsed_iter, sumrate, 10*log10(pslr_lin), status, format_time(eta));
end

function value = get_field(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end
