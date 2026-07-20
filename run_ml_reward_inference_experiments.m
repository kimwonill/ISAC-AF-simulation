function run_ml_reward_inference_experiments()
% RUN_ML_REWARD_INFERENCE_EXPERIMENTS
% Label-free reward-trained ML/RL-style inference baselines.
%
% No SDP/SCA expert labels are generated. The offline stage uses a
% cross-entropy policy search to maximize the constrained sum-rate reward
% directly over random training channels. The learned policy is then frozen.
% Test-time runtime includes only candidate construction, fixed feasibility
% certification, and metric evaluation.

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
train_CV_grid = 0:0.25:1.0;
num_train_mc = 24;
num_test_mc = params.num_mc;

opts = struct();
opts.model_version = 4;
opts.population = 28;
opts.max_iter = 9;
opts.elite_frac = 0.20;
opts.smoothing = 0.65;
opts.sigma0 = 0.85;
opts.min_sigma = 0.04;
opts.constraint_tol = 1e-4;
opts.timing_repeats = 80;
opts.penalty_qos = 60;
opts.penalty_sensing = 80;
opts.use_saved_policy = true;

policy_path = fullfile(out_data_dir, 'ml_reward_inference_policy.mat');
source_path = fullfile(out_data_dir, 'ml_reward_inference_results.mat');

fprintf('============================================================\n');
fprintf('  Label-free reward-trained ML/RL inference experiment\n');
fprintf('============================================================\n');
fprintf('  Train MC=%d, train CV=[%s]\n', num_train_mc, num2str(train_CV_grid));
fprintf('  Test MC=%d, test CV=[%s]\n', num_test_mc, num2str(CV_grid));
fprintf('  CEM population=%d, iterations=%d, timing repeats=%d\n', ...
    opts.population, opts.max_iter, opts.timing_repeats);
fprintf('------------------------------------------------------------\n');

if opts.use_saved_policy && exist(policy_path, 'file') == 2
    P = load(policy_path);
    if isfield(P, 'opts') && isfield(P.opts, 'model_version') && ...
            P.opts.model_version == opts.model_version
        fprintf('Reusing reward-trained policy: %s\n', policy_path);
        policy_cv = P.policy_cv;
        policy_direct = P.policy_direct;
        training_info = P.training_info;
    else
        [policy_cv, policy_direct, training_info] = train_reward_policies(params, train_CV_grid, num_train_mc, opts);
        save(policy_path, 'policy_cv', 'policy_direct', 'training_info', 'opts', 'params', 'train_CV_grid', 'num_train_mc');
    end
else
    [policy_cv, policy_direct, training_info] = train_reward_policies(params, train_CV_grid, num_train_mc, opts);
    save(policy_path, 'policy_cv', 'policy_direct', 'training_info', 'opts', 'params', 'train_CV_grid', 'num_train_mc');
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
        cv_result = run_reward_policy_inference(H, 'cv', cv_constraint, policy_cv, params, opts);
        ml_cv_time_grid(c, mc) = cv_result.elapsed;
        [ml_cv_sumrate_grid(c, mc), ml_cv_pslr_lin_grid(c, mc), ml_cv_islr_lin_grid(c, mc)] = ...
            summarize_result(cv_result);
        ml_cv_feasible_grid(c, mc) = cv_result.feasible;
        ml_cv_status_grid(c, mc) = string(cv_result.status);
        print_progress('RL-CV inf.', run_count, total_runs, mc, num_test_mc, CV_max, xi, ...
            cv_result.elapsed, cv_result.sumrate, min(cv_result.pslr_per_target), cv_result.status, t_global);

        run_count = run_count + 1;
        direct_constraint = struct('pslr_min', pslr_min, 'islr_max', islr_max, 'CV_hint', CV_max);
        direct_result = run_reward_policy_inference(H, 'direct', direct_constraint, policy_direct, params, opts);
        ml_direct_time_grid(c, mc) = direct_result.elapsed;
        [ml_direct_sumrate_grid(c, mc), ml_direct_pslr_lin_grid(c, mc), ml_direct_islr_lin_grid(c, mc)] = ...
            summarize_result(direct_result);
        ml_direct_feasible_grid(c, mc) = direct_result.feasible;
        ml_direct_status_grid(c, mc) = string(direct_result.status);
        print_progress('RL-Dir inf.', run_count, total_runs, mc, num_test_mc, CV_max, xi, ...
            direct_result.elapsed, direct_result.sumrate, min(direct_result.pslr_per_target), ...
            direct_result.status, t_global);
    end
end

save(source_path, ...
    'params', 'CV_grid', 'train_CV_grid', 'num_train_mc', 'num_test_mc', ...
    'opts', 'training_info', 'policy_cv', 'policy_direct', ...
    'ml_cv_sumrate_grid', 'ml_cv_pslr_lin_grid', 'ml_cv_islr_lin_grid', ...
    'ml_cv_time_grid', 'ml_cv_feasible_grid', 'ml_cv_status_grid', ...
    'ml_direct_sumrate_grid', 'ml_direct_pslr_lin_grid', 'ml_direct_islr_lin_grid', ...
    'ml_direct_time_grid', 'ml_direct_feasible_grid', 'ml_direct_status_grid');

plot_reward_inference_results(source_path, sim_dir, paper_fig_dir);
print_reward_summary(source_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved reward-trained policy: %s\n', policy_path);
fprintf('  Saved reward inference source data: %s\n', source_path);
fprintf('  Total elapsed including label-free training: %s\n', ...
    format_time(toc(t_global) + training_info.elapsed));
fprintf('============================================================\n');
end

function [policy_cv, policy_direct, info] = train_reward_policies(params, CV_grid, num_train_mc, opts)
t_train = tic;
train_set = make_training_set(params, CV_grid, num_train_mc);

fprintf('Training CV policy from constrained reward only...\n');
[theta_cv, hist_cv] = cem_train_policy(train_set, 'cv', params, opts, 3101);
fprintf('Training Direct policy from constrained reward only...\n');
[theta_direct, hist_direct] = cem_train_policy(train_set, 'direct', params, opts, 4101);

policy_cv = theta_to_policy(theta_cv);
policy_direct = theta_to_policy(theta_direct);
policy_cv.mode = 'cv';
policy_direct.mode = 'direct';

info.elapsed = toc(t_train);
info.cv_history = hist_cv;
info.direct_history = hist_direct;
info.num_train_samples = numel(train_set);
fprintf('Reward training completed in %s over %d label-free samples.\n', ...
    format_time(info.elapsed), info.num_train_samples);
end

function train_set = make_training_set(params, CV_grid, num_train_mc)
train_set = repmat(struct('H', [], 'CV_max', []), numel(CV_grid) * num_train_mc, 1);
idx = 0;
for mc = 1:num_train_mc
    rng(2000 + mc, 'twister');
    H = generate_channel(params);
    for c = 1:numel(CV_grid)
        idx = idx + 1;
        train_set(idx).H = H;
        train_set(idx).CV_max = CV_grid(c);
    end
end
end

function [theta_best, history] = cem_train_policy(train_set, mode, params, opts, seed)
rng(seed, 'twister');
D = 8;
mu = zeros(D, 1);
sigma = opts.sigma0 * ones(D, 1);
elite_count = max(2, ceil(opts.elite_frac * opts.population));
theta_best = mu;
score_best = -Inf;
history = nan(opts.max_iter, 2);

for iter = 1:opts.max_iter
    Theta = mu + sigma .* randn(D, opts.population);
    Theta(:, 1) = mu;
    scores = nan(opts.population, 1);
    for p = 1:opts.population
        policy = theta_to_policy(Theta(:, p));
        scores(p) = evaluate_policy_reward(train_set, mode, policy, params, opts);
        if scores(p) > score_best
            score_best = scores(p);
            theta_best = Theta(:, p);
        end
    end
    [~, order] = sort(scores, 'descend');
    elites = Theta(:, order(1:elite_count));
    elite_mu = mean(elites, 2);
    elite_sigma = std(elites, 0, 2);
    mu = (1 - opts.smoothing) * mu + opts.smoothing * elite_mu;
    sigma = (1 - opts.smoothing) * sigma + opts.smoothing * elite_sigma;
    sigma = max(sigma, opts.min_sigma);
    history(iter, :) = [score_best, mean(scores(order(1:elite_count)))];
    fprintf('  %s CEM iter %02d/%02d | best reward %.3f | elite %.3f\n', ...
        upper(mode), iter, opts.max_iter, score_best, history(iter, 2));
end
end

function policy = theta_to_policy(theta)
policy.theta = theta(:);
policy.rho_iso = 1.35 * sigmoid(theta(1));
policy.rho_struct = 1.35 * sigmoid(theta(2));
policy.rho_coherent = 1.35 * sigmoid(theta(3));
policy.struct_bias = theta(4);
policy.struct_tightness = theta(5);
policy.coherent_bias = theta(6);
policy.coherent_tightness = theta(7);
policy.score_temperature = 0.5 + 2.0 * sigmoid(theta(8));
end

function score = evaluate_policy_reward(train_set, mode, policy, params, opts)
score = 0;
for i = 1:numel(train_set)
    CV_max = train_set(i).CV_max;
    if strcmpi(mode, 'cv')
        constraint = struct('CV_max', CV_max, 'CV_hint', CV_max);
    else
        [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);
        constraint = struct('pslr_min', pslr_min, 'islr_max', islr_max, 'CV_hint', CV_max);
    end
    result = infer_once(train_set(i).H, mode, constraint, policy, params, opts);
    score = score + constrained_reward(result, mode, constraint, params, opts);
end
score = score / numel(train_set);
end

function result = run_reward_policy_inference(H, mode, constraint, policy, params, opts)
repeats = opts.timing_repeats;
last = [];
t_start = tic;
for r = 1:repeats
    last = infer_once(H, mode, constraint, policy, params, opts);
end
result = last;
result.elapsed = toc(t_start) / repeats;
end

function result = infer_once(H, mode, constraint, policy, params, opts)
A = compute_steering(params);
C_iso = make_comm_covariance(H, params, policy.rho_iso);
C_struct = make_comm_covariance(H, params, policy.rho_struct);
C_coherent = make_comm_covariance(H, params, policy.rho_coherent);

candidate_names = {'flat', 'struct-diag', 'struct-coherent', 'constant-coherent'};
candidate_W = cell(4, 1);
candidate_W{1} = flat_mix_candidate(C_iso, mode, constraint, A, params, opts);
candidate_W{2} = structural_equalized_candidate(H, C_struct, mode, constraint, A, params, opts, 'diagonal');
candidate_W{3} = structural_equalized_candidate(H, C_coherent, mode, constraint, A, params, opts, 'coherent');
candidate_W{4} = constant_coherent_candidate(H, C_coherent, mode, constraint, A, params, opts);

best_score = -Inf;
best_ix = 1;
best_metrics = [];
best_feasible_score = -Inf;
best_feasible_ix = [];
best_feasible_metrics = [];
for cand = 1:numel(candidate_W)
    metrics_cand = evaluate_candidate(H, candidate_W{cand}, A, params);
    score_cand = constrained_reward_from_metrics(metrics_cand, mode, constraint, params, opts);
    feasible_cand = check_feasible(metrics_cand, mode, constraint, params, opts.constraint_tol);
    if feasible_cand && score_cand > best_feasible_score
        best_feasible_score = score_cand;
        best_feasible_ix = cand;
        best_feasible_metrics = metrics_cand;
    end
    if score_cand > best_score
        best_score = score_cand;
        best_ix = cand;
        best_metrics = metrics_cand;
    end
end

if ~isempty(best_feasible_ix)
    W = candidate_W{best_feasible_ix};
    metrics = best_feasible_metrics;
    branch = candidate_names{best_feasible_ix};
else
    W = candidate_W{best_ix};
    metrics = best_metrics;
    branch = candidate_names{best_ix};
end

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
result.branch = branch;
result.mode = mode;
result.constraint = constraint;
end

function W = make_comm_covariance(H, params, rho)
N = params.N;
NT = params.NT;
gains = zeros(N, 1);
V = zeros(NT, N);
for n = 1:N
    norms = squeeze(sum(abs(H(:, :, n)).^2, 1));
    [gains(n), k] = max(norms);
    v = H(:, k, n);
    V(:, n) = v / max(norm(v), eps);
end
if rho <= 1e-6
    power = params.P_max / N * ones(N, 1);
else
    q = (gains / max(max(gains), eps)).^rho;
    q = q / sum(q);
    power = params.P_max * q;
end
W = zeros(NT, NT, N);
for n = 1:N
    W(:, :, n) = power(n) * (V(:, n) * V(:, n)');
end
end

function W = flat_mix_candidate(C, mode, constraint, A, params, opts)
W_flat = init_covariance_flat(params);
lo = 0;
hi = 1;
best = W_flat;
for i = 1:28
    eta = 0.5 * (lo + hi);
    W_try = (1 - eta) * W_flat + eta * C;
    if sensing_ok(W_try, mode, constraint, A, params, opts.constraint_tol)
        best = W_try;
        lo = eta;
    else
        hi = eta;
    end
end
W = best;
end

function W = structural_equalized_candidate(H, C, mode, constraint, A, params, opts, residual_mode)
if nargin < 8
    residual_mode = 'diagonal';
end
B = conj(A);
if size(B, 1) ~= size(B, 2) || rcond(B) < 1e-8
    W = flat_mix_candidate(C, mode, constraint, A, params, opts);
    return;
end

Binv = inv(B);
L = params.L;
N = params.N;
c = zeros(L, N);
for n = 1:N
    for l = 1:L
        c(l, n) = real(B(:, l)' * C(:, :, n) * B(:, l));
    end
end

lo = 0;
hi = 1;
best = init_covariance_flat(params);
for i = 1:36
    gamma = 0.5 * (lo + hi);
    q = max(params.P_des, max(gamma * c, [], 2));
    W_try = zeros(params.NT, params.NT, params.N);
    total_power = 0;
    ok = true;
    for n = 1:N
        residual = q - gamma * c(:, n);
        if any(residual < -1e-9)
            ok = false;
            break;
        end
        Y = residual_covariance_in_target_basis(H, residual, Binv, n, residual_mode);
        S = Binv' * Y * Binv;
        W_try(:, :, n) = gamma * C(:, :, n) + S;
        total_power = total_power + real(trace(W_try(:, :, n)));
    end
    if ok && total_power <= params.P_max + 1e-9
        best = W_try;
        lo = gamma;
    else
        hi = gamma;
    end
end

lo = 0;
hi = 1;
W_best = best;
for i = 1:28
    eta = 0.5 * (lo + hi);
    W_try = (1 - eta) * best + eta * C;
    if sensing_ok(W_try, mode, constraint, A, params, opts.constraint_tol)
        W_best = W_try;
        lo = eta;
    else
        hi = eta;
    end
end
W = W_best;
end

function W = constant_coherent_candidate(H, C, mode, constraint, A, params, opts)
B = conj(A);
if size(B, 1) ~= size(B, 2) || rcond(B) < 1e-8
    W = flat_mix_candidate(C, mode, constraint, A, params, opts);
    return;
end

Binv = inv(B);
target_user = qos_aware_target_users(H, params);
lo = params.P_des;
hi = params.P_max;
best = [];
for i = 1:34
    q0 = 0.5 * (lo + hi);
    W_try = coherent_constant_power_tensor(H, q0, Binv, params, target_user);
    total_power = total_covariance_power(W_try);
    if total_power <= params.P_max + 1e-9
        best = W_try;
        lo = q0;
    else
        hi = q0;
    end
end

if isempty(best)
    best = coherent_constant_power_tensor(H, params.P_des, Binv, params, target_user);
    scale = min(1, params.P_max / max(total_covariance_power(best), eps));
    best = scale * best;
end

lo = 0;
hi = 1;
W_best = best;
for i = 1:28
    eta = 0.5 * (lo + hi);
    W_try = (1 - eta) * best + eta * C;
    if sensing_ok(W_try, mode, constraint, A, params, opts.constraint_tol)
        W_best = W_try;
        lo = eta;
    else
        hi = eta;
    end
end
W = W_best;
end

function W = coherent_constant_power_tensor(H, q0, Binv, params, target_user)
W = zeros(params.NT, params.NT, params.N);
q = q0 * ones(params.L, 1);
for n = 1:params.N
    Y = residual_covariance_in_target_basis(H, q, Binv, n, 'coherent', target_user(n));
    W(:, :, n) = Binv' * Y * Binv;
end
end

function user_n = qos_aware_target_users(H, params)
K = params.K;
N = params.N;
R = zeros(K, N);
for n = 1:N
    for k = 1:K
        gain = (params.P_max / params.N) * norm(H(:, k, n))^2;
        R(k, n) = log2(1 + gain / params.sigma2);
    end
end

[~, user_n] = max(R, [], 1);
tol = get_param_local(params, 'dual_tol', 1e-4);
for repair_pass = 1:(K * N)
    user_rate = zeros(K, 1);
    for n = 1:N
        user_rate(user_n(n)) = user_rate(user_n(n)) + R(user_n(n), n);
    end
    [max_gap, k_v] = max(params.Q - user_rate);
    if max_gap <= tol
        break;
    end

    best_n = [];
    best_cost = Inf;
    best_safe = false;
    for n = 1:N
        if user_n(n) == k_v
            continue;
        end
        k_curr = user_n(n);
        cost = R(k_curr, n) - R(k_v, n);
        donor_safe = (user_rate(k_curr) - R(k_curr, n)) >= params.Q(k_curr) - tol;
        if isempty(best_n) || ...
                (donor_safe && ~best_safe) || ...
                (donor_safe == best_safe && cost < best_cost)
            best_n = n;
            best_cost = cost;
            best_safe = donor_safe;
        end
    end
    if isempty(best_n)
        break;
    end
    user_n(best_n) = k_v;
end
end

function value = get_param_local(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end

function total_power = total_covariance_power(W)
total_power = 0;
for n = 1:size(W, 3)
    total_power = total_power + real(trace(W(:, :, n)));
end
end

function Y = residual_covariance_in_target_basis(H, residual, Binv, n, residual_mode, target_user)
if nargin < 6
    target_user = [];
end
r = max(real(residual(:)), 0);
if strcmpi(residual_mode, 'coherent') && any(r > 0)
    if ~isempty(target_user)
        u = Binv * H(:, target_user, n);
        best_y = sqrt(r) .* exp(1j * angle(u));
    else
        K = size(H, 2);
        best_gain = -Inf;
        best_y = sqrt(r);
        for k = 1:K
            u = Binv * H(:, k, n);
            y = sqrt(r) .* exp(1j * angle(u));
            gain = abs(y' * u)^2;
            if gain > best_gain
                best_gain = gain;
                best_y = y;
            end
        end
    end
    Y = best_y * best_y';
else
    Y = diag(r);
end
Y = (Y + Y') / 2;
end

function ok = sensing_ok(W, mode, constraint, A, params, tol)
ok = true;
for l = 1:params.L
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

function metrics = evaluate_candidate(H, W, A, params)
R = zeros(params.K, params.N);
for n = 1:params.N
    Wn = W(:, :, n);
    for k = 1:params.K
        hk = H(:, k, n);
        gain = max(real(hk' * Wn * hk), 0);
        R(k, n) = log2(1 + gain / params.sigma2);
    end
end
alpha = update_alpha(H, W, params);
metrics.alpha = alpha;
metrics.sumrate = sum(alpha .* R, 'all');
metrics.user_rate = sum(alpha .* R, 2);
metrics.pslr_per_target = nan(params.L, 1);
metrics.islr_per_target = nan(params.L, 1);
metrics.cv_per_target = nan(params.L, 1);
metrics.mu_p_per_target = nan(params.L, 1);
for l = 1:params.L
    Pn = compute_directional_power(W, A(:, l));
    metrics.pslr_per_target(l) = compute_pslr(Pn, params.kappa);
    metrics.islr_per_target(l) = compute_islr(Pn, params.kappa);
    metrics.mu_p_per_target(l) = mean(Pn);
    metrics.cv_per_target(l) = sqrt(mean((Pn - metrics.mu_p_per_target(l)).^2)) / ...
        max(metrics.mu_p_per_target(l), eps);
end
end

function reward = constrained_reward(result, mode, constraint, params, opts)
reward = constrained_reward_from_metrics(result, mode, constraint, params, opts);
end

function reward = constrained_reward_from_metrics(metrics, mode, constraint, params, opts)
qos_gap = max(0, params.Q - metrics.user_rate) ./ max(params.Q, 1);
illum_gap = max(0, params.P_des - metrics.mu_p_per_target) ./ max(params.P_des, eps);
penalty = opts.penalty_qos * sum(qos_gap.^2) + opts.penalty_sensing * sum(illum_gap.^2);
if strcmpi(mode, 'cv')
    denom = max(constraint.CV_max, 0.05);
    cv_gap = max(0, metrics.cv_per_target - constraint.CV_max) ./ denom;
    penalty = penalty + opts.penalty_sensing * sum(cv_gap.^2);
else
    pslr_gap = max(0, 10*log10(constraint.pslr_min) - 10*log10(metrics.pslr_per_target));
    if isfinite(constraint.islr_max)
        islr_gap = max(0, 10*log10(metrics.islr_per_target) - 10*log10(constraint.islr_max));
    else
        islr_gap = zeros(size(metrics.islr_per_target));
    end
    penalty = penalty + opts.penalty_sensing * (sum(pslr_gap.^2) + sum(islr_gap.^2));
end
reward = metrics.sumrate - penalty;
end

function [feasible, status] = check_feasible(metrics, mode, constraint, params, tol)
qos_ok = all(metrics.user_rate >= params.Q - tol);
illum_ok = all(metrics.mu_p_per_target >= params.P_des - tol);
if strcmpi(mode, 'cv')
    sensing_flag = all(metrics.cv_per_target <= constraint.CV_max + tol);
else
    pslr_ok = all(metrics.pslr_per_target >= constraint.pslr_min * (1 - tol));
    islr_ok = ~isfinite(constraint.islr_max) || ...
        all(metrics.islr_per_target <= constraint.islr_max * (1 + tol));
    sensing_flag = pslr_ok && islr_ok;
end
feasible = qos_ok && illum_ok && sensing_flag;
if feasible
    status = 'Reward ML feasible';
elseif sensing_flag
    status = 'Reward ML QoS/illum-limited';
else
    status = 'Reward ML penalty-best';
end
end

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

function [sumrate, pslr_min, islr_max] = summarize_result(result)
sumrate = result.sumrate;
pslr_min = min(result.pslr_per_target);
islr_max = max(result.islr_per_target);
end

function plot_reward_inference_results(source_path, sim_dir, paper_fig_dir)
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
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], 'DisplayName', 'RL-CV inference');
h(4) = plot(ml_direct_sumrate, ml_direct_pslr, ':v', 'LineWidth', 2.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], 'DisplayName', 'RL-Direct inference');
xlabel('Sum-rate (bps/Hz)', 'FontSize', 13);
ylabel('PSLR (dB, worst-case across targets)', 'FontSize', 13);
title(['Label-free Reward-Trained PSLR Pareto  ' title_str], 'FontSize', 13);
legend(h, 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 12);
save_figure(fig, sim_dir, paper_fig_dir, 'ml_reward_inference_pareto_curve', 'ML_Inference_Pareto_Frontier_Result');
save_figure(fig, sim_dir, paper_fig_dir, 'ml_reward_inference_pareto_curve', 'ML_Reward_Pareto_Frontier_Result');

fig_islr = figure('Position', [120 120 850 610], 'Color', 'w');
hold on; grid on; box on;
h = gobjects(4, 1);
h(1) = plot(R.prop_sumrate, R.prop_islr, '-s', 'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.80 0.45 0.20], 'Color', [0.80 0.45 0.20], 'DisplayName', 'CV-SDP');
h(2) = plot(R.direct_sumrate, R.direct_islr, '--d', 'LineWidth', 2.1, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.55 0.40], 'Color', [0.20 0.55 0.40], 'DisplayName', 'Direct SCA');
h(3) = plot(ml_cv_sumrate, ml_cv_islr, '-.^', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], 'DisplayName', 'RL-CV inference');
h(4) = plot(ml_direct_sumrate, ml_direct_islr, ':v', 'LineWidth', 2.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], 'DisplayName', 'RL-Direct inference');
xlabel('Sum-rate (bps/Hz)', 'FontSize', 13);
ylabel('ISLR (dB, worst-case across targets)', 'FontSize', 13);
title(['Label-free Reward-Trained ISLR Pareto  ' title_str], 'FontSize', 13);
legend(h, 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 12);
save_figure(fig_islr, sim_dir, paper_fig_dir, 'ml_reward_inference_pareto_curve_islr', 'ML_Inference_ISLR_Pareto_Frontier_Result');
save_figure(fig_islr, sim_dir, paper_fig_dir, 'ml_reward_inference_pareto_curve_islr', 'ML_Reward_ISLR_Pareto_Frontier_Result');

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
    'MarkerFaceColor', [0.20 0.65 0.50], 'Color', [0.10 0.52 0.42], 'DisplayName', 'RL-CV inference');
h(4) = plot(tightness_plot, ml_direct_time_plot, ':v', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], 'Color', [0.45 0.25 0.65], 'DisplayName', 'RL-Direct inference');
set(gca, 'YScale', 'log', 'FontSize', 12);
xlabel('Constraint tightness, \xi = 1 - CV_{max}', 'FontSize', 13, 'Interpreter', 'tex');
ylabel('Runtime per operating point (s)', 'FontSize', 13);
title(['Label-free Reward-Trained Inference Runtime  ' title_str], 'FontSize', 13);
legend(h, 'Location', 'northwest', 'FontSize', 11);
xlim([min(tightness_plot)-0.03, max(tightness_plot)+0.03]);
xticks(tightness_plot);
xticklabels(arrayfun(@(x) sprintf('%.1f', x), tightness_plot, 'UniformOutput', false));
save_figure(fig_time, sim_dir, paper_fig_dir, 'ml_reward_inference_runtime_comparison', 'Runtime_Coldstart_Comparison');
save_figure(fig_time, sim_dir, paper_fig_dir, 'ml_reward_inference_runtime_comparison', 'ML_Inference_Runtime_Comparison');
save_figure(fig_time, sim_dir, paper_fig_dir, 'ml_reward_inference_runtime_comparison', 'ML_Reward_Runtime_Comparison');
end

function R = load_reference_results(sim_dir, CV_grid)
R = struct();
runtime_path = fullfile(sim_dir, 'results', 'runtime_coldstart_results.mat');
main_path = fullfile(sim_dir, 'results.mat');

R.prop_time = nan(numel(CV_grid), 1);
R.direct_time = nan(numel(CV_grid), 1);
if exist(runtime_path, 'file') == 2
    C = load(runtime_path);
    for i = 1:numel(CV_grid)
        j = match_cv_index(CV_grid(i), C.CV_grid(:));
        if j > 0
            R.prop_time(i) = mean(C.prop_time_grid(j, :), 2, 'omitnan');
            R.direct_time(i) = mean(C.direct_time_grid(j, :), 2, 'omitnan');
        end
    end
end

R.prop_sumrate = nan(numel(CV_grid), 1);
R.prop_pslr = nan(numel(CV_grid), 1);
R.prop_islr = nan(numel(CV_grid), 1);
R.direct_sumrate = nan(numel(CV_grid), 1);
R.direct_pslr = nan(numel(CV_grid), 1);
R.direct_islr = nan(numel(CV_grid), 1);
if exist(main_path, 'file') == 2
    S = load(main_path);
    for i = 1:numel(CV_grid)
        j = match_cv_index(CV_grid(i), S.CV_max_list(:));
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

function print_reward_summary(source_path)
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
fprintf('  Label-free reward-trained inference summary\n');
fprintf('============================================================\n');
fprintf('xi CVmax | RL-CV SR PSLR time feas | RL-Direct SR PSLR time feas\n');
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
fprintf(['[%5.1f%% %3d/%3d] %-10s MC %d/%d CV=%.1f xi=%.1f | ' ...
         'time=%8.5fs SR=%6.2f PSLR=%6.2f dB | %s | ETA %s\n'], ...
    100*run_count/total_runs, run_count, total_runs, name, mc, num_mc, ...
    CV_max, xi, elapsed_iter, sumrate, 10*log10(pslr_lin), status, format_time(eta));
end
