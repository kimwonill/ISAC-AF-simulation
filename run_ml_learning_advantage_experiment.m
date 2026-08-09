function run_ml_learning_advantage_experiment(num_train_mc_override, num_val_mc_override, force_rerun)
% RUN_ML_LEARNING_ADVANTAGE_EXPERIMENT
% Label-free experiment showing why the CV-reformulated constraint is easier
% to enforce than the direct PSLR constraint.
%
% The experiment compares sensing-only feasibility and sum-rate under a
% fixed training budget across a CV-threshold sweep.

if nargin < 1 || isempty(num_train_mc_override)
    num_train_mc_override = [];
end
if nargin < 2 || isempty(num_val_mc_override)
    num_val_mc_override = [];
end
if nargin < 3 || isempty(force_rerun)
    force_rerun = false;
end

clearvars -except num_train_mc_override num_val_mc_override force_rerun; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, 'figures');
paper_fig_dir = fullfile(sim_dir, '..', 'MyPaper', 'figures');
out_data_dir = fullfile(sim_dir, 'results');
if exist(out_data_dir, 'dir') ~= 7
    mkdir(out_data_dir);
end
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
if exist(paper_fig_dir, 'dir') ~= 7, mkdir(paper_fig_dir); end
addpath(genpath(sim_dir));

params = setup_params();
params.warm_start_cv = false;
params.sdp_quiet = true;

CV_grid = 0:0.1:1.0;
num_train_mc = 8;
num_val_mc = 28;
if ~isempty(num_train_mc_override)
    num_train_mc = num_train_mc_override;
end
if ~isempty(num_val_mc_override)
    num_val_mc = num_val_mc_override;
end

opts = struct();
opts.population = 18;
opts.max_iter = 8;
opts.elite_frac = 0.22;
opts.smoothing = 0.65;
opts.sigma0 = 0.85;
opts.min_sigma = 0.04;
opts.constraint_tol = 1e-4;
opts.penalty_qos = 60;
opts.penalty_sensing = 80;
opts.verbose = true;

result_path = fullfile(out_data_dir, 'ml_learning_advantage_pslr_only_results.mat');
if exist(result_path, 'file') == 2 && ~force_rerun
    cached = load(result_path, 'num_train_mc', 'num_val_mc');
    if isfield(cached, 'num_train_mc') && isfield(cached, 'num_val_mc') && ...
            cached.num_train_mc == num_train_mc && cached.num_val_mc == num_val_mc
        fprintf('Loading cached ML learning advantage result: %s\n', result_path);
        plot_learning_advantage(result_path, fig_dir, paper_fig_dir);
        print_learning_summary(result_path);
        return;
    end
    fprintf('Ignoring cache with MC train=%d, validation=%d; requested train=%d, validation=%d.\n', ...
        cached.num_train_mc, cached.num_val_mc, num_train_mc, num_val_mc);
end

fprintf('============================================================\n');
fprintf('  ML learning advantage experiment: CV vs Direct constraints\n');
fprintf('============================================================\n');
fprintf('  Train MC=%d, validation MC=%d, CV=[%s]\n', ...
    num_train_mc, num_val_mc, num2str(CV_grid));
fprintf('  CEM population=%d, iterations=%d\n', opts.population, opts.max_iter);
fprintf('------------------------------------------------------------\n');

train_set = make_learning_set(params, CV_grid, num_train_mc, 7100);
val_set = make_learning_set(params, CV_grid, num_val_mc, 8100);

t_start = tic;
[policy_cv, trace_cv] = cem_train_learning(train_set, val_set, 'cv', params, opts, 9101, 8);
[policy_direct, trace_direct] = cem_train_learning(train_set, val_set, 'direct', params, opts, 9201, 8);

sweep = evaluate_sensing_sweep(val_set, CV_grid, policy_cv, policy_direct, params, opts);

elapsed = toc(t_start);
save(result_path, 'params', 'CV_grid', 'num_train_mc', 'num_val_mc', 'opts', ...
    'policy_cv', 'policy_direct', 'trace_cv', 'trace_direct', ...
    'sweep', 'elapsed');

plot_learning_advantage(result_path, fig_dir, paper_fig_dir);
print_learning_summary(result_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', result_path);
fprintf('  Total elapsed: %s\n', format_time(elapsed));
fprintf('============================================================\n');
end

function train_set = make_learning_set(params, CV_grid, num_mc, seed0)
train_set = repmat(struct('H', [], 'CV_max', []), numel(CV_grid) * num_mc, 1);
idx = 0;
for mc = 1:num_mc
    rng(seed0 + mc, 'twister');
    H = generate_channel(params);
    for c = 1:numel(CV_grid)
        idx = idx + 1;
        train_set(idx).H = H;
        train_set(idx).CV_max = CV_grid(c);
    end
end
end

function [policy_best, trace] = cem_train_learning(train_set, val_set, mode, params, opts, seed, D)
rng(seed, 'twister');
mu = zeros(D, 1);
sigma = opts.sigma0 * ones(D, 1);
elite_count = max(2, ceil(opts.elite_frac * opts.population));

theta_best = mu;
score_best = -Inf;
trace.policy_evals = nan(opts.max_iter, 1);
trace.sample_evals = nan(opts.max_iter, 1);
trace.train_best_reward = nan(opts.max_iter, 1);
trace.train_elite_reward = nan(opts.max_iter, 1);
trace.val_weighted_sumrate = nan(opts.max_iter, 1);
trace.val_feasible_sumrate = nan(opts.max_iter, 1);
trace.val_feasibility = nan(opts.max_iter, 1);
trace.val_violation = nan(opts.max_iter, 1);

fprintf('Training %s policy (D=%d)...\n', upper(mode), D);
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

    policy_iter = theta_to_policy(theta_best);
    val = evaluate_policy_set(val_set, mode, policy_iter, params, opts);
    trace.policy_evals(iter) = iter * opts.population;
    trace.sample_evals(iter) = iter * opts.population * numel(train_set);
    trace.train_best_reward(iter) = score_best;
    trace.train_elite_reward(iter) = mean(scores(order(1:elite_count)));
    trace.val_weighted_sumrate(iter) = val.weighted_sumrate;
    trace.val_feasible_sumrate(iter) = val.feasible_sumrate;
    trace.val_feasibility(iter) = val.feasibility;
    trace.val_violation(iter) = val.violation;

    if opts.verbose
        fprintf(['  %s iter %02d/%02d | train %.3f | val Rw %.2f | ' ...
            'feas %.2f | viol %.3g\n'], upper(mode), iter, opts.max_iter, ...
            score_best, val.weighted_sumrate, val.feasibility, val.violation);
    end
end

policy_best = theta_to_policy(theta_best);
policy_best.mode = mode;
policy_best.dimension = D;
end

function score = evaluate_policy_reward(data_set, mode, policy, params, opts)
score = 0;
for i = 1:numel(data_set)
    constraint = make_constraint(mode, data_set(i).CV_max, params);
    result = infer_once(data_set(i).H, mode, constraint, policy, params, opts);
    score = score + constrained_reward(result, mode, constraint, params, opts);
end
score = score / numel(data_set);
end

function metrics = evaluate_policy_set(data_set, mode, policy, params, opts)
num_data = numel(data_set);
sumrate = nan(num_data, 1);
reward = nan(num_data, 1);
feasible = false(num_data, 1);
violation = nan(num_data, 1);

for i = 1:num_data
    constraint = make_constraint(mode, data_set(i).CV_max, params);
    result = infer_once(data_set(i).H, mode, constraint, policy, params, opts);
    sumrate(i) = result.sumrate;
    feasible(i) = result.feasible;
    reward(i) = constrained_reward(result, mode, constraint, params, opts);
    violation(i) = constraint_violation(result, mode, constraint, params);
end

metrics.sumrate = sumrate;
metrics.reward = reward;
metrics.feasible = feasible;
metrics.violation_each = violation;
metrics.feasibility = mean(double(feasible));
metrics.weighted_sumrate = mean(sumrate .* double(feasible), 'omitnan');
if any(feasible)
    metrics.feasible_sumrate = mean(sumrate(feasible), 'omitnan');
else
    metrics.feasible_sumrate = 0;
end
metrics.violation = mean(violation, 'omitnan');
metrics.reward_mean = mean(reward, 'omitnan');
end

function constraint = make_constraint(mode, CV_max, params)
if strcmpi(mode, 'cv')
    constraint = struct('CV_max', CV_max, 'CV_hint', CV_max);
else
    pslr_min = direct_thresholds_from_cv(CV_max, params);
    constraint = struct('pslr_min', pslr_min, 'CV_hint', CV_max);
end
end

function sweep = evaluate_sensing_sweep(val_set, CV_grid, policy_cv, policy_direct, params, opts)
num_cv = numel(CV_grid);
sweep.CV_grid = CV_grid(:);
sweep.cv_sensing_feasibility = nan(num_cv, 1);
sweep.direct_sensing_feasibility = nan(num_cv, 1);
sweep.cv_sumrate = nan(num_cv, 1);
sweep.direct_sumrate = nan(num_cv, 1);

for c = 1:num_cv
    subset = val_set(abs([val_set.CV_max] - CV_grid(c)) < 1e-12);
    cv_ok = false(numel(subset), 1);
    direct_ok = false(numel(subset), 1);
    cv_rate = nan(numel(subset), 1);
    direct_rate = nan(numel(subset), 1);
    for i = 1:numel(subset)
        cv_constraint = make_constraint('cv', CV_grid(c), params);
        direct_constraint = make_constraint('direct', CV_grid(c), params);
        cv_result = infer_once(subset(i).H, 'cv', cv_constraint, policy_cv, params, opts);
        direct_result = infer_once(subset(i).H, 'direct', direct_constraint, policy_direct, params, opts);
        cv_ok(i) = sensing_feasible(cv_result, 'cv', cv_constraint, opts.constraint_tol);
        direct_ok(i) = sensing_feasible(direct_result, 'direct', direct_constraint, opts.constraint_tol);
        cv_rate(i) = cv_result.sumrate;
        direct_rate(i) = direct_result.sumrate;
    end
    sweep.cv_sensing_feasibility(c) = mean(double(cv_ok));
    sweep.direct_sensing_feasibility(c) = mean(double(direct_ok));
    sweep.cv_sumrate(c) = mean(cv_rate, 'omitnan');
    sweep.direct_sumrate(c) = mean(direct_rate, 'omitnan');
end
end

function variance = reward_sensitivity_probe(data_set, CV_grid, params, opts, D, num_probe, sigma_probe)
fprintf('Running local reward-sensitivity probe...\n');
variance.CV_grid = CV_grid(:);
variance.tightness = 1 - CV_grid(:);
variance.cv_reward_std = nan(numel(CV_grid), 1);
variance.direct_reward_std = nan(numel(CV_grid), 1);
variance.cv_reward_cv = nan(numel(CV_grid), 1);
variance.direct_reward_cv = nan(numel(CV_grid), 1);
variance.cv_violation_std = nan(numel(CV_grid), 1);
variance.direct_violation_std = nan(numel(CV_grid), 1);

for c = 1:numel(CV_grid)
    subset = data_set(abs([data_set.CV_max] - CV_grid(c)) < 1e-12);
    reward_cv = nan(num_probe, 1);
    reward_direct = nan(num_probe, 1);
    viol_cv = nan(num_probe, 1);
    viol_direct = nan(num_probe, 1);
    rng(11000 + c, 'twister');
    for p = 1:num_probe
        theta = sigma_probe * randn(D, 1);
        policy = theta_to_policy(theta);
        m_cv = evaluate_policy_set(subset, 'cv', policy, params, opts);
        m_direct = evaluate_policy_set(subset, 'direct', policy, params, opts);
        reward_cv(p) = m_cv.reward_mean;
        reward_direct(p) = m_direct.reward_mean;
        viol_cv(p) = m_cv.violation;
        viol_direct(p) = m_direct.violation;
    end
    variance.cv_reward_std(c) = std(reward_cv, 0, 'omitnan');
    variance.direct_reward_std(c) = std(reward_direct, 0, 'omitnan');
    variance.cv_reward_cv(c) = variance.cv_reward_std(c) / max(abs(mean(reward_cv, 'omitnan')), 1);
    variance.direct_reward_cv(c) = variance.direct_reward_std(c) / max(abs(mean(reward_direct, 'omitnan')), 1);
    variance.cv_violation_std(c) = std(viol_cv, 0, 'omitnan');
    variance.direct_violation_std(c) = std(viol_direct, 0, 'omitnan');
    fprintf('  CV=%.2f | reward std CV %.3g, Direct %.3g | violation std CV %.3g, Direct %.3g\n', ...
        CV_grid(c), variance.cv_reward_std(c), variance.direct_reward_std(c), ...
        variance.cv_violation_std(c), variance.direct_violation_std(c));
end
end

function model_summary = model_size_sweep(train_set, val_set, model_sizes, params, opts)
fprintf('Running model-size sweep...\n');
num_sizes = numel(model_sizes);
model_summary.cv_weighted_sumrate = nan(num_sizes, 1);
model_summary.direct_weighted_sumrate = nan(num_sizes, 1);
model_summary.cv_feasibility = nan(num_sizes, 1);
model_summary.direct_feasibility = nan(num_sizes, 1);
model_summary.cv_violation = nan(num_sizes, 1);
model_summary.direct_violation = nan(num_sizes, 1);

for i = 1:num_sizes
    D = model_sizes(i);
    [policy_cv, trace_cv] = cem_train_learning(train_set, val_set, 'cv', params, opts, 12000 + D, D);
    [policy_direct, trace_direct] = cem_train_learning(train_set, val_set, 'direct', params, opts, 13000 + D, D);
    val_cv = evaluate_policy_set(val_set, 'cv', policy_cv, params, opts);
    val_direct = evaluate_policy_set(val_set, 'direct', policy_direct, params, opts);
    model_summary.cv_weighted_sumrate(i) = val_cv.weighted_sumrate;
    model_summary.direct_weighted_sumrate(i) = val_direct.weighted_sumrate;
    model_summary.cv_feasibility(i) = val_cv.feasibility;
    model_summary.direct_feasibility(i) = val_direct.feasibility;
    model_summary.cv_violation(i) = val_cv.violation;
    model_summary.direct_violation(i) = val_direct.violation;
    model_summary.cv_trace{i} = trace_cv;
    model_summary.direct_trace{i} = trace_direct;
    fprintf('  D=%d | CV Rw %.2f feas %.2f | Direct Rw %.2f feas %.2f\n', ...
        D, val_cv.weighted_sumrate, val_cv.feasibility, ...
        val_direct.weighted_sumrate, val_direct.feasibility);
end
end

function policy = theta_to_policy(theta)
theta_full = zeros(8, 1);
theta = theta(:);
theta_full(1:min(numel(theta), 8)) = theta(1:min(numel(theta), 8));
policy.theta = theta;
policy.theta_full = theta_full;
policy.rho_iso = 1.35 * sigmoid(theta_full(1));
policy.rho_struct = 1.35 * sigmoid(theta_full(2));
policy.rho_coherent = 1.35 * sigmoid(theta_full(3));
policy.struct_bias = theta_full(4);
policy.struct_tightness = theta_full(5);
policy.coherent_bias = theta_full(6);
policy.coherent_tightness = theta_full(7);
policy.score_temperature = 0.5 + 2.0 * sigmoid(theta_full(8));
end

function result = infer_once(H, mode, constraint, policy, params, opts)
A = compute_steering(params);
rho = 1.35 * sigmoid(policy.theta_full(1) + ...
    policy.theta_full(5) * (constraint.CV_hint - 0.5));
W_raw = make_comm_covariance(H, params, rho);
if strcmpi(mode, 'cv')
    W = apply_cv_safety_layer(W_raw, constraint.CV_max, A, params);
    branch = 'CV safety layer';
else
    W = W_raw;
    branch = 'Direct PSLR penalty';
end
metrics = evaluate_candidate(H, W, A, params);

[feasible, status] = check_feasible(metrics, mode, constraint, params, opts.constraint_tol);
result.W = W;
result.alpha = metrics.alpha;
result.sumrate = metrics.sumrate;
result.pslr_per_target = metrics.pslr_per_target;
result.cv_per_target = metrics.cv_per_target;
result.mu_p_per_target = metrics.mu_p_per_target;
result.user_rate = metrics.user_rate;
result.feasible = feasible;
result.status = status;
result.branch = branch;
result.mode = mode;
result.constraint = constraint;
end

function W_safe = apply_cv_safety_layer(W, CV_max, A, params)
W_bar = mean(W, 3);
rho = 1;
for l = 1:params.L
    Pn = compute_directional_power(W, A(:, l));
    mu_p = mean(Pn);
    sigma_p = sqrt(mean((Pn - mu_p).^2));
    if sigma_p > 0
        rho = min(rho, CV_max * mu_p / sigma_p);
    end
end
rho = min(max(real(rho), 0), 1);
W_safe = zeros(size(W));
for n = 1:params.N
    W_safe(:, :, n) = rho * W(:, :, n) + (1 - rho) * W_bar;
end
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
        if pslr < constraint.pslr_min * (1 - tol)
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
metrics.cv_per_target = nan(params.L, 1);
metrics.mu_p_per_target = nan(params.L, 1);
for l = 1:params.L
    Pn = compute_directional_power(W, A(:, l));
    metrics.pslr_per_target(l) = compute_pslr(Pn, params.kappa);
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
    penalty = penalty + opts.penalty_sensing * sum(pslr_gap.^2);
end
reward = metrics.sumrate - penalty;
end

function [feasible, status] = check_feasible(metrics, mode, constraint, params, tol)
qos_ok = all(metrics.user_rate >= params.Q - tol);
illum_ok = all(metrics.mu_p_per_target >= params.P_des - tol);
if strcmpi(mode, 'cv')
    sensing_flag = all(metrics.cv_per_target <= constraint.CV_max + tol);
else
    sensing_flag = all(metrics.pslr_per_target >= constraint.pslr_min * (1 - tol));
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

function violation = constraint_violation(metrics, mode, constraint, params)
qos_gap = max(0, params.Q - metrics.user_rate) ./ max(params.Q, 1);
illum_gap = max(0, params.P_des - metrics.mu_p_per_target) ./ max(params.P_des, eps);
violation = sum(qos_gap) + sum(illum_gap);
if strcmpi(mode, 'cv')
    denom = max(constraint.CV_max, 0.05);
    violation = violation + sum(max(0, metrics.cv_per_target - constraint.CV_max) ./ denom);
else
    pslr_gap = max(0, 10*log10(constraint.pslr_min) - 10*log10(metrics.pslr_per_target)) / 10;
    violation = violation + sum(pslr_gap);
end
end

function feasible = sensing_feasible(metrics, mode, constraint, tol)
if strcmpi(mode, 'cv')
    feasible = all(metrics.cv_per_target <= constraint.CV_max + tol);
else
    feasible = all(metrics.pslr_per_target >= constraint.pslr_min * (1 - tol));
end
end

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

function plot_learning_advantage(result_path, fig_dir, paper_fig_dir)
M = load(result_path);
cfg = plot_config();
font_scale = cfg.compact_panel_font_scale;
axes_font = round(font_scale * cfg.axes_font);
label_font = round(font_scale * cfg.label_font);
title_font = round(font_scale * cfg.title_font);
legend_font = round(font_scale * cfg.legend_font);
fig = figure('Position', [80 80 760 405], 'Color', 'w');
palette = paper_palette();
cv_color = palette(1, :);
direct_color = palette(2, :);
ax1 = axes(fig, 'Position', [0.125 0.225 0.320 0.485]);
hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
h_cv = plot(ax1, M.sweep.CV_grid, M.sweep.cv_sensing_feasibility, '-o', ...
    'Color', cv_color, 'MarkerFaceColor', cv_color, ...
    'LineWidth', cfg.secondary_line_width, 'MarkerSize', cfg.compact_marker_size);
h_direct = plot(ax1, M.sweep.CV_grid, M.sweep.direct_sensing_feasibility, '--s', ...
    'Color', direct_color, 'MarkerFaceColor', direct_color, ...
    'LineWidth', cfg.secondary_line_width, 'MarkerSize', cfg.compact_marker_size);
xlabel(ax1, 'CV_{max}', 'Interpreter', 'tex');
ylabel(ax1, 'Feasibility');
title(ax1, '(a) Constraint tightness', 'FontWeight', 'normal');
xlim(ax1, valid_axis_limits(M.sweep.CV_grid, cfg, 'Clip', [0 1]));
ylim(ax1, [0 1]);
xticks(ax1, [0.5 1.0]); yticks(ax1, [0 0.5 1]);

ax2 = axes(fig, 'Position', [0.635 0.225 0.320 0.485]);
hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
plot(ax2, M.sweep.CV_grid, M.sweep.cv_sumrate, '-o', ...
    'Color', cv_color, 'MarkerFaceColor', cv_color, ...
    'LineWidth', cfg.secondary_line_width, 'MarkerSize', cfg.compact_marker_size);
plot(ax2, M.sweep.CV_grid, M.sweep.direct_sumrate, '--s', ...
    'Color', direct_color, 'MarkerFaceColor', direct_color, ...
    'LineWidth', cfg.secondary_line_width, 'MarkerSize', cfg.compact_marker_size);
xlabel(ax2, 'CV_{max}', 'Interpreter', 'tex');
ylabel(ax2, 'Rate (bps/Hz)');
title(ax2, '(b) Fixed training budget', 'FontWeight', 'normal');
xlim(ax2, valid_axis_limits(M.sweep.CV_grid, cfg, 'Clip', [0 1]));
xticks(ax2, [0.5 1.0]);
rate_values = [M.sweep.cv_sumrate(:); M.sweep.direct_sumrate(:)];
ylim(ax2, valid_axis_limits(rate_values, cfg, 'Clip', [0 Inf]));

set([ax1 ax2], 'FontName', cfg.font_name, 'FontSize', axes_font, ...
    'LineWidth', cfg.axes_line_width, 'LabelFontSizeMultiplier', 1);
set([ax1.XLabel ax1.YLabel ax2.XLabel ax2.YLabel], 'FontSize', label_font);
set([ax1.Title ax2.Title], 'FontSize', title_font);
lgd = legend(ax1, [h_cv h_direct], {'CV-NN + safety', 'Direct-PSLR NN penalty'}, ...
    'Orientation', 'horizontal', 'NumColumns', 2, 'Location', 'none');
set(lgd, 'Units', 'normalized', 'Position', [0.205 0.855 0.590 0.085], ...
    'FontName', cfg.font_name, 'FontSize', legend_font, 'Box', 'on', ...
    'Color', cfg.legend_background_color, 'EdgeColor', cfg.legend_edge_color);
try
    lgd.ItemTokenSize = [18 8];
catch
end
plot_config(fig);
set([ax1 ax2], 'FontSize', axes_font);
set([ax1.XLabel ax1.YLabel ax2.XLabel ax2.YLabel], 'FontSize', label_font);
set([ax1.Title ax2.Title], 'FontSize', title_font);
set(lgd, 'FontSize', legend_font);

sim_pdf = fullfile(fig_dir, 'ML_CV_Advantage_Integrated_Result.pdf');
sim_png = fullfile(fig_dir, 'ML_CV_Advantage_Integrated_Result.png');
tight_export_figure(fig, sim_pdf, ...
    'ContentType', 'image', 'Resolution', 300, ...
    'TightLayout', false, 'TightPad', 0);
tight_export_figure(fig, sim_png, ...
    'Resolution', 300, 'TightLayout', false, 'TightPad', 0);
copyfile(sim_pdf, fullfile(paper_fig_dir, 'ML_CV_Advantage_Integrated_Result.pdf'));
copyfile(sim_png, fullfile(paper_fig_dir, 'ML_CV_Advantage_Integrated_Result.png'));
end

function save_figure(fig, sim_dir, paper_fig_dir, stem, paper_stem)
set(fig, 'PaperPositionMode', 'auto');
sim_png = fullfile(sim_dir, [stem '.png']);
sim_fig = fullfile(sim_dir, [stem '.fig']);
paper_pdf = fullfile(paper_fig_dir, [paper_stem '.pdf']);
savefig(fig, sim_fig);
tight_export_figure(fig, sim_png, 'Resolution', 300, ...
    'TightLayout', false, 'TightPad', 0);
tight_export_figure(fig, paper_pdf, 'ContentType', 'image', 'Resolution', 300, ...
    'TightLayout', false, 'TightPad', 0);
end

function print_learning_summary(result_path)
M = load(result_path);
fprintf('============================================================\n');
fprintf('  PSLR-only learning advantage summary\n');
fprintf('============================================================\n');
for c = 1:numel(M.sweep.CV_grid)
    fprintf(['CV=%.1f | sensing feasibility CV %.1f%%, Direct %.1f%% | ' ...
        'rate CV %.2f, Direct %.2f bps/Hz\n'], M.sweep.CV_grid(c), ...
        100*M.sweep.cv_sensing_feasibility(c), ...
        100*M.sweep.direct_sensing_feasibility(c), ...
        M.sweep.cv_sumrate(c), M.sweep.direct_sumrate(c));
end
end

function eval_count = first_eval_at_feasibility(trace, target)
idx = find(trace.val_feasibility >= target, 1);
if isempty(idx)
    eval_count = NaN;
else
    eval_count = trace.policy_evals(idx);
end
end

function s = format_eval(x)
if isnan(x)
    s = 'not reached';
else
    s = sprintf('%d', round(x));
end
end
