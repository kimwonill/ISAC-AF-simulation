function result = run_ml_policy_search(H, mode, constraint, params, opts)
% RUN_ML_POLICY_SEARCH  CEM policy search for beamforming baselines.
%
% This is a label-free ML baseline. A stochastic policy samples PSD
% covariance tensors from a low-dimensional basis-mixture model, and the
% Cross-Entropy Method updates the policy toward high-reward feasible
% candidates.
%
% mode = 'cv'     : solve the proposed CV-constrained problem with penalties.
% mode = 'direct' : solve the direct PSLR/ISLR-constrained problem.

if nargin < 5 || isempty(opts)
    opts = struct();
end
opts = fill_default_opts(opts);

A = compute_steering(params);
basis = make_policy_basis(H, A, params);
[mu, sigma0] = initial_policy_distribution(H, A, basis, mode, constraint, params, opts);

D = numel(mu);
sigma = sigma0 * ones(D, 1);
elite_count = max(2, ceil(opts.elite_frac * opts.population));

best_score = -Inf;
best_metrics = [];
best_W = [];
best_feasible_score = -Inf;
best_feasible_metrics = [];
best_feasible_W = [];
history_best_score = nan(opts.max_iter, 1);
history_best_feasible = false(opts.max_iter, 1);

t_start = tic;
for iter = 1:opts.max_iter
    Z = mu + sigma .* randn(D, opts.population);
    Z(:, 1) = mu;
    if iter == 1
        Z(:, 2) = initial_policy_vector(H, A, basis, mode, constraint, params, opts);
    end

    scores = nan(opts.population, 1);
    feasible = false(opts.population, 1);
    metrics_cell = cell(opts.population, 1);
    W_cell = cell(opts.population, 1);

    for p = 1:opts.population
        W = policy_vector_to_covariance(Z(:, p), basis, params);
        metrics = evaluate_policy_candidate(H, W, A, params);
        [scores(p), feasible(p)] = score_policy_candidate(metrics, mode, constraint, params, opts);

        metrics_cell{p} = metrics;
        W_cell{p} = W;

        if scores(p) > best_score
            best_score = scores(p);
            best_metrics = metrics;
            best_W = W;
        end
        if feasible(p) && metrics.sumrate > best_feasible_score
            best_feasible_score = metrics.sumrate;
            best_feasible_metrics = metrics;
            best_feasible_W = W;
        end
    end

    [~, order] = sort(scores, 'descend');
    elites = Z(:, order(1:elite_count));
    elite_mu = mean(elites, 2);
    elite_sigma = std(elites, 0, 2);

    mu = (1 - opts.smoothing) * mu + opts.smoothing * elite_mu;
    sigma = (1 - opts.smoothing) * sigma + opts.smoothing * elite_sigma;
    sigma = min(max(sigma, opts.min_sigma), opts.max_sigma);

    history_best_score(iter) = best_score;
    history_best_feasible(iter) = ~isempty(best_feasible_metrics);

    if opts.verbose
        fprintf('    CEM %s iter %02d/%02d | best score %.2f | feasible %d/%d\n', ...
            upper(mode), iter, opts.max_iter, best_score, nnz(feasible), opts.population);
    end
end

if ~isempty(best_feasible_metrics)
    metrics = best_feasible_metrics;
    W = best_feasible_W;
    status = 'ML feasible';
else
    metrics = best_metrics;
    W = best_W;
    status = 'ML penalty-best';
end

result.W = W;
result.alpha = metrics.alpha;
result.sumrate = metrics.sumrate;
result.pslr_per_target = metrics.pslr_per_target;
result.islr_per_target = metrics.islr_per_target;
result.cv_per_target = metrics.cv_per_target;
result.mu_p_per_target = metrics.mu_p_per_target;
result.user_rate = metrics.user_rate;
result.status = status;
result.elapsed = toc(t_start);
result.best_score = best_score;
result.feasible = ~isempty(best_feasible_metrics);
result.history_best_score = history_best_score;
result.history_best_feasible = history_best_feasible;
result.mode = mode;
result.constraint = constraint;
end

function opts = fill_default_opts(opts)
opts = set_default(opts, 'population', 70);
opts = set_default(opts, 'max_iter', 24);
opts = set_default(opts, 'elite_frac', 0.15);
opts = set_default(opts, 'smoothing', 0.65);
opts = set_default(opts, 'sigma0', 1.15);
opts = set_default(opts, 'min_sigma', 0.04);
opts = set_default(opts, 'max_sigma', 2.0);
opts = set_default(opts, 'penalty_qos', 80);
opts = set_default(opts, 'penalty_illumination', 80);
opts = set_default(opts, 'penalty_cv', 65);
opts = set_default(opts, 'penalty_pslr', 12);
opts = set_default(opts, 'penalty_islr', 12);
opts = set_default(opts, 'constraint_tol', 1e-4);
opts = set_default(opts, 'verbose', false);
end

function opts = set_default(opts, name, value)
if ~isfield(opts, name)
    opts.(name) = value;
end
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

function [mu, sigma0] = initial_policy_distribution(H, A, basis, mode, constraint, params, opts)
mu = initial_policy_vector(H, A, basis, mode, constraint, params, opts);
sigma0 = opts.sigma0;
end

function z0 = initial_policy_vector(H, A, basis, mode, constraint, params, opts) %#ok<INUSD>
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

function metrics = evaluate_policy_candidate(H, W, A, params)
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

function [score, feasible] = score_policy_candidate(metrics, mode, constraint, params, opts)
qos_scale = max(params.Q, 1);
qos_gap = max(0, params.Q - metrics.user_rate) ./ qos_scale;
illum_gap = max(0, params.P_des - metrics.mu_p_per_target) ./ max(params.P_des, eps);

qos_penalty = sum(qos_gap.^2);
illum_penalty = sum(illum_gap.^2);

if strcmpi(mode, 'cv')
    denom = max(constraint.CV_max, 0.05);
    cv_gap = max(0, metrics.cv_per_target - constraint.CV_max) ./ denom;
    sensing_penalty = opts.penalty_cv * sum(cv_gap.^2);
else
    pslr_min = constraint.pslr_min;
    islr_max = constraint.islr_max;
    pslr_gap_dB = max(0, 10*log10(pslr_min) - 10*log10(metrics.pslr_per_target));
    if isfinite(islr_max)
        islr_gap_dB = max(0, 10*log10(metrics.islr_per_target) - 10*log10(islr_max));
    else
        islr_gap_dB = zeros(size(metrics.islr_per_target));
    end
    sensing_penalty = opts.penalty_pslr * sum(pslr_gap_dB.^2) + ...
        opts.penalty_islr * sum(islr_gap_dB.^2);
end

score = metrics.sumrate ...
    - opts.penalty_qos * qos_penalty ...
    - opts.penalty_illumination * illum_penalty ...
    - sensing_penalty;

if strcmpi(mode, 'cv')
    sensing_ok = all(metrics.cv_per_target <= constraint.CV_max + opts.constraint_tol);
else
    pslr_ok = all(metrics.pslr_per_target >= constraint.pslr_min * (1 - opts.constraint_tol));
    islr_ok = ~isfinite(constraint.islr_max) || ...
        all(metrics.islr_per_target <= constraint.islr_max * (1 + opts.constraint_tol));
    sensing_ok = pslr_ok && islr_ok;
end

feasible = all(metrics.user_rate >= params.Q - opts.constraint_tol) && ...
    all(metrics.mu_p_per_target >= params.P_des - opts.constraint_tol) && ...
    sensing_ok;
end

function value = get_field(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end
