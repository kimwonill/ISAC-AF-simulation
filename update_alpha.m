function alpha = update_alpha(H, W, params)
% UPDATE_ALPHA  Subcarrier allocation update given the current beamforming
% covariance tensor W. Implements:
%   (1) dual-decomposition weighted assignment per subcarrier;
%   (2) greedy QoS repair: for each violated user, steal the cheapest
%       subcarrier (smallest rate loss) from another user.
%
% The relaxed allocation subproblem assigns each subcarrier to the user that
% maximizes (1 + mu_k) R_{k,n}, then updates mu_k from the QoS subgradient.

K = params.K; N = params.N;

% --- per-(k,n) achievable rate under current W ---
R = zeros(K, N);
for n = 1:N
    Wn = W(:, :, n);
    for k = 1:K
        hk     = H(:, k, n);
        gkn    = max(real(hk' * Wn * hk), 0);
        R(k,n) = log2(1 + gkn / params.sigma2);
    end
end

dual_max_iter = get_param(params, 'dual_max_iter', 50);
dual_step0    = get_param(params, 'dual_step0', 0.5);
dual_tol      = get_param(params, 'dual_tol', 1e-4);

% --- Step 1: dual-decomposition assignment ---
mu = zeros(K, 1);
alpha = zeros(K, N);
for it = 1:dual_max_iter
    weights = bsxfun(@times, 1 + mu, R);
    [~, k_best] = max(weights, [], 1);

    alpha(:) = 0;
    for n = 1:N
        alpha(k_best(n), n) = 1;
    end

    user_rate = sum(alpha .* R, 2);
    qos_gap = params.Q - user_rate;
    if max(qos_gap) <= dual_tol
        break;
    end

    step = dual_step0 / sqrt(it);
    mu = max(0, mu + step * qos_gap);
end

% --- Step 2: greedy QoS repair ---
alpha = repair_qos(alpha, R, params.Q, dual_tol);

end

function value = get_param(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end

function alpha = repair_qos(alpha, R, Q, tol)
% Prefer swaps that keep the donor feasible; fall back to the smallest loss
% only when no safe donor exists.

[K, N] = size(R);
if all(Q <= 0)
    return;
end

for repair_pass = 1:(K * N)
    user_rate = sum(alpha .* R, 2);
    [max_gap, k_v] = max(Q - user_rate);
    if max_gap <= tol
        break;
    end

    n_other = find(alpha(k_v, :) == 0);
    if isempty(n_other)
        break;
    end

    best_n = [];
    best_cost = Inf;
    best_safe = false;

    for n_idx = n_other
        k_curr = find(alpha(:, n_idx), 1);
        if isempty(k_curr)
            continue;
        end

        cost = R(k_curr, n_idx) - R(k_v, n_idx);
        donor_safe = (user_rate(k_curr) - R(k_curr, n_idx)) >= Q(k_curr) - tol;

        if isempty(best_n) || ...
                (donor_safe && ~best_safe) || ...
                (donor_safe == best_safe && cost < best_cost)
            best_n = n_idx;
            best_cost = cost;
            best_safe = donor_safe;
        end
    end

    if isempty(best_n)
        break;
    end

    alpha(:, best_n) = 0;
    alpha(k_v, best_n) = 1;
end

end
