function W = init_covariance_mrt(H, alpha, params)
% INIT_COVARIANCE_MRT  Equal-power MRT covariance initialization.
%
% This initializer is independent across operating points and does not reuse
% any solution from a neighboring CV value. For each allocated subcarrier,
% it forms a rank-one covariance along the selected user's channel.

NT = params.NT;
K = params.K;
N = params.N;
per_tone_power = params.P_max / N;
mrt_weight = get_param(params, 'mrt_warm_start_weight', 1.0);
mrt_weight = min(max(mrt_weight, 0), 1);

W_mrt = zeros(NT, NT, N);
for n = 1:N
    k = find(alpha(:, n), 1);
    if isempty(k)
        h_norms = vecnorm(H(:, :, n), 2, 1);
        [~, k] = max(h_norms);
    end
    if k < 1 || k > K
        error('Invalid MRT user index at subcarrier %d.', n);
    end

    h = H(:, k, n);
    h_norm = norm(h, 2);
    if h_norm > 0
        v = h / h_norm;
        W_mrt(:, :, n) = per_tone_power * (v * v');
    else
        W_mrt(:, :, n) = (per_tone_power / NT) * eye(NT);
    end
end
if mrt_weight < 1
    W = mrt_weight * W_mrt + (1 - mrt_weight) * init_covariance_flat(params);
else
    W = W_mrt;
end
end

function value = get_param(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end
