function metrics = evaluate_beamforming_solution(H, alpha, W, A, params)
% EVALUATE_BEAMFORMING_SOLUTION  Metrics for an implementable covariance.
%
% W may be an SDP covariance or a rank-one covariance tensor. All reported
% communication and sensing metrics are evaluated directly from W.

K = params.K;
N = params.N;
L = params.L;

rate_kn = zeros(K, N);
for n = 1:N
    Wn = W(:, :, n);
    for k = 1:K
        hk = H(:, k, n);
        gain = max(real(hk' * Wn * hk), 0);
        rate_kn(k, n) = log2(1 + gain / params.sigma2);
    end
end

allocated_rate_kn = alpha .* rate_kn;
user_rate = sum(allocated_rate_kn, 2);

P = directional_power_grid(W, A);
mean_directional_power = mean(P, 1).';
centered_power = P - mean(P, 1);
std_directional_power = sqrt(mean(centered_power.^2, 1)).';
cv_per_target = std_directional_power ./ max(mean_directional_power, eps);

pslr_per_target = zeros(L, 1);
islr_per_target = zeros(L, 1);
for l = 1:L
    pslr_per_target(l) = compute_pslr(P(:, l), params.kappa);
    islr_per_target(l) = compute_islr(P(:, l), params.kappa);
end

power_total = 0;
for n = 1:N
    power_total = power_total + real(trace(W(:, :, n)));
end

metrics.sumrate = sum(user_rate);
metrics.user_rate = user_rate;
metrics.rate_kn = rate_kn;
metrics.directional_power = P;
metrics.mean_directional_power = mean_directional_power;
metrics.cv_per_target = cv_per_target;
metrics.pslr_per_target = pslr_per_target;
metrics.islr_per_target = islr_per_target;
metrics.power_total = power_total;
end
