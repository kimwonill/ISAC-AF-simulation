function alpha = init_alpha(H, params)
% INIT_ALPHA  Initial subcarrier allocation: assign each subcarrier to the
% user with the strongest channel norm. Provides a strong warm-start for AO.

NT = params.NT; K = params.K; N = params.N; %#ok<NASGU>
alpha = zeros(K, N);
for n = 1:N
    h_norms = vecnorm(H(:, :, n), 2, 1);   % 1 x K
    [~, k_best] = max(h_norms);
    alpha(k_best, n) = 1;
end

end
