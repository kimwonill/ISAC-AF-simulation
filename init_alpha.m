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

% A positive per-user QoS constraint requires every such user to own at
% least one subcarrier before the first SDP solve.  Preserve the strongest-
% user initialization whenever it already has that support; otherwise move
% one subcarrier from a user that owns more than one.  This avoids declaring
% a channel realization infeasible solely because of the warm start.
qos_users = find(params.Q(:) > 0).';
for k = qos_users
    if any(alpha(k, :))
        continue;
    end
    owner_count = sum(alpha, 2);
    candidate = false(1, N);
    for n = 1:N
        owner = find(alpha(:, n), 1);
        candidate(n) = ~isempty(owner) && owner_count(owner) > 1;
    end
    candidate_idx = find(candidate);
    if isempty(candidate_idx)
        continue;
    end
    gains = squeeze(sum(abs(H(:, k, candidate_idx)).^2, 1));
    [~, best_local] = max(gains);
    n_best = candidate_idx(best_local);
    alpha(:, n_best) = 0;
    alpha(k, n_best) = 1;
end

end
