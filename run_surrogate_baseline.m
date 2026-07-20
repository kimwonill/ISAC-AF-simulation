function result = run_surrogate_baseline(H, mode, eta, params, alpha0)
% RUN_SURROGATE_BASELINE  AO loop for CRB-/MI-inspired non-AF baselines.

A = compute_steering(params);

if nargin >= 5 && ~isempty(alpha0) && isequal(size(alpha0), [params.K, params.N])
    alpha = alpha0;
else
    alpha = init_alpha(H, params);
end

best_sumrate = -Inf;
W_best = []; alpha_best = []; status_best = 'Not Solved';
prev_sumrate = -Inf;
stop_reason = 'max_iter';

for t = 1:params.max_iter
    [W, sumrate, status] = solve_surrogate_sdp(H, alpha, A, mode, eta, params);
    if isempty(W)
        result.status = status;
        result.sumrate = NaN;
        result.pslr_per_target = nan(params.L, 1);
        result.islr_per_target = nan(params.L, 1);
        result.W = [];
        result.alpha = alpha;
        result.iters = t;
        result.stop_reason = 'failed';
        return;
    end

    if sumrate > best_sumrate
        best_sumrate = sumrate;
        W_best = W;
        alpha_best = alpha;
        status_best = status;
    end

    if abs(sumrate - prev_sumrate) < params.tol
        stop_reason = 'sumrate_tol';
        break;
    end
    prev_sumrate = sumrate;

    alpha_next = update_alpha(H, W, params);
    if isfield(params, 'stop_if_alpha_unchanged') && params.stop_if_alpha_unchanged && isequal(alpha_next, alpha)
        stop_reason = 'alpha_fixed';
        break;
    end
    alpha = alpha_next;
end

pslr_per_target = zeros(params.L, 1);
islr_per_target = zeros(params.L, 1);
for l = 1:params.L
    Pn = compute_directional_power(W_best, A(:, l));
    pslr_per_target(l) = compute_pslr(Pn, params.kappa);
    islr_per_target(l) = compute_islr(Pn, params.kappa);
end

result.W = W_best;
result.alpha = alpha_best;
result.sumrate = best_sumrate;
result.pslr_per_target = pslr_per_target;
result.islr_per_target = islr_per_target;
result.status = status_best;
result.iters = t;
result.stop_reason = stop_reason;
result.rank_stats = compute_rank_stats(W_best);

end
