function result = run_proposed(H, CV_max, params, alpha0)
% RUN_PROPOSED  AO loop for problem (P_prop): alternates between the
% beamforming SDP and the subcarrier-allocation update until the sum-rate
% converges or max_iter is reached.

A = compute_steering(params);

if nargin >= 4 && ~isempty(alpha0) && isequal(size(alpha0), [params.K, params.N])
    alpha = alpha0;
else
    alpha = init_alpha(H, params);
end

stop_if_alpha_unchanged = get_param(params, 'stop_if_alpha_unchanged', true);
best_sumrate = -Inf;
W_best = []; alpha_best = []; status_best = 'Not Solved';
stop_reason = 'max_iter';
cvx_solver_iters = 0;
cvx_solver_iters_history = nan(params.max_iter, 1);

prev_sumrate = -Inf;
for t = 1:params.max_iter
    [W, sumrate, status, slvitr] = solve_sdp(H, alpha, A, CV_max, params);
    cvx_solver_iters_history(t) = slvitr;
    cvx_solver_iters = cvx_solver_iters + zero_if_nan(slvitr);
    if isempty(W)
        % SDP infeasible / failed -- stop early
        result.status = status;
        result.sumrate = NaN;
        result.pslr_per_target = nan(params.L, 1);
        result.islr_per_target = nan(params.L, 1);
        result.W = []; result.alpha = alpha;
        result.iters = t;
        result.cvx_solver_iters = cvx_solver_iters;
        result.cvx_solver_iters_history = cvx_solver_iters_history(1:t);
        return;
    end

    if sumrate > best_sumrate
        best_sumrate = sumrate;
        W_best = W; alpha_best = alpha; status_best = status;
    end

    if abs(sumrate - prev_sumrate) < params.tol
        stop_reason = 'sumrate_tol';
        break;
    end
    prev_sumrate = sumrate;

    alpha_next = update_alpha(H, W, params);
    if stop_if_alpha_unchanged && isequal(alpha_next, alpha)
        stop_reason = 'alpha_fixed';
        break;
    end
    alpha = alpha_next;
end

% --- final metrics from the best snapshot ---
pslr_per_target = zeros(params.L, 1);
islr_per_target = zeros(params.L, 1);
for l = 1:params.L
    Pn = compute_directional_power(W_best, A(:, l));
    pslr_per_target(l) = compute_pslr(Pn, params.kappa);
    islr_per_target(l) = compute_islr(Pn, params.kappa);
end

result.W                = W_best;
result.alpha            = alpha_best;
result.sumrate          = best_sumrate;
result.pslr_per_target  = pslr_per_target;
result.islr_per_target  = islr_per_target;
result.status           = status_best;
result.iters            = t;
result.cvx_solver_iters = cvx_solver_iters;
result.cvx_solver_iters_history = cvx_solver_iters_history(1:t);
result.stop_reason      = stop_reason;
result.rank_stats       = compute_rank_stats(W_best);

end

function value = zero_if_nan(value)
value = value(isfinite(value));
if isempty(value)
    value = 0;
else
    value = sum(value(:));
end
end

function value = get_param(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end
