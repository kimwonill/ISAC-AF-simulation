function result = run_proposed_covariance(H, CV_max, params, alpha0)
%RUN_PROPOSED_COVARIANCE Reproduce the covariance-level proposed AO loop.
%   This is the pre-rank-one-recovery trajectory used for the original
%   Pareto data: allocation updates use the SDP covariance itself.  It is
%   intentionally separate from RUN_PROPOSED, whose current implementation
%   performs rank-one recovery inside the AO loop.

A = compute_steering(params);
if nargin >= 4 && ~isempty(alpha0) && ...
        isequal(size(alpha0), [params.K, params.N])
    alpha = alpha0;
else
    alpha = init_alpha(H, params);
end

stop_if_alpha_unchanged = get_param(params, 'stop_if_alpha_unchanged', true);
best_sumrate = -Inf;
W_best = [];
alpha_best = [];
status_best = 'Not Solved';
cvx_solver_iters = 0;
cvx_solver_iters_history = nan(params.max_iter, 1);
stop_reason = 'max_iter';
prev_sumrate = -Inf;

for t = 1:params.max_iter
    [W_sdp, sumrate, status, slvitr] = solve_sdp( ...
        H, alpha, A, CV_max, params);
    cvx_solver_iters_history(t) = slvitr;
    cvx_solver_iters = cvx_solver_iters + zero_if_nan(slvitr);
    if isempty(W_sdp)
        result = failed_result(status, alpha, t, cvx_solver_iters, ...
            cvx_solver_iters_history);
        return;
    end

    if sumrate > best_sumrate
        best_sumrate = sumrate;
        W_best = W_sdp;
        alpha_best = alpha;
        status_best = status;
    end

    if abs(sumrate - prev_sumrate) < params.tol
        stop_reason = 'sumrate_tol';
        break;
    end
    prev_sumrate = sumrate;

    alpha_next = update_alpha(H, W_sdp, params);
    if stop_if_alpha_unchanged && isequal(alpha_next, alpha)
        stop_reason = 'alpha_fixed';
        break;
    end
    alpha = alpha_next;
end

result.solver_feasible = true;
result.status = status_best;
result.W_sdp = W_best;
result.alpha = alpha_best;
result.sumrate = best_sumrate;
result.iters = t;
result.cvx_solver_iters = cvx_solver_iters;
result.cvx_solver_iters_history = cvx_solver_iters_history(1:t);
result.stop_reason = stop_reason;
end

function result = failed_result(status, alpha, t, cvx_solver_iters, history)
result.solver_feasible = false;
result.status = status;
result.W_sdp = [];
result.alpha = alpha;
result.sumrate = NaN;
result.iters = t;
result.cvx_solver_iters = cvx_solver_iters;
result.cvx_solver_iters_history = history(1:t);
result.stop_reason = 'failed';
end

function value = get_param(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end

function value = zero_if_nan(value)
value = value(isfinite(value));
if isempty(value)
    value = 0;
else
    value = sum(value(:));
end
end
