function result = run_direct_sca_covariance(H, pslr_min, params, alpha0, W0)
%RUN_DIRECT_SCA_COVARIANCE Direct-SCA AO with covariance-level updates.
%   This is the pre-rank-recovery Direct-SCA path used to create the
%   original MC=100 Pareto cache. It intentionally returns the selected
%   covariance solution without EVD or Gaussian recovery.

A = compute_steering(params);

if nargin >= 4 && ~isempty(alpha0) && ...
        isequal(size(alpha0), [params.K, params.N])
    alpha = alpha0;
else
    alpha = init_alpha(H, params);
end

if nargin >= 5 && ~isempty(W0)
    W_ref = W0;
else
    W_ref = init_covariance_flat(params);
end

outer_max = get_param(params, 'direct_ao_max_iter', params.max_iter);
inner_max = get_param(params, 'direct_sca_max_iter', 5);
sca_tol = get_param(params, 'direct_sca_tol', 1e-3);
stop_if_alpha_unchanged = get_param(params, ...
    'stop_if_alpha_unchanged', true);

best_sumrate = -Inf;
W_best = [];
alpha_best = [];
status_best = 'Not Solved';
stop_reason = 'max_iter';
prev_sumrate = -Inf;
inner_used = 0;

for t = 1:outer_max
    P_ref = directional_power_grid(W_ref, A);

    for m = 1:inner_max
        [W, sumrate, status] = solve_direct_sca_sdp( ...
            H, alpha, A, pslr_min, P_ref, params);
        inner_used = inner_used + 1;
        if isempty(W)
            result = failed_result(status, alpha, t, inner_used);
            return;
        end

        P_new = directional_power_grid(W, A);
        rel_change = norm(P_new(:) - P_ref(:), 2) / ...
            max(1, norm(P_ref(:), 2));
        P_ref = P_new;
        W_ref = W;

        if rel_change < sca_tol
            break;
        end
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
    if stop_if_alpha_unchanged && isequal(alpha_next, alpha)
        stop_reason = 'alpha_fixed';
        break;
    end
    alpha = alpha_next;
end

result.W = W_best;
result.alpha = alpha_best;
result.sumrate = best_sumrate;
result.status = status_best;
result.solver_feasible = ~isempty(W_best) && isfinite(best_sumrate);
result.iters = t;
result.inner_iters = inner_used;
result.stop_reason = stop_reason;
end

function result = failed_result(status, alpha, t, inner_used)
result.W = [];
result.alpha = alpha;
result.sumrate = NaN;
result.status = status;
result.solver_feasible = false;
result.iters = t;
result.inner_iters = inner_used;
result.stop_reason = 'failed';
end

function value = get_param(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end
