function result = run_direct_sca(H, pslr_min, islr_max, params, alpha0, W0)
% RUN_DIRECT_SCA  Algorithm 2 baseline with direct PSLR/ISLR constraints.
%
% Alternates between an inner SCA beamforming loop and the same allocation
% update used by the proposed scheme.

A = compute_steering(params);

if nargin >= 5 && ~isempty(alpha0) && isequal(size(alpha0), [params.K, params.N])
    alpha = alpha0;
else
    alpha = init_alpha(H, params);
end

if nargin >= 6 && ~isempty(W0)
    W_ref = W0;
else
    W_ref = init_covariance_flat(params);
end

outer_max = get_param(params, 'direct_ao_max_iter', params.max_iter);
inner_max = get_param(params, 'direct_sca_max_iter', 5);
sca_tol = get_param(params, 'direct_sca_tol', 1e-3);
stop_if_alpha_unchanged = get_param(params, 'stop_if_alpha_unchanged', true);

best_sumrate = -Inf;
W_best = []; alpha_best = []; status_best = 'Not Solved';
stop_reason = 'max_iter';
prev_sumrate = -Inf;
inner_used = 0;
cvx_solver_iters = 0;
cvx_solver_iters_history = nan(outer_max * inner_max, 1);

for t = 1:outer_max
    P_ref = directional_power_grid(W_ref, A);

    for m = 1:inner_max
        [W, sumrate, status, slvitr] = solve_direct_sca_sdp(H, alpha, A, pslr_min, islr_max, P_ref, params);
        inner_used = inner_used + 1;
        cvx_solver_iters_history(inner_used) = slvitr;
        cvx_solver_iters = cvx_solver_iters + zero_if_nan(slvitr);
        if isempty(W)
            result = failed_result(status, alpha, params, t, inner_used, ...
                cvx_solver_iters, cvx_solver_iters_history);
            return;
        end

        P_new = directional_power_grid(W, A);
        rel_change = norm(P_new(:) - P_ref(:), 2) / max(1, norm(P_ref(:), 2));
        P_ref = P_new;
        W_ref = W;

        if rel_change < sca_tol
            break;
        end
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
result.inner_iters      = inner_used;
result.cvx_solver_iters = cvx_solver_iters;
result.cvx_solver_iters_history = cvx_solver_iters_history(1:inner_used);
result.stop_reason      = stop_reason;
result.rank_stats       = compute_rank_stats(W_best);

end

function result = failed_result(status, alpha, params, t, inner_used, ...
    cvx_solver_iters, cvx_solver_iters_history)
result.status = status;
result.sumrate = NaN;
result.pslr_per_target = nan(params.L, 1);
result.islr_per_target = nan(params.L, 1);
result.W = [];
result.alpha = alpha;
result.iters = t;
result.inner_iters = inner_used;
result.cvx_solver_iters = cvx_solver_iters;
result.cvx_solver_iters_history = cvx_solver_iters_history(1:inner_used);
result.stop_reason = 'failed';
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
