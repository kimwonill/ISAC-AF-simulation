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
W_best = []; W_sdp_best = []; w_best = [];
alpha_best = []; status_best = 'Not Solved';
metrics_best = []; metrics_sdp_best = [];
audit_best = []; audit_sdp_best = []; recovery_best = [];
metrics_evd_best = []; audit_evd_best = [];
best_feasible = false;
stop_reason = 'max_iter';
cvx_solver_iters = 0;
cvx_solver_iters_history = nan(params.max_iter, 1);
gr_attempted_any = false;
gr_used_any = false;
gr_trials_total = 0;
gr_solver_iters_total = 0;
limits = struct('CV_max', CV_max);

prev_sumrate = -Inf;
for t = 1:params.max_iter
    [W_sdp, ~, status, slvitr] = solve_sdp(H, alpha, A, CV_max, params);
    cvx_solver_iters_history(t) = slvitr;
    cvx_solver_iters = cvx_solver_iters + zero_if_nan(slvitr);
    if isempty(W_sdp)
        % SDP infeasible / failed -- stop early
        result = failed_result(status, alpha, params, t, ...
            cvx_solver_iters, cvx_solver_iters_history, ...
            gr_attempted_any, gr_used_any, gr_trials_total, ...
            gr_solver_iters_total);
        return;
    end

    [W_rank1, w_rank1, recovery] = principal_eigenmode_recovery(W_sdp);
    metrics_sdp = evaluate_beamforming_solution(H, alpha, W_sdp, A, params);
    metrics_rank1 = evaluate_beamforming_solution(H, alpha, W_rank1, A, params);
    audit_sdp = audit_solution_constraints(metrics_sdp, params, limits);
    audit_rank1 = audit_solution_constraints(metrics_rank1, params, limits);
    metrics_evd = metrics_rank1;
    audit_evd = audit_rank1;
    recovery.gaussian_randomization_attempted = false;
    recovery.gaussian_randomization_used = false;
    recovery.gaussian_randomization_status = 'Not needed';
    recovery.gaussian_randomization_trials = 0;
    recovery.gaussian_randomization_first_feasible_trial = NaN;
    recovery.gaussian_randomization_seed = NaN;
    recovery.gaussian_randomization_solver_iters = 0;
    gaussian_trials = get_param(params, 'gaussian_randomization_trials', 10);
    if ~audit_rank1.is_feasible && gaussian_trials > 0
        gr_attempted_any = true;
        recovery.gaussian_randomization_attempted = true;
        gaussian_seed = get_param(params, 'gaussian_randomization_seed', []);
        randomized = gaussian_randomization_recovery( ...
            H, alpha, W_sdp, A, CV_max, params, gaussian_trials, gaussian_seed);
        recovery.gaussian_randomization_trials = randomized.num_trials_solved;
        recovery.gaussian_randomization_first_feasible_trial = ...
            randomized.first_feasible_trial;
        recovery.gaussian_randomization_seed = randomized.random_seed;
        recovery.gaussian_randomization_solver_iters = randomized.total_solver_iters;
        gr_trials_total = gr_trials_total + randomized.num_trials_solved;
        gr_solver_iters_total = gr_solver_iters_total + ...
            randomized.total_solver_iters;
        recovery.gaussian_randomization_status = randomized.refinement_status;
        cvx_solver_iters = cvx_solver_iters + randomized.total_solver_iters;
        cvx_solver_iters_history(t) = ...
            zero_if_nan(cvx_solver_iters_history(t)) + ...
            randomized.total_solver_iters;
        if randomized.success
            gr_used_any = true;
            W_rank1 = randomized.W;
            w_rank1 = randomized.w;
            metrics_rank1 = randomized.metrics;
            audit_rank1 = randomized.audit;
            recovery.gaussian_randomization_used = true;
            recovery.gaussian_randomization_status = 'Feasible recovery';
        elseif strcmp(recovery.gaussian_randomization_status, 'Not run')
            recovery.gaussian_randomization_status = randomized.socp_status;
        end
    end
    recovery.initial_evd_max_violation = audit_evd.max_violation;
    recovery.final_rank1_max_violation = audit_rank1.max_violation;
    sumrate = metrics_rank1.sumrate;

    prefer_candidate = isempty(W_best) || ...
        (audit_rank1.is_feasible && ~best_feasible) || ...
        (audit_rank1.is_feasible == best_feasible && sumrate > best_sumrate);
    if prefer_candidate
        best_sumrate = sumrate;
        best_feasible = audit_rank1.is_feasible;
        W_best = W_rank1;
        W_sdp_best = W_sdp;
        w_best = w_rank1;
        alpha_best = alpha;
        status_best = status;
        metrics_best = metrics_rank1;
        metrics_sdp_best = metrics_sdp;
        audit_best = audit_rank1;
        audit_sdp_best = audit_sdp;
        recovery_best = recovery;
        metrics_evd_best = metrics_evd;
        audit_evd_best = audit_evd;
    end

    if abs(sumrate - prev_sumrate) < params.tol
        stop_reason = 'sumrate_tol';
        break;
    end
    prev_sumrate = sumrate;

    alpha_next = update_alpha(H, W_rank1, params);
    if stop_if_alpha_unchanged && isequal(alpha_next, alpha)
        stop_reason = 'alpha_fixed';
        break;
    end
    alpha = alpha_next;
end

result.W                = W_best;
result.W_sdp            = W_sdp_best;
result.w                = w_best;
result.alpha            = alpha_best;
result.rank1_sumrate    = metrics_best.sumrate;
result.sdp_sumrate      = metrics_sdp_best.sumrate;
result.pslr_per_target  = metrics_best.pslr_per_target;
result.islr_per_target  = metrics_best.islr_per_target;
result.metrics_rank1    = metrics_best;
result.metrics_sdp      = metrics_sdp_best;
result.metrics_evd_initial = metrics_evd_best;
result.constraint_audit_rank1 = audit_best;
result.constraint_audit_sdp = audit_sdp_best;
result.constraint_audit_evd_initial = audit_evd_best;
result.solver_feasible = true;
result.covariance_feasible = audit_sdp_best.is_feasible;
result.initial_evd_feasible = audit_evd_best.is_feasible;
result.final_rank1_feasible = audit_best.is_feasible;
result.post_evd_feasible = result.final_rank1_feasible;
result.solver_status    = status_best;
if audit_best.is_feasible
    result.sumrate = best_sumrate;
    result.status = status_best;
else
    result.sumrate = NaN;
    result.status = sprintf('Post-EVD Infeasible (max residual %.3e)', ...
        audit_best.max_violation);
end
result.iters            = t;
result.cvx_solver_iters = cvx_solver_iters;
result.cvx_solver_iters_history = cvx_solver_iters_history(1:t);
result.stop_reason      = stop_reason;
result.rank_stats       = recovery_best;
result.recovery         = recovery_best;
result.gr_attempted_any = gr_attempted_any;
result.gr_used_any      = gr_used_any;
result.gr_trials_total  = gr_trials_total;
result.gr_solver_iters_total = gr_solver_iters_total;
if recovery_best.gaussian_randomization_used
    result.recovery_method = 'gaussian-randomization';
else
    result.recovery_method = 'principal-eigenmode-lambda1';
end
result.result_schema_version = params.result_schema_version;

end

function result = failed_result(status, alpha, params, t, ...
    cvx_solver_iters, cvx_solver_iters_history, ...
    gr_attempted_any, gr_used_any, gr_trials_total, ...
    gr_solver_iters_total)
result.status = status;
result.solver_status = status;
result.sumrate = NaN;
result.rank1_sumrate = NaN;
result.sdp_sumrate = NaN;
result.pslr_per_target = nan(params.L, 1);
result.islr_per_target = nan(params.L, 1);
result.W = [];
result.W_sdp = [];
result.w = [];
result.alpha = alpha;
result.metrics_rank1 = [];
result.metrics_sdp = [];
result.metrics_evd_initial = [];
result.constraint_audit_rank1 = [];
result.constraint_audit_sdp = [];
result.constraint_audit_evd_initial = [];
result.solver_feasible = false;
result.covariance_feasible = false;
result.initial_evd_feasible = false;
result.final_rank1_feasible = false;
result.post_evd_feasible = false;
result.iters = t;
result.cvx_solver_iters = cvx_solver_iters;
result.cvx_solver_iters_history = cvx_solver_iters_history(1:t);
result.stop_reason = 'failed';
result.rank_stats = [];
result.recovery = [];
result.recovery_method = 'none';
result.gr_attempted_any = gr_attempted_any;
result.gr_used_any = gr_used_any;
result.gr_trials_total = gr_trials_total;
result.gr_solver_iters_total = gr_solver_iters_total;
result.result_schema_version = params.result_schema_version;
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
