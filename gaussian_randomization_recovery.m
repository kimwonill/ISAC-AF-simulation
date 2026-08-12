function result = gaussian_randomization_recovery( ...
    H, alpha, W_sdp, A, CV_max, params, num_trials, random_seed)
%GAUSSIAN_RANDOMIZATION_RECOVERY Search for a feasible rank-one solution.
%   Each trial draws v_n ~ CN(0,W_sdp,n), retains its direction, and solves
%   a convex power-feasibility problem. The first audited-feasible draw is
%   refined by the exact logarithmic sum-rate power optimization.

if nargin < 7 || isempty(num_trials)
    num_trials = 500;
end
if nargin < 8 || isempty(random_seed)
    random_seed = 271828;
    if isfinite(CV_max)
        random_seed = random_seed + round(1000 * CV_max);
    end
end
random_stream = RandStream('mt19937ar', 'Seed', random_seed);

NT = size(W_sdp, 1);
N = size(W_sdp, 3);
sqrt_covariance = zeros(NT, NT, N);
for n = 1:N
    Wn = (W_sdp(:, :, n) + W_sdp(:, :, n)') / 2;
    [U, D] = eig(Wn);
    eigenvalues = max(real(diag(D)), 0);
    sqrt_covariance(:, :, n) = U * diag(sqrt(eigenvalues));
end

limits = struct('CV_max', CV_max);
result.success = false;
result.first_feasible_trial = NaN;
result.num_trials_requested = num_trials;
result.num_trials_solved = 0;
result.num_sensing_feasible = 0;
result.W = [];
result.w = [];
result.metrics = [];
result.audit = [];
result.socp_status = 'Not run';
result.refinement_status = 'Not run';
result.random_seed = random_seed;
result.total_solver_iters = 0;

for trial = 1:num_trials
    directions = zeros(NT, N);
    for n = 1:N
        z = (randn(random_stream, NT, 1) + ...
            1j * randn(random_stream, NT, 1)) / sqrt(2);
        directions(:, n) = sqrt_covariance(:, :, n) * z;
    end

    [W_sensing, ~, status, sensing_info] = ...
        fixed_direction_rank_one_feasibility( ...
        H, alpha, directions, A, CV_max, params, false);
    result.total_solver_iters = result.total_solver_iters + ...
        zero_if_nan(sensing_info.solver_iters);
    result.num_trials_solved = trial;
    result.socp_status = status;
    if isempty(W_sensing)
        continue;
    end
    result.num_sensing_feasible = result.num_sensing_feasible + 1;

    % The sensing-only SOCP is a geometric prefilter. Apply the original
    % logarithmic QoS constraints and maximize sum-rate on the retained
    % directions; a conservative linear QoS prefilter would reject valid
    % candidates in these tight-CV cases.
    [W_candidate, w_candidate, refine_status, refine_info] = ...
        repair_rank_one_powers( ...
        H, alpha, directions, A, CV_max, params);
    result.total_solver_iters = result.total_solver_iters + ...
        zero_if_nan(refine_info.solver_iters);
    result.refinement_status = refine_status;
    if isempty(W_candidate)
        continue;
    end
    metrics = evaluate_beamforming_solution( ...
        H, alpha, W_candidate, A, params);
    audit = audit_solution_constraints(metrics, params, limits);
    if ~audit.is_feasible
        continue;
    end

    result.success = true;
    result.first_feasible_trial = trial;
    result.W = W_candidate;
    result.w = w_candidate;
    result.metrics = metrics;
    result.audit = audit;
    return;
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
