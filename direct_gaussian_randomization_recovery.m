function result = direct_gaussian_randomization_recovery( ...
    H, alpha, W_sdp, A, pslr_min, params, num_trials, random_seed)
%DIRECT_GAUSSIAN_RANDOMIZATION_RECOVERY GR for the Direct-PSLR baseline.

if nargin < 7 || isempty(num_trials)
    num_trials = 10;
end
if nargin < 8 || isempty(random_seed)
    random_seed = 314159;
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

limits = struct('PSLR_min', pslr_min);
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

    [W_sensing, ~, sensing_status, sensing_info] = ...
        solve_direct_fixed_direction_powers( ...
        H, alpha, directions, A, pslr_min, params, 'sensing');
    result.total_solver_iters = result.total_solver_iters + ...
        zero_if_nan(sensing_info.solver_iters);
    result.num_trials_solved = trial;
    result.socp_status = sensing_status;
    if isempty(W_sensing)
        continue;
    end
    result.num_sensing_feasible = result.num_sensing_feasible + 1;

    [W_candidate, w_candidate, refine_status, refine_info] = ...
        solve_direct_fixed_direction_powers( ...
        H, alpha, directions, A, pslr_min, params, 'refine');
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
