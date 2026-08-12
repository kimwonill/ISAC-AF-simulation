function summary = run_proposed_covariance_recovery_audit(num_trials, force_rerun)
%RUN_PROPOSED_COVARIANCE_RECOVERY_AUDIT Audit Proposed rank-one recovery.
%   Replays the original Pareto channel realization for every configuration
%   and CV point, using the original covariance-level AO trajectory.  The
%   recovery policy is principal EVD followed directly by at most
%   NUM_TRIALS Gaussian-randomization trials; no separate power-repair
%   stage is applied before GR.

if nargin < 1 || isempty(num_trials)
    num_trials = 10;
end
if nargin < 2
    force_rerun = false;
end
assert(isscalar(num_trials) && num_trials >= 0 && num_trials == floor(num_trials), ...
    'num_trials must be a nonnegative integer.');

sim_dir = fileparts(mfilename('fullpath'));
result_dir = fullfile(sim_dir, 'results');
output_path = fullfile(result_dir, sprintf( ...
    'proposed_rank_one_recovery_audit_R%d_MC100.mat', num_trials));
if exist(output_path, 'file') == 2 && ~force_rerun
    cached = load(output_path, 'summary');
    if isfield(cached, 'summary') && isfield(cached.summary, 'audit_version') && ...
            cached.summary.audit_version >= 1
        summary = cached.summary;
        print_summary(summary, output_path);
        return;
    end
end

configurations = struct('NT', {4, 4, 8, 8}, 'N', {16, 32, 16, 32});
num_configs = numel(configurations);
source_paths = cell(num_configs, 1);
sources = cell(num_configs, 1);
for q = 1:num_configs
    config = configurations(q);
    source_paths{q} = fullfile(result_dir, 'pareto_grid_1x4_pslr', sprintf( ...
        'pareto_pslr_NT%d_N%d_MC100.mat', config.NT, config.N));
    assert(exist(source_paths{q}, 'file') == 2, ...
        'Missing Pareto source: %s', source_paths{q});
    sources{q} = load(source_paths{q}, 'params', 'CV_max_list');
end

cv_grid = sources{1}.CV_max_list(:).';
num_cv = numel(cv_grid);
num_mc = 100;
for q = 2:num_configs
    assert(isequal(cv_grid, sources{q}.CV_max_list(:).'), ...
        'The Pareto CV grids must agree across configurations.');
end
grid_size = [num_cv, num_mc, num_configs];

raw.solver_feasible = false(grid_size);
raw.covariance_audit_feasible = false(grid_size);
raw.initial_evd_feasible = false(grid_size);
raw.gaussian_randomization_used = false(grid_size);
raw.final_feasible = false(grid_size);
raw.initial_evd_max_residual = nan(grid_size);
raw.final_max_residual = nan(grid_size);
raw.first_feasible_trial = nan(grid_size);
raw.randomization_seed = nan(grid_size);
raw.recovery_method = strings(grid_size);

for q = 1:num_configs
    config = configurations(q);
    params = sources{q}.params;
    params.cvx_solver = 'mosek';
    params.cvx_solver_threads = 1;
    params.sdp_quiet = true;
    params.collect_cvx_solver_log = false;
    params.post_evd_feas_tol = 1e-4;
    A = compute_steering(params);
    fprintf('\nProposed recovery audit: config %d/%d, NT=%d, N=%d\n', ...
        q, num_configs, config.NT, config.N);

    for mc = 1:num_mc
        % This is the channel seed used in the original Pareto experiment.
        rng(mc, 'twister');
        H = generate_channel(params);
        alpha_warm = [];
        for cv_index = 1:num_cv
            CV_max = cv_grid(cv_index);
            covariance_result = run_proposed_covariance( ...
                H, CV_max, params, alpha_warm);
            if ~covariance_result.solver_feasible
                raw.recovery_method(cv_index, mc, q) = "solver-failed";
                alpha_warm = [];
                continue;
            end
            raw.solver_feasible(cv_index, mc, q) = true;
            alpha_warm = covariance_result.alpha;
            W_sdp = covariance_result.W_sdp;
            alpha = covariance_result.alpha;

            limits = struct('CV_max', CV_max);
            metrics_sdp = evaluate_beamforming_solution( ...
                H, alpha, W_sdp, A, params);
            audit_sdp = audit_solution_constraints(metrics_sdp, params, limits);
            raw.covariance_audit_feasible(cv_index, mc, q) = audit_sdp.is_feasible;

            [W_rank1, ~] = principal_eigenmode_recovery(W_sdp);
            metrics_rank1 = evaluate_beamforming_solution( ...
                H, alpha, W_rank1, A, params);
            audit_rank1 = audit_solution_constraints( ...
                metrics_rank1, params, limits);
            raw.initial_evd_feasible(cv_index, mc, q) = audit_rank1.is_feasible;
            raw.initial_evd_max_residual(cv_index, mc, q) = audit_rank1.max_violation;
            recovery_method = "principal-eigenmode";

            if ~audit_rank1.is_feasible && num_trials > 0
                % Preserve the original Pareto recovery seed convention.
                random_seed = 900000 + 100 * mc + cv_index;
                raw.randomization_seed(cv_index, mc, q) = random_seed;
                randomized = gaussian_randomization_recovery( ...
                    H, alpha, W_sdp, A, CV_max, params, ...
                    num_trials, random_seed);
                raw.first_feasible_trial(cv_index, mc, q) = ...
                    randomized.first_feasible_trial;
                if randomized.success
                    audit_rank1 = randomized.audit;
                    raw.gaussian_randomization_used(cv_index, mc, q) = true;
                    recovery_method = "gaussian-randomization";
                end
            end

            raw.final_feasible(cv_index, mc, q) = audit_rank1.is_feasible;
            raw.final_max_residual(cv_index, mc, q) = audit_rank1.max_violation;
            if audit_rank1.is_feasible
                raw.recovery_method(cv_index, mc, q) = recovery_method;
            else
                raw.recovery_method(cv_index, mc, q) = "failed";
            end
        end
        if mod(mc, 10) == 0 || mc == num_mc
            fprintf('  MC %d/%d: EVD %d, GR +%d, final %d/%d\n', ...
                mc, num_mc, nnz(raw.initial_evd_feasible(:, 1:mc, q)), ...
                nnz(raw.gaussian_randomization_used(:, 1:mc, q)), ...
                nnz(raw.final_feasible(:, 1:mc, q)), num_cv * mc);
        end
    end
end

summary.audit_version = 1;
summary.recovery_policy = 'principal EVD, then GR only';
summary.num_trials = num_trials;
summary.num_points = numel(raw.solver_feasible);
summary.configurations = configurations;
summary.CV_max_list = cv_grid;
summary.channel_seed_rule = 'rng(mc, ''twister'')';
summary.gr_seed_rule = '900000 + 100*mc + cv_index';
summary.solver_feasible_by_config = squeeze(sum(sum(raw.solver_feasible, 1), 2));
summary.covariance_audit_feasible_by_config = ...
    squeeze(sum(sum(raw.covariance_audit_feasible, 1), 2));
summary.initial_evd_feasible_by_config = ...
    squeeze(sum(sum(raw.initial_evd_feasible, 1), 2));
summary.gaussian_randomization_used_by_config = ...
    squeeze(sum(sum(raw.gaussian_randomization_used, 1), 2));
summary.final_feasible_by_config = ...
    squeeze(sum(sum(raw.final_feasible, 1), 2));
summary.total_final_feasible = nnz(raw.final_feasible);

save(output_path, 'summary', 'raw', 'source_paths', '-v7.3');
print_summary(summary, output_path);
end

function print_summary(summary, output_path)
fprintf('\nProposed rank-one recovery audit (R_G=%d):\n', summary.num_trials);
for q = 1:numel(summary.configurations)
    config = summary.configurations(q);
    fprintf(['  NT=%d, N=%d: solver %d/1000, covariance audit %d/1000, ' ...
        'EVD %d/1000, GR +%d, final %d/1000\n'], ...
        config.NT, config.N, summary.solver_feasible_by_config(q), ...
        summary.covariance_audit_feasible_by_config(q), ...
        summary.initial_evd_feasible_by_config(q), ...
        summary.gaussian_randomization_used_by_config(q), ...
        summary.final_feasible_by_config(q));
end
fprintf('  Total final feasibility: %d/%d (%.2f%%)\n', ...
    summary.total_final_feasible, summary.num_points, ...
    100 * summary.total_final_feasible / summary.num_points);
fprintf('  Saved: %s\n\n', output_path);
end
