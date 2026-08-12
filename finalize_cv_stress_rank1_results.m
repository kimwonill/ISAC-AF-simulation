function source_path = finalize_cv_stress_rank1_results(num_mc)
%FINALIZE_CV_STRESS_RANK1_RESULTS Validate and merge all point files.

if nargin < 1 || isempty(num_mc), num_mc = 100; end
config = cv_stress_rank1_config(num_mc);
sim_dir = fileparts(mfilename('fullpath'));
result_dir = fullfile(sim_dir, 'results');
run_dir = fullfile(result_dir, config.run_directory);
points_dir = fullfile(run_dir, 'points');
source_path = fullfile(result_dir, config.result_filename);

S = numel(config.scenarios);
C = config.num_cv;
M = config.num_mc;
completed = false(S, C, M);

prop = initialize_method_arrays(S, C, M);
direct = initialize_method_arrays(S, C, M);
pslr_min = nan(S, C, M);
channel_seeds = nan(S, M);
recovery_seeds = nan(S, C, M);
worker_counts = nan(S, C, M);
run_ids = strings(S, C, M);
saved_utc = strings(S, C, M);

for mc = 1:M
    for s = 1:S
        for c = 1:C
            point_path = stress_point_path(points_dir, s, c, mc);
            if exist(point_path, 'file') ~= 2
                error('Missing point file: %s', point_path);
            end
            saved = load(point_path, 'point');
            if ~isfield(saved, 'point') || ~isstruct(saved.point)
                error('Invalid point payload: %s', point_path);
            end
            point = saved.point;
            validate_point(point, config, s, c, mc, point_path);
            completed(s, c, mc) = true;
            prop = assign_method(prop, point.proposed, s, c, mc);
            direct = assign_method(direct, point.direct, s, c, mc);
            pslr_min(s, c, mc) = point.pslr_min;
            channel_seeds(s, mc) = point.channel_seed;
            recovery_seeds(s, c, mc) = point.recovery_seed;
            worker_counts(s, c, mc) = point.worker_count;
            run_ids(s, c, mc) = string(point.run_id);
            saved_utc(s, c, mc) = string(point.saved_utc);
        end
    end
end

validate_feasibility_order(prop, 'CV-SDP');
validate_feasibility_order(direct, 'Direct SCA');
if ~all(completed(:))
    error('Completion mask is not full after point-file validation.');
end

archive_version = config.archive_version;
experiment_id = config.experiment_id;
result_schema_version = config.result_schema_version;
CV_grid = config.CV_grid;
scenarios = config.scenarios;
time_budget_seconds = config.time_budget_seconds;
budget_policy = config.budget_policy;
timing_protocol = config.timing_protocol;
ipm_protocol = config.ipm_protocol;
gaussian_randomization_trials = config.gaussian_randomization_trials;
cvx_solver = config.cvx_solver;
cvx_solver_threads = config.cvx_solver_threads;
channel_seed_rule = config.channel_seed_rule;
recovery_seed_rule = config.recovery_seed_rule;
num_mc = config.num_mc;
execution_worker_counts = unique(worker_counts(isfinite(worker_counts)));
if any(~isfinite(worker_counts(:))) || ...
        any(worker_counts(:) < 1) || ...
        any(worker_counts(:) ~= floor(worker_counts(:))) || ...
        numel(execution_worker_counts) ~= 1
    error('Point files mix worker counts: %s', ...
        mat2str(execution_worker_counts(:).'));
end

prop_time = prop.runtime_sec;
direct_time = direct.runtime_sec;
prop_cvx_solver_iters = prop.total_ipm_iters;
direct_cvx_solver_iters = direct.total_ipm_iters;
prop_solver_feasible = prop.solver_feasible;
direct_solver_feasible = direct.solver_feasible;
prop_covariance_feasible = prop.covariance_feasible;
direct_covariance_feasible = direct.covariance_feasible;
prop_initial_evd_feasible = prop.initial_evd_feasible;
direct_initial_evd_feasible = direct.initial_evd_feasible;
prop_final_rank1_feasible = prop.final_rank1_feasible;
direct_final_rank1_feasible = direct.final_rank1_feasible;
prop_success = prop_final_rank1_feasible;
direct_success = direct_final_rank1_feasible;
prop_gr_attempted = prop.gr_attempted;
direct_gr_attempted = direct.gr_attempted;
prop_gr_used = prop.gr_used;
direct_gr_used = direct.gr_used;
prop_gr_trials = prop.gr_trials;
direct_gr_trials = direct.gr_trials;
prop_gr_solver_iters = prop.gr_solver_iters;
direct_gr_solver_iters = direct.gr_solver_iters;
prop_evd_max_violation = prop.evd_max_violation;
direct_evd_max_violation = direct.evd_max_violation;
prop_final_max_violation = prop.final_max_violation;
direct_final_max_violation = direct.final_max_violation;
prop_status = prop.status;
direct_status = direct.status;
prop_solver_status = prop.solver_status;
direct_solver_status = direct.solver_status;
prop_recovery_method = prop.recovery_method;
direct_recovery_method = direct.recovery_method;
prop_exception_identifier = prop.exception_identifier;
direct_exception_identifier = direct.exception_identifier;
prop_exception_report = prop.exception_report;
direct_exception_report = direct.exception_report;

tmp_path = [tempname(result_dir) '.mat'];
cleanup = onCleanup(@() delete_if_present(tmp_path));
save(tmp_path, 'archive_version', 'experiment_id', ...
    'result_schema_version', 'num_mc', 'CV_grid', 'scenarios', ...
    'time_budget_seconds', 'budget_policy', 'timing_protocol', ...
    'ipm_protocol', 'gaussian_randomization_trials', ...
    'cvx_solver', 'cvx_solver_threads', 'channel_seed_rule', ...
    'recovery_seed_rule', 'execution_worker_counts', 'completed', ...
    'pslr_min', 'channel_seeds', 'recovery_seeds', 'run_ids', ...
    'saved_utc', ...
    'prop_time', 'direct_time', ...
    'prop_cvx_solver_iters', 'direct_cvx_solver_iters', ...
    'prop_solver_feasible', 'direct_solver_feasible', ...
    'prop_covariance_feasible', 'direct_covariance_feasible', ...
    'prop_initial_evd_feasible', 'direct_initial_evd_feasible', ...
    'prop_final_rank1_feasible', 'direct_final_rank1_feasible', ...
    'prop_success', 'direct_success', ...
    'prop_gr_attempted', 'direct_gr_attempted', ...
    'prop_gr_used', 'direct_gr_used', ...
    'prop_gr_trials', 'direct_gr_trials', ...
    'prop_gr_solver_iters', 'direct_gr_solver_iters', ...
    'prop_evd_max_violation', 'direct_evd_max_violation', ...
    'prop_final_max_violation', 'direct_final_max_violation', ...
    'prop_status', 'direct_status', ...
    'prop_solver_status', 'direct_solver_status', ...
    'prop_recovery_method', 'direct_recovery_method', ...
    'prop_exception_identifier', 'direct_exception_identifier', ...
    'prop_exception_report', 'direct_exception_report', '-v7.3');
[ok, message] = movefile(tmp_path, source_path, 'f');
if ~ok
    error('Failed to finalize %s: %s', source_path, message);
end
clear cleanup;

print_summary(source_path);
plot_cv_stress_rank1_results(source_path);
fprintf('Saved canonical stress result: %s\n', source_path);
end

function arrays = initialize_method_arrays(S, C, M)
arrays.runtime_sec = nan(S, C, M);
arrays.total_ipm_iters = nan(S, C, M);
arrays.solver_feasible = false(S, C, M);
arrays.covariance_feasible = false(S, C, M);
arrays.initial_evd_feasible = false(S, C, M);
arrays.final_rank1_feasible = false(S, C, M);
arrays.gr_attempted = false(S, C, M);
arrays.gr_used = false(S, C, M);
arrays.gr_trials = zeros(S, C, M);
arrays.gr_solver_iters = zeros(S, C, M);
arrays.evd_max_violation = nan(S, C, M);
arrays.final_max_violation = nan(S, C, M);
arrays.status = strings(S, C, M);
arrays.solver_status = strings(S, C, M);
arrays.recovery_method = strings(S, C, M);
arrays.exception_identifier = strings(S, C, M);
arrays.exception_report = strings(S, C, M);
end

function arrays = assign_method(arrays, record, s, c, mc)
numeric_fields = {'runtime_sec', 'total_ipm_iters', 'gr_trials', ...
    'gr_solver_iters', 'evd_max_violation', 'final_max_violation'};
logical_fields = {'solver_feasible', 'covariance_feasible', ...
    'initial_evd_feasible', 'final_rank1_feasible', ...
    'gr_attempted', 'gr_used'};
string_fields = {'status', 'solver_status', 'recovery_method', ...
    'exception_identifier', 'exception_report'};
for i = 1:numel(numeric_fields)
    name = numeric_fields{i};
    arrays.(name)(s, c, mc) = double(record.(name));
end
for i = 1:numel(logical_fields)
    name = logical_fields{i};
    arrays.(name)(s, c, mc) = logical(record.(name));
end
for i = 1:numel(string_fields)
    name = string_fields{i};
    arrays.(name)(s, c, mc) = string(record.(name));
end
end

function validate_point(point, config, s, c, mc, point_path)
expected = struct( ...
    'archive_version', config.archive_version, ...
    'experiment_id', config.experiment_id, ...
    'result_schema_version', config.result_schema_version, ...
    'num_mc', config.num_mc, ...
    'scenario_index', s, ...
    'scenario_id', config.scenarios(s).id, ...
    'cv_index', c, ...
    'CV_max', config.CV_grid(c), ...
    'mc_index', mc, ...
    'channel_seed', 2000 * s + mc, ...
    'recovery_seed', 900000 + 10000 * s + 100 * mc + c, ...
    'gaussian_randomization_trials', ...
        config.gaussian_randomization_trials, ...
    'cvx_solver', config.cvx_solver, ...
    'cvx_solver_threads', config.cvx_solver_threads);
fields = fieldnames(expected);
for i = 1:numel(fields)
    name = fields{i};
    if ~isfield(point, name) || ~isequaln(point.(name), expected.(name))
        error('Metadata mismatch for %s in %s.', name, point_path);
    end
end
required = {'completed', 'pslr_min', 'worker_count', 'run_id', ...
    'proposed', 'direct', 'saved_utc'};
for i = 1:numel(required)
    if ~isfield(point, required{i})
        error('Missing field %s in %s.', required{i}, point_path);
    end
end
if ~isequal(point.completed, true)
    error('Incomplete point: %s', point_path);
end
template = fieldnames(initialize_method_record());
for i = 1:numel(template)
    name = template{i};
    if ~isfield(point.proposed, name) || ~isfield(point.direct, name)
        error('Missing method field %s in %s.', name, point_path);
    end
end
validate_method_record(point.proposed, 'proposed', point_path);
validate_method_record(point.direct, 'direct', point_path);
params = scenario_params_for_validation(config.scenarios(s), config);
expected_pslr = direct_thresholds_from_cv(config.CV_grid(c), params);
if ~isscalar(point.pslr_min) || ~isfinite(point.pslr_min) || ...
        abs(point.pslr_min - expected_pslr) > ...
        1e-12 * max(1, abs(expected_pslr))
    error('PSLR threshold mismatch in %s.', point_path);
end
if ~isscalar(point.worker_count) || ~isfinite(point.worker_count) || ...
        point.worker_count < 1 || point.worker_count ~= floor(point.worker_count)
    error('Invalid worker_count in %s.', point_path);
end
end

function record = initialize_method_record()
record = struct( ...
    'runtime_sec', NaN, 'total_ipm_iters', NaN, ...
    'solver_feasible', false, 'covariance_feasible', false, ...
    'initial_evd_feasible', false, 'final_rank1_feasible', false, ...
    'gr_attempted', false, 'gr_used', false, ...
    'gr_trials', 0, 'gr_solver_iters', 0, ...
    'evd_max_violation', NaN, 'final_max_violation', NaN, ...
    'status', '', 'solver_status', '', 'recovery_method', '', ...
    'exception_identifier', '', 'exception_report', '');
end

function validate_method_record(record, method_name, point_path)
finite_nonnegative = {'runtime_sec', 'total_ipm_iters', 'gr_trials', ...
    'gr_solver_iters'};
for i = 1:numel(finite_nonnegative)
    name = finite_nonnegative{i};
    value = record.(name);
    if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || value < 0
        error('Invalid %s.%s in %s.', method_name, name, point_path);
    end
end
integer_fields = {'total_ipm_iters', 'gr_trials', 'gr_solver_iters'};
for i = 1:numel(integer_fields)
    name = integer_fields{i};
    if abs(record.(name) - round(record.(name))) > 1e-9
        error('Noninteger %s.%s in %s.', method_name, name, point_path);
    end
end
logical_fields = {'solver_feasible', 'covariance_feasible', ...
    'initial_evd_feasible', 'final_rank1_feasible', ...
    'gr_attempted', 'gr_used'};
for i = 1:numel(logical_fields)
    name = logical_fields{i};
    value = record.(name);
    if ~islogical(value) || ~isscalar(value)
        error('Invalid %s.%s in %s.', method_name, name, point_path);
    end
end
if ~isempty(record.exception_identifier) || ~isempty(record.exception_report)
    error('Exception record found for %s in %s.', method_name, point_path);
end
if record.solver_feasible && record.total_ipm_iters <= 0
    error('Missing IPM count for solved %s point in %s.', ...
        method_name, point_path);
end
end

function params = scenario_params_for_validation(scenario, config)
params = setup_params();
params.NT = scenario.NT;
params.N = scenario.N;
params.L = scenario.L;
params.theta = scenario.theta;
params.Q = scenario.Q * ones(params.K, 1);
params.P_des = scenario.Pdes_scale * params.P_max / params.N;
params.gaussian_randomization_trials = ...
    config.gaussian_randomization_trials;
end

function validate_feasibility_order(arrays, method_name)
if any(arrays.initial_evd_feasible(:) & ~arrays.final_rank1_feasible(:))
    error('%s has a point feasible after EVD but infeasible after GR.', ...
        method_name);
end
end

function print_summary(source_path)
S = load(source_path, 'CV_grid', 'scenarios', 'time_budget_seconds', ...
    'prop_time', 'direct_time', 'prop_cvx_solver_iters', ...
    'direct_cvx_solver_iters', 'prop_initial_evd_feasible', ...
    'direct_initial_evd_feasible', 'prop_final_rank1_feasible', ...
    'direct_final_rank1_feasible', 'prop_exception_identifier', ...
    'direct_exception_identifier');
fprintf('Stress rank-one summary over CV=0.1:0.1:1.0\n');
for s = 1:numel(S.scenarios)
    prop_budget = S.prop_final_rank1_feasible(s, :, :) & ...
        S.prop_time(s, :, :) <= S.time_budget_seconds(s);
    direct_budget = S.direct_final_rank1_feasible(s, :, :) & ...
        S.direct_time(s, :, :) <= S.time_budget_seconds(s);
    fprintf(['  %s: budget %.1f%%/%.1f%%, EVD %.1f%%/%.1f%%, ' ...
        'EVD+GR %.1f%%/%.1f%%\n'], S.scenarios(s).id, ...
        100 * mean(prop_budget(:)), 100 * mean(direct_budget(:)), ...
        100 * mean(reshape(S.prop_initial_evd_feasible(s, :, :), [], 1)), ...
        100 * mean(reshape(S.direct_initial_evd_feasible(s, :, :), [], 1)), ...
        100 * mean(reshape(S.prop_final_rank1_feasible(s, :, :), [], 1)), ...
        100 * mean(reshape(S.direct_final_rank1_feasible(s, :, :), [], 1)));
end
fprintf('  Exceptions: CV=%d, Direct=%d\n', ...
    nnz(strlength(S.prop_exception_identifier) > 0), ...
    nnz(strlength(S.direct_exception_identifier) > 0));
end

function path = stress_point_path(points_dir, s, c, mc)
path = fullfile(points_dir, sprintf( ...
    'point_s%02d_c%02d_mc%03d.mat', s, c, mc));
end

function delete_if_present(path)
if exist(path, 'file') == 2
    delete(path);
end
end
