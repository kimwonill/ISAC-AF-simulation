function run_cv_stress_rank1_experiment(num_mc, force_rerun, mc_indices)
%RUN_CV_STRESS_RANK1_EXPERIMENT Compute resumable stress-test point files.
%   Monte Carlo indices may be split across independent MATLAB processes.
%   Every point records absolute runtime, total IPM iterations, initial-EVD
%   feasibility, and final feasibility after up to R_G Gaussian trials.

if nargin < 1 || isempty(num_mc), num_mc = 100; end
if nargin < 2 || isempty(force_rerun), force_rerun = false; end
if nargin < 3 || isempty(mc_indices), mc_indices = 1:num_mc; end

config = cv_stress_rank1_config(num_mc);
mc_indices = validate_mc_indices(mc_indices, num_mc);
sim_dir = fileparts(mfilename('fullpath'));
run_dir = fullfile(sim_dir, 'results', config.run_directory);
points_dir = fullfile(run_dir, 'points');
if exist(points_dir, 'dir') ~= 7, mkdir(points_dir); end

worker_count = str2double(getenv('STRESS_WORKERS'));
if ~isfinite(worker_count) || worker_count < 1 || ...
        worker_count ~= floor(worker_count)
    error('STRESS_WORKERS must be a finite positive integer.');
end
run_id = getenv('STRESS_RUN_ID');
if isempty(run_id), run_id = 'interactive'; end
required_run_id = '';
if strcmp(getenv('STRESS_REQUIRE_RUN_ID'), '1')
    required_run_id = run_id;
end

fprintf('Stress rank-one experiment: MC %d:%d of %d, R_G=%d\n', ...
    mc_indices(1), mc_indices(end), num_mc, ...
    config.gaussian_randomization_trials);
t_worker = tic;

for mc = mc_indices
    for s = 1:numel(config.scenarios)
        scenario = config.scenarios(s);
        params = scenario_params(scenario, config);
        channel_seed = 2000 * s + mc;
        rng(channel_seed, 'twister');
        H = generate_channel(params);
        alpha0 = init_alpha_qos_safe(H, params);
        W0 = init_covariance_flat(params);

        for c = 1:config.num_cv
            CV_max = config.CV_grid(c);
            recovery_seed = 900000 + 10000 * s + 100 * mc + c;
            params.gaussian_randomization_seed = recovery_seed;
            pslr_min = direct_thresholds_from_cv(CV_max, params);
            point_path = stress_point_path(points_dir, s, c, mc);
            expected = expected_metadata(config, s, c, mc, ...
                channel_seed, recovery_seed, worker_count, pslr_min);
            if ~force_rerun && ...
                    valid_point_file(point_path, expected, required_run_id)
                fprintf('Reuse S%d CV=%.1f MC=%d\n', s, CV_max, mc);
                continue;
            end

            fprintf('Run S%d CV=%.1f MC=%d ... ', s, CV_max, mc);
            proposed = run_method(H, ...
                @run_proposed, {CV_max, params, alpha0});
            direct = run_method(H, ...
                @run_direct_sca, {pslr_min, params, alpha0, W0});

            point = expected;
            point.completed = true;
            point.run_id = run_id;
            point.proposed = proposed;
            point.direct = direct;
            point.saved_utc = utc_timestamp();
            atomic_save_point(point_path, point);
            fprintf('CV %.2fs/%d IPM, Direct %.2fs/%d IPM\n', ...
                proposed.runtime_sec, round(proposed.total_ipm_iters), ...
                direct.runtime_sec, round(direct.total_ipm_iters));
        end
    end
end

fprintf('Completed MC %d:%d in %.1f min.\n', ...
    mc_indices(1), mc_indices(end), toc(t_worker) / 60);
end

function params = scenario_params(scenario, config)
params = setup_params();
params.NT = scenario.NT;
params.N = scenario.N;
params.L = scenario.L;
params.theta = scenario.theta;
params.Q = scenario.Q * ones(params.K, 1);
params.P_des = scenario.Pdes_scale * params.P_max / params.N;
params.num_mc = 1;
params.warm_start_cv = false;
params.stop_if_alpha_unchanged = true;
params.sdp_quiet = true;
params.collect_cvx_solver_log = true;
params.cvx_solver = config.cvx_solver;
params.cvx_solver_threads = config.cvx_solver_threads;
params.max_iter = 5;
params.direct_ao_max_iter = 5;
params.direct_sca_max_iter = 5;
params.direct_sca_tol = 1e-3;
params.gaussian_randomization_trials = ...
    config.gaussian_randomization_trials;
end

function record = run_method(H, method_function, method_args)
record = empty_method_record();
params = method_args{2};
timed_params = params;
timed_params.collect_cvx_solver_log = false;
timed_params.sdp_quiet = true;
profile_params = params;
profile_params.collect_cvx_solver_log = true;
profile_params.sdp_quiet = true;
timed_args = method_args;
profile_args = method_args;
timed_args{2} = timed_params;
profile_args{2} = profile_params;
t_run = tic;
try
    result = method_function(H, timed_args{:});
    record.runtime_sec = toc(t_run);
catch ME
    record.runtime_sec = toc(t_run);
    error('StressPoint:TimedMethodException', ...
        'Timed method failed (%s): %s', ME.identifier, ME.message);
end

try
    profile = method_function(H, profile_args{:});
catch ME
    error('StressPoint:ProfileMethodException', ...
        'IPM profiling replay failed (%s): %s', ME.identifier, ME.message);
end
assert_replay_match(result, profile);

record.status = char(string(get_field(result, 'status', 'Unknown')));
record.solver_status = char(string(get_field( ...
    result, 'solver_status', record.status)));
record.solver_feasible = logical(get_field( ...
    result, 'solver_feasible', false));
record.covariance_feasible = logical(get_field( ...
    result, 'covariance_feasible', false));
record.initial_evd_feasible = logical(get_field( ...
    result, 'initial_evd_feasible', false));
record.final_rank1_feasible = logical(get_field( ...
    result, 'final_rank1_feasible', false));
record.total_ipm_iters = double(get_field( ...
    profile, 'cvx_solver_iters', NaN));
record.gr_attempted = logical(get_field( ...
    result, 'gr_attempted_any', false));
record.gr_trials = double(get_field(result, 'gr_trials_total', 0));
record.gr_solver_iters = double(get_field( ...
    profile, 'gr_solver_iters_total', 0));
record.recovery_method = char(string(get_field( ...
    result, 'recovery_method', 'none')));
record.gr_used = strcmp(record.recovery_method, ...
    'gaussian-randomization');
record.evd_max_violation = audit_value(result, ...
    'constraint_audit_evd_initial');
record.final_max_violation = audit_value(result, ...
    'constraint_audit_rank1');
end

function record = empty_method_record()
record = struct( ...
    'runtime_sec', NaN, ...
    'total_ipm_iters', NaN, ...
    'solver_feasible', false, ...
    'covariance_feasible', false, ...
    'initial_evd_feasible', false, ...
    'final_rank1_feasible', false, ...
    'gr_attempted', false, ...
    'gr_used', false, ...
    'gr_trials', 0, ...
    'gr_solver_iters', 0, ...
    'evd_max_violation', NaN, ...
    'final_max_violation', NaN, ...
    'status', 'Not run', ...
    'solver_status', 'Not run', ...
    'recovery_method', 'none', ...
    'exception_identifier', '', ...
    'exception_report', '');
end

function value = audit_value(result, field_name)
value = NaN;
if isfield(result, field_name)
    audit = result.(field_name);
    if isstruct(audit) && isfield(audit, 'max_violation')
        value = double(audit.max_violation);
    end
end
end

function value = get_field(data, name, default_value)
if isstruct(data) && isfield(data, name)
    value = data.(name);
else
    value = default_value;
end
end

function expected = expected_metadata(config, s, c, mc, ...
    channel_seed, recovery_seed, worker_count, pslr_min)
expected = struct();
expected.archive_version = config.archive_version;
expected.experiment_id = config.experiment_id;
expected.result_schema_version = config.result_schema_version;
expected.num_mc = config.num_mc;
expected.scenario_index = s;
expected.scenario_id = config.scenarios(s).id;
expected.cv_index = c;
expected.CV_max = config.CV_grid(c);
expected.mc_index = mc;
expected.channel_seed = channel_seed;
expected.recovery_seed = recovery_seed;
expected.gaussian_randomization_trials = ...
    config.gaussian_randomization_trials;
expected.cvx_solver = config.cvx_solver;
expected.cvx_solver_threads = config.cvx_solver_threads;
expected.worker_count = worker_count;
expected.pslr_min = pslr_min;
end

function tf = valid_point_file(point_path, expected, required_run_id)
tf = false;
if exist(point_path, 'file') ~= 2, return; end
try
    saved = load(point_path, 'point');
    if ~isfield(saved, 'point') || ~isstruct(saved.point), return; end
    point = saved.point;
    if ~isempty(required_run_id) && ...
            (~isfield(point, 'run_id') || ...
            ~strcmp(point.run_id, required_run_id))
        return;
    end
    fields = fieldnames(expected);
    for i = 1:numel(fields)
        name = fields{i};
        if ~isfield(point, name) || ...
                ~isequaln(point.(name), expected.(name))
            return;
        end
    end
    if ~isfield(point, 'completed') || ~isequal(point.completed, true) || ...
            ~isfield(point, 'proposed') || ~isfield(point, 'direct')
        return;
    end
    required = fieldnames(empty_method_record());
    for i = 1:numel(required)
        if ~isfield(point.proposed, required{i}) || ...
                ~isfield(point.direct, required{i})
            return;
        end
    end
    if ~valid_method_record(point.proposed) || ...
            ~valid_method_record(point.direct)
        return;
    end
    tf = true;
catch
    tf = false;
end
end

function tf = valid_method_record(record)
tf = isempty(record.exception_identifier) && ...
    isempty(record.exception_report);
numeric_fields = {'runtime_sec', 'total_ipm_iters', ...
    'gr_trials', 'gr_solver_iters'};
for i = 1:numel(numeric_fields)
    value = record.(numeric_fields{i});
    tf = tf && isnumeric(value) && isscalar(value) && ...
        isfinite(value) && value >= 0;
end
logical_fields = {'solver_feasible', 'covariance_feasible', ...
    'initial_evd_feasible', 'final_rank1_feasible', ...
    'gr_attempted', 'gr_used'};
for i = 1:numel(logical_fields)
    value = record.(logical_fields{i});
    tf = tf && islogical(value) && isscalar(value);
end
tf = tf && (~record.solver_feasible || record.total_ipm_iters > 0);
end

function assert_replay_match(timed, profile)
fields = {'solver_feasible', 'covariance_feasible', ...
    'initial_evd_feasible', 'final_rank1_feasible', ...
    'gr_attempted_any', 'gr_used_any', 'gr_trials_total', ...
    'iters', 'inner_iters', 'stop_reason', 'solver_status', ...
    'recovery_method'};
for i = 1:numel(fields)
    name = fields{i};
    if ~isequaln(get_field(timed, name, []), ...
            get_field(profile, name, []))
        error('StressPoint:ReplayMismatch', ...
            'Timed/profile replay mismatch in %s.', name);
    end
end
end

function atomic_save_point(point_path, point)
points_dir = fileparts(point_path);
tmp_path = [tempname(points_dir) '.mat'];
cleanup = onCleanup(@() delete_if_present(tmp_path));
save(tmp_path, 'point', '-v7');
[ok, message] = movefile(tmp_path, point_path, 'f');
if ~ok
    error('Failed to finalize point file %s: %s', point_path, message);
end
clear cleanup;
end

function delete_if_present(path)
if exist(path, 'file') == 2
    delete(path);
end
end

function path = stress_point_path(points_dir, s, c, mc)
path = fullfile(points_dir, sprintf( ...
    'point_s%02d_c%02d_mc%03d.mat', s, c, mc));
end

function indices = validate_mc_indices(indices, num_mc)
indices = unique(indices(:).', 'stable');
if isempty(indices) || any(indices < 1) || any(indices > num_mc) || ...
        any(indices ~= floor(indices)) || any(diff(indices) ~= 1)
    error('mc_indices must be one consecutive integer range in 1:num_mc.');
end
end

function alpha = init_alpha_qos_safe(H, params)
K = params.K;
N = params.N;
alpha = zeros(K, N);
if K > N
    alpha = init_alpha(H, params);
    return;
end
available = true(1, N);
for k = 1:K
    gains = squeeze(sum(abs(H(:, k, :)).^2, 1)).';
    gains(~available) = -Inf;
    [~, n_best] = max(gains);
    alpha(k, n_best) = 1;
    available(n_best) = false;
end
for n = find(available)
    h_norms = vecnorm(H(:, :, n), 2, 1);
    [~, k_best] = max(h_norms);
    alpha(k_best, n) = 1;
end
end

function timestamp = utc_timestamp()
timestamp = char(datetime("now", 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
end
