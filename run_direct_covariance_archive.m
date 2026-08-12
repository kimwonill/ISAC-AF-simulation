function run_direct_covariance_archive(num_mc, force_rerun, mc_indices)
%RUN_DIRECT_COVARIANCE_ARCHIVE Save H, alpha, and W for Direct-SCA MC points.
%   The function runs only the Direct-SCA baseline. Proposed CV, CRB, and
%   MI optimizations are never invoked. Direct PSLR thresholds are read from
%   the tracked MC=100 Pareto cache so the original experiment is replayed
%   with the same channel seeds and operating points.

if nargin < 1 || isempty(num_mc)
    num_mc = 100;
end
if nargin < 2 || isempty(force_rerun)
    force_rerun = false;
end
if nargin < 3 || isempty(mc_indices)
    mc_indices = 1:num_mc;
end
mc_indices = unique(mc_indices(:).', 'stable');
if any(mc_indices < 1) || any(mc_indices > num_mc) || ...
        any(mc_indices ~= floor(mc_indices))
    error('mc_indices must contain integers in 1:num_mc.');
end

sim_dir = fileparts(mfilename('fullpath'));
source_file = fullfile(sim_dir, 'results', 'pareto_grid_1x4_pslr', ...
    'pareto_pslr_NT4_N16_MC100.mat');
if exist(source_file, 'file') ~= 2
    error('Missing direct-threshold source: %s', source_file);
end
source = load(source_file, 'params', 'CV_max_list', 'pslr_lin_grid');
if num_mc > size(source.pslr_lin_grid, 2)
    error('Threshold source contains only %d channel realizations.', ...
        size(source.pslr_lin_grid, 2));
end

if ~isfield(source, 'params') || ~isstruct(source.params)
    error('Threshold source does not contain the original parameter set.');
end
params = source.params;
if params.NT ~= 4 || params.N ~= 16 || params.K ~= 5
    error('Unexpected source configuration: NT=%d, N=%d, K=%d.', ...
        params.NT, params.N, params.K);
end
if ~isfield(params, 'direct_pslr_target_relax')
    error('Source parameters do not define direct_pslr_target_relax.');
end
params.cvx_solver = 'mosek';
params.cvx_solver_threads = 1;
cv_grid = source.CV_max_list(:).';
num_cv = numel(cv_grid);
if size(source.pslr_lin_grid, 1) ~= num_cv
    error('CV grid and threshold-grid dimensions do not agree.');
end
archive_version = 1;
output_dir = fullfile(sim_dir, 'results', sprintf( ...
    'direct_covariance_archive_MC%d', num_mc));
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

fprintf(['Direct-only covariance archive: MC=%d, worker range %d:%d, ' ...
    'CV points=%d\n'], num_mc, mc_indices(1), mc_indices(end), num_cv);
fprintf('Threshold source: %s\n', source_file);

t_worker = tic;
for mc = mc_indices
    channel_seed = mc;
    rng(channel_seed, 'twister');
    H = generate_channel(params);
    alpha_warm = [];
    W_warm = [];

    for cv_index = 1:num_cv
        achieved_pslr = source.pslr_lin_grid(cv_index, mc);
        if ~isfinite(achieved_pslr)
            error('Missing proposed PSLR threshold at MC=%d, CV index=%d.', ...
                mc, cv_index);
        end
        CV_max = cv_grid(cv_index);
        pslr_min = achieved_pslr * ...
            (1 - params.direct_pslr_target_relax);
        point_file = archive_point_path( ...
            output_dir, mc, cv_index);
        if ~force_rerun && valid_existing_point( ...
                point_file, archive_version, mc, cv_index, CV_max, ...
                pslr_min, H, params)
            saved = load(point_file, 'W', 'alpha', 'solver_feasible');
            if saved.solver_feasible
                W_warm = saved.W;
                alpha_warm = saved.alpha;
            else
                W_warm = [];
                alpha_warm = [];
            end
            fprintf('  MC %03d CV %.1f: reused\n', mc, cv_grid(cv_index));
            continue;
        end

        t_point = tic;
        result = run_direct_sca_covariance( ...
            H, pslr_min, params, alpha_warm, W_warm);
        elapsed_seconds = toc(t_point);
        W = result.W;
        alpha = result.alpha;
        solver_feasible = result.solver_feasible;
        solver_status = result.status;

        temporary_file = [tempname(output_dir), '.mat'];
        save(temporary_file, 'W', 'alpha', 'H', 'solver_feasible', ...
            'solver_status', 'channel_seed', 'cv_index', 'CV_max', ...
            'pslr_min', 'archive_version', '-v7.3');
        movefile(temporary_file, point_file, 'f');

        if solver_feasible
            W_warm = W;
            alpha_warm = alpha;
        else
            W_warm = [];
            alpha_warm = [];
        end
        fprintf(['  MC %03d CV %.1f: feasible=%d, status=%s, ' ...
            '%.2fs\n'], mc, CV_max, solver_feasible, ...
            char(string(solver_status)), elapsed_seconds);
    end
end
fprintf('Worker range %d:%d completed in %.1f min.\n', ...
    mc_indices(1), mc_indices(end), toc(t_worker) / 60);
end

function path = archive_point_path(output_dir, mc, cv_index)
path = fullfile(output_dir, sprintf( ...
    'direct_covariance_MC%03d_CV%02d.mat', mc, cv_index));
end

function valid = valid_existing_point(path, archive_version, mc, cv_index, ...
    CV_max, pslr_min, H, params)
valid = false;
if exist(path, 'file') ~= 2
    return;
end
try
    saved = load(path, 'archive_version', 'channel_seed', 'cv_index', ...
        'CV_max', 'pslr_min', 'solver_feasible', 'solver_status', ...
        'W', 'alpha', 'H');
    required = {'archive_version', 'channel_seed', 'cv_index', ...
        'CV_max', 'pslr_min', 'solver_feasible', 'solver_status', ...
        'W', 'alpha', 'H'};
    valid = all(isfield(saved, required)) && ...
        saved.archive_version >= archive_version && ...
        saved.channel_seed == mc && saved.cv_index == cv_index && ...
        abs(saved.CV_max - CV_max) <= 1e-12 && ...
        abs(saved.pslr_min - pslr_min) <= ...
            1e-12 * max(1, abs(pslr_min)) && ...
        isscalar(saved.solver_feasible) && ...
        isequal(size(saved.H), [params.NT, params.K, params.N]) && ...
        isequal(saved.H, H) && ...
        isequal(size(saved.alpha), [params.K, params.N]) && ...
        ((saved.solver_feasible && isequal(size(saved.W), ...
            [params.NT, params.NT, params.N])) || ...
         (~saved.solver_feasible && isempty(saved.W)));
catch
    valid = false;
end
end
