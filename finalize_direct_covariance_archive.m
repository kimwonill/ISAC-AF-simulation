function archive = finalize_direct_covariance_archive(num_mc)
%FINALIZE_DIRECT_COVARIANCE_ARCHIVE Validate and merge Direct-SCA points.

if nargin < 1 || isempty(num_mc)
    num_mc = 100;
end
sim_dir = fileparts(mfilename('fullpath'));
threshold_source = fullfile('results', 'pareto_grid_1x4_pslr', ...
    'pareto_pslr_NT4_N16_MC100.mat');
source_file = fullfile(sim_dir, threshold_source);
source = load(source_file, 'params', 'CV_max_list', 'pslr_lin_grid');
cv_grid = source.CV_max_list(:).';
num_cv = numel(cv_grid);
params = source.params;
params.cvx_solver = 'mosek';
params.cvx_solver_threads = 1;
archive_version = 1;
output_dir = fullfile(sim_dir, 'results', sprintf( ...
    'direct_covariance_archive_MC%d', num_mc));

H_by_mc = cell(1, num_mc);
W_sdp_grid = cell(num_cv, num_mc);
alpha_grid = cell(num_cv, num_mc);
solver_feasible = false(num_cv, num_mc);
solver_status = strings(num_cv, num_mc);
pslr_min_grid = nan(num_cv, num_mc);
missing = strings(0, 1);

for mc = 1:num_mc
    rng(mc, 'twister');
    expected_H = generate_channel(params);
    for cv_index = 1:num_cv
        point_file = fullfile(output_dir, sprintf( ...
            'direct_covariance_MC%03d_CV%02d.mat', mc, cv_index));
        if exist(point_file, 'file') ~= 2
            missing(end + 1, 1) = string(point_file); %#ok<AGROW>
            continue;
        end
        saved = load(point_file, 'W', 'alpha', 'H', ...
            'solver_feasible', 'solver_status', 'channel_seed', 'cv_index', ...
            'CV_max', 'pslr_min', 'archive_version');
        required = {'W', 'alpha', 'H', 'solver_feasible', ...
            'solver_status', 'channel_seed', 'cv_index', 'CV_max', ...
            'pslr_min', 'archive_version'};
        expected_pslr_min = source.pslr_lin_grid(cv_index, mc) * ...
            (1 - params.direct_pslr_target_relax);
        if ~all(isfield(saved, required)) || saved.channel_seed ~= mc || ...
                saved.cv_index ~= cv_index || ...
                saved.archive_version ~= archive_version || ...
                abs(saved.CV_max - cv_grid(cv_index)) > 1e-12 || ...
                abs(saved.pslr_min - expected_pslr_min) > ...
                    1e-12 * max(1, abs(expected_pslr_min))
            error('Invalid point archive: %s', point_file);
        end
        if ~isequal(size(saved.H), [4, 5, 16])
            error('Unexpected H dimensions in %s.', point_file);
        elseif ~isequal(saved.H, expected_H)
            error('Channel does not match seed %d in %s.', mc, point_file);
        end
        if ~isequal(size(saved.alpha), [5, 16])
            error('Unexpected alpha dimensions in %s.', point_file);
        end
        if saved.solver_feasible && ~isequal(size(saved.W), [4, 4, 16])
            error('Unexpected W dimensions in %s.', point_file);
        elseif ~saved.solver_feasible && ~isempty(saved.W)
            error('Infeasible point contains a nonempty W in %s.', point_file);
        end
        if isempty(H_by_mc{mc})
            H_by_mc{mc} = saved.H;
        elseif ~isequal(H_by_mc{mc}, saved.H)
            error('Channel mismatch across CV points for MC=%d.', mc);
        end
        W_sdp_grid{cv_index, mc} = saved.W;
        alpha_grid{cv_index, mc} = saved.alpha;
        solver_feasible(cv_index, mc) = saved.solver_feasible;
        solver_status(cv_index, mc) = string(saved.solver_status);
        pslr_min_grid(cv_index, mc) = saved.pslr_min;
    end
end

if ~isempty(missing)
    error('Archive is incomplete: %d/%d point files are missing.', ...
        numel(missing), num_mc * num_cv);
end

H_grid = cat(4, H_by_mc{:});
CV_max_list = cv_grid;
channel_seeds = 1:num_mc;
seed_policy = 'rng(mc, ''twister'')';
point_file_pattern = 'direct_covariance_MC%03d_CV%02d.mat';

archive.archive_version = archive_version;
archive.num_mc = num_mc;
archive.CV_max_list = CV_max_list;
archive.H_grid = H_grid;
archive.W_sdp_grid = W_sdp_grid;
archive.alpha_grid = alpha_grid;
archive.solver_feasible = solver_feasible;
archive.solver_status = solver_status;
archive.pslr_min_grid = pslr_min_grid;
archive.channel_seeds = channel_seeds;
archive.params = params;
archive.seed_policy = seed_policy;
archive.threshold_source = threshold_source;
archive.point_file_pattern = point_file_pattern;

archive_path = fullfile(output_dir, sprintf( ...
    'direct_covariance_archive_MC%d.mat', num_mc));
temporary_file = [tempname(output_dir), '.mat'];
save(temporary_file, 'H_grid', 'W_sdp_grid', 'alpha_grid', ...
    'solver_feasible', 'solver_status', 'pslr_min_grid', 'CV_max_list', ...
    'num_mc', 'channel_seeds', 'params', 'archive_version', 'seed_policy', ...
    'threshold_source', 'point_file_pattern', '-v7.3');
movefile(temporary_file, archive_path, 'f');

fprintf('Direct covariance archive complete: %d/%d files.\n', ...
    num_mc * num_cv, num_mc * num_cv);
fprintf('Covariance solver feasible: %d/%d (%.2f%%).\n', ...
    nnz(solver_feasible), num_mc * num_cv, ...
    100 * nnz(solver_feasible) / (num_mc * num_cv));
fprintf('Merged archive: %s\n', archive_path);
end
