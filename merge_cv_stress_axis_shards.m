function merge_cv_stress_axis_shards(num_mc, shard_ranges, cv_step)
if nargin < 1 || isempty(num_mc), num_mc = 100; end
if nargin < 2 || isempty(shard_ranges)
    shard_ranges = [1 6; 7 12; 13 19; 20 25; 26 31; 32 37; 38 44; 45 50; ...
        51 56; 57 62; 63 69; 70 75; 76 81; 82 87; 88 94; 95 100];
end
if nargin < 3 || isempty(cv_step), cv_step = 0.1; end
if ~isscalar(cv_step) || ~isfinite(cv_step) || cv_step <= 0 || ...
        abs(round(1 / cv_step) * cv_step - 1) > 1e-10
    error('cv_step must be a positive scalar that divides 1 exactly.');
end
cv_tag = sprintf('CV%d', round(1 / cv_step));

shard_ranges = sortrows(shard_ranges, 1);
covered = [];
for shard = 1:size(shard_ranges, 1)
    covered = [covered, shard_ranges(shard, 1):shard_ranges(shard, 2)]; %#ok<AGROW>
end
if ~isequal(covered, 1:num_mc)
    error('Shard ranges must cover every MC index from 1 to %d exactly once.', num_mc);
end

sim_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(sim_dir, 'results');
canonical_path = fullfile(out_dir, sprintf( ...
    'cv_stress_axis_pslr_only_S2S3S4_%s_NT4_N16_MC%d.mat', cv_tag, num_mc));

first_path = fullfile(out_dir, sprintf( ...
    'cv_stress_axis_pslr_only_S2S3S4_%s_NT4_N16_MC%d_shard_%03d_%03d.mat', ...
    cv_tag, num_mc, shard_ranges(1, 1), shard_ranges(1, 2)));
if exist(first_path, 'file') ~= 2
    error('Missing Figure 7 shard: %s', first_path);
end
merged = load(first_path);
fields = {'prop_success', 'direct_success', 'prop_status', 'direct_status', ...
    'prop_time', 'direct_time', 'prop_cvx_solver_iters', ...
    'direct_cvx_solver_iters'};

for shard = 1:size(shard_ranges, 1)
    mc_start = shard_ranges(shard, 1);
    mc_end = shard_ranges(shard, 2);
    shard_path = fullfile(out_dir, sprintf( ...
        'cv_stress_axis_pslr_only_S2S3S4_%s_NT4_N16_MC%d_shard_%03d_%03d.mat', ...
        cv_tag, num_mc, mc_start, mc_end));
    if exist(shard_path, 'file') ~= 2
        error('Missing Figure 7 shard: %s', shard_path);
    end
    part = load(shard_path);
    if ~isequal(part.CV_grid, merged.CV_grid) || ...
            ~isequaln(part.scenarios, merged.scenarios) || ...
            part.num_mc ~= num_mc
        error('Shard metadata mismatch: %s', shard_path);
    end
    if isfield(merged, 'result_schema_version') && ...
            (~isfield(part, 'result_schema_version') || ...
             part.result_schema_version ~= merged.result_schema_version)
        error('Shard schema mismatch: %s', shard_path);
    end
    for i = 1:numel(fields)
        name = fields{i};
        if ~isfield(part, name)
            error('Shard %s is missing field %s.', shard_path, name);
        end
        merged.(name)(:,:,mc_start:mc_end) = part.(name)(:,:,mc_start:mc_end);
    end
end


numeric_fields = {'prop_time', 'direct_time', 'prop_cvx_solver_iters', ...
    'direct_cvx_solver_iters'};
for i = 1:numel(numeric_fields)
    name = numeric_fields{i};
    if any(~isfinite(merged.(name)(:)))
        error('Merged field %s contains missing or non-finite entries.', name);
    end
end

merged.num_mc = num_mc;
merged.cv_step_override = cv_step;
merged.cv_tag = cv_tag;
if ~isfield(merged, 'experiment_metadata')
    merged.experiment_metadata = struct();
end
merged.experiment_metadata.mc_indices = 1:num_mc;
merged.experiment_metadata.shard_ranges = shard_ranges;
merged.experiment_metadata.num_shards = size(shard_ranges, 1);
merged.experiment_metadata.merged_at_utc = char(datetime( ...
    'now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));

temporary_path = [tempname(out_dir), '.mat'];
save(temporary_path, '-struct', 'merged');
movefile(temporary_path, canonical_path, 'f');
fprintf('Merged %d Figure 7 shards into %s\n', size(shard_ranges, 1), canonical_path);
end
