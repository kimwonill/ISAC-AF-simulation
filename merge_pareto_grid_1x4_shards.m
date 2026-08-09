function merge_pareto_grid_1x4_shards(num_mc, config_index, shard_ranges)
% Merge disjoint MC shards into the canonical Pareto cache.
if nargin < 1 || isempty(num_mc), num_mc = 100; end
if nargin < 2 || isempty(config_index), config_index = 4; end
if nargin < 3 || isempty(shard_ranges)
    shard_ranges = [1 13; 14 25; 26 38; 39 50; 51 63; 64 75; 76 88; 89 100];
end

sim_dir = fileparts(mfilename('fullpath'));
cache_dir = fullfile(sim_dir, 'results', 'pareto_grid_1x4_pslr');
configs = [4 16; 4 32; 8 16; 8 32];
cfg = configs(config_index, :);
canonical_path = fullfile(cache_dir, sprintf( ...
    'pareto_pslr_NT%d_N%d_MC%d.mat', cfg(1), cfg(2), num_mc));

first_path = fullfile(cache_dir, sprintf( ...
    'pareto_pslr_NT%d_N%d_MC%d_shard_%03d_%03d.mat', ...
    cfg(1), cfg(2), num_mc, shard_ranges(1, 1), shard_ranges(1, 2)));
merged = load(first_path);
fields = {'sumrate_grid', 'pslr_lin_grid', 'comm_sumrate_grid', ...
    'comm_pslr_lin_grid', 'crb_sumrate_grid', 'crb_pslr_lin_grid', ...
    'mi_sumrate_grid', 'mi_pslr_lin_grid', 'direct_sumrate_grid', ...
    'direct_pslr_lin_grid', 'reference_completed_mc', 'direct_completed_mc'};

for shard = 1:size(shard_ranges, 1)
    mc_start = shard_ranges(shard, 1);
    mc_end = shard_ranges(shard, 2);
    shard_path = fullfile(cache_dir, sprintf( ...
        'pareto_pslr_NT%d_N%d_MC%d_shard_%03d_%03d.mat', ...
        cfg(1), cfg(2), num_mc, mc_start, mc_end));
    part = load(shard_path);
    for i = 1:numel(fields)
        name = fields{i};
        if isfield(part, name)
            if isvector(part.(name))
                merged.(name)(mc_start:mc_end) = part.(name)(mc_start:mc_end);
            else
                merged.(name)(:, mc_start:mc_end) = part.(name)(:, mc_start:mc_end);
            end
        end
    end
end

save(canonical_path, '-struct', 'merged', '-v7.3');
fprintf('Merged %d shards into %s\n', size(shard_ranges, 1), canonical_path);
end
