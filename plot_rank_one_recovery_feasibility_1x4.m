function plot_rank_one_recovery_feasibility_1x4(num_trials)
%PLOT_RANK_ONE_RECOVERY_FEASIBILITY_1X4 Compare EVD and EVD+GR feasibility.
%   Run RUN_PROPOSED_COVARIANCE_RECOVERY_AUDIT first.  Both inputs use the
%   Pareto CV grid and its channel rule rng(mc,'twister').

if nargin < 1 || isempty(num_trials)
    num_trials = 10;
end
sim_dir = fileparts(mfilename('fullpath'));
result_dir = fullfile(sim_dir, 'results');
proposed_path = fullfile(result_dir, sprintf( ...
    'proposed_rank_one_recovery_audit_R%d_MC100.mat', num_trials));
direct_path = fullfile(result_dir, sprintf( ...
    'direct_rank_one_recovery_audit_R%d_MC100.mat', num_trials));
assert(exist(proposed_path, 'file') == 2, ...
    'Run run_proposed_covariance_recovery_audit(%d) first.', num_trials);
assert(exist(direct_path, 'file') == 2, ...
    'Missing Direct audit: %s', direct_path);

proposed = load(proposed_path, 'summary', 'raw');
direct = load(direct_path, 'summary', 'raw');
assert(isequal(proposed.summary.CV_max_list(:), direct.summary.CV_max_list(:)), ...
    'Proposed and Direct CV grids differ.');
assert(isequal(config_pairs(proposed.summary.configurations), ...
    config_pairs(direct.summary.configurations)), ...
    'Proposed and Direct configuration orders differ.');

cv_grid = proposed.summary.CV_max_list(:).';
configs = proposed.summary.configurations;
num_configs = numel(configs);
assert(num_configs == 4, 'This plotting function expects four configurations.');

fig = figure('Color', 'w', 'Units', 'pixels', ...
    'Position', [80, 120, 1500, 360]);
tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
colors.proposed = [0.80, 0.15, 0.15];
colors.direct = [0.10, 0.35, 0.75];
for q = 1:num_configs
    nexttile;
    prop_evd = 100 * mean(proposed.raw.initial_evd_feasible(:, :, q), 2).';
    prop_final = 100 * mean(proposed.raw.final_feasible(:, :, q), 2).';
    direct_evd = 100 * mean(direct.raw.initial_evd_feasible(:, :, q), 2).';
    direct_final = 100 * mean(direct.raw.final_feasible(:, :, q), 2).';

    h(1) = plot(cv_grid, prop_evd, '--o', 'Color', colors.proposed, ...
        'MarkerFaceColor', 'w', 'LineWidth', 1.4, 'MarkerSize', 5); hold on;
    h(2) = plot(cv_grid, prop_final, '-o', 'Color', colors.proposed, ...
        'MarkerFaceColor', colors.proposed, 'LineWidth', 1.7, 'MarkerSize', 5);
    h(3) = plot(cv_grid, direct_evd, '--s', 'Color', colors.direct, ...
        'MarkerFaceColor', 'w', 'LineWidth', 1.4, 'MarkerSize', 5);
    h(4) = plot(cv_grid, direct_final, '-s', 'Color', colors.direct, ...
        'MarkerFaceColor', colors.direct, 'LineWidth', 1.7, 'MarkerSize', 5);
    grid on; box on;
    xlim([min(cv_grid), max(cv_grid)]);
    ylim([50, 101]);
    xticks(cv_grid(1:2:end));
    xlabel('$\mathrm{CV}_{\max}$', 'Interpreter', 'latex');
    if q == 1
        ylabel('Rank-one feasibility (\%)');
    end
    title(sprintf('$(N_T,N)=(%d,%d)$', configs(q).NT, configs(q).N), ...
        'Interpreter', 'latex');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, 'LineWidth', 0.8);
    if q == 1
        legend(h, {'Proposed: EVD', 'Proposed: EVD+GR', ...
            'Direct: EVD', 'Direct: EVD+GR'}, 'Location', 'southwest', ...
            'FontSize', 8);
    end
end

figure_dir = fullfile(sim_dir, 'figures');
if exist(figure_dir, 'dir') ~= 7
    mkdir(figure_dir);
end
base_name = sprintf('RankOne_Recovery_Feasibility_1x4_R%d', num_trials);
exportgraphics(fig, fullfile(figure_dir, [base_name '.pdf']), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(figure_dir, [base_name '.png']), 'Resolution', 300);
fprintf('Saved: %s\n', fullfile(figure_dir, [base_name '.pdf']));
end

function pairs = config_pairs(configs)
pairs = [[configs.NT].', [configs.N].'];
end
