function plot_computational_burden_results(source_path)
% PLOT_COMPUTATIONAL_BURDEN_RESULTS  Plot runtime and iteration burden.

if nargin < 1 || isempty(source_path)
    sim_dir = fileparts(mfilename('fullpath'));
    source_path = fullfile(sim_dir, 'results', 'computational_burden_mrt_NT4_N16_MC10.mat');
else
    sim_dir = fileparts(mfilename('fullpath'));
end

S = load(source_path);
fig_dir = fullfile(sim_dir, '..', 'figures');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end

CV = S.CV_grid(:);
prop_success = solved_mask(S.prop_status_grid);
direct_success = solved_mask(S.direct_status_grid);

prop_time_grid = mask_failed(S.prop_time_grid, prop_success);
direct_time_grid = mask_failed(S.direct_time_grid, direct_success);
prop_iters_grid = mask_failed(S.prop_iters_grid, prop_success);
direct_inner_iters_grid = mask_failed(S.direct_inner_iters_grid, direct_success);
prop_cvx_solver_iters_grid = mask_failed(S.prop_cvx_solver_iters_grid, prop_success);
direct_cvx_solver_iters_grid = mask_failed(S.direct_cvx_solver_iters_grid, direct_success);

prop_time = mean(prop_time_grid, 2, 'omitnan');
direct_time = mean(direct_time_grid, 2, 'omitnan');
prop_time_std = std(prop_time_grid, 0, 2, 'omitnan');
direct_time_std = std(direct_time_grid, 0, 2, 'omitnan');

prop_solves = mean(prop_iters_grid, 2, 'omitnan');
direct_solves = mean(direct_inner_iters_grid, 2, 'omitnan');
prop_solves_std = std(prop_iters_grid, 0, 2, 'omitnan');
direct_solves_std = std(direct_inner_iters_grid, 0, 2, 'omitnan');

prop_solver_iters = mean(prop_cvx_solver_iters_grid, 2, 'omitnan');
direct_solver_iters = mean(direct_cvx_solver_iters_grid, 2, 'omitnan');
prop_solver_iters_std = std(prop_cvx_solver_iters_grid, 0, 2, 'omitnan');
direct_solver_iters_std = std(direct_cvx_solver_iters_grid, 0, 2, 'omitnan');

valid_time = isfinite(prop_time) & isfinite(direct_time);
valid_solve = isfinite(prop_solves) & isfinite(direct_solves);
valid_solver = isfinite(prop_solver_iters) & isfinite(direct_solver_iters);
avg_speedup = mean(direct_time_grid(:), 'omitnan') / mean(prop_time_grid(:), 'omitnan');
avg_solve_ratio = mean(direct_inner_iters_grid(:), 'omitnan') / mean(prop_iters_grid(:), 'omitnan');
avg_solver_iter_ratio = mean(direct_cvx_solver_iters_grid(:), 'omitnan') / ...
    mean(prop_cvx_solver_iters_grid(:), 'omitnan');
prop_success_rate = mean(prop_success(:));
direct_success_rate = mean(direct_success(:));

fig = figure('Position', [100 100 1370 430], 'Color', 'w');
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

palette = paper_palette();
blue = palette(1, :);
red = palette(2, :);

ax1 = nexttile(tl, 1);
hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
errorbar(ax1, CV(valid_time), prop_time(valid_time), prop_time_std(valid_time), ...
    '-o', 'Color', blue, 'MarkerFaceColor', blue, 'LineWidth', 2.0, ...
    'MarkerSize', 6.0, 'DisplayName', 'CV-SDP');
errorbar(ax1, CV(valid_time), direct_time(valid_time), direct_time_std(valid_time), ...
    '--d', 'Color', red, 'MarkerFaceColor', red, 'LineWidth', 2.0, ...
    'MarkerSize', 6.0, 'DisplayName', 'Direct PSLR/ISLR SCA');
set(ax1, 'YScale', 'log', 'FontSize', 11, 'Layer', 'top');
xlabel(ax1, '$\mathrm{CV}_{\max}$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel(ax1, 'Optimization time (s)', 'FontSize', 13);
title(ax1, '(a) Cold-start runtime', 'FontSize', 13, 'FontWeight', 'bold');
legend(ax1, 'Location', 'northwest', 'FontSize', 10);
text(ax1, 0.03, 0.08, sprintf('Mean speedup: %.1fx', avg_speedup), ...
    'Units', 'normalized', 'FontSize', 11, 'Color', [0.15 0.15 0.15]);

ax2 = nexttile(tl, 2);
hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
errorbar(ax2, CV(valid_solve), prop_solves(valid_solve), prop_solves_std(valid_solve), ...
    '-o', 'Color', blue, 'MarkerFaceColor', blue, 'LineWidth', 2.0, ...
    'MarkerSize', 6.0, 'DisplayName', 'CV-SDP AO solves');
errorbar(ax2, CV(valid_solve), direct_solves(valid_solve), direct_solves_std(valid_solve), ...
    '--d', 'Color', red, 'MarkerFaceColor', red, 'LineWidth', 2.0, ...
    'MarkerSize', 6.0, 'DisplayName', 'Direct SCA solves');
set(ax2, 'FontSize', 11, 'Layer', 'top');
xlabel(ax2, '$\mathrm{CV}_{\max}$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel(ax2, 'Convex subproblem solves', 'FontSize', 13);
title(ax2, '(b) CVX calls', 'FontSize', 13, 'FontWeight', 'bold');
legend(ax2, 'Location', 'northwest', 'FontSize', 10);
text(ax2, 0.03, 0.08, sprintf('Mean solve ratio: %.1fx', avg_solve_ratio), ...
    'Units', 'normalized', 'FontSize', 11, 'Color', [0.15 0.15 0.15]);

ax3 = nexttile(tl, 3);
hold(ax3, 'on'); grid(ax3, 'on'); box(ax3, 'on');
errorbar(ax3, CV(valid_solver), prop_solver_iters(valid_solver), prop_solver_iters_std(valid_solver), ...
    '-o', 'Color', blue, 'MarkerFaceColor', blue, 'LineWidth', 2.0, ...
    'MarkerSize', 6.0, 'DisplayName', 'CV-SDP');
errorbar(ax3, CV(valid_solver), direct_solver_iters(valid_solver), direct_solver_iters_std(valid_solver), ...
    '--d', 'Color', red, 'MarkerFaceColor', red, 'LineWidth', 2.0, ...
    'MarkerSize', 6.0, 'DisplayName', 'Direct PSLR/ISLR SCA');
set(ax3, 'FontSize', 11, 'Layer', 'top');
xlabel(ax3, '$\mathrm{CV}_{\max}$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel(ax3, 'CVX solver iterations', 'FontSize', 13);
title(ax3, '(c) Interior iterations', 'FontSize', 13, 'FontWeight', 'bold');
legend(ax3, 'Location', 'northwest', 'FontSize', 10);
text(ax3, 0.03, 0.08, sprintf('Mean iter. ratio: %.1fx', avg_solver_iter_ratio), ...
    'Units', 'normalized', 'FontSize', 11, 'Color', [0.15 0.15 0.15]);

if isfield(S, 'init_mode')
    init_label = upper(string(S.init_mode));
else
    init_label = "FLAT";
end
title(tl, sprintf('Computational Burden (%s Init.), $N_T=%d$, $N=%d$, MC=%d', ...
    char(init_label), S.params.NT, S.params.N, S.num_mc), ...
    'Interpreter', 'latex', 'FontSize', 14);

out_pdf = fullfile(fig_dir, 'Computational_Burden_Comparison.pdf');
out_png = fullfile(fig_dir, 'Computational_Burden_Comparison.png');
safe_export(fig, out_pdf, 'pdf');
safe_export(fig, out_png, 'png');

summary_path = fullfile(sim_dir, 'results', 'computational_burden_summary.mat');
save(summary_path, 'CV', 'prop_time', 'direct_time', 'prop_time_std', ...
    'direct_time_std', 'prop_solves', 'direct_solves', 'prop_solves_std', ...
    'direct_solves_std', 'prop_solver_iters', 'direct_solver_iters', ...
    'prop_solver_iters_std', 'direct_solver_iters_std', ...
    'avg_speedup', 'avg_solve_ratio', 'avg_solver_iter_ratio', ...
    'prop_success_rate', 'direct_success_rate');

fprintf('Saved computational-burden figure: %s\n', out_pdf);
fprintf('Saved computational-burden figure: %s\n', out_png);
fprintf('Saved computational-burden summary: %s\n', summary_path);
end

function mask = solved_mask(status_grid)
mask = contains(string(status_grid), 'Solved');
end

function data = mask_failed(data, mask)
data(~mask) = NaN;
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        exportgraphics(fig, filename, 'ContentType', 'image', 'Resolution', 450);
    else
        exportgraphics(fig, filename, 'Resolution', 300);
    end
catch
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-vector');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
end
