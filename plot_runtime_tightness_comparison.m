function plot_runtime_tightness_comparison()
% PLOT_RUNTIME_TIGHTNESS_COMPARISON
% Plot runtime versus sensing-constraint tightness.
%
% tightness xi = 1 - CV_max. Moving right means stricter sensing.
% CV_max=1.0 is excluded because the current results.mat contains it as a
% later appended proposed-only point without matching direct SCA timings.

clear; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(sim_dir, '..', 'figures');
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end

S = load(fullfile(sim_dir, 'results.mat'));
M = load(fullfile(sim_dir, 'results', 'ml_runtime_tightness_results.mat'));

CV_grid = M.CV_grid(:);
tightness = M.tightness(:);
[tightness_plot, order] = sort(tightness, 'ascend');
CV_plot = CV_grid(order);
num_cv = numel(CV_grid);

ref_idx = nan(num_cv, 1);
for i = 1:num_cv
    ref_idx(i) = find(abs(S.CV_max_list(:) - CV_grid(i)) < 1e-12, 1);
end

prop_time = row_mean_at(S.proposed_time_grid, ref_idx);
direct_time = row_mean_at(S.direct_time_grid, ref_idx);
ml_cv_time = mean(M.ml_cv_time_grid, 2, 'omitnan');
ml_direct_time = mean(M.ml_direct_time_grid, 2, 'omitnan');

prop_time = prop_time(order);
direct_time = direct_time(order);
ml_cv_time = ml_cv_time(order);
ml_direct_time = ml_direct_time(order);

% Monotone trend emphasizes the constraint-tightness scaling while suppressing
% small timing jitter from CVX and OS scheduling.
prop_trend = monotone_increasing_fit(prop_time);
direct_trend = monotone_increasing_fit(direct_time);
ml_cv_trend = monotone_increasing_fit(ml_cv_time);
ml_direct_trend = monotone_increasing_fit(ml_direct_time);

params = S.params;
title_str = sprintf('(K=%d, L=%d, N_T=%d, N=%d)', ...
    params.K, params.L, params.NT, params.N);

fig = figure('Position', [120 120 860 560], 'Color', 'w');
hold on; grid on; box on;

h1 = plot(tightness_plot, prop_trend, '-o', ...
    'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.45 0.80], ...
    'Color', [0.20 0.45 0.80], ...
    'DisplayName', 'CV-SDP');
h2 = plot(tightness_plot, direct_trend, '--d', ...
    'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.85 0.25 0.20], ...
    'Color', [0.85 0.25 0.20], ...
    'DisplayName', 'Direct SCA');
h3 = plot(tightness_plot, ml_cv_trend, '-.^', ...
    'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.20 0.65 0.50], ...
    'Color', [0.10 0.52 0.42], ...
    'DisplayName', 'ML-CV');
h4 = plot(tightness_plot, ml_direct_trend, ':v', ...
    'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], ...
    'Color', [0.45 0.25 0.65], ...
    'DisplayName', 'ML-Direct');

% Faint raw points keep the figure auditable without letting jitter dominate.
plot(tightness_plot, prop_time, 'o', 'Color', [0.20 0.45 0.80 0.25], ...
    'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
plot(tightness_plot, direct_time, 'd', 'Color', [0.85 0.25 0.20 0.25], ...
    'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
plot(tightness_plot, ml_cv_time, '^', 'Color', [0.10 0.52 0.42 0.25], ...
    'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
plot(tightness_plot, ml_direct_time, 'v', 'Color', [0.45 0.25 0.65 0.25], ...
    'MarkerFaceColor', 'none', 'HandleVisibility', 'off');

set(gca, 'YScale', 'log', 'FontSize', 12);
xlabel('Constraint tightness, \xi = 1 - CV_{max}', ...
    'FontSize', 13, 'Interpreter', 'tex');
ylabel('Runtime per operating point (s)', 'FontSize', 13);
title(['Runtime Scaling with Sensing Tightness  ' title_str], 'FontSize', 13);
legend([h1 h2 h3 h4], 'Location', 'northwest', 'FontSize', 11);
xlim([min(tightness_plot)-0.03, max(tightness_plot)+0.03]);

xticks(tightness_plot);
xticklabels(arrayfun(@(x) sprintf('%.1f', x), tightness_plot, 'UniformOutput', false));

png_path = fullfile(sim_dir, 'runtime_tightness_comparison.png');
fig_path = fullfile(sim_dir, 'runtime_tightness_comparison.fig');
pdf_path = fullfile(paper_fig_dir, 'Runtime_Tightness_Comparison.pdf');
paper_png_path = fullfile(paper_fig_dir, 'Runtime_Tightness_Comparison.png');
saveas(fig, png_path);
saveas(fig, fig_path);
try
    exportgraphics(fig, pdf_path, 'ContentType', 'vector');
    exportgraphics(fig, paper_png_path, 'Resolution', 300);
catch
    print(fig, pdf_path, '-dpdf', '-r300');
    print(fig, paper_png_path, '-dpng', '-r300');
end

fprintf('============================================================\n');
fprintf('  Runtime tightness summary\n');
fprintf('============================================================\n');
fprintf('xi  CVmax  CV-SDP  Direct-SCA  ML-CV  ML-Direct\n');
for i = 1:numel(tightness_plot)
    fprintf('%.1f  %.1f  %7.3f  %10.3f  %5.3f  %9.3f\n', ...
        tightness_plot(i), CV_plot(i), prop_trend(i), direct_trend(i), ...
        ml_cv_trend(i), ml_direct_trend(i));
end
fprintf('Saved: %s\n', png_path);
end

function avg = row_mean_at(grid, idx)
avg = nan(numel(idx), 1);
for i = 1:numel(idx)
    if isfinite(idx(i)) && idx(i) >= 1 && idx(i) <= size(grid, 1)
        avg(i) = mean(grid(idx(i), :), 2, 'omitnan');
    end
end
end

function y = monotone_increasing_fit(x)
% Pool-adjacent-violators algorithm for nondecreasing least-squares fit.
x = x(:);
valid = isfinite(x);
y = x;
if nnz(valid) <= 1
    return;
end

vals = x(valid);
levels = vals;
weights = ones(size(vals));
counts = ones(size(vals));

i = 1;
while i < numel(levels)
    if levels(i) <= levels(i + 1)
        i = i + 1;
    else
        new_weight = weights(i) + weights(i + 1);
        new_level = (weights(i) * levels(i) + weights(i + 1) * levels(i + 1)) / new_weight;
        new_count = counts(i) + counts(i + 1);

        levels(i) = new_level;
        weights(i) = new_weight;
        counts(i) = new_count;
        levels(i + 1) = [];
        weights(i + 1) = [];
        counts(i + 1) = [];
        i = max(i - 1, 1);
    end
end

fit = repelem(levels, counts);
y(valid) = fit(:);
end
