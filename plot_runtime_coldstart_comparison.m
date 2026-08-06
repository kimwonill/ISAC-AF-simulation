function plot_runtime_coldstart_comparison()
% PLOT_RUNTIME_COLDSTART_COMPARISON
% Runtime comparison from independently initialized operating points.

clear; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(sim_dir, '..', 'figures');
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end

M = load(fullfile(sim_dir, 'results', 'runtime_coldstart_results.mat'));
reward_path = fullfile(sim_dir, 'results', 'ml_reward_inference_results.mat');
supervised_path = fullfile(sim_dir, 'results', 'ml_inference_results.mat');
if exist(reward_path, 'file') == 2
    inference_path = reward_path;
    inference_label = 'RL';
    title_prefix = 'Label-free Reward-Trained Inference Runtime';
elseif exist(supervised_path, 'file') == 2
    inference_path = supervised_path;
    inference_label = 'ML';
    title_prefix = 'Offline-trained ML Inference Runtime';
else
    inference_path = '';
    inference_label = 'ML';
    title_prefix = 'Cold-start Runtime Scaling';
end
use_inference_ml = exist(inference_path, 'file') == 2;
if use_inference_ml
    I = load(inference_path);
end

CV_grid = M.CV_grid(:);
tightness = M.tightness(:);
[tightness_plot, order] = sort(tightness, 'ascend');
CV_plot = CV_grid(order);

prop_time = mean(M.prop_time_grid, 2, 'omitnan');
direct_time = mean(M.direct_time_grid, 2, 'omitnan');
if use_inference_ml
    ml_cv_time = align_time_grid(CV_grid, I.CV_grid(:), I.ml_cv_time_grid);
    ml_direct_time = align_time_grid(CV_grid, I.CV_grid(:), I.ml_direct_time_grid);
    ml_cv_name = [inference_label '-CV inference'];
    ml_direct_name = [inference_label '-Direct inference'];
    y_label = 'Runtime per operating point (s)';
else
    ml_cv_time = mean(M.ml_cv_time_grid, 2, 'omitnan');
    ml_direct_time = mean(M.ml_direct_time_grid, 2, 'omitnan');
    ml_cv_name = 'ML-CV';
    ml_direct_name = 'ML-Direct';
    y_label = 'Cold-start runtime per operating point (s)';
end

prop_time = prop_time(order);
direct_time = direct_time(order);
ml_cv_time = ml_cv_time(order);
ml_direct_time = ml_direct_time(order);

prop_trend = monotone_increasing_fit(prop_time);
direct_trend = monotone_increasing_fit(direct_time);
ml_cv_trend = monotone_increasing_fit(ml_cv_time);
ml_direct_trend = monotone_increasing_fit(ml_direct_time);

params = M.params;
title_str = sprintf('(K=%d, L=%d, N_T=%d, N=%d)', ...
    params.K, params.L, params.NT, params.N);

fig = figure('Position', [120 120 880 570], 'Color', 'w');
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
    'DisplayName', ml_cv_name);
h4 = plot(tightness_plot, ml_direct_trend, ':v', ...
    'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.55 0.35 0.75], ...
    'Color', [0.45 0.25 0.65], ...
    'DisplayName', ml_direct_name);

plot(tightness_plot, prop_time, 'o', 'Color', [0.20 0.45 0.80 0.24], ...
    'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
plot(tightness_plot, direct_time, 'd', 'Color', [0.85 0.25 0.20 0.24], ...
    'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
plot(tightness_plot, ml_cv_time, '^', 'Color', [0.10 0.52 0.42 0.24], ...
    'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
plot(tightness_plot, ml_direct_time, 'v', 'Color', [0.45 0.25 0.65 0.24], ...
    'MarkerFaceColor', 'none', 'HandleVisibility', 'off');

set(gca, 'YScale', 'log', 'FontSize', 12);
xlabel('Constraint tightness, \xi = 1 - CV_{max}', ...
    'FontSize', 13, 'Interpreter', 'tex');
ylabel(y_label, 'FontSize', 13);
title([title_prefix '  ' title_str], 'FontSize', 13);
legend([h1 h2 h3 h4], 'Location', 'northwest', 'FontSize', 11);
xlim([min(tightness_plot)-0.03, max(tightness_plot)+0.03]);
xticks(tightness_plot);
xticklabels(arrayfun(@(x) sprintf('%.1f', x), tightness_plot, 'UniformOutput', false));

png_path = fullfile(sim_dir, 'runtime_coldstart_comparison.png');
fig_path = fullfile(sim_dir, 'runtime_coldstart_comparison.fig');
pdf_path = fullfile(paper_fig_dir, 'Runtime_Coldstart_Comparison.pdf');
paper_png_path = fullfile(paper_fig_dir, 'Runtime_Coldstart_Comparison.png');
saveas(fig, png_path);
saveas(fig, fig_path);
try
    tight_export_figure(fig, pdf_path, 'ContentType', 'image', 'Resolution', 300);
    tight_export_figure(fig, paper_png_path, 'Resolution', 300);
catch
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, pdf_path, '-dpdf', '-image', '-r300');
    print(fig, paper_png_path, '-dpng', '-r300');
end

fprintf('============================================================\n');
if use_inference_ml
    if strcmpi(inference_label, 'RL')
        fprintf('  Label-free reward-trained inference runtime summary\n');
    else
        fprintf('  Offline-trained ML inference runtime summary\n');
    end
else
    fprintf('  Cold-start runtime summary\n');
end
fprintf('============================================================\n');
fprintf('xi  CVmax  CV-SDP  Direct-SCA  %s-CV  %s-Direct\n', ...
    inference_label, inference_label);
for i = 1:numel(tightness_plot)
    fprintf('%.1f  %.1f  %7.3f  %10.3f  %5.3f  %9.3f\n', ...
        tightness_plot(i), CV_plot(i), prop_trend(i), direct_trend(i), ...
        ml_cv_trend(i), ml_direct_trend(i));
end
fprintf('Saved: %s\n', png_path);
end

function avg = align_time_grid(CV_grid, source_CV_grid, source_time_grid)
avg = nan(numel(CV_grid), 1);
source_avg = mean(source_time_grid, 2, 'omitnan');
for i = 1:numel(CV_grid)
    [gap, idx] = min(abs(source_CV_grid(:) - CV_grid(i)));
    if ~isempty(gap) && gap <= 1e-9
        avg(i) = source_avg(idx);
    end
end
end

function y = monotone_increasing_fit(x)
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
