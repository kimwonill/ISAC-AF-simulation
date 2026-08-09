function plot_pdes_pareto_sweep(num_mc_override, force_rerun)
%% PLOT_PDES_PARETO_SWEEP  PSLR Pareto curves versus illumination floor
%
% Sweeps P_des = n P_max/N and CV_max, then plots the proposed method's
% sum-rate/PSLR frontier for each illumination floor. The scenario matches
% plot_pdes_beampattern.m so the two illumination figures can be discussed
% together.
%
% Outputs:
%   pdes_pareto_sweep_results.mat
%   pdes_pareto_sweep.png/.fig
%   figures/pdes_pareto_sweep.pdf/.png

if nargin < 1 || isempty(num_mc_override)
    num_mc_override = [];
end
if nargin < 2 || isempty(force_rerun)
    force_rerun = false;
end

clearvars -except num_mc_override force_rerun; close all; clc;

addpath(genpath(fileparts(mfilename('fullpath'))));
if exist('cvx_begin', 'file') ~= 2
    error('CVX is required. Install CVX and run cvx_setup first.');
end

params = setup_params();

% Match the illumination beampattern figure.
params.NT = 8;
params.N = 32;
params.L = 2;
params.theta = [0, 30] * pi/180;
params.Q = zeros(params.K, 1);
params.run_direct_baseline = false;

pdes_multipliers = [0, 1, 2, 3];
pdes_values = pdes_multipliers * params.P_max / params.N;
CV_max_list = params.CV_max_list;
num_pdes = numel(pdes_values);
num_cv = numel(CV_max_list);
num_mc = params.num_mc;
if ~isempty(num_mc_override)
    num_mc = num_mc_override;
    params.num_mc = num_mc;
end
out_dir = fileparts(mfilename('fullpath'));
results_dir = fullfile(out_dir, 'results');
if exist(results_dir, 'dir') ~= 7
    mkdir(results_dir);
end
results_path = fullfile(results_dir, 'pdes_pareto_sweep_results.mat');
legacy_results_path = fullfile(out_dir, 'pdes_pareto_sweep_results.mat');

sumrate_grid = nan(num_pdes, num_cv, num_mc);
pslr_lin_grid = nan(num_pdes, num_cv, num_mc);
pslr_dB_grid = nan(num_pdes, num_cv, num_mc);
islr_lin_grid = nan(num_pdes, num_cv, num_mc);
islr_dB_grid = nan(num_pdes, num_cv, num_mc);
status_grid = strings(num_pdes, num_cv, num_mc);
iter_grid = nan(num_pdes, num_cv, num_mc);
runtime_grid = nan(num_pdes, num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  Illumination-Floor PSLR Pareto Sweep\n');
fprintf('============================================================\n');
fprintf('  K = %d users, targets = [%s] deg\n', ...
        params.K, num2str(rad2deg(params.theta)));
fprintf('  N_T = %d, N = %d, MC runs = %d\n', params.NT, params.N, num_mc);
fprintf('  P_des = n P_max/N, n = [%s]\n', num2str(pdes_multipliers));
fprintf('  CV_max = [%s]\n', num2str(CV_max_list));
fprintf('------------------------------------------------------------\n');

total_iters = num_pdes * num_cv * num_mc;
iter_count = 0;
t_global = tic;
use_cached_results = false;

if exist(results_path, 'file') ~= 2 && exist(legacy_results_path, 'file') == 2
    copyfile(legacy_results_path, results_path, 'f');
end

if exist(results_path, 'file') == 2 && ~force_rerun
    cached = load(results_path);
    required_fields = {'sumrate_grid', 'pslr_lin_grid', 'pslr_dB_grid', ...
        'islr_lin_grid', 'islr_dB_grid', 'status_grid', 'iter_grid', ...
        'runtime_grid', 'pdes_multipliers', 'CV_max_list', 'params'};
    has_required_fields = all(isfield(cached, required_fields));
    if has_required_fields && isequal(cached.pdes_multipliers, pdes_multipliers) && ...
            isequal(cached.CV_max_list, CV_max_list) && cached.params.num_mc == num_mc
        sumrate_grid = cached.sumrate_grid;
        pslr_lin_grid = cached.pslr_lin_grid;
        pslr_dB_grid = cached.pslr_dB_grid;
        islr_lin_grid = cached.islr_lin_grid;
        islr_dB_grid = cached.islr_dB_grid;
        status_grid = cached.status_grid;
        iter_grid = cached.iter_grid;
        runtime_grid = cached.runtime_grid;
        use_cached_results = true;
        fprintf('  Loaded cached sweep results from %s\n', results_path);
        fprintf('------------------------------------------------------------\n');
    end
end

if ~use_cached_results
    for mc = 1:num_mc
        rng(mc, 'twister');
        H = generate_channel(params);

        for p = 1:num_pdes
            params_case = params;
            params_case.P_des = pdes_values(p);
            alpha_warm = [];

            for c = 1:num_cv
                iter_count = iter_count + 1;
                CV_max = CV_max_list(c);
                t_iter = tic;

                if params_case.warm_start_cv
                    result = run_proposed(H, CV_max, params_case, alpha_warm);
                else
                    result = run_proposed(H, CV_max, params_case);
                end

                runtime_grid(p, c, mc) = toc(t_iter);
                status_grid(p, c, mc) = string(result.status);
                iter_grid(p, c, mc) = result.iters;

                if ~isnan(result.sumrate)
                    alpha_warm = result.alpha;
                    pslr_min_lin = min(result.pslr_per_target);
                    islr_max_lin = max(result.islr_per_target);
                    sumrate_grid(p, c, mc) = result.sumrate;
                    pslr_lin_grid(p, c, mc) = pslr_min_lin;
                    pslr_dB_grid(p, c, mc) = 10*log10(pslr_min_lin);
                    islr_lin_grid(p, c, mc) = islr_max_lin;
                    islr_dB_grid(p, c, mc) = 10*log10(islr_max_lin);
                    status_str = sprintf('%s, %d iters, %s', ...
                        result.status, result.iters, result.stop_reason);
                else
                    alpha_warm = [];
                    status_str = sprintf('FAIL (%s)', result.status);
                end

                elapsed = toc(t_global);
                eta_sec = elapsed / iter_count * (total_iters - iter_count);
                fprintf(['[%5.1f%% %3d/%3d] MC %d/%d  n=%g  CV_max=%.1f  ' ...
                         '| SR=%6.2f  PSLR=%6.2f dB  ISLR=%6.2f dB  ' ...
                         '| %s  | iter %5.1fs  elapsed %s  ETA %s\n'], ...
                        100*iter_count/total_iters, iter_count, total_iters, ...
                        mc, num_mc, pdes_multipliers(p), CV_max, ...
                        sumrate_grid(p, c, mc), pslr_dB_grid(p, c, mc), ...
                        islr_dB_grid(p, c, mc), status_str, ...
                        runtime_grid(p, c, mc), format_time(elapsed), ...
                        format_time(eta_sec));
            end
        end
    end
end

sumrate_avg = mean(sumrate_grid, 3, 'omitnan');
pslr_avg_lin = mean(pslr_lin_grid, 3, 'omitnan');
pslr_avg_dB = 10*log10(pslr_avg_lin);
islr_avg_lin = mean(islr_lin_grid, 3, 'omitnan');
islr_avg_dB = 10*log10(islr_avg_lin);
runtime_avg = mean(runtime_grid, 3, 'omitnan');

paper_fig_dir = fullfile(out_dir, 'figures');
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end

cfg = plot_config();
export_resolution = cfg.export_resolution;
plot_style = struct( ...
    'figure_position', [100 100 1040 560], ...
    'axes_position', [0.100 0.205 0.850 0.655], ...
    'legend_position', [0.200 0.215 0.600 0.140], ...
    'axes_font', cfg.axes_font, ...
    'label_font', cfg.label_font, ...
    'title_font', cfg.panel_caption_font, ...
    'legend_font', max(cfg.legend_font - 5, 1));

fig = figure('Position', plot_style.figure_position, 'Color', 'w');
set(fig, 'PaperPositionMode', 'auto');
hold on; grid on; box on;

colors = muted_pdes_colors(num_pdes);

for p = 1:num_pdes
    valid = ~isnan(sumrate_avg(p, :)) & ~isnan(pslr_avg_dB(p, :));
    if ~any(valid)
        continue;
    end

    plot(sumrate_avg(p, valid), pslr_avg_dB(p, valid), ...
        '-d', ...
        'LineWidth', cfg.line_width, 'MarkerSize', cfg.marker_size, ...
        'MarkerFaceColor', colors(p, :), ...
        'MarkerEdgeColor', colors(p, :), ...
        'Color', colors(p, :), ...
        'DisplayName', pdes_legend_label(pdes_multipliers(p)));
end

ax = gca;
set(ax, 'FontSize', plot_style.axes_font, 'LabelFontSizeMultiplier', 1, ...
        'LineWidth', cfg.axes_line_width, 'Layer', 'top', ...
        'Units', 'normalized', 'Position', plot_style.axes_position);
xlabel(ax, 'Sum-rate (bps/Hz)', 'FontSize', plot_style.label_font);
ylabel(ax, 'PSLR (dB)', 'FontSize', plot_style.label_font);
title_handle = title(ax, 'PSLR Pareto vs. P_{des}', ...
                     'Interpreter', 'tex', 'FontSize', plot_style.title_font, ...
                     'FontWeight', 'normal');
set(ax.XLabel, 'Units', 'normalized', 'Position', [0.5 -0.125 0]);
set(ax.YLabel, 'Units', 'normalized', 'Position', [-0.085 0.5 0]);

all_x = sumrate_avg(~isnan(sumrate_avg));
all_y = pslr_avg_dB(~isnan(pslr_avg_dB));
if ~isempty(all_x) && ~isempty(all_y)
    xlim(valid_axis_limits(all_x, cfg));
    ylim(valid_axis_limits(all_y, cfg));
    xtickangle(0);
end

set(ax.XLabel, 'FontSize', plot_style.label_font);
set(ax.YLabel, 'FontSize', plot_style.label_font);
set(title_handle, 'FontSize', plot_style.title_font);
set(ax, 'LooseInset', max(get(ax, 'TightInset'), [0.015 0.015 0.015 0.015]));
plot_config(fig);
set(ax, 'FontSize', plot_style.axes_font, ...
        'LabelFontSizeMultiplier', 1, ...
        'LineWidth', cfg.axes_line_width, ...
        'Layer', 'top', ...
        'Units', 'normalized', ...
        'Position', plot_style.axes_position);
set(ax.XLabel, 'FontSize', plot_style.label_font, ...
    'Units', 'normalized', 'Position', [0.5 -0.125 0]);
set(ax.YLabel, 'FontSize', plot_style.label_font, ...
    'Units', 'normalized', 'Position', [-0.085 0.5 0]);
set(title_handle, 'FontSize', plot_style.title_font, 'FontWeight', 'normal');
legend_labels = arrayfun(@pdes_legend_label, pdes_multipliers, ...
    'UniformOutput', false);
draw_pdes_legend(fig, plot_style.legend_position, legend_labels, ...
    colors, repmat({'-'}, 1, num_pdes), cfg, 0.020, 2);

tight_export_figure(fig, fullfile(out_dir, 'pdes_pareto_sweep.png'), ...
    'Resolution', export_resolution, 'TightPad', 0);
saveas(fig, fullfile(out_dir, 'pdes_pareto_sweep.fig'));
tight_export_figure(fig, fullfile(paper_fig_dir, 'pdes_pareto_sweep.pdf'), ...
    'ContentType', 'image', 'Resolution', export_resolution, ...
    'TightPad', 0);
tight_export_figure(fig, fullfile(paper_fig_dir, 'pdes_pareto_sweep.png'), ...
    'Resolution', export_resolution, 'TightPad', 0);

save(results_path, ...
     'sumrate_grid', 'pslr_lin_grid', 'pslr_dB_grid', ...
     'islr_lin_grid', 'islr_dB_grid', 'status_grid', 'iter_grid', ...
     'runtime_grid', 'sumrate_avg', 'pslr_avg_lin', 'pslr_avg_dB', ...
     'islr_avg_lin', 'islr_avg_dB', 'runtime_avg', ...
     'CV_max_list', 'pdes_multipliers', 'pdes_values', 'params');
copyfile(results_path, legacy_results_path, 'f');

fprintf('------------------------------------------------------------\n');
fprintf('  Saved: pdes_pareto_sweep.png/.fig, pdes_pareto_sweep_results.mat\n');
fprintf('  Updated: figures/pdes_pareto_sweep.pdf/.png\n');
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');

function pad_axes(x, y)
x_span = max(x) - min(x);
if x_span <= 0
    x_span = max(abs(x(1))*0.05, 1);
end

y_span = max(y) - min(y);
if y_span <= 0
    y_span = max(abs(y(1))*0.05, 1);
end

xlim([min(x) - 0.07*x_span, max(x) + 0.08*x_span]);
ylim([min(y) - 0.08*y_span, max(y) + 0.08*y_span]);
end

function label = pdes_legend_label(multiplier)
if multiplier == 0
    label = 'P_{des}=0';
elseif multiplier == 1
    label = 'P_{des}=P_{max}/N';
else
    label = sprintf('P_{des}=%dP_{max}/N', multiplier);
end
end

function colors = muted_pdes_colors(num_colors)
colors = paper_palette(1:num_colors);
end

end
