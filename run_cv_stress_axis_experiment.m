function run_cv_stress_axis_experiment(num_mc_override, force_rerun)
% RUN_CV_STRESS_AXIS_EXPERIMENT
% Stress-axis diagnostic with CV on the vertical axis.
%
% Each panel fixes one system stress level and sweeps CV_max. The horizontal
% axes compare feasibility rate and direct/CV computational-burden ratios.

if nargin < 1 || isempty(num_mc_override)
    num_mc_override = 5;
end
if nargin < 2 || isempty(force_rerun)
    force_rerun = false;
end

clearvars -except num_mc_override force_rerun; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
out_data_dir = fullfile(sim_dir, 'results');
fig_dir = fullfile(sim_dir, '..', 'figures');
if exist(out_data_dir, 'dir') ~= 7, mkdir(out_data_dir); end
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
addpath(genpath(sim_dir));

CV_grid = 0:0.1:1.0;
num_cv = numel(CV_grid);
num_mc = num_mc_override;
scenarios = build_scenarios();
num_scenarios = numel(scenarios);

source_path = fullfile(out_data_dir, sprintf( ...
    'cv_stress_axis_S2S3S4_CV10_NT4_N16_MC%d.mat', num_mc));

if exist(source_path, 'file') == 2 && ~force_rerun
    fprintf('Loading cached CV stress-axis result: %s\n', source_path);
    plot_cv_stress_axis_results(source_path);
    print_summary(source_path);
    return;
end

prop_success = false(num_scenarios, num_cv, num_mc);
direct_success = false(num_scenarios, num_cv, num_mc);
prop_status = strings(num_scenarios, num_cv, num_mc);
direct_status = strings(num_scenarios, num_cv, num_mc);
prop_time = nan(num_scenarios, num_cv, num_mc);
direct_time = nan(num_scenarios, num_cv, num_mc);
prop_cvx_solver_iters = nan(num_scenarios, num_cv, num_mc);
direct_cvx_solver_iters = nan(num_scenarios, num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  CV Stress-Axis Diagnostic: CV-SDP vs Direct PSLR/ISLR SCA\n');
fprintf('============================================================\n');
fprintf('  MC=%d, CV grid=[%s]\n', num_mc, num2str(CV_grid));
fprintf('  Scenarios: S2 Higher Illumination, S3 More Targets, S4 Joint Stress\n');
fprintf('  N_T is fixed at 4 for all scenarios.\n');
fprintf('  Timed runs are quiet; solver-log replay is used for total IPM iterations.\n');
fprintf('------------------------------------------------------------\n');

t_global = tic;
total_runs = num_scenarios * num_cv * num_mc * 2;
run_count = 0;

for s = 1:num_scenarios
    params = scenario_params(scenarios(s), false);
    params_profile = scenario_params(scenarios(s), true);
    fprintf('Scenario %d/%d: %s\n', s, num_scenarios, scenarios(s).label);
    fprintf('  N_T=%d, N=%d, L=%d, Q=%.2f, P_des=%.2f Pmax/N\n', ...
        params.NT, params.N, params.L, params.Q(1), params.P_des * params.N / params.P_max);

    for mc = 1:num_mc
        rng(2000*s + mc, 'twister');
        H = generate_channel(params);
        alpha0 = init_alpha_qos_safe(H, params);
        W0 = init_covariance_flat(params);

        for c = 1:num_cv
            CV_max = CV_grid(c);
            [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);

            run_count = run_count + 1;
            t_run = tic;
            prop = run_proposed(H, CV_max, params, alpha0);
            prop_time(s, c, mc) = toc(t_run);
            prop_profile = run_proposed(H, CV_max, params_profile, alpha0);
            prop_cvx_solver_iters(s, c, mc) = prop_profile.cvx_solver_iters;
            prop_status(s, c, mc) = string(prop.status);
            prop_success(s, c, mc) = is_solved(prop.status);
            print_progress('CV-SDP', run_count, total_runs, scenarios(s).short, mc, num_mc, ...
                CV_max, prop_time(s, c, mc), prop_cvx_solver_iters(s, c, mc), prop.status, t_global);

            run_count = run_count + 1;
            t_run = tic;
            direct = run_direct_sca(H, pslr_min, islr_max, params, alpha0, W0);
            direct_time(s, c, mc) = toc(t_run);
            direct_profile = run_direct_sca(H, pslr_min, islr_max, params_profile, alpha0, W0);
            direct_cvx_solver_iters(s, c, mc) = direct_profile.cvx_solver_iters;
            direct_status(s, c, mc) = string(direct.status);
            direct_success(s, c, mc) = is_solved(direct.status);
            print_progress('Direct', run_count, total_runs, scenarios(s).short, mc, num_mc, ...
                CV_max, direct_time(s, c, mc), direct_cvx_solver_iters(s, c, mc), direct.status, t_global);
        end
    end

    save(source_path, 'CV_grid', 'num_mc', 'scenarios', ...
        'prop_success', 'direct_success', 'prop_status', 'direct_status', ...
        'prop_time', 'direct_time', ...
        'prop_cvx_solver_iters', 'direct_cvx_solver_iters');
end

plot_cv_stress_axis_results(source_path);
print_summary(source_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', source_path);
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');
end

function scenarios = build_scenarios()
scenarios = struct([]);

scenarios(1).id = 'S2';
scenarios(1).short = 'S2-Illum';
scenarios(1).label = 'S2 Higher Illumination';
scenarios(1).NT = 4;
scenarios(1).N = 16;
scenarios(1).L = 4;
scenarios(1).theta = [-30, 0, 30, 60] * pi/180;
scenarios(1).Q = 1.00;
scenarios(1).Pdes_scale = 1.10;

scenarios(2).id = 'S3';
scenarios(2).short = 'S3-Tgts';
scenarios(2).label = 'S3 More Targets';
scenarios(2).NT = 4;
scenarios(2).N = 16;
scenarios(2).L = 6;
scenarios(2).theta = linspace(-60, 60, 6) * pi/180;
scenarios(2).Q = 1.25;
scenarios(2).Pdes_scale = 0.90;

scenarios(3).id = 'S4';
scenarios(3).short = 'S4-Joint';
scenarios(3).label = 'S4 Joint QoS + Illumination';
scenarios(3).NT = 4;
scenarios(3).N = 16;
scenarios(3).L = 4;
scenarios(3).theta = [-30, 0, 30, 60] * pi/180;
scenarios(3).Q = 2.00;
scenarios(3).Pdes_scale = 1.10;
end

function params = scenario_params(scenario, collect_solver_log)
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
params.sdp_quiet = ~collect_solver_log;
params.collect_cvx_solver_log = collect_solver_log;
params.max_iter = 5;
params.direct_ao_max_iter = 5;
params.direct_sca_max_iter = 5;
params.direct_sca_tol = 1e-3;
end

function tf = is_solved(status)
tf = contains(string(status), 'Solved');
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

function print_progress(name, run_count, total_runs, scenario, mc, num_mc, CV_max, elapsed, ipm_iters, status, t_global)
eta = toc(t_global) / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf('[%5.1f%% %3d/%3d] %-6s %-8s MC %d/%d CV=%.1f | time=%6.2fs | IPM=%5.0f | %s | ETA %s\n', ...
    100*run_count/total_runs, run_count, total_runs, name, scenario, mc, num_mc, ...
    CV_max, elapsed, ipm_iters, status, format_time(eta));
end

function plot_cv_stress_axis_results(source_path)
S = load(source_path);
sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, '..', 'figures');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end

CV = S.CV_grid(:);
num_scenarios = numel(S.scenarios);
[cv_feas, direct_feas, runtime_ratio, ipm_ratio] = summarize_grids(S);

palette = paper_palette();
blue = palette(1, :);
red = palette(2, :);
gold = palette(3, :);
purple = palette(4, :);

fig = figure('Position', [100 100 1480 520], 'Color', 'w');
tl = tiledlayout(fig, 1, num_scenarios, 'TileSpacing', 'loose', 'Padding', 'loose');

h_all = gobjects(1, 4);
for s = 1:num_scenarios
    ax = nexttile(tl, s);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    set(ax, 'Layer', 'top', 'FontSize', 11);

    yyaxis(ax, 'left');
    h1 = plot(ax, CV, 100*cv_feas(s, :).', '-o', ...
        'Color', blue, 'MarkerFaceColor', blue, 'LineWidth', 2.2, ...
        'MarkerSize', 6.5, 'DisplayName', 'CV feasibility');
    h2 = plot(ax, CV, 100*direct_feas(s, :).', '--d', ...
        'Color', red, 'MarkerFaceColor', red, 'LineWidth', 2.2, ...
        'MarkerSize', 6.5, 'DisplayName', 'Direct feasibility');
    ylim(ax, [-5 105]);
    ylabel(ax, 'Feasibility rate (%)');
    ax.YColor = [0.10 0.10 0.10];

    yyaxis(ax, 'right');
    h3 = plot(ax, CV, runtime_ratio(s, :).', '-s', ...
        'Color', gold, 'MarkerFaceColor', gold, 'LineWidth', 2.0, ...
        'MarkerSize', 6.0, 'DisplayName', 'Runtime ratio');
    h4 = plot(ax, CV, ipm_ratio(s, :).', '-^', ...
        'Color', purple, 'MarkerFaceColor', purple, 'LineWidth', 2.0, ...
        'MarkerSize', 6.0, 'DisplayName', 'Total IPM iteration ratio');
    set([h1, h2, h3, h4], 'Clipping', 'off');
    yline(ax, 1, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
    set(ax, 'YScale', 'log');
    valid_ratio = [runtime_ratio(s, :), ipm_ratio(s, :)];
    valid_ratio = valid_ratio(isfinite(valid_ratio) & valid_ratio > 0);
    if isempty(valid_ratio)
        ylim(ax, [0.5 10]);
    else
        ylim(ax, [0.45, max(30, 1.25*max(valid_ratio))]);
    end
    ylabel(ax, 'Direct/CV burden ratio (x, log)');
    ax.YColor = [0.10 0.10 0.10];

    xlim(ax, [min(CV)-0.02 max(CV)+0.02]);
    xlabel(ax, '$\mathrm{CV}_{\max}$', 'Interpreter', 'latex');
    if s == 1
        yyaxis(ax, 'left');
        ylabel(ax, 'Feasibility rate (%)');
    else
        yyaxis(ax, 'left');
        ylabel(ax, '');
    end
    yyaxis(ax, 'right');
    if s ~= num_scenarios
        ylabel(ax, '');
    end

    panel_title = sprintf('%s\n$L=%d$, $Q=%.2f$, $P_{\\rm des}=%.2fP_{\\max}/N$', ...
        display_label(S.scenarios(s)), S.scenarios(s).L, S.scenarios(s).Q, S.scenarios(s).Pdes_scale);
    title(ax, panel_title, 'Interpreter', 'latex', 'FontSize', 11, 'FontWeight', 'bold');

    if s == 1
        h_all = [h1, h2, h3, h4];
    end
end

leg = legend(h_all, {'CV feasibility', 'Direct feasibility', ...
    'Runtime ratio', 'Total IPM iteration ratio'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 4);
leg.Layout.Tile = 'south';

out_png = fullfile(fig_dir, 'CV_Stress_Axis_Diagnostic.png');
out_pdf = fullfile(fig_dir, 'CV_Stress_Axis_Diagnostic.pdf');
safe_export(fig, out_png, 'png');
safe_export(fig, out_pdf, 'pdf');
fprintf('Saved CV stress-axis figure: %s\n', out_pdf);
fprintf('Saved CV stress-axis figure: %s\n', out_png);
end

function label = display_label(scenario)
switch string(scenario.id)
    case "S2"
        label = 'S2 Higher Illum.';
    case "S3"
        label = 'S3 More Targets';
    case "S4"
        label = 'S4 Joint Stress';
    otherwise
        label = scenario.label;
end
end

function [cv_feas, direct_feas, runtime_ratio, ipm_ratio] = summarize_grids(S)
num_scenarios = numel(S.scenarios);
num_cv = numel(S.CV_grid);
cv_feas = nan(num_scenarios, num_cv);
direct_feas = nan(num_scenarios, num_cv);
runtime_ratio = nan(num_scenarios, num_cv);
ipm_ratio = nan(num_scenarios, num_cv);

for s = 1:num_scenarios
    for c = 1:num_cv
        cv_ok = squeeze(S.prop_success(s, c, :));
        direct_ok = squeeze(S.direct_success(s, c, :));
        cv_feas(s, c) = mean(cv_ok(:));
        direct_feas(s, c) = mean(direct_ok(:));

        cv_t = squeeze(S.prop_time(s, c, :));
        direct_t = squeeze(S.direct_time(s, c, :));
        runtime_ratio(s, c) = mean(direct_t(:), 'omitnan') / mean(cv_t(:), 'omitnan');

        cv_ipm = squeeze(S.prop_cvx_solver_iters(s, c, :));
        direct_ipm = squeeze(S.direct_cvx_solver_iters(s, c, :));
        ipm_ratio(s, c) = mean(direct_ipm(:), 'omitnan') / mean(cv_ipm(:), 'omitnan');
    end
end
end

function print_summary(source_path)
S = load(source_path);
[cv_feas, direct_feas, runtime_ratio, ipm_ratio] = summarize_grids(S);
fprintf('\nStress-axis summary by scenario:\n');
for s = 1:numel(S.scenarios)
    fprintf('  %s | CV feas %.1f%%, Direct feas %.1f%% | runtime %.1fx | IPM %.1fx\n', ...
        S.scenarios(s).short, 100*mean(cv_feas(s, :), 'omitnan'), ...
        100*mean(direct_feas(s, :), 'omitnan'), ...
        mean(runtime_ratio(s, :), 'omitnan'), mean(ipm_ratio(s, :), 'omitnan'));
end
fprintf('\nBy CV target:\n');
for s = 1:numel(S.scenarios)
    fprintf('  %s\n', S.scenarios(s).short);
    for c = 1:numel(S.CV_grid)
        fprintf('    CV=%.1f: feasibility CV %.0f%%, Direct %.0f%% | runtime %.1fx | IPM %.1fx\n', ...
            S.CV_grid(c), 100*cv_feas(s, c), 100*direct_feas(s, c), ...
            runtime_ratio(s, c), ipm_ratio(s, c));
    end
end
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        tmp_png = [tempname(fileparts(filename)), '.png'];
        exportgraphics(fig, tmp_png, 'Resolution', 300);
        if ~png_to_pdf(tmp_png, filename)
            exportgraphics(fig, filename, 'ContentType', 'image', 'Resolution', 450);
        end
        if exist(tmp_png, 'file') == 2
            delete(tmp_png);
        end
    else
        exportgraphics(fig, filename, 'Resolution', 300);
    end
catch
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-r300');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
end

function ok = png_to_pdf(png_file, pdf_file)
ok = false;
python_exe = fullfile(getenv('USERPROFILE'), '.cache', 'codex-runtimes', ...
    'codex-primary-runtime', 'dependencies', 'python', 'python.exe');
if exist(python_exe, 'file') ~= 2
    return;
end
py_code = sprintf(['from PIL import Image; ', ...
    'img=Image.open(r''%s'').convert(''RGB''); ', ...
    'img.save(r''%s'', ''PDF'', resolution=300.0)'], png_file, pdf_file);
cmd = sprintf('"%s" -c "%s"', python_exe, py_code);
[status, ~] = system(cmd);
ok = (status == 0) && exist(pdf_file, 'file') == 2;
end

function s = format_time(sec)
if ~isfinite(sec) || sec < 0
    s = 'n/a';
elseif sec < 60
    s = sprintf('%.0fs', sec);
elseif sec < 3600
    s = sprintf('%.1fmin', sec/60);
else
    s = sprintf('%.1fh', sec/3600);
end
end
