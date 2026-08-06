function run_feasibility_stress_experiment(num_mc_override, force_rerun)
% RUN_FEASIBILITY_STRESS_EXPERIMENT
% Feasibility stress test for CV-SDP and direct PSLR/ISLR SCA.
%
% This diagnostic is not used by the manuscript directly. It compares
% algorithmic feasibility under stricter QoS, illumination, and DoF settings.

if nargin < 1 || isempty(num_mc_override)
    num_mc_override = 10;
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

CV_grid = 0:0.1:0.5;
num_cv = numel(CV_grid);
num_mc = num_mc_override;
scenarios = build_scenarios();
num_scenarios = numel(scenarios);

source_path = fullfile(out_data_dir, sprintf( ...
    'feasibility_stress_MC%d.mat', num_mc));

if exist(source_path, 'file') == 2 && ~force_rerun
    fprintf('Loading cached feasibility stress result: %s\n', source_path);
    plot_feasibility_stress_results(source_path);
    print_summary(source_path);
    return;
end

prop_success = false(num_scenarios, num_cv, num_mc);
direct_success = false(num_scenarios, num_cv, num_mc);
prop_time = nan(num_scenarios, num_cv, num_mc);
direct_time = nan(num_scenarios, num_cv, num_mc);
prop_sumrate = nan(num_scenarios, num_cv, num_mc);
direct_sumrate = nan(num_scenarios, num_cv, num_mc);
prop_status = strings(num_scenarios, num_cv, num_mc);
direct_status = strings(num_scenarios, num_cv, num_mc);
direct_inner_iters = nan(num_scenarios, num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  Feasibility Stress Test: CV-SDP vs Direct PSLR/ISLR SCA\n');
fprintf('============================================================\n');
fprintf('  MC=%d, CV grid=[%s]\n', num_mc, num2str(CV_grid));
fprintf('  Direct baseline uses explicit PSLR and ISLR SCA constraints.\n');
fprintf('  No previous-solution or cross-CV warm starts are used.\n');
fprintf('------------------------------------------------------------\n');

t_global = tic;
total_runs = num_scenarios * num_cv * num_mc * 2;
run_count = 0;

for s = 1:num_scenarios
    params = scenario_params(scenarios(s));
    fprintf('Scenario %d/%d: %s\n', s, num_scenarios, scenarios(s).label);
    fprintf('  N_T=%d, N=%d, L=%d, Q=%.2f, P_des=%.2f Pmax/N\n', ...
        params.NT, params.N, params.L, params.Q(1), params.P_des * params.N / params.P_max);

    for mc = 1:num_mc
        rng(1000*s + mc, 'twister');
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
            prop_status(s, c, mc) = string(prop.status);
            prop_success(s, c, mc) = is_solved(prop.status);
            if prop_success(s, c, mc)
                prop_sumrate(s, c, mc) = prop.sumrate;
            end
            print_progress('CV-SDP', run_count, total_runs, scenarios(s).short, mc, num_mc, ...
                CV_max, prop_time(s, c, mc), prop.status, t_global);

            run_count = run_count + 1;
            t_run = tic;
            direct = run_direct_sca(H, pslr_min, islr_max, params, alpha0, W0);
            direct_time(s, c, mc) = toc(t_run);
            direct_status(s, c, mc) = string(direct.status);
            direct_success(s, c, mc) = is_solved(direct.status);
            direct_inner_iters(s, c, mc) = direct.inner_iters;
            if direct_success(s, c, mc)
                direct_sumrate(s, c, mc) = direct.sumrate;
            end
            print_progress('Direct', run_count, total_runs, scenarios(s).short, mc, num_mc, ...
                CV_max, direct_time(s, c, mc), direct.status, t_global);
        end
    end

    save(source_path, 'CV_grid', 'num_mc', 'scenarios', ...
        'prop_success', 'direct_success', ...
        'prop_time', 'direct_time', ...
        'prop_sumrate', 'direct_sumrate', ...
        'prop_status', 'direct_status', 'direct_inner_iters');
end

plot_feasibility_stress_results(source_path);
print_summary(source_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', source_path);
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');
end

function scenarios = build_scenarios()
scenarios = struct([]);
scenarios(1).short = 'QoS+illum';
scenarios(1).label = 'Tighter QoS and illumination';
scenarios(1).NT = 4;
scenarios(1).N = 16;
scenarios(1).L = 4;
scenarios(1).theta = [-30, 0, 30, 60] * pi/180;
scenarios(1).Q = 2.0;
scenarios(1).Pdes_scale = 1.10;

scenarios(2).short = 'LowDoF';
scenarios(2).label = 'Reduced transmit DoF with tight QoS';
scenarios(2).NT = 3;
scenarios(2).N = 16;
scenarios(2).L = 4;
scenarios(2).theta = [-30, 0, 30, 60] * pi/180;
scenarios(2).Q = 1.50;
scenarios(2).Pdes_scale = 1.00;

scenarios(3).short = 'MoreTgts';
scenarios(3).label = 'More target directions';
scenarios(3).NT = 4;
scenarios(3).N = 16;
scenarios(3).L = 6;
scenarios(3).theta = linspace(-60, 60, 6) * pi/180;
scenarios(3).Q = 1.25;
scenarios(3).Pdes_scale = 0.90;
end

function params = scenario_params(scenario)
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
params.sdp_quiet = true;
params.collect_cvx_solver_log = false;
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

function print_progress(name, run_count, total_runs, scenario, mc, num_mc, CV_max, elapsed, status, t_global)
eta = toc(t_global) / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf('[%5.1f%% %3d/%3d] %-6s %-8s MC %d/%d CV=%.1f | time=%6.2fs | %s | ETA %s\n', ...
    100*run_count/total_runs, run_count, total_runs, name, scenario, mc, num_mc, CV_max, ...
    elapsed, status, format_time(eta));
end

function plot_feasibility_stress_results(source_path)
S = load(source_path);
sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, '..', 'figures');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end

CV = S.CV_grid(:);
num_scenarios = numel(S.scenarios);
prop_rate = mean(S.prop_success, 3);
direct_rate = mean(S.direct_success, 3);

prop_time_success = S.prop_time;
direct_time_success = S.direct_time;
prop_time_success(~S.prop_success) = NaN;
direct_time_success(~S.direct_success) = NaN;
prop_t = mean(prop_time_success, 3, 'omitnan');
direct_t = mean(direct_time_success, 3, 'omitnan');

fig = figure('Position', [100 100 1180 720], 'Color', 'w');
tl = tiledlayout(fig, 2, num_scenarios, 'TileSpacing', 'compact', 'Padding', 'compact');
blue = [0.18 0.43 0.78];
red = [0.82 0.22 0.18];

for s = 1:num_scenarios
    ax = nexttile(tl, s);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    plot(ax, CV, 100*prop_rate(s, :).', '-o', 'Color', blue, ...
        'MarkerFaceColor', blue, 'LineWidth', 2.0, 'DisplayName', 'CV-SDP');
    plot(ax, CV, 100*direct_rate(s, :).', '--d', 'Color', red, ...
        'MarkerFaceColor', red, 'LineWidth', 2.0, 'DisplayName', 'Direct PSLR/ISLR SCA');
    ylim(ax, [-5 105]);
    xlabel(ax, '$\mathrm{CV}_{\max}$', 'Interpreter', 'latex');
    ylabel(ax, 'Feasibility rate (%)');
    title(ax, S.scenarios(s).short, 'Interpreter', 'none');
    if s == 1
        legend(ax, 'Location', 'southwest');
    end

    ax2 = nexttile(tl, num_scenarios + s);
    hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
    plot(ax2, CV, prop_t(s, :).', '-o', 'Color', blue, ...
        'MarkerFaceColor', blue, 'LineWidth', 2.0, 'DisplayName', 'CV-SDP');
    plot(ax2, CV, direct_t(s, :).', '--d', 'Color', red, ...
        'MarkerFaceColor', red, 'LineWidth', 2.0, 'DisplayName', 'Direct PSLR/ISLR SCA');
    set(ax2, 'YScale', 'log');
    xlabel(ax2, '$\mathrm{CV}_{\max}$', 'Interpreter', 'latex');
    ylabel(ax2, 'Runtime over solved cases (s)');
end

title(tl, sprintf('Feasibility Stress Test, MC=%d', S.num_mc), 'FontSize', 14, 'FontWeight', 'bold');
out_pdf = fullfile(fig_dir, 'Feasibility_Stress_Comparison.pdf');
out_png = fullfile(fig_dir, 'Feasibility_Stress_Comparison.png');
safe_export(fig, out_pdf, 'pdf');
safe_export(fig, out_png, 'png');
fprintf('Saved feasibility stress figure: %s\n', out_pdf);
fprintf('Saved feasibility stress figure: %s\n', out_png);
end

function print_summary(source_path)
S = load(source_path);
fprintf('\nFeasibility stress summary (success over all CV points and MC runs):\n');
for s = 1:numel(S.scenarios)
    prop_rate = mean(reshape(S.prop_success(s, :, :), [], 1));
    direct_rate = mean(reshape(S.direct_success(s, :, :), [], 1));
    prop_t = S.prop_time(s, :, :);
    direct_t = S.direct_time(s, :, :);
    prop_t(~S.prop_success(s, :, :)) = NaN;
    direct_t(~S.direct_success(s, :, :)) = NaN;
    direct_iters = S.direct_inner_iters(s, :, :);
    direct_iters(~S.direct_success(s, :, :)) = NaN;
    fprintf('  %-8s | CV-SDP %5.1f%%, Direct %5.1f%% | time %.2fs vs %.2fs | direct SCA solves %.2f\n', ...
        S.scenarios(s).short, 100*prop_rate, 100*direct_rate, ...
        mean(prop_t(:), 'omitnan'), mean(direct_t(:), 'omitnan'), ...
        mean(direct_iters(:), 'omitnan'));
end
fprintf('\nBy CV target:\n');
for s = 1:numel(S.scenarios)
    fprintf('  %s\n', S.scenarios(s).short);
    for c = 1:numel(S.CV_grid)
        fprintf('    CV=%.1f: CV-SDP %5.1f%%, Direct %5.1f%%\n', ...
            S.CV_grid(c), 100*mean(S.prop_success(s, c, :)), ...
            100*mean(S.direct_success(s, c, :)));
    end
end
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        tight_export_figure(fig, filename, 'ContentType', 'image', 'Resolution', 450);
    else
        tight_export_figure(fig, filename, 'Resolution', 300);
    end
catch
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-vector');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
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
