function run_integrated_stress_interior_preview(num_mc_override, force_rerun)
% RUN_INTEGRATED_STRESS_INTERIOR_PREVIEW
% Quick solver-log profiling for the integrated stress preview plot.
%
% This is intentionally lightweight. Feasibility and runtime are read from
% the MC=10 stress experiment; this script only estimates total interior-point
% iterations for the stress scenarios.

if nargin < 1 || isempty(num_mc_override)
    num_mc_override = 1;
end
if nargin < 2 || isempty(force_rerun)
    force_rerun = false;
end

clearvars -except num_mc_override force_rerun; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
out_data_dir = fullfile(sim_dir, 'results');
if exist(out_data_dir, 'dir') ~= 7, mkdir(out_data_dir); end
addpath(genpath(sim_dir));

CV_grid = 0:0.1:0.5;
num_cv = numel(CV_grid);
num_mc = num_mc_override;
scenarios = build_scenarios();
num_scenarios = numel(scenarios);

source_path = fullfile(out_data_dir, sprintf( ...
    'integrated_stress_interior_preview_MC%d.mat', num_mc));

if exist(source_path, 'file') == 2 && ~force_rerun
    fprintf('Loading cached interior profiling result: %s\n', source_path);
    plot_integrated_stress_ladder_preview();
    return;
end

prop_success = false(num_scenarios, num_cv, num_mc);
direct_success = false(num_scenarios, num_cv, num_mc);
prop_status = strings(num_scenarios, num_cv, num_mc);
direct_status = strings(num_scenarios, num_cv, num_mc);
prop_cvx_solver_iters = nan(num_scenarios, num_cv, num_mc);
direct_cvx_solver_iters = nan(num_scenarios, num_cv, num_mc);

fprintf('============================================================\n');
fprintf('  Integrated stress preview: total interior-point profiling\n');
fprintf('============================================================\n');
fprintf('  MC=%d, CV grid=[%s]\n', num_mc, num2str(CV_grid));
fprintf('  This run collects solver logs, so timing is not used here.\n');
fprintf('------------------------------------------------------------\n');

t_global = tic;
total_runs = num_scenarios * num_cv * num_mc * 2;
run_count = 0;

for s = 1:num_scenarios
    params = scenario_params(scenarios(s));
    fprintf('Scenario %d/%d: %s\n', s, num_scenarios, scenarios(s).label);

    for mc = 1:num_mc
        rng(1000*s + mc, 'twister');
        H = generate_channel(params);
        alpha0 = init_alpha_qos_safe(H, params);
        W0 = init_covariance_flat(params);

        for c = 1:num_cv
            CV_max = CV_grid(c);
            [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);

            run_count = run_count + 1;
            prop = run_proposed(H, CV_max, params, alpha0);
            prop_status(s, c, mc) = string(prop.status);
            prop_success(s, c, mc) = is_solved(prop.status);
            prop_cvx_solver_iters(s, c, mc) = prop.cvx_solver_iters;
            print_progress('CV-SDP', run_count, total_runs, scenarios(s).short, mc, num_mc, ...
                CV_max, prop.cvx_solver_iters, prop.status, t_global);

            run_count = run_count + 1;
            direct = run_direct_sca(H, pslr_min, islr_max, params, alpha0, W0);
            direct_status(s, c, mc) = string(direct.status);
            direct_success(s, c, mc) = is_solved(direct.status);
            direct_cvx_solver_iters(s, c, mc) = direct.cvx_solver_iters;
            print_progress('Direct', run_count, total_runs, scenarios(s).short, mc, num_mc, ...
                CV_max, direct.cvx_solver_iters, direct.status, t_global);
        end
    end

    save(source_path, 'CV_grid', 'num_mc', 'scenarios', ...
        'prop_success', 'direct_success', 'prop_status', 'direct_status', ...
        'prop_cvx_solver_iters', 'direct_cvx_solver_iters');
end

fprintf('------------------------------------------------------------\n');
fprintf('  Saved interior profiling data: %s\n', source_path);
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');

plot_integrated_stress_ladder_preview();
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
params.sdp_quiet = false;
params.collect_cvx_solver_log = true;
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

function print_progress(name, run_count, total_runs, scenario, mc, num_mc, CV_max, ipm_iters, status, t_global)
eta = toc(t_global) / max(run_count, 1) * max(total_runs - run_count, 0);
fprintf('[%5.1f%% %3d/%3d] %-6s %-8s MC %d/%d CV=%.1f | IPM=%5.0f | %s | ETA %s\n', ...
    100*run_count/total_runs, run_count, total_runs, name, scenario, mc, num_mc, ...
    CV_max, ipm_iters, status, format_time(eta));
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
