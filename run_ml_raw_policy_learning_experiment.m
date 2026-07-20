function run_ml_raw_policy_learning_experiment()
% RUN_ML_RAW_POLICY_LEARNING_EXPERIMENT
% Controlled label-free learning test for CV vs direct PSLR/ISLR rewards.
%
% This experiment removes the handcrafted feasibility projection used in the
% full beamforming inference baseline. A raw policy directly outputs the
% directional-power profile P_n = N softmax(z_n), and the same black-box
% evolution-strategy policy-gradient optimizer is used for both rewards.
% The goal is to isolate whether the constraint itself is easier to learn.

clear; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(sim_dir, '..', 'figures');
out_data_dir = fullfile(sim_dir, 'results');
if exist(out_data_dir, 'dir') ~= 7
    mkdir(out_data_dir);
end
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end
addpath(genpath(sim_dir));

params = setup_params();
N = params.N;
CV_grid = [0.10, 0.25, 0.50, 0.75, 1.00];
tightness = 1 - CV_grid;

num_tasks = 36;
num_iter = 90;
num_perturb = 24;
sigma_es = 0.12;
step_size = 0.18;
beta1 = 0.9;
beta2 = 0.99;
penalty_weight = 28;
snr_proxy = 12;
feas_tol = 2e-3;

fprintf('============================================================\n');
fprintf('  Raw-policy learning experiment: CV vs Direct rewards\n');
fprintf('============================================================\n');
fprintf('  N=%d, tasks/CV=%d, iterations=%d, perturbations=%d\n', ...
    N, num_tasks, num_iter, num_perturb);
fprintf('  No feasibility projection; policy output is P=N*softmax(z).\n');
fprintf('------------------------------------------------------------\n');

rng(17001, 'twister');
tasks = make_tasks(N, CV_grid, num_tasks, snr_proxy);

[trace_cv, grad_cv] = run_mode(tasks, 'cv', params, num_iter, num_perturb, ...
    sigma_es, step_size, beta1, beta2, penalty_weight, feas_tol, 18001);
[trace_direct, grad_direct] = run_mode(tasks, 'direct', params, num_iter, num_perturb, ...
    sigma_es, step_size, beta1, beta2, penalty_weight, feas_tol, 19001);

result_path = fullfile(out_data_dir, 'ml_raw_policy_learning_results.mat');
save(result_path, 'params', 'CV_grid', 'tightness', 'num_tasks', 'num_iter', ...
    'num_perturb', 'sigma_es', 'step_size', 'penalty_weight', 'snr_proxy', ...
    'feas_tol', 'trace_cv', 'trace_direct', 'grad_cv', 'grad_direct');

plot_raw_policy_learning(result_path, sim_dir, paper_fig_dir);
print_raw_policy_summary(result_path);

fprintf('------------------------------------------------------------\n');
fprintf('  Saved source data: %s\n', result_path);
fprintf('============================================================\n');
end

function tasks = make_tasks(N, CV_grid, num_tasks, snr_proxy)
tasks = repmat(struct('CV_max', [], 'gain', [], 'z0', []), numel(CV_grid) * num_tasks, 1);
idx = 0;
for c = 1:numel(CV_grid)
    for t = 1:num_tasks
        idx = idx + 1;
        g = exp(0.65 * randn(N, 1));
        g = g / mean(g);
        z0 = log(g);
        z0 = z0 - mean(z0);
        z0 = 1.5 * z0 / max(std(z0, 1), eps);
        tasks(idx).CV_max = CV_grid(c);
        tasks(idx).gain = snr_proxy * g;
        tasks(idx).z0 = z0;
    end
end
end

function [trace, grad_stats] = run_mode(tasks, mode, params, num_iter, num_perturb, ...
    sigma_es, step_size, beta1, beta2, penalty_weight, feas_tol, seed)
rng(seed, 'twister');
num_tasks = numel(tasks);
N = params.N;
Z = zeros(N, num_tasks);
for i = 1:num_tasks
    Z(:, i) = tasks(i).z0;
end
M1 = zeros(size(Z));
M2 = zeros(size(Z));

trace.reward = nan(num_iter, 1);
trace.utility = nan(num_iter, 1);
trace.feasible_utility = nan(num_iter, 1);
trace.feasibility = nan(num_iter, 1);
trace.violation = nan(num_iter, 1);
trace.policy_evals = (1:num_iter)' * num_perturb * num_tasks;

grad_stats = estimate_gradient_noise(tasks, mode, params, penalty_weight, ...
    num_perturb, sigma_es, 16, seed + 101);

fprintf('Training raw %s reward...\n', upper(mode));
for iter = 1:num_iter
    G = zeros(size(Z));
    for i = 1:num_tasks
        eps_mat = randn(N, num_perturb);
        rewards = nan(num_perturb, 1);
        for p = 1:num_perturb
            z_try = Z(:, i) + sigma_es * eps_mat(:, p);
            rewards(p) = raw_reward(z_try, tasks(i), mode, params, penalty_weight);
        end
        rewards = rewards - mean(rewards);
        G(:, i) = eps_mat * rewards / (num_perturb * sigma_es);
    end

    M1 = beta1 * M1 + (1 - beta1) * G;
    M2 = beta2 * M2 + (1 - beta2) * (G.^2);
    M1hat = M1 / (1 - beta1^iter);
    M2hat = M2 / (1 - beta2^iter);
    Z = Z + step_size * M1hat ./ (sqrt(M2hat) + 1e-8);
    Z = Z - mean(Z, 1);

    metrics = evaluate_raw_tasks(Z, tasks, mode, params, penalty_weight, feas_tol);
    trace.reward(iter) = metrics.reward;
    trace.utility(iter) = metrics.utility;
    trace.feasible_utility(iter) = metrics.feasible_utility;
    trace.feasibility(iter) = metrics.feasibility;
    trace.violation(iter) = metrics.violation;

    if mod(iter, 10) == 0 || iter == 1
        fprintf('  %s iter %03d/%03d | feas %.2f | feasible util %.3f | viol %.3g\n', ...
            upper(mode), iter, num_iter, metrics.feasibility, ...
            metrics.feasible_utility, metrics.violation);
    end
end
end

function stats = estimate_gradient_noise(tasks, mode, params, penalty_weight, num_perturb, sigma_es, num_rep, seed)
rng(seed, 'twister');
num_tasks_probe = min(numel(tasks), 20);
grad_cv = nan(num_tasks_probe, 1);
grad_var = nan(num_tasks_probe, 1);
for i = 1:num_tasks_probe
    z0 = tasks(i).z0;
    G = nan(numel(z0), num_rep);
    for r = 1:num_rep
        eps_mat = randn(numel(z0), num_perturb);
        rewards = nan(num_perturb, 1);
        for p = 1:num_perturb
            rewards(p) = raw_reward(z0 + sigma_es * eps_mat(:, p), ...
                tasks(i), mode, params, penalty_weight);
        end
        rewards = rewards - mean(rewards);
        G(:, r) = eps_mat * rewards / (num_perturb * sigma_es);
    end
    g_mean = mean(G, 2);
    grad_cv(i) = sqrt(mean(sum((G - g_mean).^2, 1))) / max(norm(g_mean), 1e-8);
    grad_var(i) = mean(sum((G - g_mean).^2, 1));
end
stats.gradient_cv = mean(grad_cv, 'omitnan');
stats.gradient_var = mean(grad_var, 'omitnan');
end

function metrics = evaluate_raw_tasks(Z, tasks, mode, params, penalty_weight, feas_tol)
num_tasks = numel(tasks);
reward = nan(num_tasks, 1);
utility = nan(num_tasks, 1);
feasible = false(num_tasks, 1);
violation = nan(num_tasks, 1);
for i = 1:num_tasks
    [reward(i), utility(i), violation(i)] = raw_reward(Z(:, i), tasks(i), mode, params, penalty_weight);
    feasible(i) = violation(i) <= feas_tol;
end
metrics.reward = mean(reward, 'omitnan');
metrics.utility = mean(utility, 'omitnan');
metrics.feasibility = mean(double(feasible));
if any(feasible)
    metrics.feasible_utility = mean(utility(feasible), 'omitnan');
else
    metrics.feasible_utility = 0;
end
metrics.violation = mean(violation, 'omitnan');
end

function [reward, utility, violation] = raw_reward(z, task, mode, params, penalty_weight)
P = z_to_power(z);
utility = mean(log2(1 + task.gain(:) .* P));
violation = raw_violation(P, task.CV_max, mode, params);
reward = utility - penalty_weight * violation.^2;
end

function violation = raw_violation(P, CV_max, mode, params)
if strcmpi(mode, 'cv')
    cv = std(P, 1) / max(mean(P), eps);
    violation = max(0, cv - CV_max) / max(CV_max, 0.05);
else
    [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);
    pslr = compute_pslr(P, params.kappa);
    islr = compute_islr(P, params.kappa);
    pslr_gap = max(0, 10*log10(pslr_min) - 10*log10(pslr)) / 10;
    islr_gap = max(0, 10*log10(islr) - 10*log10(islr_max)) / 10;
    violation = pslr_gap + islr_gap;
end
end

function P = z_to_power(z)
z = z(:) - max(z);
q = exp(z);
P = numel(z) * q / sum(q);
end

function plot_raw_policy_learning(result_path, sim_dir, paper_fig_dir)
M = load(result_path);
cv_color = [0.10 0.52 0.42];
direct_color = [0.45 0.25 0.65];

fig = figure('Position', [90 90 1120 780], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on; grid on; box on;
plot(M.trace_cv.policy_evals, M.trace_cv.feasibility, '-o', ...
    'LineWidth', 2.0, 'MarkerFaceColor', cv_color, 'Color', cv_color, ...
    'DisplayName', 'CV reward');
plot(M.trace_direct.policy_evals, M.trace_direct.feasibility, '--v', ...
    'LineWidth', 2.0, 'MarkerFaceColor', direct_color, 'Color', direct_color, ...
    'DisplayName', 'Direct reward');
yline(0.95, ':', '95%', 'LineWidth', 1.1, 'Color', [0.3 0.3 0.3]);
xlabel('Black-box policy evaluations');
ylabel('Feasibility rate');
title('(a) Feasibility learning');
ylim([0, 1.05]);
legend('Location', 'southeast');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
plot(M.trace_cv.policy_evals, M.trace_cv.violation, '-o', ...
    'LineWidth', 2.0, 'MarkerFaceColor', cv_color, 'Color', cv_color, ...
    'DisplayName', 'CV reward');
plot(M.trace_direct.policy_evals, M.trace_direct.violation, '--v', ...
    'LineWidth', 2.0, 'MarkerFaceColor', direct_color, 'Color', direct_color, ...
    'DisplayName', 'Direct reward');
xlabel('Black-box policy evaluations');
ylabel('Average normalized violation');
title('(b) Constraint violation decay');
legend('Location', 'northeast');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
plot(M.trace_cv.policy_evals, M.trace_cv.feasible_utility, '-o', ...
    'LineWidth', 2.0, 'MarkerFaceColor', cv_color, 'Color', cv_color, ...
    'DisplayName', 'CV reward');
plot(M.trace_direct.policy_evals, M.trace_direct.feasible_utility, '--v', ...
    'LineWidth', 2.0, 'MarkerFaceColor', direct_color, 'Color', direct_color, ...
    'DisplayName', 'Direct reward');
xlabel('Black-box policy evaluations');
ylabel('Feasible average utility');
title('(c) Feasible objective learning');
legend('Location', 'southeast');
set(gca, 'FontSize', 11);

nexttile;
hold on; grid on; box on;
bar_data = [M.grad_cv.gradient_cv, M.grad_direct.gradient_cv; ...
    M.grad_cv.gradient_var, M.grad_direct.gradient_var];
b = bar(bar_data);
b(1).FaceColor = cv_color;
b(2).FaceColor = direct_color;
set(gca, 'XTickLabel', {'Grad. CV', 'Grad. variance'});
ylabel('Initial ES gradient noise');
title('(d) Policy-gradient estimator noise');
legend({'CV reward', 'Direct reward'}, 'Location', 'northwest');
set(gca, 'FontSize', 11);

sgtitle(sprintf('Raw-Policy Learning Difficulty without Feasibility Projection  (N=%d)', ...
    M.params.N), 'FontSize', 14);
save_figure(fig, sim_dir, paper_fig_dir, 'ml_raw_policy_learning', ...
    'ML_Raw_Policy_Learning_Result');
end

function save_figure(fig, sim_dir, paper_fig_dir, stem, paper_stem)
set(fig, 'PaperPositionMode', 'auto');
sim_png = fullfile(sim_dir, [stem '.png']);
sim_fig = fullfile(sim_dir, [stem '.fig']);
paper_pdf = fullfile(paper_fig_dir, [paper_stem '.pdf']);
savefig(fig, sim_fig);
exportgraphics(fig, sim_png, 'Resolution', 300);
exportgraphics(fig, paper_pdf, 'ContentType', 'image', 'Resolution', 300);
end

function print_raw_policy_summary(result_path)
M = load(result_path);
fprintf('============================================================\n');
fprintf('  Raw-policy learning summary\n');
fprintf('============================================================\n');
fprintf('Final feasibility:             CV %.2f, Direct %.2f\n', ...
    M.trace_cv.feasibility(end), M.trace_direct.feasibility(end));
fprintf('Final average violation:       CV %.4g, Direct %.4g\n', ...
    M.trace_cv.violation(end), M.trace_direct.violation(end));
fprintf('Final feasible utility:        CV %.3f, Direct %.3f\n', ...
    M.trace_cv.feasible_utility(end), M.trace_direct.feasible_utility(end));
fprintf('Evals to 95%% feasibility:      CV %s, Direct %s\n', ...
    format_eval(first_eval(M.trace_cv, 0.95)), format_eval(first_eval(M.trace_direct, 0.95)));
fprintf('Initial grad. noise CV ratio:  CV %.2f, Direct %.2f\n', ...
    M.grad_cv.gradient_cv, M.grad_direct.gradient_cv);
fprintf('Initial grad. variance:        CV %.3g, Direct %.3g\n', ...
    M.grad_cv.gradient_var, M.grad_direct.gradient_var);
end

function e = first_eval(trace, target)
idx = find(trace.feasibility >= target, 1);
if isempty(idx)
    e = NaN;
else
    e = trace.policy_evals(idx);
end
end

function s = format_eval(x)
if isnan(x)
    s = 'not reached';
else
    s = sprintf('%d', round(x));
end
end
