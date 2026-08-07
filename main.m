%% MAIN  Pareto curve simulation for the proposed CV-constrained ISAC
%
% Sweeps CV_max in {0, 0.1, ..., 0.9}, runs the AO algorithm, and plots
% the resulting (sum-rate, PSLR) and (sum-rate, ISLR) trade-offs. Averages
% across Monte Carlo channel realizations.
%
% Requires CVX (http://cvxr.com/cvx) on the MATLAB path.

clear; close all; clc;

% --- path & CVX check ---
addpath(genpath(fileparts(mfilename('fullpath'))));
if exist('cvx_begin', 'file') ~= 2
    error('CVX is required. Install from http://cvxr.com/cvx and add to path.');
end

% --- setup ---
params = setup_params();

fprintf('============================================================\n');
fprintf('  MISO-OFDM ISAC: Pareto Curve for the Proposed Algorithm\n');
fprintf('============================================================\n');
fprintf('  K = %d users,  L = %d targets at [%s] deg\n', ...
        params.K, params.L, num2str(rad2deg(params.theta)));
fprintf('  N_T = %d antennas (ULA, d=lambda/2 @ %.1f GHz)\n', ...
        params.NT, params.fc/1e9);
fprintf('  N = %d subcarriers\n', params.N);
fprintf('  P_max = %.2f, sigma^2 = %.0e, P_des = %.4f\n', ...
        params.P_max, params.sigma2, params.P_des);
fprintf('  QoS floor per user = [%s] bps/Hz\n', num2str(params.Q.'));
fprintf('  kappa = %.2f (16-QAM)\n', params.kappa);
fprintf('  AO max_iter = %d, MC runs = %d\n', params.max_iter, params.num_mc);
fprintf('  Direct SCA baseline = %d (ISLR-active overlay + joint PSLR-active SCA)\n', ...
        params.run_direct_baseline);
fprintf('------------------------------------------------------------\n');

CV_max_list = params.CV_max_list;
num_cv      = numel(CV_max_list);
num_mc      = params.num_mc;

sumrate_grid = nan(num_cv, num_mc);
pslr_lin_grid = nan(num_cv, num_mc);
islr_lin_grid = nan(num_cv, num_mc);
pslr_dB_grid = nan(num_cv, num_mc);
islr_dB_grid = nan(num_cv, num_mc);
rank_eig2eig1_max_grid  = nan(num_cv, num_mc);
rank_eig2eig1_mean_grid = nan(num_cv, num_mc);
rank_toptrace_min_grid  = nan(num_cv, num_mc);
rank_toptrace_mean_grid = nan(num_cv, num_mc);
proposed_time_grid = nan(num_cv, num_mc);

% direct_* stores the PSLR-active Algorithm 2 curve; the ISLR-active curve
% is analytically equivalent to the proposed CV solution and is overlaid.
direct_sumrate_grid = nan(num_cv, num_mc);
direct_pslr_lin_grid = nan(num_cv, num_mc);
direct_islr_lin_grid = nan(num_cv, num_mc);
direct_pslr_dB_grid = nan(num_cv, num_mc);
direct_islr_dB_grid = nan(num_cv, num_mc);
direct_pslr_min_grid = nan(num_cv, num_mc);
direct_islr_max_grid = nan(num_cv, num_mc);
direct_islr_cv_grid = nan(num_cv, num_mc);
direct_inner_iter_grid = nan(num_cv, num_mc);
direct_time_grid = nan(num_cv, num_mc);

direct_equiv_sumrate_grid = nan(num_cv, num_mc);
direct_equiv_pslr_lin_grid = nan(num_cv, num_mc);
direct_equiv_islr_lin_grid = nan(num_cv, num_mc);
direct_equiv_pslr_dB_grid = nan(num_cv, num_mc);
direct_equiv_islr_dB_grid = nan(num_cv, num_mc);
direct_equiv_pslr_min_grid = nan(num_cv, num_mc);
direct_equiv_islr_max_grid = nan(num_cv, num_mc);

runs_per_point = 1 + double(params.run_direct_baseline);
total_iters = num_cv * num_mc * runs_per_point;
iter_count  = 0;
t_global    = tic;

for mc = 1:num_mc
    rng(mc, 'twister');                 % reproducible per MC realization
    H = generate_channel(params);
    alpha_warm = [];
    direct_alpha_warm = [];
    direct_W_warm = [];

    for c = 1:num_cv
        CV_max     = CV_max_list(c);
        t_iter     = tic;

        iter_count = iter_count + 1;
        if params.warm_start_cv
            result = run_proposed(H, CV_max, params, alpha_warm);
        else
            result = run_proposed(H, CV_max, params);
        end

        if ~isnan(result.sumrate)
            pslr_min_lin = min(result.pslr_per_target);  % worst-case PSLR (lower is worse)
            islr_max_lin = max(result.islr_per_target);  % worst-case ISLR (higher is worse)
            sumrate_grid(c, mc)            = result.sumrate;
            pslr_lin_grid(c, mc)           = pslr_min_lin;
            islr_lin_grid(c, mc)           = islr_max_lin;
            pslr_dB_grid(c, mc)            = 10*log10(pslr_min_lin);
            islr_dB_grid(c, mc)            = 10*log10(islr_max_lin);
            rank_eig2eig1_max_grid(c, mc)  = result.rank_stats.eig2_over_eig1_max;
            rank_eig2eig1_mean_grid(c, mc) = result.rank_stats.eig2_over_eig1_mean;
            rank_toptrace_min_grid(c, mc)  = result.rank_stats.top_to_trace_min;
            rank_toptrace_mean_grid(c, mc) = result.rank_stats.top_to_trace_mean;
            if params.warm_start_cv
                alpha_warm = result.alpha;
            end
            if params.run_direct_baseline
                [equiv_pslr_min, equiv_islr_max] = direct_thresholds_from_cv(CV_max, params);
                % ISLR(theta)<=ISLR_max is exactly CV(theta)<=CV_max. When
                % the PSLR requirement is looser than L(ISLR^{-1}(ISLR_max)),
                % Algorithm 2 has the same optimum as the proposed problem.
                direct_equiv_sumrate_grid(c, mc) = result.sumrate;
                direct_equiv_pslr_lin_grid(c, mc) = pslr_min_lin;
                direct_equiv_islr_lin_grid(c, mc) = islr_max_lin;
                direct_equiv_pslr_dB_grid(c, mc) = 10*log10(pslr_min_lin);
                direct_equiv_islr_dB_grid(c, mc) = 10*log10(islr_max_lin);
                direct_equiv_pslr_min_grid(c, mc) = equiv_pslr_min;
                direct_equiv_islr_max_grid(c, mc) = equiv_islr_max;
            end
            status_str = sprintf('%s, %d iters, %s', ...
                                 result.status, result.iters, result.stop_reason);
        else
            alpha_warm = [];
            status_str = sprintf('FAIL (%s)', result.status);
        end

        prop_elapsed = toc(t_iter);
        proposed_time_grid(c, mc) = prop_elapsed;

        % --- progress line ---
        elapsed  = toc(t_global);
        eta_sec  = elapsed / iter_count * (total_iters - iter_count);
        progress = iter_count / total_iters * 100;
        fprintf(['[%5.1f%% %3d/%3d]  Proposed  MC %d/%d  CV_max=%.1f  ' ...
                 '|  SR=%6.2f  PSLR=%6.2f dB  ISLR=%6.2f dB  ' ...
                 '|  rank(eig2/eig1 max=%.1e, top/tr min=%.4f)  ' ...
                 '|  %s  |  iter %5.1fs  elapsed %s  ETA %s\n'], ...
                progress, iter_count, total_iters, mc, num_mc, CV_max, ...
                sumrate_grid(c, mc), pslr_dB_grid(c, mc), islr_dB_grid(c, mc), ...
                rank_eig2eig1_max_grid(c, mc), rank_toptrace_min_grid(c, mc), ...
                status_str, prop_elapsed, ...
                format_time(elapsed), format_time(eta_sec));

        if params.run_direct_baseline
            iter_count = iter_count + 1;
            t_direct = tic;

            if ~isnan(result.sumrate)
                if strcmpi(params.direct_pslr_target_mode, 'proposed')
                    pslr_min = min(result.pslr_per_target) * ...
                        (1 - params.direct_pslr_target_relax);
                    islr_cv = CV_max + params.direct_pslr_active_islr_cv_gap;
                    [~, islr_max] = direct_thresholds_from_cv(islr_cv, params);
                else
                    [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params);
                    islr_cv = CV_max;
                    if isfield(params, 'direct_islr_max')
                        islr_max = params.direct_islr_max;
                        islr_cv = NaN;
                    end
                end
                direct_pslr_min_grid(c, mc) = pslr_min;
                direct_islr_max_grid(c, mc) = islr_max;
                direct_islr_cv_grid(c, mc) = islr_cv;

                if isempty(direct_W_warm)
                    direct_alpha0 = result.alpha;
                    direct_W0 = result.W;
                else
                    direct_alpha0 = direct_alpha_warm;
                    direct_W0 = direct_W_warm;
                end

                direct_result = run_direct_sca(H, pslr_min, params, direct_alpha0, direct_W0);
            else
                direct_result.status = 'Skipped: proposed warm start failed';
                direct_result.sumrate = NaN;
            end

            if ~isnan(direct_result.sumrate)
                pslr_min_lin = min(direct_result.pslr_per_target);
                islr_max_lin = max(direct_result.islr_per_target);
                direct_sumrate_grid(c, mc) = direct_result.sumrate;
                direct_pslr_lin_grid(c, mc) = pslr_min_lin;
                direct_islr_lin_grid(c, mc) = islr_max_lin;
                direct_pslr_dB_grid(c, mc) = 10*log10(pslr_min_lin);
                direct_islr_dB_grid(c, mc) = 10*log10(islr_max_lin);
                direct_inner_iter_grid(c, mc) = direct_result.inner_iters;
                direct_alpha_warm = direct_result.alpha;
                direct_W_warm = direct_result.W;
                direct_status_str = sprintf('%s, %d AO, %d SCA, %s', ...
                    direct_result.status, direct_result.iters, ...
                    direct_result.inner_iters, direct_result.stop_reason);
            else
                direct_alpha_warm = [];
                direct_W_warm = [];
                direct_status_str = sprintf('FAIL (%s)', direct_result.status);
            end

            direct_elapsed = toc(t_direct);
            direct_time_grid(c, mc) = direct_elapsed;
            elapsed  = toc(t_global);
            eta_sec  = elapsed / iter_count * (total_iters - iter_count);
            progress = iter_count / total_iters * 100;
            if isfinite(direct_islr_max_grid(c, mc))
                target_islr_str = sprintf('%6.2f dB', 10*log10(direct_islr_max_grid(c, mc)));
            else
                target_islr_str = ' loose ';
            end
            fprintf(['[%5.1f%% %3d/%3d]  DirectSCA MC %d/%d  CV_max=%.1f  ' ...
                     '|  target PSLR=%6.2f dB  target ISLR=%s  ' ...
                     '|  SR=%6.2f  PSLR=%6.2f dB  ISLR=%6.2f dB  ' ...
                     '|  %s  |  iter %5.1fs  elapsed %s  ETA %s\n'], ...
                    progress, iter_count, total_iters, mc, num_mc, CV_max, ...
                    10*log10(direct_pslr_min_grid(c, mc)), target_islr_str, ...
                    direct_sumrate_grid(c, mc), direct_pslr_dB_grid(c, mc), ...
                    direct_islr_dB_grid(c, mc), direct_status_str, ...
                    direct_elapsed, format_time(elapsed), format_time(eta_sec));
        end
    end
end

% --- average across MC ---
sumrate_avg = mean(sumrate_grid, 2, 'omitnan');
pslr_avg_lin = mean(pslr_lin_grid, 2, 'omitnan');
islr_avg_lin = mean(islr_lin_grid, 2, 'omitnan');
pslr_avg_dB = 10*log10(pslr_avg_lin);
islr_avg_dB = 10*log10(islr_avg_lin);

direct_sumrate_avg = mean(direct_sumrate_grid, 2, 'omitnan');
direct_pslr_avg_lin = mean(direct_pslr_lin_grid, 2, 'omitnan');
direct_islr_avg_lin = mean(direct_islr_lin_grid, 2, 'omitnan');
direct_pslr_avg_dB = 10*log10(direct_pslr_avg_lin);
direct_islr_avg_dB = 10*log10(direct_islr_avg_lin);

direct_equiv_sumrate_avg = mean(direct_equiv_sumrate_grid, 2, 'omitnan');
direct_equiv_pslr_avg_lin = mean(direct_equiv_pslr_lin_grid, 2, 'omitnan');
direct_equiv_islr_avg_lin = mean(direct_equiv_islr_lin_grid, 2, 'omitnan');
direct_equiv_pslr_avg_dB = 10*log10(direct_equiv_pslr_avg_lin);
direct_equiv_islr_avg_dB = 10*log10(direct_equiv_islr_avg_lin);

title_str = sprintf('(K=%d, L=%d, N_T=%d, N=%d)', ...
                    params.K, params.L, params.NT, params.N);
out_dir = fileparts(mfilename('fullpath'));
paper_fig_dir = fullfile(out_dir, '..', 'figures');
if exist(paper_fig_dir, 'dir') ~= 7
    mkdir(paper_fig_dir);
end

% --- PSLR Pareto curve ---
fig = figure('Position', [100 100 850 620], 'Color', 'w');
hold on; grid on; box on;
valid_pslr = ~isnan(sumrate_avg) & ~isnan(pslr_avg_dB);
valid_direct_equiv_pslr = params.run_direct_baseline & ...
    ~isnan(direct_equiv_sumrate_avg) & ~isnan(direct_equiv_pslr_avg_dB);
valid_direct_pslr = params.run_direct_baseline & ...
    ~isnan(direct_sumrate_avg) & ~isnan(direct_pslr_avg_dB);
h_prop_pslr = plot(sumrate_avg(valid_pslr), pslr_avg_dB(valid_pslr), '-o', ...
     'LineWidth', 2.0, 'MarkerSize', 9, ...
     'MarkerFaceColor', [0.20 0.45 0.80], ...
     'Color', [0.20 0.45 0.80], ...
     'DisplayName', 'Proposed CV');
h_active_pslr = plot(direct_sumrate_avg(valid_direct_pslr), direct_pslr_avg_dB(valid_direct_pslr), '--d', ...
     'LineWidth', 2.0, 'MarkerSize', 8, ...
     'MarkerFaceColor', [0.85 0.25 0.20], ...
     'Color', [0.85 0.25 0.20], ...
     'DisplayName', 'Alg. 2 joint PSLR-active');
h_equiv_pslr = plot(direct_equiv_sumrate_avg(valid_direct_equiv_pslr), ...
     direct_equiv_pslr_avg_dB(valid_direct_equiv_pslr), 'o', ...
     'LineStyle', 'none', 'LineWidth', 1.7, 'MarkerSize', 14, ...
     'MarkerFaceColor', 'none', 'Color', [0.10 0.10 0.10], ...
     'DisplayName', 'Alg. 2 ISLR-active (overlap)');
if any(valid_pslr)
    x_all = [sumrate_avg(valid_pslr); ...
             direct_equiv_sumrate_avg(valid_direct_equiv_pslr); ...
             direct_sumrate_avg(valid_direct_pslr)];
    y_all = [pslr_avg_dB(valid_pslr); ...
             direct_equiv_pslr_avg_dB(valid_direct_equiv_pslr); ...
             direct_pslr_avg_dB(valid_direct_pslr)];
    [label_dx, label_dy] = pad_axes_for_labels(x_all, y_all);
    for c = 1:num_cv
        if valid_pslr(c)
            text(sumrate_avg(c) + label_dx, pslr_avg_dB(c) + label_dy, ...
                 sprintf('CV_{max}=%.1f', CV_max_list(c)), ...
                 'FontSize', 10, 'Color', [0.55 0.10 0.10], ...
                 'Interpreter', 'tex');
        end
    end
end
xlabel('Sum-rate (bps/Hz)', 'FontSize', 13);
ylabel('PSLR (dB, worst-case across targets)', 'FontSize', 13);
title(['PSLR Pareto  ' title_str], 'FontSize', 13);
legend([h_prop_pslr h_equiv_pslr h_active_pslr], 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 12);

saveas(fig, fullfile(out_dir, 'pareto_curve.png'));
saveas(fig, fullfile(out_dir, 'pareto_curve.fig'));
tight_export_figure(fig, fullfile(paper_fig_dir, 'Pareto_Frontier_Result.pdf'), 'ContentType', 'vector');
tight_export_figure(fig, fullfile(paper_fig_dir, 'Pareto_Frontier_Result.png'), 'Resolution', 300);

% --- ISLR Pareto curve ---
fig_islr = figure('Position', [120 120 850 620], 'Color', 'w');
hold on; grid on; box on;
valid_islr = ~isnan(sumrate_avg) & ~isnan(islr_avg_dB);
valid_direct_equiv_islr = params.run_direct_baseline & ...
    ~isnan(direct_equiv_sumrate_avg) & ~isnan(direct_equiv_islr_avg_dB);
valid_direct_islr = params.run_direct_baseline & ...
    ~isnan(direct_sumrate_avg) & ~isnan(direct_islr_avg_dB);
h_prop_islr = plot(sumrate_avg(valid_islr), islr_avg_dB(valid_islr), '-s', ...
     'LineWidth', 2.0, 'MarkerSize', 9, ...
     'MarkerFaceColor', [0.80 0.45 0.20], ...
     'Color', [0.80 0.45 0.20], ...
     'DisplayName', 'Proposed CV');
h_active_islr = plot(direct_sumrate_avg(valid_direct_islr), direct_islr_avg_dB(valid_direct_islr), '--d', ...
     'LineWidth', 2.0, 'MarkerSize', 8, ...
     'MarkerFaceColor', [0.20 0.55 0.40], ...
     'Color', [0.20 0.55 0.40], ...
     'DisplayName', 'Alg. 2 joint PSLR-active');
h_equiv_islr = plot(direct_equiv_sumrate_avg(valid_direct_equiv_islr), ...
     direct_equiv_islr_avg_dB(valid_direct_equiv_islr), 'o', ...
     'LineStyle', 'none', 'LineWidth', 1.7, 'MarkerSize', 14, ...
     'MarkerFaceColor', 'none', 'Color', [0.10 0.10 0.10], ...
     'DisplayName', 'Alg. 2 ISLR-active (overlap)');
if any(valid_islr)
    x_all = [sumrate_avg(valid_islr); ...
             direct_equiv_sumrate_avg(valid_direct_equiv_islr); ...
             direct_sumrate_avg(valid_direct_islr)];
    y_all = [islr_avg_dB(valid_islr); ...
             direct_equiv_islr_avg_dB(valid_direct_equiv_islr); ...
             direct_islr_avg_dB(valid_direct_islr)];
    [label_dx, label_dy] = pad_axes_for_labels(x_all, y_all);
    for c = 1:num_cv
        if valid_islr(c)
            text(sumrate_avg(c) + label_dx, islr_avg_dB(c) + label_dy, ...
                 sprintf('CV_{max}=%.1f', CV_max_list(c)), ...
                 'FontSize', 10, 'Color', [0.10 0.30 0.55], ...
                 'Interpreter', 'tex');
        end
    end
end
xlabel('Sum-rate (bps/Hz)', 'FontSize', 13);
ylabel('ISLR (dB, worst-case across targets)', 'FontSize', 13);
title(['ISLR Pareto  ' title_str], 'FontSize', 13);
legend([h_prop_islr h_equiv_islr h_active_islr], 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 12);

saveas(fig_islr, fullfile(out_dir, 'pareto_curve_islr.png'));
saveas(fig_islr, fullfile(out_dir, 'pareto_curve_islr.fig'));
tight_export_figure(fig_islr, fullfile(paper_fig_dir, 'ISLR_Pareto_Frontier_Result.pdf'), 'ContentType', 'vector');
tight_export_figure(fig_islr, fullfile(paper_fig_dir, 'ISLR_Pareto_Frontier_Result.png'), 'Resolution', 300);

% --- runtime comparison ---
fig_time = figure('Position', [140 140 820 560], 'Color', 'w');
hold on; grid on; box on;
prop_time_avg = mean(proposed_time_grid, 2, 'omitnan');
direct_time_avg = mean(direct_time_grid, 2, 'omitnan');
plot(CV_max_list, prop_time_avg, '-o', 'LineWidth', 2.0, 'MarkerSize', 8, ...
     'MarkerFaceColor', [0.20 0.45 0.80], 'Color', [0.20 0.45 0.80], ...
     'DisplayName', 'Proposed CV');
plot(CV_max_list, direct_time_avg, '--d', 'LineWidth', 2.0, 'MarkerSize', 8, ...
     'MarkerFaceColor', [0.85 0.25 0.20], 'Color', [0.85 0.25 0.20], ...
     'DisplayName', 'Direct PSLR/ISLR SCA');
set(gca, 'YScale', 'log', 'FontSize', 12);
xlabel('CV_{max}', 'FontSize', 13, 'Interpreter', 'tex');
ylabel('Runtime per point (s)', 'FontSize', 13);
title(['Runtime comparison  ' title_str], 'FontSize', 13);
legend('Location', 'best', 'FontSize', 11);
saveas(fig_time, fullfile(out_dir, 'comparison_time.png'));
saveas(fig_time, fullfile(out_dir, 'comparison_time.fig'));
tight_export_figure(fig_time, fullfile(paper_fig_dir, 'comparison_time.pdf'), 'ContentType', 'vector');
tight_export_figure(fig_time, fullfile(paper_fig_dir, 'comparison_time.png'), 'Resolution', 300);

save(fullfile(out_dir, 'results.mat'), ...
     'sumrate_grid', 'pslr_lin_grid', 'islr_lin_grid', ...
     'pslr_dB_grid', 'islr_dB_grid', 'CV_max_list', 'params', ...
     'direct_equiv_sumrate_grid', 'direct_equiv_pslr_lin_grid', ...
     'direct_equiv_islr_lin_grid', 'direct_equiv_pslr_dB_grid', ...
     'direct_equiv_islr_dB_grid', ...
     'direct_equiv_pslr_min_grid', 'direct_equiv_islr_max_grid', ...
     'direct_sumrate_grid', 'direct_pslr_lin_grid', 'direct_islr_lin_grid', ...
     'direct_pslr_dB_grid', 'direct_islr_dB_grid', ...
     'direct_pslr_min_grid', 'direct_islr_max_grid', 'direct_islr_cv_grid', ...
     'direct_inner_iter_grid', ...
     'rank_eig2eig1_max_grid', 'rank_eig2eig1_mean_grid', ...
     'rank_toptrace_min_grid', 'rank_toptrace_mean_grid', ...
     'proposed_time_grid', 'direct_time_grid');

% --- SDR tightness summary ---
fprintf('------------------------------------------------------------\n');
fprintf('  SDR tightness audit (Proposition 1):\n');
all_max  = rank_eig2eig1_max_grid(:);  all_max  = all_max(~isnan(all_max));
all_mean = rank_eig2eig1_mean_grid(:); all_mean = all_mean(~isnan(all_mean));
all_top_min  = rank_toptrace_min_grid(:);  all_top_min  = all_top_min(~isnan(all_top_min));
all_top_mean = rank_toptrace_mean_grid(:); all_top_mean = all_top_mean(~isnan(all_top_mean));
if isempty(all_max)
    fprintf('    No successful SDP solves; rank statistics are unavailable.\n');
else
    fprintf('    worst lambda_2/lambda_1 across all (CV_max, mc, n):  %.3e\n', max(all_max));
    fprintf('    mean  lambda_2/lambda_1 across all (CV_max, mc, n):  %.3e\n', mean(all_mean));
    fprintf('    worst lambda_1/trace      (>=1 means rank-1 exact):  %.6f\n', min(all_top_min));
    fprintf('    mean  lambda_1/trace      :                           %.6f\n', mean(all_top_mean));
    if max(all_max) < 1e-4
        fprintf('    ==> SDR tightness HOLDS (W_n is numerically rank-1 everywhere).\n');
    else
        fprintf('    ==> Some W_n have non-trivial second eigenvalue; inspect grids.\n');
    end
end

fprintf('------------------------------------------------------------\n');
fprintf('  Saved: pareto_curve.png/.fig, pareto_curve_islr.png/.fig, comparison_time.png/.fig, results.mat\n');
fprintf('  Updated paper figures in ..\\figures\\*.pdf\n');
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');

function [label_dx, label_dy] = pad_axes_for_labels(x, y)
%PAD_AXES_FOR_LABELS Add right-side room for CV labels and return offsets.

x_span = max(x) - min(x);
if x_span <= 0
    x_span = max(abs(x(1)) * 0.05, 1);
end

y_span = max(y) - min(y);
if y_span <= 0
    y_span = max(abs(y(1)) * 0.05, 1);
end

xlim([min(x) - 0.08*x_span, max(x) + 0.35*x_span]);
ylim([min(y) - 0.12*y_span, max(y) + 0.12*y_span]);

label_dx = 0.025 * x_span;
label_dy = 0.015 * y_span;
end


