%% APPEND_CV1_RESULTS  Add CV_max=1.0 proposed samples to results.mat.
%
% This keeps the existing sweep data and computes only the missing
% CV_max=1.0 Monte Carlo points used by plot_cv_theory_bounds.m.

clear; close all; clc;

addpath(genpath(fileparts(mfilename('fullpath'))));
if exist('cvx_begin', 'file') ~= 2
    error('CVX is required. Install CVX and run cvx_setup first.');
end

out_dir = fileparts(mfilename('fullpath'));
results_path = fullfile(out_dir, 'results.mat');
if exist(results_path, 'file') ~= 2
    error('results.mat not found. Run main.m before appending CV_max=1.0.');
end

S = load(results_path);
target_cv = 1.0;

if any(abs(S.CV_max_list - target_cv) < 1e-12)
    fprintf('CV_max=%.1f already exists in results.mat. Nothing to append.\n', target_cv);
    return;
end

params = S.params;
params.CV_max_list = [S.CV_max_list, target_cv];

num_mc = params.num_mc;
new_sumrate = nan(1, num_mc);
new_pslr_lin = nan(1, num_mc);
new_islr_lin = nan(1, num_mc);
new_pslr_dB = nan(1, num_mc);
new_islr_dB = nan(1, num_mc);
new_rank_eig2eig1_max = nan(1, num_mc);
new_rank_eig2eig1_mean = nan(1, num_mc);
new_rank_toptrace_min = nan(1, num_mc);
new_rank_toptrace_mean = nan(1, num_mc);
new_time = nan(1, num_mc);

fprintf('============================================================\n');
fprintf('  Append CV_max=%.1f Proposed Samples\n', target_cv);
fprintf('============================================================\n');

t_global = tic;
for mc = 1:num_mc
    rng(mc, 'twister');
    H = generate_channel(params);

    t_iter = tic;
    result = run_proposed(H, target_cv, params);
    new_time(mc) = toc(t_iter);

    if ~isnan(result.sumrate)
        pslr_min_lin = min(result.pslr_per_target);
        islr_max_lin = max(result.islr_per_target);

        new_sumrate(mc) = result.sumrate;
        new_pslr_lin(mc) = pslr_min_lin;
        new_islr_lin(mc) = islr_max_lin;
        new_pslr_dB(mc) = 10*log10(pslr_min_lin);
        new_islr_dB(mc) = 10*log10(islr_max_lin);
        new_rank_eig2eig1_max(mc) = result.rank_stats.eig2_over_eig1_max;
        new_rank_eig2eig1_mean(mc) = result.rank_stats.eig2_over_eig1_mean;
        new_rank_toptrace_min(mc) = result.rank_stats.top_to_trace_min;
        new_rank_toptrace_mean(mc) = result.rank_stats.top_to_trace_mean;
        status_str = sprintf('%s, %d iters, %s', ...
            result.status, result.iters, result.stop_reason);
    else
        status_str = sprintf('FAIL (%s)', result.status);
    end

    fprintf(['  MC %2d/%2d  CV_max=%.1f  SR=%6.2f  PSLR=%6.2f dB  ' ...
             'ISLR=%6.2f dB  | %s | %.1fs\n'], ...
            mc, num_mc, target_cv, new_sumrate(mc), new_pslr_dB(mc), ...
            new_islr_dB(mc), status_str, new_time(mc));
end

S.CV_max_list = [S.CV_max_list, target_cv];
S.params.CV_max_list = S.CV_max_list;
S.sumrate_grid = [S.sumrate_grid; new_sumrate];
S.pslr_lin_grid = [S.pslr_lin_grid; new_pslr_lin];
S.islr_lin_grid = [S.islr_lin_grid; new_islr_lin];
S.pslr_dB_grid = [S.pslr_dB_grid; new_pslr_dB];
S.islr_dB_grid = [S.islr_dB_grid; new_islr_dB];
S.rank_eig2eig1_max_grid = [S.rank_eig2eig1_max_grid; new_rank_eig2eig1_max];
S.rank_eig2eig1_mean_grid = [S.rank_eig2eig1_mean_grid; new_rank_eig2eig1_mean];
S.rank_toptrace_min_grid = [S.rank_toptrace_min_grid; new_rank_toptrace_min];
S.rank_toptrace_mean_grid = [S.rank_toptrace_mean_grid; new_rank_toptrace_mean];
S.proposed_time_grid = [S.proposed_time_grid; new_time];

save(results_path, '-struct', 'S');

fprintf('------------------------------------------------------------\n');
fprintf('  Appended CV_max=%.1f to %s\n', target_cv, results_path);
fprintf('  Total elapsed: %s\n', format_time(toc(t_global)));
fprintf('============================================================\n');
