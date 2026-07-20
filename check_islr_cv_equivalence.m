function stats = check_islr_cv_equivalence(num_mc)
% CHECK_ISLR_CV_EQUIVALENCE  Verify exact direct-ISLR/CV equivalence.
%
% Runs the requested N_T=4, N=16 Monte Carlo experiment and solves both
% formulations from the same allocation initialization at each CV point.

if nargin < 1
    num_mc = 30;
end

params = setup_params();
params.NT = 4;
params.N = 16;
params.P_des = 0.8 * params.P_max / params.N;
params.num_mc = num_mc;

CV_list = params.CV_max_list(:);
num_cv = numel(CV_list);

out_dir = fileparts(mfilename('fullpath'));
result_dir = fullfile(out_dir, 'results');
if exist(result_dir, 'dir') ~= 7, mkdir(result_dir); end
cache_path = fullfile(result_dir, sprintf('islr_cv_equivalence_NT4_N16_MC%d.mat', num_mc));

prop_sumrate_grid = nan(num_cv, num_mc);
prop_islr_grid = nan(num_cv, num_mc);
direct_sumrate_grid = nan(num_cv, num_mc);
direct_islr_grid = nan(num_cv, num_mc);
target_islr_grid = nan(num_cv, num_mc);
cv_equiv_grid = nan(num_cv, num_mc);

t_global = tic;
total_runs = num_cv * num_mc;
run_count = 0;
fprintf('ISLR-CV equivalence check: N_T=4, N=16, MC=%d\n', num_mc);

for mc = 1:num_mc
    rng(mc, 'twister');
    H = generate_channel(params);
    alpha_warm = [];

    for c = 1:num_cv
        run_count = run_count + 1;
        CV_max = CV_list(c);
        alpha0 = alpha_warm;

        prop_result = run_proposed(H, CV_max, params, alpha0);
        [~, islr_exact_max] = direct_thresholds_from_cv(CV_max, params, false);
        direct_result = run_direct_islr_exact(H, islr_exact_max, params, alpha0);

        if ~isnan(prop_result.sumrate)
            prop_sumrate_grid(c, mc) = prop_result.sumrate;
            prop_islr_grid(c, mc) = max(prop_result.islr_per_target);
            alpha_warm = prop_result.alpha;
        else
            alpha_warm = [];
        end

        if ~isnan(direct_result.sumrate)
            direct_sumrate_grid(c, mc) = direct_result.sumrate;
            direct_islr_grid(c, mc) = max(direct_result.islr_per_target);
            cv_equiv_grid(c, mc) = direct_result.equivalent_CV_max;
        end
        target_islr_grid(c, mc) = islr_exact_max;

        delta_sr = direct_sumrate_grid(c, mc) - prop_sumrate_grid(c, mc);
        delta_islr_db = 10*log10(direct_islr_grid(c, mc)) - 10*log10(prop_islr_grid(c, mc));
        elapsed = toc(t_global);
        eta_sec = elapsed / run_count * (total_runs - run_count);
        fprintf(['[%5.1f%% %3d/%3d] MC %02d/%02d CV=%.1f | ' ...
                 'dSR=%+.3e bps/Hz dISLR=%+.3e dB | elapsed %s ETA %s\n'], ...
                100*run_count/total_runs, run_count, total_runs, mc, num_mc, CV_max, ...
                delta_sr, delta_islr_db, format_time(elapsed), format_time(eta_sec));
    end

    save(cache_path, 'params', 'CV_list', 'num_mc', ...
        'prop_sumrate_grid', 'prop_islr_grid', ...
        'direct_sumrate_grid', 'direct_islr_grid', ...
        'target_islr_grid', 'cv_equiv_grid', '-v7.3');
end

stats = summarize_equivalence(CV_list, prop_sumrate_grid, prop_islr_grid, ...
    direct_sumrate_grid, direct_islr_grid);
save(cache_path, 'stats', '-append');

fprintf('\nSummary by CV:\n');
fprintf('CVmax   mean dSR      max |dSR|     mean dISLR(dB)  max |dISLR|(dB)\n');
for c = 1:num_cv
    fprintf('%.1f   %+10.3e   %10.3e   %+13.3e   %13.3e\n', ...
        CV_list(c), stats.mean_delta_sumrate(c), stats.max_abs_delta_sumrate(c), ...
        stats.mean_delta_islr_db(c), stats.max_abs_delta_islr_db(c));
end
fprintf('Saved equivalence check to %s\n', cache_path);

end

function stats = summarize_equivalence(CV_list, prop_sumrate, prop_islr, direct_sumrate, direct_islr)
delta_sumrate = direct_sumrate - prop_sumrate;
delta_islr_db = 10*log10(direct_islr) - 10*log10(prop_islr);
valid_mask = isfinite(prop_sumrate) & isfinite(direct_sumrate) & ...
    isfinite(prop_islr) & isfinite(direct_islr);

stats.CV_list = CV_list;
stats.num_valid = sum(valid_mask, 2);
stats.mean_delta_sumrate = mean(delta_sumrate, 2, 'omitnan');
stats.max_abs_delta_sumrate = max(abs(delta_sumrate), [], 2, 'omitnan');
stats.mean_delta_islr_db = mean(delta_islr_db, 2, 'omitnan');
stats.max_abs_delta_islr_db = max(abs(delta_islr_db), [], 2, 'omitnan');
stats.overall_max_abs_delta_sumrate = max(abs(delta_sumrate(:)), [], 'omitnan');
stats.overall_max_abs_delta_islr_db = max(abs(delta_islr_db(:)), [], 'omitnan');
end
