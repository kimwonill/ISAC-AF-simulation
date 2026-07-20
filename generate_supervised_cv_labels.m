function generate_supervised_cv_labels(num_train_channels, num_test_channels, force_regenerate)
% GENERATE_SUPERVISED_CV_LABELS  Generate CV-SDP labels for NN distillation.
%
% The generated dataset stores independent cold-start solutions for each
% (channel, CV) pair:
%     input  : H, CV_max
%     label  : W*, alpha*, achieved sum-rate/PSLR/ISLR
%
% This is intentionally not used by the paper simulations unless called
% manually.

if nargin < 1 || isempty(num_train_channels)
    num_train_channels = 80;
end
if nargin < 2 || isempty(num_test_channels)
    num_test_channels = 20;
end
if nargin < 3 || isempty(force_regenerate)
    force_regenerate = false;
end

params = setup_params();
params.NT = 4;
params.N = 16;
params.P_des = 0.8 * params.P_max / params.N;
params.CV_max_list = 0:0.1:0.9;
params.sdp_quiet = true;
params.stop_if_alpha_unchanged = true;

result_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
if exist(result_dir, 'dir') ~= 7
    mkdir(result_dir);
end

out_path = fullfile(result_dir, sprintf( ...
    'supervised_cv_labels_NT%d_N%d_train%d_test%d.mat', ...
    params.NT, params.N, num_train_channels, num_test_channels));
if exist(out_path, 'file') == 2 && ~force_regenerate
    fprintf('Label file already exists: %s\n', out_path);
    fprintf('Use force_regenerate=true to overwrite it.\n');
    return;
end

CV_grid = params.CV_max_list(:);
num_cv = numel(CV_grid);
num_channels = num_train_channels + num_test_channels;
num_samples = num_channels * num_cv;

H_samples = complex(nan(params.NT, params.K, params.N, num_samples), ...
                    nan(params.NT, params.K, params.N, num_samples));
W_labels = complex(nan(params.NT, params.NT, params.N, num_samples), ...
                   nan(params.NT, params.NT, params.N, num_samples));
alpha_labels = nan(params.K, params.N, num_samples);
cv_samples = nan(num_samples, 1);
channel_index = nan(num_samples, 1);
is_train = false(num_samples, 1);
sumrate_labels = nan(num_samples, 1);
pslr_labels = nan(num_samples, 1);
islr_labels = nan(num_samples, 1);
label_time = nan(num_samples, 1);
status_ok = false(num_samples, 1);

base_seed = 57191;
fprintf('Generating supervised CV labels: NT=%d, N=%d, train=%d, test=%d, CV points=%d\n', ...
    params.NT, params.N, num_train_channels, num_test_channels, num_cv);
fprintf('Output: %s\n', out_path);
total_tic = tic;

sample_idx = 0;
for ch = 1:num_channels
    rng(base_seed + ch - 1, 'twister');
    H = generate_channel(params);
    split_is_train = ch <= num_train_channels;

    for c = 1:num_cv
        sample_idx = sample_idx + 1;
        CV_max = CV_grid(c);
        H_samples(:, :, :, sample_idx) = H;
        cv_samples(sample_idx) = CV_max;
        channel_index(sample_idx) = ch;
        is_train(sample_idx) = split_is_train;

        iter_tic = tic;
        result = run_proposed(H, CV_max, params);
        label_time(sample_idx) = toc(iter_tic);

        if ~isnan(result.sumrate) && ~isempty(result.W)
            W_labels(:, :, :, sample_idx) = result.W;
            alpha_labels(:, :, sample_idx) = result.alpha;
            sumrate_labels(sample_idx) = result.sumrate;
            pslr_labels(sample_idx) = min(result.pslr_per_target);
            islr_labels(sample_idx) = max(result.islr_per_target);
            status_ok(sample_idx) = true;
        end

        elapsed = toc(total_tic);
        avg_time = elapsed / sample_idx;
        eta = avg_time * (num_samples - sample_idx);
        fprintf(['[%4d/%4d] ch %03d/%03d CV=%.1f | ok=%d SR=%6.2f ' ...
                 'PSLR=%6.2f dB | %.2fs | ETA %s\n'], ...
            sample_idx, num_samples, ch, num_channels, CV_max, ...
            status_ok(sample_idx), sumrate_labels(sample_idx), ...
            10*log10(pslr_labels(sample_idx)), label_time(sample_idx), format_time(eta));
    end

    save(out_path, 'params', 'CV_grid', 'num_train_channels', 'num_test_channels', ...
        'base_seed', 'H_samples', 'W_labels', 'alpha_labels', 'cv_samples', ...
        'channel_index', 'is_train', 'sumrate_labels', 'pslr_labels', ...
        'islr_labels', 'label_time', 'status_ok');
end

fprintf('Done. Valid labels: %d/%d\n', nnz(status_ok), num_samples);
fprintf('Mean label time: %.2fs, median %.2fs\n', ...
    mean(label_time(status_ok), 'omitnan'), median(label_time(status_ok), 'omitnan'));
fprintf('Saved: %s\n', out_path);
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
