function [W_rank1, w, stats] = principal_eigenmode_recovery(W_sdp)
% PRINCIPAL_EIGENMODE_RECOVERY  Rank-one projection of SDP covariances.
%
% For each subcarrier n, this function applies the recovery rule used in
% the manuscript,
%   w_n = sqrt(lambda_{n,1}) u_{n,1},
%   W_rank1,n = w_n w_n^H,
% where lambda_{n,1} and u_{n,1} are the dominant eigenpair of W_sdp,n.

% The returned statistics quantify SDR tightness before recovery and the
% fraction of covariance trace retained by the principal eigenmode.

NT = size(W_sdp, 1);
N = size(W_sdp, 3);
W_rank1 = zeros(NT, NT, N);
w = zeros(NT, N);
retained_trace = ones(N, 1);
relative_frobenius_error = zeros(N, 1);

for n = 1:N
    Wn = (W_sdp(:, :, n) + W_sdp(:, :, n)') / 2;
    [U, D] = eig(Wn);
    eigenvalues = real(diag(D));
    [lambda1, idx] = max(eigenvalues);
    lambda1 = max(lambda1, 0);
    u1 = U(:, idx);

    w(:, n) = sqrt(lambda1) * u1;
    W_rank1(:, :, n) = w(:, n) * w(:, n)';

    covariance_trace = max(real(trace(Wn)), 0);
    if covariance_trace > 1e-12
        retained_trace(n) = lambda1 / covariance_trace;
    end
    covariance_norm = norm(Wn, 'fro');
    if covariance_norm > 1e-12
        relative_frobenius_error(n) = ...
            norm(Wn - W_rank1(:, :, n), 'fro') / covariance_norm;
    end
end

stats = compute_rank_stats(W_sdp);
stats.retained_trace_min = min(retained_trace);
stats.retained_trace_mean = mean(retained_trace);
stats.retained_trace_per_subcarrier = retained_trace;
stats.relative_frobenius_error_max = max(relative_frobenius_error);
stats.relative_frobenius_error_mean = mean(relative_frobenius_error);
stats.relative_frobenius_error_per_subcarrier = relative_frobenius_error;
end
