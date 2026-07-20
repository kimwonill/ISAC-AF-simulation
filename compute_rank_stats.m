function stats = compute_rank_stats(W)
% COMPUTE_RANK_STATS  Diagnostic for SDR tightness (Proposition 1 in the paper).
%
% For each subcarrier n, computes eigenvalues of W_n and returns:
%   eig2_over_eig1(n) :  ratio lambda_2 / lambda_1 -- should be ~0 if rank-1
%   top_to_trace(n)   :  lambda_1 / sum(lambda)    -- should be ~1 if rank-1
%
% Returned struct contains worst-case and mean values across n.
% A perfectly rank-1 W has eig2/eig1 = 0 and top/trace = 1.

N = size(W, 3);
eig2_over_eig1 = zeros(N, 1);
top_to_trace   = zeros(N, 1);

for n = 1:N
    e = sort(real(eig(W(:, :, n))), 'descend');
    e = max(e, 0);                              % clip tiny negative numerics
    if e(1) > 1e-12
        eig2_over_eig1(n) = e(2) / e(1);
        top_to_trace(n)   = e(1) / sum(e);
    else
        % Effectively zero matrix -- subcarrier inactive
        eig2_over_eig1(n) = 0;
        top_to_trace(n)   = 1;
    end
end

stats.eig2_over_eig1_max  = max(eig2_over_eig1);
stats.eig2_over_eig1_mean = mean(eig2_over_eig1);
stats.top_to_trace_min    = min(top_to_trace);
stats.top_to_trace_mean   = mean(top_to_trace);
stats.per_subcarrier.eig2_over_eig1 = eig2_over_eig1;
stats.per_subcarrier.top_to_trace   = top_to_trace;

end
