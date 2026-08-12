function [W_out, w_out, status, info] = repair_rank_one_powers( ...
    H, alpha, w_initial, A, CV_max, params)
%REPAIR_RANK_ONE_POWERS Re-optimize powers on fixed principal directions.
%   This low-dimensional convex repair preserves rank one on every
%   subcarrier. It is called only when direct principal-mode extraction
%   violates a proposed-design constraint beyond the numerical tolerance.

N = params.N;
K = params.K;
L = params.L;
NT = params.NT;

U = zeros(NT, N);
for n = 1:N
    wn_norm = norm(w_initial(:, n));
    if wn_norm <= eps
        status = sprintf('Repair failed: inactive principal direction at n=%d', n);
        W_out = [];
        w_out = [];
        info = struct('solver_iters', NaN, 'objective', NaN);
        return;
    end
    U(:, n) = w_initial(:, n) / wn_norm;
end

user_n = zeros(1, N);
for n = 1:N
    ix = find(alpha(:, n), 1);
    if ~isempty(ix)
        user_n(n) = ix;
    end
end

rate_gain = zeros(N, 1);
for n = 1:N
    if user_n(n) > 0
        hk = H(:, user_n(n), n);
        rate_gain(n) = abs(hk' * U(:, n))^2 / params.sigma2;
    end
end

directional_gain = zeros(N, L);
A_metric = conj(A);
for l = 1:L
    for n = 1:N
        directional_gain(n, l) = abs(A_metric(:, l)' * U(:, n))^2;
    end
end

collect_solver_log = isfield(params, 'collect_cvx_solver_log') && ...
    params.collect_cvx_solver_log;
cvx_log_text = '';
cvx_clear
configure_solver(params);
if params.sdp_quiet && ~collect_solver_log
    cvx_begin quiet
else
    cvx_begin
end
    variable q(N) nonnegative
    expression rate_n(N, 1)
    rate_n = log(1 + rate_gain .* q) / log(2); %#ok<NODEF>
    expression Pn(N, L)
    Pn = directional_gain .* repmat(q, 1, L);

    maximize sum(rate_n)

    subject to
        sum(q) <= params.P_max;                                        %#ok<VUNUS>
        for l = 1:L
            mu_l = sum(Pn(:, l)) / N;
            if isfinite(CV_max)
                norm(Pn(:, l) - mu_l, 2) <= sqrt(N) * CV_max * mu_l;  %#ok<VUNUS>
            end
            mu_l >= params.P_des;                                      %#ok<VUNUS>
        end
        for k = 1:K
            if params.Q(k) > 0
                idx_k = find(user_n == k);
                if isempty(idx_k)
                    0 >= params.Q(k);                                  %#ok<VUNUS>
                else
                    sum(rate_n(idx_k)) >= params.Q(k);                  %#ok<VUNUS>
                end
            end
        end
if collect_solver_log
    cvx_log_text = evalc('cvx_end');
else
    cvx_end
end

status = cvx_status;
info.solver_iters = NaN;
if exist('cvx_slvitr', 'var') == 1
    finite_iters = cvx_slvitr(isfinite(cvx_slvitr));
    if ~isempty(finite_iters)
        info.solver_iters = sum(finite_iters(:));
    end
end
if collect_solver_log
    parsed_solver_iters = parse_cvx_solver_iterations(cvx_log_text);
    if isfinite(parsed_solver_iters)
        info.solver_iters = parsed_solver_iters;
    end
end
info.objective = NaN;

if ~(strcmpi(status, 'Solved') || strcmpi(status, 'Inaccurate/Solved'))
    W_out = [];
    w_out = [];
    return;
end

q = max(double(q), 0);
info.objective = sum(log2(1 + rate_gain .* q));
W_out = zeros(NT, NT, N);
w_out = zeros(NT, N);
for n = 1:N
    w_out(:, n) = sqrt(q(n)) * U(:, n);
    W_out(:, :, n) = w_out(:, n) * w_out(:, n)';
end
end

function configure_solver(params)
if isfield(params, 'cvx_solver') && ~isempty(params.cvx_solver)
    cvx_solver(params.cvx_solver);
end
if isfield(params, 'cvx_solver_threads') && ...
        isfinite(params.cvx_solver_threads) && params.cvx_solver_threads > 0
    cvx_solver_settings('MSK_IPAR_NUM_THREADS', ...
        round(params.cvx_solver_threads));
end
end
