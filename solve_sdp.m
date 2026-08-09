function [W_out, sumrate, status, solver_iters] = solve_sdp(H, alpha, A, CV_max, params)
% SOLVE_SDP  Beamforming subproblem (P_SDP) for fixed subcarrier allocation.
%
%   Maximizes sum-rate subject to:
%     - CV(theta_l) <= CV_max  (SOC)
%     - mu_p(theta_l) >= P_des
%     - per-user QoS (Q_k)
%     - sum_n Tr(W_n) <= P_max
%     - W_n >= 0
%
% Uses the paper's directional-power lift
% P_n(theta) = a(theta)^T W_n a(theta)^*.

NT = params.NT; N = params.N; K = params.K; L = params.L;
solver_iters = NaN;

% Pre-compute the user index assigned to each subcarrier
user_n = zeros(1, N);
for n = 1:N
    ix = find(alpha(:, n), 1);
    if ~isempty(ix), user_n(n) = ix; end
end

for k = 1:K
    if params.Q(k) > 0 && ~any(user_n == k)
        W_out   = [];
        sumrate = NaN;
        status  = sprintf('Infeasible: user %d has no allocated subcarrier for QoS', k);
        return;
    end
end

% Metric steering vector b = a^* gives b^H W b = a^T W a^*.
A_metric = conj(A);
collect_solver_log = isfield(params, 'collect_cvx_solver_log') && params.collect_cvx_solver_log;
cvx_log_text = '';

cvx_clear   % discard any stale CVX scope from a prior interrupted solve
configure_solver(params);
if params.sdp_quiet && ~collect_solver_log, cvx_begin sdp quiet; else, cvx_begin sdp; end %#ok<NOSEMI>
    variable W(NT, NT, N) hermitian semidefinite

    % --- rate expressions ---
    expression rate_n(N, 1)
    for n = 1:N
        if user_n(n) > 0
            hk = H(:, user_n(n), n);
            rate_n(n) = log(1 + real(hk' * W(:,:,n) * hk) / params.sigma2) / log(2);
        end
    end

    % --- directional powers P_n(theta_l) ---
    expression Pn(N, L)
    for l = 1:L
        for n = 1:N
            Pn(n, l) = real(A_metric(:, l)' * W(:,:,n) * A_metric(:, l));
        end
    end

    % --- total power ---
    expression power_total
    power_total = 0;
    for n = 1:N
        power_total = power_total + real(trace(W(:,:,n)));
    end

    maximize sum(rate_n)

    subject to
        % Power budget
        power_total <= params.P_max;                                    %#ok<VUNUS>

        % CV (SOC) and illumination floor per target
        for l = 1:L
            mu_l = sum(Pn(:, l)) / N;
            norm(Pn(:, l) - mu_l, 2) <= sqrt(N) * CV_max * mu_l;        %#ok<VUNUS>
            mu_l >= params.P_des;                                       %#ok<VUNUS>
        end

        % Per-user QoS
        for k = 1:K
            if params.Q(k) > 0
                idx_k = find(user_n == k);
                sum(rate_n(idx_k)) >= params.Q(k);                      %#ok<VUNUS>
            end
        end
if collect_solver_log
    cvx_log_text = evalc('cvx_end');
else
    cvx_end
end

status = cvx_status;
if exist('cvx_slvitr', 'var') == 1
    solver_iters = sum_finite(cvx_slvitr);
end
if collect_solver_log
    parsed_solver_iters = parse_cvx_solver_iterations(cvx_log_text);
    if isfinite(parsed_solver_iters)
        solver_iters = parsed_solver_iters;
    end
end
if strcmpi(status, 'Solved') || strcmpi(status, 'Inaccurate/Solved')
    W_out   = double(W);
    sumrate = sum(double(rate_n));
else
    W_out   = [];
    sumrate = NaN;
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

function value = sum_finite(value)
value = value(isfinite(value));
if isempty(value)
    value = NaN;
else
    value = sum(value(:));
end
end
