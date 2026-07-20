function [W_out, sumrate, status, solver_iters] = solve_direct_sca_sdp(H, alpha, A, pslr_min, islr_max, P_ref, params)
% SOLVE_DIRECT_SCA_SDP  One convexified SCA beamforming subproblem.
%
% This implements Algorithm 2's inner step for fixed subcarrier allocation
% and fixed linearization point P_ref.

NT = params.NT; N = params.N; K = params.K; L = params.L;
kappa = params.kappa;
use_islr_constraint = isfinite(islr_max);
solver_iters = NaN;

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

A_metric = conj(A);
tones = (0:N-1).';
collect_solver_log = isfield(params, 'collect_cvx_solver_log') && params.collect_cvx_solver_log;
cvx_log_text = '';

cvx_clear
if params.sdp_quiet && ~collect_solver_log, cvx_begin sdp quiet; else, cvx_begin sdp; end %#ok<NOSEMI>
    variable W(NT, NT, N) hermitian semidefinite
    variable t_sl(L) nonnegative

    expression rate_n(N, 1)
    for n = 1:N
        if user_n(n) > 0
            hk = H(:, user_n(n), n);
            rate_n(n) = log(1 + real(hk' * W(:,:,n) * hk) / params.sigma2) / log(2);
        end
    end

    expression Pn(N, L)
    for l = 1:L
        for n = 1:N
            Pn(n, l) = real(A_metric(:, l)' * W(:,:,n) * A_metric(:, l));
        end
    end

    expression power_total
    power_total = 0;
    for n = 1:N
        power_total = power_total + real(trace(W(:,:,n)));
    end

    maximize sum(rate_n)

    subject to
        power_total <= params.P_max;                                    %#ok<VUNUS>

        for l = 1:L
            p = Pn(:, l);

            % Illumination floor
            sum(p) / N >= params.P_des;                                 %#ok<VUNUS>

            % Sidelobe epigraph for D_P(p)
            for tau = 1:N-1
                c_tau = exp(-1j * 2*pi * tones * tau / N);
                square_abs(c_tau.' * p) <= t_sl(l);                     %#ok<VUNUS>
            end

            p0 = max(P_ref(:, l), 0);
            s0 = sum(p0);
            q0 = sum(p0.^2);

            % PSLR SCA: PSLR_min * D_P(p) - N_P_lin(p;p0) <= 0
            NP0 = s0^2 + (kappa - 1) * q0;
            grad_NP = 2*s0*ones(N, 1) + 2*(kappa - 1)*p0;
            lin_NP = NP0 + grad_NP.' * (p - p0);
            DP = t_sl(l) + (kappa - 1) * sum_square(p);
            pslr_min * DP - lin_NP <= 0;                                %#ok<VUNUS>

            if use_islr_constraint
                % ISLR SCA: N_I(p) - (ISLR_max+1) * D_I_lin(p;p0) <= 0
                DI0 = (kappa - 1) * q0 + s0^2;
                grad_DI = 2*s0*ones(N, 1) + 2*(kappa - 1)*p0;
                lin_DI = DI0 + grad_DI.' * (p - p0);
                NI = N * ((kappa - 0.5) * sum_square(p) + 0.5 * square_pos(sum(p)));
                NI - (islr_max + 1) * lin_DI <= 0;                      %#ok<VUNUS>
            end
        end

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

function value = sum_finite(value)
value = value(isfinite(value));
if isempty(value)
    value = NaN;
else
    value = sum(value(:));
end
end
