function [W_out, sumrate, status] = solve_surrogate_sdp(H, alpha, A, mode, eta, params)
% SOLVE_SURROGATE_SDP  Literature-faithful non-AF sensing baselines.
%
% CRB-inspired: maximize a weighted angle-Fisher-information utility based
% on the derivative steering vector, following CRB-oriented ISAC beamforming.
% MI-inspired: maximize a weighted sensing mutual information log-det utility
% over the target steering subspace.
%
% Neither baseline directly constrains the AF sidelobe profile.

NT = params.NT; N = params.N; K = params.K; L = params.L;

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

% Sensing metric convention: b = a^* so b^H W b = a^T W a^*.
B = conj(A);
D = steering_derivative_metric(params);

cvx_clear
if params.sdp_quiet, cvx_begin sdp quiet; else, cvx_begin sdp; end %#ok<NOSEMI>
    variable W(NT, NT, N) hermitian semidefinite

    expression rate_n(N, 1)
    for n = 1:N
        if user_n(n) > 0
            hk = H(:, user_n(n), n);
            rate_n(n) = log(1 + real(hk' * W(:,:,n) * hk) / params.sigma2) / log(2);
        end
    end

    expression Rtot(NT, NT)
    Rtot = 0;
    expression power_total
    power_total = 0;
    for n = 1:N
        Rtot = Rtot + W(:,:,n) / N;
        power_total = power_total + real(trace(W(:,:,n)));
    end

    expression sensing_obj
    sensing_obj = 0;
    if strcmpi(mode, 'crb')
        % Angle-FI proxy: J_l proportional to dot{b}_l^H R_x dot{b}_l.
        % Maximizing J_l corresponds to minimizing a CRB upper proxy.
        for l = 1:L
            dl = D(:, l);
            sensing_obj = sensing_obj + real(dl' * Rtot * dl) / params.sigma2;
        end
    elseif strcmpi(mode, 'mi')
        % MI proxy for target-response subspace: log det(I + B^H R_x B/sigma2).
        sensing_obj = log_det(eye(L) + (B' * Rtot * B) / params.sigma2) / log(2);
    else
        error('Unknown surrogate baseline mode: %s', mode);
    end

    maximize sum(rate_n) + eta * sensing_obj

    subject to
        power_total <= params.P_max;                                    %#ok<VUNUS>

        for k = 1:K
            if params.Q(k) > 0
                idx_k = find(user_n == k);
                sum(rate_n(idx_k)) >= params.Q(k);                      %#ok<VUNUS>
            end
        end
cvx_end

status = cvx_status;
if strcmpi(status, 'Solved') || strcmpi(status, 'Inaccurate/Solved')
    W_out   = double(W);
    sumrate = sum(double(rate_n));
else
    W_out   = [];
    sumrate = NaN;
end

end

function D = steering_derivative_metric(params)
phase_step = 2*pi * params.dT / params.lambda;
idx = (0:params.NT-1).';
D = zeros(params.NT, params.L);
for l = 1:params.L
    theta = params.theta(l);
    a = exp(1j * phase_step * idx * sin(theta));
    b = conj(a);
    D(:, l) = -1j * phase_step * idx * cos(theta) .* b;
end
end
