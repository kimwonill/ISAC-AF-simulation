function [W_out, w_out, status, info] = fixed_direction_rank_one_feasibility( ...
    H, alpha, directions, A, CV_max, params, enforce_qos)
%FIXED_DIRECTION_RANK_ONE_FEASIBILITY Fast SOCP power-feasibility repair.
%   The columns of DIRECTIONS define one beam direction per subcarrier.
%   Nonnegative powers are optimized while the directions remain fixed.
%   The QoS constraints use a linear chord lower bound on log2(1+g*q), so
%   every returned solution also satisfies the exact logarithmic QoS.

N = params.N;
K = params.K;
L = params.L;
NT = params.NT;
if nargin < 7
    enforce_qos = true;
end

U = zeros(NT, N);
for n = 1:N
    direction_norm = norm(directions(:, n));
    if direction_norm <= eps
        W_out = [];
        w_out = [];
        status = sprintf('Infeasible direction at subcarrier %d', n);
        info = struct('solver_iters', NaN, 'power', NaN);
        return;
    end
    U(:, n) = directions(:, n) / direction_norm;
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

% Concavity gives log2(1+g*q) >= q/Pmax*log2(1+g*Pmax) on [0,Pmax].
% Hence these coefficients yield a conservative linear QoS constraint.
qos_chord = log2(1 + rate_gain * params.P_max) / params.P_max;

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
    expression Pn(N, L)
    Pn = directional_gain .* repmat(q, 1, L); %#ok<NODEF>

    minimize sum(q)

    subject to
        sum(q) <= params.P_max;                                        %#ok<VUNUS>
        for l = 1:L
            mu_l = sum(Pn(:, l)) / N;
            if isfinite(CV_max)
                norm(Pn(:, l) - mu_l, 2) <= ...
                    sqrt(N) * CV_max * mu_l;                           %#ok<VUNUS>
            end
            mu_l >= params.P_des;                                      %#ok<VUNUS>
        end
        if enforce_qos
            for k = 1:K
                if params.Q(k) > 0
                    idx_k = find(user_n == k);
                    if isempty(idx_k)
                        0 >= params.Q(k);                              %#ok<VUNUS>
                    else
                        qos_chord(idx_k).' * q(idx_k) >= params.Q(k);   %#ok<VUNUS>
                    end
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
info.power = NaN;

if ~(strcmpi(status, 'Solved') || strcmpi(status, 'Inaccurate/Solved'))
    W_out = [];
    w_out = [];
    return;
end

q = max(double(q), 0);
info.power = sum(q);
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
