function [W_out, w_out, status, info] = ...
    solve_direct_fixed_direction_powers( ...
    H, alpha, directions, A, pslr_min, params, mode)
%SOLVE_DIRECT_FIXED_DIRECTION_POWERS Optimize powers for fixed directions.
%   With p_{l,n}=g_{l,n}q_n, the exact zero-Doppler PSLR condition is SOC:
%
%     ||[sqrt(rho) Re(f_tau^T p_l);
%        sqrt(rho) Im(f_tau^T p_l);
%        sqrt((rho-1)(kappa-1)) p_l]||_2 <= 1^T p_l.
%
%   MODE='sensing' minimizes power without QoS as a GR prefilter.
%   MODE='refine' maximizes sum-rate subject to the original QoS constraints.

if nargin < 7 || isempty(mode)
    mode = 'refine';
end
if ~ismember(mode, {'sensing', 'refine'})
    error('mode must be ''sensing'' or ''refine''.');
end
if ~isfinite(pslr_min) || pslr_min <= 0 || params.kappa < 1
    error('PSLR_min must be positive and kappa must be at least one.');
end
enforce_pslr = pslr_min > 1;

N = params.N;
K = params.K;
L = params.L;
NT = params.NT;
U = zeros(NT, N);
for n = 1:N
    direction_norm = norm(directions(:, n));
    if direction_norm <= eps
        W_out = [];
        w_out = [];
        status = sprintf('Infeasible direction at subcarrier %d', n);
        info = struct('solver_iters', NaN, 'objective', NaN);
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
if strcmp(mode, 'refine')
    for k = 1:K
        if params.Q(k) > 0 && ~any(user_n == k)
            W_out = [];
            w_out = [];
            status = sprintf( ...
                'Infeasible: user %d has no allocated subcarrier', k);
            info = struct('solver_iters', NaN, 'objective', NaN);
            return;
        end
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
        directional_gain(n, l) = ...
            abs(A_metric(:, l)' * U(:, n))^2;
    end
end

tones = (0:N-1).';
rho_sqrt = sqrt(max(pslr_min, 1));
shape_sqrt = sqrt(max(pslr_min - 1, 0) * (params.kappa - 1));

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

    if strcmp(mode, 'sensing')
        minimize sum(q)
    else
        expression rate_n(N, 1)
        rate_n = log(1 + rate_gain .* q) / log(2);
        maximize sum(rate_n)
    end

    subject to
        sum(q) <= params.P_max;                                        %#ok<VUNUS>
        for l = 1:L
            sum(Pn(:, l)) / N >= params.P_des;                         %#ok<VUNUS>
            if enforce_pslr
                for tau = 1:N-1
                    c_tau = exp(-1j * 2*pi * tones * tau / N);
                    dft_component = c_tau.' * Pn(:, l);
                    norm([rho_sqrt * real(dft_component); ...
                        rho_sqrt * imag(dft_component); ...
                        shape_sqrt * Pn(:, l)], 2) <= sum(Pn(:, l));   %#ok<VUNUS>
                end
            end
        end
        if strcmp(mode, 'refine')
            for k = 1:K
                if params.Q(k) > 0
                    idx_k = user_n == k;
                    sum(rate_n(idx_k)) >= params.Q(k);                 %#ok<VUNUS>
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
if strcmp(mode, 'sensing')
    info.objective = sum(q);
else
    info.objective = sum(log2(1 + rate_gain .* q));
end
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
