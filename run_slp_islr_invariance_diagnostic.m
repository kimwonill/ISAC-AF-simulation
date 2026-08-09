function result = run_slp_islr_invariance_diagnostic(profile)
% RUN_SLP_ISLR_INVARIANCE_DIAGNOSTIC
% Reproduce a small instance of the MM-ADMM SLP waveform design in
% P. Li et al., IEEE TWC, 2025, with the paper's periodic 2-D AF and ISL.
%
% The purpose is diagnostic rather than a new manuscript benchmark:
%   1) optimize the paper-native instantaneous range-Doppler ISL;
%   2) normalize it by |chi(0,0)|^2;
%   3) evaluate the expected-ESL ISLR from our random-data model using the
%      optimized waveform's subcarrier directional-power profile.
%
% We use N_c=16 and N_s=4 as in Fig. 7 of the source paper. Algorithm 1 is
% reproduced without the optional SQUAREM acceleration.

sim_dir = fileparts(mfilename('fullpath'));
result_dir = fullfile(sim_dir, 'results');
ensure_dir(result_dir);

if nargin < 1 || isempty(profile)
    profile = 'paper_fig7';
end

%% Reproducible reduced paper setup
seed = 1;
rng(seed, 'twister');

switch lower(profile)
    case 'matched_square'
        % Same square N-by-N delay-Doppler grid used by our theorem.
        Nt = 4;
        Nc = 4;
        Ns = 4;
        K = 2;
        PT = 1;
        Gamma = 0.1;
        P0 = 0.50;
        output_name = 'slp_islr_invariance_matched_square.mat';
    case 'paper_fig7'
        % Nc and Ns used in Fig. 7 of the source paper.
        Nt = 6;
        Nc = 16;
        Ns = 4;
        K = 2;
        PT = 10;
        Gamma = 10^(6/10);
        P0 = 8;
        output_name = 'slp_islr_invariance_paper_fig7.mat';
    otherwise
        error('Unknown profile: %s', profile);
end
Ntf = Nc * Ns;
Ntot = Nt * Ntf;

sigma2_c = 1e-2;
Omega = 4;                      % QPSK
phi = pi / Omega;
gamma_ci = sqrt(sigma2_c * Gamma) * sin(phi);
kappa = 1.32;                   % our 16-QAM expected-ESL parameter

rho = 10;
outer_max = 10;
inner_max = 40;
inner_tol = 1e-3;
outer_tol = 1e-4;

theta0 = 0;
a = exp(1j * pi * (0:Nt-1).' * sin(theta0));
A = a * a';

Fnc = normalized_dft(Nc);
Ftilde = kron(eye(Ns), kron(Fnc, eye(Nt)));
Atilde = kron(eye(Ntf), a);
S = Atilde * Atilde';
Abar = Ftilde * kron(eye(Ntf), Nt * eye(Nt) - A) * Ftilde';
Abar = (Abar + Abar') / 2;
Pbar = Nt * PT - P0;
cm_level = sqrt(PT / Ntot);

H = (randn(Nt, K, Nc) + 1j * randn(Nt, K, Nc)) / sqrt(2);
symbol_index = randi(Omega, K, Nc, Ns);
symbols = exp(1j * pi * (2 * symbol_index - 1) / Omega);
Cci = build_ci_matrix(H, symbols, Nt, Nc, Ns, K, phi);

%% Paper Eq. (59) initialization
[x, init_margin, init_status] = solve_initialization(Cci, Ftilde, cm_level);
if isempty(x)
    error('SLP initialization failed: %s', init_status);
end
if init_margin < gamma_ci
    error(['Requested CI level %.3e exceeds the initialization margin ' ...
        '%.3e. Reduce Gamma or change the seed.'], gamma_ci, init_margin);
end

metrics_initial = waveform_metrics(x, Atilde, Nc, Ns, kappa);
metrics_initial.cm_residual = norm(abs(Ftilde' * x) - cm_level, 2);
history = nan(outer_max + 1, 7);
history(1, :) = history_row(0, metrics_initial, x, Ftilde, cm_level, ...
    Cci, gamma_ci, Abar, Pbar);

fprintf(['SLP ISLR diagnostic (%s): Nt=%d, Nc=%d, Ns=%d, K=%d, ' ...
    'seed=%d\n'], profile, Nt, Nc, Ns, K, seed);
fprintf(['  initial: paper ISLR %.6e (%.2f dB), expected ISLR ' ...
    '%.12f, CI margin %.3e\n'], metrics_initial.paper_islr, ...
    10*log10(max(metrics_initial.paper_islr, realmin)), ...
    metrics_initial.expected_islr, min(real(Cci * x)));

%% Paper Algorithm 1: outer MM and inner ADMM
previous_isl = metrics_initial.paper_isl;
outer_used = 0;
last_inner_used = 0;
last_status = init_status;

for t = 1:outer_max
    g = paper_mm_linear_term(x, Atilde, S, Nt, Ntf);

    z = Ftilde' * x;
    lambda = zeros(Ntot, 1);
    mu = zeros(Ntot, 1);
    inner_converged = false;

    for u = 1:inner_max
        mvec = g + Ftilde * (lambda - rho * z);
        [x_new, status] = solve_admm_x(mvec, rho, Abar, Pbar, ...
            Cci, gamma_ci, Ftilde, cm_level);
        if isempty(x_new)
            error('Inner ADMM x-update failed at outer %d, inner %d: %s', ...
                t, u, status);
        end

        v = Ftilde' * x_new + lambda / rho;
        r = cm_level * ones(Ntot, 1) - mu / rho;
        z_amp = max(0, 0.5 * (abs(v) + real(r)));
        z_new = z_amp .* exp(1j * angle(v));

        lambda = lambda + rho * (Ftilde' * x_new - z_new);
        mu = mu + rho * (abs(z_new) - cm_level);

        primal_res = norm(Ftilde' * x_new - z_new, 2);
        modulus_res = norm(abs(z_new) - cm_level, 2);
        x = x_new;
        z = z_new;

        if primal_res <= inner_tol && modulus_res <= inner_tol
            inner_converged = true;
            break;
        end
    end

    outer_used = t;
    last_inner_used = u;
    last_status = status;
    metrics = waveform_metrics(x, Atilde, Nc, Ns, kappa);
    history(t + 1, :) = history_row(t, metrics, x, Ftilde, cm_level, ...
        Cci, gamma_ci, Abar, Pbar);

    rel_change = abs(metrics.paper_isl - previous_isl) / ...
        max(abs(previous_isl), 1e-12);
    fprintf(['  outer %2d: paper ISLR %.6e (%7.2f dB), expected ' ...
        'ISLR %.12f, inner %2d, residuals [%.2e %.2e], rel %.2e\n'], ...
        t, metrics.paper_islr, ...
        10*log10(max(metrics.paper_islr, realmin)), ...
        metrics.expected_islr, u, primal_res, modulus_res, rel_change);

    if rel_change <= outer_tol && inner_converged
        break;
    end
    previous_isl = metrics.paper_isl;
end

metrics_final = waveform_metrics(x, Atilde, Nc, Ns, kappa);
metrics_final.cm_residual = norm(abs(Ftilde' * x) - cm_level, 2);
history = history(1:outer_used + 1, :);

result.seed = seed;
result.profile = profile;
result.Nt = Nt;
result.Nc = Nc;
result.Ns = Ns;
result.K = K;
result.PT = PT;
result.P0 = P0;
result.Gamma = Gamma;
result.gamma_ci = gamma_ci;
result.kappa = kappa;
result.rho = rho;
result.initialization_status = init_status;
result.final_status = last_status;
result.outer_iterations = outer_used;
result.last_inner_iterations = last_inner_used;
result.metrics_initial = metrics_initial;
result.metrics_final = metrics_final;
result.expected_invariant = Nc - 1;
result.history_columns = {'outer_iter', 'paper_islr', ...
    'expected_islr', 'cm_residual', 'min_ci_slack', ...
    'illumination_slack', 'af_parseval_error'};
result.history = history;
result.x = x;
result.symbols = symbols;
result.H = H;

output_path = fullfile(result_dir, output_name);
save(output_path, '-struct', 'result');

fprintf('\nFinal comparison\n');
fprintf('  paper-native instantaneous ISLR : %.12f (%.3f dB)\n', ...
    metrics_final.paper_islr, ...
    10*log10(max(metrics_final.paper_islr, realmin)));
fprintf('  our expected-ESL ISLR           : %.12f\n', ...
    metrics_final.expected_islr);
fprintf('  theorem N-1                     : %.12f\n', Nc - 1);
fprintf('  constant-modulus residual       : %.3e\n', ...
    metrics_final.cm_residual);
fprintf('  minimum CI slack                : %.3e\n', ...
    min(real(Cci * x) - gamma_ci));
fprintf('  illumination slack              : %.3e\n', ...
    Pbar - real(x' * Abar * x));
fprintf('  saved: %s\n', output_path);
end

function F = normalized_dft(N)
idx = (0:N-1).';
F = exp(-1j * 2 * pi * idx * idx.' / N) / sqrt(N);
end

function Cci = build_ci_matrix(H, symbols, Nt, Nc, Ns, K, phi)
num_constraints = 2 * K * Nc * Ns;
Ntot = Nt * Nc * Ns;
Cci = complex(zeros(num_constraints, Ntot));
row = 0;
ej = exp(-1j * pi / 2);
for m = 1:Ns
    for n = 1:Nc
        block = ((m - 1) * Nc + (n - 1)) * Nt + (1:Nt);
        for k = 1:K
            base = H(:, k, n)' * conj(symbols(k, n, m));
            row = row + 1;
            Cci(row, block) = base * (sin(phi) - ej * cos(phi));
            row = row + 1;
            Cci(row, block) = base * (sin(phi) + ej * cos(phi));
        end
    end
end
end

function [x, margin, status] = solve_initialization(Cci, Ftilde, cm_level)
Ntot = size(Ftilde, 1);
cvx_clear
cvx_solver mosek
cvx_begin quiet
    variable xvar(Ntot) complex
    variable delta
    maximize delta
    subject to
        real(Cci * xvar) >= delta;
        abs(Ftilde' * xvar) <= cm_level;
cvx_end
status = cvx_status;
if is_solved(status)
    x = double(xvar);
    margin = double(delta);
else
    x = [];
    margin = NaN;
end
end

function [x, status] = solve_admm_x(mvec, rho, Abar, Pbar, Cci, ...
    gamma_ci, Ftilde, cm_level)
Ntot = size(Ftilde, 1);
cvx_clear
cvx_solver mosek
cvx_begin quiet
    variable xvar(Ntot) complex
    minimize(real(mvec' * xvar) + (rho / 2) * sum_square_abs(xvar))
    subject to
        quad_form(xvar, Abar) <= Pbar;
        real(Cci * xvar) >= gamma_ci;
        abs(Ftilde' * xvar) <= cm_level;
cvx_end
status = cvx_status;
if is_solved(status)
    x = double(xvar);
else
    x = [];
end
end

function tf = is_solved(status)
tf = strcmpi(status, 'Solved') || strcmpi(status, 'Inaccurate/Solved');
end

function g = paper_mm_linear_term(x, Atilde, S, Nt, Ntf)
% Efficient implementation of Eqs. (41), (45), and (47).
Ntot = numel(x);
lambda_B = Ntf - 1;
lambda_C = Nt^2;

Sx = S * x;
G = 2 * (Sx * Sx' - lambda_C * (x * x'));
G = (G + G') / 2;
lambda_G = max(real(eig(G)));

y = Atilde' * x;
Y = y * y';
Mtilde = -2 * Ntf * (Y - diag(diag(Y)));
M = Atilde * Mtilde * Atilde';
M = (M + M') / 2;
lambda_M = max(real(eig(M)));

g = 2 * lambda_B * (G - lambda_G * eye(Ntot)) * x + ...
    2 * (M - lambda_M * eye(Ntot)) * x;
end

function metrics = waveform_metrics(x, Atilde, Nc, Ns, kappa)
q = abs(Atilde' * x).^2;
Ntf = Nc * Ns;
Q = reshape(q, [Nc, Ns]);

chi = fft(Q, [], 1);
chi = ifft(chi, [], 2) * Ns;
main_power = abs(chi(1, 1))^2;
paper_isl_af = sum(abs(chi(:)).^2) - main_power;
paper_isl_closed = Ntf * sum(q.^2) - sum(q)^2;
paper_isl = max(real(paper_isl_closed), 0);
paper_islr = paper_isl / max(main_power, realmin);

Pn = mean(Q, 2);
expected_islr = expected_esl_islr(Pn, kappa);

metrics.paper_isl = paper_isl;
metrics.paper_islr = paper_islr;
metrics.paper_islr_db = 10 * log10(max(paper_islr, realmin));
metrics.expected_islr = expected_islr;
metrics.main_power = main_power;
metrics.directional_power = Q;
metrics.subcarrier_power = Pn;
metrics.af_parseval_error = abs(paper_isl_af - paper_isl_closed) / ...
    max(1, abs(paper_isl_closed));
metrics.chi = chi;
metrics.cm_residual = NaN;
end

function islr = expected_esl_islr(P, kappa)
% Directly sum the corrected circular-correlation ESL over the N-by-N grid.
N = numel(P);
sq = sum(P.^2);
ESL = zeros(N, N);
for nu = 0:N-1
    corr_nu = sum(P .* circshift(P, nu));
    for tau = 0:N-1
        ESL(tau + 1, nu + 1) = corr_nu;
        if nu == 0
            Rtau = sum(P .* exp(1j * 2 * pi * (0:N-1).' * tau / N));
            ESL(tau + 1, nu + 1) = ESL(tau + 1, nu + 1) + ...
                abs(Rtau)^2 + (kappa - 2) * sq;
        end
    end
end
main = ESL(1, 1);
islr = (sum(ESL(:)) - main) / main;
end

function row = history_row(iter, metrics, x, Ftilde, cm_level, ...
    Cci, gamma_ci, Abar, Pbar)
cm_residual = norm(abs(Ftilde' * x) - cm_level, 2);
metrics.cm_residual = cm_residual;
min_ci_slack = min(real(Cci * x) - gamma_ci);
illumination_slack = Pbar - real(x' * Abar * x);
row = [iter, metrics.paper_islr, metrics.expected_islr, ...
    cm_residual, min_ci_slack, illumination_slack, ...
    metrics.af_parseval_error];
end

function ensure_dir(path_value)
if exist(path_value, 'dir') ~= 7
    mkdir(path_value);
end
end
