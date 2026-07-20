function A = compute_steering(params)
% COMPUTE_STEERING  ULA transmit steering matrix.
%
% A(:,l) = [1; exp(j*2*pi*dT*sin(theta_l)/lambda); ... ; exp(j*2*pi*(NT-1)*dT*sin(theta_l)/lambda)]
% Convention: +j (transmit beamforming convention) -- matches paper Eq. line 192.

NT = params.NT; L = params.L;
A = zeros(NT, L);
phase_step = 2*pi * params.dT / params.lambda;
for l = 1:L
    A(:, l) = exp(1j * phase_step * sin(params.theta(l)) * (0:NT-1).');
end

end
