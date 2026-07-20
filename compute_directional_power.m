function P = compute_directional_power(W, a)
% COMPUTE_DIRECTIONAL_POWER  P_n(theta) for n = 0..N-1 given beamforming
% covariance tensor W and steering vector a.
%
% Uses the paper convention P_n = a^T W_n a^* = b^H W_n b, b = a^*.

N = size(W, 3);
P = zeros(N, 1);
a_metric = conj(a);
for n = 1:N
    P(n) = real(a_metric' * W(:, :, n) * a_metric);
end

end
