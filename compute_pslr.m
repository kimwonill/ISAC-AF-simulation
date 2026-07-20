function pslr = compute_pslr(P, kappa)
% COMPUTE_PSLR  Expected zero-Doppler PSLR (linear, not dB) from a
% directional power profile P = [P_0; ...; P_{N-1}].
%
% Formula (paper Eq. line 305):
%   PSLR = [(kappa-1) sum P_n^2 + (sum P_n)^2]
%        / [(kappa-1) sum P_n^2 + max_{tau != 0} | sum_n P_n e^{j2pi n tau/N} |^2]
%
% MATLAB's fft uses the -j convention; magnitudes are unchanged.

N = length(P);
P_dft   = fft(P);
sl_pow  = abs(P_dft(2:end)).^2;     % exclude DC (tau=0)
sl_max  = max(sl_pow);

sq      = sum(P.^2);
mainlobe = (kappa - 1) * sq + sum(P)^2;
denom    = (kappa - 1) * sq + sl_max;
pslr     = mainlobe / denom;

end
