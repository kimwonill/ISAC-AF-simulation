function islr = compute_islr(P, kappa)
% COMPUTE_ISLR  Expected ISLR (linear, not dB) from a directional power
% profile P = [P_0; ...; P_{N-1}].
%
% Formula (paper Eq. line 296):
%   ISLR = N[(kappa-1/2) sum P_n^2 + 1/2 (sum P_n)^2]
%        / [(kappa-1)   sum P_n^2 +       (sum P_n)^2]   - 1
%
% Lower ISLR is better (less total sidelobe energy relative to the mainlobe).

N        = length(P);
sq       = sum(P.^2);
sumsq    = sum(P)^2;

numer    = N * ((kappa - 0.5) * sq + 0.5 * sumsq);
denom    = (kappa - 1) * sq + sumsq;
islr     = numer / denom - 1;

end
