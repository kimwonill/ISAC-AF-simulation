function islr = compute_islr(P, ~)
% COMPUTE_ISLR  Expected ISLR of the full periodic N-by-N AF.
%
% With the corrected circular correlation in the total-AF-energy
% derivation and a unitary DFT matrix, the expected sidelobe-to-mainlobe
% ratio is independent of both the directional-power profile and the
% symbol fourth moment:
%
%   ISLR = N - 1.

N = length(P);
islr = N - 1;

end
