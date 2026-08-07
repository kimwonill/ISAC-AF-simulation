function [pslr_min, islr_invariant] = direct_thresholds_from_cv(CV_max, params, apply_relax)
% DIRECT_THRESHOLDS_FROM_CV  Convert a CV target to a direct PSLR limit.
%
% The proposed CV constraint guarantees
%   PSLR >= L(N,kappa,CV_max).
% This threshold is used to compare against the direct PSLR-SCA baseline
% under the same guaranteed PSLR target.
% The optional second output is retained only for compatibility with legacy
% scripts; under the periodic N-point operator it is the invariant N-1.

if nargin < 3
    apply_relax = true;
end

N = params.N;
kappa = params.kappa;

pslr_min = (kappa - 1) / (N + kappa - 1) + ...
    (N * (N + 2*kappa - 2) / (N + kappa - 1)) / ...
    ((N + kappa - 1) * CV_max^2 + kappa - 1);
islr_invariant = N - 1;

if apply_relax
    relax = get_param(params, 'direct_constraint_relax', 1e-5);
    pslr_min = pslr_min * (1 - relax);
end

end

function value = get_param(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end
