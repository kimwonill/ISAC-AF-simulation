function [pslr_min, islr_max] = direct_thresholds_from_cv(CV_max, params, apply_relax)
% DIRECT_THRESHOLDS_FROM_CV  Convert a CV target to direct PSLR/ISLR limits.
%
% The proposed CV constraint guarantees
%   PSLR >= L(N,kappa,CV_max)
% and exactly maps to
%   ISLR <= ISLR(N,kappa,CV_max).
% These thresholds are used to compare against the direct SCA baseline under
% matched sensing requirements. Set apply_relax=false when the exact
% ISLR--CV equivalence is needed.

if nargin < 3
    apply_relax = true;
end

N = params.N;
kappa = params.kappa;

pslr_min = (kappa - 1) / (N + kappa - 1) + ...
    (N * (N + 2*kappa - 2) / (N + kappa - 1)) / ...
    ((N + kappa - 1) * CV_max^2 + kappa - 1);

islr_max = ((N * (2*kappa - 1) - 2*(kappa - 1)) * CV_max^2 + ...
    (N - 1) * (N + 2*kappa - 2)) / ...
    (2 * ((kappa - 1) * CV_max^2 + N + kappa - 1));

if apply_relax
    relax = get_param(params, 'direct_constraint_relax', 1e-5);
    pslr_min = pslr_min * (1 - relax);
    islr_max = islr_max * (1 + relax);
end

end

function value = get_param(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end
