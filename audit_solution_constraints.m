function audit = audit_solution_constraints(metrics, params, limits)
% AUDIT_SOLUTION_CONSTRAINTS  Post-recovery feasibility residuals.
%
% Optional fields in limits are CV_max, PSLR_min, and ISLR_max. A positive
% residual denotes a violation. The common numerical tolerance is set by
% params.post_evd_feas_tol (default 1e-4).

if nargin < 3 || isempty(limits)
    limits = struct();
end

tol = get_param(params, 'post_evd_feas_tol', 1e-4);
audit.power_excess = max(metrics.power_total - params.P_max, 0);
audit.qos_shortfall = max([params.Q(:) - metrics.user_rate(:); 0]);
audit.illumination_shortfall = 0;
if ~isfield(limits, 'enforce_illumination') || limits.enforce_illumination
    audit.illumination_shortfall = max([ ...
        params.P_des - metrics.mean_directional_power(:); 0]);
end

audit.cv_excess = 0;
if isfield(limits, 'CV_max') && isfinite(limits.CV_max)
    audit.cv_excess = max([metrics.cv_per_target(:) - limits.CV_max; 0]);
end

audit.pslr_shortfall = 0;
if isfield(limits, 'PSLR_min') && isfinite(limits.PSLR_min)
    audit.pslr_shortfall = max([limits.PSLR_min - metrics.pslr_per_target(:); 0]);
end

audit.islr_excess = 0;
if isfield(limits, 'ISLR_max') && isfinite(limits.ISLR_max)
    audit.islr_excess = max([metrics.islr_per_target(:) - limits.ISLR_max; 0]);
end

audit.max_violation = max([audit.power_excess, audit.qos_shortfall, ...
    audit.illumination_shortfall, audit.cv_excess, ...
    audit.pslr_shortfall, audit.islr_excess]);
audit.tolerance = tol;
audit.is_feasible = audit.max_violation <= tol;
end

function value = get_param(params, name, default_value)
if isfield(params, name)
    value = params.(name);
else
    value = default_value;
end
end
