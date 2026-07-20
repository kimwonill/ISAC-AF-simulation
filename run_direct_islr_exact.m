function result = run_direct_islr_exact(H, islr_max, params, alpha0)
% RUN_DIRECT_ISLR_EXACT  Direct ISLR formulation solved via exact CV-SOC map.
%
% Unlike the PSLR direct constraint, the ISLR constraint has no relaxation
% loss after conversion to the CV SOC constraint. This wrapper keeps that
% direct-ISLR interpretation explicit while reusing the proposed SDP solver.

CV_max = cv_from_islr_threshold(islr_max, params);

if nargin >= 4
    result = run_proposed(H, CV_max, params, alpha0);
else
    result = run_proposed(H, CV_max, params);
end

result.equivalent_CV_max = CV_max;
result.direct_islr_max = islr_max;

end
