function solver_iters = parse_cvx_solver_iterations(log_text)
% PARSE_CVX_SOLVER_ITERATIONS  Extract solver iteration counts from CVX logs.
%
% MOSEK reports one "Interior-point - iterations : ..." line for each
% optimizer call. CVX may call the solver more than once when log terms are
% handled by successive approximation, so the counts are accumulated.

solver_iters = NaN;
if isempty(log_text)
    return;
end

patterns = { ...
    'Interior-point\s*-\s*iterations\s*:\s*([0-9]+)', ...
    'number of iterations\s*=\s*([0-9]+)' ...
    };

values = [];
for p = 1:numel(patterns)
    tokens = regexp(log_text, patterns{p}, 'tokens');
    if ~isempty(tokens)
        values = cellfun(@(x) str2double(x{1}), tokens);
        break;
    end
end

values = values(isfinite(values));
if ~isempty(values)
    solver_iters = sum(values(:));
end
end
