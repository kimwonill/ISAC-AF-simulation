function limits = valid_axis_limits(values, cfg, varargin)
% VALID_AXIS_LIMITS  Tight finite-data limits with shared paper padding.
%
%   limits = valid_axis_limits(values, cfg)
%   limits = valid_axis_limits(values, cfg, 'Clip', [lower upper])
%   limits = valid_axis_limits(values, cfg, 'Include', reference_values)

parser = inputParser;
addParameter(parser, 'Clip', [-Inf Inf]);
addParameter(parser, 'Include', []);
addParameter(parser, 'PaddingFraction', cfg.axis_padding_fraction);
parse(parser, varargin{:});

all_values = [values(:); parser.Results.Include(:)];
all_values = all_values(isfinite(all_values));
clip = parser.Results.Clip;
if isempty(all_values)
    if all(isfinite(clip)) && clip(1) < clip(2)
        limits = clip;
    else
        limits = [0 1];
    end
    return;
end

lower = min(all_values);
upper = max(all_values);
span = upper - lower;
if span < cfg.axis_minimum_span
    span = max([abs(lower), abs(upper), 1]) * 0.10;
end
padding = parser.Results.PaddingFraction * span;
limits = [lower - padding, upper + padding];
limits(1) = max(limits(1), clip(1));
limits(2) = min(limits(2), clip(2));

if limits(1) >= limits(2)
    center = mean([lower upper]);
    half_span = max(span, cfg.axis_minimum_span) / 2;
    limits = [center - half_span, center + half_span];
    limits(1) = max(limits(1), clip(1));
    limits(2) = min(limits(2), clip(2));
end
end
