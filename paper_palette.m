function colors = paper_palette(indices)
% PAPER_PALETTE  Ordered color palette shared by paper figures.
%
% Usage:
%   C = paper_palette();      % all colors
%   c = paper_palette(3);     % third color
%   C = paper_palette(1:4);   % first four colors

cfg = plot_config();
palette = cfg.palette;

if nargin < 1 || isempty(indices)
    colors = palette;
    return;
end

indices = indices(:).';
colors = palette(1 + mod(indices - 1, size(palette, 1)), :);
end
