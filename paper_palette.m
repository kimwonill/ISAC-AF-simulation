function colors = paper_palette(indices)
% PAPER_PALETTE  Ordered color palette shared by paper figures.
%
% Usage:
%   C = paper_palette();      % all colors
%   c = paper_palette(3);     % third color
%   C = paper_palette(1:4);   % first four colors

palette = [
    0.82 0.22 0.18  % 1 red
    0.16 0.39 0.72  % 2 blue
    0.10 0.52 0.42  % 3 green
    0.45 0.24 0.68  % 4 purple
    0.05 0.05 0.05  % 5 black
    0.92 0.58 0.09  % 6 orange
    0.00 0.49 0.57  % 7 teal
    0.36 0.36 0.36  % 8 gray
    0.95 0.75 0.10  % 9 yellow accent
    0.55 0.20 0.58  % 10 magenta
];

if nargin < 1 || isempty(indices)
    colors = palette;
    return;
end

indices = indices(:).';
colors = palette(1 + mod(indices - 1, size(palette, 1)), :);
end
