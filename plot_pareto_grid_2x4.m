function plot_pareto_grid_2x4(num_mc_override)
% PLOT_PARETO_GRID_2X4  Backward-compatible wrapper for the PSLR-only plot.
if nargin < 1
    num_mc_override = [];
end
plot_pareto_grid_1x4(num_mc_override);
end
