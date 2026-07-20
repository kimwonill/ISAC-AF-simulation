function plot_af_surface_heatmap_corner()
% PLOT_AF_SURFACE_HEATMAP_CORNER  Shaded 3-D AF surface with heat projection.
%
% The optimized AF values are unchanged. The display grid is flipped so that
% the periodic mainlobe appears at the opposite delay-Doppler corner.

clearvars; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(sim_dir);
fig_dir = fullfile(root_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
data_path = fullfile(data_dir, 'fig2_af_simulation_example.mat');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
addpath(genpath(sim_dir));

if exist(data_path, 'file') ~= 2
    error('Missing optimized AF data: %s. Run plot_af_simulation_example first.', data_path);
end

S = load(data_path, 'ESL_dB', 'CV_max');
z_floor = -28;
Z = max(S.ESL_dB, z_floor).';
Z = rot90(Z, 2);
N = size(Z, 1);
[TAU, NU] = meshgrid(0:N-1, 0:N-1);

palette = paper_palette();
edge_color = 0.35 * palette(1, :);

fig = figure('Color', 'w', 'Position', [100 100 820 650]);
ax = axes(fig);
hold(ax, 'on');

surf(ax, TAU, NU, z_floor * ones(size(Z)), Z, ...
    'FaceColor', 'texturemap', 'EdgeColor', 'none', ...
    'FaceAlpha', 0.78, 'HandleVisibility', 'off');
surf(ax, TAU, NU, Z, Z, ...
    'FaceColor', 'interp', 'EdgeColor', edge_color, ...
    'LineWidth', 0.20, 'FaceAlpha', 0.94, ...
    'HandleVisibility', 'off');

colormap(ax, turbo(256));
clim(ax, [z_floor 0]);
cb = colorbar(ax);
cb.Label.String = 'AF (dB)';
cb.FontSize = 12;

grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'Delay index \tau');
ylabel(ax, 'Doppler index \nu');
zlabel(ax, 'AF (dB)');
title(ax, sprintf('Optimized AF Surface with Heat Projection, CV_{max}=%.1f', S.CV_max), ...
    'FontSize', 13, 'FontWeight', 'bold');
xlim(ax, [0 N-1]); ylim(ax, [0 N-1]); zlim(ax, [z_floor 1]);
xticks(ax, 0:5:N-1); yticks(ax, 0:5:N-1);
set(ax, 'FontSize', 12, 'Layer', 'top');
pbaspect(ax, [1 1 0.62]);
view(ax, [-44 30]);
camlight(ax, 'headlight');
lighting(ax, 'gouraud');

out_png = fullfile(fig_dir, 'AF_3D_Surface_Heatmap_Corner.png');
out_pdf = fullfile(fig_dir, 'AF_3D_Surface_Heatmap_Corner.pdf');
exportgraphics(fig, out_png, 'Resolution', 450);
exportgraphics(fig, out_pdf, 'ContentType', 'image', 'Resolution', 450);
fprintf('Saved corner-mainlobe AF surface: %s\n', out_png);
fprintf('Saved corner-mainlobe AF surface: %s\n', out_pdf);
end
