function plot_af_3d_variants()
% PLOT_AF_3D_VARIANTS  Preview non-heatmap AF visualizations.
%
% Uses the optimized Fig. 2 AF data and renders several 3-D alternatives.

clearvars; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(sim_dir);
fig_dir = fullfile(root_dir, 'figures');
data_dir = fullfile(sim_dir, 'results');
data_path = fullfile(data_dir, 'fig2_af_simulation_example.mat');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end
if exist(data_dir, 'dir') ~= 7, mkdir(data_dir); end
addpath(genpath(sim_dir));

if exist(data_path, 'file') ~= 2
    error('Missing optimized AF data: %s. Run plot_af_simulation_example first.', data_path);
end

S = load(data_path, 'ESL_dB', 'target_deg', 'CV_max');
Z = max(S.ESL_dB, -28);
N = size(Z, 1);
[TAU, NU] = meshgrid(0:N-1, 0:N-1);
Z_plot = Z.';

palette = paper_palette();
c_blue = palette(1, :);
c_red = palette(2, :);
c_gold = palette(3, :);
c_purple = palette(4, :);
c_gray = palette(8, :);
c_black = palette(10, :);

fig = figure('Color', 'w', 'Position', [80 80 1480 940]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl, 1);
surf(ax1, TAU, NU, Z_plot, ...
    'FaceColor', c_blue, 'EdgeColor', 'none', 'FaceAlpha', 0.90);
hold(ax1, 'on');
scatter3(ax1, 0, 0, Z_plot(1, 1), 90, c_red, 'filled', ...
    'MarkerEdgeColor', c_black, 'HandleVisibility', 'off');
style_3d_axes(ax1, '(a) Shaded surface', -28);
view(ax1, [-42 30]);
camlight(ax1, 'headlight');
lighting(ax1, 'gouraud');

ax2 = nexttile(tl, 2);
mesh(ax2, TAU, NU, Z_plot, ...
    'EdgeColor', c_blue, 'FaceAlpha', 0.10, 'LineWidth', 0.85);
hold(ax2, 'on');
contour3(ax2, TAU, NU, Z_plot, -28:4:0, ...
    'Color', c_red, 'LineWidth', 1.0);
style_3d_axes(ax2, '(b) Mesh with 3-D contours', -28);
view(ax2, [-38 34]);

ax3 = nexttile(tl, 3);
h_wf = waterfall(ax3, TAU, NU, Z_plot);
wf = [h_wf; findobj(ax3, 'Type', 'Surface')];
set(wf, 'EdgeColor', c_purple, 'FaceColor', 'none', 'LineWidth', 1.1);
style_3d_axes(ax3, '(c) Waterfall slices', -28);
view(ax3, [-33 32]);

ax4 = nexttile(tl, 4);
h_stem = stem3(ax4, TAU(:), NU(:), Z_plot(:), ...
    'Color', [c_gray 0.45], 'Marker', '.', 'MarkerSize', 7, ...
    'LineWidth', 0.7);
set(h_stem, 'HandleVisibility', 'off');
hold(ax4, 'on');
top_mask = true(size(Z_plot));
top_mask(1, 1) = false;
[~, order] = sort(Z_plot(top_mask), 'descend');
lin_idx = find(top_mask);
peak_idx = lin_idx(order(1:min(12, numel(order))));
scatter3(ax4, TAU(peak_idx), NU(peak_idx), Z_plot(peak_idx), ...
    70, c_gold, 'filled', 'MarkerEdgeColor', c_black, ...
    'DisplayName', 'Dominant sidelobes');
scatter3(ax4, 0, 0, Z_plot(1, 1), 95, c_red, 'filled', ...
    'MarkerEdgeColor', c_black, 'DisplayName', 'Mainlobe');
style_3d_axes(ax4, '(d) Sidelobe spike map', -28);
view(ax4, [-40 28]);
legend(ax4, 'Location', 'northeast', 'FontSize', 10);

title(tl, sprintf('Optimized AF 3-D Visualization Candidates (CV_{max}=%.1f)', S.CV_max), ...
    'FontSize', 17, 'FontWeight', 'bold');

out_png = fullfile(fig_dir, 'AF_3D_Visualization_Candidates.png');
out_pdf = fullfile(fig_dir, 'AF_3D_Visualization_Candidates.pdf');
exportgraphics(fig, out_png, 'Resolution', 300);
exportgraphics(fig, out_pdf, 'ContentType', 'image', 'Resolution', 300);

write_individual_figures(TAU, NU, Z_plot, fig_dir, c_blue, c_red, c_gold, c_purple, c_gray, c_black);

fprintf('Saved AF 3-D candidates: %s\n', out_png);
fprintf('Saved AF 3-D candidates: %s\n', out_pdf);
end

function style_3d_axes(ax, title_text, z_floor)
grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'Delay index \tau');
ylabel(ax, 'Doppler index \nu');
zlabel(ax, 'AF (dB)');
title(ax, title_text, 'FontSize', 13, 'FontWeight', 'bold');
xlim(ax, [0 15]); ylim(ax, [0 15]); zlim(ax, [z_floor 1]);
set(ax, 'FontSize', 10.5, 'Layer', 'top');
pbaspect(ax, [1 1 0.62]);
end

function write_individual_figures(TAU, NU, Z_plot, fig_dir, c_blue, c_red, c_gold, c_purple, c_gray, c_black)
z_floor = -28;

f1 = figure('Color', 'w', 'Position', [100 100 760 620]);
ax = axes(f1);
surf(ax, TAU, NU, Z_plot, 'FaceColor', c_blue, 'EdgeColor', 'none', 'FaceAlpha', 0.90);
hold(ax, 'on');
scatter3(ax, 0, 0, Z_plot(1, 1), 90, c_red, 'filled', ...
    'MarkerEdgeColor', c_black, 'HandleVisibility', 'off');
style_3d_axes(ax, 'Shaded surface', z_floor);
view(ax, [-42 30]); camlight(ax, 'headlight'); lighting(ax, 'gouraud');
exportgraphics(f1, fullfile(fig_dir, 'AF_3D_Surface.png'), 'Resolution', 300);

f2 = figure('Color', 'w', 'Position', [100 100 760 620]);
ax = axes(f2);
mesh(ax, TAU, NU, Z_plot, 'EdgeColor', c_blue, 'FaceAlpha', 0.10, 'LineWidth', 0.85);
hold(ax, 'on');
contour3(ax, TAU, NU, Z_plot, z_floor:4:0, 'Color', c_red, 'LineWidth', 1.0);
style_3d_axes(ax, 'Mesh with 3-D contours', z_floor);
view(ax, [-38 34]);
exportgraphics(f2, fullfile(fig_dir, 'AF_3D_Mesh_Contour.png'), 'Resolution', 300);

f3 = figure('Color', 'w', 'Position', [100 100 760 620]);
ax = axes(f3);
h_wf = waterfall(ax, TAU, NU, Z_plot);
wf = [h_wf; findobj(ax, 'Type', 'Surface')];
set(wf, 'EdgeColor', c_purple, 'FaceColor', 'none', 'LineWidth', 1.1);
style_3d_axes(ax, 'Waterfall slices', z_floor);
view(ax, [-33 32]);
exportgraphics(f3, fullfile(fig_dir, 'AF_3D_Waterfall.png'), 'Resolution', 300);

f4 = figure('Color', 'w', 'Position', [100 100 760 620]);
ax = axes(f4);
h_stem = stem3(ax, TAU(:), NU(:), Z_plot(:), 'Color', [c_gray 0.45], ...
    'Marker', '.', 'MarkerSize', 7, 'LineWidth', 0.7);
set(h_stem, 'HandleVisibility', 'off');
hold(ax, 'on');
top_mask = true(size(Z_plot));
top_mask(1, 1) = false;
[~, order] = sort(Z_plot(top_mask), 'descend');
lin_idx = find(top_mask);
peak_idx = lin_idx(order(1:min(12, numel(order))));
scatter3(ax, TAU(peak_idx), NU(peak_idx), Z_plot(peak_idx), ...
    70, c_gold, 'filled', 'MarkerEdgeColor', c_black);
scatter3(ax, 0, 0, Z_plot(1, 1), 95, c_red, 'filled', ...
    'MarkerEdgeColor', c_black);
style_3d_axes(ax, 'Sidelobe spike map', z_floor);
view(ax, [-40 28]);
exportgraphics(f4, fullfile(fig_dir, 'AF_3D_Sidelobe_Spikes.png'), 'Resolution', 300);
end
