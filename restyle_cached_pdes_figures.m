function restyle_cached_pdes_figures()
% RESTYLE_CACHED_PDES_FIGURES  Re-export cached illumination figures.
% This changes presentation only and never reruns the CVX experiments.

sim_dir = fileparts(mfilename('fullpath'));
paper_dir = fullfile(sim_dir, '..', 'MyPaper', 'figures');
sim_fig_dir = fullfile(sim_dir, 'figures');
if exist(paper_dir, 'dir') ~= 7, mkdir(paper_dir); end
if exist(sim_fig_dir, 'dir') ~= 7, mkdir(sim_fig_dir); end

cfg = plot_config();
restyle_one(fullfile(sim_dir, 'beamgain_pdes_sweep.fig'), ...
    fullfile(sim_fig_dir, 'beamgain_pdes_sweep.pdf'), ...
    fullfile(paper_dir, 'beamgain_pdes_sweep.pdf'), cfg, [-90 90]);
restyle_one(fullfile(sim_dir, 'pdes_pareto_sweep.fig'), ...
    fullfile(sim_fig_dir, 'pdes_pareto_sweep.pdf'), ...
    fullfile(paper_dir, 'pdes_pareto_sweep.pdf'), cfg, []);
end

function restyle_one(fig_path, sim_pdf, paper_pdf, cfg, x_clip)
if exist(fig_path, 'file') ~= 2
    error('Missing cached figure: %s', fig_path);
end
fig = openfig(fig_path, 'invisible');
cleanup = onCleanup(@() close(fig));
plot_config(fig);

axes_list = findall(fig, 'Type', 'axes');
for idx = 1:numel(axes_list)
    ax = axes_list(idx);
    if strcmp(get(ax, 'Tag'), 'PDESLegendAxes') || strcmp(get(ax, 'Visible'), 'off')
        continue;
    end
    set(ax.Title, 'String', '');
    lines = findall(ax, 'Type', 'line');
    x_values = [];
    y_values = [];
    for line_idx = 1:numel(lines)
        x_values = [x_values; lines(line_idx).XData(:)]; %#ok<AGROW>
        y_values = [y_values; lines(line_idx).YData(:)]; %#ok<AGROW>
        if ~strcmpi(lines(line_idx).Marker, 'none')
            lines(line_idx).MarkerSize = cfg.marker_size;
        end
    end
    if isempty(x_clip)
        xlim(ax, valid_axis_limits(x_values, cfg));
        set(ax, 'PositionConstraint', 'innerposition', ...
            'Units', 'normalized', 'Position', [0.130 0.230 0.820 0.620]);
        set(ax.XLabel, 'Units', 'normalized', 'Position', [0.5 -0.145 0]);
        set(ax.YLabel, 'Units', 'normalized', 'Position', [-0.075 0.5 0]);
    else
        xlim(ax, valid_axis_limits(x_values, cfg, 'Clip', x_clip));
        set(ax, 'PositionConstraint', 'innerposition', ...
            'Units', 'normalized', 'Position', [0.100 0.245 0.850 0.590]);
        set(ax.XLabel, 'Units', 'normalized', 'Position', [0.5 -0.115 0]);
        set(ax.YLabel, 'Units', 'normalized', 'Position', [-0.055 0.5 0]);
    end
    ylim(ax, valid_axis_limits(y_values, cfg));
end

boxes = findall(fig, 'Type', 'rectangle');
for idx = 1:numel(boxes)
    try
        set(boxes(idx), 'FaceColor', cfg.legend_background_color, ...
            'FaceAlpha', cfg.legend_face_alpha, ...
            'EdgeColor', cfg.legend_edge_color);
    catch
    end
end

legend_axes = findall(fig, 'Type', 'axes', 'Tag', 'PDESLegendAxes');
if ~isempty(legend_axes)
    if isempty(x_clip)
        legend_position = [0.200 0.215 0.600 0.140];
    else
        legend_position = [0.100 0.648 0.300 0.210];
    end
    set(legend_axes, 'Units', 'normalized', 'Position', legend_position);
    legend_text = findall(legend_axes, 'Type', 'text');
    set(legend_text, 'FontSize', cfg.pdes_legend_font);
end

tight_export_figure(fig, sim_pdf, 'ContentType', 'image', ...
    'Resolution', cfg.export_resolution, 'TightPad', 3, ...
    'TightLayout', false);
copyfile(sim_pdf, paper_pdf, 'f');
clear cleanup;
end
