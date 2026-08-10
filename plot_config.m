function cfg = plot_config(target)
% PLOT_CONFIG  Shared visual defaults for paper figures.
%
% Usage:
%   cfg = plot_config();
%   plot_config(fig);   % also applies common font defaults to a figure

cfg = struct();
cfg.font_name = 'Times New Roman';
cfg.axes_font = 28;
cfg.label_font = 32;
cfg.title_font = 33;
cfg.legend_font = 30;
cfg.pdes_legend_font = max(cfg.legend_font - 9, 1);
cfg.panel_caption_font = 31;
cfg.axes_line_width = 1.1;
cfg.line_width = 2.2;
cfg.secondary_line_width = 1.4;
cfg.marker_size = 9.4;
cfg.compact_marker_size = 6.0;
cfg.simulation_marker_size = 5.0;
cfg.simulation_scatter_size = 22;
cfg.marker_edge_width = 0.8;
cfg.boundary_marker_size = 18;
cfg.scatter_size = cfg.marker_size^2;
cfg.band_face_alpha = 0.16;
cfg.region_face_alpha = 0.10;
cfg.grid_color = [0.78 0.78 0.78];
cfg.grid_alpha = 0.45;
cfg.grid_line_style = ':';
cfg.minor_grid_color = [0.86 0.86 0.86];
cfg.minor_grid_alpha = 0.22;
cfg.legend_background_color = [1 1 1];
cfg.legend_edge_color = [0.15 0.15 0.15];
cfg.legend_face_alpha = 0.94;
cfg.axis_padding_fraction = 0.05;
cfg.axis_minimum_span = 1e-6;
cfg.full_width_font_scale = 0.65;
cfg.tall_panel_font_scale = 18 / cfg.axes_font;
cfg.compact_panel_font_scale = 12 / cfg.axes_font;
cfg.tight_layout = true;
cfg.tile_spacing = 'compact';
cfg.tile_padding = 'tight';
cfg.export_padding = 0;
cfg.export_padding_units = 'points';
cfg.trim_png = true;
cfg.export_resolution = 600;
cfg.palette = [
    0.82 0.22 0.18
    0.16 0.39 0.72
    0.10 0.52 0.42
    0.45 0.24 0.68
    0.05 0.05 0.05
    0.92 0.58 0.09
    0.00 0.49 0.57
    0.36 0.36 0.36
    0.95 0.75 0.10
    0.55 0.20 0.58
];

set(groot, ...
    'defaultAxesFontName', cfg.font_name, ...
    'defaultTextFontName', cfg.font_name, ...
    'defaultLegendFontName', cfg.font_name, ...
    'defaultColorbarFontName', cfg.font_name, ...
    'defaultAxesFontSize', cfg.axes_font, ...
    'defaultTextFontSize', cfg.label_font, ...
    'defaultLegendFontSize', cfg.legend_font, ...
    'defaultColorbarFontSize', cfg.axes_font, ...
    'defaultAxesLineWidth', cfg.axes_line_width, ...
    'defaultLineMarkerSize', cfg.marker_size, ...
    'defaultScatterSizeData', cfg.scatter_size, ...
    'defaultAxesLabelFontSizeMultiplier', 1);

if nargin >= 1 && ~isempty(target)
    apply_common_style(target, cfg);
    if isgraphics(target, 'figure')
        setappdata(target, 'PlotConfigAppliedV1', true);
    end
end
end

function apply_common_style(target, cfg)
if isgraphics(target, 'figure')
    set(target, 'Color', 'w', 'PaperPositionMode', 'auto');
end

if cfg.tight_layout
    layout_list = findall(target, '-property', 'TileSpacing');
    for i = 1:numel(layout_list)
        try
            set(layout_list(i), ...
                'TileSpacing', cfg.tile_spacing, ...
                'Padding', cfg.tile_padding);
        catch
        end
    end
end

font_objects = findall(target, '-property', 'FontName');
for i = 1:numel(font_objects)
    try
        set(font_objects(i), 'FontName', cfg.font_name);
    catch
    end
end

axes_list = findall(target, 'Type', 'axes');
for i = 1:numel(axes_list)
    try
        set(axes_list(i), 'LineWidth', cfg.axes_line_width, ...
            'Layer', 'top', ...
            'FontSize', cfg.axes_font, ...
            'GridColor', cfg.grid_color, ...
            'GridAlpha', cfg.grid_alpha, ...
            'GridLineStyle', cfg.grid_line_style, ...
            'MinorGridColor', cfg.minor_grid_color, ...
            'MinorGridAlpha', cfg.minor_grid_alpha, ...
            'LabelFontSizeMultiplier', 1);
        set(get(axes_list(i), 'XLabel'), 'FontSize', cfg.label_font);
        set(get(axes_list(i), 'YLabel'), 'FontSize', cfg.label_font);
        set(get(axes_list(i), 'ZLabel'), 'FontSize', cfg.label_font);
        set(get(axes_list(i), 'Title'), 'FontSize', cfg.title_font);
        if cfg.tight_layout
            set(axes_list(i), 'LooseInset', get(axes_list(i), 'TightInset'));
        end
    catch
    end
end

legend_list = findall(target, 'Type', 'legend');
for i = 1:numel(legend_list)
    try
        set(legend_list(i), 'FontName', cfg.font_name, ...
            'FontSize', cfg.legend_font, ...
            'Color', cfg.legend_background_color, ...
            'EdgeColor', cfg.legend_edge_color, ...
            'Box', 'on');
    catch
    end
end

colorbar_list = findall(target, 'Type', 'colorbar');
for i = 1:numel(colorbar_list)
    try
        set(colorbar_list(i), 'FontName', cfg.font_name, ...
            'FontSize', cfg.axes_font);
        set(colorbar_list(i).Label, 'FontSize', cfg.label_font);
        set(colorbar_list(i).Title, 'FontSize', cfg.label_font);
    catch
    end
end
end
