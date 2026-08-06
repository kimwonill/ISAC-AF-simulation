function cfg = plot_config(target)
% PLOT_CONFIG  Shared visual defaults for paper figures.
%
% Usage:
%   cfg = plot_config();
%   plot_config(fig);   % also applies common font defaults to a figure

cfg = struct();
cfg.font_name = 'Pretendard';
cfg.axes_font = 28;
cfg.label_font = 32;
cfg.title_font = 33;
cfg.legend_font = 30;
cfg.panel_caption_font = 31;
cfg.axes_line_width = 1.1;
cfg.line_width = 2.0;
cfg.marker_size = 9.0;
cfg.scatter_size = cfg.marker_size^2;
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

marker_objects = findall(target, '-property', 'MarkerSize');
for i = 1:numel(marker_objects)
    try
        if isprop(marker_objects(i), 'Marker') && ...
                strcmpi(get(marker_objects(i), 'Marker'), 'none')
            continue;
        end
        current_size = get(marker_objects(i), 'MarkerSize');
        if isscalar(current_size) && current_size < cfg.marker_size
            set(marker_objects(i), 'MarkerSize', cfg.marker_size);
        end
    catch
    end
end

scatter_list = findall(target, 'Type', 'scatter');
for i = 1:numel(scatter_list)
    try
        current_size = get(scatter_list(i), 'SizeData');
        set(scatter_list(i), 'SizeData', max(current_size, cfg.scatter_size));
    catch
    end
end

legend_list = findall(target, 'Type', 'legend');
for i = 1:numel(legend_list)
    try
        set(legend_list(i), 'FontName', cfg.font_name, ...
            'FontSize', cfg.legend_font);
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
