function ax_leg = draw_pdes_legend(fig, position, labels, colors, line_styles, cfg, box_inset, num_columns)
% DRAW_PDES_LEGEND  Compact paper-style legend for the P_des figures.

if nargin < 6 || isempty(cfg)
    cfg = plot_config();
end
if nargin < 7 || isempty(box_inset)
    box_inset = 0.020;
end
if nargin < 8 || isempty(num_columns)
    num_columns = 1;
end
if isstring(labels)
    labels = cellstr(labels);
end
if ischar(line_styles)
    line_styles = repmat({line_styles}, 1, numel(labels));
end

ax_leg = axes(fig, 'Position', position, ...
    'XLim', [0 1], 'YLim', [0 1], ...
    'Tag', 'PDESLegendAxes', ...
    'Visible', 'off', ...
    'Color', 'none');
hold(ax_leg, 'on');
rectangle(ax_leg, 'Position', ...
    [box_inset box_inset 1 - 2*box_inset 1 - 2*box_inset], ...
    'FaceColor', 'w', ...
    'EdgeColor', [0.15 0.15 0.15], ...
    'LineWidth', cfg.axes_line_width);

legend_font = max(cfg.legend_font - 5, 1);
num_columns = max(1, min(round(num_columns), numel(labels)));
num_rows = ceil(numel(labels) / num_columns);
if num_rows == 1
    row_y = 0.5;
elseif num_columns == 1
    row_y = linspace(0.82, 0.18, num_rows);
else
    row_y = linspace(0.72, 0.28, num_rows);
end
column_width = 1 / num_columns;

for idx = 1:numel(labels)
    row_idx = floor((idx - 1) / num_columns) + 1;
    column_idx = mod(idx - 1, num_columns);
    x_offset = column_idx * column_width;
    x_line = x_offset + column_width * [0.075 0.255];
    x_center = x_offset + column_width * 0.165;
    x_text = x_offset + column_width * 0.310;
    y = row_y(row_idx);
    style = line_styles{idx};
    plot(ax_leg, x_line, [y y], style, ...
        'Color', colors(idx, :), ...
        'LineWidth', 1.8, ...
        'Clipping', 'off');

    if ~strcmp(style, ':')
        dx = column_width * 0.020;
        dy = 0.047;
        patch(ax_leg, x_center + [0 dx 0 -dx], ...
            y + [dy 0 -dy 0], ...
            colors(idx, :), ...
            'EdgeColor', colors(idx, :), ...
            'LineWidth', 1.2, ...
            'Clipping', 'off');
    end

    text(ax_leg, x_text, y, labels{idx}, ...
        'Interpreter', 'tex', ...
        'FontName', cfg.font_name, ...
        'FontSize', legend_font, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
end

try
    uistack(ax_leg, 'top');
catch
end
end
