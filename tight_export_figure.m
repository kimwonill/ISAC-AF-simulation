function tight_export_figure(target, filename, varargin)
% TIGHT_EXPORT_FIGURE  Apply shared plot styling and export without margins.
%
% Usage:
%   tight_export_figure(fig, filename);
%   tight_export_figure(fig, filename, 'Resolution', 600);
%   tight_export_figure(fig, filename, 'ContentType', 'image', 'TightPad', 0);
%   tight_export_figure(fig, filename, 'TightLayout', false);
%
% TightPad is measured in cfg.export_padding_units. Exportgraphics options
% supplied by the caller override the shared padding defaults.

style_cache_key = 'PlotConfigAppliedV1';
if isgraphics(target, 'figure') && isappdata(target, style_cache_key)
    % Preserve any figure-specific adjustments made after plot_config(fig).
    cfg = plot_config();
else
    cfg = plot_config(target);
end
[tight_pad, trim_png, apply_tight_layout, export_args] = ...
    parse_options(cfg, varargin{:});

layout_cache_key = 'PlotConfigTightLayoutV1';
if apply_tight_layout && isgraphics(target, 'figure') && ...
        ~isappdata(target, layout_cache_key)
    tight_layout_figure(target);
    setappdata(target, layout_cache_key, true);
end

if ~has_option(export_args, 'Padding')
    export_args = [export_args, {'Padding', tight_pad}];
    if ~has_option(export_args, 'Units')
        export_args = [export_args, {'Units', cfg.export_padding_units}];
    end
end

drawnow;
exportgraphics(target, filename, export_args{:});

[~, ~, ext] = fileparts(filename);
if trim_png && tight_pad == 0 && strcmpi(ext, '.png')
    trim_png_whitespace(filename);
end
end

function [tight_pad, trim_png, apply_tight_layout, export_args] = ...
        parse_options(cfg, varargin)
tight_pad = cfg.export_padding;
trim_png = cfg.trim_png;
apply_tight_layout = cfg.tight_layout;
export_args = {};

i = 1;
while i <= numel(varargin)
    if (ischar(varargin{i}) || isstring(varargin{i})) && i < numel(varargin)
        name = char(varargin{i});
        if strcmpi(name, 'TightPad')
            tight_pad = varargin{i + 1};
            i = i + 2;
            continue;
        elseif strcmpi(name, 'TrimPNG')
            trim_png = varargin{i + 1};
            i = i + 2;
            continue;
        elseif strcmpi(name, 'TightLayout')
            apply_tight_layout = logical(varargin{i + 1});
            i = i + 2;
            continue;
        end
    end
    export_args{end + 1} = varargin{i}; %#ok<AGROW>
    i = i + 1;
end
end

function tf = has_option(args, option_name)
tf = false;
for i = 1:2:numel(args)
    if ischar(args{i}) || isstring(args{i})
        if strcmpi(char(args{i}), option_name)
            tf = true;
            return;
        end
    end
end
end

function tight_layout_figure(fig)
set_axes_loose_inset(fig);
drawnow;

% Text extents can shift slightly after the first position transform.
for iter = 1:2
    bounds = figure_content_bounds(fig);
    if isempty(bounds)
        return;
    end
    transform_top_level_positions(fig, bounds);
    set_axes_loose_inset(fig);
    drawnow;
end
end

function set_axes_loose_inset(fig)
axes_list = findall(fig, 'Type', 'axes');
for i = 1:numel(axes_list)
    try
        set(axes_list(i), 'LooseInset', get(axes_list(i), 'TightInset'));
    catch
    end
end
end

function bounds = figure_content_bounds(fig)
boxes = [];

axes_list = findall(fig, 'Type', 'axes');
for i = 1:numel(axes_list)
    boxes = append_box(boxes, axes_bounds(axes_list(i)));
end

position_types = {'colorbar', 'legend', 'textboxshape'};
for type_idx = 1:numel(position_types)
    obj_list = findall(fig, 'Type', position_types{type_idx});
    for obj_idx = 1:numel(obj_list)
        boxes = append_box(boxes, position_bounds(obj_list(obj_idx)));
    end
end

text_list = findall(fig, 'Type', 'text');
for i = 1:numel(text_list)
    boxes = append_box(boxes, text_bounds(text_list(i)));
end

if isempty(boxes)
    bounds = [];
    return;
end

left = min(boxes(:, 1));
bottom = min(boxes(:, 2));
right = max(boxes(:, 3));
top = max(boxes(:, 4));
if right <= left || top <= bottom
    bounds = [];
else
    bounds = [left bottom right top];
end
end

function b = axes_bounds(ax)
b = [];
try
    old_units = get(ax, 'Units');
    set(ax, 'Units', 'normalized');
    pos = get(ax, 'Position');
    if strcmpi(get(ax, 'Visible'), 'off')
        ti = [0 0 0 0];
    else
        ti = get(ax, 'TightInset');
    end
    set(ax, 'Units', old_units);
    b = [pos(1) - ti(1), pos(2) - ti(2), ...
         pos(1) + pos(3) + ti(3), pos(2) + pos(4) + ti(4)];
catch
end
end

function b = position_bounds(obj)
b = [];
try
    old_units = get(obj, 'Units');
    set(obj, 'Units', 'normalized');
    pos = get(obj, 'Position');
    set(obj, 'Units', old_units);
    if numel(pos) >= 4
        b = [pos(1), pos(2), pos(1) + pos(3), pos(2) + pos(4)];
    end
catch
end
end

function b = text_bounds(txt)
b = [];
try
    if strcmpi(get(txt, 'Visible'), 'off') || isempty(get(txt, 'String'))
        return;
    end

    old_units = get(txt, 'Units');
    set(txt, 'Units', 'normalized');
    ext = get(txt, 'Extent');
    parent = get(txt, 'Parent');
    set(txt, 'Units', old_units);

    parent_pos = parent_position(parent);
    if isempty(parent_pos) || numel(ext) < 4
        return;
    end

    b = [parent_pos(1) + ext(1) * parent_pos(3), ...
         parent_pos(2) + ext(2) * parent_pos(4), ...
         parent_pos(1) + (ext(1) + ext(3)) * parent_pos(3), ...
         parent_pos(2) + (ext(2) + ext(4)) * parent_pos(4)];
catch
end
end

function pos = parent_position(parent)
pos = [];
try
    if isgraphics(parent, 'figure')
        pos = [0 0 1 1];
        return;
    end
    if isprop(parent, 'Position')
        old_units = get(parent, 'Units');
        set(parent, 'Units', 'normalized');
        pos = get(parent, 'Position');
        set(parent, 'Units', old_units);
    end
catch
end
end

function boxes = append_box(boxes, b)
if isempty(b) || any(~isfinite(b)) || b(3) <= b(1) || b(4) <= b(2)
    return;
end
boxes = [boxes; b];
end

function transform_top_level_positions(fig, bounds)
left = bounds(1);
bottom = bounds(2);
width = bounds(3) - bounds(1);
height = bounds(4) - bounds(2);
if width <= 0 || height <= 0
    return;
end

objects = {};

layout_list = findall(fig, '-property', 'TileSpacing');
for i = 1:numel(layout_list)
    objects{end + 1} = layout_list(i); %#ok<AGROW>
end

axes_list = findall(fig, 'Type', 'axes');
for i = 1:numel(axes_list)
    if ~has_tiled_parent(axes_list(i))
        objects{end + 1} = axes_list(i); %#ok<AGROW>
    end
end

position_types = {'colorbar', 'legend', 'textboxshape'};
for type_idx = 1:numel(position_types)
    obj_list = findall(fig, 'Type', position_types{type_idx});
    for obj_idx = 1:numel(obj_list)
        objects{end + 1} = obj_list(obj_idx); %#ok<AGROW>
    end
end

% Cache every position before changing any object. Moving an axes can
% automatically move its legend; reading positions lazily would therefore
% apply the same transform twice and push legends beyond the canvas.
positions = cell(size(objects));
old_units = cell(size(objects));
valid = false(size(objects));
for i = 1:numel(objects)
    try
        old_units{i} = get(objects{i}, 'Units');
        set(objects{i}, 'Units', 'normalized');
        positions{i} = get(objects{i}, 'Position');
        set(objects{i}, 'Units', old_units{i});
        valid(i) = numel(positions{i}) >= 4;
    catch
    end
end

for i = 1:numel(objects)
    if ~valid(i)
        continue;
    end
    try
        pos = positions{i};
        pos(1) = (pos(1) - left) / width;
        pos(2) = (pos(2) - bottom) / height;
        pos(3) = pos(3) / width;
        pos(4) = pos(4) / height;
        set(objects{i}, 'Units', 'normalized', 'Position', pos);
        set(objects{i}, 'Units', old_units{i});
    catch
    end
end
end

function tf = has_tiled_parent(obj)
tf = false;
try
    parent = get(obj, 'Parent');
    tf = isprop(parent, 'TileSpacing');
catch
end
end

function trim_png_whitespace(filename)
try
    img = imread(filename);
    if ismatrix(img)
        return;
    end
    mask = any(img(:, :, 1:3) < 250, 3);
    [rows, cols] = find(mask);
    if isempty(rows) || isempty(cols)
        return;
    end
    img = img(min(rows):max(rows), min(cols):max(cols), :);
    imwrite(img, filename);
catch
end
end
