function standardize_pdes_figures_from_fig()
% STANDARDIZE_PDES_FIGURES_FROM_FIG  Re-export cached P_des figures with
% the paper-wide font scale without re-running the CVX experiments.

clearvars; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, 'figures');
if exist(fig_dir, 'dir') ~= 7
    mkdir(fig_dir);
end

standardize_one( ...
    fullfile(sim_dir, 'beamgain_pdes_sweep.fig'), ...
    fullfile(fig_dir, 'beamgain_pdes_sweep.pdf'), ...
    fullfile(fig_dir, 'beamgain_pdes_sweep.png'));

standardize_one( ...
    fullfile(sim_dir, 'pdes_pareto_sweep.fig'), ...
    fullfile(fig_dir, 'pdes_pareto_sweep.pdf'), ...
    fullfile(fig_dir, 'pdes_pareto_sweep.png'));
end

function standardize_one(fig_path, pdf_path, png_path)
if exist(fig_path, 'file') ~= 2
    error('Missing cached figure: %s', fig_path);
end

fig = openfig(fig_path, 'invisible');
set(fig, 'Color', 'w', 'PaperPositionMode', 'auto');
cfg = plot_config(fig);
palette = paper_palette();
[~, stem] = fileparts(fig_path);
is_pareto = contains(lower(stem), 'pareto');

if is_pareto
    legend_position = [0.285 0.215 0.460 0.140];
else
    legend_position = [0.100 0.648 0.235 0.210];
end
set(fig, 'Position', [100 100 1040 560]);
axes_position = [0.100 0.205 0.850 0.655];
axes_font = cfg.axes_font;
label_font = cfg.label_font;
title_font = cfg.panel_caption_font;

delete(findall(fig, 'Type', 'legend'));
delete(findall(fig, 'Type', 'axes', 'Tag', 'PDESLegendAxes'));
delete(findall(fig, 'Tag', 'PDESCV09Highlight'));
axes_list = findall(fig, 'Type', 'axes');
for i = 1:numel(axes_list)
    ax = axes_list(i);
    set(ax, 'FontName', cfg.font_name, 'FontSize', axes_font, ...
        'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1, ...
        'LineWidth', cfg.axes_line_width, 'Layer', 'top', ...
        'Units', 'normalized', 'Position', axes_position);
    if ~isempty(ax.XLabel)
        set(ax.XLabel, 'FontName', cfg.font_name, 'FontSize', label_font, ...
            'Units', 'normalized', 'Position', [0.5 -0.125 0]);
    end
    if ~isempty(ax.YLabel)
        if is_pareto
            ylabel_position = [-0.040 0.5 0];
        else
            ylabel_position = [-0.075 0.5 0];
            set(ax.YLabel, 'String', 'Directional power (dB)');
        end
        set(ax.YLabel, 'FontName', cfg.font_name, 'FontSize', label_font, ...
            'Units', 'normalized', 'Position', ylabel_position);
    end
    if ~isempty(ax.Title)
        set(ax.Title, 'FontName', cfg.font_name, 'FontSize', title_font, ...
            'FontWeight', 'normal');
    end
    if is_pareto
        xlim(ax, [147.7 166.8]);
        xticks(ax, 148:3:166);
    else
        xlim(ax, [-93 96]);
        xticks(ax, -90:30:90);
    end
end

if is_pareto
    line_style_objects = findall(fig, '-property', 'LineStyle');
    for j = 1:numel(line_style_objects)
        if strcmp(get(line_style_objects(j), 'LineStyle'), ':')
            delete(line_style_objects(j));
        end
    end
end

line_list = findall(fig, 'Type', 'line');
for j = 1:numel(line_list)
    name = get(line_list(j), 'DisplayName');
    if contains(name, 'P_{des}')
        color_idx = pdes_color_index(name);
        color = palette(color_idx, :);
        set(line_list(j), ...
            'LineStyle', '-', ...
            'LineWidth', 2.2, ...
            'Marker', 'd', ...
            'MarkerSize', 9.4, ...
            'MarkerFaceColor', color, ...
            'MarkerEdgeColor', color, ...
            'Color', color);
    end
end

text_list = findall(fig, 'Type', 'text');
for i = 1:numel(text_list)
    set(text_list(i), 'FontName', cfg.font_name);
end

legend_labels = { ...
    'P_{des}=0', ...
    'P_{des}=P_{max}/N', ...
    'P_{des}=2P_{max}/N', ...
    'P_{des}=3P_{max}/N'};
legend_colors = palette(1:4, :);
legend_styles = repmat({'-'}, 1, 4);
if is_pareto
    legend_box_inset = 0.020;
    legend_num_columns = 2;
else
    legend_box_inset = 0;
    legend_num_columns = 1;
end
draw_pdes_legend(fig, legend_position, legend_labels, ...
    legend_colors, legend_styles, cfg, legend_box_inset, ...
    legend_num_columns);

tight_export_figure(fig, pdf_path, 'ContentType', 'image', ...
    'Resolution', cfg.export_resolution, 'TightPad', 0);
tight_export_figure(fig, png_path, 'Resolution', cfg.export_resolution, ...
    'TightPad', 0);
close(fig);
fprintf('Standardized: %s\n', pdf_path);
end

function color_idx = pdes_color_index(name)
if contains(name, '3P_{max}')
    color_idx = 4;
elseif contains(name, '2P_{max}')
    color_idx = 3;
elseif contains(name, 'P_{max}/N')
    color_idx = 2;
else
    color_idx = 1;
end
end
