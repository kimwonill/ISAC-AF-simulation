function standardize_pdes_figures_from_fig()
% STANDARDIZE_PDES_FIGURES_FROM_FIG  Re-export cached P_des figures with
% the paper-wide font scale without re-running the CVX experiments.

clearvars; close all; clc;

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, '..', 'figures');
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
palette = paper_palette();

axes_list = findall(fig, 'Type', 'axes');
for i = 1:numel(axes_list)
    ax = axes_list(i);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 15, ...
        'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1);
    if ~isempty(ax.XLabel), set(ax.XLabel, 'FontName', 'Times New Roman', 'FontSize', 19); end
    if ~isempty(ax.YLabel), set(ax.YLabel, 'FontName', 'Times New Roman', 'FontSize', 19); end
    if ~isempty(ax.Title), set(ax.Title, 'FontName', 'Times New Roman', 'FontSize', 20); end
end

legend_list = findall(fig, 'Type', 'legend');
for i = 1:numel(legend_list)
    set(legend_list(i), 'FontName', 'Times New Roman', 'FontSize', 16.5);
end

line_list = findall(fig, 'Type', 'line');
line_names = get(line_list, 'DisplayName');
if ischar(line_names)
    line_names = {line_names};
end
named_mask = ~cellfun(@isempty, line_names);
named_lines = line_list(named_mask);
named_names = line_names(named_mask);
unnamed_lines = line_list(~named_mask);
unique_names = {};
if ~isempty(legend_list)
    legend_names = get(legend_list(1), 'String');
    if ischar(legend_names)
        legend_names = {legend_names};
    end
    unique_names = legend_names(:).';
end
remaining_names = unique(named_names, 'stable');
for k = 1:numel(remaining_names)
    if ~any(strcmp(unique_names, remaining_names{k}))
        unique_names{end + 1} = remaining_names{k}; %#ok<AGROW>
    end
end
for k = 1:numel(unique_names)
    color = palette(k, :);
    matching_lines = named_lines(strcmp(named_names, unique_names{k}));
    for j = 1:numel(matching_lines)
        set(matching_lines(j), 'Color', color);
        marker_face = get(matching_lines(j), 'MarkerFaceColor');
        if ~(ischar(marker_face) && strcmpi(marker_face, 'none'))
            set(matching_lines(j), 'MarkerFaceColor', color);
        end
    end
end
for j = 1:numel(unnamed_lines)
    if strcmp(get(unnamed_lines(j), 'LineStyle'), '--')
        set(unnamed_lines(j), 'Color', palette(5, :));
    end
end

text_list = findall(fig, 'Type', 'text');
for i = 1:numel(text_list)
    current_size = get(text_list(i), 'FontSize');
    set(text_list(i), 'FontName', 'Times New Roman');
    if current_size >= 20
        set(text_list(i), 'FontSize', 20);
    end
end

exportgraphics(fig, pdf_path, 'ContentType', 'image', 'Resolution', 1200);
exportgraphics(fig, png_path, 'Resolution', 1200);
close(fig);
fprintf('Standardized: %s\n', pdf_path);
end
