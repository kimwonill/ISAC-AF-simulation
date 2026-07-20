function plot_integrated_stress_ladder_preview()
% PLOT_INTEGRATED_STRESS_LADDER_PREVIEW
% Single-plot preview: feasibility degradation and computational burden
% ratios across increasingly stressful system settings.

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, '..', 'figures');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end

baseline_path = fullfile(sim_dir, 'results', 'computational_burden_NT4_N16_MC10.mat');
stress_path = fullfile(sim_dir, 'results', 'feasibility_stress_MC10.mat');
if exist(baseline_path, 'file') ~= 2
    error('Missing baseline data: %s', baseline_path);
end
if exist(stress_path, 'file') ~= 2
    error('Missing stress data: %s', stress_path);
end

B = load(baseline_path);
S = load(stress_path);
I = load_latest_interior_profile(sim_dir);

% Order the stress points to read like a stress ladder.
labels = ["Baseline", "Low DoF", "More Targets", "QoS + Illum."];
stress_order = [NaN, 2, 3, 1];
n_cases = numel(labels);

cv_feas = nan(1, n_cases);
direct_feas = nan(1, n_cases);
runtime_ratio = nan(1, n_cases);
ipm_iter_ratio = nan(1, n_cases);

[cv_feas(1), direct_feas(1), runtime_ratio(1), ipm_iter_ratio(1)] = ...
    summarize_baseline(B);

for k = 2:n_cases
    s = stress_order(k);
    [cv_feas(k), direct_feas(k), runtime_ratio(k)] = summarize_stress(S, s);
    if ~isempty(I)
        ipm_iter_ratio(k) = summarize_stress_interior(I, s);
    end
end

x = 1:n_cases;
blue = [0.13 0.37 0.70];
red = [0.80 0.19 0.17];
gold = [0.92 0.58 0.09];
purple = [0.45 0.24 0.68];
shade = [0.91 0.95 0.99];

fig = figure('Position', [100 100 1240 700], 'Color', 'w');
ax = axes(fig);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
set(ax, 'Layer', 'top');
ax.Position = [0.080 0.190 0.805 0.650];

yyaxis(ax, 'left');
fill(ax, [x, fliplr(x)], [100*cv_feas, fliplr(100*direct_feas)], shade, ...
    'FaceAlpha', 1.0, 'EdgeColor', 'none', 'HandleVisibility', 'off');
h_cv = plot(ax, x, 100*cv_feas, '-o', 'Color', blue, 'MarkerFaceColor', blue, ...
    'LineWidth', 2.8, 'MarkerSize', 8, 'DisplayName', 'CV-SDP feasibility');
h_direct = plot(ax, x, 100*direct_feas, '--d', 'Color', red, 'MarkerFaceColor', red, ...
    'LineWidth', 2.8, 'MarkerSize', 8, 'DisplayName', 'Direct SCA feasibility');
ylabel(ax, 'Feasibility rate (%)');
ylim(ax, [-5 105]);
ax.YColor = [0.10 0.10 0.10];

for k = 1:n_cases
    text(ax, x(k), 100*direct_feas(k)-7.0, sprintf('%.0f%%', 100*direct_feas(k)), ...
        'HorizontalAlignment', 'center', 'Color', red, 'FontSize', 10, ...
        'FontWeight', 'bold', 'Clipping', 'on');
end

yyaxis(ax, 'right');
h_runtime = plot(ax, x, runtime_ratio, '-s', 'Color', gold, 'MarkerFaceColor', gold, ...
    'LineWidth', 2.6, 'MarkerSize', 8, 'DisplayName', 'Runtime ratio');
h_ipm = plot(ax, x, ipm_iter_ratio, '-^', 'Color', purple, 'MarkerFaceColor', purple, ...
    'LineWidth', 2.6, 'MarkerSize', 8, 'DisplayName', 'Total IPM iteration ratio');
yline(ax, 1, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.1, ...
    'HandleVisibility', 'off');
set(ax, 'YScale', 'log');
ymax = max([runtime_ratio(:); ipm_iter_ratio(:)], [], 'omitnan');
if isempty(ymax) || ~isfinite(ymax), ymax = 10; end
ylim(ax, [0.8, 1.7*ymax]);
ylabel(ax, 'Direct / CV burden ratio (x, log scale)');
ax.YColor = [0.10 0.10 0.10];

for k = 1:n_cases
    if isfinite(runtime_ratio(k))
        text(ax, x(k)-0.10, runtime_ratio(k)*1.12, sprintf('%.1fx', runtime_ratio(k)), ...
            'HorizontalAlignment', 'right', 'Color', gold, 'FontSize', 10, ...
            'FontWeight', 'bold');
    end
    if isfinite(ipm_iter_ratio(k))
        text(ax, x(k)+0.10, ipm_iter_ratio(k)*1.12, sprintf('%.1fx', ipm_iter_ratio(k)), ...
            'HorizontalAlignment', 'left', 'Color', purple, 'FontSize', 10, ...
            'FontWeight', 'bold');
    end
end

set(ax, 'XTick', x, 'XTickLabel', labels, 'FontSize', 12);
xlim(ax, [0.62 n_cases + 0.38]);
xtickangle(ax, 0);
xlabel(ax, 'System stress setting');
title(ax, 'Integrated Stress Preview: CV Reformulation vs Direct PSLR/ISLR SCA', ...
    'FontSize', 13, 'FontWeight', 'bold');

leg = legend(ax, [h_cv, h_direct, h_runtime, h_ipm], ...
    'Location', 'northoutside', 'NumColumns', 4);
leg.Box = 'off';

note = 'Feasibility/runtime: MC=10. IPM ratio: total interior-point iterations from solver profiling.';
if ~isempty(I) && isfield(I, 'num_mc')
    note = sprintf('%s Stress IPM profiling MC=%d.', note, I.num_mc);
end
annotation(fig, 'textbox', [0.100 0.030 0.78 0.040], 'String', note, ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 10, ...
    'Color', [0.30 0.30 0.30]);

out_pdf = fullfile(fig_dir, 'Integrated_Stress_Ladder_Preview.pdf');
out_png = fullfile(fig_dir, 'Integrated_Stress_Ladder_Preview.png');
safe_export(fig, out_png, 'png');
safe_export(fig, out_pdf, 'pdf');
fprintf('Saved integrated stress preview: %s\n', out_pdf);
fprintf('Saved integrated stress preview: %s\n', out_png);
end

function [cv_feas, direct_feas, runtime_ratio, ipm_iter_ratio] = summarize_baseline(B)
cv_ok = contains(string(B.prop_status_grid), 'Solved');
direct_ok = contains(string(B.direct_status_grid), 'Solved');
cv_feas = mean(cv_ok(:));
direct_feas = mean(direct_ok(:));

cv_t = B.prop_time_grid;
direct_t = B.direct_time_grid;
cv_t(~cv_ok) = NaN;
direct_t(~direct_ok) = NaN;
runtime_ratio = mean(direct_t(:), 'omitnan') / mean(cv_t(:), 'omitnan');

cv_ipm = B.prop_cvx_solver_iters_grid;
direct_ipm = B.direct_cvx_solver_iters_grid;
cv_ipm(~cv_ok) = NaN;
direct_ipm(~direct_ok) = NaN;
ipm_iter_ratio = mean(direct_ipm(:), 'omitnan') / mean(cv_ipm(:), 'omitnan');
end

function [cv_feas, direct_feas, runtime_ratio] = summarize_stress(S, s)
cv_ok = squeeze(S.prop_success(s, :, :));
direct_ok = squeeze(S.direct_success(s, :, :));
cv_feas = mean(cv_ok(:));
direct_feas = mean(direct_ok(:));

cv_t = squeeze(S.prop_time(s, :, :));
direct_t = squeeze(S.direct_time(s, :, :));
cv_t(~cv_ok) = NaN;
direct_t(~direct_ok) = NaN;
runtime_ratio = mean(direct_t(:), 'omitnan') / mean(cv_t(:), 'omitnan');
end

function ipm_iter_ratio = summarize_stress_interior(I, s)
cv_ipm = squeeze(I.prop_cvx_solver_iters(s, :, :));
direct_ipm = squeeze(I.direct_cvx_solver_iters(s, :, :));
cv_ipm = cv_ipm(isfinite(cv_ipm));
direct_ipm = direct_ipm(isfinite(direct_ipm));
if isempty(cv_ipm) || isempty(direct_ipm)
    ipm_iter_ratio = NaN;
else
    ipm_iter_ratio = mean(direct_ipm(:), 'omitnan') / mean(cv_ipm(:), 'omitnan');
end
end

function S = load_latest_interior_profile(sim_dir)
result_dir = fullfile(sim_dir, 'results');
files = dir(fullfile(result_dir, 'integrated_stress_interior_preview_MC*.mat'));
if isempty(files)
    S = [];
    warning('No stress interior profiling file found. Run run_integrated_stress_interior_preview first.');
    return;
end
[~, idx] = max([files.datenum]);
S = load(fullfile(files(idx).folder, files(idx).name));
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        tmp_png = [tempname(fileparts(filename)), '.png'];
        exportgraphics(fig, tmp_png, 'Resolution', 300);
        if ~png_to_pdf(tmp_png, filename)
            exportgraphics(fig, filename, 'ContentType', 'image', 'Resolution', 450);
        end
        if exist(tmp_png, 'file') == 2
            delete(tmp_png);
        end
    else
        exportgraphics(fig, filename, 'Resolution', 300);
    end
catch
    if strcmpi(filetype, 'pdf')
        print(fig, filename, '-dpdf', '-vector');
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
end

function ok = png_to_pdf(png_file, pdf_file)
ok = false;
python_exe = fullfile(getenv('USERPROFILE'), '.cache', 'codex-runtimes', ...
    'codex-primary-runtime', 'dependencies', 'python', 'python.exe');
if exist(python_exe, 'file') ~= 2
    return;
end

py_code = sprintf(['from PIL import Image; ', ...
    'img=Image.open(r''%s'').convert(''RGB''); ', ...
    'img.save(r''%s'', ''PDF'', resolution=300.0)'], png_file, pdf_file);
cmd = sprintf('"%s" -c "%s"', python_exe, py_code);
[status, ~] = system(cmd);
ok = (status == 0) && exist(pdf_file, 'file') == 2;
end
