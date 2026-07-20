function plot_ml_nn_pareto_nt4_n16()
% PLOT_ML_NN_PARETO_NT4_N16  Diagnostic overlay of NN policies on the NT=4,N=16 Pareto curve.
%
% This figure is for inspection only.  It overlays the existing SDP/SCA
% Monte Carlo Pareto data with the full-covariance NN validation averages.

sim_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(sim_dir, '..', 'figures');
result_dir = fullfile(sim_dir, 'results');
if exist(fig_dir, 'dir') ~= 7, mkdir(fig_dir); end

pareto_path = fullfile(result_dir, 'pareto_grid_2x4', 'pareto_grid_NT4_N16_MC10.mat');
e2e_path = fullfile(result_dir, 'ml_end_to_end_learning_results.mat');
safety_path = fullfile(result_dir, 'ml_cv_safety_layer_results.mat');
target_basis_path = fullfile(result_dir, 'ml_target_basis_cv_results.mat');
supervised_path = fullfile(result_dir, 'ml_supervised_cv_results.mat');

S = load(pareto_path);
E = load(e2e_path);
M = load(safety_path);
T = struct();
if exist(target_basis_path, 'file') == 2
    T = load(target_basis_path);
end
U = struct();
if exist(supervised_path, 'file') == 2
    U = load(supervised_path);
end
S = ensure_islr_equivalent_fields(S);

max_cv = max(S.CV_max_list(:));
E = trim_ml_to_cv_grid(E, max_cv, 'e2e_cv_grid', 'e2e_cv');
E = trim_ml_to_cv_grid(E, max_cv, 'e2e_direct_grid', 'e2e_direct');
M = trim_safety_to_cv_grid(M, max_cv);
T = trim_target_basis_to_cv_grid(T, max_cv);
U = trim_supervised_to_cv_grid(U, max_cv);

data = build_plot_data(S, E, M, T, U);

fig = figure('Position', [80 80 1420 620], 'Color', 'w');
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl, 1);
[handles, labels] = plot_metric_panel(ax1, data, 'pslr');
title(ax1, 'PSLR Pareto Overlay', 'FontSize', 15, 'FontWeight', 'bold');

ax2 = nexttile(tl, 2);
plot_metric_panel(ax2, data, 'islr');
title(ax2, 'ISLR Pareto Overlay', 'FontSize', 15, 'FontWeight', 'bold');

xlabel(tl, 'Sum-rate (bps/Hz)', 'FontSize', 14);
ylabel(tl, 'AF metric (dB)', 'FontSize', 14);
title(tl, '$N_T=4,\;N=16$; SDP/SCA uses MC=10, NN curves are separate Rayleigh validation averages', ...
    'Interpreter', 'latex', 'FontSize', 15);

lgd = legend(ax1, handles, labels, 'Orientation', 'horizontal', ...
    'NumColumns', 4, 'Location', 'southoutside', 'FontSize', 11);
try
    lgd.Layout.Tile = 'south';
catch
    lgd.Location = 'southoutside';
end

out_pdf = fullfile(fig_dir, 'ML_NN_Pareto_NT4_N16_Comparison.pdf');
out_png = fullfile(fig_dir, 'ML_NN_Pareto_NT4_N16_Comparison.png');
source_path = fullfile(result_dir, 'ml_nn_pareto_nt4_n16_comparison_source.mat');
save(source_path, 'data');
safe_export(fig, out_pdf, 'pdf');
safe_export(fig, out_png, 'png');
fprintf('Saved NN/Pareto comparison: %s\n', out_pdf);
fprintf('Saved NN/Pareto comparison: %s\n', out_png);
fprintf('Saved source data: %s\n', source_path);
end

function S = ensure_islr_equivalent_fields(S)
if ~isfield(S, 'direct_islr_exact_sumrate_grid')
    S.direct_islr_exact_sumrate_grid = S.sumrate_grid;
end
if ~isfield(S, 'direct_islr_exact_islr_lin_grid')
    S.direct_islr_exact_islr_lin_grid = S.islr_lin_grid;
end
if ~isfield(S, 'direct_islr_exact_pslr_lin_grid')
    S.direct_islr_exact_pslr_lin_grid = S.pslr_lin_grid;
end
end

function E = trim_ml_to_cv_grid(E, max_cv, grid_field, prefix)
cv = E.(grid_field)(:);
keep = cv <= max_cv + 1e-12;
E.(grid_field) = cv(keep);

fields = { ...
    sprintf('%s_sumrate_grid', prefix), ...
    sprintf('%s_pslr_lin_grid', prefix), ...
    sprintf('%s_islr_lin_grid', prefix), ...
    sprintf('%s_feas_grid', prefix)};
for i = 1:numel(fields)
    if isfield(E, fields{i})
        values = E.(fields{i})(:);
        E.(fields{i}) = values(keep);
    end
end
end

function M = trim_safety_to_cv_grid(M, max_cv)
cv = M.cv_grid(:);
keep = cv <= max_cv + 1e-12;
M.cv_grid = cv(keep);
fields = {'cv_safe_sumrate', 'cv_safe_pslr', 'cv_safe_islr', ...
          'cv_safe_cv_feas', 'cv_safe_direct_feas', 'cv_safe_all_cv_feas', ...
          'direct_sumrate', 'direct_pslr', 'direct_islr', ...
          'direct_cv_feas', 'direct_direct_feas', 'direct_all_direct_feas'};
for i = 1:numel(fields)
    if isfield(M, fields{i})
        values = M.(fields{i})(:);
        M.(fields{i}) = values(keep);
    end
end
end

function T = trim_target_basis_to_cv_grid(T, max_cv)
if isempty(fieldnames(T)) || ~isfield(T, 'cv_grid')
    return;
end
cv = T.cv_grid(:);
keep = cv <= max_cv + 1e-12;
T.cv_grid = cv(keep);
fields = {'target_cv_sumrate', 'target_cv_hard_sumrate', 'target_cv_pslr', ...
          'target_cv_islr', 'target_cv_cv_feas', 'target_cv_all_feas'};
for i = 1:numel(fields)
    if isfield(T, fields{i})
        values = T.(fields{i})(:);
        T.(fields{i}) = values(keep);
    end
end
end

function U = trim_supervised_to_cv_grid(U, max_cv)
if isempty(fieldnames(U)) || ~isfield(U, 'cv_grid')
    return;
end
cv = U.cv_grid(:);
keep = cv <= max_cv + 1e-12;
U.cv_grid = cv(keep);
fields = {'supervised_cv_sumrate', 'supervised_cv_soft_sumrate', ...
          'supervised_cv_pslr', 'supervised_cv_islr', ...
          'supervised_cv_alpha_acc', 'supervised_cv_feas', ...
          'label_sumrate', 'label_pslr', 'label_islr'};
for i = 1:numel(fields)
    if isfield(U, fields{i})
        values = U.(fields{i})(:);
        U.(fields{i}) = values(keep);
    end
end
end

function data = build_plot_data(S, E, M, T, U)
data.cv_grid = S.CV_max_list(:);
data.params = S.params;

data.prop.sumrate = row_mean(S.sumrate_grid);
data.prop.pslr = db(row_mean(S.pslr_lin_grid));
data.prop.islr = db(row_mean(S.islr_lin_grid));

data.direct.sumrate_pslr = row_mean(S.direct_sumrate_grid);
data.direct.pslr = db(row_mean(S.direct_pslr_lin_grid));
data.direct.sumrate_islr = row_mean(S.direct_islr_exact_sumrate_grid);
data.direct.islr = db(row_mean(S.direct_islr_exact_islr_lin_grid));

data.equiv.sumrate = row_mean(S.direct_equiv_sumrate_grid);
data.equiv.pslr = db(row_mean(S.direct_equiv_pslr_lin_grid));
data.equiv.islr = db(row_mean(S.direct_equiv_islr_lin_grid));

data.crb.sumrate = row_mean(S.crb_sumrate_grid);
data.crb.pslr = db(row_mean(S.crb_pslr_lin_grid));
data.crb.islr = db(row_mean(S.crb_islr_lin_grid));

data.mi.sumrate = row_mean(S.mi_sumrate_grid);
data.mi.pslr = db(row_mean(S.mi_pslr_lin_grid));
data.mi.islr = db(row_mean(S.mi_islr_lin_grid));

data.comm.sumrate = mean(S.comm_sumrate_grid(:), 'omitnan');
data.comm.pslr = db(mean(S.comm_pslr_lin_grid(:), 'omitnan'));
data.comm.islr = db(mean(S.comm_islr_lin_grid(:), 'omitnan'));

data.e2e_cv.cv = E.e2e_cv_grid(:);
data.e2e_cv.sumrate = E.e2e_cv_sumrate_grid(:);
data.e2e_cv.pslr = db(E.e2e_cv_pslr_lin_grid(:));
data.e2e_cv.islr = db(E.e2e_cv_islr_lin_grid(:));
data.e2e_cv.feas = E.e2e_cv_feas_grid(:);

data.e2e_direct.cv = E.e2e_direct_grid(:);
data.e2e_direct.sumrate = E.e2e_direct_sumrate_grid(:);
data.e2e_direct.pslr = db(E.e2e_direct_pslr_lin_grid(:));
data.e2e_direct.islr = db(E.e2e_direct_islr_lin_grid(:));
data.e2e_direct.feas = E.e2e_direct_feas_grid(:);

data.safe.cv = M.cv_grid(:);
data.safe.sumrate = M.cv_safe_sumrate(:);
data.safe.pslr = db(M.cv_safe_pslr(:));
data.safe.islr = db(M.cv_safe_islr(:));
data.safe.feas = M.cv_safe_cv_feas(:);

data.safe_direct.cv = M.cv_grid(:);
data.safe_direct.sumrate = M.direct_sumrate(:);
data.safe_direct.pslr = db(M.direct_pslr(:));
data.safe_direct.islr = db(M.direct_islr(:));
data.safe_direct.feas = M.direct_direct_feas(:);

data.has_target_basis = ~isempty(fieldnames(T)) && isfield(T, 'target_cv_sumrate');
if data.has_target_basis
    data.target_basis.cv = T.cv_grid(:);
    data.target_basis.sumrate = T.target_cv_sumrate(:);
    data.target_basis.hard_sumrate = T.target_cv_hard_sumrate(:);
    data.target_basis.pslr = db(T.target_cv_pslr(:));
    data.target_basis.islr = db(T.target_cv_islr(:));
    data.target_basis.feas = T.target_cv_cv_feas(:);
else
    data.target_basis.cv = [];
    data.target_basis.sumrate = [];
    data.target_basis.hard_sumrate = [];
    data.target_basis.pslr = [];
    data.target_basis.islr = [];
    data.target_basis.feas = [];
end

data.has_supervised = ~isempty(fieldnames(U)) && isfield(U, 'supervised_cv_sumrate');
if data.has_supervised
    data.supervised.cv = U.cv_grid(:);
    data.supervised.sumrate = U.supervised_cv_sumrate(:);
    data.supervised.pslr = db(U.supervised_cv_pslr(:));
    data.supervised.islr = db(U.supervised_cv_islr(:));
    data.supervised.feas = U.supervised_cv_feas(:);
    data.supervised.label_sumrate = U.label_sumrate(:);
    data.supervised.label_pslr = db(U.label_pslr(:));
    data.supervised.label_islr = db(U.label_islr(:));
else
    data.supervised.cv = [];
    data.supervised.sumrate = [];
    data.supervised.pslr = [];
    data.supervised.islr = [];
    data.supervised.feas = [];
    data.supervised.label_sumrate = [];
    data.supervised.label_pslr = [];
    data.supervised.label_islr = [];
end

data.pslr_bound = db(1 + S.params.N / (S.params.kappa - 1));
data.islr_bound = db((S.params.N - 1) * (S.params.N + 2*S.params.kappa - 2) / ...
    (2 * (S.params.N + S.params.kappa - 1)));
end

function [handles, labels] = plot_metric_panel(ax, data, metric)
axes(ax);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

blue = [0.18 0.43 0.78];
red = [0.82 0.22 0.18];
gray = [0.45 0.45 0.45];
purple = [0.45 0.25 0.65];
cyan = [0.00 0.49 0.57];
magenta = [0.55 0.20 0.58];
green = [0.12 0.55 0.25];
orange = [0.86 0.45 0.08];
navy = [0.05 0.18 0.33];
gold = [0.70 0.48 0.05];

if strcmpi(metric, 'pslr')
    y_prop = data.prop.pslr;
    y_direct = data.direct.pslr;
    x_direct = data.direct.sumrate_pslr;
    y_equiv = data.equiv.pslr;
    y_crb = data.crb.pslr;
    y_mi = data.mi.pslr;
    y_comm = data.comm.pslr;
    y_e2e_cv = data.e2e_cv.pslr;
    y_e2e_direct = data.e2e_direct.pslr;
    y_safe = data.safe.pslr;
    y_safe_direct = data.safe_direct.pslr;
    y_target_basis = data.target_basis.pslr;
    y_supervised = data.supervised.pslr;
    y_bound = data.pslr_bound;
    bound_label = 'Global bound';
    direction_note = 'higher is better';
else
    y_prop = data.prop.islr;
    y_direct = data.direct.islr;
    x_direct = data.direct.sumrate_islr;
    y_equiv = data.equiv.islr;
    y_crb = data.crb.islr;
    y_mi = data.mi.islr;
    y_comm = data.comm.islr;
    y_e2e_cv = data.e2e_cv.islr;
    y_e2e_direct = data.e2e_direct.islr;
    y_safe = data.safe.islr;
    y_safe_direct = data.safe_direct.islr;
    y_target_basis = data.target_basis.islr;
    y_supervised = data.supervised.islr;
    y_bound = data.islr_bound;
    bound_label = 'Global bound';
    direction_note = 'lower is better';
end

h_prop = plot(ax, data.prop.sumrate, y_prop, '-o', ...
    'Color', blue, 'MarkerFaceColor', blue, 'LineWidth', 2.3, ...
    'MarkerSize', 6.2, 'DisplayName', 'CV-SDP optimum');
h_direct = plot(ax, x_direct, y_direct, '--d', ...
    'Color', red, 'MarkerFaceColor', red, 'LineWidth', 2.0, ...
    'MarkerSize', 5.8, 'DisplayName', 'Direct SCA');
h_equiv = plot(ax, data.equiv.sumrate, y_equiv, 'o', ...
    'LineStyle', 'none', 'Color', [0.05 0.05 0.05], ...
    'MarkerFaceColor', 'none', 'LineWidth', 1.4, 'MarkerSize', 8.0, ...
    'DisplayName', 'ISLR-active point');
h_crb = plot(ax, data.crb.sumrate, y_crb, '-.^', ...
    'Color', gray, 'MarkerFaceColor', [0.70 0.70 0.70], ...
    'LineWidth', 1.1, 'MarkerSize', 4.8, 'DisplayName', 'CRB-inspired');
h_mi = plot(ax, data.mi.sumrate, y_mi, ':v', ...
    'Color', purple, 'MarkerFaceColor', [0.65 0.55 0.78], ...
    'LineWidth', 1.4, 'MarkerSize', 5.0, 'DisplayName', 'MI-inspired');
h_e2e_cv = plot(ax, data.e2e_cv.sumrate, y_e2e_cv, '-x', ...
    'Color', cyan, 'LineWidth', 1.8, 'MarkerSize', 7.0, ...
    'DisplayName', 'E2E CV-NN');
h_e2e_direct = plot(ax, data.e2e_direct.sumrate, y_e2e_direct, '--x', ...
    'Color', magenta, 'LineWidth', 1.8, 'MarkerSize', 7.0, ...
    'DisplayName', 'E2E Direct-NN');
h_safe = plot(ax, data.safe.sumrate, y_safe, '-p', ...
    'Color', green, 'MarkerFaceColor', green, 'LineWidth', 2.1, ...
    'MarkerSize', 7.0, 'DisplayName', 'CV-NN + safety');
h_safe_direct = plot(ax, data.safe_direct.sumrate, y_safe_direct, '--p', ...
    'Color', orange, 'MarkerFaceColor', orange, 'LineWidth', 2.0, ...
    'MarkerSize', 7.0, 'DisplayName', 'Direct-NN');
if data.has_target_basis
    h_target_basis = plot(ax, data.target_basis.sumrate, y_target_basis, '-h', ...
        'Color', navy, 'MarkerFaceColor', [0.35 0.55 0.75], 'LineWidth', 1.9, ...
        'MarkerSize', 6.8, 'DisplayName', 'Heavy target-basis CV-NN');
else
    h_target_basis = gobjects(0);
end
if data.has_supervised
    h_supervised = plot(ax, data.supervised.sumrate, y_supervised, '-s', ...
        'Color', gold, 'MarkerFaceColor', [0.95 0.75 0.20], 'LineWidth', 2.0, ...
        'MarkerSize', 6.5, 'DisplayName', 'Supervised CV-NN');
else
    h_supervised = gobjects(0);
end
h_comm = plot(ax, data.comm.sumrate, y_comm, 'kp', ...
    'MarkerFaceColor', [0.95 0.75 0.10], 'LineWidth', 1.5, ...
    'MarkerSize', 13, 'DisplayName', 'Communication-only');
h_comm_line = xline(ax, data.comm.sumrate, '-.', 'Color', [0 0 0], ...
    'LineWidth', 1.0, 'DisplayName', 'Comm. rate');
h_bound = yline(ax, y_bound, '--', 'Color', [0.15 0.15 0.15], ...
    'LineWidth', 1.0, 'DisplayName', bound_label);

handles = [h_prop h_direct h_equiv h_crb h_mi h_e2e_cv h_e2e_direct ...
           h_safe h_safe_direct h_target_basis h_supervised h_comm h_comm_line h_bound];
labels = get(handles, 'DisplayName');

x_all = [data.prop.sumrate; x_direct; data.equiv.sumrate; data.crb.sumrate; ...
         data.mi.sumrate; data.e2e_cv.sumrate; data.e2e_direct.sumrate; ...
         data.safe.sumrate; data.safe_direct.sumrate; data.target_basis.sumrate; ...
         data.supervised.sumrate; data.comm.sumrate];
y_all = [y_prop; y_direct; y_equiv; y_crb; y_mi; y_e2e_cv; y_e2e_direct; ...
         y_safe; y_safe_direct; y_target_basis; y_supervised; y_comm; y_bound];
pad_axes(ax, x_all, y_all, metric);

text(ax, 0.02, 0.04, direction_note, 'Units', 'normalized', ...
    'FontSize', 10, 'Color', [0.25 0.25 0.25]);
set(ax, 'FontSize', 11, 'Layer', 'top');
end

function values = row_mean(x)
values = mean(x, 2, 'omitnan');
values = values(:);
end

function y = db(x)
y = 10 * log10(max(real(x), realmin));
end

function pad_axes(ax, x, y, metric)
x = x(isfinite(x));
y = y(isfinite(y));
xr = max(x) - min(x);
yr = max(y) - min(y);
if xr <= 0, xr = max(1, abs(max(x)) * 0.05); end
if yr <= 0, yr = max(0.1, abs(max(y)) * 0.05); end

xlim(ax, [min(x) - 0.06*xr, max(x) + 0.06*xr]);
if strcmpi(metric, 'pslr')
    ylim(ax, [min(y) - 0.08*yr, max(y) + 0.08*yr]);
else
    ylim(ax, [min(y) - max(0.08*yr, 0.03), max(y) + max(0.10*yr, 0.03)]);
end
end

function safe_export(fig, filename, filetype)
try
    if strcmpi(filetype, 'pdf')
        exportgraphics(fig, filename, 'ContentType', 'image', 'Resolution', 450);
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
