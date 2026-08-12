function config = cv_stress_rank1_config(num_mc)
%CV_STRESS_RANK1_CONFIG Shared configuration for the MC stress experiment.

if nargin < 1 || isempty(num_mc)
    num_mc = 100;
end
if ~isscalar(num_mc) || num_mc < 1 || num_mc ~= floor(num_mc)
    error('num_mc must be a positive integer.');
end

base_params = setup_params();
config.archive_version = 1;
config.experiment_id = 'cv-stress-rank1-gr10-v1';
config.result_schema_version = base_params.result_schema_version;
config.num_mc = num_mc;
config.CV_grid = 0.1:0.1:1.0;
config.num_cv = numel(config.CV_grid);
config.gaussian_randomization_trials = 10;
config.cvx_solver = 'mosek';
config.cvx_solver_threads = 1;
config.time_budget_seconds = [3; 4; 3];
config.budget_policy = ['Common wall-clock deadline for both methods: ' ...
    '3 s in S1/S3 and 4 s in S2. Runtime includes EVD/GR recovery.'];
config.timing_protocol = ['Quiet timed execution per method and point; runtime ' ...
    'includes optimization, EVD, audits, and GR. A deterministic untimed ' ...
    'solver-log replay collects IPM iterations.'];
config.ipm_protocol = ['Accumulated MOSEK interior-point iterations over all ' ...
    'AO/SCA and GR feasibility/refinement solves.'];
config.channel_seed_rule = '2000*scenario_index + mc_index';
config.recovery_seed_rule = ...
    '900000 + 10000*scenario_index + 100*mc_index + cv_index';
config.result_filename = sprintf( ...
    'cv_stress_rank1_S1S2S3_CV10_NT4_N16_MC%d.mat', num_mc);
config.run_directory = sprintf('cv_stress_rank1_MC%d', num_mc);

scenarios = struct([]);
scenarios(1).id = 'S1';
scenarios(1).short = 'S1-Illum';
scenarios(1).label = 'S1 Higher Illumination';
scenarios(1).NT = 4;
scenarios(1).N = 16;
scenarios(1).L = 4;
scenarios(1).theta = [-30, 0, 30, 60] * pi/180;
scenarios(1).Q = 1.00;
scenarios(1).Pdes_scale = 1.10;

scenarios(2).id = 'S2';
scenarios(2).short = 'S2-Tgts';
scenarios(2).label = 'S2 More Targets';
scenarios(2).NT = 4;
scenarios(2).N = 16;
scenarios(2).L = 6;
scenarios(2).theta = linspace(-60, 60, 6) * pi/180;
scenarios(2).Q = 1.25;
scenarios(2).Pdes_scale = 0.90;

scenarios(3).id = 'S3';
scenarios(3).short = 'S3-Joint';
scenarios(3).label = 'S3 Joint Stress';
scenarios(3).NT = 4;
scenarios(3).N = 16;
scenarios(3).L = 4;
scenarios(3).theta = [-30, 0, 30, 60] * pi/180;
scenarios(3).Q = 2.00;
scenarios(3).Pdes_scale = 1.10;
config.scenarios = scenarios;
end
