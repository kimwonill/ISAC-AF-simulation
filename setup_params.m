function params = setup_params()
% SETUP_PARAMS  System and algorithm parameters for the MISO-OFDM ISAC simulation.
%
% Physical scenario: 5G FR1 mid-band ISAC base station serving K=5 users
% while sensing L=2 stationary targets.

%% Array / waveform (realistic 5G FR1 setup)
params.fc        = 3.5e9;             % carrier frequency [Hz]
params.lambda    = 3e8 / params.fc;   % wavelength [m]
params.NT        = 4;                 % # TX antennas (ULA) -- reduced for DoF-constrained regime
params.dT        = params.lambda/2;   % half-wavelength spacing
params.N         = 16;                % # OFDM subcarriers

%% Users and targets
params.K         = 5;
params.L         = 4;
params.theta     = [-30, 0, 30, 60] * pi/180;  % target azimuths [rad]

%% Power / noise
% Normalized: P_max = 1, sigma2 chosen so nominal per-element SNR is ~20 dB.
params.P_max     = 1.0;
params.sigma2    = 1e-2;              % AWGN variance per receive subcarrier
params.kappa     = 1.32;              % 16-QAM kurtosis (paper)

%% QoS and illumination
% A light 1 bps/Hz/user floor is a common normalized QoS baseline in
% OFDMA/ISAC resource-allocation simulations. It prevents user starvation
% without dominating the sensing--communication Pareto sweep.
params.Q         = ones(params.K, 1);
params.P_des     = 0.8 * params.P_max / params.N;     % illumination floor (paper)

%% Algorithm
params.CV_max_list = [0, 0.1:0.1:0.9]; % use coarser grids for quick drafts
params.CV_max_pdes_plot = 0.5; % fixed CV limit for the P_des beam plot
params.max_iter  = 5;       % AO iterations
params.tol       = 1e-3;    % sum-rate convergence tolerance (bps/Hz)
params.num_mc    = 10;      % Monte Carlo channel realizations
params.sdp_quiet = true;    % suppress CVX banner per call
params.dual_max_iter = 50;   % subcarrier-allocation dual updates
params.dual_step0    = 0.5;  % diminishing step size: step0 / sqrt(i)
params.dual_tol      = 1e-4; % QoS violation tolerance in allocation
params.warm_start_cv = true; % initialize each CV point from the previous one
params.stop_if_alpha_unchanged = true; % skip redundant SDP solves at AO fixed points
params.run_direct_baseline = true; % Algorithm 2 baselines
params.direct_ao_max_iter = 5;
params.direct_sca_max_iter = 5;
params.direct_sca_tol = 1e-3;
params.direct_constraint_relax = 1e-5;
params.direct_pslr_target_mode = 'proposed'; % PSLR-active: match each proposed point's PSLR
params.direct_pslr_target_relax = 1e-4;
params.direct_pslr_active_islr_cv_gap = 0.2; % finite ISLR cap: c_I = CV_max + gap
params.direct_islr_max = Inf; % fallback override; ISLR-active curve overlaps proposed CV

end
