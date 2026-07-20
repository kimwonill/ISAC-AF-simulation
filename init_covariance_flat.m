function W = init_covariance_flat(params)
% INIT_COVARIANCE_FLAT  Isotropic feasible-ish covariance warm start.
%
% This is used only to provide a sensing-feasible linearization point when a
% better SCA warm start is unavailable.

W = zeros(params.NT, params.NT, params.N);
Wn = (params.P_max / params.N / params.NT) * eye(params.NT);
for n = 1:params.N
    W(:, :, n) = Wn;
end

end
