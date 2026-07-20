function Pn = directional_power_grid(W, A)
% DIRECTIONAL_POWER_GRID  Directional powers for all subcarriers/targets.
%
% Returns Pn(n,l) = a(theta_l)^T W_n a(theta_l)^*.

N = size(W, 3);
L = size(A, 2);
Pn = zeros(N, L);
for l = 1:L
    Pn(:, l) = compute_directional_power(W, A(:, l));
end

end
