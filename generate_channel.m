function H = generate_channel(params)
% GENERATE_CHANNEL  Rayleigh frequency-domain MISO channel.
%
% Returns H of size NT x K x N with H(:,k,n) ~ CN(0, I_NT).
% Assumes rich-scattering NLoS per the paper system model.

NT = params.NT; K = params.K; N = params.N;
H = (randn(NT, K, N) + 1j*randn(NT, K, N)) / sqrt(2);

end
