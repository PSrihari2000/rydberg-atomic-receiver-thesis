function [G, S, Bc, Lk] = generate_trial_1D(p, P)
% GENERATE_TRIAL_1D  Draw one Monte Carlo realization of the 1D channel,
% pilot matrix, and complex reference term.
%
% Inputs:
%   p - struct from params_1D(), with an added field p.sb2 (reference
%       pilot power, set by the caller to hit the desired |b|>>|a+n| margin)
%   P - pilot length
%
% Outputs:
%   G  (I x K complex)  - true channel, Eq.(7)
%   S  (K x P complex)  - pilot matrix, entries iid CN(0,1)
%   Bc (I x P complex)  - true complex reference term b_{i,p}, Eq.(9) (with
%                         the 1/hbar factor restored, see chat discussion)
%   Lk (K x 1)          - realized number of paths per user

I = p.I; K = p.K;
i_idx = (0:I-1).';

% ---- channel G, Eq.(7) ----
Lk = randi([p.Lmin, p.Lmax], K, 1);
G = zeros(I, K);
for k = 1:K
    Lkk = Lk(k);
    theta = 2*pi*rand(Lkk,1);                       % AOA per path (ASSUMPTION: isotropic)
    phi   = 2*pi*p.d_over_lambda*cos(theta);         % phase step per path
    eps_y = sqrt(p.eps_var)*randn(Lkk,1);            % mu_eg has only a y-component,
                                                      % so mu_eg^T*eps = mu*eps_y
    alpha = sqrt(p.alpha_var/2)*(randn(Lkk,1)+1i*randn(Lkk,1));  % CN(0,1)

    coeff  = (p.mu/p.hbar) * eps_y .* alpha;         % Lkk x 1
    Ephase = exp(-1i * i_idx * phi.');               % I x Lkk
    G(:,k) = Ephase * coeff;                         % sum over paths -> I x 1
end

% ---- pilots S, iid CN(0,1) ----
S = sqrt(1/2)*(randn(K,P) + 1i*randn(K,P));

% ---- reference term Bc, Eq.(9) (single LOS path, no path sum) ----
theta_b = 2*pi*rand();
phi_b   = 2*pi*p.d_over_lambda*cos(theta_b);
eps_b_y = sqrt(p.eps_var)*randn(I,1);                 % per-antenna polarization sample
alpha_b = sqrt(p.alphab_var/2)*(randn()+1i*randn());  % CN(0,10), single draw shared by all i,p
sb      = sqrt(p.sb2/2)*(randn(1,P)+1i*randn(1,P));   % reference pilot symbols, CN(0,sb2)

coeff_b = (p.mu/p.hbar) * eps_b_y * alpha_b .* exp(-1i*i_idx*phi_b);  % I x 1
Bc = coeff_b * sb;                                     % I x P (outer product)

end
