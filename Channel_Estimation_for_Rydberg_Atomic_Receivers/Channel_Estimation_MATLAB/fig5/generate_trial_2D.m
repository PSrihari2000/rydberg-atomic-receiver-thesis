function [Gflat, S, Bflat, Lk] = generate_trial_2D(p, P)
% GENERATE_TRIAL_2D  Draw one Monte Carlo realization of the 2D channel,
% pilot matrix, and complex reference term, Eq.(16)-(17).
%
% Inputs:
%   p - struct from params_2D(), with an added field p.sb2 (reference
%       pilot power, set by the caller to hit the desired |b|>>|a+n| margin)
%   P - pilot length
%
% Outputs:
%   Gflat (I1*I2 x K complex) - true channel, FLATTENED with i1 cycling
%                                fastest (column-major reshape convention,
%                                matches G_(3)' in the paper's mode-3
%                                unfolding, Eq.20). Row idx = i1+(i2-1)*I1.
%   S     (K x P complex)     - pilot matrix, entries iid CN(0,1)
%   Bflat (I1*I2 x P complex) - true complex reference term b_{i1,i2,p}, Eq.(17)
%   Lk    (K x 1)             - realized number of paths per user (also the
%                                TRUE rank of each user's I1xI2 channel slice,
%                                Sec.III-B -- PGD is given this as a known input,
%                                matching Algorithm 1's own input list)

I1 = p.I1; I2 = p.I2; K = p.K;
i1idx = (0:I1-1).';   % I1 x 1
i2idx = (0:I2-1).';   % I2 x 1

% ---- channel G, Eq.(16); each path contributes an EXACT RANK-1 I1xI2 outer
% product (phase_i1 * phase_i2), which is why each user's slice has rank<=Lk
% (Sec.III-B) -- built explicitly that way here rather than incidentally. ----
Lk = randi([p.Lmin, p.Lmax], K, 1);
Gflat = zeros(I1*I2, K);
for k = 1:K
    Lkk = Lk(k);
    theta = 2*pi*rand(Lkk,1);              % elevation (ASSUMPTION: isotropic)
    phi   = 2*pi*rand(Lkk,1);              % azimuth   (ASSUMPTION: isotropic)
    u = 2*pi*p.d1_over_lambda*cos(theta);              % Eq.16
    v = 2*pi*p.d2_over_lambda*sin(theta).*cos(phi);    % Eq.16

    eps_y = sqrt(p.eps_var)*randn(Lkk,1);  % mu_eg^T*eps = mu*eps_y (only y-component nonzero)
    alpha = sqrt(p.alpha_var/2)*(randn(Lkk,1)+1i*randn(Lkk,1));  % CN(0,1)
    coeff = (p.mu/p.hbar) * eps_y .* alpha;   % Lkk x 1

    Gslice = zeros(I1, I2);
    for l = 1:Lkk
        phase_i1 = exp(-1i * i1idx * u(l));     % I1 x 1
        phase_i2 = exp(-1i * i2idx * v(l)).';   % 1  x I2
        Gslice = Gslice + coeff(l) * (phase_i1 * phase_i2);  % rank-1 outer product
    end
    Gflat(:,k) = reshape(Gslice, I1*I2, 1);     % i1 fastest (MATLAB column-major)
end

% ---- pilots S, iid CN(0,1) ----
S = sqrt(1/2)*(randn(K,P) + 1i*randn(K,P));

% ---- reference term Bflat, Eq.(17) (single LOS path, no path sum) ----
theta_b = 2*pi*rand();
phi_b   = 2*pi*rand();
u_b = 2*pi*p.d1_over_lambda*cos(theta_b);
v_b = 2*pi*p.d2_over_lambda*sin(theta_b)*cos(phi_b);

eps_b_y = sqrt(p.eps_var)*randn(I1,I2);               % per-antenna polarization sample
alpha_b = sqrt(p.alphab_var/2)*(randn()+1i*randn());  % CN(0,10), single draw shared by all i1,i2,p
sb      = sqrt(p.sb2/2)*(randn(1,P)+1i*randn(1,P));   % reference pilot symbols, CN(0,sb2)

phase_b_i1 = exp(-1i * i1idx * u_b);     % I1 x 1
phase_b_i2 = exp(-1i * i2idx * v_b).';   % 1  x I2
coeff_b_grid = (p.mu/p.hbar) * eps_b_y .* alpha_b .* (phase_b_i1 * phase_b_i2);  % I1 x I2
coeff_b_flat = reshape(coeff_b_grid, I1*I2, 1);

Bflat = coeff_b_flat * sb;   % (I1*I2) x P (outer product)

end
