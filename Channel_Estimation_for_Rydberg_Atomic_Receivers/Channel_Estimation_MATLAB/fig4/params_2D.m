function p = params_2D()
% PARAMS_2D  Fixed physical/system parameters for Fig.4/Fig.5 (2D array),
% Xu et al. Sec.V. All values are paper-exact unless commented as an
% ASSUMPTION. Identical physical constants to the 1D case (fig3/params_1D.m)
% -- only the array geometry changes.

p.q    = 1.602e-19;        % electron charge [C]
p.a0   = 5.292e-11;        % Bohr radius [m]
p.h    = 6.626e-34;        % Planck constant [J.s]
p.hbar = p.h/(2*pi);       % reduced Planck constant

p.mu = 1785.9 * p.q * p.a0;    % |mu_eg|, dipole moment magnitude (mu_eg=[0,mu,0]^T)

p.I1 = 8;                  % 2D array size, Sec.V: "8 x 8 vapor elements"
p.I2 = 8;
p.K  = 3;                  % number of users

p.Lmin = 3;                % Lk ~ discrete Uniform{Lmin,...,Lmax}
p.Lmax = 7;

p.alpha_var  = 1;          % Var(alpha_l,k), alpha_l,k ~ CN(0,1)
p.alphab_var = 10;         % Var(alpha_b),   alpha_b   ~ CN(0,10)
p.eps_var    = 1/3;        % Var of each Cartesian component of polarization vectors

% ASSUMPTION: paper does not give inter-antenna spacing / wavelength numerically,
% nor the elevation/azimuth angle distributions for 2D paths. Half-wavelength
% spacing on both axes (matching fig3's 1D assumption) and isotropic elevation/
% azimuth (Uniform(0,2*pi)) are used -- phases only, doesn't affect NMSE power
% statistics.
p.d1_over_lambda = 0.5;
p.d2_over_lambda = 0.5;

end
