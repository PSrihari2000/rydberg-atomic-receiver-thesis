function p = params_1D()
% PARAMS_1D  Fixed physical/system parameters for Fig.3 (1D array), Xu et al.
% Sec.V. All values are paper-exact unless commented as an ASSUMPTION.

p.q    = 1.602e-19;        % electron charge [C]
p.a0   = 5.292e-11;        % Bohr radius [m]
p.h    = 6.626e-34;        % Planck constant [J.s]
p.hbar = p.h/(2*pi);       % reduced Planck constant

p.mu = 1785.9 * p.q * p.a0;    % |mu_eg|, dipole moment magnitude (mu_eg=[0,mu,0]^T)

p.I = 8;                   % number of antennas (1D array, Fig.3)
p.K = 3;                   % number of users

p.Lmin = 3;                % Lk ~ discrete Uniform{Lmin,...,Lmax}
p.Lmax = 7;

p.alpha_var  = 1;          % Var(alpha_l,k), alpha_l,k ~ CN(0,1)
p.alphab_var = 10;         % Var(alpha_b),   alpha_b   ~ CN(0,10)
p.eps_var    = 1/3;        % Var of each Cartesian component of polarization vectors

% ASSUMPTION: paper does not give inter-antenna spacing / wavelength numerically.
% Half-wavelength spacing is the standard array choice; AOA/path phases are random
% per path anyway, so this choice does not materially affect NMSE statistics.
p.d_over_lambda = 0.5;

end
