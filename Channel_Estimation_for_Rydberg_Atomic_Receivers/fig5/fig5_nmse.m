%% fig5_nmse.m
% Fig. 5: NMSE vs pilot length P for a 2D array (I1 x I2 = 8 x 8, K = 3 users)
% at SNR = 5 dB, PGD (415)-(429a) vs CRLB (454)/(457a).
% Equation numbers refer to ce_derivation_1D_2D_PGD_CRLB.pdf.
% Same model and estimator as fig4/fig4_nmse.m; only the sweep variable changes.
%
% Measurements are generated with the exact model (385), y = |sum_k g s + b + n|.
% PGD works on the linearised model (401). GD (no projection) is computed too and
% printed in the table for reference; the paper's Fig. 5 plots PGD and CRLB only.
% The exact real-parameter bound (463a) is also printed; it ignores the rank
% constraint and is infinite when P < 2K (fewer real equations per cell than
% real unknowns).
%
% Settings that can be given before running (defaults in brackets):
%   RSR_dB  reference-to-signal ratio in dB, NaN = paper alpha_b ~ CN(0,10)  [40]
%   POL     'path': eps_{k,l} per path, one eps_b for all cells, eq. (374)   ['path']
%           'cell': eps_{i1,i2,k,l} and eps_{b,i1,i2} per cell, Xu eq. (16)-(17)
%   T_gd, T_pgd  iterations, one number or one per entry of P_list           [3000]
%   L_proj  rank used in the PGD projection: NaN = true L_k of each user,
%           a number = same rank for every user (e.g. 7 = L_max, L_k unknown)  [NaN]
% Example:  T_gd = 50; T_pgd = 50; fig5_nmse

clearvars -except RSR_dB POL T_gd T_pgd L_proj; clc; rng(1);

%% Parameters
MC      = 100;                  % Monte Carlo trials per point
K       = 3;                    % users
I1      = 8;  I2 = 8;           % array size
P_list  = 5:5:50;               % pilot lengths
SNR_dB  = 5;
d1_lam  = 0.5;  d2_lam = 0.5;   % element spacings / lambda
if ~exist('RSR_dB', 'var'), RSR_dB = 40;     end
if ~exist('POL',    'var'), POL    = 'path'; end
if ~exist('T_gd',   'var'), T_gd   = 3000;   end
if ~exist('T_pgd',  'var'), T_pgd  = 3000;   end
if ~exist('L_proj', 'var'), L_proj = NaN;    end
tol = 1e-12;                    % relative stop rule ||G(t+1) - G(t)||^2 < tol ||G||^2

q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34;
mu_eg = [0; 1785.9*q*a0; 0];

nP = numel(P_list); Na = I1*I2;
T_gd  = T_gd.*ones(1, nP);  T_pgd = T_pgd.*ones(1, nP);
err_gd = zeros(1, nP); err_pgd = zeros(1, nP);
crb = zeros(1, nP); crb_ex = zeros(1, nP); energy = zeros(1, nP);

tic;
for ip = 1:nP
    P = P_list(ip);
    for mc = 1:MC
        [G3, S, B3, Lk] = gen_trial(K, I1, I2, P, RSR_dB, d1_lam, d2_lam, mu_eg, hbar, POL);
        A3 = S.'*G3;  Z3 = exp(-1j*angle(B3));              % (400), (399)
        energy(ip) = energy(ip) + norm(G3, 'fro')^2;

        sigma2 = mean(abs(A3(:)).^2)/10^(SNR_dB/10);        % complex noise variance (390)
        N3 = sqrt(sigma2/2)*(randn(P, Na) + 1j*randn(P, Na));
        Y3 = abs(A3 + B3 + N3);                             % (385), unfolded

        G_gd  = est_pgd(Y3, S, B3, Z3, Lk, I1, I2, T_gd(ip),  tol, false);
        if ~isnan(L_proj), Lk = L_proj*ones(1, K); end     % rank assumed by PGD
        G_pgd = est_pgd(Y3, S, B3, Z3, Lk, I1, I2, T_pgd(ip), tol, true);
        err_gd(ip)  = err_gd(ip)  + norm(G_gd  - G3, 'fro')^2;
        err_pgd(ip) = err_pgd(ip) + norm(G_pgd - G3, 'fro')^2;
        crb(ip)    = crb(ip) + 2*sigma2*Na*real(trace(inv(conj(S)*S.')));   % (454),(457)
        crb_ex(ip) = crb_ex(ip) + crb_exact(S, Z3, sigma2/2);               % (463a)
    end
    fprintf('P = %2d done (%.1f s)\n', P, toc);
end

nmse   = @(x) 10*log10(x./energy);                          % NMSE of Xu Sec. V, (457a)
NM_gd  = nmse(err_gd); NM_pgd = nmse(err_pgd);
NM_crb = nmse(crb);    NM_ex  = nmse(crb_ex);

%% Table
fprintf('\nSNR = %g dB   (NMSE in dB, %d trials)\n', SNR_dB, MC);
fprintf('   P     PGD      GD     CRLB(454)  exact(463a)\n');
for ip = 1:nP
    fprintf('  %2d  %7.2f  %7.2f  %8.2f  %9.2f\n', P_list(ip), ...
        NM_pgd(ip), NM_gd(ip), NM_crb(ip), NM_ex(ip));
end

%% Plot
figure('Color', 'w'); hold on; grid on; box on;
plot(P_list, NM_pgd, '-^', 'Color', [0 0.45 0.74], 'LineWidth', 1.2, ...
     'MarkerFaceColor', [0 0.45 0.74]);
plot(P_list, NM_crb, '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
xlabel('Pilot Length'); ylabel('NMSE [dB]');
legend('PGD', 'CRLB', 'Location', 'northeast');

if isnan(RSR_dB)
    tag = 'paperref'; rsr_txt = 'paper reference values';
else
    tag = sprintf('rsr%d', RSR_dB); rsr_txt = sprintf('RSR = %g dB', RSR_dB);
end
tag = [tag '_pol' POL];
if any(T_pgd ~= 3000)
    if all(T_pgd == T_pgd(1))
        tag = [tag sprintf('_Tpgd%d', T_pgd(1))];
    else
        tag = [tag '_Tpgd' sprintf('%d-', T_pgd)]; tag(end) = [];
    end
end
if ~isnan(L_proj), tag = [tag sprintf('_Lproj%d', L_proj)]; end
if all(T_pgd == T_pgd(1))
    it_txt = sprintf('PGD %d it.', T_pgd(1));
else
    it_txt = ['PGD it. per P: ' mat2str(T_pgd)];
end
if isnan(L_proj), rk_txt = 'projection rank L_k'; else, rk_txt = sprintf('projection rank %d', L_proj); end
title({sprintf('2D array, %d x %d, K = %d, SNR = %g dB, %s, polarization per %s', ...
    I1, I2, K, SNR_dB, rsr_txt, POL), sprintf('%s, %s, %d trials', it_txt, rk_txt, MC)});
xlim([P_list(1) P_list(end)]); xticks(P_list);

exportgraphics(gcf, ['fig5_nmse_' tag '.png'], 'Resolution', 200);
save(['fig5_nmse_' tag '.mat'], 'P_list', 'SNR_dB', 'NM_gd', 'NM_pgd', 'NM_crb', 'NM_ex', ...
     'MC', 'K', 'I1', 'I2', 'RSR_dB', 'POL', 'T_gd', 'T_pgd', 'L_proj');
fprintf('\nsaved fig5_nmse_%s.png and .mat\n', tag);

%% ------------------------------------------------------------------------
function [G3, S, B3, Lk] = gen_trial(K, I1, I2, P, RSR_dB, d1, d2, mu_eg, hbar, POL)
% 2D channel (370)-(380), mode-3 unfolding (398a), pilots (395), reference (383)
i1 = (0:I1-1).'; i2 = (0:I2-1).';
Lk = randi([3 7], 1, K);
G3 = zeros(K, I1*I2);
for k = 1:K
    L     = Lk(k);
    theta = pi*rand(1, L);  phi = 2*pi*rand(1, L);          % elevation, azimuth
    u     = 2*pi*d1*cos(theta);                             % (370)
    v     = 2*pi*d2*sin(theta).*cos(phi);
    alpha = (randn(1, L) + 1j*randn(1, L))/sqrt(2);         % CN(0,1)
    Gk = zeros(I1, I2);
    for l = 1:L
        a1 = exp(-1j*i1*u(l));  a2 = exp(-1j*i2*v(l));      % (376), (377)
        if strcmp(POL, 'path')
            c = (mu_eg.'*(sqrt(1/3)*randn(3, 1)))/hbar;     % beta/alpha, eq. (373)
        else
            c = reshape(mu_eg.'*(sqrt(1/3)*randn(3, I1*I2)), I1, I2)/hbar;
        end
        Gk = Gk + c.*alpha(l).*(a1*a2.');                   % (378), (380)
    end
    G3(k, :) = reshape(Gk, 1, []);                          % column-major, (398a)
end
S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);                 % CN(0,1)

% reference: one line-of-sight path, (383)
th_b = pi*rand; ph_b = 2*pi*rand;
u_b  = 2*pi*d1*cos(th_b);  v_b = 2*pi*d2*sin(th_b)*cos(ph_b);
alpha_b = sqrt(10)*(randn + 1j*randn)/sqrt(2);              % CN(0,10)
steer_b = exp(-1j*i1*u_b)*exp(-1j*i2*v_b).';                % I1 x I2
if strcmp(POL, 'path')
    c_b = (mu_eg.'*(sqrt(1/3)*randn(3, 1)))/hbar;
else
    c_b = reshape(mu_eg.'*(sqrt(1/3)*randn(3, I1*I2)), I1, I2)/hbar;
end
g_b = reshape(c_b.*alpha_b.*steer_b, 1, []);                % 1 x I1I2
B3  = ones(P, 1)*g_b;                                       % s_b,p = 1, P x I1I2
A3  = S.'*G3;
if ~isnan(RSR_dB)
    B3 = B3*sqrt(10^(RSR_dB/10)*mean(abs(A3(:)).^2)/mean(abs(B3(:)).^2));
end
end

function G3 = est_pgd(Y3, S, B3, Z3, Lk, I1, I2, T, tol, project)
% gradient step (415)-(418) with step size (418a); if project = true, followed by
% the rank-L_k projection of every user's I1 x I2 slice, (423)-(428)
K  = size(S, 1);
G3 = zeros(K, I1*I2);
zeta = 1/max(eig(S*S'));
Yc = Y3 - abs(B3);
for t = 1:T
    E = Yc - real(Z3.*(S.'*G3));                % (415)
    M = G3 + zeta*conj(S)*(E.*conj(Z3));        % (418)
    if project
        for k = 1:K
            Mk = reshape(M(k, :), I1, I2);      % (423)
            [U, Sg, V] = svd(Mk);               % (424)
            L = Lk(k);
            M(k, :) = reshape(U(:,1:L)*Sg(1:L,1:L)*V(:,1:L)', 1, []);   % (426), (428)
        end
    end
    if norm(M - G3, 'fro')^2 < tol*norm(M, 'fro')^2, G3 = M; break; end
    G3 = M;
end
end

function c = crb_exact(S, Z3, sr2)
% exact real-parameter bound (460)-(463a) for the unconstrained G_(3),
% summed over the I1 I2 cells; each cell sees y = Re(z .* (S^T g)) + noise
c = 0;
for q = 1:size(Z3, 2)
    Aq = diag(Z3(:, q))*S.';
    D  = [real(Aq), -imag(Aq)];
    F  = (D.'*D)/sr2;                           % (462)
    if rank(F) < size(F, 1), c = Inf; return; end
    c = c + trace(inv(F));                      % (463a)
end
end
