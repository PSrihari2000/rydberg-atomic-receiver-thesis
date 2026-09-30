%% fig3_fixes_budget.m
% Fig. 3 of Xu et al., "Channel Estimation for Rydberg Atomic Receivers":
% NMSE vs SNR, 1D array, P = 10 and 30, GS vs GD vs CRLB,
% with GS and GD given the SAME iteration budget (fair comparison, equal effort).
% Same model and settings as fig3_paper_tuned.m except the iteration budget.
% Equation numbers refer to ce_derivation_1D_2D_PGD_CRLB.pdf, Xu equation
% numbers are marked "Xu".
%
% Stated in the paper (Sec. II, V) and used here:
%   I = 8 vapor cells, K = 3 users, L_k ~ U{3,...,7}, P = 10 and 30, SNR -5:5:30 dB
%   channel Xu (7) = (75), reference Xu (9) = (88)
%   polarization eps ~ N(0,1/3) per component
%   path gains alpha_{l,k} ~ CN(0,1), s_b = 1
%   dipole moment mu_eg = [0, 1785.9 q a0, 0]^T
%   pilots i.i.d. CN(0,1), initial G0 i.i.d. CN(0,0.1)
%   GD, Xu (13)-(15) = (216)-(228)
%   GS, Xu [10] Algorithm 1 = (351)-(369): spectral initialisation + iterations
%   CRLB, Xu (29)-(31) = (454), (457a)
%
% CHANGED from the paper (as in fig3_paper_tuned.m):
%   1. reference strength rescaled to RSR = 40 dB (alpha_b ~ CN(0,10) is drawn first)
%   2. polarization drawn once per path and once for the reference, the same in
%      every vapor cell, instead of per cell, Xu (7), (9)
%
% Iteration budget, the SAME for GD and GS at both P:
%   exactly T = 50 iterations, no early stop (T = t0 = 50 is the value used for GS
%   in [10]). One GD iteration and one GS iteration cost about the same (a few
%   I x K by K x P matrix products); GS also has a one-time spectral initialisation.
%
% Not stated in the paper, chosen here (same as fig3_paper_tuned.m):
%   SNR = E|a|^2 / sigma^2 with a = G S                          (90)
%   step size eta = 1/lambda_max(S S^H)                          (418a)
%   G0 ~ CN(0,0.1) is stated for PGD; it is used for GD here as well
%   element spacing lambda/2, angle of arrival ~ U(0,2*pi)
%   500 Monte Carlo trials; channel, pilots and reference drawn once per trial
%   and reused for every SNR
%
% The table gives 95% bootstrap confidence intervals over the trials and the
% share of trials in which each method had already converged within the 50
% iterations, checked with ||G(t+1) - G(t)||_F^2 < 1e-12 ||G(t+1)||_F^2 but
% without stopping (not part of the paper, printed as a check only).
%
% Measurements are generated with the exact magnitude model (94) = Xu (8);
% GD works on the linearised model (163) = Xu (12), GS on (94) directly.

clear; clc; rng(1);

%% Parameters
MC      = 500;                  % Monte Carlo trials per point
K       = 3;                    % users
I       = 8;                    % vapor cells
P_list  = [10 30];              % pilot lengths
SNR_dB  = -5:5:30;
RSR_dB  = 40;                   % CHANGED 1: reference-to-signal ratio
d_lam   = 0.5;                  % element spacing / lambda
T       = 50;                   % iterations, GD and GS, both P
tol     = 1e-12;                % convergence check only, no early stop

q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34;
mu_eg = [0; 1785.9*q*a0; 0];

nP = numel(P_list); nS = numel(SNR_dB);
e_gd = zeros(MC, nS, nP); e_gs = zeros(MC, nS, nP); e_crb = zeros(MC, nS, nP);
c_gd = false(MC, nS, nP);  c_gs = false(MC, nS, nP);       % converged within T
en   = zeros(MC, nP); rsr = zeros(MC, nP);

tic;
for ip = 1:nP
    P = P_list(ip);
    for mc = 1:MC
        [G, S, B] = gen_trial(K, I, P, RSR_dB, d_lam, mu_eg, hbar);
        A = G*S;  Z = exp(-1j*angle(B));                    % (91), (160)
        en(mc, ip)  = norm(G, 'fro')^2;
        rsr(mc, ip) = mean(abs(B(:)).^2)/mean(abs(A(:)).^2);
        Ea = mean(abs(A(:)).^2);

        for is = 1:nS
            sigma2 = Ea/10^(SNR_dB(is)/10);                 % complex noise variance (90)
            N = sqrt(sigma2/2)*(randn(I, P) + 1j*randn(I, P));
            Y = abs(A + B + N);                             % (94)

            G0 = sqrt(0.1/2)*(randn(I, K) + 1j*randn(I, K));    % G0 ~ CN(0,0.1)
            [G_gd, c_gd(mc, is, ip)] = est_gd(Y, S, B, Z, G0, T, tol);
            [G_gs, c_gs(mc, is, ip)] = est_gs(Y, S, B, T, tol);
            e_gd(mc, is, ip)  = norm(G_gd - G, 'fro')^2;
            e_gs(mc, is, ip)  = norm(G_gs - G, 'fro')^2;
            e_crb(mc, is, ip) = 2*sigma2*I*real(trace(inv(conj(S)*S.')));   % (454), (457)
        end
    end
    fprintf('P = %d done (%.1f s)\n', P, toc);
end

%% NMSE = sum of errors / sum of energies (457a), with 95% bootstrap intervals
bs = RandStream('mt19937ar', 'Seed', 7);                    % separate stream
NM_gd = zeros(nP, nS); NM_gs = zeros(nP, nS); NM_crb = zeros(nP, nS);
CI_gd = zeros(2, nS, nP); CI_gs = zeros(2, nS, nP); CI_crb = zeros(2, nS, nP);
for ip = 1:nP
    E = repmat(en(:, ip), 1, nS);
    [NM_gd(ip,:),  CI_gd(:,:,ip)]  = nmse_ci(e_gd(:,:,ip),  E, bs);
    [NM_gs(ip,:),  CI_gs(:,:,ip)]  = nmse_ci(e_gs(:,:,ip),  E, bs);
    [NM_crb(ip,:), CI_crb(:,:,ip)] = nmse_ci(e_crb(:,:,ip), E, bs);
end

%% Table
for ip = 1:nP
    fprintf('\nP = %d, RSR = %g dB, polarization per path, GD and GS %d it., %d trials (NMSE in dB, [95%% CI])\n', ...
        P_list(ip), RSR_dB, T, MC);
    fprintf('  SNR         GS                   GD                 CRLB (454)   converged within %d it. GS / GD\n', T);
    for is = 1:nS
        fprintf('  %3d  %6.2f [%6.2f,%6.2f]  %6.2f [%6.2f,%6.2f]  %7.2f      %5.1f %% / %5.1f %%\n', ...
            SNR_dB(is), NM_gs(ip,is), CI_gs(:,is,ip), NM_gd(ip,is), CI_gd(:,is,ip), NM_crb(ip,is), ...
            100*mean(c_gs(:,is,ip)), 100*mean(c_gd(:,is,ip)));
    end
end

%% Plot (as Xu Fig. 3: GS, GD and CRLB for P = 10 and 30)
figure('Color', 'w'); hold on; grid on; box on;
for ip = 1:nP
    plot(SNR_dB, NM_gs(ip,:),  '--d', 'Color', 'k', 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
    plot(SNR_dB, NM_gd(ip,:),  '-.s', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.5, ...
         'MarkerFaceColor', 'w');
    plot(SNR_dB, NM_crb(ip,:), '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);
end
% dashed ellipse around each pilot length, as in Xu Fig. 3
j = SNR_dB == 10;  mark_group(10, [NM_gs(1,j) NM_gd(1,j) NM_crb(1,j)], sprintf('P = %d', P_list(1)), 'above');
j = SNR_dB == 15;  mark_group(15, [NM_gs(2,j) NM_gd(2,j) NM_crb(2,j)], sprintf('P = %d', P_list(2)), 'left');
xlabel('SNR [dB]'); ylabel('NMSE [dB]');
legend('GS', 'GD', 'CRLB', 'Location', 'southwest');
title({sprintf('Same budget: 1D array, I = %d, K = %d, polarization per path, RSR = %g dB', I, K, RSR_dB), ...
       sprintf('GD and GS %d iterations each at both P; %d trials', T, MC)});
xlim([SNR_dB(1) SNR_dB(end)]); xticks(SNR_dB);

exportgraphics(gcf, 'fig3_fixes_budget.png', 'Resolution', 200);
save('fig3_fixes_budget.mat', 'P_list', 'SNR_dB', 'RSR_dB', 'NM_gs', 'NM_gd', 'NM_crb', ...
     'CI_gs', 'CI_gd', 'CI_crb', 'c_gs', 'c_gd', 'rsr', 'MC', 'K', 'I', 'T', 'tol');
fprintf('\nsaved fig3_fixes_budget.png and .mat\n');

%% ------------------------------------------------------------------------
function [G, S, B] = gen_trial(K, I, P, RSR_dB, d_lam, mu_eg, hbar)
% channel Xu (7) = (75), pilots (145), reference Xu (9) = (88) rescaled to RSR_dB,
% polarization drawn once per path and once for the reference
idx = (0:I-1).';
G = zeros(I, K);
for k = 1:K
    L     = randi([3 7]);                                   % L_k ~ U{3,...,7}
    phi   = 2*pi*d_lam*cos(2*pi*rand(1, L));                  % phase shift, AoA ~ U(0,2*pi)
    alpha = (randn(1, L) + 1j*randn(1, L))/sqrt(2);         % CN(0,1)
    eps_  = sqrt(1/3)*randn(3, L);                          % CHANGED 2: eps_{k,l}, one per path
    coup  = repmat(mu_eg.'*eps_, I, 1)/hbar;                % same in every cell
    G(:, k) = sum(coup.*alpha.*exp(-1j*idx*phi), 2);        % (75)
end
S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);                 % CN(0,1)

% reference: one path, (88)
phi_b   = 2*pi*d_lam*cos(2*pi*rand);
alpha_b = sqrt(10)*(randn + 1j*randn)/sqrt(2);              % CN(0,10)
eps_b   = repmat(sqrt(1/3)*randn(3, 1), 1, I);              % CHANGED 2: one eps_b
g_b     = (mu_eg.'*eps_b/hbar).'*alpha_b.*exp(-1j*idx*phi_b);
B       = g_b*ones(1, P);                                   % s_b,p = 1
A       = G*S;
B       = B*sqrt(10^(RSR_dB/10)*mean(abs(A(:)).^2)/mean(abs(B(:)).^2));   % CHANGED 1
end

function [G, converged] = est_gd(Y, S, B, Z, G0, T, tol)
% gradient descent on the linearised model, (216)-(228), starting from G0,
% exactly T iterations. converged = true if the change fell below the
% threshold at some iteration (checked only, no early stop).
G = G0;  converged = false;
eta = 1/max(eig(S*S'));                         % step size (418a)
Yc  = Y - abs(B);
for t = 1:T
    E     = Yc - real(Z.*(G*S));                % (220)
    G_new = G + eta*(E.*conj(Z))*S';            % (223)
    if norm(G_new - G, 'fro')^2 < tol*norm(G_new, 'fro')^2, converged = true; end
    G = G_new;
end
end

function [G, converged] = est_gs(Y, S, B, T, tol)
% biased Gerchberg-Saxton of [10], spectral init (351)-(358) + exactly T iterations
% (360)-(364). converged = true if the change fell below the threshold at some
% iteration (checked only, no early stop).
[I, ~] = size(Y); K = size(S, 1);
G = zeros(I, K);  converged = false;
for i = 1:I
    y  = Y(i, :).';
    Ab = [S.', B(i, :).'];                      % (351), P x (K+1)
    M  = Ab'*diag(y)*Ab;                        % (352) = sum_p y_p a_p a_p^H
    [V, D] = eig((M + M')/2);
    [~, m] = max(real(diag(D)));
    v  = V(:, m);                               % principal eigenvector (353)
    q  = abs(Ab*v);
    r  = (q.'*y)/(q.'*q);                       % (354)
    gt = exp(-1j*angle(r*v(end)))*r*v;          % (355)-(356)
    G(i, :) = gt(1:K).';                        % (357)-(358)
end
SSinv = S'/(S*S');
for t = 1:T
    X = G*S + B;                                % (360)
    R = Y.*exp(1j*angle(X));                    % (361)-(362)
    G_new = (R - B)*SSinv;                      % (363)-(364)
    if norm(G_new - G, 'fro')^2 < tol*norm(G_new, 'fro')^2, converged = true; end
    G = G_new;
end
end

function [nm, ci] = nmse_ci(e, en, bs)
% NMSE in dB per column (sum of errors / sum of energies) and its 95% bootstrap
% interval from 2000 resamples of the trials
[MC, n] = size(e);  nB = 2000;
nm = 10*log10(sum(e, 1)./sum(en, 1));
ci = nan(2, n);
for j = 1:n
    idx = randi(bs, MC, MC, nB);
    ej = e(:, j);  enj = en(:, j);
    r  = sort(10*log10(sum(ej(idx), 1)./sum(enj(idx), 1)));
    ci(:, j) = [r(round(0.025*nB)); r(round(0.975*nB))];
end
end

function mark_group(x0, y, txt, side)
% dashed ellipse around the curves of one pilot length at SNR = x0 (y = their
% NMSE values there), with the label txt 'right', 'left', 'below' or 'above'
rx = 0.9;  pad = 1.5;                           % half-width [dB SNR], margin [dB NMSE]
cy = (min(y) + max(y))/2;  ry = (max(y) - min(y))/2 + pad;
t  = linspace(0, 2*pi, 200);
plot(x0 + rx*cos(t), cy + ry*sin(t), 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
switch side
    case 'right', text(x0 + rx + 0.8, cy - 0.3*ry, txt, 'FontWeight', 'bold');
    case 'left',  text(x0 - rx - 0.8, cy, txt, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
    case 'below', text(x0, cy - ry - 2, txt, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    case 'above', text(x0, cy + ry + 1.5, txt, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end
end
