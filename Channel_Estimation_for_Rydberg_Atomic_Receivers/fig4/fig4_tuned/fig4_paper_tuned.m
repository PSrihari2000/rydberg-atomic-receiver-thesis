%% fig4_paper_tuned.m
% Fig. 4 of Xu et al., "Channel Estimation for Rydberg Atomic Receivers":
% NMSE vs SNR, 2D array, P = 10 and 30, GD vs PGD vs CRLB.
% Same equations and values as fig4_paper_literal.m, except the settings
% marked CHANGED below. Equation numbers refer to ce_derivation_1D_2D_PGD_CRLB.pdf,
% Xu equation numbers are marked "Xu".
%
% Stated in the paper (Sec. V) and used here:
%   array 8 x 8, K = 3 users, L_k ~ U{3,...,7}, P = 10 and 30, SNR -5:5:30 dB
%   path gains alpha_{k,l} ~ CN(0,1), s_b = 1
%   polarization eps ~ N(0,1/3) per component
%   dipole moment mu_eg = [0, 1785.9 q a0, 0]^T
%   pilots i.i.d. CN(0,1), initial G0 i.i.d. CN(0,0.1)
%   PGD, Xu Algorithm 1: gradient step (415)-(418), rank-L_k projection (423)-(428)
%     with the rank-L_k set of Xu (21), (25)
%   GD: the same gradient step without the projection
%   CRLB, Xu (29)-(31) = (454), (457a)
%
% CHANGED from the paper:
%   1. reference strength: alpha_b ~ CN(0,10) is drawn as in the paper, then the
%      reference is rescaled so that E|b|^2 / E|a|^2 = RSR = 40 dB. With the paper's
%      value the RSR is about -1 dB and the linearisation (401) does not hold.
%   2. polarization: eps drawn once per path (and once for the reference), the same
%      in every vapor cell, eq. (374), instead of per cell, Xu (16)-(17). Per cell,
%      every slice G_k is full rank and the rank-L_k model of PGD does not hold.
%   3. iterations: GD and PGD both stop after T = 50 iterations (the paper gives a
%      threshold but no value). With 50 iterations neither method has converged
%      at P = 10, which gives the error floor seen in the paper at P = 10.
%
% Differs from the printed equations:
%   the SVD in Xu (26)-(27) is written U Sigma V^T; for complex matrices the
%   reconstruction is U Sigma V^H, which is used here (424)-(426).
%
% Not stated in the paper, chosen here (same as fig4_paper_literal.m):
%   SNR = E|a|^2 / sigma^2 with a = S^T G_(3)                    (390)
%   step size zeta = 1/lambda_max(S S^H)                         (418a)
%   element spacing lambda/2, elevation ~ U(0,pi), azimuth ~ U(0,2pi)
%   500 Monte Carlo trials; channel, pilots and reference drawn once per trial
%   and reused for every SNR
%
% The table gives 95% bootstrap confidence intervals over the trials and the
% share of trials that reached the threshold within 50 iterations (not part of
% the paper, printed as a check only).
%
% Measurements are generated with the exact magnitude model (385);
% GD and PGD work on the linearised model (401), as in the paper.

clear; clc; rng(1);

%% Parameters
MC      = 500;                  % Monte Carlo trials per point
K       = 3;                    % users
I1      = 8;  I2 = 8;           % array size
P_list  = [10 30];              % pilot lengths
SNR_dB  = -5:5:30;
RSR_dB  = 40;                   % CHANGED 1: reference-to-signal ratio
d1_lam  = 0.5;  d2_lam = 0.5;   % element spacings / lambda
T_max   = 50;                   % CHANGED 3: GD and PGD iterations
tol     = 1e-12;                % relative stop rule

q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34;
mu_eg = [0; 1785.9*q*a0; 0];

nP = numel(P_list); nS = numel(SNR_dB); Na = I1*I2;
e_gd = zeros(MC, nS, nP); e_pgd = zeros(MC, nS, nP); e_crb = zeros(MC, nS, nP);
c_gd = false(MC, nS, nP); c_pgd = false(MC, nS, nP);     % threshold reached
en   = zeros(MC, nP); rsr = zeros(MC, nP);

tic;
for ip = 1:nP
    P = P_list(ip);
    for mc = 1:MC
        [G3, S, B3, Lk] = gen_trial(K, I1, I2, P, RSR_dB, d1_lam, d2_lam, mu_eg, hbar);
        A3 = S.'*G3;  Z3 = exp(-1j*angle(B3));              % (400), (399)
        en(mc, ip)  = norm(G3, 'fro')^2;
        rsr(mc, ip) = mean(abs(B3(:)).^2)/mean(abs(A3(:)).^2);
        Ea = mean(abs(A3(:)).^2);

        for is = 1:nS
            sigma2 = Ea/10^(SNR_dB(is)/10);                 % complex noise variance (390)
            N3 = sqrt(sigma2/2)*(randn(P, Na) + 1j*randn(P, Na));
            Y3 = abs(A3 + B3 + N3);                         % (385), unfolded

            G0 = sqrt(0.1/2)*(randn(K, Na) + 1j*randn(K, Na));   % G0 ~ CN(0,0.1)
            [G_gd,  ~, c_gd(mc, is, ip)]  = est_pgd(Y3, S, B3, Z3, Lk, I1, I2, G0, T_max, tol, false);
            [G_pgd, ~, c_pgd(mc, is, ip)] = est_pgd(Y3, S, B3, Z3, Lk, I1, I2, G0, T_max, tol, true);
            e_gd(mc, is, ip)  = norm(G_gd  - G3, 'fro')^2;
            e_pgd(mc, is, ip) = norm(G_pgd - G3, 'fro')^2;
            e_crb(mc, is, ip) = 2*sigma2*Na*real(trace(inv(conj(S)*S.')));   % (454), (457)
        end
    end
    fprintf('P = %d done (%.1f s)\n', P, toc);
end

%% NMSE = sum of errors / sum of energies (457a), with 95% bootstrap intervals
bs = RandStream('mt19937ar', 'Seed', 7);                    % separate stream
NM_gd = zeros(nP, nS); NM_pgd = zeros(nP, nS); NM_crb = zeros(nP, nS);
CI_gd = zeros(2, nS, nP); CI_pgd = zeros(2, nS, nP); CI_crb = zeros(2, nS, nP);
for ip = 1:nP
    E = repmat(en(:, ip), 1, nS);
    [NM_gd(ip,:),  CI_gd(:,:,ip)]  = nmse_ci(e_gd(:,:,ip),  E, bs);
    [NM_pgd(ip,:), CI_pgd(:,:,ip)] = nmse_ci(e_pgd(:,:,ip), E, bs);
    [NM_crb(ip,:), CI_crb(:,:,ip)] = nmse_ci(e_crb(:,:,ip), E, bs);
end

%% Table
for ip = 1:nP
    fprintf('\nP = %d, RSR = %g dB, polarization per path, %d iterations, %d trials (NMSE in dB, [95%% CI])\n', ...
        P_list(ip), RSR_dB, T_max, MC);
    fprintf('  SNR         GD                   PGD                 CRLB (454)     converged GD / PGD\n');
    for is = 1:nS
        fprintf('  %3d  %6.2f [%6.2f,%6.2f]  %6.2f [%6.2f,%6.2f]  %7.2f        %5.1f %% / %5.1f %%\n', ...
            SNR_dB(is), NM_gd(ip,is), CI_gd(:,is,ip), NM_pgd(ip,is), CI_pgd(:,is,ip), ...
            NM_crb(ip,is), 100*mean(c_gd(:,is,ip)), 100*mean(c_pgd(:,is,ip)));
    end
end

%% Plot (as Xu Fig. 4: GD, PGD and CRLB for P = 10 and 30)
figure('Color', 'w'); hold on; grid on; box on;
for ip = 1:nP
    plot(SNR_dB, NM_gd(ip,:),  '-.s', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.5, ...
         'MarkerFaceColor', 'w');
    plot(SNR_dB, NM_pgd(ip,:), '-^', 'Color', [0 0.45 0.74], 'LineWidth', 1.5, ...
         'MarkerFaceColor', [0 0.45 0.74]);
    plot(SNR_dB, NM_crb(ip,:), '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);
end
% dashed ellipse around each pilot length, as in Xu Fig. 4
j = SNR_dB == 10;  mark_group(10, [NM_gd(1,j) NM_pgd(1,j) NM_crb(1,j)], sprintf('P = %d', P_list(1)), 'above');
j = SNR_dB == 15;  mark_group(15, [NM_gd(2,j) NM_pgd(2,j) NM_crb(2,j)], sprintf('P = %d', P_list(2)), 'below');
xlabel('SNR [dB]'); ylabel('NMSE [dB]');
legend('GD', 'PGD', 'CRLB', 'Location', 'southwest');
title({sprintf('Tuned: %d x %d array, K = %d', I1, I2, K), ...
       sprintf('polarization per path, RSR = %g dB, GD and PGD %d iterations, %d trials', ...
       RSR_dB, T_max, MC)});
xlim([SNR_dB(1) SNR_dB(end)]); xticks(SNR_dB);

exportgraphics(gcf, 'fig4_paper_tuned.png', 'Resolution', 200);
save('fig4_paper_tuned.mat', 'P_list', 'SNR_dB', 'RSR_dB', 'NM_gd', 'NM_pgd', 'NM_crb', ...
     'CI_gd', 'CI_pgd', 'CI_crb', 'c_gd', 'c_pgd', 'rsr', 'MC', 'K', 'I1', 'I2', 'T_max', 'tol');
fprintf('\nsaved fig4_paper_tuned.png and .mat\n');

%% ------------------------------------------------------------------------
function [G3, S, B3, Lk] = gen_trial(K, I1, I2, P, RSR_dB, d1, d2, mu_eg, hbar)
% 2D channel (370)-(380) with polarization drawn per path (374),
% mode-3 unfolding (398a), pilots (395), reference (383) rescaled to RSR_dB
i1 = (0:I1-1).'; i2 = (0:I2-1).';
Lk = randi([3 7], 1, K);                                    % L_k ~ U{3,...,7}
G3 = zeros(K, I1*I2);
for k = 1:K
    L     = Lk(k);
    theta = pi*rand(1, L);  phi = 2*pi*rand(1, L);          % elevation, azimuth
    u     = 2*pi*d1*cos(theta);                             % (370)
    v     = 2*pi*d2*sin(theta).*cos(phi);
    alpha = (randn(1, L) + 1j*randn(1, L))/sqrt(2);         % CN(0,1)
    Gk = zeros(I1, I2);
    for l = 1:L
        a1  = exp(-1j*i1*u(l));  a2 = exp(-1j*i2*v(l));     % (376), (377)
        c   = (mu_eg.'*(sqrt(1/3)*randn(3, 1)))/hbar;       % CHANGED 2: eps_{k,l}, one per path
        Gk  = Gk + c*alpha(l)*(a1*a2.');                    % (378), (380), rank 1 per path
    end
    G3(k, :) = reshape(Gk, 1, []);                          % column-major, (398a)
end
S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);                 % CN(0,1)

% reference: one line-of-sight path, (383)
th_b = pi*rand; ph_b = 2*pi*rand;
u_b  = 2*pi*d1*cos(th_b);  v_b = 2*pi*d2*sin(th_b)*cos(ph_b);
alpha_b = sqrt(10)*(randn + 1j*randn)/sqrt(2);              % CN(0,10)
steer_b = exp(-1j*i1*u_b)*exp(-1j*i2*v_b).';                % I1 x I2
c_b = (mu_eg.'*(sqrt(1/3)*randn(3, 1)))/hbar;               % CHANGED 2: one eps_b
g_b = reshape(c_b*alpha_b*steer_b, 1, []);                  % 1 x I1I2
B3  = ones(P, 1)*g_b;                                       % s_b,p = 1, P x I1I2
A3  = S.'*G3;
B3  = B3*sqrt(10^(RSR_dB/10)*mean(abs(A3(:)).^2)/mean(abs(B3(:)).^2));   % CHANGED 1
end

function [G3, t, converged] = est_pgd(Y3, S, B3, Z3, Lk, I1, I2, G0, T, tol, project)
% Xu Algorithm 1: gradient step (415)-(418) with step size (418a), then the
% rank-L_k projection of every user's I1 x I2 slice (423)-(428);
% project = false gives plain GD. Returns the number of iterations used and
% whether the threshold was reached.
K  = size(S, 1);
G3 = G0;  converged = false;
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
    if norm(M - G3, 'fro')^2 < tol*norm(M, 'fro')^2
        G3 = M;  converged = true;  break;
    end
    G3 = M;
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
