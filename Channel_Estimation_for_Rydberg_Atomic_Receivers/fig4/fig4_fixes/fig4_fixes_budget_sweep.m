%% fig4_fixes_budget_sweep.m
% Equal-budget comparison of PGD and GD for the 2D array of Xu et al. Fig. 4,
% P = 10: for which iteration budget T and SNR does PGD beat GD, and by how much?
% Both methods always get the SAME number of iterations T. Same model and
% settings as fig4_paper_tuned.m; T is swept instead of fixed at 50.
% Equation numbers refer to ce_derivation_1D_2D_PGD_CRLB.pdf, Xu equation
% numbers are marked "Xu".
%
% Model (as fig4_paper_tuned.m):
%   8 x 8 array, K = 3 users, L_k ~ U{3,...,7}, P = 10, SNR -5:5:30 dB
%   2D channel (370)-(380), reference (383), pilots i.i.d. CN(0,1)
%   CHANGED from the paper: reference rescaled to RSR = 40 dB; polarization drawn
%   once per path and once for the reference instead of per cell
%   PGD, Xu Algorithm 1 (415)-(428), and GD (the same step without projection),
%   both from the same G0 ~ CN(0,0.1), step size 1/lambda_max(S S^H);
%   SVD reconstruction with V^H (Xu (26)-(27) print V^T)
%
% Budgets T = 10, 20, 50, 100, 200. Each method runs 200 iterations without early
% stop and its error is recorded after T iterations for every T in the list;
% this is the same as separate runs with budget T.
%
% Output: the difference NMSE_PGD - NMSE_GD in dB (negative = PGD better) for each
% T and SNR, with a paired 95% bootstrap interval (both methods on the same
% trials). Filled markers: the interval excludes 0 (significant difference).

clear; clc; rng(1);

%% Parameters
MC      = 500;                  % Monte Carlo trials per point
K       = 3;                    % users
I1      = 8;  I2 = 8;           % array size
P       = 10;                   % pilot length
SNR_dB  = -5:5:30;
RSR_dB  = 40;                   % reference-to-signal ratio
d1_lam  = 0.5;  d2_lam = 0.5;   % element spacings / lambda
T_list  = [10 20 50 100 200];   % iteration budgets, same for GD and PGD

q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34;
mu_eg = [0; 1785.9*q*a0; 0];

nS = numel(SNR_dB); nT = numel(T_list); Na = I1*I2;
e_gd = zeros(MC, nS, nT); e_pgd = zeros(MC, nS, nT); en = zeros(MC, 1);

tic;
for mc = 1:MC
    [G3, S, B3, Lk] = gen_trial(K, I1, I2, P, RSR_dB, d1_lam, d2_lam, mu_eg, hbar);
    A3 = S.'*G3;  Z3 = exp(-1j*angle(B3));                  % (400), (399)
    en(mc) = norm(G3, 'fro')^2;
    Ea = mean(abs(A3(:)).^2);

    for is = 1:nS
        sigma2 = Ea/10^(SNR_dB(is)/10);                     % complex noise variance (390)
        N3 = sqrt(sigma2/2)*(randn(P, Na) + 1j*randn(P, Na));
        Y3 = abs(A3 + B3 + N3);                             % (385), unfolded

        G0 = sqrt(0.1/2)*(randn(K, Na) + 1j*randn(K, Na)); % G0 ~ CN(0,0.1)
        e_gd(mc, is, :)  = pgd_path(Y3, S, B3, Z3, Lk, I1, I2, G0, G3, T_list, false);
        e_pgd(mc, is, :) = pgd_path(Y3, S, B3, Z3, Lk, I1, I2, G0, G3, T_list, true);
    end
    if mod(mc, 100) == 0, fprintf('%d trials done (%.1f s)\n', mc, toc); end
end

%% NMSE per method and the paired difference PGD - GD with 95% bootstrap intervals
bs = RandStream('mt19937ar', 'Seed', 7);  nB = 2000;
NM_gd = zeros(nT, nS); NM_pgd = zeros(nT, nS); D = zeros(nT, nS); CI_D = zeros(2, nT, nS);
idx = randi(bs, MC, MC, nB);                                % same resamples for both methods
Eb  = sum(en(idx), 1);
for it = 1:nT
    for is = 1:nS
        eg = e_gd(:, is, it);  ep = e_pgd(:, is, it);
        NM_gd(it, is)  = 10*log10(sum(eg)/sum(en));
        NM_pgd(it, is) = 10*log10(sum(ep)/sum(en));
        D(it, is) = NM_pgd(it, is) - NM_gd(it, is);
        d = sort(10*log10(sum(ep(idx), 1)./Eb) - 10*log10(sum(eg(idx), 1)./Eb));
        CI_D(:, it, is) = [d(round(0.025*nB)); d(round(0.975*nB))];
    end
end
sig = squeeze(CI_D(2,:,:) < 0 | CI_D(1,:,:) > 0);          % interval excludes 0

%% Table
fprintf('\nP = %d, RSR = %g dB, polarization per path, %d trials; NMSE in dB\n', P, RSR_dB, MC);
fprintf('difference PGD - GD [95%% paired CI]; negative = PGD better, * = significant\n');
for it = 1:nT
    fprintf('\nT = %d iterations (GD and PGD)\n', T_list(it));
    fprintf('  SNR     GD      PGD      PGD - GD\n');
    for is = 1:nS
        mk = ' '; if sig(it, is), mk = '*'; end
        fprintf('  %3d  %7.2f  %7.2f   %6.2f [%6.2f,%6.2f] %s\n', SNR_dB(is), ...
            NM_gd(it, is), NM_pgd(it, is), D(it, is), CI_D(1, it, is), CI_D(2, it, is), mk);
    end
end

%% Plot: NMSE_PGD - NMSE_GD vs SNR, one line per budget T
figure('Color', 'w'); hold on; grid on; box on;
col = lines(nT);  h = gobjects(nT, 1);
for it = 1:nT
    h(it) = plot(SNR_dB, D(it,:), '-o', 'Color', col(it,:), 'LineWidth', 1.5, ...
                 'MarkerFaceColor', 'w');
    s = sig(it, :);
    plot(SNR_dB(s), D(it, s), 'o', 'Color', col(it,:), 'MarkerFaceColor', col(it,:), ...
         'HandleVisibility', 'off');
end
yline(0, 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
xlabel('SNR [dB]'); ylabel('NMSE_{PGD} - NMSE_{GD} [dB]');
legend(h, arrayfun(@(t) sprintf('T = %d', t), T_list, 'UniformOutput', false), ...
       'Location', 'southwest');
title({sprintf('Equal budget T for GD and PGD: %d x %d array, K = %d, P = %d', I1, I2, K, P), ...
       sprintf('RSR = %g dB, polarization per path, %d trials; below 0: PGD better; filled: significant', ...
       RSR_dB, MC)});
xlim([SNR_dB(1) SNR_dB(end)]); xticks(SNR_dB);

exportgraphics(gcf, 'fig4_fixes_budget_sweep.png', 'Resolution', 200);
save('fig4_fixes_budget_sweep.mat', 'P', 'SNR_dB', 'RSR_dB', 'T_list', 'NM_gd', 'NM_pgd', ...
     'D', 'CI_D', 'sig', 'MC', 'K', 'I1', 'I2');
fprintf('\nsaved fig4_fixes_budget_sweep.png and .mat\n');

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
        c   = (mu_eg.'*(sqrt(1/3)*randn(3, 1)))/hbar;       % eps_{k,l}, one per path
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
c_b = (mu_eg.'*(sqrt(1/3)*randn(3, 1)))/hbar;               % one eps_b
g_b = reshape(c_b*alpha_b*steer_b, 1, []);                  % 1 x I1I2
B3  = ones(P, 1)*g_b;                                       % s_b,p = 1, P x I1I2
A3  = S.'*G3;
B3  = B3*sqrt(10^(RSR_dB/10)*mean(abs(A3(:)).^2)/mean(abs(B3(:)).^2));
end

function e = pgd_path(Y3, S, B3, Z3, Lk, I1, I2, G0, Gtrue, T_list, project)
% Xu Algorithm 1 (project = true) or GD (project = false) from G0 for
% max(T_list) iterations; e(j) = squared error after T_list(j) iterations
K  = size(S, 1);
G3 = G0;  e = zeros(1, numel(T_list));  j = 1;
zeta = 1/max(eig(S*S'));
Yc = Y3 - abs(B3);
for t = 1:max(T_list)
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
    G3 = M;
    if t == T_list(j), e(j) = norm(G3 - Gtrue, 'fro')^2; j = j + 1; end
end
end
