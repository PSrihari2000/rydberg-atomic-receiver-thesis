%% fig3_fixes_budget_sweep.m
% Equal-budget comparison of GD and GS for the 1D array of Xu et al. Fig. 3,
% P = 10: for which iteration budget T and SNR does GD beat GS?
% Both methods always get the SAME number of iterations T. Same model and
% settings as fig3_fixes_budget.m; T is swept instead of fixed at 50.
% Equation numbers refer to ce_derivation_1D_2D_PGD_CRLB.pdf, Xu equation
% numbers are marked "Xu".
%
% Model (as fig3_paper_tuned.m):
%   I = 8 vapor cells, K = 3 users, L_k ~ U{3,...,7}, P = 10, SNR -5:5:30 dB
%   channel Xu (7) = (75), reference Xu (9) = (88), pilots i.i.d. CN(0,1)
%   CHANGED from the paper: reference rescaled to RSR = 40 dB; polarization drawn
%   once per path and once for the reference instead of per cell
%   GD (216)-(228) from G0 ~ CN(0,0.1), step size 1/lambda_max(S S^H)
%   GS of [10] (351)-(369): spectral initialisation + iterations
%
% Budgets T = 10, 20, 50, 100, 200. Each method runs 200 iterations without early
% stop and its error is recorded after T iterations for every T in the list;
% this is the same as separate runs with budget T.
%
% Output: the difference NMSE_GD - NMSE_GS in dB (negative = GD better) for each
% T and SNR, with a paired 95% bootstrap interval (both methods on the same
% trials). Filled markers: the interval excludes 0 (significant difference).

clear; clc; rng(1);

%% Parameters
MC      = 500;                  % Monte Carlo trials per point
K       = 3;                    % users
I       = 8;                    % vapor cells
P       = 10;                   % pilot length
SNR_dB  = -5:5:30;
RSR_dB  = 40;                   % reference-to-signal ratio
d_lam   = 0.5;                  % element spacing / lambda
T_list  = [10 20 50 100 200];   % iteration budgets, same for GD and GS

q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34;
mu_eg = [0; 1785.9*q*a0; 0];

nS = numel(SNR_dB); nT = numel(T_list);
e_gd = zeros(MC, nS, nT); e_gs = zeros(MC, nS, nT); en = zeros(MC, 1);

tic;
for mc = 1:MC
    [G, S, B] = gen_trial(K, I, P, RSR_dB, d_lam, mu_eg, hbar);
    A = G*S;  Z = exp(-1j*angle(B));                        % (91), (160)
    en(mc) = norm(G, 'fro')^2;
    Ea = mean(abs(A(:)).^2);

    for is = 1:nS
        sigma2 = Ea/10^(SNR_dB(is)/10);                     % complex noise variance (90)
        N = sqrt(sigma2/2)*(randn(I, P) + 1j*randn(I, P));
        Y = abs(A + B + N);                                 % (94)

        G0 = sqrt(0.1/2)*(randn(I, K) + 1j*randn(I, K));   % G0 ~ CN(0,0.1)
        e_gd(mc, is, :) = gd_path(Y, S, B, Z, G0, G, T_list);
        e_gs(mc, is, :) = gs_path(Y, S, B, G, T_list);
    end
end
fprintf('done (%.1f s)\n', toc);

%% NMSE per method and the paired difference GD - GS with 95% bootstrap intervals
bs = RandStream('mt19937ar', 'Seed', 7);  nB = 2000;
NM_gd = zeros(nT, nS); NM_gs = zeros(nT, nS); D = zeros(nT, nS); CI_D = zeros(2, nT, nS);
idx = randi(bs, MC, MC, nB);                                % same resamples for both methods
Eb  = sum(en(idx), 1);
for it = 1:nT
    for is = 1:nS
        eg = e_gd(:, is, it);  es = e_gs(:, is, it);
        NM_gd(it, is) = 10*log10(sum(eg)/sum(en));
        NM_gs(it, is) = 10*log10(sum(es)/sum(en));
        D(it, is) = NM_gd(it, is) - NM_gs(it, is);
        d = sort(10*log10(sum(eg(idx), 1)./Eb) - 10*log10(sum(es(idx), 1)./Eb));
        CI_D(:, it, is) = [d(round(0.025*nB)); d(round(0.975*nB))];
    end
end
sig = squeeze(CI_D(2,:,:) < 0 | CI_D(1,:,:) > 0);          % interval excludes 0

%% Table
fprintf('\nP = %d, RSR = %g dB, polarization per path, %d trials; NMSE in dB\n', P, RSR_dB, MC);
fprintf('difference GD - GS [95%% paired CI]; negative = GD better, * = significant\n');
for it = 1:nT
    fprintf('\nT = %d iterations (GD and GS)\n', T_list(it));
    fprintf('  SNR     GS       GD       GD - GS\n');
    for is = 1:nS
        mk = ' '; if sig(it, is), mk = '*'; end
        fprintf('  %3d  %7.2f  %7.2f   %6.2f [%6.2f,%6.2f] %s\n', SNR_dB(is), ...
            NM_gs(it, is), NM_gd(it, is), D(it, is), CI_D(1, it, is), CI_D(2, it, is), mk);
    end
end

%% Plot: NMSE_GD - NMSE_GS vs SNR, one line per budget T
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
xlabel('SNR [dB]'); ylabel('NMSE_{GD} - NMSE_{GS} [dB]');
legend(h, arrayfun(@(t) sprintf('T = %d', t), T_list, 'UniformOutput', false), ...
       'Location', 'northwest');
title({sprintf('Equal budget T for GD and GS: 1D array, I = %d, K = %d, P = %d', I, K, P), ...
       sprintf('RSR = %g dB, polarization per path, %d trials; below 0: GD better; filled: significant', ...
       RSR_dB, MC)});
xlim([SNR_dB(1) SNR_dB(end)]); xticks(SNR_dB);

exportgraphics(gcf, 'fig3_fixes_budget_sweep.png', 'Resolution', 200);
save('fig3_fixes_budget_sweep.mat', 'P', 'SNR_dB', 'RSR_dB', 'T_list', 'NM_gd', 'NM_gs', ...
     'D', 'CI_D', 'sig', 'MC', 'K', 'I');
fprintf('\nsaved fig3_fixes_budget_sweep.png and .mat\n');

%% ------------------------------------------------------------------------
function [G, S, B] = gen_trial(K, I, P, RSR_dB, d_lam, mu_eg, hbar)
% channel Xu (7) = (75), pilots (145), reference Xu (9) = (88) rescaled to RSR_dB,
% polarization drawn once per path and once for the reference
idx = (0:I-1).';
G = zeros(I, K);
for k = 1:K
    L     = randi([3 7]);                                   % L_k ~ U{3,...,7}
    phi   = 2*pi*d_lam*cos(pi*rand(1, L));                  % phase shift, AoA ~ U(0,pi)
    alpha = (randn(1, L) + 1j*randn(1, L))/sqrt(2);         % CN(0,1)
    eps_  = sqrt(1/3)*randn(3, L);                          % eps_{k,l}, one per path
    coup  = repmat(mu_eg.'*eps_, I, 1)/hbar;                % same in every cell
    G(:, k) = sum(coup.*alpha.*exp(-1j*idx*phi), 2);        % (75)
end
S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);                 % CN(0,1)

% reference: one path, (88)
phi_b   = 2*pi*d_lam*cos(pi*rand);
alpha_b = sqrt(10)*(randn + 1j*randn)/sqrt(2);              % CN(0,10)
eps_b   = repmat(sqrt(1/3)*randn(3, 1), 1, I);              % one eps_b
g_b     = (mu_eg.'*eps_b/hbar).'*alpha_b.*exp(-1j*idx*phi_b);
B       = g_b*ones(1, P);                                   % s_b,p = 1
A       = G*S;
B       = B*sqrt(10^(RSR_dB/10)*mean(abs(A(:)).^2)/mean(abs(B(:)).^2));
end

function e = gd_path(Y, S, B, Z, G0, Gtrue, T_list)
% GD on the linearised model, (216)-(228), from G0, max(T_list) iterations;
% e(j) = squared error after T_list(j) iterations
G = G0;  e = zeros(1, numel(T_list));  j = 1;
eta = 1/max(eig(S*S'));                         % step size (418a)
Yc  = Y - abs(B);
for t = 1:max(T_list)
    E = Yc - real(Z.*(G*S));                    % (220)
    G = G + eta*(E.*conj(Z))*S';                % (223)
    if t == T_list(j), e(j) = norm(G - Gtrue, 'fro')^2; j = j + 1; end
end
end

function e = gs_path(Y, S, B, Gtrue, T_list)
% biased GS of [10]: spectral init (351)-(358), then max(T_list) iterations
% (360)-(364); e(j) = squared error after T_list(j) iterations
[I, ~] = size(Y); K = size(S, 1);
G = zeros(I, K);  e = zeros(1, numel(T_list));  j = 1;
for i = 1:I
    y  = Y(i, :).';
    Ab = [S.', B(i, :).'];                      % (351), P x (K+1)
    M  = Ab'*diag(y)*Ab;                        % (352)
    [V, Dg] = eig((M + M')/2);
    [~, m] = max(real(diag(Dg)));
    v  = V(:, m);                               % principal eigenvector (353)
    q  = abs(Ab*v);
    r  = (q.'*y)/(q.'*q);                       % (354)
    gt = exp(-1j*angle(r*v(end)))*r*v;          % (355)-(356)
    G(i, :) = gt(1:K).';                        % (357)-(358)
end
SSinv = S'/(S*S');
for t = 1:max(T_list)
    X = G*S + B;                                % (360)
    R = Y.*exp(1j*angle(X));                    % (361)-(362)
    G = (R - B)*SSinv;                          % (363)-(364)
    if t == T_list(j), e(j) = norm(G - Gtrue, 'fro')^2; j = j + 1; end
end
end
