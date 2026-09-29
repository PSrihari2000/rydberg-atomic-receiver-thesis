%% fig3_nmse.m
% Fig. 3: NMSE vs SNR for a 1D array (I = 8 cells, K = 3 users),
% GD (216)-(228) vs biased GS (351)-(369) vs CRLB (454)/(457a) and (463a).
% Equation numbers refer to ce_derivation_1D_2D_PGD_CRLB.pdf.
%
% Measurements are generated with the exact model (94), y = |GS + B + N|.
% GD works on the linearised model (163); GS works on (94) directly.

clearvars -except RSR_dB POL T_gd T_gs; clc; rng(1);

%% Parameters
MC      = 200;                  % Monte Carlo trials per point
K       = 3;                    % users
I       = 8;                    % vapor cells
P_list  = [10 30];              % pilot lengths
SNR_dB  = -5:5:30;
if ~exist('RSR_dB', 'var')
RSR_dB  = 30;                   % reference-to-signal ratio E|b|^2 / E|a|^2
end                             % RSR_dB = NaN -> paper values, alpha_b ~ CN(0,10), no rescaling
if ~exist('POL', 'var')
POL     = 'cell';               % 'cell': eps_{i,k,l} and eps_{b,i} per cell, literal (75), (88)
end                             % 'path': eps_{k,l} per path and one eps_b for all cells
d_lam   = 0.5;
if ~exist('T_gd', 'var'), T_gd = 3000; end   % GD iterations (3000 = run to convergence)
if ~exist('T_gs', 'var'), T_gs = 50;   end   % GS iterations, t0 = 50 as in Cui et al. Sec. VI
% T_gd and T_gs may be one number (same for every P) or one number per entry
% of P_list, e.g. T_gs = [15 50] -> 15 iterations at P = 10, 50 at P = 30
T_gd = T_gd.*ones(1, numel(P_list));
T_gs = T_gs.*ones(1, numel(P_list));
tol_gd  = 1e-12;                             % relative stop rule (225)

q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34;
mu_eg = [0; 1785.9*q*a0; 0];

nS = numel(SNR_dB); nP = numel(P_list);
err_gd = zeros(nP, nS); err_gs = zeros(nP, nS);
crb_xu = zeros(nP, nS); crb_ex = zeros(nP, nS); energy = zeros(nP, 1);

tic;
for ip = 1:nP
    P = P_list(ip);
    for mc = 1:MC
        % channel, pilots and reference are drawn once per trial and reused
        % for every SNR, so the curves differ only through the noise
        [G, S, B] = gen_trial(K, I, P, RSR_dB, d_lam, mu_eg, hbar, POL);
        A = G*S;  Z = exp(-1j*angle(B));
        energy(ip) = energy(ip) + norm(G, 'fro')^2;
        Ea = mean(abs(A(:)).^2);

        for is = 1:nS
            sigma2 = Ea/10^(SNR_dB(is)/10);                     % (90)
            N = sqrt(sigma2/2)*(randn(I, P) + 1j*randn(I, P));
            Y = abs(A + B + N);                                 % (94)

            G_gd = est_gd(Y, S, B, Z, T_gd(ip), tol_gd);
            G_gs = est_gs(Y, S, B, T_gs(ip));
            err_gd(ip, is) = err_gd(ip, is) + norm(G_gd - G, 'fro')^2;
            err_gs(ip, is) = err_gs(ip, is) + norm(G_gs - G, 'fro')^2;

            % CRLB, same bound for every cell of the 1D array
            crb_xu(ip, is) = crb_xu(ip, is) + 2*sigma2*I*real(trace(inv(conj(S)*S.')));  % (454),(457)
            crb_ex(ip, is) = crb_ex(ip, is) + crb_exact(S, Z, sigma2/2);                  % (463a)
        end
    end
    fprintf('P = %d done (%.1f s)\n', P, toc);
end

nmse   = @(x) 10*log10(x./energy);                          % NMSE of Xu Sec. V
NM_gd  = nmse(err_gd); NM_gs = nmse(err_gs);
NM_crb = nmse(crb_xu); NM_ex = nmse(crb_ex);

%% Table
for ip = 1:nP
    fprintf('\nP = %d   (NMSE in dB, %d trials, RSR = %g dB)\n', P_list(ip), MC, RSR_dB);
    fprintf('  SNR     GD       GS      CRLB(454)  exact CRLB(463a)\n');
    for is = 1:nS
        fprintf('  %3d  %7.2f  %7.2f  %8.2f  %9.2f\n', SNR_dB(is), ...
            NM_gd(ip,is), NM_gs(ip,is), NM_crb(ip,is), NM_ex(ip,is));
    end
end

%% Plot
figure('Color', 'w'); hold on; grid on; box on;
for ip = 1:nP
    plot(SNR_dB, NM_gs(ip,:),  'k--d', 'LineWidth', 1.2, 'MarkerFaceColor', 'w');
    plot(SNR_dB, NM_gd(ip,:),  '--s', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.2, ...
         'MarkerFaceColor', [0.93 0.69 0.13]);
    plot(SNR_dB, NM_crb(ip,:), '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
    text(SNR_dB(end-2), NM_gs(ip,end-2) + 2.5, sprintf('P = %d', P_list(ip)), ...
         'FontWeight', 'bold');
end
xlabel('SNR [dB]'); ylabel('NMSE [dB]');
legend('GS', 'GD', 'CRLB', 'Location', 'southwest');
if isnan(RSR_dB)
    tag = 'paperref'; rsr_txt = 'paper reference values';
else
    tag = sprintf('rsr%d', RSR_dB); rsr_txt = sprintf('RSR = %g dB', RSR_dB);
end
tag = [tag '_pol' POL];
if any(T_gd ~= 3000) || any(T_gs ~= 50)
    tag = [tag '_Tgd' sprintf('%d-', T_gd)]; tag(end) = [];
    tag = [tag '_Tgs' sprintf('%d-', T_gs)]; tag(end) = [];
end
it_txt = '';
for ip = 1:nP
    it_txt = [it_txt sprintf('P = %d: GD %d, GS %d it.   ', P_list(ip), T_gd(ip), T_gs(ip))]; %#ok<AGROW>
end
title({sprintf('1D array, I = %d, K = %d, %s, polarization per %s, %d trials', ...
    I, K, rsr_txt, POL, MC), strtrim(it_txt)});
xlim([SNR_dB(1) SNR_dB(end)]);

exportgraphics(gcf, ['fig3_nmse_' tag '.png'], 'Resolution', 200);
save(['fig3_nmse_' tag '.mat'], 'SNR_dB', 'P_list', 'NM_gd', 'NM_gs', 'NM_crb', 'NM_ex', ...
     'MC', 'K', 'I', 'RSR_dB');
fprintf('\nsaved fig3_nmse_%s.png and .mat\n', tag);

%% ------------------------------------------------------------------------
function [G, S, B] = gen_trial(K, I, P, RSR_dB, d_lam, mu_eg, hbar, POL)
% channel (75), pilots (145), reference (88)-(89)
% POL = 'cell': eps_{i,k,l} and eps_{b,i} drawn per cell (literal 1D model)
% POL = 'path': eps_{k,l} per path and one eps_b, same in every cell
idx = (0:I-1).';
G = zeros(I, K);
for k = 1:K
    L     = randi([3 7]);
    phi   = 2*pi*d_lam*cos(pi*rand(1, L));                  % (8)
    alpha = (randn(1, L) + 1j*randn(1, L))/sqrt(2);         % CN(0,1)
    if strcmp(POL, 'cell')
        eps_ = sqrt(1/3)*randn(3, I*L);                     % eps_{i,k,l} ~ N(0,1/3)
        coup = reshape(mu_eg.'*eps_, I, L)/hbar;            % I x L
    else
        eps_ = sqrt(1/3)*randn(3, L);                       % eps_{k,l} ~ N(0,1/3)
        coup = repmat(mu_eg.'*eps_, I, 1)/hbar;             % same row for every cell
    end
    G(:, k) = sum(coup.*alpha.*exp(-1j*idx*phi), 2);        % (75)
end
S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);                 % CN(0,1)

% reference: one path, (88)
phi_b   = 2*pi*d_lam*cos(pi*rand);
alpha_b = sqrt(10)*(randn + 1j*randn)/sqrt(2);              % CN(0,10)
if strcmp(POL, 'cell')
    eps_b = sqrt(1/3)*randn(3, I);                          % eps_{b,i}
else
    eps_b = repmat(sqrt(1/3)*randn(3, 1), 1, I);            % one eps_b for all cells
end
g_b     = (mu_eg.'*eps_b/hbar).'*alpha_b.*exp(-1j*idx*phi_b);
B       = g_b*ones(1, P);                                   % s_b,p = 1
A       = G*S;
if ~isnan(RSR_dB)
    B = B*sqrt(10^(RSR_dB/10)*mean(abs(A(:)).^2)/mean(abs(B(:)).^2));
end
end

function G = est_gd(Y, S, B, Z, T, tol)
% gradient descent on the linearised model, (216)-(228)
[I, ~] = size(Y); K = size(S, 1);
scale = sqrt(mean(Y(:).^2));                    % only sets the size of G0
G   = sqrt(0.1/2)*(randn(I, K) + 1j*randn(I, K))*1e-6*scale;   % ~ CN(0,0.1), tiny
eta = 1/max(eig(S*S'));                         % step size (418a)
Yc  = Y - abs(B);
for t = 1:T
    E     = Yc - real(Z.*(G*S));                % (220)
    G_new = G + eta*(E.*conj(Z))*S';            % (223)
    if norm(G_new - G, 'fro')^2 < tol*norm(G_new, 'fro')^2, G = G_new; break; end
    G = G_new;
end
end

function G = est_gs(Y, S, B, T)
% biased Gerchberg-Saxton, spectral init (351)-(358) + iterations (360)-(364)
[I, ~] = size(Y); K = size(S, 1);
G = zeros(I, K);
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
    G = (R - B)*SSinv;                          % (363)-(364)
end
end

function c = crb_exact(S, Z, sr2)
% exact real-parameter bound (460)-(463a), summed over the cells
c = 0;
for i = 1:size(Z, 1)
    Ai = diag(Z(i, :))*S.';
    D  = [real(Ai), -imag(Ai)];
    F  = (D.'*D)/sr2;                           % (462)
    if rank(F) < size(F, 1), c = Inf; return; end
    c = c + trace(inv(F));                      % (463a)
end
end
