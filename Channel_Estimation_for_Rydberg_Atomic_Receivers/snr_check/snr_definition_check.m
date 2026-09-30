%% snr_definition_check.m
% Which SNR definition did Xu et al. use? The CRLB of Xu (29)-(31) = (454) depends
% only on the noise variance, the pilots and the channel energy, not on any
% algorithm. Comparing the published CRLB curves with the CRLB under our SNR
% definition therefore gives, for every figure, the noise offset
%   Delta = 10 log10( sigma2_paper / sigma2_ours )   [dB]
% that the paper used at the same nominal SNR. Delta is then compared with the
% offsets implied by common SNR definitions.
%
% Our definition (all figure scripts): SNR = E|a|^2 / sigma^2, a = entries of the
% noiseless users' signal G S (1D) or S^T G_(3) (2D), sigma^2 = complex noise variance.
% With it, E{CRLB} = 2 sigma^2 N E{tr((S^* S^T)^{-1})} / E||G||^2 with
% E|a|^2 = ||G||^2 / N, so NMSE_CRLB = 2 K/(P-K) / SNR, i.e.
%   CRLB_ours [dB] = 10 log10(2K/(P-K)) - SNR_dB.
% This closed form is checked here against a Monte Carlo evaluation of (454).
%
% Candidate definitions and their offset Delta (sigma2_def / sigma2_ours):
%   D1 noise per real component (sigma^2/2 called the noise power)    +3.01 dB
%   D2 signal of one user instead of all K users                      -10log10(K)
%   D3 signal summed over all N cells, noise per cell                 +10log10(N)
%   D4 D3 and noise per real component                                +10log10(N) + 3.01
%   D5 D3 with the signal summed over users once more (N K)          +10log10(N K)
%   D6 signal including the reference, E|a|^2 + E|b|^2               +10log10(1 + RSR)
% Delta is estimated from the Monte Carlo channels where it is not exact.
%
% Paper CRLB values are read from the published figures (accuracy about 0.5 dB).

clear; clc; rng(1);

K = 3;  MC = 2000;
q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34;
mu_eg = [0; 1785.9*q*a0; 0];

%% Published CRLB values (read from Xu Figs. 3-5)
SNR_34 = -5:5:30;
paper.f3P10 = [14.2  9.1  4.1 -0.9  -5.9 -10.9 -15.9 -20.9];   % Fig. 3, 1D, P = 10
paper.f3P30 = [ 8.7  3.7 -1.3 -6.3 -11.3 -16.3 -21.3 -26.3];   % Fig. 3, 1D, P = 30
paper.f4P10 = [28.3 23.3 18.3 13.3   8.3   3.3  -1.7  -6.7];   % Fig. 4, 2D, P = 10
paper.f4P30 = [21.2 16.2 11.2  6.2   1.2  -3.8  -8.8 -13.8];   % Fig. 4, 2D, P = 30
P_5 = 5:5:50;                                                  % Fig. 5, 2D, SNR = 5 dB
paper.f5 = [15.0 -1.3 -4.5 -6.2 -7.5 -8.4 -9.3 -9.9 -10.5 -11.0];

crlb = @(P, snr) 10*log10(2*K./(P - K)) - snr;                 % ours, closed form

%% Check the closed form against a Monte Carlo evaluation of (454)
fprintf('closed form vs Monte Carlo CRLB (454), our SNR definition\n');
fprintf('  array  P   SNR   closed form   Monte Carlo\n');
chk = [1 10 10; 1 30 30; 2 10 30; 2 30 15; 2 50 5];            % [dim P SNR]
for c = 1:size(chk, 1)
    dim = chk(c,1); P = chk(c,2); snr = chk(c,3);
    num = 0; den = 0;
    for mc = 1:MC
        [G, S] = gen_channel(dim, K, P, mu_eg, hbar);
        N = size(G, 1);
        A = G*S;  sigma2 = mean(abs(A(:)).^2)/10^(snr/10);
        num = num + 2*sigma2*N*real(trace(inv(conj(S)*S.')));
        den = den + norm(G, 'fro')^2;
    end
    fprintf('   %dD  %3d  %3d    %8.2f      %8.2f\n', dim, P, snr, crlb(P, snr), 10*log10(num/den));
end

%% Offset Delta required by each published figure
D.f3P10 = paper.f3P10 - crlb(10, SNR_34);
D.f3P30 = paper.f3P30 - crlb(30, SNR_34);
D.f4P10 = paper.f4P10 - crlb(10, SNR_34);
D.f4P30 = paper.f4P30 - crlb(30, SNR_34);
D.f5    = paper.f5    - crlb(P_5, 5);

fprintf('\nrequired offset Delta = paper CRLB - our CRLB  [dB]\n');
fprintf('  Fig. 3, P = 10: %s   mean %.1f\n', mat2str(round(D.f3P10, 1)), mean(D.f3P10));
fprintf('  Fig. 3, P = 30: %s   mean %.1f\n', mat2str(round(D.f3P30, 1)), mean(D.f3P30));
fprintf('  Fig. 4, P = 10: %s   mean %.1f\n', mat2str(round(D.f4P10, 1)), mean(D.f4P10));
fprintf('  Fig. 4, P = 30: %s   mean %.1f\n', mat2str(round(D.f4P30, 1)), mean(D.f4P30));
fprintf('  Fig. 5, P = 5..50: %s\n', mat2str(round(D.f5, 1)));
fprintf('  Fig. 5, mean over P >= 20: %.1f\n', mean(D.f5(P_5 >= 20)));

%% Offsets of the candidate definitions
% ratio of one user's signal power to the total, from the Monte Carlo channels
ru = zeros(1, 2);
for dim = 1:2
    s1 = 0; sK = 0;
    for mc = 1:MC
        [G, S] = gen_channel(dim, K, 30, mu_eg, hbar);
        A = G*S;  sK = sK + mean(abs(A(:)).^2);
        for k = 1:K, Ak = G(:,k)*S(k,:); s1 = s1 + mean(abs(Ak(:)).^2)/K; end
    end
    ru(dim) = s1/sK;
end
N1 = 8; N2 = 64; RSR_lit = 10^(-1.2/10);                       % measured literal RSR
cand = {'D1 noise per real component',          3.01,                     3.01;
        'D2 signal of one user',                10*log10(ru(1)),          10*log10(ru(2));
        'D3 signal summed over cells',          10*log10(N1),             10*log10(N2);
        'D4 D3 + real component',               10*log10(N1) + 3.01,      10*log10(N2) + 3.01;
        'D5 signal summed over cells, x K',     10*log10(N1*K),           10*log10(N2*K);
        'D6 signal incl. reference (literal)',  10*log10(1 + RSR_lit),    10*log10(1 + RSR_lit)};

fprintf('\ncandidate definitions: offset Delta [dB]      1D (Fig. 3)   2D (Figs. 4, 5)\n');
for c = 1:size(cand, 1)
    fprintf('  %-38s   %6.2f        %6.2f\n', cand{c,1}, cand{c,2}, cand{c,3});
end
fprintf('\nrequired: Fig. 3 ~ %.1f, Fig. 4 ~ %.1f, Fig. 5 (P >= 20) ~ %.1f\n', ...
    mean([D.f3P10 D.f3P30]), mean([D.f4P10 D.f4P30]), mean(D.f5(P_5 >= 20)));

%% Best candidate per figure
req = [mean([D.f3P10 D.f3P30]), mean([D.f4P10 D.f4P30]), mean(D.f5(P_5 >= 20))];
dimf = [1 2 2];  names = {'Fig. 3', 'Fig. 4', 'Fig. 5'};
for f = 1:3
    off = cell2mat(cand(:, 1 + dimf(f)));
    [e, b] = min(abs(off - req(f)));
    fprintf('  %s: closest candidate %s (%.2f dB, mismatch %.1f dB)\n', names{f}, cand{b,1}, off(b), e);
end

%% Plot: required offset per figure with the candidate offsets
figure('Color', 'w', 'Position', [100 100 1100 420]);
subplot(1, 2, 1); hold on; grid on; box on;
plot(SNR_34, D.f3P10, '-d', 'Color', 'k', 'LineWidth', 1.4, 'MarkerFaceColor', 'w');
plot(SNR_34, D.f3P30, '--d', 'Color', 'k', 'LineWidth', 1.4, 'MarkerFaceColor', 'k');
plot(SNR_34, D.f4P10, '-^', 'Color', [0 0.45 0.74], 'LineWidth', 1.4, 'MarkerFaceColor', 'w');
plot(SNR_34, D.f4P30, '--^', 'Color', [0 0.45 0.74], 'LineWidth', 1.4, 'MarkerFaceColor', [0 0.45 0.74]);
yl = [cand{3,2} cand{5,3} cand{4,3} cand{3,3}];
yline(cand{3,2}, ':', 'D3 (1D)', 'Color', 'k', 'LabelHorizontalAlignment', 'left');
yline(cand{5,3}, ':', 'D5 (2D)', 'Color', [0 0.45 0.74], 'LabelHorizontalAlignment', 'left');
xlabel('SNR [dB]'); ylabel('\Delta = CRLB_{paper} - CRLB_{ours} [dB]');
legend('Fig. 3, P = 10', 'Fig. 3, P = 30', 'Fig. 4, P = 10', 'Fig. 4, P = 30', 'Location', 'east');
title('Figs. 3 and 4'); ylim([0 30]); xlim([-5 30]); xticks(SNR_34);
subplot(1, 2, 2); hold on; grid on; box on;
plot(P_5, D.f5, '-^', 'Color', [0 0.45 0.74], 'LineWidth', 1.4, 'MarkerFaceColor', [0 0.45 0.74]);
yline(cand{1,3}, ':', 'D1', 'Color', 'k', 'LabelHorizontalAlignment', 'left');
xlabel('Pilot length P'); ylabel('\Delta = CRLB_{paper} - CRLB_{ours} [dB]');
title('Fig. 5 (SNR = 5 dB)'); ylim([0 20]); xlim([5 50]); xticks(P_5);
sgtitle('Noise offset implied by the published CRLB curves');

exportgraphics(gcf, 'snr_definition_check.png', 'Resolution', 200);
save('snr_definition_check.mat', 'paper', 'D', 'cand', 'req', 'SNR_34', 'P_5', 'K');
fprintf('\nsaved snr_definition_check.png and .mat\n');

%% ------------------------------------------------------------------------
function [G, S] = gen_channel(dim, K, P, mu_eg, hbar)
% channel and pilots of the tuned scripts (polarization per path); dim = 1: 1D,
% I = 8, G is I x K; dim = 2: 8 x 8, G is (I1 I2) x K (transpose of G_(3))
S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);
if dim == 1
    I = 8; idx = (0:I-1).'; G = zeros(I, K);
    for k = 1:K
        L = randi([3 7]);
        phi = 2*pi*0.5*cos(pi*rand(1, L));
        alpha = (randn(1, L) + 1j*randn(1, L))/sqrt(2);
        c = (mu_eg.'*(sqrt(1/3)*randn(3, L)))/hbar;
        G(:, k) = exp(-1j*idx*phi)*(c.*alpha).';
    end
else
    I1 = 8; I2 = 8; i1 = (0:I1-1).'; i2 = (0:I2-1).'; G = zeros(I1*I2, K);
    for k = 1:K
        L = randi([3 7]); Gk = zeros(I1, I2);
        for l = 1:L
            th = pi*rand; ph = 2*pi*rand;
            u = 2*pi*0.5*cos(th); v = 2*pi*0.5*sin(th)*cos(ph);
            c = (mu_eg.'*(sqrt(1/3)*randn(3, 1)))/hbar;
            alpha = (randn + 1j*randn)/sqrt(2);
            Gk = Gk + c*alpha*(exp(-1j*i1*u)*exp(-1j*i2*v).');
        end
        G(:, k) = Gk(:);
    end
end
end
