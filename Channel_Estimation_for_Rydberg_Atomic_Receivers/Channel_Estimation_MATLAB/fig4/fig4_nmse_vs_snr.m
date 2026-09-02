% FIG4_NMSE_VS_SNR  Reproduce Fig.4 of Xu et al., "Channel Estimation for
% Rydberg Atomic Receivers" (IEEE WCL, Sept 2025): NMSE vs SNR for a 2D
% Rydberg atom-based antenna array (I1=I2=8), comparing GD (unconstrained),
% PGD (rank-projected, Algorithm 1), and CRLB, at P = 10 and P = 30 pilots.
%
% Structure mirrors fig3/fig3_nmse_vs_snr.m: SIX underlying curves (GD/PGD/
% CRLB x P=10/P=30), THREE legend entries (paper's own Fig.4 legend has just
% GD/PGD/CRLB), P=10 and P=30 clusters marked with a dashed ellipse + label.
%
% Key qualitative difference from Fig.3 expected here (paper's own Fig.4):
% at P=10, BOTH GD and PGD saturate almost immediately and stay roughly FLAT
% across the whole SNR range (~25dB) -- this is a genuine identifiability
% limit of the 2D rank-constrained recovery problem (I1*I2=64 unknowns per
% user, only P=10 measurements), not a bug, and is far more severe than
% Fig.3's 1D "knee" shape. PGD sits slightly below GD at P=10 (the rank
% projection helps); at P=30 both nearly coincide and track CRLB.
%
% Run this script directly (it adds its own folder to the path).

clear; clc; close all;
addpath(fileparts(mfilename('fullpath')));

p = params_2D();
IJ = p.I1*p.I2;

% ---------------- simulation controls (not given numerically by the paper) ----------------
SNR_dB_list = -5:5:30;
P_list      = [10 30];
numTrials   = 200;      % Monte Carlo trials per (P, SNR) point (2D PGD is heavier than fig3's
                         % closed-form GD, so fewer trials than fig3's 300 to keep runtime sane)
maxIter_PGD = 2000;     % PGD iteration cap (raised from 500 to rule out under-convergence
                        % specifically for PGD -- unlike fig3/fig4's GD, PGD is iterative by
                        % genuine necessity (the rank projection has no closed form), so its
                        % convergence is real behavior, not an implementation shortcut, and
                        % is worth checking cleanly before blaming the model/margin)
reltol_PGD  = 1e-10;    % PGD relative stopping tolerance (tightened alongside maxIter)

% ASSUMPTION 1: same |b|/|a| power margin trade-off as fig3 (Eq.10's linearization
% requirement vs. keeping the model genuinely nonlinear). Raised from 45 to 60dB here:
% fig3's GD is now an EXACT closed-form solve, and by the same argument, fig4's GD
% (also exact -- unconstrained 2D decomposes into 64 independent per-antenna-pair
% closed-form LS problems, see chat) should NOT show a hard SNR-independent floor at
% all -- an exact LS solution to an over-determined problem (P=10 real eqns vs 6 real
% unknowns per antenna-pair) keeps improving as noise shrinks. If GD is still capping
% out here, that's the Eq.10 Taylor-truncation bias (margin-controlled), same mechanism
% as fig3, needing a larger margin -- NOT re-derived analytically for 2D, being
% re-tested empirically same as fig3 was.
marginDB = 60;

% ASSUMPTION 2: same "no numeric SNR-to-noise-variance mapping in the paper" situation
% as fig3 -- anchor chosen from an approximate visual read of the paper's own Fig.4
% (P=30 CRLB cluster, left edge); flagged as approximate, not paper-derived. See
% fig3_nmse_vs_snr.m's ASSUMPTION 2 comment for the full reasoning (Wishart trace
% formula, sigma2/2 real-noise-variance convention, etc.) -- unchanged here.
P_anchor       = 30;
SNR_anchor_dB  = -5;
NMSE_anchor_dB = 22;

% ---------------- reference pilot power calibration (marginDB) ----------------
% Unchanged formula from fig3 -- per-antenna power statistics don't depend on I1,I2.
EL = mean(p.Lmin:p.Lmax);
Pa_analytical = p.K * EL * p.mu^2 / (3*p.hbar^2);

Pb_target = Pa_analytical * 10^(marginDB/10);
p.sb2 = Pb_target / (p.alphab_var * p.mu^2 / (3*p.hbar^2));

% ---------------- noise-scale anchor ----------------
traceMinv_anchor = p.K/(P_anchor - p.K);
SNR_lin_anchor   = 10^(SNR_anchor_dB/10);
NMSE_lin_anchor  = 10^(NMSE_anchor_dB/10);
calib = NMSE_lin_anchor * SNR_lin_anchor / (2*traceMinv_anchor);   % "2" folds in the sigma2/2 fix (see fig3)

fprintf('Pa_analytical=%.4g, margin=%ddB, calib=%.4g (anchored to %ddB NMSE at %ddB SNR, P=%d)\n', ...
    Pa_analytical, marginDB, calib, NMSE_anchor_dB, SNR_anchor_dB, P_anchor);

sigma2_of_SNR = @(SNR_dB) calib * Pa_analytical / 10^(SNR_dB/10);

% ---------------- GD / PGD / CRLB Monte Carlo sweep (per P) ----------------
results = struct('P', {}, 'SNR_dB', {}, 'GD', {}, 'PGD', {}, 'CRLB', {});

for iP = 1:numel(P_list)
    P = P_list(iP);
    nmse_GD   = zeros(size(SNR_dB_list));
    nmse_PGD  = zeros(size(SNR_dB_list));
    nmse_CRLB = zeros(size(SNR_dB_list));

    for iSNR = 1:numel(SNR_dB_list)
        SNR_dB = SNR_dB_list(iSNR);
        sigma2 = sigma2_of_SNR(SNR_dB);

        sumSqErr_GD = 0; sumSqErr_PGD = 0; sumSqG = 0; sumCRB = 0;

        for trial = 1:numTrials
            [Gflat, S, Bflat, Lk] = generate_trial_2D(p, P);

            A = Gflat*S;                                          % (I1I2) x P, true noiseless signal
            N = sqrt(sigma2/2)*(randn(IJ,P)+1i*randn(IJ,P));
            Y = abs(A + Bflat + N);                                % exact model (Eq.18-analogue)

            absB = abs(Bflat);
            Z    = exp(-1i*angle(Bflat));

            Ghat_GD  = gd_estimator_1D(Y, absB, Z, S);                                    % unconstrained
            Ghat_PGD = pgd_estimator_2D(Y, absB, Z, S, p.I1, p.I2, Lk, maxIter_PGD, reltol_PGD);

            sumSqErr_GD  = sumSqErr_GD  + norm(Gflat-Ghat_GD, 'fro')^2;
            sumSqErr_PGD = sumSqErr_PGD + norm(Gflat-Ghat_PGD,'fro')^2;
            sumSqG       = sumSqG + norm(Gflat,'fro')^2;
            sumCRB       = sumCRB + crlb_trace_1D(sigma2/2, S, IJ);   % sigma2/2: see fig3 BUGFIX note
        end

        nmse_GD(iSNR)   = 10*log10(sumSqErr_GD/sumSqG);
        nmse_PGD(iSNR)  = 10*log10(sumSqErr_PGD/sumSqG);
        nmse_CRLB(iSNR) = 10*log10(sumCRB/sumSqG);

        fprintf('P=%2d  SNR=%3d dB  ->  GD=%.2f dB  PGD=%.2f dB  CRLB=%.2f dB\n', ...
            P, SNR_dB, nmse_GD(iSNR), nmse_PGD(iSNR), nmse_CRLB(iSNR));
    end

    results(iP).P      = P;
    results(iP).SNR_dB = SNR_dB_list;
    results(iP).GD     = nmse_GD;
    results(iP).PGD    = nmse_PGD;
    results(iP).CRLB   = nmse_CRLB;
end

% ---------------- plot (6 curves, 3 legend entries, matching the paper) ----------------
figure; hold on; grid on; box on;
colGD = [0.85 0.55 0.1]; colPGD = [0.1 0.35 0.75]; colCRLB = [0.80 0.15 0.15];

for iP = 1:numel(results)
    isFirst = (iP == 1);
    if isFirst
        gdName = 'GD'; pgdName = 'PGD'; crlbName = 'CRLB'; onoff = 'on';
    else
        gdName = ''; pgdName = ''; crlbName = ''; onoff = 'off';
    end
    plot(results(iP).SNR_dB, results(iP).GD,   '--s', 'Color', colGD, ...
        'DisplayName', gdName,   'HandleVisibility', onoff);
    plot(results(iP).SNR_dB, results(iP).PGD,  '-^',  'Color', colPGD, ...
        'DisplayName', pgdName,  'HandleVisibility', onoff);
    plot(results(iP).SNR_dB, results(iP).CRLB, '-o',  'Color', colCRLB, ...
        'DisplayName', crlbName, 'HandleVisibility', onoff);

    idx = round(numel(results(iP).SNR_dB)*0.55);
    cx = results(iP).SNR_dB(idx);
    cluster = [results(iP).GD(idx), results(iP).PGD(idx), results(iP).CRLB(idx)];
    cy = mean(cluster);
    ry = max(2.5, (max(cluster)-min(cluster))/2 + 1.5);
    rx = 3;
    drawDashedEllipse(cx, cy, rx, ry, [0.3 0.3 0.3]);
    text(cx+rx+0.5, cy, sprintf('P = %d', results(iP).P), 'FontSize', 9);
end

xlim([-5 30]); ylim([-15 30]);
xlabel('SNR [dB]'); ylabel('NMSE [dB]');
title('Fig.4 reproduction: 2D Rydberg atomic array (I1=I2=8, K=3)');
legend('Location', 'northeast');

save('fig4_results.mat', 'results', 'p', ...
    'Pa_analytical', 'marginDB', 'calib', 'SNR_anchor_dB', 'NMSE_anchor_dB', 'P_anchor');

function drawDashedEllipse(cx, cy, rx, ry, color)
% DRAWDASHEDELLIPSE  Dashed ellipse in data coordinates, matching the
% paper's oval annotations that circle each P-cluster of curves.
theta = linspace(0, 2*pi, 120);
plot(cx + rx*cos(theta), cy + ry*sin(theta), '--', ...
    'Color', color, 'LineWidth', 1, 'HandleVisibility', 'off');
end
