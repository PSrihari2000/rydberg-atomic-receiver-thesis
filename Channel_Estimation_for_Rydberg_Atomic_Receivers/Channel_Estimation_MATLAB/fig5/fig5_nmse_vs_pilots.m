% FIG5_NMSE_VS_PILOTS  Reproduce Fig.5 of Xu et al., "Channel Estimation for
% Rydberg Atomic Receivers" (IEEE WCL, Sept 2025): NMSE vs pilot length P at
% fixed SNR = 5 dB, 2D Rydberg atom-based antenna array (I1=I2=8), comparing
% PGD (Algorithm 1) and CRLB. Paper's own Fig.5 legend has only these two
% curves (no GD) -- confirmed directly from the figure crop.
%
% Same physical setup/calibration as fig4/fig4_nmse_vs_snr.m (same marginDB,
% same noise-scale anchor) so this figure's SNR=5dB operating point is
% consistent with fig4's own SNR=5dB column, not independently re-tuned.
%
% Run this script directly (it adds its own folder to the path).

clear; clc; close all;
addpath(fileparts(mfilename('fullpath')));

p = params_2D();
IJ = p.I1*p.I2;

% ---------------- simulation controls ----------------
P_list      = 5:5:50;   % pilot length sweep, matching paper's Fig.5 x-axis
SNR_dB      = 5;        % fixed, per paper's Fig.5 caption
numTrials   = 200;
maxIter_PGD = 2000;
reltol_PGD  = 1e-10;

% Same margin/anchor assumptions as fig4 -- see that script's ASSUMPTION 1/2
% comments for the full reasoning; reused verbatim here for consistency.
marginDB       = 60;
P_anchor       = 30;
SNR_anchor_dB  = -5;
NMSE_anchor_dB = 22;

EL = mean(p.Lmin:p.Lmax);
Pa_analytical = p.K * EL * p.mu^2 / (3*p.hbar^2);

Pb_target = Pa_analytical * 10^(marginDB/10);
p.sb2 = Pb_target / (p.alphab_var * p.mu^2 / (3*p.hbar^2));

traceMinv_anchor = p.K/(P_anchor - p.K);
SNR_lin_anchor   = 10^(SNR_anchor_dB/10);
NMSE_lin_anchor  = 10^(NMSE_anchor_dB/10);
calib = NMSE_lin_anchor * SNR_lin_anchor / (2*traceMinv_anchor);

sigma2 = calib * Pa_analytical / 10^(SNR_dB/10);   % fixed, since SNR is fixed here

fprintf('Pa_analytical=%.4g, margin=%ddB, calib=%.4g, sigma2=%.4g (SNR=%ddB fixed)\n', ...
    Pa_analytical, marginDB, calib, sigma2, SNR_dB);

% ---------------- PGD / CRLB Monte Carlo sweep over pilot length ----------------
nmse_PGD  = zeros(size(P_list));
nmse_CRLB = zeros(size(P_list));

for iP = 1:numel(P_list)
    P = P_list(iP);
    sumSqErr_PGD = 0; sumSqG = 0; sumCRB = 0;

    for trial = 1:numTrials
        [Gflat, S, Bflat, Lk] = generate_trial_2D(p, P);

        A = Gflat*S;
        N = sqrt(sigma2/2)*(randn(IJ,P)+1i*randn(IJ,P));
        Y = abs(A + Bflat + N);

        absB = abs(Bflat);
        Z    = exp(-1i*angle(Bflat));

        Ghat_PGD = pgd_estimator_2D(Y, absB, Z, S, p.I1, p.I2, Lk, maxIter_PGD, reltol_PGD);

        sumSqErr_PGD = sumSqErr_PGD + norm(Gflat-Ghat_PGD,'fro')^2;
        sumSqG       = sumSqG + norm(Gflat,'fro')^2;
        sumCRB       = sumCRB + crlb_trace_1D(sigma2/2, S, IJ);
    end

    nmse_PGD(iP)  = 10*log10(sumSqErr_PGD/sumSqG);
    nmse_CRLB(iP) = 10*log10(sumCRB/sumSqG);

    fprintf('P=%3d  ->  PGD=%.2f dB  CRLB=%.2f dB\n', P, nmse_PGD(iP), nmse_CRLB(iP));
end

% ---------------- plot ----------------
figure; hold on; grid on; box on;
plot(P_list, nmse_PGD,  '-^', 'Color', [0.1 0.35 0.75], 'DisplayName', 'PGD');
plot(P_list, nmse_CRLB, '-o', 'Color', [0.80 0.15 0.15], 'DisplayName', 'CRLB');

xlim([5 50]); ylim([-15 30]);
xlabel('Pilot Length'); ylabel('NMSE [dB]');
title('Fig.5 reproduction: NMSE vs pilot length, SNR=5dB, 2D array (I1=I2=8, K=3)');
legend('Location', 'northeast');

save('fig5_results.mat', 'P_list', 'nmse_PGD', 'nmse_CRLB', 'p', ...
    'Pa_analytical', 'marginDB', 'calib', 'SNR_dB');
