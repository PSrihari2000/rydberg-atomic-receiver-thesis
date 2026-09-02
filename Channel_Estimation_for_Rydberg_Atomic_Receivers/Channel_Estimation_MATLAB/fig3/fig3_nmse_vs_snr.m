% FIG3_NMSE_VS_SNR  Reproduce Fig.3 of Xu et al., "Channel Estimation for
% Rydberg Atomic Receivers" (IEEE WCL, Sept 2025): NMSE vs SNR for a 1D
% Rydberg atom-based antenna array, comparing GS, GD, and CRLB, at
% P = 10 and P = 30 pilots.
%
% There are SIX underlying curves (GS/GD/CRLB x P=10/P=30) but only THREE
% legend entries, exactly like the paper's own Fig.3: each curve-type uses
% one style for both P values, distinguished only by cluster position. At
% P=30, GS, GD, and CRLB nearly coincide (paper's own text: "both the
% proposed GD and the GS algorithm perform almost identically and close
% to the CRLB"); at P=10 they visibly separate, with GS saturating higher
% than GD, both well above the (higher, since fewer pilots) P=10 CRLB.
%
% Run this script directly (it adds its own folder to the path).

clear; clc; close all;
addpath(fileparts(mfilename('fullpath')));

p = params_1D();

% ---------------- simulation controls (not given numerically by the paper) ----------------
SNR_dB_list = -5:5:30;
P_list      = [10 30];
numTrials   = 300;      % Monte Carlo trials per (P, SNR) point
t0_GS       = 100;      % GS iterations (Cui et al. use 50 for their own, better-conditioned
                         % setting; doubled here since P=10,K=3 is a tighter regime)
% NOTE: GD no longer needs iteration/tolerance controls -- gd_estimator_1D.m now
% solves Eq.13 in closed form (exact per-antenna real LS) instead of iterating,
% since GD was found not fully converging within a fixed iteration budget, which
% was masquerading as a modeling-bias floor. See gd_estimator_1D.m's docstring.

% ASSUMPTION 1: |b|/|a| power margin enforcing the paper's "reference dominates"
% linearization requirement (Eq.10). This is a genuine trade-off: if margin is too
% LARGE, the GS phase-update theta=angle(S^T*g+b_i) locks onto angle(b_i) almost
% independent of g, collapsing GS's "exact nonlinear" problem into the same linear
% problem GD solves, erasing GS's own local-optima saturation. If margin is too
% SMALL, GD's own Taylor-truncation bias (Eq.10's dropped O(x^2) term, does NOT
% shrink with SNR) dominates -- and empirically (margin=25, with gd_estimator_1D.m's
% now-exact closed-form solve, so this floor is a genuine model-bias effect, not
% under-convergence) that bias came out LARGER than GS's own floor, inverting the
% paper's expected GD<=GS ordering. NOTE: the earlier "margin=60 erases GS's
% nonlinearity" conclusion was drawn from a run where GD was NOT yet using the
% closed-form solve (500 iterations only), so it's being re-tested here now that
% GD is trustworthy -- that conclusion may itself have been a convergence artifact.
marginDB = 45;

% ASSUMPTION 2: the paper never states a numeric SNR-to-noise-variance mapping
% anywhere (no equation defines "SNR" in terms of sigma^2 and other quantities), so
% the ABSOLUTE vertical position of these curves is fundamentally undetermined from
% the paper text/equations alone -- only the SLOPE is fixed by the model (exactly
% -1 dB/dB per curve, since CRLB is linear in sigma^2, and the P=10 vs P=30 CRLB
% curves are exactly parallel, offset by 10*log10((P30-K)/(P10-K)) dB). We anchor
% our noise scale so the P=30 CRLB curve passes through the paper's own Fig.3 at
% one easily-read point (its bottom cluster's left edge); everything else (the
% P=10 saturation gap, the P=30 tracking, the GS-vs-GD ordering, the P=10/P=30
% CRLB gap) then follows from the physics, not from this calibration.
P_anchor       = 30;
SNR_anchor_dB  = -5;
NMSE_anchor_dB = 15;

% ---------------- reference pilot power calibration (marginDB) ----------------
% E{|a_i,p|^2} = K * E{Lk} * mu^2 / (3*hbar^2)   (derived from Eq.7, see chat)
EL = mean(p.Lmin:p.Lmax);
Pa_analytical = p.K * EL * p.mu^2 / (3*p.hbar^2);

% E{|b_i,p|^2} = alphab_var * sb2 * mu^2 / (3*hbar^2)   (derived from Eq.9)
Pb_target = Pa_analytical * 10^(marginDB/10);
p.sb2 = Pb_target / (p.alphab_var * p.mu^2 / (3*p.hbar^2));

% ---------------- noise-scale anchor (SNR_anchor_dB, NMSE_anchor_dB, at P_anchor) ----------------
% NMSE_CRLB(P) = 4*sigma2_real*trace(Minv_P)/Pa_analytical, where sigma2_real is the
% REAL-valued residual noise variance the CRLB (Eq.31) is actually derived for. With
% sigma2 = calib*Pa_analytical/SNR_lin, Pa_analytical cancels completely -- confirms
% the physical constants (mu, hbar, ...) do NOT set the CRLB curves' height or
% spacing, only trace(Minv_P) (i.e. P,K) and this calibration constant do. trace(Minv_P)
% for iid CN(0,1) pilots has the exact closed form E{trace((S*S^H)^-1)} = K/(P-K)
% (complex Wishart) -- this also sets the exact, paper-independent P=10 vs P=30 CRLB
% gap: 10*log10((P_anchor-K)/(min(P_list)-K)) dB.
%
% BUGFIX: sigma2_real = sigma2/2, NOT sigma2. Eq.11 derives the real residual noise as
% n_bar ~ N(0, sigma2/2) from the complex n~CN(0,sigma2) of Eq.8, but Eq.28 (which the
% CRLB is built from) declares it N(0,sigma2*I) -- an internal inconsistency in the
% paper's own notation chain (the /2 from Eq.11 appears to get dropped by Eq.28). We
% follow Eq.11's explicit derivation, since it's the more carefully-derived equation:
% CRLB must be evaluated at the REAL noise variance actually present in GD/GS's
% residual, sigma2/2, not the complex-model sigma2. Passing the wrong (2x too large)
% value here is exactly what made CRLB sit above GD/GS instead of bounding them from
% below -- caught by comparing against a real run, see chat.
traceMinv_anchor = p.K/(P_anchor - p.K);
SNR_lin_anchor   = 10^(SNR_anchor_dB/10);
NMSE_lin_anchor  = 10^(NMSE_anchor_dB/10);
calib = NMSE_lin_anchor * SNR_lin_anchor / (2*traceMinv_anchor);   % "2" = 4*(1/2) folding in the sigma2/2 fix

fprintf('Pa_analytical=%.4g, margin=%ddB, calib=%.4g (anchored to %ddB NMSE at %ddB SNR, P=%d)\n', ...
    Pa_analytical, marginDB, calib, NMSE_anchor_dB, SNR_anchor_dB, P_anchor);

sigma2_of_SNR = @(SNR_dB) calib * Pa_analytical / 10^(SNR_dB/10);

% ---------------- GD / GS / CRLB Monte Carlo sweep (per P) ----------------
results = struct('P', {}, 'SNR_dB', {}, 'GD', {}, 'GS', {}, 'CRLB', {});

for iP = 1:numel(P_list)
    P = P_list(iP);
    nmse_GD   = zeros(size(SNR_dB_list));
    nmse_GS   = zeros(size(SNR_dB_list));
    nmse_CRLB = zeros(size(SNR_dB_list));

    for iSNR = 1:numel(SNR_dB_list)
        SNR_dB = SNR_dB_list(iSNR);
        sigma2 = sigma2_of_SNR(SNR_dB);

        sumSqErr_GD = 0; sumSqErr_GS = 0; sumSqG = 0; sumCRB = 0;

        for trial = 1:numTrials
            [G, S, Bc] = generate_trial_1D(p, P);

            A = G*S;                                        % I x P, true noiseless signal term
            N = sqrt(sigma2/2)*(randn(p.I,P)+1i*randn(p.I,P));
            Y = abs(A + Bc + N);                             % exact model, Eq.(8)

            absB = abs(Bc);
            Z    = exp(-1i*angle(Bc));

            Ghat_GD = gd_estimator_1D(Y, absB, Z, S);
            Ghat_GS = gs_estimator_1D(Y, Bc, S, t0_GS);

            sumSqErr_GD = sumSqErr_GD + norm(G-Ghat_GD,'fro')^2;
            sumSqErr_GS = sumSqErr_GS + norm(G-Ghat_GS,'fro')^2;
            sumSqG      = sumSqG + norm(G,'fro')^2;
            sumCRB      = sumCRB + crlb_trace_1D(sigma2/2, S, p.I);  % sigma2/2: see BUGFIX note above
        end

        nmse_GD(iSNR)   = 10*log10(sumSqErr_GD/sumSqG);
        nmse_GS(iSNR)   = 10*log10(sumSqErr_GS/sumSqG);
        nmse_CRLB(iSNR) = 10*log10(sumCRB/sumSqG);

        fprintf('P=%2d  SNR=%3d dB  ->  GD=%.2f dB  GS=%.2f dB  CRLB=%.2f dB\n', ...
            P, SNR_dB, nmse_GD(iSNR), nmse_GS(iSNR), nmse_CRLB(iSNR));
    end

    results(iP).P      = P;
    results(iP).SNR_dB = SNR_dB_list;
    results(iP).GD     = nmse_GD;
    results(iP).GS     = nmse_GS;
    results(iP).CRLB   = nmse_CRLB;
end

% ---------------- plot (6 curves, 3 legend entries, matching the paper) ----------------
figure; hold on; grid on; box on;
colGS = 'k'; colGD = [0.85 0.55 0.1]; colCRLB = [0.80 0.15 0.15];

for iP = 1:numel(results)
    isFirst = (iP == 1);
    if isFirst
        gsName = 'GS'; gdName = 'GD'; crlbName = 'CRLB'; onoff = 'on';
    else
        gsName = ''; gdName = ''; crlbName = ''; onoff = 'off';
    end
    plot(results(iP).SNR_dB, results(iP).GS,   '--d', 'Color', colGS, ...
        'DisplayName', gsName,   'HandleVisibility', onoff);
    plot(results(iP).SNR_dB, results(iP).GD,   '--s', 'Color', colGD, ...
        'DisplayName', gdName,   'HandleVisibility', onoff);
    plot(results(iP).SNR_dB, results(iP).CRLB, '-o',  'Color', colCRLB, ...
        'DisplayName', crlbName, 'HandleVisibility', onoff);

    % dashed ellipse + "P = 10"/"P = 30" label circling each cluster, like the paper
    idx = round(numel(results(iP).SNR_dB)*0.55);
    cx = results(iP).SNR_dB(idx);
    cluster = [results(iP).GS(idx), results(iP).GD(idx), results(iP).CRLB(idx)];
    cy = mean(cluster);
    ry = max(2.5, (max(cluster)-min(cluster))/2 + 1.5);   % vertical radius: spans the 3 curves + padding
    rx = 3;                                                % horizontal radius [dB SNR]
    drawDashedEllipse(cx, cy, rx, ry, [0.3 0.3 0.3]);
    text(cx+rx+0.5, cy, sprintf('P = %d', results(iP).P), 'FontSize', 9);
end

xlim([-5 30]); ylim([-30 20]);
xlabel('SNR [dB]'); ylabel('NMSE [dB]');
title('Fig.3 reproduction: 1D Rydberg atomic array (I=8, K=3)');
legend('Location', 'northeast');

save('fig3_results.mat', 'results', 'p', ...
    'Pa_analytical', 'marginDB', 'calib', 'SNR_anchor_dB', 'NMSE_anchor_dB', 'P_anchor');

function drawDashedEllipse(cx, cy, rx, ry, color)
% DRAWDASHEDELLIPSE  Dashed ellipse in data coordinates, matching the
% paper's oval annotations that circle each P-cluster of curves.
theta = linspace(0, 2*pi, 120);
plot(cx + rx*cos(theta), cy + ry*sin(theta), '--', ...
    'Color', color, 'LineWidth', 1, 'HandleVisibility', 'off');
end
