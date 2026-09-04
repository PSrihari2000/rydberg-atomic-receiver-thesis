%% ================================================================
% FIG. 3 REPRODUCTION
% Channel Estimation for 1D Rydberg Atomic Receiver
%
% Source paper (main):
%   B. Xu, J. Zhang, Z. Chen, B. Cheng, Z. Liu, Y.-C. Wu, B. Ai,
%   "Channel Estimation for Rydberg Atomic Receivers,"
%   IEEE Wireless Communications Letters, vol. 14, no. 9, 2025.
%   -- channel model Eq.(7)-(9), linearized GD model Eq.(10)-(15),
%      CRLB Eq.(28)-(31).
%
% Source paper (GS baseline):
%   M. Cui et al., "Towards Atomic MIMO Receivers," arXiv:2404.04864.
%   -- biased Gerchberg-Saxton phase retrieval, Algorithm 1.
%
% Comparison:
%   1. Gerchberg-Saxton (GS)   -- exact nonlinear magnitude model,
%                                 alternating phase/LS minimization
%   2. Gradient Descent (GD)   -- exact nonlinear magnitude model,
%                                 direct Wirtinger-flow-style gradient
%   3. Cramer-Rao Lower Bound (CRLB)
%
% Two pilot lengths: P = 10, P = 30
%
% NOTE ON THIS VERSION vs the paper's own Eq.(13)-(15):
%   The paper derives GD from a FIRST-ORDER LINEARIZATION of the
%   magnitude measurement (Eq.10-11), then does closed-form/iterative
%   least squares on that linearized proxy. This script instead runs
%   gradient descent DIRECTLY on the exact (non-linearized) magnitude
%   objective, min_g || |S^T*g+b| - y ||^2 -- i.e. it never linearizes
%   at all, so it also never introduces the Eq.10 Taylor-truncation
%   bias GD's linearized version has to contend with. This is a
%   deliberate, disclosed departure from the literal Eq.13 objective,
%   not an error -- it is closer in spirit to "Wirtinger Flow" phase
%   retrieval than to the paper's own printed closed-form.
%
% TWO BUGS FIXED vs the version this was built from:
%   1. CRLB must be evaluated at the REAL residual noise variance
%      sigma^2/2 (Eq.11: n_bar ~ N(0,sigma^2/2)), not the complex-model
%      sigma^2 (Eq.8: n ~ CN(0,sigma^2)). Eq.28 sloppily drops this /2
%      when restating the noise model for the CRLB derivation -- Eq.11
%      is the more carefully-derived equation, so it is trusted here.
%      Using raw sigma^2 makes the reported CRLB curve sit ~3dB too
%      high (too loose a bound).
%   2. NMSE must be computed as a RATIO OF SUMS across trials --
%      NMSE = sum(||Ghat-G||^2) / sum(||G||^2) -- matching the paper's
%      own definition (Sec.V): NMSE = E{||G-Ghat||^2} / E{||G||^2}.
%      Averaging the PER-TRIAL ratio ||Ghat-G||^2/||G||^2 instead is a
%      different (and here, biased) statistical quantity, since ||G||^2
%      genuinely varies trial-to-trial (Lk ~ Uniform(3,7) per user).
%
% MARGIN/ITERATION DIAGNOSTIC (this session, P=30 SNR=20-30, 100 trials
% per combo): the first version of this script (reference_ratio=10,
% GD_iterations=500, GS_iterations=50) showed GS/GD plateauing ~10-20dB
% ABOVE CRLB at high SNR for P=30, which the paper says should not
% happen (P=30 should track CRLB closely). Isolated the cause:
%   - Raising GD_iterations alone (500->3000, GS 50->150) barely moved
%     the gap (10.7dB -> 10.1dB @ SNR=20) -- NOT an under-convergence
%     problem.
%   - Raising reference_ratio alone (10->50) did almost all the work
%     (10.7dB -> 1.7dB @ SNR=20) -- margin was the real bottleneck.
%   - Combining both (ref=50, GD=3000, GS=150) gave the best result
%     (0.8-1.4dB gap across SNR=20-30) -- adopted below.
%   - Note: reference_ratio=200 (with iterations still at 500) was
%     WORSE than 50 (5.6-14dB gap) -- margin is not simply "more is
%     better" here. Likely cause: as the reference term's absolute
%     magnitude grows, the FIXED step size mu0 becomes poorly scaled
%     for the resulting gradient magnitudes, needing proportionally
%     more iterations to converge -- raising margin without also
%     raising the iteration budget can make things worse.
%   - Re-checked P=10 under the new settings (150 trials): GD still
%     beats GS at 4 of 5 SNR points (one near-tie at SNR=15, +0.11dB,
%     well within MC noise) -- the fix for P=30 doesn't break P=10.
%
% STATUS AS OF THIS COMMIT (verified against the paper, genuine Monte
% Carlo, no visual tuning):
%   - P=10: MATCHES the paper qualitatively -- GD beats GS at essentially
%     every SNR point, both estimators sit well above CRLB (the paper's
%     own claimed P=10 behavior). Considered correct/settled.
%   - P=30: STILL AN OPEN PROBLEM. The margin=50/GD=3000/GS=150 fix above
%     looked good in an isolated 3-point spot-check (SNR=20/25/30, 100
%     trials: 0.8-1.4dB gap to CRLB) but a FULL 8-point/300-trial run
%     showed a much larger gap (7-16.6dB at SNR=20-30) with GS/GD
%     plateauing around -20dB instead of tracking CRLB down to -36dB.
%     Root-caused via a per-trial error distribution check (NOT margin,
%     NOT iteration count): GD/GS's MEDIAN per-trial performance is
%     excellent (~-36dB, right at CRLB) but 93.7%/89.6% of the TOTAL
%     squared-error sum comes from just the worst 5% of trials -- rare
%     catastrophic-failure realizations dominate the paper's own
%     ratio-of-sums NMSE definition. Tested whether a backtracking line
%     search (reject any step that doesn't reduce the true loss) fixes
%     this -- it made ZERO difference (identical results to the
%     decimal), which rules out step overshoot/instability and points
%     instead to genuine LOCAL MINIMA: this magnitude-only estimation
%     problem is fundamentally non-convex phase retrieval, and both
%     estimators' fixed starting points (GD: g=0; GS: single spectral
%     init) occasionally converge smoothly into the wrong basin. NEXT
%     STEP (not yet implemented in this file): multiple random restarts
%     per antenna, keeping whichever converges to the lowest true loss
%     -- the standard fix for local-minima issues in non-convex
%     estimation, unlike margin/iteration tuning which does not address
%     this failure mode at all.
%
% IMPORTANT:
% - CRLB is calculated separately for P=10 and P=30.
% - P=30 should generally provide better estimation performance.
% - GS/GD are NOT artificially forced to coincide with CRLB.
% - All curves result from genuine Monte-Carlo simulation.
%% ================================================================

clear;
clc;
close all;

rng(2026);   % reproducibility

%% ================================================================
% 1. SYSTEM PARAMETERS (Xu et al., Sec.V -- paper-exact unless noted)
%% ================================================================

p.q    = 1.602e-19;       % electron charge [C]
p.a0   = 5.292e-11;       % Bohr radius [m]
p.h    = 6.626e-34;       % Planck constant [J.s]
p.hbar = p.h/(2*pi);      % reduced Planck constant

% Atomic dipole moment magnitude: mu_eg = [0, 1785.9*q*a0, 0]^T (Sec.V)
p.mu = 1785.9 * p.q * p.a0;

p.I = 8;                  % number of Rydberg atomic antennas (1D array)
p.K = 3;                  % number of users

p.Lmin = 3;                % Lk ~ discrete Uniform{Lmin,...,Lmax}, Sec.V
p.Lmax = 7;

p.alpha_var  = 1;          % Var(alpha_l,k), alpha_l,k ~ CN(0,1),  Sec.V
p.alphab_var = 10;         % Var(alpha_b),   alpha_b   ~ CN(0,10), Sec.V
p.eps_var    = 1/3;        % Var of each Cartesian polarization component

% ASSUMPTION: paper gives no numeric inter-antenna spacing / wavelength.
% Half-wavelength spacing is the standard array choice; AOA/path phases
% are random per path anyway, so this does not affect NMSE statistics.
p.d_over_lambda = 0.5;


%% ================================================================
% 2. SIMULATION PARAMETERS (free/undocumented -- disclosed here)
%% ================================================================

SNR_dB = -5:5:30;          % SNR sweep, matches paper's Fig.3 x-axis
P_list = [10 30];          % pilot lengths, matches paper's Fig.3

Nmc = 300;                 % Monte Carlo trials per (P,SNR) point.
                            % Verified this session (independent 300-
                            % trial batches at several margins/points
                            % agree to within ~0.1-0.3dB, and going to
                            % 1500/6000 trials on a fixed point moved
                            % the estimate by <0.07dB) -- 300 is enough
                            % precision for the effect sizes here.

GS_iterations = 150;        % Cui et al. use t0=50 for their own problem;
                            % raised from 50 -- see margin/iteration
                            % diagnostic note above.

GD_iterations = 3000;       % GD is now literal gradient descent on the
GD_step       = 0.15;      % exact objective (see header note), so it
                            % does need an iteration count / step size,
                            % unlike a closed-form solve. Not given by
                            % the paper -- raised from 500 -- see
                            % diagnostic note above.

% ASSUMPTION: |b|/|a| power margin, enforcing the "reference dominates"
% requirement (footnote 1 of Xu et al., under Eq.10). The paper gives no
% numeric value. reference_ratio=50 means |b|^2/|a|^2 = 2500 (34dB) is
% enforced EXACTLY in every trial (not just in expectation), by rescaling
% the reference term Bc to hit this ratio against that trial's own
% realized signal power. Raised from 10 -- see diagnostic note above
% (this was the dominant fix for P=30's high-SNR floor).
reference_ratio = 50;

eps_num = 1e-12;           % small number for numerical safety


%% ================================================================
% 3. STORAGE VARIABLES
%% ================================================================

numSNR = length(SNR_dB);
numP   = length(P_list);

NMSE_GS   = zeros(numP,numSNR);
NMSE_GD   = zeros(numP,numSNR);
NMSE_CRLB = zeros(numP,numSNR);


%% ================================================================
% 4. MAIN SIMULATION
%% ================================================================

fprintf('====================================================\n');
fprintf(' FIG. 3: Rydberg Atomic Receiver Channel Estimation\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n', Nmc);
fprintf('Antennas I         = %d\n', p.I);
fprintf('Users K            = %d\n', p.K);
fprintf('Pilot lengths      = [%d %d]\n', P_list(1), P_list(2));
fprintf('====================================================\n\n');

for pp = 1:numP

    P = P_list(pp);

    fprintf('\n============================================\n');
    fprintf('Simulating P = %d\n', P);
    fprintf('============================================\n');

    % ---- BUGFIX 2: accumulate RAW SUMS across trials (numerator and
    % denominator separately), not per-trial ratios. NMSE is formed
    % ONCE at the end as sum(err)/sum(energy), matching the paper's own
    % E{...}/E{...} definition (Sec.V). ----
    sumSqErr_GS   = zeros(1,numSNR);
    sumSqErr_GD   = zeros(1,numSNR);
    sumSqG        = zeros(1,numSNR);
    sumCRB        = zeros(1,numSNR);

    %% ------------------------------------------------------------
    % MONTE CARLO LOOP
    %% ------------------------------------------------------------

    for mc = 1:Nmc

        %% ========================================================
        % STEP A: GENERATE TRUE CHANNEL G, PILOTS S, REFERENCE Bbase
        %
        % G(i,k), Eq.(7):
        %   g_{i,k} = sum_l (1/hbar) * mu_eg^T*eps_{i,k,l} * alpha_{l,k}
        %             * exp(-j*(i-1)*psi_{l,k})
        % (only the y-component of mu_eg survives the dot product,
        %  since mu_eg = [0, mu, 0]^T -- hence "eps_y" below)
        %% ========================================================

        [G,S,Bbase] = generate_trial_1D(p,P);

        % Desired complex pilot signal, Eq.(8)'s a_{i,p} term:
        %   X(i,p) = sum_k G(i,k)*S(k,p)
        X = G*S;

        %% ========================================================
        % STEP B: NORMALIZE CHANNEL SCALE
        %
        % Physical constants (mu/hbar etc.) create very large raw
        % numbers, but NMSE is scale-invariant. Normalize so the
        % OBSERVABLE signal X has unit average power; everything
        % downstream (SNR, margin) is then defined relative to this
        % normalized scale, sidestepping the need to track how the
        % physical constants would otherwise cancel out of NMSE.
        %% ========================================================

        signal_power = mean(abs(X(:)).^2);
        if signal_power < eps_num
            continue;
        end
        scale_factor = sqrt(signal_power);
        G = G / scale_factor;
        X = G*S;

        %% ========================================================
        % STEP C: CREATE STRONG KNOWN REFERENCE Bc, Eq.(9)
        %
        % Rescale Bbase so mean(|Bc|^2)/mean(|X|^2) = reference_ratio^2
        % EXACTLY for this trial -- the "reference dominates" condition
        % (footnote 1, under Eq.10) that the whole linearization/phase-
        % retrieval framework depends on for identifiability.
        %% ========================================================

        Bbase_power = mean(abs(Bbase(:)).^2);
        desired_B_power = reference_ratio^2 * mean(abs(X(:)).^2);
        if Bbase_power < eps_num
            continue;
        end
        Bc = Bbase * sqrt(desired_B_power/Bbase_power);

        %% ========================================================
        % STEP D: LOOP OVER SNR
        %% ========================================================

        for ss = 1:numSNR

            snr_linear = 10^(SNR_dB(ss)/10);

            % SNR := (signal power of X) / (noise power), a direct,
            % textbook definition -- the paper gives no equation
            % mapping "SNR" to sigma^2, so this fills that gap without
            % needing to anchor to a point on the paper's own figure.
            signal_power = mean(abs(X(:)).^2);
            sigma2 = signal_power/snr_linear;   % Eq.(8)'s n ~ CN(0,sigma^2)

            N = sqrt(sigma2/2) * (randn(p.I,P) + 1i*randn(p.I,P));

            %% ====================================================
            % STEP E: RYDBERG MAGNITUDE-ONLY OBSERVATION, Eq.(8)
            %   Y(i,p) = | sum_k G(i,k)*S(k,p) + B(i,p) + N(i,p) |
            %% ====================================================

            Y = abs(X + Bc + N);

            %% ====================================================
            % STEP F: GS CHANNEL ESTIMATION (Cui et al. Algorithm 1)
            %% ====================================================

            Ghat_GS = gs_estimator_1D(Y, Bc, S, GS_iterations);

            %% ====================================================
            % STEP G: GD CHANNEL ESTIMATION (exact-model Wirtinger
            % gradient descent -- see header note)
            %% ====================================================

            Ghat_GD = gd_estimator_1D(Y, Bc, S, GD_iterations, GD_step);

            %% ====================================================
            % STEP H: ACCUMULATE NMSE numerator/denominator, Eq.(Sec.V):
            %   NMSE = E{||G-Ghat||_F^2} / E{||G||_F^2}
            %% ====================================================

            channel_energy = norm(G,'fro')^2;

            sumSqErr_GS(ss) = sumSqErr_GS(ss) + norm(Ghat_GS-G,'fro')^2;
            sumSqErr_GD(ss) = sumSqErr_GD(ss) + norm(Ghat_GD-G,'fro')^2;
            sumSqG(ss)      = sumSqG(ss)      + channel_energy;

            %% ====================================================
            % STEP I: CRLB, Eq.(28)-(31)
            %
            %   CRB(g) = 4*sigma_r^2 * (I_I kron (S*.S^T)^-1)
            %   trace(CRB) = I * 4*sigma_r^2 * trace((S*.S^T)^-1)
            %
            % BUGFIX 1: sigma_r^2 = sigma2/2 is the REAL residual noise
            % variance Eq.11 derives (n_bar ~ N(0,sigma2/2) from Eq.8's
            % complex n~CN(0,sigma2)) -- NOT sigma2 itself. Eq.28 restates
            % the noise model as N(0,sigma2*I), sloppily dropping the /2
            % that Eq.11 explicitly established a few equations earlier;
            % Eq.11's derivation is trusted as the more careful one.
            %% ====================================================

            FIM_part = conj(S)*S.';         % = S* S^T, Eq.(31)
            FIM_inv  = pinv(FIM_part);

            sigma2_real = sigma2/2;          % BUGFIX 1
            trace_CRLB = 4 * sigma2_real * p.I * real(trace(FIM_inv));

            sumCRB(ss) = sumCRB(ss) + trace_CRLB;

        end

        if mod(mc,50)==0
            fprintf('P=%d : Monte Carlo %d / %d\n', P,mc,Nmc);
        end

    end

    %% ============================================================
    % STEP J: FORM NMSE AS A RATIO OF SUMS (BUGFIX 2), THEN TO dB
    %% ============================================================

    NMSE_GS(pp,:)   = 10*log10(max(sumSqErr_GS ./ sumSqG, eps_num));
    NMSE_GD(pp,:)   = 10*log10(max(sumSqErr_GD ./ sumSqG, eps_num));
    NMSE_CRLB(pp,:) = 10*log10(max(sumCRB      ./ sumSqG, eps_num));

end


%% ================================================================
% 5. DISPLAY NUMERICAL RESULTS
%% ================================================================

fprintf('\n\n============================================\n');
fprintf('FINAL RESULTS\n');
fprintf('============================================\n');

for pp = 1:numP
    P = P_list(pp);
    fprintf('\nP = %d\n',P);
    fprintf('SNR     GS        GD        CRLB\n');
    for ss = 1:numSNR
        fprintf('%3d   %8.3f   %8.3f   %8.3f\n', ...
            SNR_dB(ss), NMSE_GS(pp,ss), NMSE_GD(pp,ss), NMSE_CRLB(pp,ss));
    end
end


%% ================================================================
% 6. PAPER-STYLE FIGURE
%% ================================================================

figure('Color','w','Position',[150 100 850 620]);
hold on; grid on; box on;

% ---- P = 10 ----
hGS = plot(SNR_dB,NMSE_GS(1,:), '--d', 'Color',[0.1 0.1 0.1], ...
    'LineWidth',1.8, 'MarkerSize',6, 'MarkerFaceColor','w');
hGD = plot(SNR_dB,NMSE_GD(1,:), '-.s', 'Color',[0.85 0.55 0.05], ...
    'LineWidth',1.8, 'MarkerSize',6, 'MarkerFaceColor','w');
hCRLB = plot(SNR_dB,NMSE_CRLB(1,:), '-o', 'Color',[0.75 0.25 0.05], ...
    'LineWidth',2.0, 'MarkerSize',6, 'MarkerFaceColor','w');

% ---- P = 30 (same styles, no duplicate legend entries) ----
plot(SNR_dB,NMSE_GS(2,:), '--d', 'Color',[0.1 0.1 0.1], ...
    'LineWidth',1.8, 'MarkerSize',6, 'MarkerFaceColor','w', 'HandleVisibility','off');
plot(SNR_dB,NMSE_GD(2,:), '-.s', 'Color',[0.85 0.55 0.05], ...
    'LineWidth',1.8, 'MarkerSize',6, 'MarkerFaceColor','w', 'HandleVisibility','off');
plot(SNR_dB,NMSE_CRLB(2,:), '-o', 'Color',[0.75 0.25 0.05], ...
    'LineWidth',2.0, 'MarkerSize',6, 'MarkerFaceColor','w', 'HandleVisibility','off');

xlabel('SNR [dB]', 'FontSize',13, 'FontWeight','bold');
ylabel('NMSE [dB]', 'FontSize',13, 'FontWeight','bold');
title('Channel Estimation for 1D Rydberg Atomic Receiver', 'FontSize',14, 'FontWeight','bold');
xlim([-5 30]); xticks(-5:5:30);
ylim([-30 20]); yticks(-30:5:20);
set(gca, 'FontSize',11, 'LineWidth',1.0);

legend([hGS hGD hCRLB], {'GS','GD','CRLB'}, 'Location','southwest', 'FontSize',11);

% Pilot-length annotations -- adjust positions if your curves differ
text(12,5,'P = 10', 'FontSize',11, 'FontWeight','bold');
text(11,-15,'P = 30', 'FontSize',11, 'FontWeight','bold');

hold off;


%% ================================================================
% 7. SAVE RESULTS
%% ================================================================

results.SNR_dB = SNR_dB;
results.P_list = P_list;
results.NMSE_GS_dB = NMSE_GS;
results.NMSE_GD_dB = NMSE_GD;
results.NMSE_CRLB_dB = NMSE_CRLB;

save('fig3_results.mat','results');

fprintf('\n============================================\n');
fprintf('Simulation completed.\n');
fprintf('Results saved to fig3_results.mat\n');
fprintf('============================================\n');


%% ================================================================
%% LOCAL FUNCTIONS
%% ================================================================

function [G,S,Bc] = generate_trial_1D(p,P)
% GENERATE_TRIAL_1D  One Monte Carlo draw of the channel, pilots, and
% reference term.
%
%   G  (I x K complex) -- true channel, Eq.(7)
%   S  (K x P complex) -- pilot matrix, entries iid CN(0,1)
%   Bc (I x P complex) -- reference term Bbase at UNIT reference-symbol
%                         power (final power scaling is applied by the
%                         caller via reference_ratio, Step C)

I = p.I; K = p.K;
i_idx = (0:I-1).';   % physical antenna index, 0..I-1

%% ---- CHANNEL MATRIX G, Eq.(7) ----
%   g_{i,k} = sum_{l=1}^{Lk} (1/hbar)*mu_eg^T*eps_{i,k,l}*alpha_{l,k}
%             * exp(-j*(i-1)*psi_{l,k}),   psi_{l,k} = 2*pi*(d/lambda)*cos(theta_l,k)
Lk = randi([p.Lmin,p.Lmax],K,1);
G = zeros(I,K);
for k = 1:K
    Lkk = Lk(k);
    theta = 2*pi*rand(Lkk,1);                 % AOA per path (isotropic)
    phi   = 2*pi*p.d_over_lambda*cos(theta);   % spatial phase step per path
    eps_y = sqrt(p.eps_var)*randn(Lkk,1);      % mu_eg=[0,mu,0]^T -> only y-component of eps matters
    alpha = sqrt(p.alpha_var/2)*(randn(Lkk,1)+1i*randn(Lkk,1));  % alpha_l,k ~ CN(0,1)
    coeff = (p.mu/p.hbar) .* eps_y .* alpha;   % Lkk x 1
    Ephase = exp(-1i*i_idx*phi.');             % I x Lkk, antenna spatial response
    G(:,k) = Ephase*coeff;                     % sum over paths -> I x 1
end

%% ---- PILOT MATRIX S, iid CN(0,1) ----
S = sqrt(1/2) * (randn(K,P) + 1i*randn(K,P));

%% ---- REFERENCE TERM Bc, Eq.(9) (single dominant LOS path) ----
%   b_{i,p} = s_{b,p} * mu_eg^T*eps_{b,i} * alpha_b * exp(-j*(i-1)*psi_b)
theta_b = 2*pi*rand();
phi_b   = 2*pi*p.d_over_lambda*cos(theta_b);
eps_b_y = sqrt(p.eps_var)*randn(I,1);                 % per-antenna polarization sample
alpha_b = sqrt(p.alphab_var/2)*(randn()+1i*randn());  % alpha_b ~ CN(0,10)
sb      = sqrt(1/2)*(randn(1,P)+1i*randn(1,P));       % UNIT-power reference symbols;
                                                       % final scaling done by caller
coeff_b = (p.mu/p.hbar) * eps_b_y * alpha_b .* exp(-1i*i_idx*phi_b);  % I x 1
Bc = coeff_b*sb;                                       % I x P (outer product)

end


function Ghat = gd_estimator_1D(Y,Bc,S,maxIter,mu0)
% GD_ESTIMATOR_1D  Gradient descent directly on the EXACT (non-linearized)
% magnitude objective, per antenna:
%
%   min_g  f(g) = sum_p ( |S(:,p)^T*g + b_p| - y_p )^2
%
% Gradient (Wirtinger calculus, treating g and conj(g) as independent):
% z_p = S(:,p)^T*g + b_p is HOLOMORPHIC in g, so
%   d|z_p|/d(conj g) = (z_p/(2|z_p|)) * conj(S(:,p))
%   df/d(conj g)     = sum_p (|z_p|-y_p) * (z_p/|z_p|) * conj(S(:,p))
%                     = conj(S) * ( residual .* (z./|z|) )
% Moving opposite this direction decreases f -- standard real steepest
% descent for a real-valued function of a complex variable.
%
% This does NOT linearize the magnitude relationship the way the paper's
% own Eq.(10)-(15) does, so it carries no Eq.(10) Taylor-truncation bias,
% and its effective "phase reference" z/|z| is recomputed fresh every
% iteration from the CURRENT g (adaptive), unlike a fixed-point
% linearization anchored only at g=0.

[I,~] = size(Y);
K = size(S,1);
eps_num = 1e-12;

Ghat = zeros(I,K);

for i = 1:I
    y = Y(i,:).';
    b = Bc(i,:).';

    g = zeros(K,1);   % start at g=0 (z=b, a reasonable start since b dominates)

    for iter = 1:maxIter
        z = S.'*g + b;                  % P x 1, predicted complex signal
        mag_z = abs(z);
        residual = mag_z - y;           % P x 1
        phase_dir = z ./ max(mag_z,eps_num);   % z/|z|, adaptive phase reference

        grad = conj(S) * (residual .* phase_dir);   % K x 1
        grad = grad/size(S,2);          % normalize by pilot count P

        step = mu0 / sqrt(1 + 0.01*(iter-1));  % mild decaying step size

        g_new = g - step*grad;
        if any(~isfinite(g_new))
            break;
        end
        g = g_new;
    end

    Ghat(i,:) = g.';
end

end


function Ghat = gs_estimator_1D(Y,Bc,S,t0)
% GS_ESTIMATOR_1D  Biased Gerchberg-Saxton channel estimator, Cui et al.
% "Towards Atomic MIMO Receivers," Algorithm 1, adapted per-antenna:
% treat the antenna's channel row g_i (K unknowns) as the unknown
% "symbol vector," and A=conj(S) as the known "channel matrix."
%
% Spectral initialization (Cui et al. Eq.26, Algorithm 1 steps 1-4):
%   Abar = [A^H, b]^H  ->  top K rows = A exactly (traced from
%   Abar^H = [A^H, b]), so under A=conj(S), Sbar = [conj(S); conj(b)^T].
%   M = sum_p y_p * abar_p * abar_p^H ; v = principal eigenvector of M.
%   rbar = |v^H*Abar|*y / ||Abar^H*v||^2 ;  s0 = rbar*v.
%   g = exp(-j*angle(s0(K+1))) * s0(1:K)  -- de-rotate by bias phase.
%
% Alternating minimization (Algorithm 1 steps 5-8):
%   theta^t = angle(A^H*g^{t-1} + b)
%   g^t     = (A*A^H)^-1 * A * (y.*exp(j*theta^t) - b)

[I,~] = size(Y);
K = size(S,1);

M = conj(S)*S.';      % = A*A^H under A=conj(S)
Minv = pinv(M);        % pseudoinverse for numerical stability
Sc = conj(S);           % = A

Ghat = zeros(I,K);

for i = 1:I
    yi = Y(i,:).';
    bi = Bc(i,:).';

    %% ---- spectral initialization ----
    Sbar  = [conj(S); conj(bi).'];        % (K+1) x P
    Mfull = Sbar * diag(yi) * Sbar';      % (K+1) x (K+1), Hermitian PSD
    [V,D] = eig(Mfull);
    [~,idx] = max(real(diag(D)));
    v = V(:,idx);

    numerator   = abs(v'*Sbar)*yi;
    denominator = norm(Sbar'*v)^2;
    rbar = numerator/max(denominator,1e-12);

    s0 = rbar*v;
    g = exp(-1i*angle(s0(K+1))) * s0(1:K);

    %% ---- alternating phase / least-squares iteration ----
    for t = 1:t0
        theta = angle(S.'*g + bi);
        complex_observation = yi .* exp(1i*theta);
        desired_part = complex_observation - bi;
        g = Minv * (Sc*desired_part);
    end

    Ghat(i,:) = g.';
end

end
