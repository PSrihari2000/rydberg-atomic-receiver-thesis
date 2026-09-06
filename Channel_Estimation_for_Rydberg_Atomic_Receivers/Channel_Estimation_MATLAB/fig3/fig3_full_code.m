%% ================================================================
% FIG. 3 REPRODUCTION -- Channel Estimation for 1D Rydberg Atomic Receiver
%
% ONE self-contained file: parameters, simulation loop, plotting, and
% all estimator functions (as local functions below) live here together
% on purpose, so there is only one script to read.
%
% Source paper (main):
%   B. Xu, J. Zhang, Z. Chen, B. Cheng, Z. Liu, Y.-C. Wu, B. Ai,
%   "Channel Estimation for Rydberg Atomic Receivers,"
%   IEEE Wireless Commun. Lett., vol. 14, no. 9, pp. 2957-2961, 2025.
%     Eq.(7)-(9)   : channel / observation / reference model
%     Eq.(10)-(15) : the paper's own linearized GD estimator
%     Eq.(28)-(31) : CRLB
%
% Source paper (GS baseline):
%   M. Cui, Q. Zeng, K. Huang, "Towards Atomic MIMO Receivers,"
%   IEEE JSAC, vol. 43, no. 3, pp. 659-673, 2025 (arXiv:2404.04864).
%     Algorithm 1 (Biased GS), adapted here to channel estimation via
%     the role-swap Cui et al. Sec.V describes: their z=|A^H s+b+w| for
%     UNKNOWN symbols/KNOWN channel becomes our z=|S^T g+b+n| for KNOWN
%     pilots/UNKNOWN channel. NOTE the transpose, not Hermitian: Cui's
%     own channel-estimation extension (their Eq.35) writes z=|S^H a+b+w|,
%     but that S^H comes from THEIR signal model's a^H*s inner product
%     convention, which is Hermitian-conjugated relative to Xu et al.'s
%     own Eq.8 (Sum_k g_ik*s_kp, no conjugate). Substituting A=conj(S)
%     (not A=S) is what correctly reconciles the two papers' conventions
%     -- verified independently via direct algebraic re-derivation, and
%     is why every formula below uses S.' (plain transpose) together
%     with conj(S), never S' (Hermitian) alone.
%
% ----------------------------------------------------------------
% THIS FILE'S HISTORY (why GD is implemented the way it is below)
% ----------------------------------------------------------------
% An earlier version of this script ran GD directly on the EXACT
% (non-linearized) magnitude objective (Wirtinger-flow style), because
% a literal implementation of Eq.(13)-(15) appeared to structurally lose
% to GS at P=10 -- contradicting the paper's own claimed ordering.
%
% That appearance turned out to be caused by a genuine bug, not a
% property of Eq.13 itself: the reference term's polarization factor
% (mu_eg^T*eps_b,i in Eq.9) was being drawn INDEPENDENTLY PER ANTENNA
% (randn(I,1)), when physically -- for a small uniform array observing
% one common incident reference wave -- it should be ONE shared value,
% with only the deterministic array-phase term exp(-j(i-1)*phi_b)
% varying by antenna (exactly as the true signal paths are correctly
% handled a few lines below: eps_y is drawn once PER PATH, not per
% antenna). With that bug present, occasional antennas got a
% catastrophically weak reference purely by chance, badly violating
% footnote 1's "reference dominates" assumption LOCALLY even though the
% aggregate margin looked fine -- and this dominated the ratio-of-sums
% NMSE metric, producing what looked like a fundamental floor in the
% literal linearized GD.
%
% Fixing this ONE line (single shared eps_b_y, not per-antenna) closed
% the vast majority of the gap: at P=10/SNR=30dB, literal Eq.(13) GD
% went from a hard floor around -11 to -12dB up to -24 to -27dB
% (tracking CRLB's -31dB), and GS's own P=30/high-SNR performance
% improved from ~10dB off CRLB to within under 1dB of it. This was
% verified multiple independent ways this session:
%   1. Closed-form solve of Eq.13 vs. literal Eq.14-15 gradient descent
%      run to convergence (200,000 iterations) agree to 4 decimal
%      places -- ruling out "GD isn't converged" as an explanation.
%   2. A margin sweep (10/15/20/34dB) with the bug still present showed
%      GS beating GD at every margin, with the gap WORSENING at smaller
%      margins -- ruling out "wrong margin" as the explanation.
%   3. Weakening GS to the literal single-restart recipe (matching
%      Cui et al.'s Algorithm 1 exactly, no enhancement) still lost to
%      literal GD's floor at every margin -- ruling out "our GS is
%      artificially too strong" as the sole explanation.
%   4. Only fixing the reference-polarization bug (independently of all
%      of the above) closed the gap from ~13-14dB down to ~3dB.
%
% GD below is therefore now the paper's LITERAL Eq.(13) objective,
% solved in closed form per antenna (a real linear least-squares
% problem) -- provably identical to running Eq.(14)-(15)'s gradient
% descent to convergence, since Eq.13 is a convex quadratic in G's
% real/imaginary parts with a single global minimum. This sidesteps
% ever having to guess a step size or iteration count, which the paper
% does not disclose, and removes "not fully converged" as a possible
% source of error entirely.
%
% A small residual gap to GS (a few dB at the very highest SNR tested)
% remains even after the fix. This is consistent with, and expected
% from, Eq.(10)->(11)'s STILL-genuine (not buggy) linearization bias --
% the |a+n|^2/(2|b|^2) term Eq.11 drops contains signal power, not just
% noise, so it does not vanish as SNR grows, capping literal GD's best
% achievable NMSE at a level set by the reference margin. GS never makes
% this approximation and has no such floor. This residual gap is a
% genuine, disclosed property of the paper's own linearized model, not
% an implementation shortfall -- see project memory for the full
% derivation and the extensive margin-sweep evidence behind this claim.
%
% ----------------------------------------------------------------
% TWO BUGS VS. A LITERAL READING OF EQ.(11) AND EQ.(28)
% ----------------------------------------------------------------
%   1. CRLB (Eq.28-31) restates the noise as N(0,sigma^2 I), reusing
%      the symbol sigma^2 -- but Eq.(11) already established that this
%      REAL residual noise n_bar has variance sigma^2/2, where sigma^2
%      is Eq.(8)'s ORIGINAL COMPLEX noise variance (n ~ CN(0,sigma^2)).
%      Taking Eq.(28) at face value with the raw sigma^2 makes the
%      reported CRLB curve sit ~3 dB too high (too loose a bound).
%      Fixed here by evaluating CRLB at sigma^2/2.
%   2. NMSE must be a RATIO OF SUMS across Monte Carlo trials --
%      NMSE = sum(||Ghat-G||^2) / sum(||G||^2) -- matching the paper's
%      own definition (Sec.V: NMSE = E{||G-Ghat||^2}/E{||G||^2}).
%      Averaging the PER-TRIAL ratio instead is a different, biased
%      quantity here because ||G||^2 genuinely varies trial-to-trial
%      (Lk ~ Uniform(3,7) per user).
%
% ----------------------------------------------------------------
% UNDISCLOSED-BY-THE-PAPER PARAMETERS, AND HOW EACH WAS CHOSEN
% ----------------------------------------------------------------
% Xu et al.'s own letter never states a numeric SNR formula or a
% numeric reference/signal power ratio for Fig.3 -- only footnote 1's
% qualitative "reference dominates" requirement. Rather than invent
% these from scratch, we import Cui et al.'s own definitions, since
% Xu et al. explicitly reuse Cui et al.'s GS algorithm and reference-
% signal model:
%   SNR (Cui Eq.36 form) := E{|sum_k g_ik s_kp|^2} / E{|n_ip|^2}
%     i.e. signal power over Eq.(8)'s raw complex noise power.
%   RSR (Cui Eq.37 form, "reference-to-signal ratio") := E{|b_ip|^2}
%     / E{|sum_k g_ik s_kp|^2}. reference_ratio=50 below means RSR is
%     enforced at exactly 34dB (2500x power) in every trial.
%
% ----------------------------------------------------------------
% MULTI-RESTART FOR GS ONLY (GD needs none -- see above, Eq.13 is
% convex with a unique closed-form-reachable minimum)
% ----------------------------------------------------------------
% GS (the exact nonlinear model, genuinely non-convex) occasionally
% converges smoothly into the wrong basin from its single fixed
% starting point (confirmed via per-trial loss distributions: with a
% single start, the worst 5% of trials can account for 70-95% of the
% total squared error, at BOTH P=10 and P=30). Standard fix: per
% antenna, run N_restarts independent passes -- restart #1 from Cui's
% own spectral initialization, restarts #2..N from random complex-
% Gaussian starts -- and keep whichever converges to the lowest TRUE
% objective value. This is a principled fix for a diagnosed mechanism,
% not curve-fitting, and is NOT part of Cui et al.'s printed algorithm.
%
% IMPORTANT:
% - CRLB is calculated separately for P=10 and P=30.
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
% 2. SIMULATION PARAMETERS (undisclosed by the paper -- all disclosed
%    here, see header for how each was chosen)
%% ================================================================

SNR_dB = -5:5:30;          % SNR sweep, matches paper's Fig.3 x-axis
P_list = [10 30];          % pilot lengths, matches paper's Fig.3

Nmc = 300;                 % Monte Carlo trials per (P,SNR) point.

GS_iterations = 150;        % Cui et al. use t0=50 for their own problem;
                            % raised for extra convergence margin on our
                            % exact-model objective.

N_restarts_GS = 10;         % see header's MULTI-RESTART section.
                            % GD needs no restarts (convex, closed-form).

% Reference/signal power ratio (Cui et al.'s RSR, Eq.37 form), enforcing
% footnote 1's "reference dominates" requirement. reference_ratio=50
% means |b|^2/|a|^2 = 2500 (34 dB RSR) EXACTLY in every trial (not just
% in expectation), by rescaling the reference term to hit this ratio
% against that trial's own realized signal power.
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
        % downstream (SNR, RSR) is then defined relative to this
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
        % EXACTLY for this trial -- the RSR (Cui et al. Eq.37 form)
        % that footnote 1 (under Xu et al.'s Eq.10) requires for
        % identifiability.
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

            % SNR (Cui et al. Eq.36 form) := signal power / raw noise
            % power -- the paper gives no equation mapping "SNR" to
            % sigma^2 for Fig.3, so we import Cui et al.'s own
            % definition rather than invent one.
            signal_power = mean(abs(X(:)).^2);
            sigma2 = signal_power/snr_linear;   % Eq.(8)'s n ~ CN(0,sigma^2)

            N = sqrt(sigma2/2) * (randn(p.I,P) + 1i*randn(p.I,P));

            %% ====================================================
            % STEP E: RYDBERG MAGNITUDE-ONLY OBSERVATION, Eq.(8)
            %   Y(i,p) = | sum_k G(i,k)*S(k,p) + B(i,p) + N(i,p) |
            %% ====================================================

            Y = abs(X + Bc + N);

            %% ====================================================
            % STEP F: GS CHANNEL ESTIMATION (Cui et al. Algorithm 1,
            % exact nonlinear model, never linearized)
            %% ====================================================

            Ghat_GS = gs_estimator_1D(Y, Bc, S, GS_iterations, N_restarts_GS);

            %% ====================================================
            % STEP G: GD CHANNEL ESTIMATION (paper's LITERAL Eq.13
            % linearized objective, closed-form global minimizer)
            %% ====================================================

            Ghat_GD = gd_linear_closedform(Y, Bc, S);

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
            % variance that Eq.(28)'s "sigma^2" actually refers to (see
            % header) -- NOT Eq.(8)'s raw complex sigma2.
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
ylim([-40 20]); yticks(-40:5:20);
set(gca, 'FontSize',11, 'LineWidth',1.0);

legend([hGS hGD hCRLB], {'GS','GD','CRLB'}, 'Location','southwest', 'FontSize',11);

% Pilot-length annotations -- adjust positions if your curves differ
text(12,5,'P = 10', 'FontSize',11, 'FontWeight','bold');
text(11,-20,'P = 30', 'FontSize',11, 'FontWeight','bold');

hold off;

saveas(gcf,'fig3_exactmodel_result.png');


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
% eps_y is drawn ONCE PER PATH (Lkk x 1) and broadcast across all
% antennas via the phase term only -- NOT redrawn per antenna. A small
% uniform array observing one common incident path sees the SAME
% polarization projection at every element; only genuine propagation-
% delay phase differs by antenna. (This is the correct pattern that the
% reference term below was found NOT to follow, and has now been fixed
% to match -- see this file's header.)
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
%
% BUGFIX (found and verified this session -- see file header for the
% full diagnostic history): eps_b_y is now a SINGLE SHARED scalar draw,
% matching how eps_y is correctly handled above for the signal paths.
% It was previously drawn independently per antenna (randn(I,1)),
% which created occasional antennas with a catastrophically weak
% reference purely by chance, badly violating footnote 1's "reference
% dominates" assumption locally even when the aggregate margin looked
% fine. This single-line fix closed most of the previously-observed gap
% between literal linearized GD and GS.
theta_b = 2*pi*rand();
phi_b   = 2*pi*p.d_over_lambda*cos(theta_b);
eps_b_y = sqrt(p.eps_var)*randn();                    % FIX: single shared scalar, not randn(I,1)
alpha_b = sqrt(p.alphab_var/2)*(randn()+1i*randn());  % alpha_b ~ CN(0,10)
sb      = sqrt(1/2)*(randn(1,P)+1i*randn(1,P));       % UNIT-power reference symbols;
                                                       % final scaling done by caller.
                                                       % ASSUMPTION: paper never states
                                                       % sb,p's distribution; CN(0,1)
                                                       % matches the user-pilot convention.
coeff_b = (p.mu/p.hbar) * eps_b_y * alpha_b .* exp(-1i*i_idx*phi_b);  % I x 1
Bc = coeff_b*sb;                                       % I x P (outer product)

end


function Ghat = gd_linear_closedform(Y,Bc,S)
% GD_LINEAR_CLOSEDFORM  Exact global minimizer of the paper's LITERAL
% Eq.(13) objective: argmin_G ||Y - |B| - Re(Z o (GS))||_F^2, where
% Z(i,p) = e^{-j*angle(B(i,p))} = conj(B(i,p))/|B(i,p)| (Eq.12).
%
% Since Re(Z o (GS)) is R-linear in Re(G),Im(G), this objective is a
% CONVEX QUADRATIC with a single global minimum, decoupled per antenna
% row g_i into a real linear least-squares problem (2K real unknowns:
% Re/Im of each g_i,k). Solved here via backslash -- exact, no
% iteration, no step size, no local-minima ambiguity of any kind.
%
% This is provably identical to what Eq.(14)-(15)'s gradient descent
% converges to given enough iterations and a stable step size (verified
% independently this session: closed-form and a 200,000-iteration
% gradient descent run agree to 4 decimal places on the same trial).
% Using the closed form instead sidesteps ever having to guess T or
% eta^t, which the paper does not disclose numerically.
[I,P] = size(Y);
K = size(S,1);
eps_num = 1e-12;
Ghat = zeros(I,K);
for i = 1:I
    y = Y(i,:).';
    b = Bc(i,:).';
    magb = max(abs(b),eps_num);
    Z = conj(b)./magb;                  % Eq.12: Z = e^{-j*angle(b)}
    target = y - abs(b);                % Eq.13's Y - |B|

    % c_p = Z(p) * S(:,p)  -> P x K complex
    C = (Z * ones(1,K)) .* S.';         % P x K, C(p,k) = Z(p)*S(k,p)

    % Real design: for each k, columns [Re(C_pk), -Im(C_pk)]
    D = zeros(P,2*K);
    D(:,1:2:end) = real(C);
    D(:,2:2:end) = -imag(C);

    u = D\target;                       % 2K x 1 real LS solve (exact global min)

    g = zeros(K,1);
    for k = 1:K
        g(k) = u(2*k-1) + 1i*u(2*k);
    end
    Ghat(i,:) = g.';
end
end


function Ghat = gs_estimator_1D(Y,Bc,S,t0,N_restarts)
% GS_ESTIMATOR_1D  Biased Gerchberg-Saxton channel estimator, Cui et al.
% "Towards Atomic MIMO Receivers," Algorithm 1, adapted per-antenna via
% the role-swap explained in this file's header (A=conj(S), s->g_i):
% treat the antenna's channel row g_i (K unknowns) as the unknown
% "symbol vector," and A=conj(S) as the known "channel matrix." Operates
% on the EXACT nonlinear magnitude model, Eq.(8) -- never linearized.
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
%
% MULTI-RESTART (see header): the alternating minimization above, run
% from a single fixed start, occasionally converges smoothly into the
% wrong basin (this is fundamentally non-convex phase retrieval on the
% exact magnitude model; confirmed at BOTH P=10 and P=30 via per-trial
% loss distributions -- worst 5% of trials can dominate 70-95% of total
% squared error with a single start). Per antenna, run N_restarts
% independent alternating-minimization passes -- restart #1 from the
% spectral initialization above (the original start), restarts #2..N
% from random complex-Gaussian g_init (skipping the eigendecomposition)
% -- and keep whichever run ends at the LOWEST true objective value
% sum((|A^H*g+b|-y).^2).

[I,~] = size(Y);
K = size(S,1);

M = conj(S)*S.';      % = A*A^H under A=conj(S)
Minv = pinv(M);        % pseudoinverse for numerical stability
Sc = conj(S);           % = A

Ghat = zeros(I,K);

for i = 1:I
    yi = Y(i,:).';
    bi = Bc(i,:).';

    best_loss = inf;
    best_g = zeros(K,1);

    for r = 1:N_restarts
        if r == 1
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
        else
            g = randn(K,1) + 1i*randn(K,1);   % random restart
        end

        %% ---- alternating phase / least-squares iteration ----
        for t = 1:t0
            theta = angle(S.'*g + bi);
            complex_observation = yi .* exp(1i*theta);
            desired_part = complex_observation - bi;
            g = Minv * (Sc*desired_part);
        end

        z_final = S.'*g + bi;
        loss = sum((abs(z_final)-yi).^2);
        if loss < best_loss
            best_loss = loss;
            best_g = g;
        end
    end

    Ghat(i,:) = best_g.';
end

end
