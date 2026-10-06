%% fig3_paper_literal.m
% Fig. 3 of Xu et al., "Channel Estimation for Rydberg Atomic Receivers":

clear; clc; close all;
rng(1);

%% ---- Parameters ----
MC = 500;                       % Monte Carlo trials

I = 8;                          % vapor cells
K = 3;                          % users

P_list = [10 30];               % pilot lengths
numP = length(P_list);

SNR_dB = -5:5:30;
numSNR = length(SNR_dB);

GS_iterations = 50;             % t0 = 50 of [10] (Cui et al.)
GD_max_iterations = 3000;       % upper limit, GD normally stops earlier
GD_tol = 1e-12;                 % stop when ||G_new - G||^2 < GD_tol*||G_new||^2

G0_var = 0.1;                   % G0 ~ CN(0,0.1)
d_over_lambda = 0.5;            % element spacing / wavelength

%% ---- Physical constants ----
hbar = 6.62607015e-34/(2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];       % 3 x 1, transition dipole moment

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FIG. 3 (paper values): NMSE vs SNR, 1D antenna array\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n', MC);
fprintf('Vapor cells I      = %d\n', I);
fprintf('Users K            = %d\n', K);
fprintf('Pilot lengths      = [%s]\n', num2str(P_list));
fprintf('====================================================\n');

%% ---- Result arrays ----
NMSE_GS = zeros(numP, numSNR);
NMSE_GD = zeros(numP, numSNR);
NMSE_CRLB = zeros(numP, numSNR);
refRatio = zeros(numP, 1);                     % measured E|b|^2 / E|GS|^2 (check only)

%% ---- Main loop over pilot length ----
for pIndex = 1:numP

    P = P_list(pIndex);

    fprintf('\n============================================\n');
    fprintf('Simulating P = %d\n', P);
    fprintf('============================================\n');

    mseGS_acc = zeros(1, numSNR);
    mseGD_acc = zeros(1, numSNR);
    crlb_acc = zeros(1, numSNR);
    channelPower_acc = zeros(1, numSNR);
    signalPower_acc = 0;
    refPower_acc = 0;

    for mc = 1:MC

        %% ---- True channel, Eq.(7) ----
        G = zeros(I, K);                                    % I x K
        for k = 1:K
            Lk = randi([3, 7]);                             % L_k ~ Uniform{3,...,7}
            for l = 1:Lk
                alpha_lk = (randn + 1j*randn)/sqrt(2);      % alpha_{l,k} ~ CN(0,1)
                theta_lk = 2*pi*rand;                       % AoA ~ U(0,2*pi)
                phi_lk = 2*pi*d_over_lambda*cos(theta_lk);  % phase shift
                for i = 1:I
                    epsilon_ikl = sqrt(1/3)*randn(3, 1);    % 3 x 1, epsilon_{i,k,l} ~ N(0,1/3), own draw per cell
                    projection = mu_eg.'*epsilon_ikl;       % (1 x 3)(3 x 1) = 1 x 1
                    G(i, k) = G(i, k) + (1/hbar)*projection*alpha_lk*exp(-1j*(i-1)*phi_lk);
                end
            end
        end

        %% ---- Pilots ----
        S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);         % K x P, s_{k,p} ~ CN(0,1)
        A_signal = G*S;                                     % (I x K)(K x P) = I x P

        %% ---- Reference signal, Eq.(9) ----
        alpha_b = sqrt(10/2)*(randn + 1j*randn);            % alpha_b ~ CN(0,10)
        theta_b = 2*pi*rand;                                % AoA ~ U(0,2*pi)
        phi_b = 2*pi*d_over_lambda*cos(theta_b);
        s_b = ones(1, P);                                   % 1 x P, s_{b,p} = 1 (not given in paper, assumed)

        B = zeros(I, P);                                    % I x P
        for i = 1:I
            epsilon_bi = sqrt(1/3)*randn(3, 1);             % 3 x 1, epsilon_{b,i} ~ N(0,1/3), own draw per cell
            projection_b = mu_eg.'*epsilon_bi;              % 1 x 1
            for p = 1:P
                B(i, p) = s_b(p)*(1/hbar)*projection_b*alpha_b*exp(-1j*(i-1)*phi_b);
            end
        end

        Z = exp(-1j*angle(B));                              % I x P, reference phase

        signalPower = mean(abs(A_signal(:)).^2);            % E|GS|^2
        refPower = mean(abs(B(:)).^2);                      % E|B|^2
        signalPower_acc = signalPower_acc + signalPower;
        refPower_acc = refPower_acc + refPower;

        %% ---- Loop over SNR ----
        for snrIndex = 1:numSNR

            snrLinear = 10^(SNR_dB(snrIndex)/10);
            sigma2_complex = signalPower/snrLinear;         % complex noise variance

            N_complex = sqrt(sigma2_complex/2)*(randn(I, P) + 1j*randn(I, P));   % I x P, CN(0,sigma2)

            Y = abs(A_signal + B + N_complex);              % I x P, magnitude observation, Eq.(8)

            %% ---- Biased GS (one cell at a time) ----
            G_hat_GS = zeros(I, K);                         % I x K

            for i = 1:I

                y_i = Y(i, :).';                            % P x 1, row i of Y = the P readings of cell i
                b_i = B(i, :).';                            % P x 1, reference at cell i

                Abar = [S.', b_i];                          % P x (K+1)

                M = zeros(K+1, K+1);                        % (K+1) x (K+1)
                for p = 1:P
                    a_bar_p = Abar(p, :)';                  % (K+1) x 1, p-th row as a column
                    M = M + y_i(p)*(a_bar_p*a_bar_p');      % y_p * a_p * a_p^H
                end

                [V, D] = eig(M);
                [~, maxIndex] = max(real(diag(D)));
                v = V(:, maxIndex);                         % (K+1) x 1, principal eigenvector

                r_bar = (abs(Abar*v).'*y_i)/(norm(Abar*v)^2);    % scale factor
                gbar0 = r_bar*v;                            % (K+1) x 1

                x = exp(-1j*angle(gbar0(K+1)))*gbar0(1:K);  % K x 1, initial estimate of g_i

                for gsIter = 1:GS_iterations
                    theta = angle(S.'*x + b_i);             % P x 1, estimated phase
                    x = inv(conj(S)*S.')*(conj(S)*(y_i.*exp(1j*theta) - b_i));   % K x 1
                end

                G_hat_GS(i, :) = x.';                       % 1 x K

            end

            %% ---- GD, Eq.(13)-(15) ----
            G_hat_GD = sqrt(G0_var/2)*(randn(I, K) + 1j*randn(I, K));   % I x K, G0 ~ CN(0,0.1)
            step = 1/max(eig(S*S'));                        % step size
            Y_centered = Y - abs(B);                        % I x P

            for gdIter = 1:GD_max_iterations

                residual = Y_centered - real(Z.*(G_hat_GD*S));   % I x P
                gradient = -(residual.*conj(Z))*S';         % I x K

                G_new = G_hat_GD - step*gradient;           % I x K
                relChange = norm(G_new - G_hat_GD, 'fro')^2/norm(G_new, 'fro')^2;
                G_hat_GD = G_new;

                if relChange < GD_tol
                    break;
                end

            end

            %% ---- CRLB, Eq.(31) ----
            sigma2_real = sigma2_complex/2;
            CRB_g = 4*sigma2_real*kron(eye(I), inv(conj(S)*S.'));    % IK x IK

            %% ---- Accumulate ----
            mseGS_acc(snrIndex) = mseGS_acc(snrIndex) + norm(G - G_hat_GS, 'fro')^2;
            mseGD_acc(snrIndex) = mseGD_acc(snrIndex) + norm(G - G_hat_GD, 'fro')^2;
            crlb_acc(snrIndex) = crlb_acc(snrIndex) + real(trace(CRB_g));
            channelPower_acc(snrIndex) = channelPower_acc(snrIndex) + norm(G, 'fro')^2;

        end

        if mod(mc, 50) == 0 || mc == MC
            fprintf('P = %d : Monte Carlo %d / %d\n', P, mc, MC);
        end

    end

    NMSE_GS(pIndex, :) = mseGS_acc./channelPower_acc;
    NMSE_GD(pIndex, :) = mseGD_acc./channelPower_acc;
    NMSE_CRLB(pIndex, :) = crlb_acc./channelPower_acc;
    refRatio(pIndex) = refPower_acc/signalPower_acc;

    fprintf('\nResults for P = %d (measured E|b|^2/E|GS|^2 = %.1f dB)\n', P, 10*log10(refRatio(pIndex)));
    fprintf('------------------------------------------------------------------------\n');
    for snrIndex = 1:numSNR
        fprintf('P = %2d | SNR = %3d dB | GS = %7.2f dB | GD = %7.2f dB | CRLB = %7.2f dB\n', ...
            P, SNR_dB(snrIndex), 10*log10(NMSE_GS(pIndex, snrIndex)), 10*log10(NMSE_GD(pIndex, snrIndex)), ...
            10*log10(NMSE_CRLB(pIndex, snrIndex)));
    end

end

%% ---- Plot ----
NMSE_GS_dB = 10*log10(NMSE_GS);
NMSE_GD_dB = 10*log10(NMSE_GD);
NMSE_CRLB_dB = 10*log10(NMSE_CRLB);

colorGD = [0.93 0.69 0.13];
colorCRLB = [0.85 0.33 0.10];

figure('Color', 'w');
hold on;
grid on;
box on;

% P = 10 (these three lines give the legend)
plot(SNR_dB, NMSE_GS_dB(1, :), '--d', 'Color', 'k', 'LineWidth', 1.5, 'MarkerFaceColor', 'w', 'DisplayName', 'GS');
plot(SNR_dB, NMSE_GD_dB(1, :), '-.s', 'Color', colorGD, 'LineWidth', 1.5, 'MarkerFaceColor', 'w', 'DisplayName', 'GD');
plot(SNR_dB, NMSE_CRLB_dB(1, :), '-o', 'Color', colorCRLB, 'LineWidth', 1.5, 'DisplayName', 'CRLB');

% P = 30
plot(SNR_dB, NMSE_GS_dB(2, :), '--d', 'Color', 'k', 'LineWidth', 1.5, 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
plot(SNR_dB, NMSE_GD_dB(2, :), '-.s', 'Color', colorGD, 'LineWidth', 1.5, 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
plot(SNR_dB, NMSE_CRLB_dB(2, :), '-o', 'Color', colorCRLB, 'LineWidth', 1.5, 'HandleVisibility', 'off');

% dashed ellipses marking P = 10 (at SNR = 10 dB) and P = 30 (at SNR = 15 dB)
index10 = find(SNR_dB == 10);
index15 = find(SNR_dB == 15);
drawEllipse(10, [NMSE_GS_dB(1, index10), NMSE_GD_dB(1, index10), NMSE_CRLB_dB(1, index10)], 'P = 10', 'above');
drawEllipse(15, [NMSE_GS_dB(2, index15), NMSE_GD_dB(2, index15), NMSE_CRLB_dB(2, index15)], 'P = 30', 'below');

xlabel('SNR [dB]');
ylabel('NMSE [dB]');
legend('Location', 'southwest');
title({sprintf('Paper values: 1D array, I = %d, K = %d', I, K), ...
       sprintf('polarization per cell, \\alpha_b ~ CN(0,10), G_0 ~ CN(0,0.1), %d trials', MC)});
xlim([SNR_dB(1) SNR_dB(end)]);
xticks(SNR_dB);

hold off;

exportgraphics(gcf, 'fig3_paper_literal.png', 'Resolution', 200);

%% ---- Save results ----
save('fig3_paper_literal.mat', 'SNR_dB', 'P_list', 'NMSE_GS', 'NMSE_GD', 'NMSE_CRLB', ...
     'refRatio', 'MC', 'I', 'K', 'GS_iterations', 'GD_max_iterations', 'GD_tol');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig3_paper_literal.mat\n');
fprintf('Plot saved to fig3_paper_literal.png\n');
fprintf('====================================================\n');

%% ---- Local function: dashed ellipse with a label ----
function drawEllipse(x0, yValues, label, side)
% x0: SNR where the ellipse is drawn; yValues: NMSE values [dB] of the curves it encloses

halfWidth = 0.9;                                    % [dB on the SNR axis]
margin = 1.5;                                       % [dB on the NMSE axis]

yCenter = (min(yValues) + max(yValues))/2;
halfHeight = (max(yValues) - min(yValues))/2 + margin;

t = linspace(0, 2*pi, 200);
plot(x0 + halfWidth*cos(t), yCenter + halfHeight*sin(t), 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');

if strcmp(side, 'above')
    text(x0, yCenter + halfHeight + 1.5, label, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
else
    text(x0, yCenter - halfHeight - 2, label, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

end
