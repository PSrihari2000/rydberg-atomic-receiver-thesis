%% fig4_literal.m
% Fig. 4 of Xu et al., "Channel Estimation for Rydberg Atomic Receivers":

clear; clc; close all;
rng(1);

%% ---- Parameters ----
MC = 500;                       % Monte Carlo trials

I1 = 8;                         % vapor cells along dimension 1
I2 = 8;                         % vapor cells along dimension 2
N = I1*I2;                      % total number of vapor cells
K = 3;                          % users

P_list = [10 30];               % pilot lengths
numP = length(P_list);

SNR_dB = -5:5:30;
numSNR = length(SNR_dB);

GD_max_iterations = 3000;       % upper limit, GD normally stops earlier
PGD_max_iterations = 3000;      % upper limit, PGD normally stops earlier
tol = 1e-12;                    % stop when ||G_new - G||^2 < tol*||G_new||^2

G0_var = 0.1;                   % G0 ~ CN(0,0.1)
d_over_lambda = 0.5;            % element spacing / wavelength (both dimensions)

%% ---- Physical constants ----
hbar = 6.62607015e-34/(2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];       % 3 x 1, transition dipole moment

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FIG. 4 (paper values): NMSE vs SNR, 2D antenna array\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n', MC);
fprintf('Array              = %d x %d\n', I1, I2);
fprintf('Users K            = %d\n', K);
fprintf('Pilot lengths      = [%s]\n', num2str(P_list));
fprintf('====================================================\n');

%% ---- Result arrays ----
NMSE_GD = zeros(numP, numSNR);
NMSE_PGD = zeros(numP, numSNR);
NMSE_CRLB = zeros(numP, numSNR);
refRatio = zeros(numP, 1);                     % measured E|b|^2 / E|S^T G3|^2 (check only)

%% ---- Main loop over pilot length ----
for pIndex = 1:numP

    P = P_list(pIndex);

    fprintf('\n============================================\n');
    fprintf('Simulating P = %d\n', P);
    fprintf('============================================\n');

    mseGD_acc = zeros(1, numSNR);
    msePGD_acc = zeros(1, numSNR);
    crlb_acc = zeros(1, numSNR);
    channelPower_acc = zeros(1, numSNR);
    signalPower_acc = 0;
    refPower_acc = 0;

    for mc = 1:MC

        %% ---- True channel, Eq.(16) ----
        Lk = zeros(1, K);                                   % number of paths of each user
        G3 = zeros(K, N);                                   % K x I1I2, mode-3 unfolded channel
        for k = 1:K
            Lk(k) = randi([3, 7]);                          % L_k ~ Uniform{3,...,7}
            Gk = zeros(I1, I2);                             % I1 x I2, channel of user k
            for l = 1:Lk(k)
                alpha_lk = (randn + 1j*randn)/sqrt(2);      % alpha_{l,k} ~ CN(0,1)
                theta_lk = pi*rand;                         % elevation ~ U(0,pi)
                phi_lk = 2*pi*rand;                         % azimuth ~ U(0,2*pi)
                u_lk = 2*pi*d_over_lambda*cos(theta_lk);
                v_lk = 2*pi*d_over_lambda*sin(theta_lk)*cos(phi_lk);
                for i1 = 1:I1
                    for i2 = 1:I2
                        epsilon_cell = sqrt(1/3)*randn(3, 1);   % 3 x 1, epsilon_{i1,i2,k,l} ~ N(0,1/3), own draw per cell
                        projection = mu_eg.'*epsilon_cell;      % 1 x 1
                        Gk(i1, i2) = Gk(i1, i2) + (1/hbar)*projection*alpha_lk*exp(-1j*((i1-1)*u_lk + (i2-1)*v_lk));
                    end
                end
            end
            % mode-3 unfolding, Eq.(20): row k of G3, column q = i1 + (i2-1)*I1
            for i2 = 1:I2
                for i1 = 1:I1
                    q = i1 + (i2-1)*I1;
                    G3(k, q) = Gk(i1, i2);
                end
            end
        end

        %% ---- Pilots ----
        S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);         % K x P, s_{k,p} ~ CN(0,1)
        A3 = S.'*G3;                                        % (P x K)(K x I1I2) = P x I1I2

        %% ---- Reference signal, Eq.(17) ----
        alpha_b = sqrt(10/2)*(randn + 1j*randn);            % alpha_b ~ CN(0,10)
        theta_b = pi*rand;                                  % elevation ~ U(0,pi)
        phi_b = 2*pi*rand;                                  % azimuth ~ U(0,2*pi)
        u_b = 2*pi*d_over_lambda*cos(theta_b);
        v_b = 2*pi*d_over_lambda*sin(theta_b)*cos(phi_b);
        s_b = ones(1, P);                                   % 1 x P, s_{b,p} = 1 (not given in paper, assumed)

        B3 = zeros(P, N);                                   % P x I1I2
        for i2 = 1:I2
            for i1 = 1:I1
                epsilon_b = sqrt(1/3)*randn(3, 1);          % 3 x 1, epsilon_{b,i1,i2} ~ N(0,1/3), own draw per cell
                projection_b = mu_eg.'*epsilon_b;           % 1 x 1
                q = i1 + (i2-1)*I1;
                for p = 1:P
                    B3(p, q) = s_b(p)*(1/hbar)*projection_b*alpha_b*exp(-1j*((i1-1)*u_b + (i2-1)*v_b));
                end
            end
        end

        Z3 = exp(-1j*angle(B3));                            % P x I1I2, reference phase

        signalPower = mean(abs(A3(:)).^2);                  % E|S^T G3|^2
        refPower = mean(abs(B3(:)).^2);                     % E|B3|^2
        signalPower_acc = signalPower_acc + signalPower;
        refPower_acc = refPower_acc + refPower;

        %% ---- Loop over SNR ----
        for snrIndex = 1:numSNR

            snrLinear = 10^(SNR_dB(snrIndex)/10);
            sigma2_complex = signalPower/snrLinear;         % complex noise variance

            N3 = sqrt(sigma2_complex/2)*(randn(P, N) + 1j*randn(P, N));   % P x I1I2, CN(0,sigma2)

            Y3 = abs(A3 + B3 + N3);                         % P x I1I2, magnitude observation, Eq.(18)-(20)

            G0 = sqrt(G0_var/2)*(randn(K, N) + 1j*randn(K, N));   % K x I1I2, G0 ~ CN(0,0.1)
            step = 1/max(eig(S*S'));                        % step size
            Y_centered = Y3 - abs(B3);                      % P x I1I2

            %% ---- GD (gradient step only) ----
            G_hat_GD = G0;                                  % K x I1I2
            for iter = 1:GD_max_iterations

                residual = Y_centered - real(Z3.*(S.'*G_hat_GD));    % P x I1I2
                gradient = -conj(S)*(residual.*conj(Z3));   % K x I1I2, Eq.(24)

                G_new = G_hat_GD - step*gradient;           % K x I1I2
                relChange = norm(G_new - G_hat_GD, 'fro')^2/norm(G_new, 'fro')^2;
                G_hat_GD = G_new;

                if relChange < tol
                    break;
                end

            end

            %% ---- PGD, Algorithm 1, Eq.(23)-(27) ----
            G_hat_PGD = G0;                                 % K x I1I2
            for iter = 1:PGD_max_iterations

                residual = Y_centered - real(Z3.*(S.'*G_hat_PGD));   % P x I1I2
                gradient = -conj(S)*(residual.*conj(Z3));   % K x I1I2, Eq.(24)
                M_step = G_hat_PGD - step*gradient;         % K x I1I2, gradient step

                G_new = zeros(K, N);                        % K x I1I2
                for k = 1:K

                    % row k of M_step back to an I1 x I2 matrix
                    Mk = zeros(I1, I2);
                    for i2 = 1:I2
                        for i1 = 1:I1
                            Mk(i1, i2) = M_step(k, i1 + (i2-1)*I1);
                        end
                    end

                    % best rank-L_k approximation, Eq.(26)-(27)
                    [U, Sigma, V] = svd(Mk);
                    L = Lk(k);
                    Mk_rank = U(:, 1:L)*Sigma(1:L, 1:L)*V(:, 1:L)';    % I1 x I2, V' = V^H

                    % back to row k
                    for i2 = 1:I2
                        for i1 = 1:I1
                            G_new(k, i1 + (i2-1)*I1) = Mk_rank(i1, i2);
                        end
                    end

                end

                relChange = norm(G_new - G_hat_PGD, 'fro')^2/norm(G_new, 'fro')^2;
                G_hat_PGD = G_new;

                if relChange < tol
                    break;
                end

            end

            %% ---- CRLB, Eq.(31) ----
            sigma2_real = sigma2_complex/2;
            CRB_g = 4*sigma2_real*kron(eye(N), inv(conj(S)*S.'));    % KI1I2 x KI1I2

            %% ---- Accumulate ----
            mseGD_acc(snrIndex) = mseGD_acc(snrIndex) + norm(G3 - G_hat_GD, 'fro')^2;
            msePGD_acc(snrIndex) = msePGD_acc(snrIndex) + norm(G3 - G_hat_PGD, 'fro')^2;
            crlb_acc(snrIndex) = crlb_acc(snrIndex) + real(trace(CRB_g));
            channelPower_acc(snrIndex) = channelPower_acc(snrIndex) + norm(G3, 'fro')^2;

        end

        if mod(mc, 50) == 0 || mc == MC
            fprintf('P = %d : Monte Carlo %d / %d\n', P, mc, MC);
        end

    end

    NMSE_GD(pIndex, :) = mseGD_acc./channelPower_acc;
    NMSE_PGD(pIndex, :) = msePGD_acc./channelPower_acc;
    NMSE_CRLB(pIndex, :) = crlb_acc./channelPower_acc;
    refRatio(pIndex) = refPower_acc/signalPower_acc;

    fprintf('\nResults for P = %d (measured E|b|^2/E|S^T G3|^2 = %.1f dB)\n', P, 10*log10(refRatio(pIndex)));
    fprintf('------------------------------------------------------------------------\n');
    for snrIndex = 1:numSNR
        fprintf('P = %2d | SNR = %3d dB | GD = %7.2f dB | PGD = %7.2f dB | CRLB = %7.2f dB\n', ...
            P, SNR_dB(snrIndex), 10*log10(NMSE_GD(pIndex, snrIndex)), 10*log10(NMSE_PGD(pIndex, snrIndex)), ...
            10*log10(NMSE_CRLB(pIndex, snrIndex)));
    end

end

%% ---- Plot ----
NMSE_GD_dB = 10*log10(NMSE_GD);
NMSE_PGD_dB = 10*log10(NMSE_PGD);
NMSE_CRLB_dB = 10*log10(NMSE_CRLB);

colorGD = [0.93 0.69 0.13];
colorPGD = [0 0.45 0.74];
colorCRLB = [0.85 0.33 0.10];
lw = 1.5;                                           % line width, as in the paper
ms = 6;                                             % marker size

figure('Color', 'w', 'Position', [100 100 640 480]);
hold on;
grid on;
box on;

% P = 10 (these three lines give the legend)
plot(SNR_dB, NMSE_GD_dB(1, :), '-.s', 'Color', colorGD, 'LineWidth', lw, 'MarkerSize', ms, 'MarkerFaceColor', 'w', 'DisplayName', 'GD');
plot(SNR_dB, NMSE_PGD_dB(1, :), '-^', 'Color', colorPGD, 'LineWidth', lw, 'MarkerSize', ms, 'MarkerFaceColor', colorPGD, 'DisplayName', 'PGD');
plot(SNR_dB, NMSE_CRLB_dB(1, :), '-o', 'Color', colorCRLB, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'CRLB');

% P = 30
plot(SNR_dB, NMSE_GD_dB(2, :), '-.s', 'Color', colorGD, 'LineWidth', lw, 'MarkerSize', ms, 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
plot(SNR_dB, NMSE_PGD_dB(2, :), '-^', 'Color', colorPGD, 'LineWidth', lw, 'MarkerSize', ms, 'MarkerFaceColor', colorPGD, 'HandleVisibility', 'off');
plot(SNR_dB, NMSE_CRLB_dB(2, :), '-o', 'Color', colorCRLB, 'LineWidth', lw, 'MarkerSize', ms, 'HandleVisibility', 'off');

% dashed ellipses marking P = 10 (at SNR = 10 dB) and P = 30 (at SNR = 15 dB)
index10 = find(SNR_dB == 10);
index15 = find(SNR_dB == 15);
drawEllipse(10, [NMSE_GD_dB(1, index10), NMSE_PGD_dB(1, index10), NMSE_CRLB_dB(1, index10)], 'P = 10', 'above');
drawEllipse(15, [NMSE_GD_dB(2, index15), NMSE_PGD_dB(2, index15), NMSE_CRLB_dB(2, index15)], 'P = 30', 'below');

xlabel('SNR [dB]', 'FontWeight', 'bold');
ylabel('NMSE [dB]', 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontWeight', 'bold');
set(gca, 'FontWeight', 'bold', 'FontSize', 11, 'LineWidth', 1);
xlim([SNR_dB(1) SNR_dB(end)]);
xticks(SNR_dB);
ylim([-40 15]);
yticks(-40:5:15);

hold off;

exportgraphics(gcf, 'fig4_literal.png', 'Resolution', 200);

%% ---- Save results ----
save('fig4_literal.mat', 'SNR_dB', 'P_list', 'NMSE_GD', 'NMSE_PGD', 'NMSE_CRLB', ...
     'refRatio', 'MC', 'I1', 'I2', 'K', 'GD_max_iterations', 'PGD_max_iterations', 'tol');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig4_literal.mat\n');
fprintf('Plot saved to fig4_literal.png\n');
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
