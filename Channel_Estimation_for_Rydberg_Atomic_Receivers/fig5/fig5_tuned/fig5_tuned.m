%% fig5_tuned.m
% Fig. 5 of Xu et al., "Channel Estimation for Rydberg Atomic Receivers":
% same as fig5_literal.m except the changes marked TUNED; plot in the paper's style.

clear; clc; close all;
rng(1);

%% ---- Parameters ----
MC = 500;                       % Monte Carlo trials

I1 = 8;                         % vapor cells along dimension 1
I2 = 8;                         % vapor cells along dimension 2
N = I1*I2;                      % total number of vapor cells
K = 3;                          % users

P_list = 5:5:50;                % pilot lengths
numP = length(P_list);

SNR_dB = 5;

RSR_dB = 50;                    % TUNED: reference-to-signal ratio E|b|^2 / E|S^T G3|^2

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
fprintf(' FIG. 5 (tuned): NMSE vs pilot length, 2D antenna array\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n', MC);
fprintf('Array              = %d x %d\n', I1, I2);
fprintf('Users K            = %d\n', K);
fprintf('SNR                = %d dB\n', SNR_dB);
fprintf('RSR                = %d dB\n', RSR_dB);
fprintf('====================================================\n');

%% ---- Result arrays ----
NMSE_PGD = zeros(1, numP);
NMSE_CRLB = zeros(1, numP);

%% ---- Main loop over pilot length ----
for pIndex = 1:numP

    P = P_list(pIndex);

    msePGD_acc = 0;
    crlb_acc = 0;
    channelPower_acc = 0;

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
                epsilon_kl = sqrt(1/3)*randn(3, 1);         % TUNED: 3 x 1, epsilon_{k,l} ~ N(0,1/3), one draw per path
                projection = mu_eg.'*epsilon_kl;            % 1 x 1
                for i1 = 1:I1
                    for i2 = 1:I2
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

        epsilon_b = sqrt(1/3)*randn(3, 1);                  % TUNED: 3 x 1, epsilon_b ~ N(0,1/3), one draw for all cells
        projection_b = mu_eg.'*epsilon_b;                   % 1 x 1

        B3 = zeros(P, N);                                   % P x I1I2
        for i2 = 1:I2
            for i1 = 1:I1
                q = i1 + (i2-1)*I1;
                for p = 1:P
                    B3(p, q) = s_b(p)*(1/hbar)*projection_b*alpha_b*exp(-1j*((i1-1)*u_b + (i2-1)*v_b));
                end
            end
        end

        signalPower = mean(abs(A3(:)).^2);                  % E|S^T G3|^2

        % TUNED: scale the reference so that E|b|^2 / E|S^T G3|^2 = RSR
        B3 = B3*sqrt(10^(RSR_dB/10)*signalPower/mean(abs(B3(:)).^2));

        Z3 = exp(-1j*angle(B3));                            % P x I1I2, reference phase

        %% ---- Noise and measurement ----
        snrLinear = 10^(SNR_dB/10);
        sigma2_complex = signalPower/snrLinear;             % complex noise variance

        N3 = sqrt(sigma2_complex/2)*(randn(P, N) + 1j*randn(P, N));   % P x I1I2, CN(0,sigma2)

        Y3 = abs(A3 + B3 + N3);                             % P x I1I2, magnitude observation, Eq.(18)-(20)

        %% ---- PGD, Algorithm 1, Eq.(23)-(27) ----
        G_hat_PGD = sqrt(G0_var/2)*(randn(K, N) + 1j*randn(K, N));   % K x I1I2, G0 ~ CN(0,0.1)
        step = 1/max(eig(S*S'));                            % step size
        Y_centered = Y3 - abs(B3);                          % P x I1I2

        for iter = 1:PGD_max_iterations

            residual = Y_centered - real(Z3.*(S.'*G_hat_PGD));   % P x I1I2
            gradient = -conj(S)*(residual.*conj(Z3));       % K x I1I2, Eq.(24)
            M_step = G_hat_PGD - step*gradient;             % K x I1I2, gradient step

            G_new = zeros(K, N);                            % K x I1I2
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
        msePGD_acc = msePGD_acc + norm(G3 - G_hat_PGD, 'fro')^2;
        crlb_acc = crlb_acc + real(trace(CRB_g));
        channelPower_acc = channelPower_acc + norm(G3, 'fro')^2;

    end

    NMSE_PGD(pIndex) = msePGD_acc/channelPower_acc;
    NMSE_CRLB(pIndex) = crlb_acc/channelPower_acc;

    fprintf('P = %2d | PGD = %7.2f dB | CRLB = %7.2f dB\n', ...
        P, 10*log10(NMSE_PGD(pIndex)), 10*log10(NMSE_CRLB(pIndex)));

end

%% ---- Plot ----
NMSE_PGD_dB = 10*log10(NMSE_PGD);
NMSE_CRLB_dB = 10*log10(NMSE_CRLB);

colorPGD = [0 0.45 0.74];
colorCRLB = [0.85 0.33 0.10];

lw = 2;                                             % line width, as in the paper
ms = 7;                                             % marker size

figure('Color', 'w', 'Position', [100 100 640 480]);
hold on;
grid on;
box on;

plot(P_list, NMSE_PGD_dB, '-^', 'Color', colorPGD, 'LineWidth', lw, 'MarkerSize', ms, 'MarkerFaceColor', colorPGD, 'DisplayName', 'PGD');
plot(P_list, NMSE_CRLB_dB, '-o', 'Color', colorCRLB, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'CRLB');

xlabel('Pilot Length', 'FontWeight', 'bold');
ylabel('NMSE [dB]', 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontWeight', 'bold');
set(gca, 'FontWeight', 'bold', 'FontSize', 11, 'LineWidth', 1);
xlim([P_list(1) P_list(end)]);
xticks(P_list);
ylim([-15 30]);
yticks(-15:5:30);

hold off;

exportgraphics(gcf, 'fig5_tuned.png', 'Resolution', 200);

%% ---- Save results ----
save('fig5_tuned.mat', 'P_list', 'SNR_dB', 'NMSE_PGD', 'NMSE_CRLB', ...
     'MC', 'I1', 'I2', 'K', 'RSR_dB', 'PGD_max_iterations', 'tol');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig5_tuned.mat\n');
fprintf('Plot saved to fig5_tuned.png\n');
fprintf('====================================================\n');
