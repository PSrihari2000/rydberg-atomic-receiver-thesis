%% fig3_sb_check.m
% Test: does the choice of the reference pilot s_{b,p} change Fig. 3?
%   case 1: s_{b,p} = 1 (used in the Fig. 3 scripts)
%   case 2: s_{b,p} ~ CN(0,1) (same distribution as the user pilots)
% Both cases use the same channel, pilots, noise and G0 in every trial, so the
% only difference is s_b. Done for the paper-literal and the tuned settings.

clear; clc; close all;
rng(1);

%% ---- Parameters ----
MC = 500;
I = 8;
K = 3;
P_list = [10 30];
numP = length(P_list);
SNR_dB = -5:5:30;
numSNR = length(SNR_dB);
d_over_lambda = 0.5;
G0_var = 0.1;
GD_tol = 1e-12;

% settings of the two Fig. 3 scripts
modelName = {'literal', 'tuned'};
perCell = [true, false];            % polarization per cell (literal) or per path (tuned)
RSR_dB = [NaN, 40];                 % NaN = no rescaling of the reference
GD_iterations = [3000, 50];         % literal: to convergence, tuned: 50
GS_iterations = [50 50; 15 50];     % rows: model, columns: P = 10, 30

%% ---- Physical constants ----
hbar = 6.62607015e-34/(2*pi);
mu_eg = [0; 1785.9*1.602e-19*5.292e-11; 0];     % 3 x 1

sbStream = RandStream('mt19937ar', 'Seed', 3);  % separate stream for the random s_b

%% ---- Result arrays: (model, P, SNR, case) ----
NMSE_GS = zeros(2, numP, numSNR, 2);
NMSE_GD = zeros(2, numP, numSNR, 2);
NMSE_CRLB = zeros(2, numP, numSNR);

for m = 1:2
    for pIndex = 1:numP

        P = P_list(pIndex);
        fprintf('%s, P = %d\n', modelName{m}, P);

        mseGS = zeros(numSNR, 2);
        mseGD = zeros(numSNR, 2);
        crlb = zeros(numSNR, 1);
        channelPower = 0;

        for mc = 1:MC

            %% ---- True channel ----
            G = zeros(I, K);                                    % I x K
            for k = 1:K
                Lk = randi([3, 7]);
                for l = 1:Lk
                    alpha_lk = (randn + 1j*randn)/sqrt(2);      % CN(0,1)
                    phi_lk = 2*pi*d_over_lambda*cos(2*pi*rand); % AoA ~ U(0,2*pi)
                    epsilon_kl = sqrt(1/3)*randn(3, 1);         % per path
                    for i = 1:I
                        if perCell(m)
                            epsilon_kl = sqrt(1/3)*randn(3, 1); % per cell
                        end
                        G(i, k) = G(i, k) + (1/hbar)*(mu_eg.'*epsilon_kl)*alpha_lk*exp(-1j*(i-1)*phi_lk);
                    end
                end
            end
            channelPower = channelPower + norm(G, 'fro')^2;

            S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);         % K x P, CN(0,1)
            A_signal = G*S;                                     % I x P
            signalPower = mean(abs(A_signal(:)).^2);

            %% ---- Reference channel (without s_b) ----
            alpha_b = sqrt(10/2)*(randn + 1j*randn);            % CN(0,10)
            phi_b = 2*pi*d_over_lambda*cos(2*pi*rand);
            epsilon_b = sqrt(1/3)*randn(3, 1);
            g_b = zeros(I, 1);                                  % I x 1
            for i = 1:I
                if perCell(m)
                    epsilon_b = sqrt(1/3)*randn(3, 1);
                end
                g_b(i) = (1/hbar)*(mu_eg.'*epsilon_b)*alpha_b*exp(-1j*(i-1)*phi_b);
            end

            %% ---- The two reference pilots ----
            s_b_case = cell(1, 2);
            s_b_case{1} = ones(1, P);                                               % s_b = 1
            s_b_case{2} = (randn(sbStream, 1, P) + 1j*randn(sbStream, 1, P))/sqrt(2); % s_b ~ CN(0,1)

            B_case = cell(1, 2);
            for c = 1:2
                B = g_b*s_b_case{c};                            % I x P
                if ~isnan(RSR_dB(m))
                    B = B*sqrt(10^(RSR_dB(m)/10)*signalPower/mean(abs(B(:)).^2));   % RSR on average
                end
                B_case{c} = B;
            end

            for snrIndex = 1:numSNR

                sigma2 = signalPower/10^(SNR_dB(snrIndex)/10);
                N = sqrt(sigma2/2)*(randn(I, P) + 1j*randn(I, P));   % same noise for both cases
                G0 = sqrt(G0_var/2)*(randn(I, K) + 1j*randn(I, K));  % same G0 for both cases

                for c = 1:2
                    B = B_case{c};
                    Y = abs(A_signal + B + N);                  % I x P
                    G_GS = run_gs(Y, S, B, GS_iterations(m, pIndex));
                    G_GD = run_gd(Y, S, B, G0, GD_iterations(m), GD_tol);
                    mseGS(snrIndex, c) = mseGS(snrIndex, c) + norm(G - G_GS, 'fro')^2;
                    mseGD(snrIndex, c) = mseGD(snrIndex, c) + norm(G - G_GD, 'fro')^2;
                end

                crlb(snrIndex) = crlb(snrIndex) + 2*sigma2*I*real(trace(inv(conj(S)*S.')));
            end
        end

        NMSE_GS(m, pIndex, :, :) = 10*log10(mseGS/channelPower);
        NMSE_GD(m, pIndex, :, :) = 10*log10(mseGD/channelPower);
        NMSE_CRLB(m, pIndex, :) = 10*log10(crlb/channelPower);
    end
end

%% ---- Table ----
for m = 1:2
    for pIndex = 1:numP
        fprintf('\n%s, P = %d (NMSE in dB)\n', modelName{m}, P_list(pIndex));
        fprintf('  SNR |  GS: s_b=1  s_b~CN  diff |  GD: s_b=1  s_b~CN  diff | CRLB\n');
        for snrIndex = 1:numSNR
            gs1 = NMSE_GS(m, pIndex, snrIndex, 1);  gs2 = NMSE_GS(m, pIndex, snrIndex, 2);
            gd1 = NMSE_GD(m, pIndex, snrIndex, 1);  gd2 = NMSE_GD(m, pIndex, snrIndex, 2);
            fprintf('  %3d | %9.2f %7.2f %5.2f | %9.2f %7.2f %5.2f | %6.2f\n', SNR_dB(snrIndex), ...
                gs1, gs2, gs2 - gs1, gd1, gd2, gd2 - gd1, NMSE_CRLB(m, pIndex, snrIndex));
        end
    end
end

%% ---- Plot: solid = s_b = 1, dashed = s_b ~ CN(0,1) ----
figure('Color', 'w', 'Position', [100 100 1200 480]);
for m = 1:2
    subplot(1, 2, m);
    hold on; grid on; box on;
    for pIndex = 1:numP
        plot(SNR_dB, squeeze(NMSE_GS(m, pIndex, :, 1)), '-d', 'Color', 'k', 'LineWidth', 1.3);
        plot(SNR_dB, squeeze(NMSE_GS(m, pIndex, :, 2)), '--d', 'Color', 'k', 'LineWidth', 1.3, 'MarkerFaceColor', 'k');
        plot(SNR_dB, squeeze(NMSE_GD(m, pIndex, :, 1)), '-s', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.3);
        plot(SNR_dB, squeeze(NMSE_GD(m, pIndex, :, 2)), '--s', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.3, ...
             'MarkerFaceColor', [0.93 0.69 0.13]);
        plot(SNR_dB, squeeze(NMSE_CRLB(m, pIndex, :)), '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.3);
    end
    xlabel('SNR [dB]'); ylabel('NMSE [dB]');
    legend('GS, s_b = 1', 'GS, s_b ~ CN(0,1)', 'GD, s_b = 1', 'GD, s_b ~ CN(0,1)', 'CRLB', 'Location', 'southwest');
    title(sprintf('Fig. 3 %s settings, P = 10 and 30, %d trials', modelName{m}, MC));
    xlim([SNR_dB(1) SNR_dB(end)]); xticks(SNR_dB);
    hold off;
end

exportgraphics(gcf, 'fig3_sb_check.png', 'Resolution', 200);
save('fig3_sb_check.mat', 'SNR_dB', 'P_list', 'NMSE_GS', 'NMSE_GD', 'NMSE_CRLB', 'MC');
fprintf('\nsaved fig3_sb_check.png and .mat\n');

%% ---- Local functions ----
function G = run_gs(Y, S, B, T)
% biased GS of [10]: spectral initialisation + T iterations, cell by cell
[I, ~] = size(Y);
K = size(S, 1);
G = zeros(I, K);
for i = 1:I
    z_i = Y(i, :).';                                % P x 1
    b_i = B(i, :).';                                % P x 1
    Abar = [S.', b_i];                              % P x (K+1)
    M = Abar'*diag(z_i)*Abar;
    M = (M + M')/2;
    [V, D] = eig(M);
    [~, maxIndex] = max(real(diag(D)));
    v = V(:, maxIndex);
    r_bar = (abs(Abar*v).'*z_i)/(norm(Abar*v)^2);
    gbar0 = r_bar*v;
    x = exp(-1j*angle(gbar0(K+1)))*gbar0(1:K);     % K x 1
    for t = 1:T
        theta = angle(S.'*x + b_i);
        x = (conj(S)*S.')\(conj(S)*(z_i.*exp(1j*theta) - b_i));
    end
    G(i, :) = x.';
end
end

function G = run_gd(Y, S, B, G0, T, tol)
% GD on the linearised model, step 1/lambda_max(S*S^H)
G = G0;
Z = exp(-1j*angle(B));
step = 1/max(eig(S*S'));
Y_centered = Y - abs(B);
for t = 1:T
    residual = Y_centered - real(Z.*(G*S));
    G_new = G + step*(residual.*conj(Z))*S';
    relChange = norm(G_new - G, 'fro')^2/norm(G_new, 'fro')^2;
    G = G_new;
    if relChange < tol
        break;
    end
end
end
