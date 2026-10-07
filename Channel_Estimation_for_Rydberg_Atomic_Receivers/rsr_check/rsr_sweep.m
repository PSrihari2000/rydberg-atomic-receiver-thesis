%% rsr_sweep.m
% How strong must the reference be? NMSE of GD (linearised model, Eq.(12)) and
% GS (exact magnitude model) vs the reference-to-signal ratio RSR = E|b|^2/E|GS|^2.
% Fig. 3 model with polarization per path; GD and GS run to convergence.
% Same channel, pilots, noise and G0 for every RSR (only the reference strength changes).

clear; clc; close all;
rng(1);

%% ---- Parameters ----
MC = 200;                       % Monte Carlo trials

I = 8;                          % vapor cells
K = 3;                          % users

P_list = [10 30];
numP = length(P_list);

SNR_list = [10 20 30];          % dB
numSNR = length(SNR_list);

RSR_list = [-2 0 5 10 15 20 25 30 35 40 50];   % dB (-2 dB is about the paper's value)
numRSR = length(RSR_list);

max_iterations = 3000;          % GD and GS, upper limit
tol = 1e-12;                    % stop when ||G_new - G||^2 < tol*||G_new||^2

G0_var = 0.1;
d_over_lambda = 0.5;

hbar = 6.62607015e-34/(2*pi);
mu_eg = [0; 1785.9*1.602e-19*5.292e-11; 0];       % 3 x 1

%% ---- Result arrays: (P, SNR, RSR) ----
NMSE_GD = zeros(numP, numSNR, numRSR);
NMSE_GS = zeros(numP, numSNR, numRSR);
NMSE_CRLB = zeros(numP, numSNR);

for pIndex = 1:numP

    P = P_list(pIndex);
    fprintf('P = %d\n', P);

    mseGD = zeros(numSNR, numRSR);
    mseGS = zeros(numSNR, numRSR);
    crlb = zeros(numSNR, 1);
    channelPower = 0;

    for mc = 1:MC

        %% ---- True channel (polarization per path) ----
        G = zeros(I, K);                                    % I x K
        for k = 1:K
            Lk = randi([3, 7]);
            for l = 1:Lk
                alpha_lk = (randn + 1j*randn)/sqrt(2);      % CN(0,1)
                phi_lk = 2*pi*d_over_lambda*cos(2*pi*rand); % AoA ~ U(0,2*pi)
                projection = mu_eg.'*(sqrt(1/3)*randn(3, 1));
                for i = 1:I
                    G(i, k) = G(i, k) + (1/hbar)*projection*alpha_lk*exp(-1j*(i-1)*phi_lk);
                end
            end
        end
        channelPower = channelPower + norm(G, 'fro')^2;

        S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);         % K x P, CN(0,1)
        A_signal = G*S;                                     % I x P
        signalPower = mean(abs(A_signal(:)).^2);

        %% ---- Reference shape (one path, one epsilon_b, s_b = 1) ----
        alpha_b = sqrt(10/2)*(randn + 1j*randn);            % CN(0,10)
        phi_b = 2*pi*d_over_lambda*cos(2*pi*rand);
        projection_b = mu_eg.'*(sqrt(1/3)*randn(3, 1));
        B_shape = zeros(I, P);                              % I x P
        for i = 1:I
            for p = 1:P
                B_shape(i, p) = (1/hbar)*projection_b*alpha_b*exp(-1j*(i-1)*phi_b);
            end
        end

        for snrIndex = 1:numSNR

            sigma2 = signalPower/10^(SNR_list(snrIndex)/10);
            N_complex = sqrt(sigma2/2)*(randn(I, P) + 1j*randn(I, P));   % same noise for every RSR
            G0 = sqrt(G0_var/2)*(randn(I, K) + 1j*randn(I, K));          % same G0 for every RSR

            crlb(snrIndex) = crlb(snrIndex) + 2*sigma2*I*real(trace(inv(conj(S)*S.')));

            for rIndex = 1:numRSR

                % scale the reference to the wanted RSR
                B = B_shape*sqrt(10^(RSR_list(rIndex)/10)*signalPower/mean(abs(B_shape(:)).^2));
                Y = abs(A_signal + B + N_complex);          % I x P

                G_GD = run_gd(Y, S, B, G0, max_iterations, tol);
                G_GS = run_gs(Y, S, B, max_iterations, tol);

                mseGD(snrIndex, rIndex) = mseGD(snrIndex, rIndex) + norm(G - G_GD, 'fro')^2;
                mseGS(snrIndex, rIndex) = mseGS(snrIndex, rIndex) + norm(G - G_GS, 'fro')^2;
            end
        end
    end

    NMSE_GD(pIndex, :, :) = 10*log10(mseGD/channelPower);
    NMSE_GS(pIndex, :, :) = 10*log10(mseGS/channelPower);
    NMSE_CRLB(pIndex, :) = 10*log10(crlb/channelPower);
end

%% ---- Table ----
for pIndex = 1:numP
    for snrIndex = 1:numSNR
        fprintf('\nP = %d, SNR = %d dB (CRLB = %.2f dB)\n', P_list(pIndex), SNR_list(snrIndex), NMSE_CRLB(pIndex, snrIndex));
        fprintf('  RSR [dB]     GD       GS     GD - GS\n');
        for rIndex = 1:numRSR
            gd = NMSE_GD(pIndex, snrIndex, rIndex);
            gs = NMSE_GS(pIndex, snrIndex, rIndex);
            fprintf('   %4d     %7.2f  %7.2f   %6.2f\n', RSR_list(rIndex), gd, gs, gd - gs);
        end
    end
end

%% ---- Plot: NMSE vs RSR, solid = GD, dashed = GS, dotted = CRLB ----
colors = [0 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];
figure('Color', 'w', 'Position', [100 100 1200 480]);
for pIndex = 1:numP
    subplot(1, 2, pIndex);
    hold on; grid on; box on;
    for snrIndex = 1:numSNR
        c = colors(snrIndex, :);
        plot(RSR_list, squeeze(NMSE_GD(pIndex, snrIndex, :)), '-s', 'Color', c, 'LineWidth', 1.5, ...
             'MarkerFaceColor', c, 'DisplayName', sprintf('GD, SNR = %d dB', SNR_list(snrIndex)));
        plot(RSR_list, squeeze(NMSE_GS(pIndex, snrIndex, :)), '--d', 'Color', c, 'LineWidth', 1.2, ...
             'DisplayName', sprintf('GS, SNR = %d dB', SNR_list(snrIndex)));
        yline(NMSE_CRLB(pIndex, snrIndex), ':', 'Color', c, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
    xline(-2, 'k--', 'paper value', 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
    xlabel('RSR = E|b|^2 / E|GS|^2 [dB]');
    ylabel('NMSE [dB]');
    title(sprintf('1D, P = %d, polarization per path, converged, %d trials (dotted: CRLB)', P_list(pIndex), MC));
    legend('Location', 'northeast');
    xlim([RSR_list(1) RSR_list(end)]);
    hold off;
end

exportgraphics(gcf, 'rsr_sweep.png', 'Resolution', 200);
save('rsr_sweep.mat', 'RSR_list', 'SNR_list', 'P_list', 'NMSE_GD', 'NMSE_GS', 'NMSE_CRLB', 'MC');
fprintf('\nsaved rsr_sweep.png and .mat\n');

%% ---- Local functions ----
function G = run_gd(Y, S, B, G0, T, tol)
% GD on the linearised model, step 1/lambda_max(S*S^H), run to convergence
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

function G = run_gs(Y, S, B, T, tol)
% biased GS of [10]: spectral initialisation, then iterations until convergence
[I, P] = size(Y);
K = size(S, 1);
G = zeros(I, K);
for i = 1:I
    y_i = Y(i, :).';
    b_i = B(i, :).';
    Abar = [S.', b_i];
    M = zeros(K+1, K+1);
    for p = 1:P
        a_bar_p = Abar(p, :)';
        M = M + y_i(p)*(a_bar_p*a_bar_p');
    end
    [V, D] = eig(M);
    [~, maxIndex] = max(real(diag(D)));
    v = V(:, maxIndex);
    r_bar = (abs(Abar*v).'*y_i)/(norm(Abar*v)^2);
    gbar0 = r_bar*v;
    x = exp(-1j*angle(gbar0(K+1)))*gbar0(1:K);
    for t = 1:T
        theta = angle(S.'*x + b_i);
        x_new = inv(conj(S)*S.')*(conj(S)*(y_i.*exp(1j*theta) - b_i));
        relChange = norm(x_new - x)^2/norm(x_new)^2;
        x = x_new;
        if relChange < tol
            break;
        end
    end
    G(i, :) = x.';
end
end
