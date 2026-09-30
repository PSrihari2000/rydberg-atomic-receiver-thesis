%% fig3_herm_check.m
% Test: does removing  M = (M + M')/2  in the GS initialisation change the GS results?
% Same data for both versions (literal Fig. 3 settings); compares the GS NMSE.

clear; clc; close all;
rng(1);

MC = 500;
I = 8;
K = 3;
P_list = [10 30];
SNR_dB = -5:5:30;
numSNR = length(SNR_dB);
GS_iterations = 50;
d_over_lambda = 0.5;

hbar = 6.62607015e-34/(2*pi);
mu_eg = [0; 1785.9*1.602e-19*5.292e-11; 0];

maxDiffG = 0;                       % largest change of any GS estimate (relative)
nComplexEig = 0;                    % how often eig returned complex eigenvalues without the line
nCalls = 0;

for pIndex = 1:length(P_list)

    P = P_list(pIndex);
    mse_with = zeros(1, numSNR);
    mse_without = zeros(1, numSNR);
    channelPower = 0;

    for mc = 1:MC

        G = zeros(I, K);
        for k = 1:K
            Lk = randi([3, 7]);
            for l = 1:Lk
                alpha_lk = (randn + 1j*randn)/sqrt(2);
                phi_lk = 2*pi*d_over_lambda*cos(2*pi*rand);
                for i = 1:I
                    epsilon_ikl = sqrt(1/3)*randn(3, 1);
                    G(i, k) = G(i, k) + (1/hbar)*(mu_eg.'*epsilon_ikl)*alpha_lk*exp(-1j*(i-1)*phi_lk);
                end
            end
        end
        channelPower = channelPower + norm(G, 'fro')^2;

        S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);
        A_signal = G*S;

        alpha_b = sqrt(10/2)*(randn + 1j*randn);
        phi_b = 2*pi*d_over_lambda*cos(2*pi*rand);
        B = zeros(I, P);
        for i = 1:I
            epsilon_bi = sqrt(1/3)*randn(3, 1);
            B(i, :) = (1/hbar)*(mu_eg.'*epsilon_bi)*alpha_b*exp(-1j*(i-1)*phi_b)*ones(1, P);
        end

        signalPower = mean(abs(A_signal(:)).^2);

        for snrIndex = 1:numSNR
            sigma2 = signalPower/10^(SNR_dB(snrIndex)/10);
            N = sqrt(sigma2/2)*(randn(I, P) + 1j*randn(I, P));
            Y = abs(A_signal + B + N);

            [G_with, ~] = run_gs(Y, S, B, GS_iterations, true);
            [G_without, nc] = run_gs(Y, S, B, GS_iterations, false);

            nComplexEig = nComplexEig + nc;
            nCalls = nCalls + I;
            maxDiffG = max(maxDiffG, norm(G_with - G_without, 'fro')/norm(G_with, 'fro'));

            mse_with(snrIndex) = mse_with(snrIndex) + norm(G - G_with, 'fro')^2;
            mse_without(snrIndex) = mse_without(snrIndex) + norm(G - G_without, 'fro')^2;
        end
    end

    fprintf('\nP = %d: GS NMSE [dB]\n', P);
    fprintf('  SNR    with line   without line   difference\n');
    for snrIndex = 1:numSNR
        a = 10*log10(mse_with(snrIndex)/channelPower);
        b = 10*log10(mse_without(snrIndex)/channelPower);
        fprintf('  %3d   %9.4f   %11.4f   %10.2e\n', SNR_dB(snrIndex), a, b, b - a);
    end
end

fprintf('\nlargest relative change of a GS estimate: %.2e\n', maxDiffG);
fprintf('eig returned complex eigenvalues (without the line) in %d of %d calls\n', nComplexEig, nCalls);

function [G, nComplex] = run_gs(Y, S, B, T, symmetrize)
[I, ~] = size(Y);
K = size(S, 1);
G = zeros(I, K);
nComplex = 0;
for i = 1:I
    z_i = Y(i, :).';
    b_i = B(i, :).';
    Abar = [S.', b_i];
    M = Abar'*diag(z_i)*Abar;
    if symmetrize
        M = (M + M')/2;
    end
    [V, D] = eig(M);
    if any(imag(diag(D)) ~= 0)
        nComplex = nComplex + 1;
    end
    [~, maxIndex] = max(real(diag(D)));
    v = V(:, maxIndex);
    r_bar = (abs(Abar*v).'*z_i)/(norm(Abar*v)^2);
    gbar0 = r_bar*v;
    x = exp(-1j*angle(gbar0(K+1)))*gbar0(1:K);
    for t = 1:T
        theta = angle(S.'*x + b_i);
        x = (conj(S)*S.')\(conj(S)*(z_i.*exp(1j*theta) - b_i));
    end
    G(i, :) = x.';
end
end
