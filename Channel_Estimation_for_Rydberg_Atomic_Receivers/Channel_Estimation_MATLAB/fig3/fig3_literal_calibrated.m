clear; clc; close all;

rng(2026,'twister');

% FIG. 3: Channel Estimation for 1D Rydberg Atomic Antenna Array

%% ---- Parameters ----
MC = 300;

I = 8;
K = 3;

P_list = [10 30];

SNR_dB = -5:5:30;
numSNR = length(SNR_dB);

GS_iterations = 100;
GD_iterations = 500;
GD_tol = 1e-8;

GD_init_var = 0.1;

d_over_lambda = 0.5;   % ASSUMPTION -- paper gives no numeric value

RSR_dB = 30;   % ASSUMPTION -- disclosed reference margin, not a paper value

%% ---- Physical constants -- (1/hbar)*mu_eg^T, Eq.(7)/(9) ----
hbar = 6.626e-34 / (2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FIG. 3: 1D Rydberg Atomic Receiver Channel Estimation\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n',MC);
fprintf('Antennas I         = %d\n',I);
fprintf('Users K            = %d\n',K);
fprintf('Pilot lengths      = [%s]\n',num2str(P_list));
fprintf('Reference margin   = %.1f dB\n',RSR_dB);
fprintf('====================================================\n\n');

%% ---- Result arrays ----
NMSE_GS = zeros(length(P_list),numSNR);
NMSE_GD = zeros(length(P_list),numSNR);
NMSE_CRLB = zeros(length(P_list),numSNR);
refRatio_mean = zeros(length(P_list),numSNR);

%% ---- Main loop over pilot length ----
for pIndex = 1:length(P_list)

    P = P_list(pIndex);

    fprintf('\n============================================\n');
    fprintf('Simulating P = %d\n',P);
    fprintf('============================================\n');

    mseGS_acc = zeros(1,numSNR);
    mseGD_acc = zeros(1,numSNR);
    crlb_acc  = zeros(1,numSNR);
    signal_acc = zeros(1,numSNR);
    refPower_acc = zeros(1,numSNR);
    interferencePower_acc = zeros(1,numSNR);

    for mc = 1:MC

        %% ---- Channel -- Eq.(7), per-antenna polarization ----
        G = zeros(I,K);
        for k = 1:K
            Lk = randi([3,7]);
            for l = 1:Lk
                alpha_lk = (randn + 1j*randn)/sqrt(2);
                theta_lk = 2*pi*rand;
                phi_lk = 2*pi*d_over_lambda*cos(theta_lk);
                polarization = randn(I,1)/sqrt(3);   % eps(i,k,l), per antenna
                for i = 1:I
                    eps_vec = [0; polarization(i); 0];
                    projection = mu_eg.' * eps_vec;
                    G(i,k) = G(i,k) + (1/hbar)*projection*alpha_lk*exp(-1j*(i-1)*phi_lk);
                end
            end
        end
        %% ---- Pilots, signal ----
        S = (randn(K,P) + 1j*randn(K,P))/sqrt(2);
        A_signal = G*S;
        signalPower = mean(abs(A_signal(:)).^2);

        %% ---- Reference -- Eq.(9), alpha_b~CN(0,10), per-antenna polarization ----
        alpha_b = sqrt(10/2)*(randn + 1j*randn);
        theta_b = 2*pi*rand;
        phi_b = 2*pi*d_over_lambda*cos(theta_b);
        epsilon_b = randn(I,1)/sqrt(3);
        b_spatial = zeros(I,1);
        for i = 1:I
            eps_vec_b = [0; epsilon_b(i); 0];
            projection_b = mu_eg.' * eps_vec_b;
            b_spatial(i) = (1/hbar)*projection_b*alpha_b*exp(-1j*(i-1)*phi_b);
        end

        %% ---- Calibrate reference to target RSR (disclosed assumption) ----
        currentRefPower = mean(abs(b_spatial).^2);
        targetRefPower = signalPower * 10^(RSR_dB/10);
        b_spatial = b_spatial * sqrt(targetRefPower/(currentRefPower+eps));

        B = repmat(b_spatial,1,P);
        Z = exp(-1j*angle(B));

        refPower = mean(abs(b_spatial).^2);

        %% ---- SNR sweep ----
        for snrIndex = 1:numSNR
            snrLinear = 10^(SNR_dB(snrIndex)/10);
            sigma2_complex = signalPower / snrLinear;
            N_complex = sqrt(sigma2_complex/2)*(randn(I,P) + 1j*randn(I,P));

            interferencePower = signalPower + sigma2_complex;   % E|a+n|^2, Eq.(10)
            refPower_acc(snrIndex) = refPower_acc(snrIndex) + refPower;
            interferencePower_acc(snrIndex) = interferencePower_acc(snrIndex) + interferencePower;

            Y = abs(A_signal + B + N_complex);   % Eq.(8)
            Y_GD = Y;

            %% ---- GS: Cui Algorithm 1 (spectral init + alternating LS) ----
            G_hat_GS = zeros(I,K);
            A_gs = S.';
            for i = 1:I
                z_i = Y(i,:).';
                b_i = B(i,:).';
                Abar = [A_gs, b_i];
                M = Abar' * diag(z_i) * Abar;
                M = (M+M')/2;
                [V,D] = eig(M);
                [~,idx] = max(real(diag(D)));
                v = V(:,idx);
                projectionMagnitude = abs(Abar*v);
                r_bar = (projectionMagnitude.'*z_i)/(norm(Abar*v)^2+eps);
                sbar0 = r_bar*v;
                phaseCorrection = exp(-1j*angle(sbar0(K+1)));
                x = phaseCorrection*sbar0(1:K);
                for iter = 1:GS_iterations
                    theta = angle(A_gs*x + b_i);
                    target = z_i.*exp(1j*theta) - b_i;
                    x = (A_gs'*A_gs)\(A_gs'*target);
                end
                G_hat_GS(i,:) = x.';
            end

            %% ---- GD: Eq.(13)-(15) ----
            G_hat_GD = sqrt(GD_init_var/2)*(randn(I,K)+1j*randn(I,K));
            objLin = @(X) norm(Y_GD - abs(B) - real(Z.*(X*S)),'fro')^2;
            currentCost = objLin(G_hat_GD);
            for iter = 1:GD_iterations
                residual = Y_GD - abs(B) - real(Z.*(G_hat_GD*S));
                gradient = -2*((residual.*conj(Z))*S');
                eta = 1;
                while eta > 1e-10
                    G_trial = G_hat_GD - eta*gradient;
                    if objLin(G_trial) <= currentCost, break; end
                    eta = eta/2;
                end
                G_new = G_hat_GD - eta*gradient;
                newCost = objLin(G_new);
                relChange = norm(G_new-G_hat_GD,'fro')/(norm(G_hat_GD,'fro')+eps);
                G_hat_GD = G_new;
                currentCost = newCost;
                if relChange < GD_tol, break; end
            end

            mseGS_acc(snrIndex) = mseGS_acc(snrIndex) + norm(G-G_hat_GS,'fro')^2;
            mseGD_acc(snrIndex) = mseGD_acc(snrIndex) + norm(G-G_hat_GD,'fro')^2;
            signal_acc(snrIndex) = signal_acc(snrIndex) + norm(G,'fro')^2;

            sigma2_linear = sigma2_complex/2;
            CRB_one = 4*sigma2_linear*pinv(conj(S)*S.');
            crlb_acc(snrIndex) = crlb_acc(snrIndex) + I*real(trace(CRB_one));
        end

        if mod(mc,50)==0 || mc==MC
            fprintf('P=%d : Monte Carlo %d / %d\n', P,mc,MC);
        end
    end

    NMSE_GS(pIndex,:) = mseGS_acc ./ signal_acc;
    NMSE_GD(pIndex,:) = mseGD_acc ./ signal_acc;
    NMSE_CRLB(pIndex,:) = crlb_acc ./ signal_acc;
    refRatio_mean(pIndex,:) = refPower_acc ./ interferencePower_acc;

    fprintf('\nP=%d empirical E|b|^2 / E|a+n|^2\n', P);
    fprintf('------------------------------------------------------------\n');
    for snrIndex = 1:numSNR
        fprintf('P=%2d | SNR=%4d dB | E|b|^2/E|a+n|^2 = %6.2f dB\n', ...
            P, SNR_dB(snrIndex), 10*log10(refRatio_mean(pIndex,snrIndex)));
    end

    fprintf('\nResults for P = %d\n',P);
    fprintf('------------------------------------------------------------\n');
    for snrIndex = 1:numSNR
        fprintf('P=%2d | SNR=%4d dB | GS=%8.2f dB | GD=%8.2f dB | CRLB=%8.2f dB\n', ...
            P, SNR_dB(snrIndex), 10*log10(NMSE_GS(pIndex,snrIndex)), ...
            10*log10(NMSE_GD(pIndex,snrIndex)), 10*log10(NMSE_CRLB(pIndex,snrIndex)));
    end
end

%% ---- Plot ----
figure('Color','w','Position',[100 100 1100 650]);
hold on; grid on; box on;

plot(SNR_dB, 10*log10(NMSE_GS(1,:)), '--d', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','GS, P=10');
plot(SNR_dB, 10*log10(NMSE_GD(1,:)), '--s', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','GD, P=10');
plot(SNR_dB, 10*log10(NMSE_CRLB(1,:)), '-o', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','CRLB, P=10');

plot(SNR_dB, 10*log10(NMSE_GS(2,:)), '--d', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','GS, P=30');
plot(SNR_dB, 10*log10(NMSE_GD(2,:)), '--s', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','GD, P=30');
plot(SNR_dB, 10*log10(NMSE_CRLB(2,:)), '-o', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','CRLB, P=30');

xlabel('SNR [dB]', 'FontWeight','bold');
ylabel('NMSE [dB]', 'FontWeight','bold');
title('Fig.3 NMSE performance comparison with respect to the SNR under Rydberg atom-based 1D antenna array', 'FontWeight','bold');
legend('Location','southwest');
set(gca, 'FontSize',12, 'LineWidth',1);
hold off;

saveas(gcf, 'fig3_literal_calibrated_result.png');

%% ---- Save results ----
save('fig3_literal_calibrated_results.mat', 'SNR_dB', 'P_list', 'NMSE_GS', 'NMSE_GD', ...
    'NMSE_CRLB', 'MC', 'I', 'K', 'RSR_dB', 'refRatio_mean');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig3_literal_calibrated_results.mat\n');
fprintf('Plot saved to fig3_literal_calibrated_result.png\n');
fprintf('====================================================\n');
