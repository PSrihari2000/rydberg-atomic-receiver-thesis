clear; clc; close all;

rng(2026,'twister');

% FIG. 4: Channel Estimation for 2D Rydberg Atomic Antenna Array

%% ---- Physical constants -- (1/hbar)*mu_eg^T, Eq.(16)/(17) ----
hbar = 6.626e-34 / (2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg_y = 1785.9 * q_charge * a0_bohr;   % mu_eg=[0,mu_eg_y,0]^T

%% ---- Parameters ----
MC = 100;

I1 = 8; I2 = 8;
K = 3;

P_list = [10 30];
SNR_dB = -5:5:30;
numSNR = length(SNR_dB);

GD_iterations = 500;
PGD_iterations = 300;
tol = 1e-8;

GD_init_var = 0.1;

d1_over_lambda = 0.5;   % ASSUMPTION -- paper gives no numeric value
d2_over_lambda = 0.5;   % ASSUMPTION -- paper gives no numeric value

IJ = I1*I2;

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FIG. 4: 2D Rydberg Atomic Receiver Channel Estimation\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n',MC);
fprintf('Array I1 x I2      = %d x %d\n',I1,I2);
fprintf('Users K            = %d\n',K);
fprintf('Pilot lengths      = [%s]\n',num2str(P_list));
fprintf('====================================================\n\n');

%% ---- Result arrays ----
NMSE_GD = zeros(length(P_list),numSNR);
NMSE_PGD = zeros(length(P_list),numSNR);
NMSE_CRLB = zeros(length(P_list),numSNR);
refRatio_mean = zeros(length(P_list),numSNR);

%% ---- Main loop over pilot length ----
for pIndex = 1:length(P_list)

    P = P_list(pIndex);

    fprintf('\n============================================\n');
    fprintf('Simulating P = %d\n',P);
    fprintf('============================================\n');

    mseGD_acc = zeros(1,numSNR);
    msePGD_acc = zeros(1,numSNR);
    crlb_acc = zeros(1,numSNR);
    signal_acc = zeros(1,numSNR);
    refPower_acc = zeros(1,numSNR);
    interferencePower_acc = zeros(1,numSNR);

    for mc = 1:MC

        %% ---- Channel -- Eq.(16), scalar-per-path polarization ----
        % ASSUMPTION: eps(i1,i2,k,l) taken constant across the array per
        % path (not independent per antenna pair). Required for
        % rank(Gk)<=Lk (Sec.III-B), which fails otherwise -- verified
        % empirically: per-antenna-pair polarization gives rank(Gk)=8
        % (full rank) regardless of Lk, contradicting the paper's own
        % stated rank property.
        Gflat = zeros(IJ,K);
        Lk = zeros(K,1);
        i1idx = (0:I1-1).';
        i2idx = (0:I2-1).';

        for k = 1:K
            Lkk = randi([3,7]);
            Lk(k) = Lkk;
            Gslice = zeros(I1,I2);
            for l = 1:Lkk
                alpha_lk = (randn + 1j*randn)/sqrt(2);
                theta_lk = 2*pi*rand;
                phi_lk = 2*pi*rand;
                u_lk = 2*pi*d1_over_lambda*cos(theta_lk);
                v_lk = 2*pi*d2_over_lambda*sin(theta_lk)*cos(phi_lk);

                pol = randn/sqrt(3);   % eps(k,l), scalar per path

                phase_i1 = exp(-1j*i1idx*u_lk);
                phase_i2 = exp(-1j*i2idx*v_lk).';
                Gslice = Gslice + (mu_eg_y/hbar)*pol*alpha_lk*(phase_i1*phase_i2);
            end
            Gflat(:,k) = reshape(Gslice, IJ, 1);
        end

        %% ---- Pilots, signal ----
        S = (randn(K,P) + 1j*randn(K,P))/sqrt(2);
        A_signal = Gflat*S;
        signalPower = mean(abs(A_signal(:)).^2);

        %% ---- Reference -- Eq.(17), alpha_b~CN(0,10), per-antenna-pair polarization ----
        alpha_b = sqrt(10/2)*(randn + 1j*randn);
        theta_b = 2*pi*rand;
        phi_b = 2*pi*rand;
        u_b = 2*pi*d1_over_lambda*cos(theta_b);
        v_b = 2*pi*d2_over_lambda*sin(theta_b)*cos(phi_b);

        eps_b = randn(I1,I2)/sqrt(3);   % eps(b,i1,i2)

        phase_b_i1 = exp(-1j*i1idx*u_b);
        phase_b_i2 = exp(-1j*i2idx*v_b).';
        Bslice = (mu_eg_y/hbar)*eps_b.*alpha_b.*(phase_b_i1*phase_b_i2);
        b_flat = reshape(Bslice, IJ, 1);

        Bmat = repmat(b_flat,1,P);
        Zmat = exp(-1j*angle(Bmat));

        refPower = mean(abs(b_flat).^2);

        %% ---- SNR sweep ----
        for snrIndex = 1:numSNR
            snrLinear = 10^(SNR_dB(snrIndex)/10);
            sigma2_complex = signalPower / snrLinear;
            N_complex = sqrt(sigma2_complex/2)*(randn(IJ,P) + 1j*randn(IJ,P));

            interferencePower = signalPower + sigma2_complex;   % E|a+n|^2, Eq.(10)-analogue
            refPower_acc(snrIndex) = refPower_acc(snrIndex) + refPower;
            interferencePower_acc(snrIndex) = interferencePower_acc(snrIndex) + interferencePower;

            Y = abs(A_signal + Bmat + N_complex);   % Eq.(18)-(19)
            absB = abs(Bmat);

            %% ---- GD: Eq.(13)-(15) form, unconstrained ----
            Ghat_GD = sqrt(GD_init_var/2)*(randn(IJ,K)+1j*randn(IJ,K));
            objGD = @(X) norm(Y - absB - real(Zmat.*(X*S)),'fro')^2;
            curCostGD = objGD(Ghat_GD);
            for iter = 1:GD_iterations
                residual = Y - absB - real(Zmat.*(Ghat_GD*S));
                gradient = -2*((residual.*conj(Zmat))*S');
                eta = 1;
                while eta > 1e-10
                    Gtrial = Ghat_GD - eta*gradient;
                    if objGD(Gtrial) <= curCostGD, break; end
                    eta = eta/2;
                end
                Gnew = Ghat_GD - eta*gradient;
                newCost = objGD(Gnew);
                relChange = norm(Gnew-Ghat_GD,'fro')/(norm(Ghat_GD,'fro')+eps);
                Ghat_GD = Gnew;
                curCostGD = newCost;
                if relChange < tol, break; end
            end

            %% ---- PGD: Eq.(22)-(27), gradient step + per-user SVD rank projection ----
            Ghat_PGD = sqrt(GD_init_var/2)*(randn(IJ,K)+1j*randn(IJ,K));
            objPGD = @(X) norm(Y - absB - real(Zmat.*(X*S)),'fro')^2;
            curCostPGD = objPGD(Ghat_PGD);
            for iter = 1:PGD_iterations
                residual = Y - absB - real(Zmat.*(Ghat_PGD*S));
                gradient = -2*((residual.*conj(Zmat))*S');
                eta = 1;
                while eta > 1e-10
                    Mtrial = Ghat_PGD - eta*gradient;
                    if objPGD(Mtrial) <= curCostPGD, break; end
                    eta = eta/2;
                end
                M = Ghat_PGD - eta*gradient;

                Gnew = zeros(IJ,K);
                for k = 1:K
                    Mk = reshape(M(:,k), I1, I2);
                    [U,Sig,V] = svd(Mk);
                    r = Lk(k);
                    Uk = U(:,1:r); Sigk = Sig(1:r,1:r); Vk = V(:,1:r);
                    Xk = Uk*Sigk*Vk';
                    Gnew(:,k) = reshape(Xk, IJ, 1);
                end

                relChange = norm(Gnew-Ghat_PGD,'fro')/(norm(Ghat_PGD,'fro')+eps);
                Ghat_PGD = Gnew;
                curCostPGD = objPGD(Ghat_PGD);
                if relChange < tol, break; end
            end

            mseGD_acc(snrIndex) = mseGD_acc(snrIndex) + norm(Gflat-Ghat_GD,'fro')^2;
            msePGD_acc(snrIndex) = msePGD_acc(snrIndex) + norm(Gflat-Ghat_PGD,'fro')^2;
            signal_acc(snrIndex) = signal_acc(snrIndex) + norm(Gflat,'fro')^2;

            sigma2_linear = sigma2_complex/2;
            CRB_one = 4*sigma2_linear*pinv(conj(S)*S.');
            crlb_acc(snrIndex) = crlb_acc(snrIndex) + IJ*real(trace(CRB_one));
        end

        if mod(mc,25)==0 || mc==MC
            fprintf('P=%d : Monte Carlo %d / %d\n', P,mc,MC);
        end
    end

    NMSE_GD(pIndex,:) = mseGD_acc ./ signal_acc;
    NMSE_PGD(pIndex,:) = msePGD_acc ./ signal_acc;
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
        fprintf('P=%2d | SNR=%4d dB | GD=%8.2f dB | PGD=%8.2f dB | CRLB=%8.2f dB\n', ...
            P, SNR_dB(snrIndex), 10*log10(NMSE_GD(pIndex,snrIndex)), ...
            10*log10(NMSE_PGD(pIndex,snrIndex)), 10*log10(NMSE_CRLB(pIndex,snrIndex)));
    end
end

%% ---- Plot ----
figure('Color','w','Position',[100 100 1100 650]);
hold on; grid on; box on;

plot(SNR_dB, 10*log10(NMSE_GD(1,:)), '--s', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','GD, P=10');
plot(SNR_dB, 10*log10(NMSE_PGD(1,:)), '-^', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','PGD, P=10');
plot(SNR_dB, 10*log10(NMSE_CRLB(1,:)), '-o', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','CRLB, P=10');

plot(SNR_dB, 10*log10(NMSE_GD(2,:)), '--s', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','GD, P=30');
plot(SNR_dB, 10*log10(NMSE_PGD(2,:)), '-^', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','PGD, P=30');
plot(SNR_dB, 10*log10(NMSE_CRLB(2,:)), '-o', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','CRLB, P=30');

xlabel('SNR [dB]', 'FontWeight','bold');
ylabel('NMSE [dB]', 'FontWeight','bold');
title('fig4. NMSE performance comparison with respect to the SNR under Rydberg atom-based 2D antenna array', 'FontWeight','bold');
legend('Location','southwest');
set(gca, 'FontSize',12, 'LineWidth',1);
hold off;

saveas(gcf, 'fig4_literal_result.png');

%% ---- Save results ----
save('fig4_literal_results.mat', 'SNR_dB', 'P_list', 'NMSE_GD', 'NMSE_PGD', ...
    'NMSE_CRLB', 'MC', 'I1', 'I2', 'K', 'refRatio_mean');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig4_literal_results.mat\n');
fprintf('Plot saved to fig4_literal_result.png\n');
fprintf('====================================================\n');
