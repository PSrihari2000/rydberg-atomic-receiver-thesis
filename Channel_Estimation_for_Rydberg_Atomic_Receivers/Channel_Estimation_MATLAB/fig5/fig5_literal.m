clear; clc; close all;

rng(2026,'twister');

% FIG. 5: Channel Estimation for 2D Rydberg Atomic Antenna Array

%% ---- Physical constants -- (1/hbar)*mu_eg^T, Eq.(16)/(17) ----
hbar = 6.626e-34 / (2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg_y = 1785.9 * q_charge * a0_bohr;   % mu_eg=[0,mu_eg_y,0]^T

%% ---- Parameters ----
MC = 200;

I1 = 8; I2 = 8;
K = 3;

P_list = 5:5:50;
numP = length(P_list);

SNR_fixed_dB = 5;   % Section V: Fig.5 fixed at SNR = 5 dB

PGD_iterations = 300;
tol = 1e-8;

GD_init_var = 0.1;

d1_over_lambda = 0.5;   % ASSUMPTION -- paper gives no numeric value
d2_over_lambda = 0.5;   % ASSUMPTION -- paper gives no numeric value

IJ = I1*I2;

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FIG. 5: 2D Rydberg Atomic Receiver Channel Estimation\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n',MC);
fprintf('Array I1 x I2      = %d x %d\n',I1,I2);
fprintf('Users K            = %d\n',K);
fprintf('Fixed SNR          = %d dB\n',SNR_fixed_dB);
fprintf('Pilot lengths      = [%s]\n',num2str(P_list));
fprintf('====================================================\n\n');

%% ---- Result arrays ----
NMSE_PGD = zeros(1,numP);
NMSE_CRLB = zeros(1,numP);
refRatio_mean = zeros(1,numP);

snrLinear = 10^(SNR_fixed_dB/10);

%% ---- Main loop over pilot length ----
for pIndex = 1:numP

    P = P_list(pIndex);

    msePGD_acc = 0;
    crlb_acc = 0;
    signal_acc = 0;
    refPower_acc = 0;
    interferencePower_acc = 0;

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

        %% ---- Noise at fixed SNR = 5 dB ----
        sigma2_complex = signalPower / snrLinear;
        N_complex = sqrt(sigma2_complex/2)*(randn(IJ,P) + 1j*randn(IJ,P));

        interferencePower = signalPower + sigma2_complex;   % E|a+n|^2, Eq.(10)-analogue
        refPower_acc = refPower_acc + refPower;
        interferencePower_acc = interferencePower_acc + interferencePower;

        Y = abs(A_signal + Bmat + N_complex);   % Eq.(18)-(19)
        absB = abs(Bmat);

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

        msePGD_acc = msePGD_acc + norm(Gflat-Ghat_PGD,'fro')^2;
        signal_acc = signal_acc + norm(Gflat,'fro')^2;

        sigma2_linear = sigma2_complex/2;
        CRB_one = 4*sigma2_linear*pinv(conj(S)*S.');
        crlb_acc = crlb_acc + IJ*real(trace(CRB_one));

        if mod(mc,50)==0 || mc==MC
            fprintf('P=%2d : Monte Carlo %d / %d\n', P,mc,MC);
        end
    end

    NMSE_PGD(pIndex) = msePGD_acc / signal_acc;
    NMSE_CRLB(pIndex) = crlb_acc / signal_acc;
    refRatio_mean(pIndex) = refPower_acc / interferencePower_acc;
end

fprintf('\nEmpirical E|b|^2 / E|a+n|^2 (SNR=%d dB fixed)\n', SNR_fixed_dB);
fprintf('------------------------------------------------------------\n');
for pIndex = 1:numP
    fprintf('P=%2d | E|b|^2/E|a+n|^2 = %6.2f dB\n', ...
        P_list(pIndex), 10*log10(refRatio_mean(pIndex)));
end

fprintf('\nResults\n');
fprintf('------------------------------------------------------------\n');
for pIndex = 1:numP
    fprintf('P=%2d | PGD=%8.2f dB | CRLB=%8.2f dB\n', ...
        P_list(pIndex), 10*log10(NMSE_PGD(pIndex)), 10*log10(NMSE_CRLB(pIndex)));
end

%% ---- Plot ----
figure('Color','w','Position',[100 100 1100 650]);
hold on; grid on; box on;

plot(P_list, 10*log10(NMSE_PGD), '-^', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','PGD');
plot(P_list, 10*log10(NMSE_CRLB), '-o', 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','CRLB');

xlabel('Pilot Length', 'FontWeight','bold');
ylabel('NMSE [dB]', 'FontWeight','bold');
title('Channel Estimation for 2D Rydberg Atomic Receiver', 'FontWeight','bold');
legend('Location','northeast');
set(gca, 'FontSize',12, 'LineWidth',1);
hold off;

saveas(gcf, 'fig5_literal_result.png');

%% ---- Save results ----
save('fig5_literal_results.mat', 'P_list', 'SNR_fixed_dB', 'NMSE_PGD', ...
    'NMSE_CRLB', 'MC', 'I1', 'I2', 'K', 'refRatio_mean');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig5_literal_results.mat\n');
fprintf('Plot saved to fig5_literal_result.png\n');
fprintf('====================================================\n');
