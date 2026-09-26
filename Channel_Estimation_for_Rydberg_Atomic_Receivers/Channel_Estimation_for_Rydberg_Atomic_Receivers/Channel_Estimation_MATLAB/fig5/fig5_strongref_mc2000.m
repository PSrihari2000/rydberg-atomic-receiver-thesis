%% STRONG-REFERENCE VARIANT (added 2026-09-23 for the mid-review report)
% Identical to the literal script except:
%  (1) s_{b,p} = 1 for all p (reference constant over pilots, as in Cui et al. Eq.35)
%  (2) reference rescaled per trial so E|b|^2 / E|a|^2 = ref_ratio^2 (34 dB),
%      i.e. the strong-reference regime |b| >> |a+n| that Xu et al. Eq.(10) assumes.
%      With the paper's literal priors this ratio is only about -2 dB.
clear; clc; close all;

%% ---- Physical constants ----
hbar = 6.626e-34 / (2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg_y = 1785.9 * q_charge * a0_bohr;

%% ---- Parameters ----
MC = 2000;

I1 = 8; I2 = 8;
K = 3;

P_list = 5:5:50;
numP = length(P_list);

SNR_fixed_dB = 5;

PGD_iterations = 500;
tol = 1e-8;

PGD_init_var = 0.1;

d1_over_lambda = 0.5;
d2_over_lambda = 0.5;  

IJ = I1*I2;   % total antennas = 8*8 = 64

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FIG. 5 (s_b,p~CN(0,1) variant): 2D Rydberg Channel Estimation\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n',MC);
fprintf('Array I1 x I2      = %d x %d\n',I1,I2);
fprintf('Users K            = %d\n',K);
fprintf('Fixed SNR          = %d dB\n',SNR_fixed_dB);
fprintf('Pilot lengths      = [%s]\n',num2str(P_list));
fprintf('Reference pilot s_b,p ~ CN(0,1) (matching user pilot distribution)\n');
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
    channelPower_acc = 0;
    refPower_acc = 0;
    signalPlusNoisePower_acc = 0;

    for mc = 1:MC

        % ---- Channel -- Eq.(16)  ----
        G3 = zeros(K,IJ);        % K x IJ
        Lk = zeros(K,1);
        i1idx = (1:I1).';
        i2idx = (1:I2).';

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
                pol = randn/sqrt(3);  % Pol ~ N(0,1/3)
                phase_i1 = exp(-1j*(i1idx-1)*u_lk);       % I1 x 1
                phase_i2 = exp(-1j*(i2idx-1)*v_lk).';     % 1 x I2
                Gslice = Gslice + (mu_eg_y/hbar)*pol*alpha_lk*(phase_i1*phase_i2);   % I1 x I2
            end
            G3(k,:) = reshape(Gslice,1,IJ);
        end

        % ---- Pilot signal ----
        S = (randn(K,P) + 1j*randn(K,P))/sqrt(2);   % K x P
        A_signal3 = S.' * G3;                        % (PxK)*(KxIJ) = P x IJ
        signalPower = mean(abs(A_signal3(:)).^2);

        % ---- Reference signal parameters -- Eq.(17) ----
        alpha_b = sqrt(10/2)*(randn + 1j*randn);   % alpha_b~CN(0,10)
        theta_b = 2*pi*rand;
        phi_b = 2*pi*rand;
        u_b = 2*pi*d1_over_lambda*cos(theta_b);
        v_b = 2*pi*d2_over_lambda*sin(theta_b)*cos(phi_b);

        % -- Reference pilot symbols --
        S_b = ones(1,P);   % s_{b,p}=1: reference constant over pilot slots (Cui et al. Eq.35)

        Bslice = zeros(I1,I2);   % I1 x I2, per-antenna part of the reference
        for i1 = 1:I1
            for i2 = 1:I2
                epsilon_b_i1i2 = randn/sqrt(3);   % epsilon_{b,i1,i2} ~ N(0,1/3)
                projection_b = mu_eg_y*epsilon_b_i1i2;
                phase_val = exp(-1j*(i1-1)*u_b)*exp(-1j*(i2-1)*v_b);
                Bslice(i1,i2) = projection_b*phase_val;
            end
        end

        b_flat = reshape(Bslice,1,IJ);   % 1 x IJ 

        B3 = zeros(P,IJ);   % P x IJ, reference signal matrix
        for p = 1:P
            B3(p,:) = (S_b(p)/hbar)*alpha_b*b_flat;   % 1×IJ, picking out row p of B3, which is P×IJ
        end

        ref_ratio = 50;   % ASSUMPTION: strong-reference margin, E|b|^2/E|a|^2 = 34 dB
        B3 = B3*sqrt(ref_ratio^2*mean(abs(A_signal3(:)).^2)/mean(abs(B3(:)).^2));
        Z3 = exp(-1j*angle(B3));         % P x IJ

        refPower = mean(abs(B3(:)).^2);

        % ---- Noise at fixed SNR ----
        sigma2_complex = signalPower / snrLinear;    % scalar
        N3 = sqrt(sigma2_complex/2)*(randn(P,IJ) + 1j*randn(P,IJ));   % P x IJ

        signalPlusNoisePower = mean(abs(A_signal3(:)+N3(:)).^2);  
        refPower_acc = refPower_acc + refPower;
        signalPlusNoisePower_acc = signalPlusNoisePower_acc + signalPlusNoisePower;

        Y3 = abs(A_signal3 + B3 + N3);   % P x IJ -- raw nonlinear observation
        absB3 = abs(B3);                  % P x IJ

        % ---- PGD: Eq.(23)-(27), gradient step (Eq.24) + per-user SVD rank projection ----
        Ghat_PGD = sqrt(PGD_init_var/2)*(randn(K,IJ)+1j*randn(K,IJ));   % K x IJ
        objPGD = @(X) norm(Y3 - absB3 - real(Z3.*(S.'*X)),'fro')^2;
        curCostPGD = objPGD(Ghat_PGD);   % scalar

        for iter = 1:PGD_iterations
            residual = Y3 - absB3 - real(Z3.*(S.'*Ghat_PGD));   % P x IJ
            gradient = -conj(S)*(residual.*conj(Z3));            % K x IJ
            eta = 1;   % Initial step size

            while eta > 1e-10
                Mtrial = Ghat_PGD - eta*gradient;   % K x IJ
                if objPGD(Mtrial) <= curCostPGD, break; end
                eta = eta/2;
            end
            M = Ghat_PGD - eta*gradient;   % K x IJ

            Gnew = zeros(K,IJ);   % K x IJ
            for k = 1:K
                Mk = reshape(M(k,:),I1,I2);   % I1 x I2
                [U,Sig,V] = svd(Mk);             % U: I1xI1, Sig: I1xI2, V: I2xI2
                r = Lk(k);                        % scalar, true rank for user k
                Uk = U(:,1:r);       % I1 x r
                Sigk = Sig(1:r,1:r); % r x r
                Vk = V(:,1:r);       % I2 x r
                Xk = Uk*Sigk*Vk';    % I1 x I2 -- rank-r reconstruction
                Gnew(k,:) = reshape(Xk,1,IJ);   % 1 x IJ
            end

            relChange = norm(Ghat_PGD-Gnew,'fro')/(norm(Ghat_PGD,'fro'));   % scalar
            Ghat_PGD = Gnew;
            curCostPGD = objPGD(Ghat_PGD);
            if relChange < tol, break; end
        end

        msePGD_acc = msePGD_acc + norm(G3-Ghat_PGD,'fro')^2;
        channelPower_acc = channelPower_acc + norm(G3,'fro')^2;

        sigma2_linear = sigma2_complex/2;                                    % scalar
        CRB_full = 4*sigma2_linear*kron(eye(IJ), pinv(conj(S)*S.'));        % (IJ*K) x (IJ*K), literal Eq.(31)
        crlb_acc = crlb_acc + real(trace(CRB_full));                        % scalar

        if mod(mc,50)==0 || mc==MC
            fprintf('P=%2d : Monte Carlo %d / %d\n', P,mc,MC);
        end
    end

    NMSE_PGD(pIndex) = msePGD_acc / channelPower_acc;
    NMSE_CRLB(pIndex) = crlb_acc / channelPower_acc;
    refRatio_mean(pIndex) = refPower_acc / signalPlusNoisePower_acc;
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

plot(P_list, 10*log10(NMSE_PGD), '-^','Color',[0 0.45 0.74],'MarkerFaceColor',[0 0.45 0.74], 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','PGD');
plot(P_list, 10*log10(NMSE_CRLB), '-o','Color',[0.85 0.33 0.10], 'LineWidth',1.8, 'MarkerSize',7, 'DisplayName','CRLB');

xlabel('Pilot Length', 'FontWeight','bold');
ylabel('NMSE [dB]', 'FontWeight','bold');
legend('Location','northeast');
set(gca, 'FontSize',12, 'LineWidth',1);
hold off;

exportgraphics(gcf,'fig5_strongref_mc2000.png','Resolution',300);

%% ---- Save results ----
save('fig5_strongref_mc2000.mat', 'P_list', 'SNR_fixed_dB', 'NMSE_PGD', ...
    'NMSE_CRLB', 'MC', 'I1', 'I2', 'K', 'refRatio_mean');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig5.mat\n');
fprintf('Plot saved to fig5.png\n');
fprintf('====================================================\n');
