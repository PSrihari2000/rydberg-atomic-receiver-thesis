clear; clc; close all;

%% ---- Parameters ----
MC = 500;

I = 8;
K = 3;

P_list = [10 30];

SNR_dB = -5:5:30;
numSNR = length(SNR_dB);

GS_iterations = 100;
GD_iterations = 100;
GD_tol = 1e-8;

GD_init_var = 0.1;   % ASSUMPTION -- paper states this CN(0,0.1) init only for 2D PGD; no 1D GD init is given
d_over_lambda = 0.5;   

%% ---- Physical constants ----
hbar = 6.626e-34/(2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0, 1785.9*q_charge*a0_bohr, 0].'; 

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FIG. 3 NMSE vs SNR for Rydberg atom-based 1D antenna array.\n');
fprintf('====================================================\n');
fprintf('Monte Carlo trials = %d\n',MC);
fprintf('Antennas I         = %d\n',I);
fprintf('Users K            = %d\n',K);
fprintf('Pilot lengths      = [%s]\n',num2str(P_list));
fprintf('====================================================\n\n');

%% ---- Result arrays ----
NMSE_GS = zeros(length(P_list),numSNR);
NMSE_GD = zeros(length(P_list),numSNR);
NMSE_CRLB = zeros(length(P_list),numSNR);
refRatio_mean = zeros(length(P_list),numSNR);
refMagRatio_mean = zeros(length(P_list),numSNR);

%% ---- Main loop over pilot length ----
for pIndex = 1:length(P_list)

    P = P_list(pIndex);

    fprintf('\n============================================\n');
    fprintf('Simulating P = %d\n',P);
    fprintf('============================================\n');

    mseGS_acc = zeros(1,numSNR);
    mseGD_acc = zeros(1,numSNR);
    crlb_acc = zeros(1,numSNR);
    channelPower_acc = zeros(1,numSNR);
    refPower_acc = zeros(1,numSNR);
    signalPlusNoisePower_acc = zeros(1,numSNR);
    refMag_acc = zeros(1,numSNR);
    signalPlusNoisePowerMag_acc = zeros(1,numSNR);

    for mc = 1:MC

        G = zeros(I,K);   % I x K, True channel
        for k = 1:K
            Lk = randi([3,7]);   % L_k ~ Uniform(3,7)
            for l = 1:Lk
                alpha_lk = (randn + 1j*randn)/sqrt(2);   % alpha_lk ~ CN(0,1)
                theta_lk = 2*pi*rand;   % theta_{l,k} -- Assumption, AOA: Uniform(0,2*pi)
                phi_lk = 2*pi*d_over_lambda*cos(theta_lk);
                for i = 1:I
                    epsilon_ikl = randn/sqrt(3);   % epsilon_{i,k,l} ~ N(0,1/3), this antenna's own draw
                    eps_vec = [0;epsilon_ikl;0];    % 3 x 1
                    projection = mu_eg.'*eps_vec;    % (1x3)(3x1) = 1x1
                    G(i,k) = G(i,k) + (1/hbar)*projection*alpha_lk*exp(-1j*(i-1)*phi_lk);
                end
            end
         end

         % transmitted signal 
        S = (randn(K,P) + 1j*randn(K,P))/sqrt(2);   % K x P, s_{k,p} ~ CN(0,1)
        A_signal = G*S;                             % (I x K)(K x P) = I x P , True channel x transmitted signal

        % Reference signal
        alpha_b = sqrt(10/2)*(randn + 1j*randn);    % alpha_b ~ CN(0,10)
        theta_b = 2*pi*rand;                        % theta_b assumed: Uniform(0,2*pi)
        phi_b = 2*pi*d_over_lambda*cos(theta_b);

        S_b = (randn(1,P) + 1j*randn(1,P))/sqrt(2);      % 1 x P, s_{b,p} ~ CN(0,1), refernce pilots

        B = zeros(I,P);   % Initial reference signal matrix
        for i = 1:I
            epsilon_bi = randn/sqrt(3);        % epsilon_{b,i} ~ N(0,1/3)
            eps_vec_b = [0;epsilon_bi;0];    % 3x1
            projection_b = mu_eg.'*eps_vec_b;  % (1x3)(3x1) = 1x1 
            for p = 1:P
                B(i,p) = (S_b(p)/hbar)*projection_b*alpha_b*exp(-1j*(i-1)*phi_b);   % I x P, reference signal matrix
            end
        end

        Z = exp(-1j*angle(B));                           % I x P, reference phase matrix

        signalPower = mean(abs(A_signal(:)).^2);       
        refPower = mean(abs(B(:)).^2);                  

        for snrIndex = 1:numSNR

            snrLinear = 10^(SNR_dB(snrIndex)/10);  
            sigma2_complex = signalPower/snrLinear;   

            N_complex = sqrt(sigma2_complex/2)*(randn(I,P) + 1j*randn(I,P));   % N ~ CN(0,sigma2_complex)

            signalPlusNoisePower = mean(abs(A_signal(:) + N_complex(:)).^2);   % E{|GS+N|^2}
            refPower_acc(snrIndex) = refPower_acc(snrIndex) + refPower;
            signalPlusNoisePower_acc(snrIndex) = signalPlusNoisePower_acc(snrIndex) + signalPlusNoisePower;

            refMag = mean(abs(B(:)));   
            signalplusNoiseMag = mean(abs(A_signal(:) + N_complex(:)));   
            refMag_acc(snrIndex) = refMag_acc(snrIndex) + refMag;
            signalPlusNoisePowerMag_acc(snrIndex) = signalPlusNoisePowerMag_acc(snrIndex) + signalplusNoiseMag;

            Y = abs(A_signal + B + N_complex);   % I x P, nonlinear magnitude observation, Eq.(8)
            Y_GD = Y;   % I x P

            %% ---- Biased GS Algorithm ----
            G_hat_GS = zeros(I,K);   % I x K, GS estimated matrix

            for i = 1:I

                z_i = Y(i,:).';   % P x 1, magnitude observation for antenna i
                b_i = B(i,:).';   % P x 1, known reference for antenna i

                Abar = [S.',b_i];   % Px(K+1), augmented matrix 
                M = Abar'*diag(z_i)*Abar;   % (K+1)x(K+1)
                
                v = eigs(M,1,'largestreal'); % (K+1)x1  
                r_bar = (abs(Abar*v).'*z_i)/(norm(Abar*v)^2);  % scale factor 
                gbar0 = r_bar*v;   % (K+1)x1, Initial Augmented channel estimate vector

                x = exp(-1j*angle(gbar0(K+1)))*gbar0(1:K); % Initial channel estimate

                for gsIter = 1:GS_iterations

                    theta = angle(S.'*x + b_i);       % Estimated phase
                    x = (conj(S)*S.')\(conj(S)*(z_i.*exp(1j*theta) - b_i));  % Updated channel estimate

                end

                G_hat_GS(i,:) = x.';   % 1 x K, store estimated g_i as row i

            end

            %% ---- GD: Eq.(13)-(15) ----
            G_hat_GD = sqrt(GD_init_var/2)*(randn(I,K) + 1j*randn(I,K));   % I x K

            cost = @(X) norm(Y_GD - abs(B) - real(Z.*(X*S)),'fro')^2;   
            currentCost = cost(G_hat_GD);   

            for gdIter = 1:GD_iterations

                residual = Y_GD - abs(B) - real(Z.*(G_hat_GD*S));   
                gradient = -2*((residual.*conj(Z))*S');

                eta = 1;   % Initial step size, assumed
                while eta > 1e-8         % assumption threshold = 1e-8
                    G_trial = G_hat_GD - eta*gradient;      % I x K
                    if cost(G_trial) <= currentCost, break; end
                    eta = eta/2;
                end

                G_new = G_hat_GD - eta*gradient;   % I x K
                newCost = cost(G_new);   
                relChange = norm(G_hat_GD - G_new,'fro')/(norm(G_hat_GD,'fro'));   % Relative channel-update change

                G_hat_GD = G_new;
                currentCost = newCost;

                if relChange < GD_tol, break; end

            end

            mseGS_acc(snrIndex) = mseGS_acc(snrIndex) + norm(G-G_hat_GS,'fro')^2;  
            mseGD_acc(snrIndex) = mseGD_acc(snrIndex) + norm(G-G_hat_GD,'fro')^2;   
            channelPower_acc(snrIndex) = channelPower_acc(snrIndex) + norm(G,'fro')^2;   

            sigma2_linear = sigma2_complex/2;

            CRB_g = 4*sigma2_linear*kron(eye(I),inv(conj(S)*S.'));

            crlb_acc(snrIndex) = crlb_acc(snrIndex) + real(trace(CRB_g));

        end

        if mod(mc,50) == 0 || mc == MC
            fprintf('P=%d : Monte Carlo %d / %d\n',P,mc,MC);
        end

    end

    NMSE_GS(pIndex,:) = mseGS_acc./channelPower_acc;
    NMSE_GD(pIndex,:) = mseGD_acc./channelPower_acc;
    NMSE_CRLB(pIndex,:) = crlb_acc./channelPower_acc;

    refRatio_mean(pIndex,:) = refPower_acc./signalPlusNoisePower_acc;   % E{|B|^2}/E{|GS+N|^2}
    refMagRatio_mean(pIndex,:) = refMag_acc./signalPlusNoisePowerMag_acc;   % E{|B|}/E{|GS+N|}

    fprintf('\nP=%d empirical E|b|^2 / E|GS+n|^2\n',P);
    fprintf('------------------------------------------------------------\n');

    for snrIndex = 1:numSNR
        fprintf('P=%2d | SNR=%4d dB | E|b|^2/E|GS+n|^2 = %6.3f (%6.2f dB) | E|b|/E|GS+n| = %6.3f\n',P,SNR_dB(snrIndex),refRatio_mean(pIndex,snrIndex),10*log10(refRatio_mean(pIndex,snrIndex)),refMagRatio_mean(pIndex,snrIndex));
    end

    fprintf('\nResults for P = %d\n',P);
    fprintf('------------------------------------------------------------\n');

    for snrIndex = 1:numSNR
        fprintf('P=%2d | SNR=%4d dB | GS=%8.2f dB | GD=%8.2f dB | CRLB=%8.2f dB\n',P,SNR_dB(snrIndex),10*log10(NMSE_GS(pIndex,snrIndex)),10*log10(NMSE_GD(pIndex,snrIndex)),10*log10(NMSE_CRLB(pIndex,snrIndex)));
    end

end

%% ---- Plot ----
figure('Color','w','Position',[100 100 1100 650]);
hold on;
grid on;
box on;

plot(SNR_dB,10*log10(NMSE_GS(1,:)),'--d','LineWidth',1.8,'MarkerSize',7,'DisplayName','GS, P=10');
plot(SNR_dB,10*log10(NMSE_GD(1,:)),'--s','LineWidth',1.8,'MarkerSize',7,'DisplayName','GD, P=10');
plot(SNR_dB,10*log10(NMSE_CRLB(1,:)),'-o','LineWidth',1.8,'MarkerSize',7,'DisplayName','CRLB, P=10');

plot(SNR_dB,10*log10(NMSE_GS(2,:)),'--d','LineWidth',1.8,'MarkerSize',7,'DisplayName','GS, P=30');
plot(SNR_dB,10*log10(NMSE_GD(2,:)),'--s','LineWidth',1.8,'MarkerSize',7,'DisplayName','GD, P=30');
plot(SNR_dB,10*log10(NMSE_CRLB(2,:)),'-o','LineWidth',1.8,'MarkerSize',7,'DisplayName','CRLB, P=30');

xlabel('SNR [dB]','FontWeight','bold');
ylabel('NMSE [dB]','FontWeight','bold');
title('Fig.3 NMSE vs SNR, 1D antenna array','FontWeight','bold');
legend('Location','southwest');
set(gca,'FontSize',12,'LineWidth',1);

hold off;

saveas(gcf,'fig3.png');

%% ---- Save results ----
save('fig3.mat','SNR_dB','P_list','NMSE_GS','NMSE_GD','NMSE_CRLB','MC','I','K','refRatio_mean','refMagRatio_mean');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig3.mat\n');
fprintf('Plot saved to fig3.png\n');
fprintf('====================================================\n');
