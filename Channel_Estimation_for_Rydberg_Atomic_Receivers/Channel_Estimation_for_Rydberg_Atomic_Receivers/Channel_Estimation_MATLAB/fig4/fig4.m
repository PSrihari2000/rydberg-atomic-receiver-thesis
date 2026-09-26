clear; clc; close all;

%% ---- Physical constants ----
hbar = 6.626e-34 / (2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg_y = 1785.9 * q_charge * a0_bohr;  

%% ---- Parameters ----
MC = 100;

I1 = 8; I2 = 8;
K = 3;

P_list = [10 30];
SNR_dB = -5:5:30;
numSNR = length(SNR_dB);

GD_iterations = 500;
PGD_iterations = 300;
tol = 1e-8;  % Assumption

GD_init_var = 0.1; % Assumption to match PGD
PGD_init_var = 0.1; % Given in sec V

d1_over_lambda = 0.5; 
d2_over_lambda = 0.5; 

IJ = I1*I2;   % total antennas = 8*8 = 64

fprintf('\n');
fprintf('====================================================\n');
fprintf(' Fig.4 NMSE performance comparison w.r.t SNR under Rydberg atom-based 2D antenna array.\n');
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
    channelPower_acc = zeros(1,numSNR);
    refPower_acc = zeros(1,numSNR);
    signalPlusNoisePower_acc = zeros(1,numSNR);

    for mc = 1:MC

        % ---- Channel -- G(3) is K x IJ ----
        G3 = zeros(K,IJ);        % K x IJ
        Lk = zeros(K,1);         % K x 1
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

                pol = randn/sqrt(3);  
                phase_i1 = exp(-1j*(i1idx-1)*u_lk);       % I1 x 1
                phase_i2 = exp(-1j*(i2idx-1)*v_lk).';     % 1 x I2

                Gslice = Gslice + (mu_eg_y/hbar)*pol*alpha_lk*(phase_i1*phase_i2);   % I1xI2 , Eq.(16)

            end

            G3(k,:) = reshape(Gslice,1,IJ);  % Mode(3) unfolded 2D matrix

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
        S_b = (randn(1,P) + 1j*randn(1,P))/sqrt(2);   % 1xP, s_{b,p} ~ CN(0,1)-assumed to match transmitted symbols (not given in paper)

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

        B3 = zeros(P,IJ);   % P x IJ
        for p = 1:P
            B3(p,:) = (S_b(p)/hbar)*alpha_b*b_flat;  % PxIJ, reference signal matrix
        end

        Z3 = exp(-1j*angle(B3));         % P x IJ, reference phase matrix

        refPower = mean(abs(B3(:)).^2);  

        % ---- SNR sweep ----
        for snrIndex = 1:numSNR

            snrLinear = 10^(SNR_dB(snrIndex)/10);     
            sigma2_complex = signalPower/snrLinear;   
            N3 = sqrt(sigma2_complex/2)*(randn(P,IJ) + 1j*randn(P,IJ));   % P x IJ N~CN(0, Sigma^2)

            signalPlusNoisePower = mean(abs(A_signal3(:)+N3(:)).^2);   

            refPower_acc(snrIndex) = refPower_acc(snrIndex) + refPower;
            signalPlusNoisePower_acc(snrIndex) = signalPlusNoisePower_acc(snrIndex) + signalPlusNoisePower;

            Y3 = abs(A_signal3 + B3 + N3);   % P x IJ -- nonlinear observation
            absB3 = abs(B3);                  % P x IJ

            %% ---- GD: Eq.(22), gradient Eq.(24) ----
            Ghat_GD = sqrt(GD_init_var/2)*(randn(K,IJ)+1j*randn(K,IJ));   % K x IJ

            objGD = @(X) norm(Y3 - absB3 - real(Z3.*(S.'*X)),'fro')^2;    

            curCostGD = objGD(Ghat_GD);  

            for gdIter = 1:GD_iterations

                residual = Y3 - absB3 - real(Z3.*(S.'*Ghat_GD));   % P x IJ
                gradient = -conj(S)*(residual.*conj(Z3));           % (KxP)*(PxIJ) = K x IJ, Eq.(24)

                eta = 1;   % initialize step size

                while eta > 1e-10

                    Gtrial = Ghat_GD - eta*gradient;   % K x IJ

                    if objGD(Gtrial) <= curCostGD, break; end

                    eta = eta/2;

                end

                Gnew = Ghat_GD - eta*gradient;    % K x IJ
                newCost = objGD(Gnew);            

                relChange = norm(Ghat_GD-Gnew,'fro')/(norm(Ghat_GD,'fro'));   

                Ghat_GD = Gnew;
                curCostGD = newCost;

                if relChange < tol, break; end

            end

            %% ---- PGD: Eq.(23)-(27), gradient step (Eq.24) + per-user SVD rank projection ----
            Ghat_PGD = sqrt(PGD_init_var/2)*(randn(K,IJ)+1j*randn(K,IJ));   % K x IJ

            objPGD = @(X) norm(Y3 - absB3 - real(Z3.*(S.'*X)),'fro')^2;

            curCostPGD = objPGD(Ghat_PGD);   % scalar

            for pgdIter = 1:PGD_iterations

                residual = Y3 - absB3 - real(Z3.*(S.'*Ghat_PGD));   % P x IJ
                gradient = -conj(S)*(residual.*conj(Z3));            % K x IJ
                
                eta = 1;   % initialization

                while eta > 1e-10

                    Mtrial = Ghat_PGD - eta*gradient;   % K x IJ

                    if objPGD(Mtrial) <= curCostPGD, break; end

                    eta = eta/2;

                end

                M = Ghat_PGD - eta*gradient;   % K x IJ 

                Gnew = zeros(K,IJ);   % K x IJ, empty matrix for the projected result

                for k = 1:K

                    Mk = reshape(M(k,:),I1,I2);   % I1 x I2 -- mat([M]_{k,:}), Eq.(26)

                    [U,Sig,V] = svd(Mk);          % U: I1xI1, Sig: I1xI2, V: I2xI2

                    r = Lk(k);                    % scalar, number of propagation paths for user k

                    Uk = U(:,1:r);                % I1xr
                    Sigk = Sig(1:r,1:r);          % rxr
                    Vk = V(:,1:r);                % I2xr

                    Xk = Uk*Sigk*Vk';             % I1 x I2 -- Low rank matrix

                    Gnew(k,:) = reshape(Xk,1,IJ);   % K x IJ -- Eq.(27)  

                end

                relChange = norm(Ghat_PGD-Gnew,'fro')/(norm(Ghat_PGD,'fro'));  

                Ghat_PGD = Gnew;
                curCostPGD = objPGD(Ghat_PGD);

                if relChange < tol, break; end

            end

            mseGD_acc(snrIndex) = mseGD_acc(snrIndex) + norm(G3-Ghat_GD,'fro')^2;
            msePGD_acc(snrIndex) = msePGD_acc(snrIndex) + norm(G3-Ghat_PGD,'fro')^2;
            channelPower_acc(snrIndex) = channelPower_acc(snrIndex) + norm(G3,'fro')^2;

            sigma2_linear = sigma2_complex/2;                                    

            CRB_full = 4*sigma2_linear*kron(eye(IJ), inv(conj(S)*S.'));         % (IJ*K) x (IJ*K), Eq.(31)

            crlb_acc(snrIndex) = crlb_acc(snrIndex) + real(trace(CRB_full));   

        end

        if mod(mc,25)==0 || mc==MC
            fprintf('P=%d : Monte Carlo %d / %d\n',P,mc,MC);
        end

    end

    NMSE_GD(pIndex,:) = mseGD_acc./channelPower_acc;
    NMSE_PGD(pIndex,:) = msePGD_acc./channelPower_acc;
    NMSE_CRLB(pIndex,:) = crlb_acc./channelPower_acc;
    refRatio_mean(pIndex,:) = refPower_acc./signalPlusNoisePower_acc;

    fprintf('\nP=%d empirical E|b|^2 / E|a+n|^2\n',P);
    fprintf('------------------------------------------------------------\n');

    for snrIndex = 1:numSNR

        fprintf('P=%2d | SNR=%4d dB | E|b|^2/E|a+n|^2 = %6.2f dB\n',P,SNR_dB(snrIndex),10*log10(refRatio_mean(pIndex,snrIndex)));

    end

    fprintf('\nResults for P = %d\n',P);
    fprintf('------------------------------------------------------------\n');

    for snrIndex = 1:numSNR

        fprintf('P=%2d | SNR=%4d dB | GD=%8.2f dB | PGD=%8.2f dB | CRLB=%8.2f dB\n',P,SNR_dB(snrIndex),10*log10(NMSE_GD(pIndex,snrIndex)),10*log10(NMSE_PGD(pIndex,snrIndex)),10*log10(NMSE_CRLB(pIndex,snrIndex)));

    end

end

%% ---- Plot ----
figure('Color','w','Position',[100 100 1100 650]);
hold on; grid on; box on;

plot(SNR_dB,10*log10(NMSE_GD(1,:)),'--s','LineWidth',1.8,'MarkerSize',7,'DisplayName','GD, P=10');
plot(SNR_dB,10*log10(NMSE_PGD(1,:)),'-^','LineWidth',1.8,'MarkerSize',7,'DisplayName','PGD, P=10');
plot(SNR_dB,10*log10(NMSE_CRLB(1,:)),'-o','LineWidth',1.8,'MarkerSize',7,'DisplayName','CRLB, P=10');

plot(SNR_dB,10*log10(NMSE_GD(2,:)),'--s','LineWidth',1.8,'MarkerSize',7,'DisplayName','GD, P=30');
plot(SNR_dB,10*log10(NMSE_PGD(2,:)),'-^','LineWidth',1.8,'MarkerSize',7,'DisplayName','PGD, P=30');
plot(SNR_dB,10*log10(NMSE_CRLB(2,:)),'-o','LineWidth',1.8,'MarkerSize',7,'DisplayName','CRLB, P=30');

xlabel('SNR [dB]','FontWeight','bold');
ylabel('NMSE [dB]','FontWeight','bold');
title('fig4 NMSE vs SNR, 2D antenna array','FontWeight','bold');
legend('Location','southwest');
set(gca,'FontSize',12,'LineWidth',1);

hold off;

saveas(gcf,'fig4.png');

%% ---- Save results ----
save('fig4.mat','SNR_dB','P_list','NMSE_GD','NMSE_PGD','NMSE_CRLB','MC','I1','I2','K','refRatio_mean');

fprintf('\n====================================================\n');
fprintf('Simulation completed. Results saved to fig4.mat\n');
fprintf('Plot saved to fig4.png\n');
fprintf('====================================================\n');