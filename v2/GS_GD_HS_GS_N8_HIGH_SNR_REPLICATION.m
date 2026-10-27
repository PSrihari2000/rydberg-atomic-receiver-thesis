%% Independent high-SNR replication: GS, GD, and adaptive soft-rank HS-GS, N = 8
% Generates fresh paired channel, pilot, reference, and noise samples.
% The adaptive soft-rank rule is frozen from the previous experiment:
% each sample selects a threshold using held-out pilot magnitude prediction,
% then refits HS-GS with all pilots. Channel truth is used only for NMSE.
% This focused replication tests SNR = 20, 25, and 30 dB with 2000 samples
% per P/SNR point. No prior validation or result file is loaded.
%
% Run in MATLAB:
% cd('D:\MTech_Thesis_Results\v2')
% GS_GD_HS_GS_N8_HIGH_SNR_REPLICATION

clear;
clc;
close all;
rng(1);

%% -------------------- SIMULATION PARAMETERS --------------------
MC = 2000;
N = 8;
K = 3;
P_list = [10 30];
num_P = length(P_list);
SNR_dB = [20 25 30];
num_SNR = length(SNR_dB);
RSR_dB = 25;
p_H = 3;
maximum_iterations = 5000;
change_tolerance = 1e-10;
d_over_lambda = 0.5;
G0_var = 0.1;
workers_if_starting_pool = 6;
samples_per_checkpoint = 20;
test_seed_base = 3200000000;
soft_shrink_multiplier_list = [0 0.25 0.5 1 2];
pilot_training_fraction = 0.70;
threshold_selection_updates = 25;
number_of_pilot_splits = 3;
bootstrap_replicates = 5000;

%% -------------------- PHYSICAL CONSTANTS --------------------
hbar = 6.62607015e-34/(2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0;1785.9*q_charge*a0_bohr;0];

%% -------------------- OUTPUT AND CHECKPOINT FILES --------------------
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~isfolder(result_folder)
    mkdir(result_folder);
end
checkpoint_file = fullfile(result_folder,'GS_GD_HS_GS_N8_high_snr_replication_checkpoint.mat');
result_file = fullfile(result_folder,'GS_GD_HS_GS_N8_high_snr_replication_results.mat');
table_file = fullfile(result_folder,'GS_GD_HS_GS_N8_high_snr_replication_table.csv');
figure_png = fullfile(result_folder,'GS_GD_HS_GS_N8_high_snr_replication.png');
figure_file = fullfile(result_folder,'GS_GD_HS_GS_N8_high_snr_replication.fig');
if isfile(result_file) || isfile(table_file)
    error('Results already exist. Rename or inspect them before starting a new run.');
end

fprintf('Fresh paired high-SNR replication: GS, GD, and HS-GS.\n');
fprintf('N = %d, K = %d, P = [%s], RSR = %g dB, MC = %d.\n',N,K,num2str(P_list),RSR_dB,MC);
fprintf('SNR points = [%s] dB; threshold candidates = [%s].\n',num2str(SNR_dB),num2str(soft_shrink_multiplier_list));
fprintf('Selection rule is frozen; no NMSE-based tuning is performed.\n');

%% -------------------- STORAGE --------------------
GS_error_sum = zeros(num_P,num_SNR);
GD_error_sum = zeros(num_P,num_SNR);
HS_GS_error_sum = zeros(num_P,num_SNR);
G_energy_sum = zeros(num_P,num_SNR);
selected_multiplier_sum = zeros(num_P,num_SNR);
per_sample_error = nan(num_P,num_SNR,MC,3);
per_sample_energy = nan(num_P,num_SNR,MC);
completed_samples = false(num_P,num_SNR,MC);

run_configuration = struct('MC',MC,'N',N,'K',K,'P_list',P_list,'SNR_dB',SNR_dB,'RSR_dB',RSR_dB,'p_H',p_H,'maximum_iterations',maximum_iterations,'change_tolerance',change_tolerance,'d_over_lambda',d_over_lambda,'G0_var',G0_var,'soft_shrink_multiplier_list',soft_shrink_multiplier_list,'pilot_training_fraction',pilot_training_fraction,'threshold_selection_updates',threshold_selection_updates,'number_of_pilot_splits',number_of_pilot_splits,'bootstrap_replicates',bootstrap_replicates,'test_seed_base',test_seed_base);

if isfile(checkpoint_file)
    saved = load(checkpoint_file);
    if ~isequal(saved.run_configuration,run_configuration)
        error('Checkpoint settings differ from this script. Preserve the checkpoint.');
    end
    GS_error_sum = saved.GS_error_sum;
    GD_error_sum = saved.GD_error_sum;
    HS_GS_error_sum = saved.HS_GS_error_sum;
    G_energy_sum = saved.G_energy_sum;
    selected_multiplier_sum = saved.selected_multiplier_sum;
    per_sample_error = saved.per_sample_error;
    per_sample_energy = saved.per_sample_energy;
    completed_samples = saved.completed_samples;
    fprintf('Resuming checkpoint: %d of %d samples complete.\n',nnz(completed_samples),numel(completed_samples));
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('local',workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('Parallel pool has %d workers.\n',pool.NumWorkers);

%% -------------------- MONTE CARLO SIMULATION --------------------
for P_index = 1:num_P
    P = P_list(P_index);
    fprintf('\nStarting P = %d.\n',P);
    for SNR_index = 1:num_SNR
        current_SNR_dB = SNR_dB(SNR_index);
        fprintf('Starting P = %d, SNR = %g dB.\n',P,current_SNR_dB);
        for block_start = 1:samples_per_checkpoint:MC
            sample_indices = block_start:min(block_start+samples_per_checkpoint-1,MC);
            sample_indices = sample_indices(~reshape(completed_samples(P_index,SNR_index,sample_indices),1,[]));
            if isempty(sample_indices)
                continue;
            end

            block_GS_error = zeros(numel(sample_indices),1);
            block_GD_error = zeros(numel(sample_indices),1);
            block_HS_GS_error = zeros(numel(sample_indices),1);
            block_G_energy = zeros(numel(sample_indices),1);
            block_selected_multiplier = zeros(numel(sample_indices),1);
            block_sample_error = zeros(numel(sample_indices),3);

            parfor block_index = 1:numel(sample_indices)
                mc = sample_indices(block_index);
                sample_seed = test_seed_base+1000000*P+10000*SNR_index+mc;
                rng(sample_seed,'twister');

                %% Generate true channel G, N-by-K
                G = zeros(N,K);
                for user = 1:K
                    number_of_paths = randi([3 7]);
                    for path_index = 1:number_of_paths
                        alpha = (randn+1j*randn)/sqrt(2);
                        theta = 2*pi*rand;
                        phase_step = 2*pi*d_over_lambda*cos(theta);
                        polarization = sqrt(1/3)*randn(3,1);
                        for antenna = 1:N
                            G(antenna,user) = G(antenna,user)+(mu_eg.'*polarization/hbar)*alpha*exp(-1j*(antenna-1)*phase_step);
                        end
                    end
                end

                %% Generate pilots S, K-by-P
                S = (randn(K,P)+1j*randn(K,P))/sqrt(2);

                %% Generate known reference B, N-by-P, and scale to the requested RSR
                alpha_reference = sqrt(10/2)*(randn+1j*randn);
                theta_reference = 2*pi*rand;
                reference_phase_step = 2*pi*d_over_lambda*cos(theta_reference);
                reference_polarization = sqrt(1/3)*randn(3,1);
                reference_vector = zeros(N,1);
                for antenna = 1:N
                    reference_vector(antenna) = (mu_eg.'*reference_polarization/hbar)*alpha_reference*exp(-1j*(antenna-1)*reference_phase_step);
                end
                B = reference_vector*ones(1,P);
                average_single_user_power = 0;
                for user = 1:K
                    average_single_user_power = average_single_user_power+mean(abs(G(:,user)*S(user,:)).^2,'all')/K;
                end
                target_reference_power = 10^(RSR_dB/10)*average_single_user_power;
                B = B*sqrt(target_reference_power/mean(abs(B).^2,'all'));

                %% Generate noisy magnitude measurements Y, N-by-P
                signal_power = mean(abs(G*S).^2,'all');
                sigma2 = signal_power/10^(current_SNR_dB/10);
                W = sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P));
                Y = abs(G*S+B+W);

                %% Common spectral initialization used by the iterative estimators
                G_initial = zeros(N,K);
                for antenna = 1:N
                    y_i = Y(antenna,:).';
                    b_i = B(antenna,:).';
                    A_bar_i = [S.',b_i];
                    M_i = A_bar_i'*diag(y_i)*A_bar_i;
                    [V_i,D_i] = eig(M_i);
                    [~,principal_index] = max(real(diag(D_i)));
                    v_i = V_i(:,principal_index);
                    v_i = v_i/max(norm(v_i),eps);
                    scale_i = (abs(A_bar_i*v_i).'*y_i)/max(norm(A_bar_i*v_i)^2,eps);
                    G_initial(antenna,:) = (scale_i*exp(-1j*angle(v_i(K+1)))*v_i(1:K)).';
                end

                %% Algorithm 1: GS
                G_hat_GS = G_initial;
                right_inverse = S'/(S*S');
                for iteration = 1:maximum_iterations
                    previous_G = G_hat_GS;
                    G_hat_GS = (Y.*exp(1j*angle(G_hat_GS*S+B))-B)*right_inverse;
                    relative_change = norm(G_hat_GS-previous_G,'fro')^2/max(norm(G_hat_GS,'fro')^2,eps);
                    if relative_change < change_tolerance
                        break;
                    end
                end

                %% Algorithm 2: linearized GD
                rng(mod(sample_seed+720000,2^32-1),'twister');
                G_hat_GD = sqrt(G0_var/2)*(randn(N,K)+1j*randn(N,K));
                phase_reference = exp(-1j*angle(B));
                GD_step_size = 1/max(real(eig(S*S')));
                for iteration = 1:maximum_iterations
                    previous_G = G_hat_GD;
                    G_hat_GD = G_hat_GD+GD_step_size*((Y-abs(B)-real(phase_reference.*(G_hat_GD*S))).*conj(phase_reference))*S';
                    relative_change = norm(G_hat_GD-previous_G,'fro')^2/max(norm(G_hat_GD,'fro')^2,eps);
                    if relative_change < change_tolerance
                        break;
                    end
                end

                %% Select this sample's threshold using held-out pilots only
                candidate_prediction_error_sum = zeros(1,numel(soft_shrink_multiplier_list));
                for split_index = 1:number_of_pilot_splits
                    rng(mod(sample_seed+360000+1000*split_index,2^32-1),'twister');
                    pilot_order = randperm(P);
                    number_of_training_pilots = max(K+1,round(pilot_training_fraction*P));
                    number_of_training_pilots = min(number_of_training_pilots,P-1);
                    training_indices = pilot_order(1:number_of_training_pilots);
                    heldout_indices = pilot_order(number_of_training_pilots+1:P);
                    Y_training = Y(:,training_indices);
                    S_training = S(:,training_indices);
                    B_training = B(:,training_indices);
                    Y_heldout = Y(:,heldout_indices);
                    S_heldout = S(:,heldout_indices);
                    B_heldout = B(:,heldout_indices);
                    right_inverse_training = S_training'/(S_training*S_training');
                    G_candidate_initial = zeros(N,K);
                    for antenna = 1:N
                        y_i = Y_training(antenna,:).';
                        b_i = B_training(antenna,:).';
                        A_bar_i = [S_training.',b_i];
                        M_i = A_bar_i'*diag(y_i)*A_bar_i;
                        [V_i,D_i] = eig(M_i);
                        [~,principal_index] = max(real(diag(D_i)));
                        v_i = V_i(:,principal_index);
                        v_i = v_i/max(norm(v_i),eps);
                        scale_i = (abs(A_bar_i*v_i).'*y_i)/max(norm(A_bar_i*v_i)^2,eps);
                        G_candidate_initial(antenna,:) = (scale_i*exp(-1j*angle(v_i(K+1)))*v_i(1:K)).';
                    end
                    candidate_prediction_error = inf(1,numel(soft_shrink_multiplier_list));
                    training_noise_scale = sqrt(sigma2)*norm(right_inverse_training,'fro')*sqrt(max(N-p_H,p_H+1)/K);
                    for candidate_index = 1:numel(soft_shrink_multiplier_list)
                        candidate_multiplier = soft_shrink_multiplier_list(candidate_index);
                        candidate_threshold = candidate_multiplier*training_noise_scale;
                        G_candidate = G_candidate_initial;
                        for candidate_update = 1:threshold_selection_updates
                            candidate_field = G_candidate*S_training+B_training;
                            candidate_kappa = 2/max(sigma2,eps)*Y_training.*abs(candidate_field);
                            candidate_ratio = besseli(1,candidate_kappa,1)./max(besseli(0,candidate_kappa,1),realmin);
                            G_candidate = (Y_training.*exp(1j*angle(candidate_field)).*candidate_ratio-B_training)*right_inverse_training;
                            if candidate_multiplier > 0
                                for user = 1:K
                                    candidate_channel = G_candidate(:,user);
                                    H = zeros(N-p_H,p_H+1);
                                    for row = 1:N-p_H
                                        H(row,:) = candidate_channel(row:row+p_H).';
                                    end
                                    [U_H,Sigma_H,V_H] = svd(H,'econ');
                                    shrunk_singular_values = max(diag(Sigma_H)-candidate_threshold,0);
                                    H_soft = U_H*diag(shrunk_singular_values)*V_H';
                                    reconstructed_channel = zeros(N,1);
                                    copy_count = zeros(N,1);
                                    for row = 1:N-p_H
                                        for column = 1:p_H+1
                                            index = row+column-1;
                                            reconstructed_channel(index) = reconstructed_channel(index)+H_soft(row,column);
                                            copy_count(index) = copy_count(index)+1;
                                        end
                                    end
                                    G_candidate(:,user) = reconstructed_channel./copy_count;
                                end
                            end
                        end
                        candidate_prediction = abs(G_candidate*S_heldout+B_heldout);
                        candidate_prediction_error(candidate_index) = mean((Y_heldout-candidate_prediction).^2,'all');
                    end
                    candidate_prediction_error_sum = candidate_prediction_error_sum+candidate_prediction_error;
                end
                candidate_prediction_error = candidate_prediction_error_sum/number_of_pilot_splits;
                [~,best_candidate_index] = min(candidate_prediction_error);
                selected_multiplier = soft_shrink_multiplier_list(best_candidate_index);
                block_selected_multiplier(block_index) = selected_multiplier;

                %% Refit HS-GS using all pilots and the selected soft threshold
                G_hat_HS_GS = G_initial;
                full_noise_scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
                full_soft_threshold = selected_multiplier*full_noise_scale;
                for iteration = 1:maximum_iterations
                    previous_G = G_hat_HS_GS;
                    field = G_hat_HS_GS*S+B;
                    kappa = 2/max(sigma2,eps)*Y.*abs(field);
                    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
                    G_hat_HS_GS = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
                    if selected_multiplier > 0
                        for user = 1:K
                            channel_vector = G_hat_HS_GS(:,user);
                            H = zeros(N-p_H,p_H+1);
                            for row = 1:N-p_H
                                H(row,:) = channel_vector(row:row+p_H).';
                            end
                            [U_H,Sigma_H,V_H] = svd(H,'econ');
                            shrunk_singular_values = max(diag(Sigma_H)-full_soft_threshold,0);
                            H_soft = U_H*diag(shrunk_singular_values)*V_H';
                            reconstructed_channel = zeros(N,1);
                            copy_count = zeros(N,1);
                            for row = 1:N-p_H
                                for column = 1:p_H+1
                                    index = row+column-1;
                                    reconstructed_channel(index) = reconstructed_channel(index)+H_soft(row,column);
                                    copy_count(index) = copy_count(index)+1;
                                end
                            end
                            G_hat_HS_GS(:,user) = reconstructed_channel./copy_count;
                        end
                    end
                    relative_change = norm(G_hat_HS_GS-previous_G,'fro')^2/max(norm(G_hat_HS_GS,'fro')^2,eps);
                    if relative_change < change_tolerance
                        break;
                    end
                end

                %% Save this sample's squared errors for the paired comparison
                block_GS_error(block_index) = norm(G_hat_GS-G,'fro')^2;
                block_GD_error(block_index) = norm(G_hat_GD-G,'fro')^2;
                block_HS_GS_error(block_index) = norm(G_hat_HS_GS-G,'fro')^2;
                block_G_energy(block_index) = norm(G,'fro')^2;
                block_sample_error(block_index,:) = [block_GS_error(block_index),block_GD_error(block_index),block_HS_GS_error(block_index)];
            end

            GS_error_sum(P_index,SNR_index) = GS_error_sum(P_index,SNR_index)+sum(block_GS_error);
            GD_error_sum(P_index,SNR_index) = GD_error_sum(P_index,SNR_index)+sum(block_GD_error);
            HS_GS_error_sum(P_index,SNR_index) = HS_GS_error_sum(P_index,SNR_index)+sum(block_HS_GS_error);
            G_energy_sum(P_index,SNR_index) = G_energy_sum(P_index,SNR_index)+sum(block_G_energy);
            selected_multiplier_sum(P_index,SNR_index) = selected_multiplier_sum(P_index,SNR_index)+sum(block_selected_multiplier);
            per_sample_error(P_index,SNR_index,sample_indices,:) = reshape(block_sample_error,[1 1 numel(sample_indices) 3]);
            per_sample_energy(P_index,SNR_index,sample_indices) = block_G_energy;
            completed_samples(P_index,SNR_index,sample_indices) = true;
            save(checkpoint_file,'run_configuration','GS_error_sum','GD_error_sum','HS_GS_error_sum','G_energy_sum','selected_multiplier_sum','per_sample_error','per_sample_energy','completed_samples','-v7.3');
            fprintf('%d/%d samples done for P=%d, SNR=%g dB.\n',nnz(completed_samples(P_index,SNR_index,:)),MC,P,current_SNR_dB);
        end
    end
end

%% -------------------- COMPUTE NMSE, PLOT, AND SAVE --------------------
if ~all(completed_samples,'all')
    error('Some samples are incomplete. The checkpoint has been retained.');
end
GS_NMSE_dB = 10*log10(GS_error_sum./G_energy_sum);
GD_NMSE_dB = 10*log10(GD_error_sum./G_energy_sum);
HS_GS_NMSE_dB = 10*log10(HS_GS_error_sum./G_energy_sum);
gain_over_GS_dB = GS_NMSE_dB-HS_GS_NMSE_dB;
gain_over_GD_dB = GD_NMSE_dB-HS_GS_NMSE_dB;
gain_GS_CI_low = zeros(num_P,num_SNR);
gain_GS_CI_high = zeros(num_P,num_SNR);
gain_GD_CI_low = zeros(num_P,num_SNR);
gain_GD_CI_high = zeros(num_P,num_SNR);
rng(620000000,'twister');
for P_index = 1:num_P
    for SNR_index = 1:num_SNR
        point_errors = reshape(per_sample_error(P_index,SNR_index,:,:),MC,3);
        bootstrap_GS_gain = zeros(bootstrap_replicates,1);
        bootstrap_GD_gain = zeros(bootstrap_replicates,1);
        for bootstrap_index = 1:bootstrap_replicates
            sample_indices = randi(MC,MC,1);
            bootstrap_GS_gain(bootstrap_index) = 10*log10(sum(point_errors(sample_indices,1))/sum(point_errors(sample_indices,3)));
            bootstrap_GD_gain(bootstrap_index) = 10*log10(sum(point_errors(sample_indices,2))/sum(point_errors(sample_indices,3)));
        end
        bootstrap_GS_gain = sort(bootstrap_GS_gain);
        bootstrap_GD_gain = sort(bootstrap_GD_gain);
        gain_GS_CI_low(P_index,SNR_index) = bootstrap_GS_gain(max(1,round(0.025*bootstrap_replicates)));
        gain_GS_CI_high(P_index,SNR_index) = bootstrap_GS_gain(min(bootstrap_replicates,round(0.975*bootstrap_replicates)));
        gain_GD_CI_low(P_index,SNR_index) = bootstrap_GD_gain(max(1,round(0.025*bootstrap_replicates)));
        gain_GD_CI_high(P_index,SNR_index) = bootstrap_GD_gain(min(bootstrap_replicates,round(0.975*bootstrap_replicates)));
    end
end

figure_handle = figure('Color','w','Name','High-SNR GS, GD, and HS-GS replication');
tiledlayout(1,num_P,'TileSpacing','compact','Padding','compact');
for P_index = 1:num_P
    nexttile;
    hold on;
    plot(SNR_dB,GS_NMSE_dB(P_index,:),'-o','LineWidth',1.5,'MarkerSize',5,'DisplayName','GS');
    plot(SNR_dB,GD_NMSE_dB(P_index,:),'-s','LineWidth',1.5,'MarkerSize',5,'DisplayName','GD');
    plot(SNR_dB,HS_GS_NMSE_dB(P_index,:),'-^','LineWidth',1.5,'MarkerSize',5,'DisplayName','HS-GS');
    grid on;
    xlabel('SNR (dB)');
    ylabel('NMSE (dB)');
    title(sprintf('N = %d, P = %d',N,P_list(P_index)));
    legend('Location','best');
end
sgtitle(sprintf('Fresh paired high-SNR replication, RSR = %g dB',RSR_dB));

comparison_table = table();
for P_index = 1:num_P
    P_column = repmat(P_list(P_index),num_SNR,1);
    mean_selected_multiplier = selected_multiplier_sum(P_index,:).'/MC;
    point_table = table(P_column,SNR_dB(:),GS_NMSE_dB(P_index,:).',GD_NMSE_dB(P_index,:).',HS_GS_NMSE_dB(P_index,:).',gain_over_GS_dB(P_index,:).',gain_GS_CI_low(P_index,:).',gain_GS_CI_high(P_index,:).',gain_over_GD_dB(P_index,:).',gain_GD_CI_low(P_index,:).',gain_GD_CI_high(P_index,:).',mean_selected_multiplier,'VariableNames',{'P','SNR_dB','GS_NMSE_dB','GD_NMSE_dB','HS_GS_NMSE_dB','Gain_over_GS_dB','Gain_GS_CI_low','Gain_GS_CI_high','Gain_over_GD_dB','Gain_GD_CI_low','Gain_GD_CI_high','Mean_selected_multiplier'});
    comparison_table = [comparison_table;point_table];
end
writetable(comparison_table,table_file);
save(result_file,'run_configuration','comparison_table','GS_error_sum','GD_error_sum','HS_GS_error_sum','G_energy_sum','selected_multiplier_sum','per_sample_error','per_sample_energy','gain_over_GS_dB','gain_over_GD_dB','gain_GS_CI_low','gain_GS_CI_high','gain_GD_CI_low','gain_GD_CI_high','GS_NMSE_dB','GD_NMSE_dB','HS_GS_NMSE_dB','-v7.3');
saveas(figure_handle,figure_png);
savefig(figure_handle,figure_file);
disp(comparison_table);
fprintf('Figure saved to: %s\n',figure_png);
fprintf('Table saved to: %s\n',table_file);
