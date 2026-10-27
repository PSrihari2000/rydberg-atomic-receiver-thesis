%% Paired biased-GS versus EM-GS channel-estimation comparison
% N = 8, K = 3, P = 10 or 30, RSR = 25 dB.
% Each method sees exactly the same channel, pilots, reference, and noise.
% Both methods use the same spectral initialization and convergence rule.
% No validation or prior result files are loaded.
% The paired bootstrap interval is for NMSE_GS_dB - NMSE_EM_GS_dB;
% a positive gain means EM-GS has lower NMSE.

clear;
clc;
close all;

%% -------------------- EXPERIMENT SETTINGS --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_dB = -5:5:30;
RSR_dB = 25;
MC = 500;
d_over_lambda = 0.5;
maximum_iterations = 5000;
change_tolerance = 1e-10;
samples_per_checkpoint = 20;
bootstrap_replicates = 2000;
workers_if_starting_pool = 4;
sample_seed_base = 5100000000;

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
checkpoint_file = fullfile(result_folder,'GS_EM_GS_N8_RSR25_paired_checkpoint.mat');
result_file = fullfile(result_folder,'GS_EM_GS_N8_RSR25_paired_results.mat');
table_file = fullfile(result_folder,'GS_EM_GS_N8_RSR25_paired_table.csv');
figure_png = fullfile(result_folder,'GS_EM_GS_N8_RSR25_paired_nmse.png');
figure_file = fullfile(result_folder,'GS_EM_GS_N8_RSR25_paired_nmse.fig');
if isfile(result_file) || isfile(table_file)
    error('Final outputs already exist. Inspect or rename them before starting a new run.');
end

run_configuration = struct('N',N,'K',K,'P_list',P_list,'SNR_dB',SNR_dB,'RSR_dB',RSR_dB,'MC',MC,'d_over_lambda',d_over_lambda,'maximum_iterations',maximum_iterations,'change_tolerance',change_tolerance,'samples_per_checkpoint',samples_per_checkpoint,'bootstrap_replicates',bootstrap_replicates,'sample_seed_base',sample_seed_base);

number_of_P = numel(P_list);
number_of_SNR = numel(SNR_dB);
sample_squared_error = nan(number_of_P,number_of_SNR,MC,2);
sample_channel_energy = nan(number_of_P,number_of_SNR,MC);
sample_iterations = zeros(number_of_P,number_of_SNR,MC,2);
sample_converged = false(number_of_P,number_of_SNR,MC,2);
sample_completed = false(number_of_P,number_of_SNR,MC);

if isfile(checkpoint_file)
    C = load(checkpoint_file);
    if ~isequal(C.run_configuration,run_configuration)
        error('Checkpoint settings differ from this script. Preserve the checkpoint and use matching settings.');
    end
    sample_squared_error = C.sample_squared_error;
    sample_channel_energy = C.sample_channel_energy;
    sample_iterations = C.sample_iterations;
    sample_converged = C.sample_converged;
    sample_completed = C.sample_completed;
    fprintf('Resuming checkpoint: %d of %d samples complete.\n',nnz(sample_completed),numel(sample_completed));
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('local',workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('Paired GS/EM-GS comparison started with %d workers.\n',pool.NumWorkers);
fprintf('N = %d, P = [%s], RSR = %g dB, %d samples per P/SNR point.\n',N,num2str(P_list),RSR_dB,MC);

%% -------------------- PAIRED MONTE CARLO SIMULATION --------------------
for P_index = 1:number_of_P
    P = P_list(P_index);
    for SNR_index = 1:number_of_SNR
        current_SNR_dB = SNR_dB(SNR_index);
        point_timer = tic;
        fprintf('\nStarting P = %d, SNR = %g dB.\n',P,current_SNR_dB);

        for block_start = 1:samples_per_checkpoint:MC
            block_indices = block_start:min(block_start+samples_per_checkpoint-1,MC);
            pending_indices = block_indices(~reshape(sample_completed(P_index,SNR_index,block_indices),1,[]));
            if isempty(pending_indices)
                continue;
            end

            block_error = zeros(numel(pending_indices),2);
            block_energy = zeros(numel(pending_indices),1);
            block_iterations = zeros(numel(pending_indices),2);
            block_converged = false(numel(pending_indices),2);

            parfor q = 1:numel(pending_indices)
                sample_index = pending_indices(q);
                sample_seed = sample_seed_base+1000000*P+10000*SNR_index+sample_index;
                rng(mod(double(sample_seed),2^32-1),'twister');

                %% Generate the channel realization G, size N-by-K
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

                %% Generate pilots S and known reference B
                S = (randn(K,P)+1j*randn(K,P))/sqrt(2);
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

                %% Generate one noisy magnitude observation shared by both methods
                signal_power = mean(abs(G*S).^2,'all');
                sigma2 = signal_power/10^(current_SNR_dB/10);
                W = sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P));
                Y = abs(G*S+B+W);

                %% Common spectral initialization and common LS matrix
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
                right_inverse = S'/(S*S');

                %% Biased GS
                G_hat_GS = G_initial;
                GS_converged = false;
                GS_iterations = maximum_iterations;
                for iteration = 1:maximum_iterations
                    previous_G = G_hat_GS;
                    G_hat_GS = (Y.*exp(1j*angle(G_hat_GS*S+B))-B)*right_inverse;
                    relative_change = norm(G_hat_GS-previous_G,'fro')^2/max(norm(G_hat_GS,'fro')^2,eps);
                    if relative_change < change_tolerance
                        GS_iterations = iteration;
                        GS_converged = true;
                        break;
                    end
                end

                %% EM-GS
                G_hat_EM_GS = G_initial;
                EM_GS_converged = false;
                EM_GS_iterations = maximum_iterations;
                for iteration = 1:maximum_iterations
                    previous_G = G_hat_EM_GS;
                    predicted_field = G_hat_EM_GS*S+B;
                    kappa = (2/max(sigma2,eps))*Y.*abs(predicted_field);
                    bessel_ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
                    G_hat_EM_GS = (Y.*exp(1j*angle(predicted_field)).*bessel_ratio-B)*right_inverse;
                    relative_change = norm(G_hat_EM_GS-previous_G,'fro')^2/max(norm(G_hat_EM_GS,'fro')^2,eps);
                    if relative_change < change_tolerance
                        EM_GS_iterations = iteration;
                        EM_GS_converged = true;
                        break;
                    end
                end

                %% Save paired errors for this identical sample
                block_error(q,:) = [norm(G_hat_GS-G,'fro')^2,norm(G_hat_EM_GS-G,'fro')^2];
                block_energy(q) = norm(G,'fro')^2;
                block_iterations(q,:) = [GS_iterations,EM_GS_iterations];
                block_converged(q,:) = [GS_converged,EM_GS_converged];
            end

            sample_squared_error(P_index,SNR_index,pending_indices,:) = reshape(block_error,[1 1 numel(pending_indices) 2]);
            sample_channel_energy(P_index,SNR_index,pending_indices) = block_energy;
            sample_iterations(P_index,SNR_index,pending_indices,:) = reshape(block_iterations,[1 1 numel(pending_indices) 2]);
            sample_converged(P_index,SNR_index,pending_indices,:) = reshape(block_converged,[1 1 numel(pending_indices) 2]);
            sample_completed(P_index,SNR_index,pending_indices) = true;

            save(checkpoint_file,'run_configuration','sample_squared_error','sample_channel_energy','sample_iterations','sample_converged','sample_completed','-v7.3');
            fprintf('%d/%d samples complete; %.1f seconds elapsed for this point.\n',nnz(sample_completed(P_index,SNR_index,:)),MC,toc(point_timer));
        end
    end
end

if ~all(sample_completed(:))
    error('Some samples are incomplete. The checkpoint is saved; rerun this script to resume.');
end

%% -------------------- PAIRED NMSE AND CONFIDENCE INTERVALS --------------------
GS_NMSE_dB = zeros(number_of_P,number_of_SNR);
EM_GS_NMSE_dB = zeros(number_of_P,number_of_SNR);
gain_EM_over_GS_dB = zeros(number_of_P,number_of_SNR);
gain_CI_low = zeros(number_of_P,number_of_SNR);
gain_CI_high = zeros(number_of_P,number_of_SNR);
mean_GS_iterations = zeros(number_of_P,number_of_SNR);
mean_EM_GS_iterations = zeros(number_of_P,number_of_SNR);
GS_convergence_fraction = zeros(number_of_P,number_of_SNR);
EM_GS_convergence_fraction = zeros(number_of_P,number_of_SNR);

rng(261026,'twister');
number_of_rows = number_of_P*number_of_SNR;
result_rows = zeros(number_of_rows,13);
row_number = 0;
for P_index = 1:number_of_P
    for SNR_index = 1:number_of_SNR
        row_number = row_number+1;
        errors = reshape(sample_squared_error(P_index,SNR_index,:,:),MC,2);
        energies = reshape(sample_channel_energy(P_index,SNR_index,:),MC,1);
        total_energy = sum(energies);
        GS_NMSE = sum(errors(:,1))/max(total_energy,realmin);
        EM_GS_NMSE = sum(errors(:,2))/max(total_energy,realmin);
        GS_NMSE_dB(P_index,SNR_index) = 10*log10(max(GS_NMSE,realmin));
        EM_GS_NMSE_dB(P_index,SNR_index) = 10*log10(max(EM_GS_NMSE,realmin));
        gain_EM_over_GS_dB(P_index,SNR_index) = GS_NMSE_dB(P_index,SNR_index)-EM_GS_NMSE_dB(P_index,SNR_index);

        bootstrap_gain = zeros(bootstrap_replicates,1);
        for replicate = 1:bootstrap_replicates
            resampled_indices = randi(MC,MC,1);
            resampled_energy = sum(energies(resampled_indices));
            resampled_GS_NMSE = sum(errors(resampled_indices,1))/max(resampled_energy,realmin);
            resampled_EM_GS_NMSE = sum(errors(resampled_indices,2))/max(resampled_energy,realmin);
            bootstrap_gain(replicate) = 10*log10(max(resampled_GS_NMSE,realmin))-10*log10(max(resampled_EM_GS_NMSE,realmin));
        end
        bootstrap_gain = sort(bootstrap_gain);
        gain_CI_low(P_index,SNR_index) = bootstrap_gain(max(1,floor(0.025*bootstrap_replicates)));
        gain_CI_high(P_index,SNR_index) = bootstrap_gain(min(bootstrap_replicates,ceil(0.975*bootstrap_replicates)));

        iterations = reshape(sample_iterations(P_index,SNR_index,:,:),MC,2);
        convergence = reshape(sample_converged(P_index,SNR_index,:,:),MC,2);
        mean_GS_iterations(P_index,SNR_index) = mean(iterations(:,1));
        mean_EM_GS_iterations(P_index,SNR_index) = mean(iterations(:,2));
        GS_convergence_fraction(P_index,SNR_index) = mean(convergence(:,1));
        EM_GS_convergence_fraction(P_index,SNR_index) = mean(convergence(:,2));

        result_rows(row_number,:) = [P_list(P_index),SNR_dB(SNR_index),GS_NMSE_dB(P_index,SNR_index),EM_GS_NMSE_dB(P_index,SNR_index),gain_EM_over_GS_dB(P_index,SNR_index),gain_CI_low(P_index,SNR_index),gain_CI_high(P_index,SNR_index),mean_GS_iterations(P_index,SNR_index),mean_EM_GS_iterations(P_index,SNR_index),GS_convergence_fraction(P_index,SNR_index),EM_GS_convergence_fraction(P_index,SNR_index),MC,RSR_dB];
    end
end

comparison_table = array2table(result_rows,'VariableNames',{'P','SNR_dB','GS_NMSE_dB','EM_GS_NMSE_dB','Gain_EM_over_GS_dB','Gain_CI_low_dB','Gain_CI_high_dB','Mean_GS_iterations','Mean_EM_GS_iterations','GS_convergence_fraction','EM_GS_convergence_fraction','Samples_per_point','RSR_dB'});
writetable(comparison_table,table_file);
disp(comparison_table);

%% -------------------- PLOT BOTH CURVES --------------------
figure_handle = figure('Color','w','Name','Paired GS and EM-GS channel estimation');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
for P_index = 1:number_of_P
    nexttile;
    plot(SNR_dB,GS_NMSE_dB(P_index,:),'-o','LineWidth',1.6,'MarkerSize',5,'DisplayName','Biased GS');
    hold on;
    plot(SNR_dB,EM_GS_NMSE_dB(P_index,:),'-s','LineWidth',1.6,'MarkerSize',5,'DisplayName','EM-GS');
    grid on;
    box on;
    xlabel('SNR (dB)');
    ylabel('NMSE (dB)');
    title(sprintf('N = %d, P = %d',N,P_list(P_index)));
    xticks(SNR_dB);
    legend('Location','best');
end
sgtitle(sprintf('Paired biased GS and EM-GS channel estimation; RSR = %g dB',RSR_dB));
exportgraphics(figure_handle,figure_png,'Resolution',300);
savefig(figure_handle,figure_file);

save(result_file,'run_configuration','comparison_table','sample_squared_error','sample_channel_energy','sample_iterations','sample_converged','GS_NMSE_dB','EM_GS_NMSE_dB','gain_EM_over_GS_dB','gain_CI_low','gain_CI_high','-v7.3');

fprintf('\nLast row: P = %d, SNR = %g dB: GS = %.3f dB, EM-GS = %.3f dB, EM gain = %.3f dB, paired 95%% CI = [%.3f, %.3f].\n',comparison_table.P(end),comparison_table.SNR_dB(end),comparison_table.GS_NMSE_dB(end),comparison_table.EM_GS_NMSE_dB(end),comparison_table.Gain_EM_over_GS_dB(end),comparison_table.Gain_CI_low_dB(end),comparison_table.Gain_CI_high_dB(end));
fprintf('Table saved to: %s\n',table_file);
fprintf('Figure saved to: %s\n',figure_png);
fprintf('MAT results saved to: %s\n',result_file);
