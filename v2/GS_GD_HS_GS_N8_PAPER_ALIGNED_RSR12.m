%% GS, GD, and HS-GS comparison at literature-referenced RSR = 12 dB
% The supplied Hankel study uses RSR = 12 dB in its main NMSE and array-size
% experiments (PDF pp. 5-6). Cui et al., Towards Atomic MIMO Receivers, define
% RSR as reference power divided by single-user received signal power (Eq. 37)
% and use 12 dB in their BER experiment (PDF pp. 11-12). Xu et al. state a
% strong-reference assumption but do not specify a numerical RSR.
% Here RSR is implemented per generated sample as mean reference power divided
% by the average single-user received signal power, using all users, antennas,
% and pilot slots. GS, GD, and HS-GS use the same generated sample.
% Rank is selected separately for each sample using held-out pilot magnitudes.
% This script loads no external validation or result file. The held-out pilots
% are part of the estimator; true channel G is used only for NMSE scoring.
% HS-GS uses a hard Hankel rank projection. At maximum rank, it returns the
% converged EM-GS estimate, matching the fallback described in the paper.
% One Cadzow sweep is used per HS update: the earlier ablation selected one
% sweep most often. Final estimator updates stop by convergence, capped at 5000.
%
% Run in MATLAB:
% cd('D:\MTech_Thesis_Results\v2')
% GS_GD_HS_GS_N8_PAPER_ALIGNED_RSR12

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end

checkpoint_file = fullfile(result_folder,'GS_GD_HS_GS_N8_PAPER_ALIGNED_RSR12_checkpoint.mat');
result_file = fullfile(result_folder,'GS_GD_HS_GS_N8_PAPER_ALIGNED_RSR12_results.mat');
table_file = fullfile(result_folder,'GS_GD_HS_GS_N8_PAPER_ALIGNED_RSR12_table.csv');
figure_file = fullfile(result_folder,'GS_GD_HS_GS_N8_PAPER_ALIGNED_RSR12.png');
figure_editable_file = fullfile(result_folder,'GS_GD_HS_GS_N8_PAPER_ALIGNED_RSR12.fig');
if isfile(result_file) || isfile(table_file)
    error('Final outputs already exist. Inspect them or change the run name before starting again.');
end

N = 8;
p_H = 3;
P_list = [10 30];
SNR_dB = -5:5:30;
K = 3;
RSR_dB = 12; % Literature-referenced operating point; shared by all estimators.
samples_per_test_point = 500;
samples_per_checkpoint = 20;
sample_seed_base = 4270000000;
training_fraction = 0.70;
rank_selection_updates = 25;
cadzow_sweeps = 1;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
bootstrap_replicates = 2000;
workers_if_starting_pool = 6;
d_over_lambda = 0.5;
mu_eg = [0; 1785.9*1.602e-19*5.292e-11; 0];
hbar = 6.62607015e-34/(2*pi);
gd_initial_variance = 0.1;
method_names = {'GS','GD','HS-GS'};
number_of_methods = numel(method_names);
rank_list = 1:min(N-p_H,p_H+1);

number_of_P = numel(P_list);
number_of_SNR = numel(SNR_dB);
test_squared_error = nan(number_of_P,number_of_SNR,samples_per_test_point,number_of_methods);
test_channel_energy = nan(number_of_P,number_of_SNR,samples_per_test_point);
test_selected_rank = nan(number_of_P,number_of_SNR,samples_per_test_point);
test_heldout_error = nan(number_of_P,number_of_SNR,samples_per_test_point);
test_iterations = zeros(number_of_P,number_of_SNR,samples_per_test_point,number_of_methods);
test_converged = false(number_of_P,number_of_SNR,samples_per_test_point,number_of_methods);
test_completed = false(number_of_P,number_of_SNR,samples_per_test_point);

run_configuration = struct('N',N,'p_H',p_H,'P_list',P_list,'SNR_dB',SNR_dB,'K',K, ...
    'RSR_dB',RSR_dB,'samples_per_test_point',samples_per_test_point, ...
    'sample_seed_base',sample_seed_base,'training_fraction',training_fraction, ...
    'rank_selection_updates',rank_selection_updates,'cadzow_sweeps',cadzow_sweeps, ...
    'maximum_updates',maximum_updates,'convergence_tolerance',convergence_tolerance, ...
    'gd_initial_variance',gd_initial_variance);
if isfile(checkpoint_file)
    saved = load(checkpoint_file);
    if ~isequal(saved.run_configuration,run_configuration)
        error('Checkpoint settings do not match. Preserve the checkpoint and use a new run name.');
    end
    test_squared_error = saved.test_squared_error;
    test_channel_energy = saved.test_channel_energy;
    test_selected_rank = saved.test_selected_rank;
    test_heldout_error = saved.test_heldout_error;
    test_iterations = saved.test_iterations;
    test_converged = saved.test_converged;
    test_completed = saved.test_completed;
    fprintf('Resuming: %d/%d samples complete.\n',nnz(test_completed),numel(test_completed));
end

pool = gcp('nocreate');
if isempty(pool), parpool('local',workers_if_starting_pool); pool = gcp('nocreate'); end
fprintf('Three-method paired run: GS, GD, and hard-rank HS-GS; N=%d, RSR=%g dB.\n',N,RSR_dB);
fprintf('Pilot rank selection is per sample (70/30 split); no external validation file is loaded. Workers=%d.\n',pool.NumWorkers);
fprintf('Rank candidates = [%s], rank-fit updates = %d, Cadzow sweeps per HS update = %d.\n',num2str(rank_list),rank_selection_updates,cadzow_sweeps);
fprintf('Final GS/GD/HS-GS iterations stop at relative-change tolerance %g, with cap %d.\n',convergence_tolerance,maximum_updates);

for P_index = 1:number_of_P
    P = P_list(P_index);
    for SNR_index = 1:number_of_SNR
        snr_db = SNR_dB(SNR_index);
        point_timer = tic;
        fprintf('Starting P=%d, SNR=%g dB.\n',P,snr_db);
        for block_start = 1:samples_per_checkpoint:samples_per_test_point
            positions = block_start:min(block_start+samples_per_checkpoint-1,samples_per_test_point);
            was_completed = reshape(test_completed(P_index,SNR_index,positions),1,[]);
            pending = positions(~was_completed);
            if isempty(pending), continue; end
            block_error = nan(numel(pending),number_of_methods);
            block_energy = nan(numel(pending),1);
            block_rank = nan(numel(pending),1);
            block_heldout_error = nan(numel(pending),1);
            block_iterations = zeros(numel(pending),number_of_methods);
            block_converged = false(numel(pending),number_of_methods);
            parfor local_index = 1:numel(pending)
                sample_index = pending(local_index);
                seed = sample_seed_base+1000000*P+10000*SNR_index+sample_index;
                [Y,G,S,B,sigma2] = paper_generate_sample(N,K,P,snr_db,RSR_dB,d_over_lambda,mu_eg,hbar,seed);
                [chosen_rank,heldout_error] = paper_select_rank(Y,S,B,sigma2,p_H,rank_list, ...
                    rank_selection_updates,training_fraction,cadzow_sweeps,seed);
                if chosen_rank == rank_list(end)
                    [G_em,em_updates,em_ok] = paper_em_converged(Y,S,B,sigma2,maximum_updates,convergence_tolerance);
                    G_hs = G_em;
                    hs_updates = em_updates;
                    hs_ok = em_ok;
                else
                    [G_hs,hs_updates,hs_ok] = paper_hs_converged(Y,S,B,sigma2,p_H,chosen_rank, ...
                        cadzow_sweeps,maximum_updates,convergence_tolerance);
                end
                [G_gs,gs_updates,gs_ok] = paper_gs_converged(Y,S,B,maximum_updates,convergence_tolerance);
                [G_gd,gd_updates,gd_ok] = paper_gd_converged(Y,S,B,maximum_updates, ...
                    convergence_tolerance,gd_initial_variance,seed);
                block_error(local_index,:) = [norm(G_hs-G,'fro')^2,norm(G_gs-G,'fro')^2,norm(G_gd-G,'fro')^2];
                block_energy(local_index) = norm(G,'fro')^2;
                block_rank(local_index) = chosen_rank;
                block_heldout_error(local_index) = heldout_error;
                block_iterations(local_index,:) = [hs_updates,gs_updates,gd_updates];
                block_converged(local_index,:) = [hs_ok,gs_ok,gd_ok];
            end
            test_squared_error(P_index,SNR_index,pending,:) = reshape(block_error,[1 1 numel(pending) number_of_methods]);
            test_channel_energy(P_index,SNR_index,pending) = block_energy;
            test_selected_rank(P_index,SNR_index,pending) = block_rank;
            test_heldout_error(P_index,SNR_index,pending) = block_heldout_error;
            test_iterations(P_index,SNR_index,pending,:) = reshape(block_iterations,[1 1 numel(pending) number_of_methods]);
            test_converged(P_index,SNR_index,pending,:) = reshape(block_converged,[1 1 numel(pending) number_of_methods]);
            test_completed(P_index,SNR_index,pending) = true;
            save(checkpoint_file,'run_configuration','N','p_H','P_list','SNR_dB','K','RSR_dB', ...
                'rank_list','training_fraction','rank_selection_updates','cadzow_sweeps', ...
                'samples_per_test_point','sample_seed_base','maximum_updates','convergence_tolerance', ...
                'test_squared_error','test_channel_energy','test_selected_rank','test_heldout_error', ...
                'test_iterations','test_converged','test_completed','-v7.3');
            fprintf('  %d/%d samples complete; %.1f seconds for this point.\n', ...
                sum(test_completed(P_index,SNR_index,:),'all'),samples_per_test_point,toc(point_timer));
        end
    end
end

if ~all(test_completed,'all') || any(~isfinite(test_squared_error),'all') || ...
        any(~isfinite(test_channel_energy),'all') || any(test_channel_energy<=0,'all')
    error('Run has incomplete or invalid values. Checkpoint is preserved.');
end

rng(2026102731,'twister');
rows = zeros(number_of_P*number_of_SNR,19);
row_number = 0;
for P_index = 1:number_of_P
    for SNR_index = 1:number_of_SNR
        row_number = row_number+1;
        errors = reshape(test_squared_error(P_index,SNR_index,:,:),samples_per_test_point,number_of_methods);
        energy = reshape(test_channel_energy(P_index,SNR_index,:),[],1);
        nmse_db = 10*log10(sum(errors,1)/sum(energy));
        gain_gs = nmse_db(2)-nmse_db(1);
        gain_gd = nmse_db(3)-nmse_db(1);
        bootstrap_gs = zeros(bootstrap_replicates,1);
        bootstrap_gd = zeros(bootstrap_replicates,1);
        for draw = 1:bootstrap_replicates
            sample_indices = randi(samples_per_test_point,samples_per_test_point,1);
            bootstrap_gs(draw) = 10*log10(sum(errors(sample_indices,2))/sum(errors(sample_indices,1)));
            bootstrap_gd(draw) = 10*log10(sum(errors(sample_indices,3))/sum(errors(sample_indices,1)));
        end
        bootstrap_gs = sort(bootstrap_gs);
        bootstrap_gd = sort(bootstrap_gd);
        ranks = reshape(test_selected_rank(P_index,SNR_index,:),[],1);
        mean_iterations = reshape(mean(test_iterations(P_index,SNR_index,:,:),3),1,[]);
        converged_fraction = reshape(mean(test_converged(P_index,SNR_index,:,:),3),1,[]);
        rows(row_number,:) = [P_list(P_index),SNR_dB(SNR_index),nmse_db, ...
            gain_gs,gain_gd,bootstrap_gs(max(1,round(0.025*bootstrap_replicates)),1), ...
            bootstrap_gs(min(bootstrap_replicates,round(0.975*bootstrap_replicates)),1), ...
            bootstrap_gd(max(1,round(0.025*bootstrap_replicates)),1), ...
            bootstrap_gd(min(bootstrap_replicates,round(0.975*bootstrap_replicates)),1), ...
            mean(ranks),mean(ranks==rank_list(end)),mean_iterations,converged_fraction];
    end
end
result_table = array2table(rows,'VariableNames',{'P','SNR_dB','HS_GS_NMSE_dB','GS_NMSE_dB', ...
    'GD_NMSE_dB','Gain_HS_over_GS_dB','Gain_HS_over_GD_dB','CI_low_GS','CI_high_GS', ...
    'CI_low_GD','CI_high_GD','Mean_selected_rank','Fraction_max_rank','Mean_HS_GS_updates', ...
    'Mean_GS_updates','Mean_GD_updates','HS_GS_convergence_fraction','GS_convergence_fraction', ...
    'GD_convergence_fraction'});

figure_handle = figure('Color','w','Name','GS, GD, HS-GS N=8 paper-aligned comparison');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
method_columns = {'GS_NMSE_dB','GD_NMSE_dB','HS_GS_NMSE_dB'};
for P_index = 1:number_of_P
    nexttile; hold on;
    selected_rows = result_table.P==P_list(P_index);
    for method_index = 1:numel(method_columns)
        plot(SNR_dB,result_table{selected_rows,method_columns{method_index}},'-o', ...
            'LineWidth',1.5,'MarkerSize',5,'DisplayName',method_names{method_index});
    end
    grid on; xlabel('SNR (dB)'); ylabel('NMSE (dB)');
    title(sprintf('N = %d, P = %d',N,P_list(P_index)));
    legend('Location','southwest');
end
sgtitle(sprintf('GS, GD, and hard-rank HS-GS; RSR = %g dB',RSR_dB));
writetable(result_table,table_file);
save(result_file,'result_table','run_configuration','rank_list','test_squared_error', ...
    'test_channel_energy','test_selected_rank','test_heldout_error','test_iterations', ...
    'test_converged','-v7.3');
saveas(figure_handle,figure_file);
savefig(figure_handle,figure_editable_file);
disp(result_table);
fprintf('Table: %s\nFigure: %s\nMAT results: %s\n',table_file,figure_file,result_file);

function [selected_rank,best_error] = paper_select_rank(Y,S,B,sigma2,p_H,rank_list,selection_updates,training_fraction,sweeps,sample_seed)
P = size(S,2);
rng(mod(double(sample_seed)+1234567,2^32-1),'twister');
pilot_order = randperm(P);
training_count = round(training_fraction*P);
training_pilots = pilot_order(1:training_count);
heldout_pilots = pilot_order(training_count+1:end);
candidate_error = inf(numel(rank_list),1);
for rank_index = 1:numel(rank_list)
    G_candidate = paper_em_fixed(Y(:,training_pilots),S(:,training_pilots),B(:,training_pilots), ...
        sigma2,selection_updates,p_H,rank_list(rank_index),sweeps);
    prediction = abs(G_candidate*S(:,heldout_pilots)+B(:,heldout_pilots));
    candidate_error(rank_index) = sum((Y(:,heldout_pilots)-prediction).^2,'all');
end
[best_error,best_index] = min(candidate_error);
selected_rank = rank_list(best_index);
end

function G = paper_em_fixed(Y,S,B,sigma2,number_of_updates,p_H,rank_value,sweeps)
G = paper_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
for update_index = 1:number_of_updates
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if rank_value < min(size(G,1)-p_H,p_H+1)
        G = paper_hankel_project(G,p_H,rank_value,sweeps);
    end
end
end

function [G,iteration,converged] = paper_hs_converged(Y,S,B,sigma2,p_H,rank_value,sweeps,max_updates,tolerance)
G = paper_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    G = paper_hankel_project(G,p_H,rank_value,sweeps);
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = paper_em_converged(Y,S,B,sigma2,max_updates,tolerance)
G = paper_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = paper_gs_converged(Y,S,B,max_updates,tolerance)
G = paper_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    G = (Y.*exp(1j*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = paper_gd_converged(Y,S,B,max_updates,tolerance,initial_variance,seed)
rng(mod(double(seed)+720000,2^32-1),'twister');
N = size(Y,1); K = size(S,1);
G = sqrt(initial_variance/2)*(randn(N,K)+1j*randn(N,K));
phase_reference = exp(-1j*angle(B));
step_size = 1/max(real(eig(S*S')));
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function G = paper_spectral_initialize(Y,S,B)
N = size(Y,1); K = size(S,1); G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).'; b = B(antenna,:).'; A = [S.',b]; M = A'*diag(y)*A;
    [V,D] = eig(M); [~,principal_index] = max(real(diag(D))); v = V(:,principal_index);
    v = v/max(norm(v),eps); scale = (abs(A*v).'*y)/max(norm(A*v)^2,eps);
    G(antenna,:) = (scale*exp(-1j*angle(v(K+1)))*v(1:K)).';
end
end

function G = paper_hankel_project(G,p_H,rank_value,sweeps)
for sweep = 1:sweeps
    for user = 1:size(G,2)
        x = G(:,user); N = length(x); number_of_rows = N-p_H; number_of_columns = p_H+1;
        H = zeros(number_of_rows,number_of_columns);
        for row = 1:number_of_rows, H(row,:) = x(row:row+number_of_columns-1).'; end
        [U,Sigma,V] = svd(H,'econ');
        H_rank = U(:,1:rank_value)*Sigma(1:rank_value,1:rank_value)*V(:,1:rank_value)';
        reconstructed = zeros(N,1); counts = zeros(N,1);
        for row = 1:number_of_rows
            for column = 1:number_of_columns
                index = row+column-1; reconstructed(index) = reconstructed(index)+H_rank(row,column);
                counts(index) = counts(index)+1;
            end
        end
        G(:,user) = reconstructed./counts;
    end
end
end

function [Y,G,S,B,sigma2] = paper_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
% Reduce derived per-sample seeds to MATLAB rng's valid 32-bit range.
rng(mod(double(seed),2^32),'twister'); G = zeros(N,K);
for user = 1:K
    number_of_paths = randi([3 7]);
    for path_index = 1:number_of_paths
        alpha = (randn+1j*randn)/sqrt(2); theta = 2*pi*rand;
        phase_step = 2*pi*d_over_lambda*cos(theta); polarization = sqrt(1/3)*randn(3,1);
        for antenna = 1:N
            G(antenna,user) = G(antenna,user)+(mu_eg.'*polarization/hbar)*alpha*exp(-1j*(antenna-1)*phase_step);
        end
    end
end
S = (randn(K,P)+1j*randn(K,P))/sqrt(2);
alpha_b = sqrt(10/2)*(randn+1j*randn); theta_b = 2*pi*rand;
phase_step_b = 2*pi*d_over_lambda*cos(theta_b); polarization_b = sqrt(1/3)*randn(3,1); b = zeros(N,1);
for antenna = 1:N
    b(antenna) = (mu_eg.'*polarization_b/hbar)*alpha_b*exp(-1j*(antenna-1)*phase_step_b);
end
B = b*ones(1,P); average_signal_power = 0;
for user = 1:K
    average_signal_power = average_signal_power+mean(abs(G(:,user)*S(user,:)).^2,'all')/K;
end
B = B*sqrt(10^(RSR_db/10)*average_signal_power/mean(abs(B).^2,'all'));
sigma2 = mean(abs(G*S).^2,'all')/10^(snr_db/10);
W = sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P)); Y = abs(G*S+B+W);
end
