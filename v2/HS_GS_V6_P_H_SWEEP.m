%% HS_GS_V6_P_H_SWEEP.m
% Diagnostic: test whether the Hankel pencil p_H causes the high-SNR gap.
% Uses V5's frozen adaptive updates/sweeps; p_H is the only changed factor.
% Validation/test data remain independent. This is a diagnostic, not a final
% benchmark claim.
clear; clc; close all;

script_folder = fileparts(mfilename('fullpath'));
source_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir'), mkdir(result_folder); end

P_list = [10 30]; SNR_dB = -5:5:30; N = 8; K = 3; RSR_dB = 25;
p_H_list = [2 3 4 5];
rank_selection_updates = 50; training_fraction = 0.70; random_seed_base = 910000;
maximum_test_samples_per_snr = inf;
% These are frozen from the completed V5 validation run.
selected_updates = [100 100 100 100 100 100 100 1000; 100 100 100 100 500 100 100 1000];
selected_sweeps  = [4 2 1 1 1 1 1 0;       2 2 1 1 1 1 1 1];
requested_workers = 5;
pool = gcp('nocreate');
if isempty(pool), parpool('local', requested_workers);
elseif pool.NumWorkers ~= requested_workers, delete(pool); parpool('local', requested_workers); end
pool = gcp('nocreate');
fprintf('HS-GS v6 p_H sweep started with %d workers.\n', pool.NumWorkers);

checkpoint_file = fullfile(result_folder, 'HS_GS_V6_p_H_sweep_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V6_p_H_sweep_results.mat');
test_NMSE = zeros(length(p_H_list), length(P_list), length(SNR_dB));
test_NMSE_dB = zeros(size(test_NMSE)); test_mean_rank = zeros(size(test_NMSE));
completed_points = false(size(test_NMSE));
if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    if isequal(checkpoint.p_H_list, p_H_list) && isequal(checkpoint.P_list, P_list) && isequal(checkpoint.SNR_dB, SNR_dB)
        test_NMSE = checkpoint.test_NMSE; test_NMSE_dB = checkpoint.test_NMSE_dB;
        test_mean_rank = checkpoint.test_mean_rank; completed_points = checkpoint.completed_points;
        fprintf('Loaded checkpoint: %d of %d points complete.\n', sum(completed_points, 'all'), numel(completed_points));
    end
end

for p_index = 1:length(p_H_list)
    p_H = p_H_list(p_index); rank_list = 1:min(N-p_H, p_H+1);
    for P_index = 1:length(P_list)
        P = P_list(P_index); test_source = load(fullfile(source_folder, sprintf('test_P%d.mat', P)));
        for snr_index = 1:length(SNR_dB)
            if completed_points(p_index,P_index,snr_index), continue; end
            test_samples = find(test_source.SNR_dB_data(:) == SNR_dB(snr_index));
            test_samples = limit_samples(test_samples, maximum_test_samples_per_snr);
            squared_error = zeros(length(test_samples),1); channel_energy = zeros(length(test_samples),1); selected_rank = zeros(length(test_samples),1);
            update_count = selected_updates(P_index,snr_index); sweep_count = selected_sweeps(P_index,snr_index);
            fprintf('p_H=%d, P=%d, SNR=%3d dB\n', p_H, P, SNR_dB(snr_index));
            parfor sample_number = 1:length(test_samples)
                sample_index = test_samples(sample_number);
                Y = double(test_source.Y_data(:,:,sample_index)); G = double(test_source.G_data(:,:,sample_index));
                S = double(test_source.S_data(:,:,sample_index)); B = double(test_source.B_data(:,:,sample_index));
                sigma2 = double(test_source.sigma2_data(sample_index));
                [training_pilots, held_out_pilots] = split_pilots(P, training_fraction, random_seed_base, sample_index);
                candidate_error = zeros(length(rank_list),1);
                for rank_index = 1:length(rank_list)
                    G_candidate = run_hs_gs(Y(:,training_pilots), S(:,training_pilots), B(:,training_pilots), sigma2, rank_list(rank_index), rank_selection_updates, sweep_count, p_H);
                    candidate_error(rank_index) = norm(Y(:,held_out_pilots)-abs(G_candidate*S(:,held_out_pilots)+B(:,held_out_pilots)),'fro')^2;
                end
                [~, best_rank_index] = min(candidate_error); selected_rank(sample_number) = rank_list(best_rank_index);
                G_hat = run_hs_gs(Y,S,B,sigma2,selected_rank(sample_number),update_count,sweep_count,p_H);
                squared_error(sample_number) = norm(G_hat-G,'fro')^2; channel_energy(sample_number) = norm(G,'fro')^2;
            end
            test_NMSE(p_index,P_index,snr_index) = sum(squared_error)/max(sum(channel_energy),eps);
            test_NMSE_dB(p_index,P_index,snr_index) = 10*log10(test_NMSE(p_index,P_index,snr_index));
            test_mean_rank(p_index,P_index,snr_index) = mean(selected_rank);
            fprintf('Finished p_H=%d, P=%d, SNR=%3d dB: NMSE=%8.3f dB, mean rank=%.3f\n',p_H,P,SNR_dB(snr_index),test_NMSE_dB(p_index,P_index,snr_index),test_mean_rank(p_index,P_index,snr_index));
            completed_points(p_index,P_index,snr_index) = true;
            save(checkpoint_file,'test_NMSE','test_NMSE_dB','test_mean_rank','completed_points','p_H_list','P_list','SNR_dB','selected_updates','selected_sweeps','rank_selection_updates','training_fraction','N','K','RSR_dB');
        end
    end
end
save(result_file,'test_NMSE','test_NMSE_dB','test_mean_rank','p_H_list','P_list','SNR_dB','selected_updates','selected_sweeps','rank_selection_updates','training_fraction','N','K','RSR_dB');
fprintf('Saved V6 results to %s\n', result_file);

function samples = limit_samples(samples, maximum_samples)
if isfinite(maximum_samples), samples = samples(1:min(length(samples),maximum_samples)); end
end

function [training_pilots, held_out_pilots] = split_pilots(P, training_fraction, seed_base, sample_index)
rng(seed_base + 1000*P + sample_index,'twister'); order = randperm(P); training_count = round(training_fraction*P);
training_pilots = order(1:training_count); held_out_pilots = order(training_count+1:P);
end

function G_hat = run_hs_gs(Y,S,B,sigma2,rank_value,em_updates,cadzow_sweeps,p_H)
N = size(Y,1); K = size(S,1); G_hat = zeros(N,K);
for cell_index = 1:N
    y_i=Y(cell_index,:).'; b_i=B(cell_index,:).'; A=[S.',b_i]; M=A'*diag(y_i)*A;
    [V,D]=eig(M); [~,idx]=max(real(diag(D))); v=V(:,idx); v=v/max(norm(v),eps);
    r=(abs(A*v).'*y_i)/max(norm(A*v)^2,eps); G_hat(cell_index,:)=(r*exp(-1j*angle(v(K+1)))*v(1:K)).';
end
for update_index=1:em_updates
    field=G_hat*S+B; kappa=(2/max(sigma2,eps))*Y.*abs(field); R=besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    U=Y.*exp(1j*angle(field)).*R; G_tilde=(U-B)*S'/(S*S'); G_processed=zeros(N,K);
    for user_index=1:K
        x=G_tilde(:,user_index);
        for sweep_index=1:cadzow_sweeps
            H_rows=N-p_H; H_columns=p_H+1; H=zeros(H_rows,H_columns);
            for row_index=1:H_rows, H(row_index,:)=x(row_index:row_index+H_columns-1).'; end
            [U_H,Sigma_H,V_H]=svd(H,'econ'); H_rank=U_H(:,1:rank_value)*Sigma_H(1:rank_value,1:rank_value)*V_H(:,1:rank_value)';
            x_reconstructed=zeros(N,1); copies=zeros(N,1);
            for row_index=1:H_rows
                for column_index=1:H_columns, q=row_index+column_index-1; x_reconstructed(q)=x_reconstructed(q)+H_rank(row_index,column_index); copies(q)=copies(q)+1; end
            end
            x=x_reconstructed./max(copies,eps);
        end
        G_processed(:,user_index)=x;
    end
    G_hat=G_processed;
end
end
