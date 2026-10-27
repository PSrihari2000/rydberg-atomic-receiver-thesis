%% HS_GS_V24_MULTI_SPLIT_RANK_VALIDATION.m
% Validation-only study of Hankel rank and projection duration.
% Uses three independent validation seed groups and freezes one policy per
% N/P/SNR using only held-out validation channel NMSE. It does not load or
% score any test data. All candidates use 50 EM updates and four Cadzow
% sweeps whenever a projection is applied.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V24_MULTI_SPLIT_RANK_VALIDATION

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end

checkpoint_file = fullfile(result_folder,'HS_GS_V24_validation_checkpoint.mat');
policy_file = fullfile(result_folder,'HS_GS_V24_frozen_validation_policy.mat');
table_file = fullfile(result_folder,'HS_GS_V24_validation_candidates.csv');
if isfile(policy_file) || isfile(table_file)
    error('V24 output already exists. Inspect it before starting a new run.');
end

% Keep the physical/simulation settings matched to the previous v2 studies.
N_list = [8 16];
p_H_list = [3 7];
P_list = [10 30];
SNR_dB = -5:5:30;
K = 3;
RSR_dB = 25;
d_over_lambda = 0.5;
mu_eg = [0; 1785.9 * 1.602e-19 * 5.292e-11; 0];
hbar = 6.62607015e-34 / (2*pi);

% Multiple seed groups test whether the selected rank rule is stable.
seed_base_by_split = [2500000000 2600000000 2700000000];
number_of_splits = numel(seed_base_by_split);
samples_per_split = 100;
samples_per_checkpoint = 20;
total_updates = 50;
cadzow_sweeps = 4;
workers_if_starting_pool = 6;

% Candidate columns are [rank, projected EM updates, total EM updates].
candidate_settings_by_N = cell(1,numel(N_list));
for n_index = 1:numel(N_list)
    N = N_list(n_index);
    p_H = p_H_list(n_index);
    maximum_rank = min(N-p_H,p_H+1);
    projection_durations = [5 25];
    if N == 8
        ranks_to_test = [2 3];
    else
        ranks_to_test = 2:7;
    end
    candidates = [maximum_rank 0 total_updates]; % Unprojected EM-GS control.
    for rank_value = ranks_to_test
        for projection_index = 1:numel(projection_durations)
            candidates(end+1,:) = [rank_value projection_durations(projection_index) total_updates]; %#ok<SAGROW>
        end
    end
    candidate_settings_by_N{n_index} = candidates;
end

number_of_candidates_by_N = cellfun(@(x) size(x,1),candidate_settings_by_N);
validation_squared_error = cell(1,numel(N_list));
validation_channel_energy = cell(1,numel(N_list));
validation_completed = cell(1,numel(N_list));
for n_index = 1:numel(N_list)
    number_of_candidates = number_of_candidates_by_N(n_index);
    validation_squared_error{n_index} = nan(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split,number_of_candidates);
    validation_channel_energy{n_index} = nan(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split);
    validation_completed{n_index} = false(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split);
end

if isfile(checkpoint_file)
    C = load(checkpoint_file);
    if ~isequal(C.N_list,N_list) || ~isequal(C.p_H_list,p_H_list) || ...
            ~isequal(C.P_list,P_list) || ~isequal(C.SNR_dB,SNR_dB) || ...
            ~isequal(C.seed_base_by_split,seed_base_by_split) || ...
            C.samples_per_split~=samples_per_split || C.total_updates~=total_updates || ...
            ~isequal(C.candidate_settings_by_N,candidate_settings_by_N)
        error('V24 checkpoint settings do not match this script; preserve and inspect it.');
    end
    validation_squared_error = C.validation_squared_error;
    validation_channel_energy = C.validation_channel_energy;
    validation_completed = C.validation_completed;
    fprintf('Resuming V24: %d/%d samples complete.\n', ...
        sum(cellfun(@(x) sum(x,'all'),validation_completed)), ...
        numel(P_list)*numel(SNR_dB)*number_of_splits*samples_per_split*numel(N_list));
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('local',workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('V24 multi-split rank validation: %d workers; no test data will be used.\n',pool.NumWorkers);
fprintf('Samples per N/P/SNR point: %d (%d splits x %d samples).\n', ...
    number_of_splits*samples_per_split,number_of_splits,samples_per_split);

for n_index = 1:numel(N_list)
    N = N_list(n_index);
    p_H = p_H_list(n_index);
    candidate_settings = candidate_settings_by_N{n_index};
    number_of_candidates = size(candidate_settings,1);
    for P_index = 1:numel(P_list)
        P = P_list(P_index);
        for SNR_index = 1:numel(SNR_dB)
            snr_db = SNR_dB(SNR_index);
            point_timer = tic;
            for split_index = 1:number_of_splits
                for block_start = 1:samples_per_checkpoint:samples_per_split
                    positions = block_start:min(block_start+samples_per_checkpoint-1,samples_per_split);
                    completed_here = reshape(validation_completed{n_index}( ...
                        P_index,SNR_index,split_index,positions),1,[]);
                    pending = positions(~completed_here);
                    if isempty(pending), continue; end

                    block_error = nan(numel(pending),number_of_candidates);
                    block_energy = nan(numel(pending),1);
                    parfor local_index = 1:numel(pending)
                        sample_index = pending(local_index);
                        seed = seed_base_by_split(split_index) + ...
                            100000000*n_index + 1000000*P + sample_index;
                        [Y,G,S,B,sigma2] = v24_generate_sample( ...
                            N,K,P,snr_db,RSR_dB,d_over_lambda,mu_eg,hbar,seed);
                        block_energy(local_index) = norm(G,'fro')^2;
                        for candidate_index = 1:number_of_candidates
                            rank_value = candidate_settings(candidate_index,1);
                            projected_updates = candidate_settings(candidate_index,2);
                            candidate_total_updates = candidate_settings(candidate_index,3);
                            G_candidate = v24_hybrid_em(Y,S,B,sigma2,rank_value, ...
                                projected_updates,candidate_total_updates,p_H,cadzow_sweeps);
                            block_error(local_index,candidate_index) = norm(G_candidate-G,'fro')^2;
                        end
                    end

                    validation_squared_error{n_index}(P_index,SNR_index,split_index,pending,:) = ...
                        reshape(block_error,[1 1 1 numel(pending) number_of_candidates]);
                    validation_channel_energy{n_index}(P_index,SNR_index,split_index,pending) = block_energy;
                    validation_completed{n_index}(P_index,SNR_index,split_index,pending) = true;

                    save(checkpoint_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB', ...
                        'seed_base_by_split','number_of_splits','samples_per_split', ...
                        'total_updates','cadzow_sweeps','candidate_settings_by_N', ...
                        'validation_squared_error','validation_channel_energy', ...
                        'validation_completed','-v7.3');
                    fprintf('N=%d, P=%d, SNR=%d dB, split %d/%d: %d/%d samples; %.1f s.\n', ...
                        N,P,snr_db,split_index,number_of_splits, ...
                        sum(validation_completed{n_index}(P_index,SNR_index,split_index,:),'all'), ...
                        samples_per_split,toc(point_timer));
                end
            end
            fprintf('Validated N=%d, P=%d, SNR=%d dB.\n',N,P,snr_db);
        end
    end
end

% Select settings from pooled validation errors (ratio of total error energy).
selected_candidate = zeros(numel(N_list),numel(P_list),numel(SNR_dB));
selected_rank = zeros(size(selected_candidate));
selected_projected_updates = zeros(size(selected_candidate));
selected_total_updates = zeros(size(selected_candidate));
selected_validation_nmse_db = zeros(size(selected_candidate));
candidate_rows = cell(0,9);
for n_index = 1:numel(N_list)
    candidates = candidate_settings_by_N{n_index};
    for P_index = 1:numel(P_list)
        for SNR_index = 1:numel(SNR_dB)
            candidate_error_sum = squeeze(sum(sum(validation_squared_error{n_index}( ...
                P_index,SNR_index,:,:,:),3,'omitnan'),4,'omitnan'));
            energy_sum = sum(validation_channel_energy{n_index}(P_index,SNR_index,:,:),'all','omitnan');
            candidate_nmse = candidate_error_sum ./ max(energy_sum,realmin);
            [best_nmse,best_index] = min(candidate_nmse);
            split_wins = zeros(size(candidates,1),1);
            for split_index = 1:number_of_splits
                split_errors = squeeze(sum(validation_squared_error{n_index}( ...
                    P_index,SNR_index,split_index,:,:),4,'omitnan'));
                split_energy = sum(validation_channel_energy{n_index}( ...
                    P_index,SNR_index,split_index,:),'all','omitnan');
                split_nmse = split_errors ./ max(split_energy,realmin);
                [~,split_best_index] = min(split_nmse);
                split_wins(split_best_index) = split_wins(split_best_index)+1;
            end
            selected_candidate(n_index,P_index,SNR_index) = best_index;
            selected_rank(n_index,P_index,SNR_index) = candidates(best_index,1);
            selected_projected_updates(n_index,P_index,SNR_index) = candidates(best_index,2);
            selected_total_updates(n_index,P_index,SNR_index) = candidates(best_index,3);
            selected_validation_nmse_db(n_index,P_index,SNR_index) = 10*log10(max(best_nmse,realmin));
            for candidate_index = 1:size(candidates,1)
                is_selected = candidate_index == best_index;
                candidate_rows(end+1,:) = {N_list(n_index),P_list(P_index),SNR_dB(SNR_index), ...
                    candidates(candidate_index,1),candidates(candidate_index,2), ...
                    candidates(candidate_index,3),10*log10(max(candidate_nmse(candidate_index),realmin)), ...
                    is_selected,split_wins(candidate_index)}; %#ok<SAGROW>
            end
        end
    end
end

candidate_table = cell2table(candidate_rows,'VariableNames', ...
    {'N','P','SNR_dB','Rank','Projected_EM_updates','Total_EM_updates', ...
    'Pooled_Validation_NMSE_dB','Selected_Pooled','Number_of_Split_Wins'});
writetable(candidate_table,table_file);
save(policy_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB', ...
    'seed_base_by_split','number_of_splits','samples_per_split','total_updates', ...
    'cadzow_sweeps','candidate_settings_by_N','selected_candidate','selected_rank', ...
    'selected_projected_updates','selected_total_updates','selected_validation_nmse_db', ...
    'candidate_table');

fprintf('\nFrozen V24 validation policy (selected settings only):\n');
selected_rows = candidate_table(candidate_table.Selected_Pooled,:);
disp(selected_rows);
fprintf('Candidate table: %s\nFrozen policy: %s\n',table_file,policy_file);

function [Y,G,S,B,sigma2] = v24_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
rng(double(seed),'twister');
G = zeros(N,K);
for user = 1:K
    number_of_paths = randi([3 7]);
    for path_index = 1:number_of_paths
        alpha = (randn+1j*randn)/sqrt(2);
        theta = 2*pi*rand;
        phase_step = 2*pi*d_over_lambda*cos(theta);
        polarization = sqrt(1/3)*randn(3,1);
        for antenna = 1:N
            G(antenna,user) = G(antenna,user) + ...
                (mu_eg.'*polarization/hbar)*alpha*exp(-1j*(antenna-1)*phase_step);
        end
    end
end
S = (randn(K,P)+1j*randn(K,P))/sqrt(2);
alpha_b = sqrt(10/2)*(randn+1j*randn);
theta_b = 2*pi*rand;
phase_step_b = 2*pi*d_over_lambda*cos(theta_b);
polarization_b = sqrt(1/3)*randn(3,1);
b = zeros(N,1);
for antenna = 1:N
    b(antenna) = (mu_eg.'*polarization_b/hbar)*alpha_b*exp(-1j*(antenna-1)*phase_step_b);
end
B = b*ones(1,P);
average_signal_power = 0;
for user = 1:K
    average_signal_power = average_signal_power + mean(abs(G(:,user)*S(user,:)).^2,'all')/K;
end
B = B*sqrt(10^(RSR_db/10)*average_signal_power/mean(abs(B).^2,'all'));
sigma2 = mean(abs(G*S).^2,'all')/10^(snr_db/10);
W = sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P));
Y = abs(G*S+B+W);
end

function G = v24_hybrid_em(Y,S,B,sigma2,rank_value,projected_updates,total_updates,p_H,sweeps)
G = v24_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
for iteration = 1:total_updates
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if iteration <= projected_updates
        G = v24_hankel_project(G,p_H,rank_value,sweeps);
    end
end
end

function G = v24_spectral_initialize(Y,S,B)
N = size(Y,1);
K = size(S,1);
G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).';
    b = B(antenna,:).';
    A = [S.',b];
    M = A'*diag(y)*A;
    [V,D] = eig(M);
    [~,principal_index] = max(real(diag(D)));
    v = V(:,principal_index);
    v = v/max(norm(v),eps);
    scale = (abs(A*v).'*y)/max(norm(A*v)^2,eps);
    G(antenna,:) = (scale*exp(-1j*angle(v(K+1)))*v(1:K)).';
end
end

function G = v24_hankel_project(G,p_H,rank_value,sweeps)
for sweep = 1:sweeps
    for user = 1:size(G,2)
        x = G(:,user);
        N = length(x);
        number_of_rows = N-p_H;
        number_of_columns = p_H+1;
        H = zeros(number_of_rows,number_of_columns);
        for row = 1:number_of_rows
            H(row,:) = x(row:row+number_of_columns-1).';
        end
        [U,Sigma,V] = svd(H,'econ');
        H_rank = U(:,1:rank_value)*Sigma(1:rank_value,1:rank_value)*V(:,1:rank_value)';
        reconstructed = zeros(N,1);
        counts = zeros(N,1);
        for row = 1:number_of_rows
            for column = 1:number_of_columns
                index = row+column-1;
                reconstructed(index) = reconstructed(index)+H_rank(row,column);
                counts(index) = counts(index)+1;
            end
        end
        G(:,user) = reconstructed./counts;
    end
end
end
