%% Validation-only threshold selection for frozen V4 vs soft-rank HS-GS.
% This script never loads a test file or changes a V4 checkpoint.
clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
experiment_root = fileparts(script_folder);
case_studies_root = fileparts(experiment_root);
project_root = fileparts(case_studies_root);
antenna_data_root = fullfile(case_studies_root,'Antenna_sweep','data','ANTENNA_COUNT_N8_16_24_32_K3_RSR25DB_SNR_m10_0_10_20_30');
user_data_root = fullfile(case_studies_root,'Users_sweep','data','USER_COUNT_CONFIRMATORY_V2_P10_K1_2_3_4_5_P30_K1_2_3_4_5_6_7_RSR25_SNR_m10_0_10_20_30');
output_root = fullfile(experiment_root,'Validation_results');
log_root = fullfile(experiment_root,'Experiment_logs');
if ~isfolder(output_root) || ~isfolder(log_root)
    error('Create the experiment directory structure before running this script.');
end
if local_has_entries(output_root)
    error('Validation output folder is not empty; inspect it and choose a new experiment directory.');
end

N_antenna = [8 16 24 32];
K_antenna = 3;
K_user_by_P = {[1 2 3 4 5],[1 2 3 4 5 6 7]};
P_list = [10 30];
SNR_grid = [-10 0 10 20 30];
RSR_dB = 25;
p_H_rule = 'floor((N-1)/2)';
threshold_multipliers = 0:0.1:2;
tuning_samples_per_condition = 30;
maximum_gs_updates = 20000;
gs_convergence_tolerance = 1e-10;
number_of_stages = 16;
selection_seed = 2026111301;
maximum_parallel_workers = 6;
pool = gcp('nocreate');
if isempty(pool)
    parpool('Processes',maximum_parallel_workers);
elseif pool.NumWorkers > maximum_parallel_workers
    warning('Existing parallel pool has %d workers; each parfor is capped at %d.', ...
        pool.NumWorkers,maximum_parallel_workers);
end

% Each row is one dimension/P configuration. Validation uses the corresponding
% dimension-matched split; the N=8,K=3 antenna validation cell defines transfer.
configurations = struct('study',{},'N',{},'K',{},'P',{},'validation_file',{});
for P = P_list
    for N = N_antenna
        configurations(end+1) = struct('study','Antenna','N',N,'K',K_antenna,'P',P, ...
            'validation_file',fullfile(antenna_data_root,sprintf('N%d_K%d_RSR25dB',N,K_antenna),sprintf('validation_P%d.mat',P))); %#ok<SAGROW>
    end
    p_index = find(P_list == P,1);
    for K = K_user_by_P{p_index}
        configurations(end+1) = struct('study','User','N',8,'K',K,'P',P, ...
            'validation_file',fullfile(user_data_root,sprintf('N8_K%d_P%d_RSR25dB',K,P),'validation.mat')); %#ok<SAGROW>
    end
end

if numel(configurations) ~= 20
    error('Unexpected number of dimension/P validation configurations.');
end
number_of_configs = numel(configurations);
number_of_snr = numel(SNR_grid);
number_of_candidates = numel(threshold_multipliers);
candidate_nmse = nan(number_of_configs,number_of_snr,number_of_candidates);
candidate_samples = zeros(number_of_configs,number_of_snr);
candidate_rows = cell(number_of_configs*number_of_snr*number_of_candidates,1);
candidate_row_index = 0;

fprintf('Validation-only tuning of conventional soft Hankel threshold.\n');
fprintf('Configs=%d; samples/condition=%d; threshold candidates=%d; stages=%d.\n', ...
    number_of_configs,tuning_samples_per_condition,number_of_candidates,number_of_stages);
fprintf('No test data or frozen checkpoint contents are used for threshold selection.\n');

for config_index = 1:number_of_configs
    cfg = configurations(config_index);
    if ~isfile(cfg.validation_file)
        error('Missing dimension-matched validation file: %s',cfg.validation_file);
    end
    data = load(cfg.validation_file,'Y_data','G_data','S_data','B_data','sigma2_data', ...
        'SNR_dB_data','N','K','P','RSR_dB','SNR_dB','split_name','number_of_samples');
    local_validate_split(data,cfg,SNR_grid,RSR_dB);
    p_H = floor((cfg.N-1)/2);
    candidate_error = zeros(number_of_snr,number_of_candidates);
    energy_sum = zeros(number_of_snr,1);
    for snr_index = 1:number_of_snr
        condition_indices = find(double(data.SNR_dB_data(:)) == SNR_grid(snr_index));
        rng(selection_seed + config_index*100 + snr_index,'twister');
        selected_indices = condition_indices(randperm(numel(condition_indices), ...
            min(tuning_samples_per_condition,numel(condition_indices))));
        if numel(selected_indices) ~= tuning_samples_per_condition
            error('Not enough validation samples at N=%d,K=%d,P=%d,SNR=%g.', ...
                cfg.N,cfg.K,cfg.P,SNR_grid(snr_index));
        end
        candidate_samples(config_index,snr_index) = numel(selected_indices);
        Y_selected = data.Y_data(:,:,selected_indices);
        G_selected = data.G_data(:,:,selected_indices);
        S_selected = data.S_data(:,:,selected_indices);
        B_selected = data.B_data(:,:,selected_indices);
        sigma2_selected = data.sigma2_data(selected_indices);
        sample_candidate_error = zeros(tuning_samples_per_condition,number_of_candidates);
        sample_energy = zeros(tuning_samples_per_condition,1);
        parfor (sample_position = 1:tuning_samples_per_condition,maximum_parallel_workers)
            Y = double(Y_selected(:,:,sample_position));
            G_true = double(G_selected(:,:,sample_position));
            S = double(S_selected(:,:,sample_position));
            B = double(B_selected(:,:,sample_position));
            sigma2 = double(sigma2_selected(sample_position));
            G0 = local_sweep_gs_initialize(Y,S,B,maximum_gs_updates,gs_convergence_tolerance);
            energy = sum(abs(G_true).^2,'all');
            sample_energy(sample_position) = energy;
            local_candidate_error = zeros(1,number_of_candidates);
            for candidate_index = 1:number_of_candidates
                tau_scale = local_noise_scale(sigma2,S,cfg.N,cfg.K,p_H);
                estimate = local_soft_rank_hsgs(G0,Y,S,B,sigma2, ...
                    threshold_multipliers(candidate_index)*tau_scale,p_H,number_of_stages);
                local_candidate_error(candidate_index) = sum(abs(estimate-G_true).^2,'all');
            end
            sample_candidate_error(sample_position,:) = local_candidate_error;
        end
        energy_sum(snr_index) = sum(sample_energy);
        candidate_error(snr_index,:) = sum(sample_candidate_error,1);
        for candidate_index = 1:number_of_candidates
            candidate_nmse(config_index,snr_index,candidate_index) = ...
                candidate_error(snr_index,candidate_index)/max(energy_sum(snr_index),realmin);
            candidate_row_index = candidate_row_index+1;
            candidate_rows{candidate_row_index} = {string(cfg.study),cfg.N,cfg.K,cfg.P, ...
                SNR_grid(snr_index),threshold_multipliers(candidate_index), ...
                candidate_nmse(config_index,snr_index,candidate_index),candidate_samples(config_index,snr_index)};
        end
        fprintf('Tuned candidates: %s N=%d K=%d P=%d SNR=%g dB.\n', ...
            cfg.study,cfg.N,cfg.K,cfg.P,SNR_grid(snr_index));
    end
end

% Policy A: strongest configuration-specific validation selection.
policy_rows = cell(number_of_configs*number_of_snr*3,1);
policy_row_index = 0;
policy_nmse = nan(number_of_configs,number_of_snr,3);
policy_lambda = nan(number_of_configs,number_of_snr,3);
policy_names = {'configuration_specific','sweep_wide_per_P','N8K3_transfer'};
for config_index = 1:number_of_configs
    cfg = configurations(config_index);
    for snr_index = 1:number_of_snr
        [policy_nmse(config_index,snr_index,1),candidate_index] = min( ...
            candidate_nmse(config_index,snr_index,:));
        policy_lambda(config_index,snr_index,1) = threshold_multipliers(candidate_index);
    end
end

% Policy B: one lambda per P within each sweep, selected from all dimensions
% and SNRs in that sweep. Each dimension/SNR validation condition has equal
% weight in this macro-average.
study_names = {'Antenna','User'};
for study_index = 1:numel(study_names)
    for P = P_list
        config_indices = find(strcmp({configurations.study},study_names{study_index}) & ...
            [configurations.P] == P);
        candidate_score = zeros(number_of_candidates,1);
        for candidate_index = 1:number_of_candidates
            values = candidate_nmse(config_indices,:,candidate_index);
            candidate_score(candidate_index) = mean(values(:));
        end
        [~,selected_candidate] = min(candidate_score);
        for config_index = config_indices
            policy_lambda(config_index,:,2) = threshold_multipliers(selected_candidate);
            policy_nmse(config_index,:,2) = candidate_nmse(config_index,:,selected_candidate);
        end
    end
end

% Policy C: tune only at the original N=8,K=3 configuration and transfer the
% resulting P/SNR-specific lambda unchanged across the matching sweep.
for P = P_list
    source_index = find(strcmp({configurations.study},'Antenna') & ...
        [configurations.N] == 8 & [configurations.K] == 3 & [configurations.P] == P,1);
    if isempty(source_index)
        error('Could not locate N=8,K=3 validation source for P=%d.',P);
    end
    for snr_index = 1:number_of_snr
        source_values = squeeze(candidate_nmse(source_index,snr_index,:));
        [~,selected_candidate] = min(source_values);
        selected_lambda = threshold_multipliers(selected_candidate);
        for config_index = find([configurations.P] == P)
            policy_lambda(config_index,snr_index,3) = selected_lambda;
            policy_nmse(config_index,snr_index,3) = candidate_nmse(config_index,snr_index,selected_candidate);
        end
    end
end

for config_index = 1:number_of_configs
    cfg = configurations(config_index);
    for snr_index = 1:number_of_snr
        for policy_index = 1:3
            policy_row_index = policy_row_index+1;
            policy_rows{policy_row_index} = {string(cfg.study),cfg.N,cfg.K,cfg.P, ...
                SNR_grid(snr_index),string(policy_names{policy_index}), ...
                policy_lambda(config_index,snr_index,policy_index), ...
                policy_nmse(config_index,snr_index,policy_index), ...
                candidate_samples(config_index,snr_index)};
        end
    end
end

candidate_table = cell2table(vertcat(candidate_rows{:}),'VariableNames', ...
    {'Study','N','K','P','SNR_dB','ThresholdMultiplier','ValidationPooledNMSE','SamplesUsed'});
policy_table = cell2table(vertcat(policy_rows{:}),'VariableNames', ...
    {'Study','N','K','P','SNR_dB','Policy','ThresholdMultiplier','ValidationPooledNMSEAtSelectedMultiplier','SamplesUsedForSelection'});
writetable(candidate_table,fullfile(output_root,'soft_rank_validation_results.csv'));
writetable(policy_table,fullfile(output_root,'soft_rank_selected_thresholds.csv'));
configuration = struct('N_antenna',[8 16 24 32],'K_antenna',3, ...
    'K_user_P10',[1 2 3 4 5],'K_user_P30',[1 2 3 4 5 6 7], ...
    'P_list',P_list,'SNR_grid',SNR_grid,'RSR_dB',RSR_dB, ...
    'p_H_rule',p_H_rule,'threshold_multipliers',threshold_multipliers, ...
    'tuning_samples_per_condition',tuning_samples_per_condition, ...
    'maximum_gs_updates',maximum_gs_updates,'gs_convergence_tolerance',gs_convergence_tolerance, ...
    'number_of_stages',number_of_stages,'selection_seed',selection_seed, ...
    'maximum_parallel_workers',maximum_parallel_workers, ...
    'policy_names',{policy_names},'test_data_loaded',false);
save(fullfile(output_root,'experiment_configuration.mat'),'configuration','configurations', ...
    'candidate_nmse','candidate_samples','policy_lambda','policy_nmse','threshold_multipliers','-v7.3');

fprintf('\nValidation threshold selection complete. No test files were loaded.\n');
fprintf('Candidate validation table: %s\n',fullfile(output_root,'soft_rank_validation_results.csv'));
fprintf('Frozen policy table: %s\n',fullfile(output_root,'soft_rank_selected_thresholds.csv'));
fprintf('Policy C is the transfer policy; A/B use validation from swept configurations.\n');

function local_validate_split(data,cfg,SNR_grid,RSR_dB)
if data.N ~= cfg.N || data.K ~= cfg.K || data.P ~= cfg.P || data.RSR_dB ~= RSR_dB || ...
        ~isequal(double(data.SNR_dB(:)).',SNR_grid) || ~strcmp(data.split_name,'validation')
    error('Validation metadata mismatch for %s N=%d K=%d P=%d.',cfg.study,cfg.N,cfg.K,cfg.P);
end
sample_count = data.number_of_samples;
if ~isequal(size(data.Y_data),[cfg.N cfg.P sample_count]) || ...
        ~isequal(size(data.G_data),[cfg.N cfg.K sample_count]) || ...
        ~isequal(size(data.S_data),[cfg.K cfg.P sample_count]) || ...
        ~isequal(size(data.B_data),[cfg.N cfg.P sample_count]) || ...
        numel(data.sigma2_data) ~= sample_count || numel(data.SNR_dB_data) ~= sample_count
    error('Validation tensor dimensions or per-sample metadata mismatch for %s N=%d K=%d P=%d.', ...
        cfg.study,cfg.N,cfg.K,cfg.P);
end
for snr_index = 1:numel(SNR_grid)
    if nnz(double(data.SNR_dB_data(:)) == SNR_grid(snr_index)) < 30
        error('Validation file has too few samples at SNR=%g.',SNR_grid(snr_index));
    end
end
end

function G = local_sweep_gs_initialize(Y,S,B,maximum_updates,tolerance)
G = local_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
for iteration = 1:maximum_updates
    previous = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        break;
    end
end
end

function G = local_spectral_initialize(Y,S,B)
N = size(Y,1);
K = size(S,1);
G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).';
    b = B(antenna,:).';
    augmented = [S.',b];
    spectral_matrix = augmented'*diag(y)*augmented;
    [vectors,values] = eig(spectral_matrix);
    [~,principal_index] = max(real(diag(values)));
    principal = vectors(:,principal_index);
    principal = principal/max(norm(principal),eps);
    scale = (abs(augmented*principal).'*y)/max(norm(augmented*principal)^2,eps);
    G(antenna,:) = (scale*exp(-1i*angle(principal(K+1)))*principal(1:K)).';
end
end

function scale = local_noise_scale(sigma2,S,N,K,p_H)
right_inverse = S'/(S*S');
scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
end

function G = local_emgs_update(G,Y,S,B,sigma2)
right_inverse = S'/(S*S');
field = G*S+B;
kappa = 2/max(sigma2,eps)*Y.*abs(field);
ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
G = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
end

function G = local_soft_rank_hsgs(G,Y,S,B,sigma2,threshold,p_H,number_of_stages)
for stage = 1:number_of_stages
    G_em = local_emgs_update(G,Y,S,B,sigma2);
    G = local_soft_hankel_projection(G_em,p_H,threshold);
end
end

function G = local_soft_hankel_projection(G,p_H,threshold)
[N,K] = size(G);
rows = N-p_H;
columns = p_H+1;
for user = 1:K
    vector = G(:,user);
    hankel = zeros(rows,columns);
    for row = 1:rows
        hankel(row,:) = vector(row:row+columns-1).';
    end
    [U,Sigma,V] = svd(hankel,'econ');
    shrunk = max(diag(Sigma)-threshold,0);
    hankel_shrunk = U*diag(shrunk)*V';
    reconstructed = zeros(N,1);
    counts = zeros(N,1);
    for row = 1:rows
        for column = 1:columns
            antenna = row+column-1;
            reconstructed(antenna) = reconstructed(antenna)+hankel_shrunk(row,column);
            counts(antenna) = counts(antenna)+1;
        end
    end
    G(:,user) = reconstructed./counts;
end
end

function has_entries = local_has_entries(folder)
entries = dir(fullfile(folder,'*'));
entries = entries(~ismember({entries.name},{'.','..'}));
has_entries = ~isempty(entries);
end
