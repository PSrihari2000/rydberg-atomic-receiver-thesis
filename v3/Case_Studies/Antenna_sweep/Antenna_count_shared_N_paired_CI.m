%% Paired uncertainty analysis on the already-used shared-N test split
% This is a post-test inferential analysis only. It does not select or alter
% models. It re-scores the same test observations to recover paired sample
% NMSEs, then bootstraps the ratio of arithmetic mean linear NMSE values.

clear;
clc;

N_list = [8 16 24 32];
K = 3;
P_list = [10 30];
SNR_grid = [-10 0 10 20 30];
RSR_dB = 25;
number_of_samples_per_condition = 500;
number_of_bootstrap_replicates = 2000;
maximum_updates = 10000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
maximum_parallel_workers = 6;

script_folder = fileparts(mfilename('fullpath'));
repository_folder = fileparts(fileparts(script_folder));
fresh_root = fullfile(script_folder, 'data', 'ANTENNA_COUNT_SHARED_N_FRESH');
model_root = fullfile(script_folder, 'results', 'ANTENNA_COUNT_SHARED_N_PI_DU_RACE_FROM_V4');
v4_model_root = fullfile(repository_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
policy_file = fullfile(script_folder, 'results', 'ANTENNA_COUNT_CLASSICAL', 'validation_policy', 'antenna_sweep_hard_rank_policy.mat');
test_result_file = fullfile(script_folder, 'results', 'ANTENNA_COUNT_SHARED_N_TEST_EVALUATION', 'shared_N_test_results.mat');
output_folder = fullfile(script_folder, 'results', 'ANTENNA_COUNT_SHARED_N_PAIRED_CI');

if ~isfolder(fresh_root) || ~isfolder(model_root) || ~isfolder(v4_model_root) || ~isfile(policy_file) || ~isfile(test_result_file)
    error('Required fresh test data, selected models, frozen rank policy, or original test results are missing.');
end
if isfolder(output_folder)
    existing_entries = dir(output_folder);
    existing_names = string({existing_entries.name});
    existing_names = existing_names(~ismember(existing_names, ["." ".."]));
    allowed_checkpoint_names = strings(0, 1);
    for N_value = N_list
        for P_value = P_list
            allowed_checkpoint_names(end+1, 1) = sprintf('paired_sample_nmse_N%d_P%d.mat', N_value, P_value);
        end
    end
    allowed_output_names = ["shared_N_paired_test_sample_nmse.csv"; ...
        "shared_N_paired_CI_by_condition.csv"; ...
        "shared_N_paired_CI_SNRavg.csv"; ...
        "shared_N_paired_CI_results.mat"];
    if any(~ismember(existing_names, [allowed_checkpoint_names; allowed_output_names]))
        error('Paired-CI output contains final or unexpected files. Inspect them before rerunning: %s', output_folder);
    end
    fprintf('Using paired-CI output folder; compatible checkpoints will be resumed and known summary outputs refreshed.\n');
else
    mkdir(output_folder);
end

policy = load(policy_file, 'hard_rank_policy', 'N_list', 'K_list', 'P_list', 'SNR_dB', 'RSR_dB', 'maximum_updates', 'convergence_tolerance');
if ~isequal(policy.N_list, N_list) || policy.K_list ~= K || ~isequal(policy.P_list, P_list) || ~isequal(policy.SNR_dB, SNR_grid) || policy.RSR_dB ~= RSR_dB
    error('Frozen rank-policy metadata does not match this test grid.');
end
maximum_updates = policy.maximum_updates;
convergence_tolerance = policy.convergence_tolerance;
method_names = {'V4 weights (N=8)','Shared-N DL','Hard-rank HS-GS'};
number_of_methods = numel(method_names);
number_of_conditions = numel(N_list)*numel(P_list)*numel(SNR_grid);
condition_rows = zeros(number_of_conditions*2, 11);
summary_rows = zeros(numel(N_list)*numel(P_list)*2, 9);
condition_row = 0;
summary_row = 0;
all_sample_rows = zeros(number_of_conditions*number_of_samples_per_condition, 8);
sample_row = 0;

fprintf('Paired test uncertainty analysis only; no model selection or tuning.\n');
fprintf('Re-scoring %d samples per N/P/SNR cell; %d bootstrap replicates.\n', number_of_samples_per_condition, number_of_bootstrap_replicates);
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes', maximum_parallel_workers);
else
    maximum_parallel_workers = min(maximum_parallel_workers, pool.NumWorkers);
end
fprintf('Scoring with up to %d parallel workers.\n', maximum_parallel_workers);

for P_index = 1:numel(P_list)
    P = P_list(P_index);
    selected_model_file = fullfile(model_root, sprintf('selected_shared_N_model_P%d.mat', P));
    v4_model_file = fullfile(v4_model_root, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    shared = load(selected_model_file, 'best_parameters_numeric', 'N_supported', 'K', 'P', 'SNR_grid', 'RSR_dB', 'number_of_stages', 'maximum_gs_updates', 'gs_convergence_tolerance', 'data_version_validation');
    v4 = load(v4_model_file, 'best_parameters_numeric', 'N', 'K', 'P', 'RSR_dB', 'number_of_stages', 'initialization_method');
    if ~isequal(shared.N_supported, N_list) || shared.K ~= K || shared.P ~= P || ~isequal(shared.SNR_grid, SNR_grid) || shared.RSR_dB ~= RSR_dB || ~strcmp(shared.data_version_validation, 'V3_ANTENNA_COUNT_SHARED_N_FRESH_V1')
        error('Shared-N selected-model metadata mismatch for P=%d.', P);
    end
    if v4.N ~= 8 || v4.K ~= K || v4.P ~= P || v4.RSR_dB ~= RSR_dB || v4.number_of_stages ~= shared.number_of_stages || ~strcmp(v4.initialization_method, 'GS')
        error('Frozen V4 model metadata mismatch for P=%d.', P);
    end

    for N_index = 1:numel(N_list)
        N = N_list(N_index);
        p_H = floor((N-1)/2);
        test_file = fullfile(fresh_root, sprintf('N%d_K%d_RSR25dB', N, K), sprintf('test_P%d.mat', P));
        test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version', 'split_name', 'samples_per_SNR');
        if test_data.N ~= N || test_data.K ~= K || test_data.P ~= P || test_data.RSR_dB ~= RSR_dB || ~isequal(double(test_data.SNR_dB), SNR_grid) || ~strcmp(test_data.data_version, 'V3_ANTENNA_COUNT_SHARED_N_FRESH_V1') || ~strcmp(test_data.split_name, 'test') || test_data.samples_per_SNR ~= number_of_samples_per_condition || test_data.number_of_samples ~= number_of_samples_per_condition*numel(SNR_grid)
            error('Test split metadata mismatch for N=%d,P=%d.', N, P);
        end

        checkpoint_file = fullfile(output_folder, sprintf('paired_sample_nmse_N%d_P%d.mat', N, P));
        if isfile(checkpoint_file)
            checkpoint = load(checkpoint_file, 'sample_nmse_by_index', 'sample_snr_index_by_index', 'N_saved', 'K_saved', 'P_saved', 'RSR_dB_saved', 'SNR_grid_saved', 'number_of_samples_per_condition_saved');
            if checkpoint.N_saved ~= N || checkpoint.K_saved ~= K || checkpoint.P_saved ~= P || checkpoint.RSR_dB_saved ~= RSR_dB || ~isequal(checkpoint.SNR_grid_saved, SNR_grid) || checkpoint.number_of_samples_per_condition_saved ~= number_of_samples_per_condition || ~isequal(size(checkpoint.sample_nmse_by_index), [test_data.number_of_samples number_of_methods]) || numel(checkpoint.sample_snr_index_by_index) ~= test_data.number_of_samples
                error('Paired-score checkpoint metadata mismatch for N=%d,P=%d. Preserve it and use a new output folder.', N, P);
            end
            sample_nmse_by_index = checkpoint.sample_nmse_by_index;
            sample_snr_index_by_index = checkpoint.sample_snr_index_by_index;
            fprintf('Resumed saved paired scores: N=%d, P=%d.\n', N, P);
        else
            sample_nmse_by_index = zeros(test_data.number_of_samples, number_of_methods);
            sample_snr_index_by_index = zeros(test_data.number_of_samples, 1);
            fprintf('Re-scoring paired test samples in parallel: N=%d, P=%d.\n', N, P);
            parfor (sample_index = 1:test_data.number_of_samples, maximum_parallel_workers)
                snr_db = double(test_data.SNR_dB_data(sample_index));
                snr_index = find(SNR_grid == snr_db, 1);
                if isempty(snr_index)
                    error('Unexpected SNR value for test sample %d.', sample_index);
                end

                Y = double(test_data.Y_data(:, :, sample_index));
                G_true = double(test_data.G_data(:, :, sample_index));
                S = double(test_data.S_data(:, :, sample_index));
                B = double(test_data.B_data(:, :, sample_index));
                sigma2 = double(test_data.sigma2_data(sample_index));
                G0 = local_biased_gs(Y, S, B, shared.maximum_gs_updates, shared.gs_convergence_tolerance);
                G_v4 = local_unfolded_forward(v4.best_parameters_numeric, Y, S, B, sigma2, G0, N, K, p_H, shared.number_of_stages);
                G_shared = local_unfolded_forward(shared.best_parameters_numeric, Y, S, B, sigma2, G0, N, K, p_H, shared.number_of_stages);
                selected_rank = policy.hard_rank_policy(N_index, P_index, snr_index);
                G_hard_rank = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, selected_rank, maximum_updates, convergence_tolerance);
                truth_energy = max(sum(abs(G_true).^2, 'all'), eps);
                sample_nmse_by_index(sample_index, :) = [sum(abs(G_v4-G_true).^2, 'all') sum(abs(G_shared-G_true).^2, 'all') sum(abs(G_hard_rank-G_true).^2, 'all')]/truth_energy;
                sample_snr_index_by_index(sample_index) = snr_index;
            end
            N_saved = N;
            K_saved = K;
            P_saved = P;
            RSR_dB_saved = RSR_dB;
            SNR_grid_saved = SNR_grid;
            number_of_samples_per_condition_saved = number_of_samples_per_condition;
            save(checkpoint_file, 'sample_nmse_by_index', 'sample_snr_index_by_index', 'N_saved', 'K_saved', 'P_saved', 'RSR_dB_saved', 'SNR_grid_saved', 'number_of_samples_per_condition_saved', '-v7.3');
            fprintf('Saved paired-score checkpoint: %s\n', checkpoint_file);
        end

        nmse_by_snr_and_method = zeros(number_of_samples_per_condition, numel(SNR_grid), number_of_methods);

        sample_position_by_snr = zeros(1, numel(SNR_grid));
        for sample_index = 1:test_data.number_of_samples
            snr_index = sample_snr_index_by_index(sample_index);
            snr_db = SNR_grid(snr_index);
            sample_position_by_snr(snr_index) = sample_position_by_snr(snr_index)+1;
            position = sample_position_by_snr(snr_index);
            nmse_by_snr_and_method(position, snr_index, :) = reshape(sample_nmse_by_index(sample_index, :), 1, 1, []);

            sample_row = sample_row+1;
            sample_nmse = sample_nmse_by_index(sample_index, :);
            all_sample_rows(sample_row, :) = [N K P snr_db sample_index sample_nmse];
        end
        if any(sample_position_by_snr ~= number_of_samples_per_condition)
            error('Expected exactly %d samples per SNR for N=%d,P=%d.', number_of_samples_per_condition, N, P);
        end

        %% Pointwise paired bootstrap intervals by SNR
        for snr_index = 1:numel(SNR_grid)
            for baseline_index = [1 3]
                baseline_nmse = nmse_by_snr_and_method(:, snr_index, baseline_index);
                shared_nmse = nmse_by_snr_and_method(:, snr_index, 2);
                point_gain = 10*log10(mean(baseline_nmse)/mean(shared_nmse));
                bootstrap_gain = zeros(number_of_bootstrap_replicates, 1);
                rng(2026114000+100000*N+1000*P+10*SNR_grid(snr_index)+baseline_index, 'twister');
                for bootstrap_index = 1:number_of_bootstrap_replicates
                    resample = randi(number_of_samples_per_condition, number_of_samples_per_condition, 1);
                    bootstrap_gain(bootstrap_index) = 10*log10(mean(baseline_nmse(resample))/mean(shared_nmse(resample)));
                end
                interval = local_percentile(bootstrap_gain, [2.5 97.5]);
                condition_row = condition_row+1;
                condition_rows(condition_row, :) = [N K P SNR_grid(snr_index) baseline_index number_of_samples_per_condition mean(baseline_nmse) mean(shared_nmse) point_gain interval];
            end
        end

        %% Paired bootstrap interval for equal-SNR mean linear NMSE
        for baseline_index = [1 3]
            baseline_by_snr = squeeze(nmse_by_snr_and_method(:, :, baseline_index));
            shared_by_snr = squeeze(nmse_by_snr_and_method(:, :, 2));
            point_baseline = mean(mean(baseline_by_snr, 1));
            point_shared = mean(mean(shared_by_snr, 1));
            point_gain = 10*log10(point_baseline/point_shared);
            bootstrap_gain = zeros(number_of_bootstrap_replicates, 1);
            rng(2026115000+100000*N+1000*P+baseline_index, 'twister');
            for bootstrap_index = 1:number_of_bootstrap_replicates
                resampled_baseline_means = zeros(1, numel(SNR_grid));
                resampled_shared_means = zeros(1, numel(SNR_grid));
                for snr_index = 1:numel(SNR_grid)
                    resample = randi(number_of_samples_per_condition, number_of_samples_per_condition, 1);
                    resampled_baseline_means(snr_index) = mean(baseline_by_snr(resample, snr_index));
                    resampled_shared_means(snr_index) = mean(shared_by_snr(resample, snr_index));
                end
                bootstrap_gain(bootstrap_index) = 10*log10(mean(resampled_baseline_means)/mean(resampled_shared_means));
            end
            interval = local_percentile(bootstrap_gain, [2.5 97.5]);
            summary_row = summary_row+1;
            summary_rows(summary_row, :) = [N K P baseline_index point_baseline point_shared point_gain interval];
        end
    end
end

sample_nmse_table = array2table(all_sample_rows, 'VariableNames', {'N','K','P','SNR_dB','SampleIndex','V4_NMSE_linear','SharedN_NMSE_linear','HardRankHSGS_NMSE_linear'});
writetable(sample_nmse_table, fullfile(output_folder, 'shared_N_paired_test_sample_nmse.csv'));

condition_table = array2table(condition_rows, 'VariableNames', {'N','K','P','SNR_dB','BaselineIndex','Samples','BaselineMeanNMSE_linear','SharedNMeanNMSE_linear','SharedN_Gain_dB','CI95_Lower_dB','CI95_Upper_dB'});
condition_table.Baseline = reshape(method_names(condition_table.BaselineIndex), [], 1);
condition_table = movevars(condition_table, 'Baseline', 'After', 'SNR_dB');
writetable(condition_table, fullfile(output_folder, 'shared_N_paired_CI_by_condition.csv'));

summary_table = array2table(summary_rows, 'VariableNames', {'N','K','P','BaselineIndex','BaselineMeanLinearNMSE_over_SNR','SharedNMeanLinearNMSE_over_SNR','SharedN_Gain_dB','CI95_Lower_dB','CI95_Upper_dB'});
summary_table.Baseline = reshape(method_names(summary_table.BaselineIndex), [], 1);
summary_table = movevars(summary_table, 'Baseline', 'After', 'P');
writetable(summary_table, fullfile(output_folder, 'shared_N_paired_CI_SNRavg.csv'));

save(fullfile(output_folder, 'shared_N_paired_CI_results.mat'), 'sample_nmse_table', 'condition_table', 'summary_table', 'method_names', 'N_list', 'K', 'P_list', 'SNR_grid', 'RSR_dB', 'number_of_bootstrap_replicates', 'maximum_updates', 'convergence_tolerance', 'policy_file');
fprintf('Paired bootstrap analysis complete. Positive gain favors shared-N DL.\n');
fprintf('Results: %s\n', output_folder);
fprintf('This analysis did not change model selection or parameters.\n');

function interval = local_percentile(values, probabilities)
values = sort(values(:));
positions = 1+(numel(values)-1)*probabilities/100;
interval = zeros(size(probabilities));
for index = 1:numel(probabilities)
    lower_index = floor(positions(index));
    upper_index = ceil(positions(index));
    fraction = positions(index)-lower_index;
    interval(index) = values(lower_index)*(1-fraction)+values(upper_index)*fraction;
end
end

function G = local_biased_gs(Y,S,B,maximum_updates,tolerance)
G = local_spectral_initialize(Y,S,B); right_inverse = S'/(S*S');
for iteration = 1:maximum_updates
    previous = G; G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance, break; end
end
end

function G = local_hard_rank_hsgs(Y,S,B,sigma2,p_H,rank,maximum_updates,tolerance)
N = size(Y,1); K = size(S,1); right_inverse = S'/(S*S'); G = local_spectral_initialize(Y,S,B);
for iteration = 1:maximum_updates
    previous = G; field = G*S+B; kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G_emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    projected = zeros(N,K); rows = N-p_H; columns = p_H+1;
    for user = 1:K
        channel = G_emgs(:,user); hankel = zeros(rows,columns);
        for row = 1:rows, hankel(row,:) = channel(row:row+columns-1).'; end
        [U,Sigma,V] = svd(hankel,'econ'); hankel = U(:,1:rank)*Sigma(1:rank,1:rank)*V(:,1:rank)';
        reconstructed = zeros(N,1); counts = zeros(N,1);
        for row = 1:rows
            for column = 1:columns
                antenna = row+column-1; reconstructed(antenna) = reconstructed(antenna)+hankel(row,column); counts(antenna) = counts(antenna)+1;
            end
        end
        projected(:,user) = reconstructed./counts;
    end
    G = projected;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance, break; end
end
end

function G = local_unfolded_forward(parameters,Y,S,B,sigma2,G0,N,K,p_H,number_of_stages)
right_inverse = S'/(S*S'); G = G0;
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2,'all'),eps),eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
map = local_hankel_map(N,p_H); counts = sum(map,1).';
for stage = 1:number_of_stages
    previous = G; field = G*S+B; kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    hankel = zeros(N,K);
    for user = 1:K
        H = reshape(map*emgs(:,user),N-p_H,p_H+1); gram = H'*H;
        [vectors,values] = eig(gram); singular = sqrt(max(real(sum(values,2)),0)); safe = max(singular,1e-12);
        relative = tanh(log(safe/max(mean(safe),1e-12)));
        logits = parameters(1,stage)+parameters(2,stage)*noise_feature+parameters(3,stage)*relative;
        thresholds = noise_scale*(2./(1+exp(-logits))); factors = max(1-thresholds./safe,0);
        H_shrunk = H*(vectors.*reshape(factors,1,[]))*vectors';
        hankel(:,user) = (map'*H_shrunk(:))./counts;
    end
    emgs_change = tanh(log(1+sqrt(sum(abs(emgs-previous).^2,'all')/max(sum(abs(previous).^2,'all'),eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(hankel-emgs).^2,'all')/max(sum(abs(emgs).^2,'all'),eps)))/3);
    coefficients = reshape(parameters(4:15,stage),4,3);
    gate_logits = coefficients(1,:).'*noise_feature+coefficients(2,:).'*emgs_change+coefficients(3,:).'*hankel_change+coefficients(4,:).';
    weights = exp(gate_logits-max(gate_logits)); weights = weights/sum(weights);
    G = weights(1)*previous+weights(2)*emgs+weights(3)*hankel;
end
end

function map = local_hankel_map(N,p_H)
rows = N-p_H; columns = p_H+1; map = zeros(rows*columns,N);
for row = 1:rows
    for column = 1:columns
        map(row+(column-1)*rows,row+column-1) = 1;
    end
end
end

function G = local_spectral_initialize(Y,S,B)
N = size(Y,1); K = size(S,1); G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).'; b = B(antenna,:).'; augmented = [S.',b];
    matrix = augmented'*diag(y)*augmented; [vectors,values] = eig(matrix);
    [~,index] = max(real(diag(values))); vector = vectors(:,index); vector = vector/max(norm(vector),eps);
    scale = (abs(augmented*vector).'*y)/max(norm(augmented*vector)^2,eps);
    G(antenna,:) = (scale*exp(-1i*angle(vector(K+1)))*vector(1:K)).';
end
end
