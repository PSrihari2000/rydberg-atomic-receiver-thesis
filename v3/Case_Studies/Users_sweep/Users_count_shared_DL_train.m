%% Shared PI-DU-RACE model training across user counts
% Trains one 16-stage model for P=10 across K=1:5 and one for P=30 across
% K=1:7. Each model is warm-started from the frozen V4 model at N=8,K=3.
% Training and validation files only are loaded. No test file is opened.

clear;
clc;
close all;

%% -------------------- FROZEN CONFIGURATION --------------------
N = 8;
P_list = [10 30];
K_by_P = {[1 2 3 4 5], [1 2 3 4 5 6 7]};
SNR_grid = [-10 0 10 20 30];
RSR_dB = 25;
number_of_stages = 16;
parameters_per_stage = 15;
number_of_gate_features = 3;
number_of_gate_outputs = 3;
number_of_shrinkage_parameters = 3;
initialization_method = 'GS';
maximum_gs_updates = 5000;
gs_convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;

%% -------------------- TRAINING SETTINGS --------------------
training_samples_per_snr_per_K_per_epoch = 80;
validation_samples_per_snr_per_K_for_selection = 32;
batch_size = 8;
maximum_epochs = 12;
early_stopping_patience = 4;
learning_rate = 0.001;
beta1 = 0.9;
beta2 = 0.999;
gradient_limit = 5;
epsilon = 1e-8;

%% -------------------- PATHS --------------------
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
repository_folder = fileparts(case_studies_folder);
scripts_folder = fullfile(repository_folder, 'scripts');
scenario_name = 'USER_COUNT_RECOMMENDED_P10_K1_2_3_4_5_P30_K1_2_3_4_5_6_7_RSR25_SNR_m10_0_10_20_30';
data_version = 'V3_USER_COUNT_RECOMMENDED_SWEEP_V1';
scenario_folder = fullfile(script_folder, 'data', scenario_name);
warm_start_folder = fullfile(repository_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
output_folder = fullfile(script_folder, 'results', 'USER_COUNT_SHARED_PI_DU_RACE_K_SWEEP_FROM_V4');

if ~isfolder(scenario_folder)
    error('Recommended-sweep data folder not found: %s', scenario_folder);
end
if ~isfolder(warm_start_folder)
    error('Frozen V4 model folder not found: %s', warm_start_folder);
end
if ~isfile(fullfile(scripts_folder, 'V3_BesselRatioFunction.m')) || ~isfile(fullfile(scripts_folder, 'V3_CLASSICAL_ESTIMATOR.m'))
    error('Required estimator functions are missing from %s.', scripts_folder);
end
if isfolder(output_folder)
    existing_entries = dir(output_folder);
    existing_names = string({existing_entries.name});
    existing_names = existing_names(~ismember(existing_names, ["." ".."]));
    allowed_checkpoints = ["checkpoint_P10.mat" "checkpoint_P30.mat"];
    completed_outputs = contains(existing_names, "selected_shared_model_") | contains(existing_names, "validation_by_K_SNR_") | contains(existing_names, "training_history_");
    unexpected_entries = ~ismember(existing_names, allowed_checkpoints);
    if any(completed_outputs) || any(unexpected_entries)
        error('Output folder contains completed or unexpected files. Preserve and inspect it; choose a new output-folder name to start a separate run: %s', output_folder);
    end
else
    mkdir(output_folder);
end
addpath(scripts_folder);

fprintf('Shared 16-stage PI-DU-RACE training across user counts.\n');
fprintf('N=%d; P=10 K=%s; P=30 K=%s; SNR=%s dB; RSR=%g dB.\n', N, mat2str(K_by_P{1}), mat2str(K_by_P{2}), mat2str(SNR_grid), RSR_dB);
fprintf('Trainable weights per shared model: %d. Models: one per P.\n', parameters_per_stage*number_of_stages);
fprintf('Initialization: frozen V4 selected model for the matching P.\n');
fprintf('Only train and validation splits will be loaded.\n');
fprintf('Output folder: %s\n', output_folder);

%% -------------------- TRAIN A SHARED MODEL FOR EACH PILOT COUNT --------------------
for P_index = 1:numel(P_list)
    P = P_list(P_index);
    K_list = K_by_P{P_index};
    number_of_configurations = numel(K_list);
    warm_start_file = fullfile(warm_start_folder, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    if ~isfile(warm_start_file)
        error('Frozen V4 warm-start model is missing: %s', warm_start_file);
    end

    warm_start_data = load(warm_start_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs', 'initialization_method', 'maximum_gs_updates', 'gs_convergence_tolerance');
    if warm_start_data.N ~= N || warm_start_data.K ~= 3 || warm_start_data.P ~= P || warm_start_data.RSR_dB ~= RSR_dB || warm_start_data.p_H ~= floor((N-1)/2) || warm_start_data.number_of_stages ~= number_of_stages || warm_start_data.parameters_per_stage ~= parameters_per_stage || warm_start_data.number_of_gate_features ~= number_of_gate_features || warm_start_data.number_of_gate_outputs ~= number_of_gate_outputs || ~strcmp(warm_start_data.initialization_method, initialization_method) || ~isequal(size(warm_start_data.best_parameters_numeric), [parameters_per_stage number_of_stages])
        error('Frozen V4 model metadata or parameter shape does not match shared-model settings for P=%d.', P);
    end
    p_H = floor((N-1)/2);
    parameters = dlarray(double(warm_start_data.best_parameters_numeric));
    parameters_initial = double(warm_start_data.best_parameters_numeric);
    average_gradient = zeros(size(parameters_initial));
    average_squared_gradient = zeros(size(parameters_initial));
    best_parameters = parameters;
    best_validation_loss = inf;
    best_epoch = 0;
    epochs_without_improvement = 0;
    parameter_update_count = 0;
    history = zeros(maximum_epochs, 6);
    start_epoch = 1;
    checkpoint_file = fullfile(output_folder, sprintf('checkpoint_P%d.mat', P));
    if isfile(checkpoint_file)
        checkpoint_data = load(checkpoint_file);
        if checkpoint_data.P ~= P || checkpoint_data.N ~= N || ~isequal(checkpoint_data.K_list, K_list) || checkpoint_data.RSR_dB ~= RSR_dB || ~isequal(checkpoint_data.SNR_grid, SNR_grid) || checkpoint_data.number_of_stages ~= number_of_stages || checkpoint_data.parameters_per_stage ~= parameters_per_stage || checkpoint_data.training_samples_per_snr_per_K_per_epoch ~= training_samples_per_snr_per_K_per_epoch || checkpoint_data.validation_samples_per_snr_per_K_for_selection ~= validation_samples_per_snr_per_K_for_selection || checkpoint_data.batch_size ~= batch_size || checkpoint_data.learning_rate ~= learning_rate || ~strcmp(checkpoint_data.warm_start_file, warm_start_file)
            error('Checkpoint metadata mismatch for P=%d. Preserve it and use a new output folder.', P);
        end
        parameters = dlarray(checkpoint_data.current_parameters);
        if isfield(checkpoint_data, 'average_gradient')
            average_gradient = checkpoint_data.average_gradient;
            average_squared_gradient = checkpoint_data.average_squared_gradient;
        else
            warning('P=%d checkpoint predates optimizer-state saving. Resuming from its current weights with Adam moments reset.', P);
        end
        best_parameters = dlarray(checkpoint_data.best_parameters_saved);
        best_validation_loss = checkpoint_data.best_validation_loss;
        best_epoch = checkpoint_data.best_epoch;
        if isfield(checkpoint_data, 'epochs_without_improvement')
            epochs_without_improvement = checkpoint_data.epochs_without_improvement;
        else
            epochs_without_improvement = 0;
        end
        parameter_update_count = checkpoint_data.parameter_update_count;
        history_saved = checkpoint_data.history_saved;
        history(1:size(history_saved, 1), :) = history_saved;
        start_epoch = checkpoint_data.epoch+1;
        fprintf('Resuming P=%d from checkpoint after epoch %d; next epoch=%d.\n', P, checkpoint_data.epoch, start_epoch);
    end

    train_data_by_K = cell(1, number_of_configurations);
    validation_data_by_K = cell(1, number_of_configurations);
    training_initial_cache = cell(1, number_of_configurations);
    training_initial_ready = cell(1, number_of_configurations);
    validation_initial_cache = cell(1, number_of_configurations);
    validation_initial_ready = cell(1, number_of_configurations);
    selection_indices_by_K = cell(1, number_of_configurations);

    %% Load only the matching K-specific train and validation splits
    for configuration_index = 1:number_of_configurations
        K = K_list(configuration_index);
        dimension_folder = fullfile(scenario_folder, sprintf('N%d_K%d_P%d_RSR25dB', N, K, P));
        train_file = fullfile(dimension_folder, 'train.mat');
        validation_file = fullfile(dimension_folder, 'validation.mat');
        if ~isfile(train_file) || ~isfile(validation_file)
            error('Missing train or validation split for N=%d,K=%d,P=%d.', N, K, P);
        end
        train_data_by_K{configuration_index} = load(train_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version');
        validation_data_by_K{configuration_index} = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version');
        local_validate_dimension_data(train_data_by_K{configuration_index}, N, K, P, SNR_grid, RSR_dB, 2000);
        local_validate_dimension_data(validation_data_by_K{configuration_index}, N, K, P, SNR_grid, RSR_dB, 500);

        train_count = train_data_by_K{configuration_index}.number_of_samples;
        validation_count = validation_data_by_K{configuration_index}.number_of_samples;
        training_initial_cache{configuration_index} = complex(zeros(N, K, train_count));
        training_initial_ready{configuration_index} = false(1, train_count);
        validation_initial_cache{configuration_index} = complex(zeros(N, K, validation_count));
        validation_initial_ready{configuration_index} = false(1, validation_count);
        selection_indices = [];
        rng(2026120100+1000*K+P, 'twister');
        for snr_index = 1:numel(SNR_grid)
            snr_candidates = find(double(validation_data_by_K{configuration_index}.SNR_dB_data) == SNR_grid(snr_index));
            if numel(snr_candidates) < validation_samples_per_snr_per_K_for_selection
                error('Insufficient validation samples for K=%d,P=%d,SNR=%g.', K, P, SNR_grid(snr_index));
            end
            snr_candidates = snr_candidates(randperm(numel(snr_candidates)));
            selection_indices = [selection_indices; snr_candidates(1:validation_samples_per_snr_per_K_for_selection)];
        end
        selection_indices_by_K{configuration_index} = selection_indices;

        %% Cache GS initializations for the validation subset only
        for selection_position = 1:numel(selection_indices)
            sample_index = selection_indices(selection_position);
            Y = double(validation_data_by_K{configuration_index}.Y_data(:, :, sample_index));
            S = double(validation_data_by_K{configuration_index}.S_data(:, :, sample_index));
            B = double(validation_data_by_K{configuration_index}.B_data(:, :, sample_index));
            sigma2 = double(validation_data_by_K{configuration_index}.sigma2_data(sample_index));
            [validation_initial_cache{configuration_index}(:, :, sample_index), ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', Y, S, B, sigma2, p_H, 0, maximum_gs_updates, gs_convergence_tolerance, gd_initial_variance, sample_index);
            validation_initial_ready{configuration_index}(sample_index) = true;
        end
        fprintf('Loaded train/validation for P=%d,K=%d. Validation selection samples=%d.\n', P, K, numel(selection_indices));
    end

    fprintf('\n============================================================\n');
    fprintf('Shared model training for P=%d over K=%s; V4 warm-start epoch=%d.\n', P, mat2str(K_list), warm_start_data.best_epoch_saved);

    %% Balanced mixed-dimension training
    epoch = start_epoch-1;
    for epoch = start_epoch:maximum_epochs
        epoch_start_time = tic;
        batch_schedule = struct('configuration_index', {}, 'sample_indices', {});
        rng(2026121100+100*P+epoch, 'twister');
        for configuration_index = 1:number_of_configurations
            train_data = train_data_by_K{configuration_index};
            for snr_index = 1:numel(SNR_grid)
                snr_candidates = find(double(train_data.SNR_dB_data) == SNR_grid(snr_index));
                if numel(snr_candidates) < training_samples_per_snr_per_K_per_epoch
                    error('Insufficient training samples for K=%d,P=%d,SNR=%g.', K_list(configuration_index), P, SNR_grid(snr_index));
                end
                snr_candidates = snr_candidates(randperm(numel(snr_candidates)));
                selected_samples = snr_candidates(1:training_samples_per_snr_per_K_per_epoch);
                for batch_start = 1:batch_size:training_samples_per_snr_per_K_per_epoch
                    batch_end = min(batch_start+batch_size-1, training_samples_per_snr_per_K_per_epoch);
                    batch_schedule(end+1) = struct('configuration_index', configuration_index, 'sample_indices', selected_samples(batch_start:batch_end));
                end
            end
        end
        batch_schedule = batch_schedule(randperm(numel(batch_schedule)));
        batch_loss_sum = 0;

        for batch_index = 1:numel(batch_schedule)
            configuration_index = batch_schedule(batch_index).configuration_index;
            sample_indices = batch_schedule(batch_index).sample_indices;
            train_data = train_data_by_K{configuration_index};
            K = K_list(configuration_index);
            Y_batch = double(train_data.Y_data(:, :, sample_indices));
            G_batch = double(train_data.G_data(:, :, sample_indices));
            S_batch = double(train_data.S_data(:, :, sample_indices));
            B_batch = double(train_data.B_data(:, :, sample_indices));
            sigma2_batch = double(train_data.sigma2_data(sample_indices));
            G0_batch = complex(zeros(N, K, numel(sample_indices)));

            for batch_sample_index = 1:numel(sample_indices)
                sample_index = sample_indices(batch_sample_index);
                if ~training_initial_ready{configuration_index}(sample_index)
                    Y_initial = double(train_data.Y_data(:, :, sample_index));
                    S_initial = double(train_data.S_data(:, :, sample_index));
                    B_initial = double(train_data.B_data(:, :, sample_index));
                    sigma2_initial = double(train_data.sigma2_data(sample_index));
                    [training_initial_cache{configuration_index}(:, :, sample_index), ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', Y_initial, S_initial, B_initial, sigma2_initial, p_H, 0, maximum_gs_updates, gs_convergence_tolerance, gd_initial_variance, sample_index);
                    training_initial_ready{configuration_index}(sample_index) = true;
                end
                G0_batch(:, :, batch_sample_index) = training_initial_cache{configuration_index}(:, :, sample_index);
            end

            [batch_loss, gradients] = dlfeval(@local_training_loss, parameters, Y_batch, G_batch, S_batch, B_batch, sigma2_batch, G0_batch, p_H, N, K, number_of_stages);
            gradients = real(gradients);
            gradient_norm = sqrt(sum(gradients.^2, 'all'));
            gradient_norm_value = double(extractdata(gradient_norm));
            if gradient_norm_value > gradient_limit
                gradients = gradients*(gradient_limit/gradient_norm_value);
            end

            parameter_update_count = parameter_update_count+1;
            average_gradient = beta1*average_gradient+(1-beta1)*gradients;
            average_squared_gradient = beta2*average_squared_gradient+(1-beta2)*(gradients.^2);
            corrected_average_gradient = average_gradient/(1-beta1^parameter_update_count);
            corrected_average_squared_gradient = average_squared_gradient/(1-beta2^parameter_update_count);
            parameters = parameters-learning_rate*corrected_average_gradient./(sqrt(corrected_average_squared_gradient)+epsilon);
            batch_loss_sum = batch_loss_sum+double(extractdata(batch_loss));
        end

        mean_training_loss = batch_loss_sum/max(numel(batch_schedule), 1);
        numeric_parameters = double(extractdata(parameters));
        validation_loss = local_mixed_validation_loss(numeric_parameters, validation_data_by_K, selection_indices_by_K, validation_initial_cache, validation_initial_ready, K_list, SNR_grid, N, p_H, number_of_stages);
        history(epoch, :) = [epoch mean_training_loss validation_loss 10*log10(max(validation_loss, realmin)) numel(batch_schedule) toc(epoch_start_time)];
        fprintf('P=%d epoch=%d balanced train NMSE=%.6g validation macro-NMSE=%.6g (%.3f dB) batches=%d time=%.1f s.\n', P, epoch, mean_training_loss, validation_loss, history(epoch, 4), numel(batch_schedule), history(epoch, 6));

        if isfinite(validation_loss) && validation_loss < best_validation_loss
            best_validation_loss = validation_loss;
            best_parameters = parameters;
            best_epoch = epoch;
            epochs_without_improvement = 0;
        else
            epochs_without_improvement = epochs_without_improvement+1;
        end

        current_parameters = double(extractdata(parameters));
        best_parameters_saved = double(extractdata(best_parameters));
        history_saved = history(1:epoch, :);
        save(checkpoint_file, 'current_parameters', 'best_parameters_saved', 'best_epoch', 'best_validation_loss', 'history_saved', 'average_gradient', 'average_squared_gradient', 'epochs_without_improvement', 'parameter_update_count', 'epoch', 'P', 'N', 'K_list', 'RSR_dB', 'P_list', 'SNR_grid', 'number_of_stages', 'parameters_per_stage', 'training_samples_per_snr_per_K_per_epoch', 'validation_samples_per_snr_per_K_for_selection', 'batch_size', 'learning_rate', 'warm_start_file', 'maximum_gs_updates', 'gs_convergence_tolerance');

        if epochs_without_improvement >= early_stopping_patience
            fprintf('P=%d early stopping after %d epochs without validation improvement.\n', P, early_stopping_patience);
            break;
        end
    end

    if best_epoch == 0
        error('No finite validation checkpoint selected for P=%d.', P);
    end

    %% Full validation readout for each K and SNR; no test files are touched
    best_parameters_numeric = double(extractdata(best_parameters));
    validation_rows = zeros(number_of_configurations*numel(SNR_grid), 6);
    validation_row_index = 0;
    for configuration_index = 1:number_of_configurations
        K = K_list(configuration_index);
        validation_data = validation_data_by_K{configuration_index};
        for snr_index = 1:numel(SNR_grid)
            sample_indices = find(double(validation_data.SNR_dB_data) == SNR_grid(snr_index));
            sample_nmse = zeros(numel(sample_indices), 1);
            warm_start_sample_nmse = zeros(numel(sample_indices), 1);
            for sample_position = 1:numel(sample_indices)
                sample_index = sample_indices(sample_position);
                if ~validation_initial_ready{configuration_index}(sample_index)
                    Y_initial = double(validation_data.Y_data(:, :, sample_index));
                    S_initial = double(validation_data.S_data(:, :, sample_index));
                    B_initial = double(validation_data.B_data(:, :, sample_index));
                    sigma2_initial = double(validation_data.sigma2_data(sample_index));
                    [validation_initial_cache{configuration_index}(:, :, sample_index), ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', Y_initial, S_initial, B_initial, sigma2_initial, p_H, 0, maximum_gs_updates, gs_convergence_tolerance, gd_initial_variance, sample_index);
                    validation_initial_ready{configuration_index}(sample_index) = true;
                end
                Y = double(validation_data.Y_data(:, :, sample_index));
                G_true = double(validation_data.G_data(:, :, sample_index));
                S = double(validation_data.S_data(:, :, sample_index));
                B = double(validation_data.B_data(:, :, sample_index));
                sigma2 = double(validation_data.sigma2_data(sample_index));
                G_estimate = local_unfolded_forward_numeric(best_parameters_numeric, Y, S, B, sigma2, validation_initial_cache{configuration_index}(:, :, sample_index), p_H, N, K, number_of_stages);
                sample_nmse(sample_position) = sum(abs(G_estimate-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
                G_warm_start = local_unfolded_forward_numeric(warm_start_data.best_parameters_numeric, Y, S, B, sigma2, validation_initial_cache{configuration_index}(:, :, sample_index), p_H, N, K, number_of_stages);
                warm_start_sample_nmse(sample_position) = sum(abs(G_warm_start-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
            end
            validation_row_index = validation_row_index+1;
            validation_rows(validation_row_index, :) = [P K SNR_grid(snr_index) numel(sample_indices) mean(warm_start_sample_nmse) mean(sample_nmse)];
        end
    end

    validation_by_K_SNR = array2table(validation_rows(1:validation_row_index, :), 'VariableNames', {'P','K','SNR_dB','Samples','V4WarmStartNMSE_linear','SharedModelNMSE_linear'});
    validation_by_K_SNR.V4WarmStartNMSE_dB = 10*log10(max(validation_by_K_SNR.V4WarmStartNMSE_linear, realmin));
    validation_by_K_SNR.SharedModelNMSE_dB = 10*log10(max(validation_by_K_SNR.SharedModelNMSE_linear, realmin));
    validation_csv = fullfile(output_folder, sprintf('validation_by_K_SNR_P%d.csv', P));
    writetable(validation_by_K_SNR, validation_csv);
    history_table = array2table(history(1:epoch, :), 'VariableNames', {'Epoch','TrainingMacroNMSE','ValidationSubsetMacroNMSE','ValidationSubsetMacroNMSE_dB','BatchCount','EpochRuntimeSeconds'});
    history_csv = fullfile(output_folder, sprintf('training_history_P%d.csv', P));
    writetable(history_table, history_csv);

    validation_macro_nmse = mean(validation_by_K_SNR.SharedModelNMSE_linear);
    validation_macro_nmse_dB = 10*log10(max(validation_macro_nmse, realmin));
    selected_model_file = fullfile(output_folder, sprintf('selected_shared_model_P%d.mat', P));
    N_supported = N;
    K_supported = K_list;
    best_epoch_saved = best_epoch;
    save(selected_model_file, 'best_parameters_numeric', 'best_epoch_saved', 'best_validation_loss', 'validation_macro_nmse', 'validation_macro_nmse_dB', 'validation_by_K_SNR', 'P', 'N', 'N_supported', 'K_supported', 'P_list', 'SNR_grid', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs', 'number_of_shrinkage_parameters', 'initialization_method', 'maximum_gs_updates', 'gs_convergence_tolerance', 'training_samples_per_snr_per_K_per_epoch', 'validation_samples_per_snr_per_K_for_selection', 'batch_size', 'learning_rate', 'warm_start_file', 'data_version');
    fprintf('P=%d shared model selected at epoch %d; validation macro NMSE=%.4f dB.\n', P, best_epoch, validation_macro_nmse_dB);
    fprintf('Saved shared model: %s\n', selected_model_file);
    fprintf('Saved per-K/SNR validation: %s\n', validation_csv);
end

fprintf('\nShared K-sweep model training and validation complete.\n');
fprintf('No test files were loaded. Selected models are stored in %s\n', output_folder);

%% -------------------- LOCAL FUNCTIONS --------------------
function local_validate_dimension_data(data, N, K, P, SNR_grid, RSR_dB, expected_samples_per_SNR)
if data.N ~= N || data.K ~= K || data.P ~= P || data.RSR_dB ~= RSR_dB || ~isequal(double(data.SNR_dB), double(SNR_grid)) || ~strcmp(data.data_version, 'V3_USER_COUNT_RECOMMENDED_SWEEP_V1')
    error('Data metadata mismatch for N=%d,K=%d,P=%d.', N, K, P);
end
if data.number_of_samples ~= expected_samples_per_SNR*numel(SNR_grid)
    error('Unexpected sample total for N=%d,K=%d,P=%d.', N, K, P);
end
for snr_index = 1:numel(SNR_grid)
    if sum(double(data.SNR_dB_data) == SNR_grid(snr_index)) ~= expected_samples_per_SNR
        error('Unexpected per-SNR sample count for K=%d,P=%d,SNR=%g.', K, P, SNR_grid(snr_index));
    end
end
end

function [loss, gradients] = local_training_loss(parameters, Y_batch, G_batch, S_batch, B_batch, sigma2_batch, G0_batch, p_H, N, K, number_of_stages)
batch_count = size(Y_batch, 3);
total_loss = 0;
for sample_index = 1:batch_count
    Y = Y_batch(:, :, sample_index);
    G_true = G_batch(:, :, sample_index);
    S = S_batch(:, :, sample_index);
    B = B_batch(:, :, sample_index);
    sigma2 = sigma2_batch(sample_index);
    G_initial = G0_batch(:, :, sample_index);
    G_estimate = local_unfolded_forward_dl(parameters, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);
    total_loss = total_loss+sum(abs(G_estimate-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
end
loss = total_loss/batch_count;
gradients = dlgradient(loss, parameters);
end

function G_estimate = local_unfolded_forward_dl(parameters, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages)
right_inverse = S'/(S*S');
G_estimate = dlarray(G_initial);
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
bessel_ratio_operation = V3_BesselRatioFunction();
hankel_map = local_hankel_map(N, p_H);
hankel_diagonal_counts = sum(hankel_map, 1).';
for stage_index = 1:number_of_stages
    G_previous = G_estimate;
    field = G_estimate*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = bessel_ratio_operation(kappa);
    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    G_hankel = local_hankel_shrink_dl(G_emgs, parameters(1:3, stage_index), noise_feature, noise_scale, hankel_map, hankel_diagonal_counts, N, K, p_H);
    emgs_change = tanh(log(1+sqrt(sum(abs(G_emgs-G_previous).^2, 'all')/max(sum(abs(G_previous).^2, 'all'), eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_emgs).^2, 'all')/max(sum(abs(G_emgs).^2, 'all'), eps)))/3);
    gate_coefficients = reshape(parameters(4:15, stage_index), 4, 3);
    gate_logits = gate_coefficients(1, :).'*noise_feature+gate_coefficients(2, :).'*emgs_change+gate_coefficients(3, :).'*hankel_change+gate_coefficients(4, :).';
    gate_logits = gate_logits-max(gate_logits);
    gate_weights = exp(gate_logits)./sum(exp(gate_logits));
    G_estimate = gate_weights(1)*G_previous+gate_weights(2)*G_emgs+gate_weights(3)*G_hankel;
end
end

function validation_loss = local_mixed_validation_loss(parameters, validation_data_by_K, selection_indices_by_K, initial_cache, initial_ready, K_list, SNR_grid, N, p_H, number_of_stages)
configuration_losses = zeros(numel(K_list), 1);
for configuration_index = 1:numel(K_list)
    K = K_list(configuration_index);
    data = validation_data_by_K{configuration_index};
    sample_indices = selection_indices_by_K{configuration_index};
    sample_nmse = zeros(numel(sample_indices), 1);
    for sample_position = 1:numel(sample_indices)
        sample_index = sample_indices(sample_position);
        if ~initial_ready{configuration_index}(sample_index)
            error('Missing cached validation GS initialization for K=%d, sample=%d.', K, sample_index);
        end
        Y = double(data.Y_data(:, :, sample_index));
        G_true = double(data.G_data(:, :, sample_index));
        S = double(data.S_data(:, :, sample_index));
        B = double(data.B_data(:, :, sample_index));
        sigma2 = double(data.sigma2_data(sample_index));
        G_estimate = local_unfolded_forward_numeric(parameters, Y, S, B, sigma2, initial_cache{configuration_index}(:, :, sample_index), p_H, N, K, number_of_stages);
        sample_nmse(sample_position) = sum(abs(G_estimate-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
    end
    snr_losses = zeros(numel(SNR_grid), 1);
    selected_snr = double(data.SNR_dB_data(sample_indices));
    for snr_index = 1:numel(SNR_grid)
        snr_losses(snr_index) = mean(sample_nmse(selected_snr == SNR_grid(snr_index)));
    end
    configuration_losses(configuration_index) = mean(snr_losses);
end
validation_loss = mean(configuration_losses);
end

function G_estimate = local_unfolded_forward_numeric(parameters, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages)
right_inverse = S'/(S*S');
G_estimate = G_initial;
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
hankel_map = local_hankel_map(N, p_H);
hankel_diagonal_counts = sum(hankel_map, 1).';
for stage_index = 1:number_of_stages
    G_previous = G_estimate;
    field = G_estimate*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    G_hankel = local_hankel_shrink_numeric(G_emgs, parameters(1:3, stage_index), noise_feature, noise_scale, hankel_map, hankel_diagonal_counts, N, K, p_H);
    emgs_change = tanh(log(1+sqrt(sum(abs(G_emgs-G_previous).^2, 'all')/max(sum(abs(G_previous).^2, 'all'), eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_emgs).^2, 'all')/max(sum(abs(G_emgs).^2, 'all'), eps)))/3);
    gate_coefficients = reshape(parameters(4:15, stage_index), 4, 3);
    gate_logits = gate_coefficients(1, :).'*noise_feature+gate_coefficients(2, :).'*emgs_change+gate_coefficients(3, :).'*hankel_change+gate_coefficients(4, :).';
    gate_logits = gate_logits-max(gate_logits);
    gate_weights = exp(gate_logits)./sum(exp(gate_logits));
    G_estimate = gate_weights(1)*G_previous+gate_weights(2)*G_emgs+gate_weights(3)*G_hankel;
end
end

function G_output = local_hankel_shrink_dl(G_input, shrinkage_parameters, noise_feature, noise_scale, hankel_map, hankel_diagonal_counts, N, K, p_H)
G_output_parts = cell(1, K);
for user_index = 1:K
    hankel_matrix = reshape(hankel_map*G_input(:, user_index), N-p_H, p_H+1);
    gram_matrix = hankel_matrix'*hankel_matrix;
    [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
    eigenvalues = real(sum(eigenvalue_matrix, 2));
    singular_values = sqrt(max(eigenvalues, 0));
    safe_singular_values = max(singular_values, 1e-12);
    mean_singular_value = max(mean(safe_singular_values), 1e-12);
    relative_spectrum_feature = tanh(log(safe_singular_values./mean_singular_value));
    threshold_logits = shrinkage_parameters(1)+shrinkage_parameters(2)*noise_feature+shrinkage_parameters(3)*relative_spectrum_feature;
    mode_thresholds = noise_scale*(2./(1+exp(-threshold_logits)));
    shrink_factors = max(1-mode_thresholds./safe_singular_values, 0);
    scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
    hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
    G_output_parts{user_index} = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
end
G_output = cat(2, G_output_parts{:});
end

function G_output = local_hankel_shrink_numeric(G_input, shrinkage_parameters, noise_feature, noise_scale, hankel_map, hankel_diagonal_counts, N, K, p_H)
G_output = zeros(N, K);
for user_index = 1:K
    hankel_matrix = reshape(hankel_map*G_input(:, user_index), N-p_H, p_H+1);
    gram_matrix = hankel_matrix'*hankel_matrix;
    [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
    eigenvalues = real(sum(eigenvalue_matrix, 2));
    singular_values = sqrt(max(eigenvalues, 0));
    safe_singular_values = max(singular_values, 1e-12);
    mean_singular_value = max(mean(safe_singular_values), 1e-12);
    relative_spectrum_feature = tanh(log(safe_singular_values./mean_singular_value));
    threshold_logits = shrinkage_parameters(1)+shrinkage_parameters(2)*noise_feature+shrinkage_parameters(3)*relative_spectrum_feature;
    mode_thresholds = noise_scale*(2./(1+exp(-threshold_logits)));
    shrink_factors = max(1-mode_thresholds./safe_singular_values, 0);
    scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
    hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
    G_output(:, user_index) = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
end
end

function hankel_map = local_hankel_map(N, p_H)
number_of_rows = N-p_H;
number_of_columns = p_H+1;
hankel_map = zeros(number_of_rows*number_of_columns, N);
for row_index = 1:number_of_rows
    for column_index = 1:number_of_columns
        linear_index = row_index+(column_index-1)*number_of_rows;
        antenna_index = row_index+column_index-1;
        hankel_map(linear_index, antenna_index) = 1;
    end
end
end
