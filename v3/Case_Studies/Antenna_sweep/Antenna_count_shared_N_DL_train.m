%% Train one shared-N PI-DU-RACE model per pilot count
% Each P-specific model uses one set of 240 weights jointly trained over
% N={8,16,24,32}; the Hankel aperture is adjusted using p_H=floor((N-1)/2).
% Existing, previously unused antenna-sweep train data are reused. Fresh
% validation data are required. This script never loads a test file.

clear;
clc;
close all;

%% Fixed experiment settings
N_list = [8 16 24 32];
K = 3;
P_list = [10 30];
SNR_grid = [-10 0 10 20 30];
RSR_dB = 25;
number_of_stages = 16;
parameters_per_stage = 15;
initialization_method = 'GS';
maximum_gs_updates = 5000;
gs_convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;

%% Screening budget
training_samples_per_N_SNR_per_epoch = 40;
validation_samples_per_N_SNR_for_selection = 32;
batch_size = 8;
maximum_epochs = 10;
early_stopping_patience = 3;
learning_rate = 0.001;
beta1 = 0.9;
beta2 = 0.999;
gradient_limit = 5;
epsilon = 1e-8;

script_folder = fileparts(mfilename('fullpath'));
repository_folder = fileparts(fileparts(script_folder));
train_release = 'ANTENNA_COUNT_N8_16_24_32_K3_RSR25DB_SNR_m10_0_10_20_30';
fresh_release = 'ANTENNA_COUNT_SHARED_N_FRESH';
train_root = fullfile(script_folder, 'data', train_release);
validation_root = fullfile(script_folder, 'data', fresh_release);
v4_model_root = fullfile(repository_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
output_folder = fullfile(script_folder, 'results', 'ANTENNA_COUNT_SHARED_N_PI_DU_RACE_FROM_V4');

if ~isfolder(train_root) || ~isfolder(validation_root) || ~isfolder(v4_model_root)
    error('Missing antenna train data, fresh validation data, or frozen V4 warm-start models.');
end

if isfolder(output_folder)
    existing_entries = dir(output_folder);
    existing_names = string({existing_entries.name});
    existing_names = existing_names(~ismember(existing_names, ["." ".."]));
    allowed_names = ["checkpoint_P10.mat" "checkpoint_P30.mat" "selected_shared_N_model_P10.mat" "selected_shared_N_model_P30.mat" "training_history_P10.csv" "training_history_P30.csv" "validation_by_N_SNR_P10.csv" "validation_by_N_SNR_P30.csv"];
    if any(~ismember(existing_names, allowed_names))
        error('Output folder has completed or unexpected files. Preserve them and choose a new output folder: %s', output_folder);
    end
else
    mkdir(output_folder);
end

fprintf('Shared-N 16-stage PI-DU-RACE training and validation.\n');
fprintf('N=%s; K=%d; P=%s; SNR=%s dB; RSR=%g dB.\n', mat2str(N_list), K, mat2str(P_list), mat2str(SNR_grid), RSR_dB);
fprintf('One model per P, jointly trained across all N; %d weights per model.\n', parameters_per_stage*number_of_stages);
fprintf('Initialization: corresponding frozen V4 parameters; test files will not be loaded.\n');
fprintf('Training samples per N/SNR/epoch=%d; validation selection samples per N/SNR=%d; max epochs=%d.\n', training_samples_per_N_SNR_per_epoch, validation_samples_per_N_SNR_for_selection, maximum_epochs);
fprintf('Outputs: %s\n', output_folder);

%% Train a separate shared-N model for each pilot count
for P_index = 1:numel(P_list)
    P = P_list(P_index);
    selected_model_file = fullfile(output_folder, sprintf('selected_shared_N_model_P%d.mat', P));
    if isfile(selected_model_file)
        completed_model = load(selected_model_file, 'N_supported', 'K', 'P', 'SNR_grid', 'RSR_dB', 'number_of_stages', 'data_version_train', 'data_version_validation');
        if ~isequal(completed_model.N_supported, N_list) || completed_model.K ~= K || completed_model.P ~= P || ~isequal(completed_model.SNR_grid, SNR_grid) || completed_model.RSR_dB ~= RSR_dB || completed_model.number_of_stages ~= number_of_stages || ~strcmp(completed_model.data_version_train, 'V3_ANTENNA_COUNT_SWEEP') || ~strcmp(completed_model.data_version_validation, 'V3_ANTENNA_COUNT_SHARED_N_FRESH_V1')
            error('Existing selected model metadata mismatch for P=%d. Preserve it and use a new output folder.', P);
        end
        fprintf('P=%d already has a compatible validation-selected model; preserving it and continuing.\n', P);
        continue;
    end
    v4_model_file = fullfile(v4_model_root, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    if ~isfile(v4_model_file)
        error('Missing frozen V4 model: %s', v4_model_file);
    end
    v4_model = load(v4_model_file, 'best_parameters_numeric', 'best_epoch_saved', 'N', 'K', 'P', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'initialization_method');
    if v4_model.N ~= 8 || v4_model.K ~= K || v4_model.P ~= P || v4_model.RSR_dB ~= RSR_dB || v4_model.number_of_stages ~= number_of_stages || v4_model.parameters_per_stage ~= parameters_per_stage || ~strcmp(v4_model.initialization_method, initialization_method) || ~isequal(size(v4_model.best_parameters_numeric), [parameters_per_stage number_of_stages])
        error('Frozen V4 model metadata mismatch for P=%d.', P);
    end

    train_data_by_N = cell(size(N_list));
    validation_data_by_N = cell(size(N_list));
    validation_selection_indices_by_N = cell(size(N_list));
    train_initial_cache_by_N = cell(size(N_list));
    train_initial_ready_by_N = cell(size(N_list));
    validation_initial_cache_by_N = cell(size(N_list));
    validation_initial_ready_by_N = cell(size(N_list));

    %% Load train and fresh validation splits, never test files
    for n_index = 1:numel(N_list)
        N = N_list(n_index);
        train_dimension_folder = fullfile(train_root, sprintf('N%d_K%d_RSR25dB', N, K));
        validation_dimension_folder = fullfile(validation_root, sprintf('N%d_K%d_RSR25dB', N, K));
        train_file = fullfile(train_dimension_folder, sprintf('train_P%d.mat', P));
        validation_file = fullfile(validation_dimension_folder, sprintf('validation_P%d.mat', P));
        if ~isfile(train_file) || ~isfile(validation_file)
            error('Missing train or fresh validation file for N=%d,P=%d.', N, P);
        end
        train_data_by_N{n_index} = load(train_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version', 'split_name', 'samples_this_split_per_snr');
        validation_data_by_N{n_index} = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version', 'split_name', 'samples_per_SNR');
        local_validate_train_split(train_data_by_N{n_index}, N, K, P, SNR_grid, RSR_dB);
        local_validate_fresh_validation(validation_data_by_N{n_index}, N, K, P, SNR_grid, RSR_dB);

        train_count = train_data_by_N{n_index}.number_of_samples;
        validation_count = validation_data_by_N{n_index}.number_of_samples;
        train_initial_cache_by_N{n_index} = complex(zeros(N, K, train_count));
        train_initial_ready_by_N{n_index} = false(1, train_count);
        validation_initial_cache_by_N{n_index} = complex(zeros(N, K, validation_count));
        validation_initial_ready_by_N{n_index} = false(1, validation_count);

        rng(2026111800+100*N+P, 'twister');
        selection_indices = [];
        for snr_index = 1:numel(SNR_grid)
            candidates = find(double(validation_data_by_N{n_index}.SNR_dB_data(:)) == SNR_grid(snr_index));
            if numel(candidates) < validation_samples_per_N_SNR_for_selection
                error('Insufficient validation samples for N=%d,P=%d,SNR=%g.', N, P, SNR_grid(snr_index));
            end
            candidates = candidates(randperm(numel(candidates)));
            selection_indices = [selection_indices; candidates(1:validation_samples_per_N_SNR_for_selection)];
        end
        validation_selection_indices_by_N{n_index} = selection_indices;

        for selection_position = 1:numel(selection_indices)
            sample_index = selection_indices(selection_position);
            data = validation_data_by_N{n_index};
            Y = double(data.Y_data(:, :, sample_index));
            S = double(data.S_data(:, :, sample_index));
            B = double(data.B_data(:, :, sample_index));
            [validation_initial_cache_by_N{n_index}(:, :, sample_index), ~, ~] = local_biased_gs(Y, S, B, maximum_gs_updates, gs_convergence_tolerance);
            validation_initial_ready_by_N{n_index}(sample_index) = true;
        end
        fprintf('P=%d loaded N=%d: train=%d, validation-selection=%d.\n', P, N, train_count, numel(selection_indices));
    end

    parameters = dlarray(double(v4_model.best_parameters_numeric));
    average_gradient = zeros(parameters_per_stage, number_of_stages);
    average_squared_gradient = zeros(parameters_per_stage, number_of_stages);
    best_parameters = parameters;
    best_validation_loss = inf;
    best_epoch = 0;
    epochs_without_improvement = 0;
    parameter_update_count = 0;
    history = zeros(maximum_epochs, 6);
    checkpoint_file = fullfile(output_folder, sprintf('checkpoint_P%d.mat', P));
    start_epoch = 1;

    %% Resume only from a compatible epoch checkpoint
    if isfile(checkpoint_file)
        checkpoint = load(checkpoint_file);
        if checkpoint.P ~= P || ~isequal(checkpoint.N_list, N_list) || checkpoint.K ~= K || checkpoint.RSR_dB ~= RSR_dB || ~isequal(checkpoint.SNR_grid, SNR_grid) || checkpoint.number_of_stages ~= number_of_stages || checkpoint.training_samples_per_N_SNR_per_epoch ~= training_samples_per_N_SNR_per_epoch || checkpoint.learning_rate ~= learning_rate
            error('Checkpoint settings do not match. Preserve it and use a new output folder.');
        end
        parameters = dlarray(checkpoint.current_parameters);
        average_gradient = checkpoint.average_gradient;
        average_squared_gradient = checkpoint.average_squared_gradient;
        best_parameters = dlarray(checkpoint.best_parameters_saved);
        best_validation_loss = checkpoint.best_validation_loss;
        best_epoch = checkpoint.best_epoch;
        epochs_without_improvement = checkpoint.epochs_without_improvement;
        parameter_update_count = checkpoint.parameter_update_count;
        history(1:size(checkpoint.history_saved, 1), :) = checkpoint.history_saved;
        start_epoch = checkpoint.epoch+1;
        fprintf('P=%d resuming after epoch %d.\n', P, checkpoint.epoch);
    end

    %% Balanced mixed-N training
    final_epoch = start_epoch-1;
    for epoch = start_epoch:maximum_epochs
        epoch_timer = tic;
        batch_schedule = struct('n_index', {}, 'sample_indices', {});
        rng(2026111900+P*100+epoch, 'twister');
        for n_index = 1:numel(N_list)
            train_data = train_data_by_N{n_index};
            for snr_index = 1:numel(SNR_grid)
                candidates = find(double(train_data.SNR_dB_data(:)) == SNR_grid(snr_index));
                if numel(candidates) < training_samples_per_N_SNR_per_epoch
                    error('Insufficient training samples for N=%d,P=%d,SNR=%g.', N_list(n_index), P, SNR_grid(snr_index));
                end
                candidates = candidates(randperm(numel(candidates)));
                chosen = candidates(1:training_samples_per_N_SNR_per_epoch);
                for batch_start = 1:batch_size:training_samples_per_N_SNR_per_epoch
                    batch_end = min(batch_start+batch_size-1, training_samples_per_N_SNR_per_epoch);
                    batch_schedule(end+1) = struct('n_index', n_index, 'sample_indices', chosen(batch_start:batch_end));
                end
            end
        end
        batch_schedule = batch_schedule(randperm(numel(batch_schedule)));
        batch_loss_sum = 0;

        for batch_index = 1:numel(batch_schedule)
            n_index = batch_schedule(batch_index).n_index;
            N = N_list(n_index);
            p_H = floor((N-1)/2);
            data = train_data_by_N{n_index};
            sample_indices = batch_schedule(batch_index).sample_indices;
            batch_count = numel(sample_indices);
            Y_batch = double(data.Y_data(:, :, sample_indices));
            G_batch = double(data.G_data(:, :, sample_indices));
            S_batch = double(data.S_data(:, :, sample_indices));
            B_batch = double(data.B_data(:, :, sample_indices));
            sigma2_batch = double(data.sigma2_data(sample_indices));
            G0_batch = complex(zeros(N, K, batch_count));

            for batch_sample_index = 1:batch_count
                sample_index = sample_indices(batch_sample_index);
                if ~train_initial_ready_by_N{n_index}(sample_index)
                    [train_initial_cache_by_N{n_index}(:, :, sample_index), ~, ~] = local_biased_gs(Y_batch(:, :, batch_sample_index), S_batch(:, :, batch_sample_index), B_batch(:, :, batch_sample_index), maximum_gs_updates, gs_convergence_tolerance);
                    train_initial_ready_by_N{n_index}(sample_index) = true;
                end
                G0_batch(:, :, batch_sample_index) = train_initial_cache_by_N{n_index}(:, :, sample_index);
            end

            [batch_loss, gradients] = dlfeval(@local_training_loss, parameters, Y_batch, G_batch, S_batch, B_batch, sigma2_batch, G0_batch, N, K, p_H, number_of_stages);
            gradients = real(gradients);
            gradient_norm = sqrt(sum(gradients.^2, 'all'));
            gradient_norm_value = double(extractdata(gradient_norm));
            if gradient_norm_value > gradient_limit
                gradients = gradients*(gradient_limit/gradient_norm_value);
            end
            parameter_update_count = parameter_update_count+1;
            average_gradient = beta1*average_gradient+(1-beta1)*gradients;
            average_squared_gradient = beta2*average_squared_gradient+(1-beta2)*(gradients.^2);
            corrected_gradient = average_gradient/(1-beta1^parameter_update_count);
            corrected_squared_gradient = average_squared_gradient/(1-beta2^parameter_update_count);
            parameters = parameters-learning_rate*corrected_gradient./(sqrt(corrected_squared_gradient)+epsilon);
            batch_loss_sum = batch_loss_sum+double(extractdata(batch_loss));
        end

        mean_training_loss = batch_loss_sum/max(numel(batch_schedule), 1);
        current_parameters = double(extractdata(parameters));
        validation_loss = local_mixed_validation_loss(current_parameters, validation_data_by_N, validation_selection_indices_by_N, validation_initial_cache_by_N, validation_initial_ready_by_N, N_list, K, SNR_grid, number_of_stages);
        history(epoch, :) = [epoch mean_training_loss validation_loss 10/log(10)*validation_loss numel(batch_schedule) toc(epoch_timer)];
        fprintf('P=%d epoch=%d train log-NMSE=%.6g validation macro log-NMSE=%.6g (%.3f dB) batches=%d time=%.1f s.\n', P, epoch, mean_training_loss, validation_loss, history(epoch, 4), numel(batch_schedule), history(epoch, 6));

        if validation_loss < best_validation_loss
            best_validation_loss = validation_loss;
            best_parameters = parameters;
            best_epoch = epoch;
            epochs_without_improvement = 0;
        else
            epochs_without_improvement = epochs_without_improvement+1;
        end

        final_epoch = epoch;
        best_parameters_saved = double(extractdata(best_parameters));
        history_saved = history(1:epoch, :);
        save(checkpoint_file, 'current_parameters', 'best_parameters_saved', 'best_epoch', 'best_validation_loss', 'history_saved', 'average_gradient', 'average_squared_gradient', 'epochs_without_improvement', 'parameter_update_count', 'epoch', 'P', 'N_list', 'K', 'RSR_dB', 'SNR_grid', 'number_of_stages', 'parameters_per_stage', 'training_samples_per_N_SNR_per_epoch', 'learning_rate');
        if epochs_without_improvement >= early_stopping_patience
            fprintf('P=%d early stopping after %d validation checks without improvement.\n', P, early_stopping_patience);
            break;
        end
    end

    if best_epoch == 0
        error('No finite validation-selected checkpoint for P=%d.', P);
    end

    %% Validation readout by N and SNR on the selected validation subset
    best_parameters_numeric = double(extractdata(best_parameters));
    validation_rows = zeros(numel(N_list)*numel(SNR_grid), 7);
    validation_row = 0;
    for n_index = 1:numel(N_list)
        N = N_list(n_index);
        p_H = floor((N-1)/2);
        data = validation_data_by_N{n_index};
        indices = validation_selection_indices_by_N{n_index};
        nmse_by_sample = zeros(numel(indices), 1);
        v4_nmse_by_sample = zeros(numel(indices), 1);
        for position = 1:numel(indices)
            sample_index = indices(position);
            Y = double(data.Y_data(:, :, sample_index));
            G_true = double(data.G_data(:, :, sample_index));
            S = double(data.S_data(:, :, sample_index));
            B = double(data.B_data(:, :, sample_index));
            sigma2 = double(data.sigma2_data(sample_index));
            G0 = validation_initial_cache_by_N{n_index}(:, :, sample_index);
            G_candidate = local_unfolded_forward_numeric(best_parameters_numeric, Y, S, B, sigma2, G0, N, K, p_H, number_of_stages);
            G_warm = local_unfolded_forward_numeric(v4_model.best_parameters_numeric, Y, S, B, sigma2, G0, N, K, p_H, number_of_stages);
            true_energy = max(sum(abs(G_true).^2, 'all'), eps);
            nmse_by_sample(position) = sum(abs(G_candidate-G_true).^2, 'all')/true_energy;
            v4_nmse_by_sample(position) = sum(abs(G_warm-G_true).^2, 'all')/true_energy;
        end
        selected_snr = double(data.SNR_dB_data(indices));
        for snr_index = 1:numel(SNR_grid)
            condition = selected_snr == SNR_grid(snr_index);
            validation_row = validation_row+1;
            validation_rows(validation_row, :) = [N K P SNR_grid(snr_index) sum(condition) mean(v4_nmse_by_sample(condition)) mean(nmse_by_sample(condition))];
        end
    end
    validation_by_N_SNR = array2table(validation_rows, 'VariableNames', {'N','K','P','SNR_dB','Samples','V4WarmStartNMSE_linear','SharedNModelNMSE_linear'});
    validation_by_N_SNR.V4WarmStartNMSE_dB = 10*log10(max(validation_by_N_SNR.V4WarmStartNMSE_linear, realmin));
    validation_by_N_SNR.SharedNModelNMSE_dB = 10*log10(max(validation_by_N_SNR.SharedNModelNMSE_linear, realmin));
    validation_csv = fullfile(output_folder, sprintf('validation_by_N_SNR_P%d.csv', P));
    writetable(validation_by_N_SNR, validation_csv);
    history_table = array2table(history(1:final_epoch, :), 'VariableNames', {'Epoch','TrainingMeanLogNMSE','ValidationMacroLogNMSE','ValidationMacroLogNMSE_dB','BatchCount','EpochRuntimeSeconds'});
    history_csv = fullfile(output_folder, sprintf('training_history_P%d.csv', P));
    writetable(history_table, history_csv);

    validation_macro_nmse = mean(validation_by_N_SNR.SharedNModelNMSE_linear);
    validation_macro_nmse_dB = 10*log10(max(validation_macro_nmse, realmin));
    best_epoch_saved = best_epoch;
    N_supported = N_list;
    p_H_by_N = floor((N_list-1)/2);
    warm_start_file = v4_model_file;
    data_version_train = 'V3_ANTENNA_COUNT_SWEEP';
    data_version_validation = 'V3_ANTENNA_COUNT_SHARED_N_FRESH_V1';
    save(selected_model_file, 'best_parameters_numeric', 'best_epoch_saved', 'best_validation_loss', 'validation_macro_nmse', 'validation_macro_nmse_dB', 'validation_by_N_SNR', 'N', 'N_supported', 'K', 'P', 'P_list', 'SNR_grid', 'RSR_dB', 'p_H_by_N', 'number_of_stages', 'parameters_per_stage', 'initialization_method', 'maximum_gs_updates', 'gs_convergence_tolerance', 'training_samples_per_N_SNR_per_epoch', 'validation_samples_per_N_SNR_for_selection', 'batch_size', 'learning_rate', 'warm_start_file', 'data_version_train', 'data_version_validation');
    delete(checkpoint_file);
    fprintf('P=%d shared-N model selected at epoch %d. Validation macro NMSE=%+.4f dB.\n', P, best_epoch, validation_macro_nmse_dB);
    fprintf('Saved model: %s\n', selected_model_file);
end

fprintf('\nShared-N training/validation complete. No test files were loaded.\n');

%% Local functions
function local_validate_train_split(data, N, K, P, SNR_grid, RSR_dB)
if data.N ~= N || data.K ~= K || data.P ~= P || data.RSR_dB ~= RSR_dB || ~isequal(double(data.SNR_dB), SNR_grid) || ~strcmp(data.data_version, 'V3_ANTENNA_COUNT_SWEEP') || ~strcmp(data.split_name, 'train') || data.samples_this_split_per_snr ~= 2000 || data.number_of_samples ~= 2000*numel(SNR_grid)
    error('Training-data metadata mismatch at N=%d,P=%d.', N, P);
end
end

function local_validate_fresh_validation(data, N, K, P, SNR_grid, RSR_dB)
if data.N ~= N || data.K ~= K || data.P ~= P || data.RSR_dB ~= RSR_dB || ~isequal(double(data.SNR_dB), SNR_grid) || ~strcmp(data.data_version, 'V3_ANTENNA_COUNT_SHARED_N_FRESH_V1') || ~strcmp(data.split_name, 'validation') || data.number_of_samples ~= 500*numel(SNR_grid)
    error('Fresh validation-data metadata mismatch at N=%d,P=%d.', N, P);
end
end

function [loss, gradients] = local_training_loss(parameters, Y_batch, G_batch, S_batch, B_batch, sigma2_batch, G0_batch, N, K, p_H, number_of_stages)
batch_count = size(Y_batch, 3);
total_loss = 0;
for sample_index = 1:batch_count
    G_estimate = local_unfolded_forward_dl(parameters, Y_batch(:, :, sample_index), S_batch(:, :, sample_index), B_batch(:, :, sample_index), sigma2_batch(sample_index), G0_batch(:, :, sample_index), N, K, p_H, number_of_stages);
    G_true = G_batch(:, :, sample_index);
    sample_nmse = sum(abs(G_estimate-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
    total_loss = total_loss+log(max(sample_nmse, eps));
end
loss = total_loss/batch_count;
gradients = dlgradient(loss, parameters);
end

function G_estimate = local_unfolded_forward_dl(parameters, Y, S, B, sigma2, G_initial, N, K, p_H, number_of_stages)
right_inverse = S'/(S*S');
G_estimate = dlarray(G_initial);
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
hankel_map = local_hankel_map(N, p_H);
hankel_diagonal_counts = sum(hankel_map, 1).';
for stage_index = 1:number_of_stages
    G_previous = G_estimate;
    field = G_estimate*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = local_bessel_ratio_dl(kappa);
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

function ratio = local_bessel_ratio_dl(kappa)
% Preserve the exact scaled-Bessel forward value and provide its analytic
% derivative because besseli does not accept dlarray inputs in MATLAB.
kappa_value = extractdata(kappa);
i0_value = besseli(0, kappa_value, 1);
i1_value = besseli(1, kappa_value, 1);
ratio_value = i1_value./max(i0_value, realmin);
ratio_derivative = 1-ratio_value.^2;
positive_kappa = kappa_value > 1e-6;
ratio_derivative(positive_kappa) = 1-ratio_value(positive_kappa).^2-ratio_value(positive_kappa)./kappa_value(positive_kappa);
ratio_derivative(~positive_kappa) = 0.5;
ratio = ratio_value+(kappa-kappa_value).*ratio_derivative;
end

function G_output = local_hankel_shrink_dl(G_input, shrinkage_parameters, noise_feature, noise_scale, hankel_map, hankel_diagonal_counts, N, K, p_H)
G_output_parts = cell(1, K);
for user_index = 1:K
    hankel_matrix = reshape(hankel_map*G_input(:, user_index), N-p_H, p_H+1);
    gram_matrix = hankel_matrix'*hankel_matrix;
    [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
    singular_values = sqrt(max(real(sum(eigenvalue_matrix, 2)), 0));
    safe_singular_values = max(singular_values, 1e-12);
    mean_singular_value = max(mean(safe_singular_values), 1e-12);
    relative_spectrum_feature = tanh(log(safe_singular_values./mean_singular_value));
    threshold_logits = shrinkage_parameters(1)+shrinkage_parameters(2)*noise_feature+shrinkage_parameters(3)*relative_spectrum_feature;
    mode_thresholds = noise_scale*(2./(1+exp(-threshold_logits)));
    shrink_factors = max(1-mode_thresholds./safe_singular_values, 0);
    hankel_shrunk = hankel_matrix*(eigenvectors.*reshape(shrink_factors, 1, []))*eigenvectors';
    G_output_parts{user_index} = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
end
G_output = cat(2, G_output_parts{:});
end

function validation_loss = local_mixed_validation_loss(parameters, data_by_N, indices_by_N, initial_cache_by_N, initial_ready_by_N, N_list, K, SNR_grid, number_of_stages)
cell_losses = zeros(numel(N_list)*numel(SNR_grid), 1);
cell_index = 0;
for n_index = 1:numel(N_list)
    N = N_list(n_index);
    p_H = floor((N-1)/2);
    data = data_by_N{n_index};
    indices = indices_by_N{n_index};
    sample_nmse = zeros(numel(indices), 1);
    for position = 1:numel(indices)
        sample_index = indices(position);
        if ~initial_ready_by_N{n_index}(sample_index)
            error('Missing GS initialization for validation sample %d at N=%d.', sample_index, N);
        end
        Y = double(data.Y_data(:, :, sample_index));
        G_true = double(data.G_data(:, :, sample_index));
        S = double(data.S_data(:, :, sample_index));
        B = double(data.B_data(:, :, sample_index));
        sigma2 = double(data.sigma2_data(sample_index));
        G_estimate = local_unfolded_forward_numeric(parameters, Y, S, B, sigma2, initial_cache_by_N{n_index}(:, :, sample_index), N, K, p_H, number_of_stages);
        sample_nmse(position) = sum(abs(G_estimate-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
    end
    selected_snr = double(data.SNR_dB_data(indices));
    for snr_index = 1:numel(SNR_grid)
        cell_index = cell_index+1;
        cell_losses(cell_index) = mean(log(max(sample_nmse(selected_snr == SNR_grid(snr_index)), realmin)));
    end
end
validation_loss = mean(cell_losses);
end

function G_estimate = local_unfolded_forward_numeric(parameters, Y, S, B, sigma2, G_initial, N, K, p_H, number_of_stages)
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

function G_output = local_hankel_shrink_numeric(G_input, shrinkage_parameters, noise_feature, noise_scale, hankel_map, hankel_diagonal_counts, N, K, p_H)
G_output = zeros(N, K);
for user_index = 1:K
    hankel_matrix = reshape(hankel_map*G_input(:, user_index), N-p_H, p_H+1);
    gram_matrix = hankel_matrix'*hankel_matrix;
    [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
    singular_values = sqrt(max(real(sum(eigenvalue_matrix, 2)), 0));
    safe_singular_values = max(singular_values, 1e-12);
    mean_singular_value = max(mean(safe_singular_values), 1e-12);
    relative_spectrum_feature = tanh(log(safe_singular_values./mean_singular_value));
    threshold_logits = shrinkage_parameters(1)+shrinkage_parameters(2)*noise_feature+shrinkage_parameters(3)*relative_spectrum_feature;
    mode_thresholds = noise_scale*(2./(1+exp(-threshold_logits)));
    shrink_factors = max(1-mode_thresholds./safe_singular_values, 0);
    hankel_shrunk = hankel_matrix*(eigenvectors.*reshape(shrink_factors, 1, []))*eigenvectors';
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

function [G, iteration_count, converged] = local_biased_gs(Y, S, B, maximum_updates, tolerance)
G = local_spectral_initialize(Y, S, B);
right_inverse = S'/(S*S');
converged = false;
iteration_count = maximum_updates;
for update_index = 1:maximum_updates
    previous_G = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous_G, 'fro')^2/max(norm(G, 'fro')^2, eps) < tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

function G = local_spectral_initialize(Y, S, B)
N = size(Y, 1);
K = size(S, 1);
G = zeros(N, K);
for antenna_index = 1:N
    y = Y(antenna_index, :).';
    b = B(antenna_index, :).';
    augmented_matrix = [S.', b];
    spectral_matrix = augmented_matrix'*diag(y)*augmented_matrix;
    [spectral_vectors, spectral_values] = eig(spectral_matrix);
    [~, principal_index] = max(real(diag(spectral_values)));
    principal_vector = spectral_vectors(:, principal_index);
    principal_vector = principal_vector/max(norm(principal_vector), eps);
    spectral_scale = (abs(augmented_matrix*principal_vector).'*y)/max(norm(augmented_matrix*principal_vector)^2, eps);
    G(antenna_index, :) = (spectral_scale*exp(-1i*angle(principal_vector(K+1)))*principal_vector(1:K)).';
end
end
