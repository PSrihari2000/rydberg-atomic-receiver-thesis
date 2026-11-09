%% V5 sixteen-stage unfolding fine-tuned from the frozen V4 models
% Separate P=10 and P=30 models warm-start directly from the frozen V4 16-stage models.
% Each stage learns a noise- and singular-spectrum-conditioned Hankel shrinkage rule and a three-way gate.
% The gate chooses among the previous estimate, the EM-GS update, and the Hankel estimate.
% The gate uses only measurement/noise and update-difference features, never the true channel.
% Training and validation files only are loaded. Reserved test files are never loaded.
% P=10 and P=30 run independently in parfor; each model's minibatches remain sequential.

clear;
clc;
close all;
rng(20261109, 'twister');

%% -------------------- FIXED V5 SCENARIO --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
number_of_stages = 16;
number_of_gate_features = 3;
number_of_gate_outputs = 3;
number_of_shrinkage_parameters = 3;
parameters_per_stage = number_of_shrinkage_parameters+number_of_gate_outputs*(number_of_gate_features+1);
initialization_method = 'GS';
training_samples_per_epoch = 4000;
validation_samples_per_SNR_for_selection = 128;
batch_size = 8;
maximum_epochs = 30;
early_stopping_patience = 6;
learning_rate = 0.003;
beta1 = 0.9;
beta2 = 0.999;
gradient_limit = 5;
epsilon = 1e-8;

script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
v3_folder = fileparts(case_studies_folder);
scenario_folder = fullfile(script_folder, 'data', 'N8_K3_RSR25dB');
base_checkpoint_folder = fullfile(v3_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
maximum_gs_updates = 5000;
gs_convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
output_folder = fullfile(script_folder, 'results', 'V5_PI_DU_RACE_16STAGE_FRESH_DATA');
if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it or choose a new output-folder name before running.');
end
if ~isfile(fullfile(script_folder, 'V3_BesselRatioFunction.m'))
    error('V3_BesselRatioFunction.m must be in the same folder as this trainer.');
end
if ~isfolder(scenario_folder)
    error('V5 scenario data folder was not found: %s', scenario_folder);
end
if ~isfolder(base_checkpoint_folder)
    error('Frozen V4 model folder was not found: %s', base_checkpoint_folder);
end
if ~isfile(fullfile(script_folder, 'V3_CLASSICAL_ESTIMATOR.m'))
    error('V3_CLASSICAL_ESTIMATOR.m must be in the scripts folder.');
end
addpath(script_folder);
mkdir(output_folder);

fprintf('V5 sixteen-stage PI-DU-RACE with spectrum-adaptive soft Hankel shrinkage.\n');
fprintf('N=%d, K=%d, P=[%s], RSR=%g dB, AoA=U(-90,90 deg).\n', N, K, num2str(P_list), RSR_dB);
fprintf('Stages=%d; trainable parameters=%d per stage, %d per P model.\n', number_of_stages, parameters_per_stage, parameters_per_stage*number_of_stages);
fprintf('Each Hankel mode uses one shared learned logistic rule conditioned on noise and relative singular value.\n');
fprintf('Warm start uses all 15 learned parameters per stage from the matching frozen V4 model.\n');
fprintf('The gate mixes previous estimate, EM-GS update, and Hankel estimate.\n');
fprintf('Training samples per epoch=%d; batch size=%d; maximum epochs=%d; early-stop patience=%d.\n', training_samples_per_epoch, batch_size, maximum_epochs, early_stopping_patience);
fprintf('GS initialization uses up to %d updates and relative-change tolerance %g.\n', maximum_gs_updates, gs_convergence_tolerance);
fprintf('Training/validation only; reserved test files will not be loaded.\n');
fprintf('Outputs: %s\n', output_folder);

%% -------------------- TRAIN SEPARATE MODELS FOR THE PILOT COUNTS --------------------
parfor (p_index = 1:numel(P_list), min(2, numel(P_list)))
    P = P_list(p_index);
    rng(20261109+P, 'twister');
    train_file = fullfile(scenario_folder, sprintf('train_P%d.mat', P));
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    warm_start_file = fullfile(base_checkpoint_folder, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    if ~isfile(train_file) || ~isfile(validation_file) || ~isfile(warm_start_file)
        error('Missing training data, validation data, or selected 16-stage model for P=%d.', P);
    end

    train_data = load(train_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P');
    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P');
    warm_start_data = load(warm_start_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs', 'initialization_method');
    if train_data.number_of_samples ~= 40000
        error('Expected exactly 40000 training samples for P=%d.', P);
    end
    if validation_data.number_of_samples ~= 4000
        error('Expected exactly 4000 fresh validation samples for P=%d.', P);
    end
    if train_data.N ~= N || train_data.K ~= K || train_data.P ~= P || validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P
        error('Dataset metadata does not match the configured N, K, and P values.');
    end
    if warm_start_data.N ~= N || warm_start_data.K ~= K || warm_start_data.P ~= P || warm_start_data.RSR_dB ~= RSR_dB || warm_start_data.p_H ~= p_H || warm_start_data.number_of_stages ~= number_of_stages || warm_start_data.parameters_per_stage ~= parameters_per_stage || warm_start_data.number_of_gate_features ~= number_of_gate_features || warm_start_data.number_of_gate_outputs ~= number_of_gate_outputs || ~strcmp(warm_start_data.initialization_method, 'GS') || ~isequal(size(warm_start_data.best_parameters_numeric), [parameters_per_stage number_of_stages])
        error('Frozen V4 16-stage model metadata does not match the V5 experiment settings.');
    end

    fprintf('\n============================================================\n');
    fprintf('Fine-tuning spectrum-adaptive 16-stage model for P=%d\n', P);
    fprintf('Training samples=%d; validation samples=%d; warm-start epoch=%d\n', train_data.number_of_samples, validation_data.number_of_samples, warm_start_data.best_epoch_saved);

    selection_indices = local_select_fresh_validation_indices(validation_data.SNR_dB_data, SNR_grid, validation_samples_per_SNR_for_selection);
    validation_initial_cache = complex(zeros(N, K, validation_data.number_of_samples));
    validation_initial_ready = false(1, validation_data.number_of_samples);
    for selection_index = 1:numel(selection_indices)
        data_index = selection_indices(selection_index);
        Y_initial = double(validation_data.Y_data(:, :, data_index));
        S_initial = double(validation_data.S_data(:, :, data_index));
        B_initial = double(validation_data.B_data(:, :, data_index));
        sigma2_initial = double(validation_data.sigma2_data(data_index));
        [validation_initial_cache(:, :, data_index), ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', Y_initial, S_initial, B_initial, sigma2_initial, p_H, 0, maximum_gs_updates, gs_convergence_tolerance, gd_initial_variance, data_index);
        validation_initial_ready(data_index) = true;
    end
    training_initial_cache = complex(zeros(N, K, train_data.number_of_samples));
    training_initial_ready = false(1, train_data.number_of_samples);
    parameters_initial = warm_start_data.best_parameters_numeric;
    parameters = dlarray(parameters_initial);
    average_gradient = zeros(size(parameters_initial));
    average_squared_gradient = zeros(size(parameters_initial));
    best_parameters = parameters;
    best_validation_loss = inf;
    best_epoch = 0;
    epochs_without_improvement = 0;
    parameter_update_count = 0;
    history = zeros(maximum_epochs, 5);

    for epoch = 1:maximum_epochs
        epoch_start_time = tic;
        sample_order = randperm(train_data.number_of_samples, min(training_samples_per_epoch, train_data.number_of_samples));
        batch_loss_sum = 0;
        batch_count = 0;

        for batch_start = 1:batch_size:numel(sample_order)
            batch_end = min(batch_start+batch_size-1, numel(sample_order));
            sample_indices = sample_order(batch_start:batch_end);
            Y_batch = double(train_data.Y_data(:, :, sample_indices));
            G_batch = double(train_data.G_data(:, :, sample_indices));
            S_batch = double(train_data.S_data(:, :, sample_indices));
            B_batch = double(train_data.B_data(:, :, sample_indices));
            sigma2_batch = double(train_data.sigma2_data(sample_indices));
            G0_batch = complex(zeros(N, K, numel(sample_indices)));
            for batch_sample_index = 1:numel(sample_indices)
                data_index = sample_indices(batch_sample_index);
                if ~training_initial_ready(data_index)
                    Y_initial = double(train_data.Y_data(:, :, data_index));
                    S_initial = double(train_data.S_data(:, :, data_index));
                    B_initial = double(train_data.B_data(:, :, data_index));
                    sigma2_initial = double(train_data.sigma2_data(data_index));
                    [training_initial_cache(:, :, data_index), ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', Y_initial, S_initial, B_initial, sigma2_initial, p_H, 0, maximum_gs_updates, gs_convergence_tolerance, gd_initial_variance, data_index);
                    training_initial_ready(data_index) = true;
                end
                G0_batch(:, :, batch_sample_index) = training_initial_cache(:, :, data_index);
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
            batch_count = batch_count+1;
        end

        mean_training_loss = batch_loss_sum/max(batch_count, 1);
        numeric_parameters = double(extractdata(parameters));
        validation_loss = local_evaluate_loss(numeric_parameters, validation_data, selection_indices, validation_initial_cache, validation_initial_ready, p_H, N, K, number_of_stages);
        history(epoch, :) = [epoch mean_training_loss validation_loss 10*log10(validation_loss) toc(epoch_start_time)];
        fprintf('P=%d epoch=%d train NMSE=%.6g GS-cache samples=%d validation subset NMSE=%.6g (%.3f dB) time=%.1f s\n', P, epoch, mean_training_loss, sum(training_initial_ready), validation_loss, 10*log10(validation_loss), history(epoch, 5));

        if isfinite(validation_loss) && validation_loss < best_validation_loss
            best_validation_loss = validation_loss;
            best_parameters = parameters;
            best_epoch = epoch;
            epochs_without_improvement = 0;
        else
            epochs_without_improvement = epochs_without_improvement+1;
        end

        checkpoint_file = fullfile(output_folder, sprintf('checkpoint_P%d.mat', P));
        current_parameters = double(extractdata(parameters));
        best_parameters_saved = double(extractdata(best_parameters));
        history_saved = history(1:epoch, :);
        local_save_checkpoint(checkpoint_file, current_parameters, best_parameters_saved, best_epoch, best_validation_loss, history_saved, parameter_update_count, epoch, P, N, K, RSR_dB, p_H, number_of_stages, training_samples_per_epoch, batch_size, learning_rate, warm_start_file, initialization_method, maximum_gs_updates, gs_convergence_tolerance, parameters_per_stage);

        if epochs_without_improvement >= early_stopping_patience
            fprintf('P=%d early stopping after %d epochs without validation improvement.\n', P, early_stopping_patience);
            break;
        end
    end

    if best_epoch == 0
        error('No finite validation checkpoint was selected for P=%d. Inspect the checkpoint before retrying with a new output folder.', P);
    end
    best_parameters_numeric = double(extractdata(best_parameters));
    history_table = array2table(history(1:epoch, :), 'VariableNames', {'Epoch', 'TrainingNMSE', 'ValidationSubsetNMSE', 'ValidationSubsetNMSE_dB', 'EpochRuntimeSeconds'});
    history_file = fullfile(output_folder, sprintf('training_history_P%d.csv', P));
    writetable(history_table, history_file);
    selected_model_file = fullfile(output_folder, sprintf('selected_model_P%d.mat', P));
    best_epoch_saved = best_epoch;
    [full_validation_nmse, validation_by_snr] = local_evaluate_full_validation(best_parameters_numeric, validation_data, selection_indices, validation_initial_cache, SNR_grid, p_H, N, K, number_of_stages);
    [~, v4_validation_by_snr] = local_evaluate_full_validation(warm_start_data.best_parameters_numeric, validation_data, selection_indices, validation_initial_cache, SNR_grid, p_H, N, K, number_of_stages);
    validation_gain_dB = v4_validation_by_snr-validation_by_snr;
    validation_table = table(repmat(P, numel(SNR_grid), 1), SNR_grid(:), v4_validation_by_snr(:), validation_by_snr(:), validation_gain_dB(:), 'VariableNames', {'P', 'SNR_dB', 'V4WarmStartValidationNMSE_dB', 'V5SelectedValidationNMSE_dB', 'V5GainOverV4_dB'});
    comparison_csv = fullfile(output_folder, sprintf('validation_V4_vs_V5_P%d.csv', P));
    writetable(validation_table, comparison_csv);
    validation_csv = fullfile(output_folder, sprintf('validation_by_snr_P%d.csv', P));
    writetable(validation_table, validation_csv);
    validation_nmse_dB = 10*log10(full_validation_nmse);
    local_save_selected_model(selected_model_file, best_parameters_numeric, best_epoch_saved, best_validation_loss, validation_table, validation_nmse_dB, P, N, K, RSR_dB, p_H, number_of_stages, training_samples_per_epoch, batch_size, learning_rate, warm_start_file, initialization_method, maximum_gs_updates, gs_convergence_tolerance, parameters_per_stage, number_of_gate_features, number_of_gate_outputs);
    fprintf('P=%d selected epoch=%d; fresh validation-subset aggregate NMSE=%.4f dB.\n', P, best_epoch, validation_nmse_dB);
    fprintf('Saved model: %s\n', selected_model_file);
end

fprintf('\nV5 sixteen-stage spectrum-adaptive Hankel training and validation complete.\n');
fprintf('No reserved test files were loaded.\n');
fprintf('Results are in: %s\n', output_folder);

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
    sample_loss = sum(abs(G_estimate-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
    total_loss = total_loss+sample_loss;
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

function validation_loss = local_evaluate_loss(parameters, data, sample_indices, initial_cache, initial_ready, p_H, N, K, number_of_stages)
total_loss = 0;
for sample_index = 1:numel(sample_indices)
    data_index = sample_indices(sample_index);
    Y = double(data.Y_data(:, :, data_index));
    G_true = double(data.G_data(:, :, data_index));
    S = double(data.S_data(:, :, data_index));
    B = double(data.B_data(:, :, data_index));
    sigma2 = double(data.sigma2_data(data_index));
    if ~initial_ready(data_index)
        error('Missing GS validation initialization for sample %d.', data_index);
    end
    G_initial = initial_cache(:, :, data_index);
    G_estimate = local_unfolded_forward_numeric(parameters, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);
    total_loss = total_loss+sum(abs(G_estimate-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
end
validation_loss = total_loss/numel(sample_indices);
end

function [aggregate_nmse, validation_by_snr] = local_evaluate_full_validation(parameters, data, sample_indices, initial_cache, SNR_grid, p_H, N, K, number_of_stages)
number_of_snr_points = numel(SNR_grid);
validation_by_snr = zeros(number_of_snr_points, 1);
all_nmse = zeros(numel(sample_indices), 1);
for sample_position = 1:numel(sample_indices)
    data_index = sample_indices(sample_position);
    Y = double(data.Y_data(:, :, data_index));
    G_true = double(data.G_data(:, :, data_index));
    S = double(data.S_data(:, :, data_index));
    B = double(data.B_data(:, :, data_index));
    sigma2 = double(data.sigma2_data(data_index));
    G_initial = initial_cache(:, :, data_index);
    G_estimate = local_unfolded_forward_numeric(parameters, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);
    all_nmse(sample_position) = sum(abs(G_estimate-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
end
for snr_index = 1:number_of_snr_points
    snr_sample_positions = find(double(data.SNR_dB_data(sample_indices)) == SNR_grid(snr_index));
    validation_by_snr(snr_index) = 10*log10(mean(all_nmse(snr_sample_positions)));
end
aggregate_nmse = mean(all_nmse);
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

function selected_indices = local_select_fresh_validation_indices(SNR_dB_data, SNR_grid, samples_per_SNR)
selected_indices = [];
for snr_index = 1:numel(SNR_grid)
    candidates = find(double(SNR_dB_data) == SNR_grid(snr_index));
    if numel(candidates) <= 2*samples_per_SNR
        error('Not enough validation samples to reserve separate subsets for SNR=%g dB.', SNR_grid(snr_index));
    end
    previous_selection_positions = round(linspace(1, numel(candidates), samples_per_SNR));
    fresh_candidates = candidates;
    fresh_candidates(previous_selection_positions) = [];
    fresh_selection_positions = round(linspace(1, numel(fresh_candidates), samples_per_SNR));
    selected_indices = [selected_indices; fresh_candidates(fresh_selection_positions(:))];
end
end

function local_save_checkpoint(file_name, current_parameters, best_parameters_saved, best_epoch, best_validation_loss, history_saved, parameter_update_count, epoch, P, N, K, RSR_dB, p_H, number_of_stages, training_samples_per_epoch, batch_size, learning_rate, warm_start_file, initialization_method, maximum_gs_updates, gs_convergence_tolerance, parameters_per_stage)
save(file_name, 'current_parameters', 'best_parameters_saved', 'best_epoch', 'best_validation_loss', 'history_saved', 'parameter_update_count', 'epoch', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'training_samples_per_epoch', 'batch_size', 'learning_rate', 'warm_start_file', 'initialization_method', 'maximum_gs_updates', 'gs_convergence_tolerance', 'parameters_per_stage');
end

function local_save_selected_model(file_name, best_parameters_numeric, best_epoch_saved, best_validation_loss, validation_table, validation_nmse_dB, P, N, K, RSR_dB, p_H, number_of_stages, training_samples_per_epoch, batch_size, learning_rate, warm_start_file, initialization_method, maximum_gs_updates, gs_convergence_tolerance, parameters_per_stage, number_of_gate_features, number_of_gate_outputs)
save(file_name, 'best_parameters_numeric', 'best_epoch_saved', 'best_validation_loss', 'validation_table', 'validation_nmse_dB', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'training_samples_per_epoch', 'batch_size', 'learning_rate', 'warm_start_file', 'initialization_method', 'maximum_gs_updates', 'gs_convergence_tolerance', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs');
end
