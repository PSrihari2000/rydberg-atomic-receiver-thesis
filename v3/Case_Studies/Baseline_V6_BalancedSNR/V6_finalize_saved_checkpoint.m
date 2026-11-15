%% Finalize the best saved V6 checkpoints after an interrupted screening run
% This script reads only the V6 validation data, saved training checkpoints,
% and frozen V4 model files. It never loads a test file.

clear;
clc;

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
training_samples_per_epoch = 500;
validation_samples_per_SNR_for_selection = 32;
batch_size = 8;
learning_rate = 0.003;
maximum_gs_updates = 5000;
gs_convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
training_objective = 'mean_per_sample_log_nmse';

script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
v3_folder = fileparts(case_studies_folder);
data_folder = fullfile(script_folder, 'data', 'N8_K3_RSR25dB');
result_folder = fullfile(script_folder, 'results', 'V6_PI_DU_RACE_BALANCED_LOG_NMSE_SCREENING');
v4_model_folder = fullfile(v3_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');

if ~isfolder(data_folder) || ~isfolder(result_folder) || ~isfolder(v4_model_folder)
    error('V6 data, checkpoint folder, or frozen V4 models are missing.');
end
if ~isfile(fullfile(script_folder, 'V3_CLASSICAL_ESTIMATOR.m'))
    error('The local V3_CLASSICAL_ESTIMATOR.m helper is missing.');
end
addpath(script_folder);

fprintf('Finalize V6 screening from saved best checkpoints. Validation only; no test files are loaded.\n');

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    checkpoint_file = fullfile(result_folder, sprintf('checkpoint_P%d.mat', P));
    validation_file = fullfile(data_folder, sprintf('validation_P%d.mat', P));
    v4_model_file = fullfile(v4_model_folder, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    selected_model_file = fullfile(result_folder, sprintf('selected_model_P%d.mat', P));
    if ~isfile(checkpoint_file) || ~isfile(validation_file) || ~isfile(v4_model_file)
        error('Checkpoint, V6 validation data, or V4 model is missing for P=%d.', P);
    end
    if isfile(selected_model_file)
        error('Selected V6 model already exists for P=%d. Inspect it; refusing to overwrite.', P);
    end

    checkpoint = load(checkpoint_file);
    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'RSR_dB', 'SNR_dB', 'split_name', 'data_version');
    v4_model = load(v4_model_file, 'best_parameters_numeric', 'N', 'K', 'P', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs', 'initialization_method');

    if ~isfield(checkpoint, 'best_parameters_saved') || ~isfield(checkpoint, 'best_epoch') || ~isfield(checkpoint, 'best_validation_loss') || isempty(checkpoint.best_parameters_saved)
        error('Checkpoint does not contain a complete best-model state for P=%d.', P);
    end
    if checkpoint.P ~= P || checkpoint.N ~= N || checkpoint.K ~= K || checkpoint.RSR_dB ~= RSR_dB || checkpoint.p_H ~= p_H || checkpoint.number_of_stages ~= number_of_stages || ~isequal(size(checkpoint.best_parameters_saved), [parameters_per_stage number_of_stages])
        error('V6 checkpoint metadata mismatch for P=%d.', P);
    end
    if validation_data.number_of_samples ~= 4000 || validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ~isequal(double(validation_data.SNR_dB), SNR_grid) || ~strcmp(validation_data.split_name, 'validation') || ~strcmp(validation_data.data_version, 'V6_balanced_log_nmse_fresh')
        error('V6 validation data metadata mismatch for P=%d.', P);
    end
    if v4_model.N ~= N || v4_model.K ~= K || v4_model.P ~= P || v4_model.RSR_dB ~= RSR_dB || v4_model.p_H ~= p_H || v4_model.number_of_stages ~= number_of_stages || v4_model.parameters_per_stage ~= parameters_per_stage || ~strcmp(v4_model.initialization_method, 'GS')
        error('Frozen V4 model metadata mismatch for P=%d.', P);
    end

    selection_indices = local_select_fresh_validation_indices(double(validation_data.SNR_dB_data), SNR_grid, validation_samples_per_SNR_for_selection);
    initial_cache = complex(zeros(N, K, validation_data.number_of_samples));
    for sample_position = 1:numel(selection_indices)
        data_index = selection_indices(sample_position);
        [initial_cache(:, :, data_index), ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', double(validation_data.Y_data(:, :, data_index)), double(validation_data.S_data(:, :, data_index)), double(validation_data.B_data(:, :, data_index)), double(validation_data.sigma2_data(data_index)), p_H, 0, maximum_gs_updates, gs_convergence_tolerance, gd_initial_variance, data_index);
    end

    v6_parameters = checkpoint.best_parameters_saved;
    v4_nmse_by_snr = zeros(numel(SNR_grid), 1);
    v6_nmse_by_snr = zeros(numel(SNR_grid), 1);
    for snr_index = 1:numel(SNR_grid)
        condition_indices = selection_indices(double(validation_data.SNR_dB_data(selection_indices)) == SNR_grid(snr_index));
        condition_v4_nmse = zeros(numel(condition_indices), 1);
        condition_v6_nmse = zeros(numel(condition_indices), 1);
        for sample_position = 1:numel(condition_indices)
            data_index = condition_indices(sample_position);
            Y = double(validation_data.Y_data(:, :, data_index));
            G_true = double(validation_data.G_data(:, :, data_index));
            S = double(validation_data.S_data(:, :, data_index));
            B = double(validation_data.B_data(:, :, data_index));
            sigma2 = double(validation_data.sigma2_data(data_index));
            G_initial = initial_cache(:, :, data_index);
            true_energy = max(sum(abs(G_true).^2, 'all'), eps);
            G_v4 = local_unfolded_forward_numeric(v4_model.best_parameters_numeric, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);
            G_v6 = local_unfolded_forward_numeric(v6_parameters, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);
            condition_v4_nmse(sample_position) = sum(abs(G_v4-G_true).^2, 'all')/true_energy;
            condition_v6_nmse(sample_position) = sum(abs(G_v6-G_true).^2, 'all')/true_energy;
        end
        v4_nmse_by_snr(snr_index) = 10*log10(mean(condition_v4_nmse));
        v6_nmse_by_snr(snr_index) = 10*log10(mean(condition_v6_nmse));
    end

    validation_gain_dB = v4_nmse_by_snr-v6_nmse_by_snr;
    validation_table = table(repmat(P, numel(SNR_grid), 1), SNR_grid(:), v4_nmse_by_snr, v6_nmse_by_snr, validation_gain_dB, 'VariableNames', {'P','SNR_dB','V4WarmStartValidationNMSE_dB','V6SelectedValidationNMSE_dB','V6GainOverV4_dB'});
    validation_csv = fullfile(result_folder, sprintf('validation_V4_vs_V6_P%d.csv', P));
    history_csv = fullfile(result_folder, sprintf('training_history_P%d.csv', P));
    if isfile(validation_csv) || isfile(history_csv)
        error('A final validation output already exists for P=%d. Inspect the folder before continuing.', P);
    end
    writetable(validation_table, validation_csv);
    if isfield(checkpoint, 'history_saved')
        history_table = array2table(checkpoint.history_saved, 'VariableNames', {'Epoch','TrainingMeanLogNMSE','ValidationMeanLogNMSE','ValidationMeanSampleNMSE_dB','EpochRuntimeSeconds'});
        writetable(history_table, history_csv);
    end

    best_parameters_numeric = v6_parameters;
    best_epoch_saved = checkpoint.best_epoch;
    best_validation_loss = checkpoint.best_validation_loss;
    full_validation_nmse = mean(10.^(v6_nmse_by_snr/10));
    validation_nmse_dB = 10*log10(full_validation_nmse);
    warm_start_file = v4_model_file;
    initialization_method = 'GS';
    validation_selection_indices = selection_indices;
    save(selected_model_file, 'best_parameters_numeric', 'best_epoch_saved', 'best_validation_loss', 'validation_table', 'validation_nmse_dB', 'validation_selection_indices', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'training_samples_per_epoch', 'batch_size', 'learning_rate', 'warm_start_file', 'initialization_method', 'maximum_gs_updates', 'gs_convergence_tolerance', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs', 'training_objective');
    fprintf('P=%d: finalized checkpoint from epoch %d; saved %s\n', P, best_epoch_saved, selected_model_file);
end

fprintf('V6 checkpoints finalized. Test data were not loaded.\n');

function selected_indices = local_select_fresh_validation_indices(SNR_dB_data, SNR_grid, samples_per_SNR)
selected_indices = [];
for snr_index = 1:numel(SNR_grid)
    candidates = find(double(SNR_dB_data) == SNR_grid(snr_index));
    if numel(candidates) <= 2*samples_per_SNR
        error('Not enough validation samples for the V6 selection subset.');
    end
    first_selection_positions = round(linspace(1, numel(candidates), samples_per_SNR));
    unused_candidates = candidates;
    unused_candidates(first_selection_positions) = [];
    second_selection_positions = round(linspace(1, numel(unused_candidates), samples_per_SNR));
    selected_indices = [selected_indices; unused_candidates(second_selection_positions(:))];
end
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
