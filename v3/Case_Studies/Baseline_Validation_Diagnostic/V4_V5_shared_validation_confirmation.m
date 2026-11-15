%% Matched validation comparison of frozen V4 and screened V5 models
% Evaluates both models on the exact 64-sample-per-SNR confirmation subset
% saved by V4_stage_gate_validation_confirmation.m.
% Training, validation-selection, and test files are not modified or loaded.

clear;
clc;
close all;

%% -------------------- FIXED SCENARIO --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
number_of_stages = 16;
parameters_per_stage = 15;
maximum_gs_updates = 5000;
gs_convergence_tolerance = 1e-10;
samples_per_condition = 64;
t_critical_95_df63 = 1.998;

%% -------------------- INPUT AND OUTPUT PATHS --------------------
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
v3_folder = fileparts(case_studies_folder);
validation_folder = fullfile(case_studies_folder, 'Baseline_V5', 'data', 'N8_K3_RSR25dB');
v4_model_folder = fullfile(v3_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
v5_model_folder = fullfile(case_studies_folder, 'Baseline_V5', 'results', 'V5_PI_DU_RACE_16STAGE_SCREENING');
confirmation_sample_file = fullfile(script_folder, 'results', 'V4_STAGE_GATE_VALIDATION_CONFIRMATION', 'V4_validation_confirmation_paired_samples.csv');
output_folder = fullfile(script_folder, 'results', 'V4_V5_SHARED_VALIDATION_CONFIRMATION');

if isfolder(output_folder)
    error('Output folder already exists. Inspect it or choose a new output-folder name.');
end
if ~isfolder(validation_folder) || ~isfolder(v4_model_folder) || ~isfolder(v5_model_folder) || ~isfile(confirmation_sample_file)
    error('Validation data, V4/V5 selected models, or the saved confirmation sample table is missing.');
end

confirmation_table = readtable(confirmation_sample_file);
required_confirmation_columns = {'P','SNR_dB','ValidationSampleIndex'};
if ~all(ismember(required_confirmation_columns, confirmation_table.Properties.VariableNames))
    error('The saved confirmation table is missing required sample-index columns.');
end

fprintf('Matched V4 versus V5 comparison; validation data only.\n');
fprintf('Only the held-out confirmation indices from the validation split are scored. Training and test files are not loaded.\n');
fprintf('P=[%s], N=%d, K=%d, RSR=%g dB; %d samples per P/SNR condition.\n', num2str(P_list), N, K, RSR_dB, samples_per_condition);
fprintf('V5 model source: %s\n', v5_model_folder);
fprintf('Output folder: %s\n', output_folder);

%% -------------------- OUTPUT STORAGE --------------------
sample_rows = cell(0, 9);
condition_rows = cell(0, 12);
comparison_config = struct;
comparison_config.N = N;
comparison_config.K = K;
comparison_config.P_list = P_list;
comparison_config.SNR_grid = SNR_grid;
comparison_config.RSR_dB = RSR_dB;
comparison_config.p_H = p_H;
comparison_config.number_of_stages = number_of_stages;
comparison_config.samples_per_condition = samples_per_condition;
comparison_config.validation_split = 'V5_fresh_validation_selected';
comparison_config.test_files_loaded = false;
comparison_config.V4_model_folder = v4_model_folder;
comparison_config.V5_model_folder = v5_model_folder;
comparison_config.confirmation_sample_file = confirmation_sample_file;

%% -------------------- LOAD MODELS AND SCORE MATCHED SAMPLES --------------------
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(validation_folder, sprintf('validation_P%d.mat', P));
    v4_model_file = fullfile(v4_model_folder, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    v5_model_file = fullfile(v5_model_folder, sprintf('selected_model_P%d.mat', P));
    if ~isfile(validation_file) || ~isfile(v4_model_file) || ~isfile(v5_model_file)
        error('A required validation or selected-model file is missing for P=%d.', P);
    end

    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'SNR_dB', 'number_of_samples', 'split_name', 'data_version');
    v4_model = load(v4_model_file, 'best_parameters_numeric', 'N', 'K', 'P', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'initialization_method');
    v5_model = load(v5_model_file, 'best_parameters_numeric', 'N', 'K', 'P', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'initialization_method', 'best_epoch_saved');

    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || validation_data.number_of_samples ~= 4000 || ~strcmp(validation_data.split_name, 'validation') || ~strcmp(validation_data.data_version, 'V5_fresh_validation_selected') || ~isequal(double(validation_data.SNR_dB), SNR_grid)
        error('Validation-data metadata mismatch for P=%d.', P);
    end
    if ~local_model_metadata_valid(v4_model, N, K, P, RSR_dB, p_H, number_of_stages, parameters_per_stage)
        error('Frozen V4 model metadata mismatch for P=%d.', P);
    end
    if ~local_model_metadata_valid(v5_model, N, K, P, RSR_dB, p_H, number_of_stages, parameters_per_stage)
        error('Screening V5 model metadata mismatch for P=%d.', P);
    end

    p_confirmation_table = confirmation_table(confirmation_table.P == P, :);
    if height(p_confirmation_table) ~= samples_per_condition*numel(SNR_grid)
        error('Expected %d saved confirmation samples for P=%d.', samples_per_condition*numel(SNR_grid), P);
    end
    fprintf('\nP=%d: V5 selected epoch=%d; scoring %d matched validation samples.\n', P, v5_model.best_epoch_saved, height(p_confirmation_table));

    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        condition_rows_for_snr = p_confirmation_table(abs(p_confirmation_table.SNR_dB-current_SNR_dB) < 1e-9, :);
        if height(condition_rows_for_snr) ~= samples_per_condition
            error('Expected %d saved confirmation samples for P=%d, SNR=%g dB.', samples_per_condition, P, current_SNR_dB);
        end
        sample_indices = double(condition_rows_for_snr.ValidationSampleIndex(:));
        if numel(unique(sample_indices)) ~= samples_per_condition || any(sample_indices < 1) || any(sample_indices > validation_data.number_of_samples)
            error('Confirmation sample indices are invalid for P=%d, SNR=%g dB.', P, current_SNR_dB);
        end
        if any(abs(double(validation_data.SNR_dB_data(sample_indices))-current_SNR_dB) > 1e-9)
            error('A saved confirmation index has a different SNR for P=%d, SNR=%g dB.', P, current_SNR_dB);
        end

        v4_nmse = zeros(samples_per_condition, 1);
        v5_nmse = zeros(samples_per_condition, 1);
        gs_updates = zeros(samples_per_condition, 1);
        gs_converged = false(samples_per_condition, 1);

        for sample_position = 1:samples_per_condition
            sample_index = sample_indices(sample_position);
            Y = double(validation_data.Y_data(:, :, sample_index));
            G_true = double(validation_data.G_data(:, :, sample_index));
            S = double(validation_data.S_data(:, :, sample_index));
            B = double(validation_data.B_data(:, :, sample_index));
            sigma2 = double(validation_data.sigma2_data(sample_index));

            G_spectral = local_spectral_initialize(Y, S, B);
            [G_initial, gs_updates(sample_position), gs_converged(sample_position)] = local_biased_gs(Y, S, B, G_spectral, maximum_gs_updates, gs_convergence_tolerance);
            G_v4 = local_unfolded_forward_numeric(v4_model.best_parameters_numeric, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);
            G_v5 = local_unfolded_forward_numeric(v5_model.best_parameters_numeric, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);

            true_energy = max(sum(abs(G_true).^2, 'all'), eps);
            v4_nmse(sample_position) = sum(abs(G_v4-G_true).^2, 'all')/true_energy;
            v5_nmse(sample_position) = sum(abs(G_v5-G_true).^2, 'all')/true_energy;
            paired_gain_dB = 10*log10(max(v4_nmse(sample_position), realmin)/max(v5_nmse(sample_position), realmin));
            sample_rows(end+1, :) = {P, current_SNR_dB, sample_index, v4_nmse(sample_position), v5_nmse(sample_position), 10*log10(max(v4_nmse(sample_position), realmin)), 10*log10(max(v5_nmse(sample_position), realmin)), paired_gain_dB, gs_converged(sample_position)};
        end

        paired_gains = 10*log10(max(v4_nmse, realmin)./max(v5_nmse, realmin));
        mean_v4_nmse = mean(v4_nmse);
        mean_v5_nmse = mean(v5_nmse);
        mean_paired_gain = mean(paired_gains);
        standard_error = std(paired_gains, 0)/sqrt(samples_per_condition);
        confidence_half_width = t_critical_95_df63*standard_error;
        v5_win_count = sum(paired_gains > 0);
        condition_rows(end+1, :) = {P, current_SNR_dB, samples_per_condition, mean_v4_nmse, mean_v5_nmse, 10*log10(max(mean_v4_nmse, realmin)), 10*log10(max(mean_v5_nmse, realmin)), mean_paired_gain, mean_paired_gain-confidence_half_width, mean_paired_gain+confidence_half_width, v5_win_count, sum(~gs_converged)};
    end
end

%% -------------------- SAVE VALIDATION COMPARISON --------------------
sample_table = cell2table(sample_rows, 'VariableNames', {'P','SNR_dB','ValidationSampleIndex','V4_NMSE_linear','V5_NMSE_linear','V4_NMSE_dB','V5_NMSE_dB','PairedGain_V5overV4_dB','GS_Converged'});
condition_table = cell2table(condition_rows, 'VariableNames', {'P','SNR_dB','SampleCount','V4_MeanNMSE_linear','V5_MeanNMSE_linear','V4_MeanNMSE_dB','V5_MeanNMSE_dB','MeanPairedGain_V5overV4_dB','PairedGain_CI95_Lower_dB','PairedGain_CI95_Upper_dB','V5_WinCount','GS_NonconvergedCount'});

mkdir(output_folder);
sample_csv = fullfile(output_folder, 'V4_V5_confirmation_paired_samples.csv');
condition_csv = fullfile(output_folder, 'V4_V5_confirmation_by_condition.csv');
mat_file = fullfile(output_folder, 'V4_V5_confirmation_results.mat');
writetable(sample_table, sample_csv);
writetable(condition_table, condition_csv);
save(mat_file, 'sample_table', 'condition_table', 'comparison_config', '-v7.3');

fprintf('\nSaved paired validation samples: %s\n', sample_csv);
fprintf('Saved per-condition comparison: %s\n', condition_csv);
fprintf('Saved MATLAB results: %s\n', mat_file);
fprintf('Positive paired gain means V5 has lower NMSE than V4. No test data were loaded.\n');

%% -------------------- LOCAL MODEL METADATA CHECK --------------------
function valid = local_model_metadata_valid(model, N, K, P, RSR_dB, p_H, number_of_stages, parameters_per_stage)
valid = model.N == N && model.K == K && model.P == P && model.RSR_dB == RSR_dB && model.p_H == p_H && model.number_of_stages == number_of_stages && model.parameters_per_stage == parameters_per_stage && isequal(size(model.best_parameters_numeric), [parameters_per_stage number_of_stages]) && strcmp(model.initialization_method, 'GS');
end

%% -------------------- BIASED-GS SPECTRAL INITIALIZATION --------------------
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

%% -------------------- ITERATIVE BIASED GAUSS-SEIDEL --------------------
function [G, iteration_count, converged] = local_biased_gs(Y, S, B, G_initial, maximum_updates, convergence_tolerance)
G = G_initial;
right_inverse = S'/(S*S');
iteration_count = maximum_updates;
converged = false;
for update_index = 1:maximum_updates
    G_previous = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-G_previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

%% -------------------- SAME NUMERICAL INFERENCE FOR BOTH MODELS --------------------
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

%% -------------------- SPECTRUM-ADAPTIVE HANKEL SHRINKAGE --------------------
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

%% -------------------- HANKEL EMBEDDING MAP --------------------
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
