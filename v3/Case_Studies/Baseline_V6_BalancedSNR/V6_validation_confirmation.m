%% Paired validation confirmation for the completed V6 screening model
% Uses validation samples excluded from the checkpoint-selection subset.
% Compares the frozen V4 model and selected V6 model on identical samples.
% No test files are loaded.

clear;
clc;

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
gd_initial_variance = 0.1;
confirmation_samples_per_SNR = 64;
t_critical_95_df63 = 1.998;

script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
v3_folder = fileparts(case_studies_folder);
data_folder = fullfile(script_folder, 'data', 'N8_K3_RSR25dB');
v4_model_folder = fullfile(v3_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
v6_model_folder = fullfile(script_folder, 'results', 'V6_PI_DU_RACE_BALANCED_LOG_NMSE_SCREENING');
output_folder = fullfile(script_folder, 'results', 'V6_VALIDATION_CONFIRMATION');
if isfolder(output_folder)
    error('Confirmation output folder already exists. Inspect it or choose a new folder name.');
end
if ~isfolder(data_folder) || ~isfolder(v4_model_folder) || ~isfolder(v6_model_folder) || ~isfile(fullfile(script_folder, 'V3_CLASSICAL_ESTIMATOR.m'))
    error('V6 validation data, V4/V6 models, or the local classical-estimator helper is missing.');
end
addpath(script_folder);
mkdir(output_folder);

fprintf('V4 versus V6 paired confirmation; unused V6 validation samples only.\n');
fprintf('No training data or test files are loaded.\n');
fprintf('64 samples per SNR and P; output: %s\n', output_folder);

sample_rows = cell(0, 9);
condition_rows = cell(0, 12);

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_data = load(fullfile(data_folder, sprintf('validation_P%d.mat', P)), 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'RSR_dB', 'SNR_dB', 'split_name', 'data_version');
    v4_model = load(fullfile(v4_model_folder, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P)), 'best_parameters_numeric', 'N', 'K', 'P', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'initialization_method');
    v6_model = load(fullfile(v6_model_folder, sprintf('selected_model_P%d.mat', P)), 'best_parameters_numeric', 'validation_selection_indices', 'best_epoch_saved', 'N', 'K', 'P', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'initialization_method');

    if validation_data.number_of_samples ~= 4000 || validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ~isequal(double(validation_data.SNR_dB), SNR_grid) || ~strcmp(validation_data.split_name, 'validation') || ~strcmp(validation_data.data_version, 'V6_balanced_log_nmse_fresh')
        error('Fresh V6 validation metadata mismatch for P=%d.', P);
    end
    if ~local_model_metadata_valid(v4_model, N, K, P, RSR_dB, p_H, number_of_stages, parameters_per_stage) || ~local_model_metadata_valid(v6_model, N, K, P, RSR_dB, p_H, number_of_stages, parameters_per_stage)
        error('V4 or V6 model metadata mismatch for P=%d.', P);
    end

    selected_indices = double(v6_model.validation_selection_indices(:));
    if numel(unique(selected_indices)) ~= numel(selected_indices) || any(selected_indices < 1) || any(selected_indices > validation_data.number_of_samples)
        error('Saved V6 selection indices are invalid for P=%d.', P);
    end

    fprintf('\nP=%d: V6 selected epoch=%d.\n', P, v6_model.best_epoch_saved);
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        all_condition_indices = find(double(validation_data.SNR_dB_data(:)) == current_SNR_dB);
        condition_candidates = setdiff(all_condition_indices, selected_indices, 'stable');
        if numel(condition_candidates) < confirmation_samples_per_SNR
            error('Not enough unused validation samples for P=%d, SNR=%g dB.', P, current_SNR_dB);
        end
        rng(2026111600+P+snr_index, 'twister');
        chosen_positions = randperm(numel(condition_candidates), confirmation_samples_per_SNR);
        sample_indices = condition_candidates(chosen_positions(:));

        v4_nmse = zeros(confirmation_samples_per_SNR, 1);
        v6_nmse = zeros(confirmation_samples_per_SNR, 1);
        gs_updates = zeros(confirmation_samples_per_SNR, 1);
        gs_converged = false(confirmation_samples_per_SNR, 1);

        for sample_position = 1:confirmation_samples_per_SNR
            sample_index = sample_indices(sample_position);
            Y = double(validation_data.Y_data(:, :, sample_index));
            G_true = double(validation_data.G_data(:, :, sample_index));
            S = double(validation_data.S_data(:, :, sample_index));
            B = double(validation_data.B_data(:, :, sample_index));
            sigma2 = double(validation_data.sigma2_data(sample_index));
            [G_initial, gs_updates(sample_position), gs_converged(sample_position)] = V3_CLASSICAL_ESTIMATOR('GS', Y, S, B, sigma2, p_H, 0, maximum_gs_updates, gs_convergence_tolerance, gd_initial_variance, sample_index);
            true_energy = max(sum(abs(G_true).^2, 'all'), eps);
            G_v4 = local_unfolded_forward_numeric(v4_model.best_parameters_numeric, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);
            G_v6 = local_unfolded_forward_numeric(v6_model.best_parameters_numeric, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages);
            v4_nmse(sample_position) = sum(abs(G_v4-G_true).^2, 'all')/true_energy;
            v6_nmse(sample_position) = sum(abs(G_v6-G_true).^2, 'all')/true_energy;
            paired_gain_dB = 10*log10(max(v4_nmse(sample_position), realmin)/max(v6_nmse(sample_position), realmin));
            sample_rows(end+1, :) = {P, current_SNR_dB, sample_index, v4_nmse(sample_position), v6_nmse(sample_position), 10*log10(max(v4_nmse(sample_position), realmin)), 10*log10(max(v6_nmse(sample_position), realmin)), paired_gain_dB, gs_converged(sample_position)};
        end

        paired_gains = 10*log10(max(v4_nmse, realmin)./max(v6_nmse, realmin));
        mean_paired_gain = mean(paired_gains);
        confidence_half_width = t_critical_95_df63*std(paired_gains)/sqrt(confirmation_samples_per_SNR);
        condition_rows(end+1, :) = {P, current_SNR_dB, confirmation_samples_per_SNR, mean(v4_nmse), mean(v6_nmse), 10*log10(max(mean(v4_nmse), realmin)), 10*log10(max(mean(v6_nmse), realmin)), mean_paired_gain, mean_paired_gain-confidence_half_width, mean_paired_gain+confidence_half_width, sum(paired_gains > 0), sum(~gs_converged)};
        fprintf('P=%d SNR=%g dB: paired V6 gain=%+.4f dB; GS nonconverged=%d/%d.\n', P, current_SNR_dB, mean_paired_gain, sum(~gs_converged), confirmation_samples_per_SNR);
    end
end

sample_table = cell2table(sample_rows, 'VariableNames', {'P','SNR_dB','ValidationSampleIndex','V4_NMSE_linear','V6_NMSE_linear','V4_NMSE_dB','V6_NMSE_dB','PairedGain_V6overV4_dB','GS_Converged'});
condition_table = cell2table(condition_rows, 'VariableNames', {'P','SNR_dB','SampleCount','V4_MeanNMSE_linear','V6_MeanNMSE_linear','V4_MeanNMSE_dB','V6_MeanNMSE_dB','MeanPairedGain_V6overV4_dB','PairedGain_CI95_Lower_dB','PairedGain_CI95_Upper_dB','V6_WinCount','GS_NonconvergedCount'});
writetable(sample_table, fullfile(output_folder, 'V4_V6_unused_validation_paired_samples.csv'));
writetable(condition_table, fullfile(output_folder, 'V4_V6_unused_validation_by_condition.csv'));
save(fullfile(output_folder, 'V4_V6_unused_validation_confirmation.mat'), 'sample_table', 'condition_table', 'N', 'K', 'P_list', 'SNR_grid', 'RSR_dB', 'p_H', 'number_of_stages', 'confirmation_samples_per_SNR');
fprintf('\nSaved paired results and condition summary: %s\n', output_folder);
fprintf('Positive gain favors V6. No test files were loaded.\n');

function valid = local_model_metadata_valid(model, N, K, P, RSR_dB, p_H, number_of_stages, parameters_per_stage)
valid = model.N == N && model.K == K && model.P == P && model.RSR_dB == RSR_dB && model.p_H == p_H && model.number_of_stages == number_of_stages && model.parameters_per_stage == parameters_per_stage && strcmp(model.initialization_method, 'GS') && isequal(size(model.best_parameters_numeric), [parameters_per_stage number_of_stages]);
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
