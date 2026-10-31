%% Held-out V3 validation comparison: classical methods and 8/16-stage PI-DU-RACE
% Excludes 64 validation samples per SNR used to select each DL checkpoint.
% Uses the same remaining validation observations for every method.
% The frozen HS-GS policy was selected on validation; final claims require test.
% This script loads validation and selected-model files only.

clear;
clc;
close all;
rng(20261030, 'twister');

N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
validation_samples_per_SNR_for_checkpoint = 64;
bootstrap_replicates = 2000;
method_names = {'LinearInit','GS','GD','EM-GS','HS-GS','PI-DU-RACE-8stage','PI-DU-RACE-16stage'};
gain_baseline_indices = [1 2 3 4 5 6];
gain_baseline_names = {'LinearInit','GS','GD','EMGS','HSGS','EightStage'};

script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
addpath(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
classical_folder = fullfile(v3_folder, 'results', 'N8_CLASSICAL_COMPARISON_RSR25DB');
eight_stage_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_8STAGE_ADAPTIVE_GATE_RSR25DB');
sixteen_stage_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_ADAPTIVE_GATE_RSR25DB');
policy_file = fullfile(classical_folder, 'N8_frozen_HSGS_validation_policy.mat');
output_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_VALIDATION_CONFIRMATION_RSR25DB');

if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it before running again.');
end
if ~isfolder(scenario_folder) || ~isfolder(eight_stage_folder) || ~isfolder(sixteen_stage_folder) || ~isfile(policy_file)
    error('A required validation data, model, or frozen HS-GS policy path is missing.');
end
if ~isfile(fullfile(script_folder, 'V3_CLASSICAL_ESTIMATOR.m'))
    error('V3_CLASSICAL_ESTIMATOR.m must be in the scripts folder.');
end
mkdir(output_folder);

policy_data = load(policy_file, 'configuration', 'frozen_multiplier');
configuration = policy_data.configuration;
if configuration.N ~= N || configuration.K ~= K || configuration.RSR_dB ~= RSR_dB || ~isequal(configuration.P_list, P_list) || ~isequal(configuration.SNR_dB, SNR_grid) || ~strcmp(configuration.data_version, 'V3_AoA_Uminus90to90')
    error('The frozen HS-GS policy does not match this V3 scenario.');
end
p_H = configuration.p_H;
maximum_updates = configuration.maximum_updates;
convergence_tolerance = configuration.convergence_tolerance;
gd_initial_variance = configuration.gd_initial_variance;

fprintf('Held-out validation: classical, 8-stage, and 16-stage estimators.\n');
fprintf('The 64 checkpoint-selection samples per SNR are excluded.\n');
fprintf('No training or reserved test files will be loaded.\n');
fprintf('HS-GS policy was selected on validation; final evidence requires test.\n');

number_of_hankel_rows = N-p_H;
number_of_hankel_columns = p_H+1;
hankel_map = zeros(number_of_hankel_rows*number_of_hankel_columns, N);
for row_index = 1:number_of_hankel_rows
    for column_index = 1:number_of_hankel_columns
        linear_index = row_index+(column_index-1)*number_of_hankel_rows;
        antenna_index = row_index+column_index-1;
        hankel_map(linear_index, antenna_index) = 1;
    end
end
hankel_diagonal_counts = sum(hankel_map, 1).';

number_of_methods = numel(method_names);
number_of_gains = numel(gain_baseline_indices);
result_rows = zeros(numel(P_list)*numel(SNR_grid), 5+number_of_methods+3*number_of_gains);
row_index = 0;
all_errors = cell(numel(P_list), 1);
all_true_energy = cell(numel(P_list), 1);
all_snr_values = cell(numel(P_list), 1);
all_confirmation_masks = cell(numel(P_list), 1);

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    eight_stage_file = fullfile(eight_stage_folder, sprintf('selected_model_P%d.mat', P));
    sixteen_stage_file = fullfile(sixteen_stage_folder, sprintf('selected_model_P%d.mat', P));
    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'AoA_deg_range', 'data_version');
    eight_stage_data = load(eight_stage_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs');
    sixteen_stage_data = load(sixteen_stage_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs');

    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ~isequal(double(validation_data.AoA_deg_range), [-90 90]) || ~strcmp(validation_data.data_version, 'V3_AoA_Uminus90to90')
        error('Validation metadata mismatch for P=%d.', P);
    end
    if eight_stage_data.N ~= N || eight_stage_data.K ~= K || eight_stage_data.P ~= P || eight_stage_data.RSR_dB ~= RSR_dB || eight_stage_data.p_H ~= p_H || eight_stage_data.number_of_stages ~= 8 || eight_stage_data.parameters_per_stage ~= 13 || eight_stage_data.number_of_gate_features ~= 3 || eight_stage_data.number_of_gate_outputs ~= 3 || ~isequal(size(eight_stage_data.best_parameters_numeric), [13 8])
        error('Eight-stage model metadata mismatch for P=%d.', P);
    end
    if sixteen_stage_data.N ~= N || sixteen_stage_data.K ~= K || sixteen_stage_data.P ~= P || sixteen_stage_data.RSR_dB ~= RSR_dB || sixteen_stage_data.p_H ~= p_H || sixteen_stage_data.number_of_stages ~= 16 || sixteen_stage_data.parameters_per_stage ~= 13 || sixteen_stage_data.number_of_gate_features ~= 3 || sixteen_stage_data.number_of_gate_outputs ~= 3 || ~isequal(size(sixteen_stage_data.best_parameters_numeric), [13 16])
        error('Sixteen-stage model metadata mismatch for P=%d.', P);
    end

    sample_count = size(validation_data.Y_data, 3);
    confirmation_mask = false(sample_count, 1);
    for snr_index = 1:numel(SNR_grid)
        candidates = find(double(validation_data.SNR_dB_data) == SNR_grid(snr_index));
        if numel(candidates) <= validation_samples_per_SNR_for_checkpoint
            error('Not enough validation samples for P=%d, SNR=%g dB.', P, SNR_grid(snr_index));
        end
        selected_positions = round(linspace(1, numel(candidates), validation_samples_per_SNR_for_checkpoint));
        confirmation_mask(candidates) = true;
        confirmation_mask(candidates(selected_positions)) = false;
    end
    confirmation_indices = find(confirmation_mask);
    sample_errors = nan(sample_count, number_of_methods);
    true_energy = nan(sample_count, 1);
    fprintf('\nP=%d; held-out samples=%d; epochs 8/16-stage=%d/%d.\n', P, numel(confirmation_indices), eight_stage_data.best_epoch_saved, sixteen_stage_data.best_epoch_saved);

    for confirmation_index = 1:numel(confirmation_indices)
        sample_index = confirmation_indices(confirmation_index);
        Y = double(validation_data.Y_data(:, :, sample_index));
        G_true = double(validation_data.G_data(:, :, sample_index));
        S = double(validation_data.S_data(:, :, sample_index));
        B = double(validation_data.B_data(:, :, sample_index));
        sigma2 = double(validation_data.sigma2_data(sample_index));
        snr_db = double(validation_data.SNR_dB_data(sample_index));
        snr_index = find(SNR_grid == snr_db, 1);
        random_seed = 861000000+1000000*P+10000*snr_index+sample_index;
        right_inverse = S'/(S*S');
        G_estimate = (Y-abs(B))*right_inverse;
        sample_errors(sample_index, 1) = norm(G_estimate-G_true, 'fro')^2;
        [G_estimate, ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', Y, S, B, sigma2, p_H, 0, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
        sample_errors(sample_index, 2) = norm(G_estimate-G_true, 'fro')^2;
        [G_estimate, ~, ~] = V3_CLASSICAL_ESTIMATOR('GD', Y, S, B, sigma2, p_H, 0, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
        sample_errors(sample_index, 3) = norm(G_estimate-G_true, 'fro')^2;
        [G_estimate, ~, ~] = V3_CLASSICAL_ESTIMATOR('EM-GS', Y, S, B, sigma2, p_H, 0, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
        sample_errors(sample_index, 4) = norm(G_estimate-G_true, 'fro')^2;
        [G_estimate, ~, ~] = V3_CLASSICAL_ESTIMATOR('HS-GS', Y, S, B, sigma2, p_H, policy_data.frozen_multiplier(p_index, snr_index), maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
        sample_errors(sample_index, 5) = norm(G_estimate-G_true, 'fro')^2;
        G_estimate = local_forward_unfolded(eight_stage_data.best_parameters_numeric, 8, Y, S, B, sigma2, p_H, N, K, hankel_map, hankel_diagonal_counts);
        sample_errors(sample_index, 6) = norm(G_estimate-G_true, 'fro')^2;
        G_estimate = local_forward_unfolded(sixteen_stage_data.best_parameters_numeric, 16, Y, S, B, sigma2, p_H, N, K, hankel_map, hankel_diagonal_counts);
        sample_errors(sample_index, 7) = norm(G_estimate-G_true, 'fro')^2;
        true_energy(sample_index) = norm(G_true, 'fro')^2;
        if mod(confirmation_index, 100) == 0 || confirmation_index == numel(confirmation_indices)
            fprintf('P=%d validation samples completed: %d/%d.\n', P, confirmation_index, numel(confirmation_indices));
        end
    end

    if any(~isfinite(sample_errors(confirmation_indices, :)), 'all') || any(~isfinite(true_energy(confirmation_indices))) || any(true_energy(confirmation_indices) <= 0)
        error('P=%d validation comparison produced invalid estimates.', P);
    end
    all_errors{p_index} = sample_errors;
    all_true_energy{p_index} = true_energy;
    all_snr_values{p_index} = double(validation_data.SNR_dB_data);
    all_confirmation_masks{p_index} = confirmation_mask;

    for snr_index = 1:numel(SNR_grid)
        snr_db = SNR_grid(snr_index);
        snr_indices = confirmation_indices(double(validation_data.SNR_dB_data(confirmation_indices)) == snr_db);
        errors = sample_errors(snr_indices, :);
        energy = true_energy(snr_indices);
        nmse_db = 10*log10(sum(errors, 1)/sum(energy));
        bootstrap_gain = zeros(bootstrap_replicates, number_of_gains);
        rng(971000+1000*P+snr_index, 'twister');
        for draw_index = 1:bootstrap_replicates
            resampled_indices = randi(numel(snr_indices), numel(snr_indices), 1);
            resampled_errors = errors(resampled_indices, :);
            for gain_index = 1:number_of_gains
                baseline_index = gain_baseline_indices(gain_index);
                bootstrap_gain(draw_index, gain_index) = 10*log10(sum(resampled_errors(:, baseline_index))/max(sum(resampled_errors(:, 7)), realmin));
            end
        end
        bootstrap_gain = sort(bootstrap_gain, 1);
        ci_low = bootstrap_gain(max(1, floor(0.025*bootstrap_replicates)), :);
        ci_high = bootstrap_gain(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)), :);
        point_gain = 10*log10(sum(errors(:, gain_baseline_indices), 1)./sum(errors(:, 7)));
        row_index = row_index+1;
        result_rows(row_index, :) = [P, snr_db, numel(snr_indices), eight_stage_data.best_epoch_saved, sixteen_stage_data.best_epoch_saved, nmse_db, point_gain, ci_low, ci_high];
        fprintf('P=%d SNR=%g dB: 16-stage NMSE=%.3f dB; gain vs 8-stage=%.3f dB, CI=[%.3f, %.3f].\n', P, snr_db, nmse_db(7), point_gain(6), ci_low(6), ci_high(6));
    end
end

base_names = {'P','SNR_dB','ConfirmationSamples','EightStageSelectedEpoch','SixteenStageSelectedEpoch','LinearInit_NMSE_dB','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','HSGS_NMSE_dB','PI_DU_RACE_8Stage_NMSE_dB','PI_DU_RACE_16Stage_NMSE_dB'};
gain_names = cell(1, number_of_gains);
low_names = cell(1, number_of_gains);
high_names = cell(1, number_of_gains);
for gain_index = 1:number_of_gains
    gain_names{gain_index} = ['Gain_16Stage_vs_' gain_baseline_names{gain_index} '_dB'];
    low_names{gain_index} = ['CI_Low_Gain_16Stage_vs_' gain_baseline_names{gain_index} '_dB'];
    high_names{gain_index} = ['CI_High_Gain_16Stage_vs_' gain_baseline_names{gain_index} '_dB'];
end
variable_names = [base_names, gain_names, low_names, high_names];
result_table = array2table(result_rows, 'VariableNames', variable_names);
csv_file = fullfile(output_folder, 'V3_PI_DU_RACE_16stage_validation_confirmation.csv');
mat_file = fullfile(output_folder, 'V3_PI_DU_RACE_16stage_validation_confirmation.mat');
writetable(result_table, csv_file);
save(mat_file, 'result_table', 'all_errors', 'all_true_energy', 'all_snr_values', 'all_confirmation_masks', 'P_list', 'SNR_grid', 'method_names', 'gain_baseline_names', 'maximum_updates', 'convergence_tolerance', 'gd_initial_variance', 'bootstrap_replicates', 'validation_samples_per_SNR_for_checkpoint', '-v7.3');
disp(result_table);
fprintf('\nHeld-out validation comparison complete. No training or reserved test files were loaded.\n');
fprintf('HS-GS was previously tuned on validation; use independent test results for final claims.\n');
fprintf('Results are in: %s\n', output_folder);

function G_estimate = local_forward_unfolded(parameters, number_of_stages, Y, S, B, sigma2, p_H, N, K, hankel_map, hankel_diagonal_counts)
right_inverse = S'/(S*S');
G_estimate = (Y-abs(B))*right_inverse;
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
for stage_index = 1:number_of_stages
    G_previous = G_estimate;
    threshold_multiplier = 2/(1+exp(-parameters(1, stage_index)));
    threshold = threshold_multiplier*noise_scale;
    field = G_estimate*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    G_hankel = local_hankel_shrink(G_emgs, threshold, hankel_map, hankel_diagonal_counts, N, K, p_H);
    emgs_change = tanh(log(1+sqrt(sum(abs(G_emgs-G_previous).^2, 'all')/max(sum(abs(G_previous).^2, 'all'), eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_emgs).^2, 'all')/max(sum(abs(G_emgs).^2, 'all'), eps)))/3);
    gate_coefficients = reshape(parameters(2:13, stage_index), 4, 3);
    gate_logits = gate_coefficients(1, :).'*noise_feature+gate_coefficients(2, :).'*emgs_change+gate_coefficients(3, :).'*hankel_change+gate_coefficients(4, :).';
    gate_logits = gate_logits-max(gate_logits);
    gate_weights = exp(gate_logits)./sum(exp(gate_logits));
    G_estimate = gate_weights(1)*G_previous+gate_weights(2)*G_emgs+gate_weights(3)*G_hankel;
end
end

function G_output = local_hankel_shrink(G_input, threshold, hankel_map, hankel_diagonal_counts, N, K, p_H)
G_output = zeros(N, K);
for user_index = 1:K
    hankel_matrix = reshape(hankel_map*G_input(:, user_index), N-p_H, p_H+1);
    gram_matrix = hankel_matrix'*hankel_matrix;
    [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
    eigenvalues = real(sum(eigenvalue_matrix, 2));
    singular_values = sqrt(max(eigenvalues, 0));
    safe_singular_values = max(singular_values, 1e-12);
    shrink_factors = max(1-threshold./safe_singular_values, 0);
    scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
    hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
    G_output(:, user_index) = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
end
end
