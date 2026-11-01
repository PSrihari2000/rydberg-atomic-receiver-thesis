%% Independent validation confirmation for the V3 GS-initialized 16-stage model
% Compares raw GS, the earlier full-data model, and the GS-init fine-tuned model.
% Each pilot/SNR confirmation subset excludes both 64-sample model-selection subsets.
% All methods use the same validation observations and per-sample relative NMSE.
% No training data or reserved test data are loaded.

clear;
clc;
close all;
rng(20261102, 'twister');

%% -------------------- FIXED V3 SCENARIO --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
number_of_stages = 16;
parameters_per_stage = 13;
selection_samples_per_SNR = 64;
confirmation_samples_per_SNR = 64;
bootstrap_replicates = 2000;

script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
previous_model_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_FULLDATA_ADAPTIVE_GATE_RSR25DB');
new_model_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_GS_INIT_RSR25DB');
classical_folder = fullfile(v3_folder, 'results', 'N8_CLASSICAL_COMPARISON_RSR25DB');
policy_file = fullfile(classical_folder, 'N8_frozen_HSGS_validation_policy.mat');
output_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_GS_INIT_CONFIRMATION_VALIDATION');

if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it before running this confirmation again.');
end
if ~isfolder(scenario_folder) || ~isfolder(previous_model_folder) || ~isfolder(new_model_folder) || ~isfile(policy_file)
    error('A required V3 validation data, model, or frozen GS settings file is missing.');
end
if ~isfile(fullfile(script_folder, 'V3_CLASSICAL_ESTIMATOR.m'))
    error('V3_CLASSICAL_ESTIMATOR.m must be in the scripts folder.');
end
addpath(script_folder);

policy_data = load(policy_file, 'configuration');
configuration = policy_data.configuration;
if configuration.N ~= N || configuration.K ~= K || configuration.RSR_dB ~= RSR_dB || ~isequal(configuration.P_list, P_list) || ~isequal(configuration.SNR_dB, SNR_grid) || ~strcmp(configuration.data_version, 'V3_AoA_Uminus90to90') || configuration.p_H ~= p_H
    error('Frozen GS settings do not match this V3 scenario.');
end
maximum_gs_updates = configuration.maximum_updates;
convergence_tolerance = configuration.convergence_tolerance;
gd_initial_variance = configuration.gd_initial_variance;
mkdir(output_folder);

fprintf('V3 GS-init 16-stage confirmation: validation data only.\n');
fprintf('The previous checkpoint-selection and GS-init fine-tuning subsets are excluded.\n');
fprintf('%d confirmation samples per SNR; %d samples per P value.\n', confirmation_samples_per_SNR, confirmation_samples_per_SNR*numel(SNR_grid));
fprintf('Raw GS uses the frozen validation settings: max updates=%d, tolerance=%g.\n', maximum_gs_updates, convergence_tolerance);
fprintf('Reserved test files will not be loaded.\n');
fprintf('Output folder: %s\n', output_folder);

number_of_rows = numel(P_list)*numel(SNR_grid);
result_values = zeros(number_of_rows, 15);
row_index = 0;

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    previous_checkpoint_file = fullfile(previous_model_folder, sprintf('checkpoint_P%d.mat', P));
    new_model_file = fullfile(new_model_folder, sprintf('selected_model_P%d.mat', P));

    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'RSR_dB', 'AoA_deg_range', 'data_version');
    previous_model = load(previous_checkpoint_file, 'best_parameters_saved', 'best_epoch', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage');
    new_model = load(new_model_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'initialization_method');

    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ~isequal(double(validation_data.AoA_deg_range), [-90 90]) || ~strcmp(validation_data.data_version, 'V3_AoA_Uminus90to90') || validation_data.number_of_samples ~= 4000
        error('Validation data metadata mismatch for P=%d.', P);
    end
    if previous_model.N ~= N || previous_model.K ~= K || previous_model.P ~= P || previous_model.RSR_dB ~= RSR_dB || previous_model.p_H ~= p_H || previous_model.number_of_stages ~= number_of_stages || previous_model.parameters_per_stage ~= parameters_per_stage || ~isequal(size(previous_model.best_parameters_saved), [parameters_per_stage number_of_stages])
        error('Previous model metadata mismatch for P=%d.', P);
    end
    if new_model.N ~= N || new_model.K ~= K || new_model.P ~= P || new_model.RSR_dB ~= RSR_dB || new_model.p_H ~= p_H || new_model.number_of_stages ~= number_of_stages || new_model.parameters_per_stage ~= parameters_per_stage || ~isequal(size(new_model.best_parameters_numeric), [parameters_per_stage number_of_stages]) || ~strcmp(new_model.initialization_method, 'GS')
        error('GS-init fine-tuned model metadata mismatch for P=%d.', P);
    end

    %% Make the two model-selection subsets and confirmation subset disjoint
    confirmation_indices = [];
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        candidates = find(double(validation_data.SNR_dB_data) == current_SNR_dB);
        if numel(candidates) ~= 500
            error('Expected 500 validation observations for P=%d, SNR=%g dB.', P, current_SNR_dB);
        end

        previous_selection_positions = round(linspace(1, numel(candidates), selection_samples_per_SNR));
        remaining_after_previous = candidates;
        remaining_after_previous(previous_selection_positions) = [];
        new_selection_positions = round(linspace(1, numel(remaining_after_previous), selection_samples_per_SNR));
        remaining_for_confirmation = remaining_after_previous;
        remaining_for_confirmation(new_selection_positions) = [];

        if numel(remaining_for_confirmation) < confirmation_samples_per_SNR
            error('Not enough untouched confirmation observations for P=%d, SNR=%g dB.', P, current_SNR_dB);
        end
        confirmation_positions = round(linspace(1, numel(remaining_for_confirmation), confirmation_samples_per_SNR));
        confirmation_indices = [confirmation_indices; remaining_for_confirmation(confirmation_positions(:))];
    end

    previous_linear_error = zeros(numel(confirmation_indices), 1);
    previous_gs_init_error = zeros(numel(confirmation_indices), 1);
    new_gs_init_error = zeros(numel(confirmation_indices), 1);
    gs_error = zeros(numel(confirmation_indices), 1);
    true_energy = zeros(numel(confirmation_indices), 1);

    fprintf('\nP=%d; untouched confirmation observations=%d; previous epoch=%d; new epoch=%d.\n', P, numel(confirmation_indices), previous_model.best_epoch, new_model.best_epoch_saved);
    for confirmation_position = 1:numel(confirmation_indices)
        sample_index = confirmation_indices(confirmation_position);
        Y = double(validation_data.Y_data(:, :, sample_index));
        G_true = double(validation_data.G_data(:, :, sample_index));
        S = double(validation_data.S_data(:, :, sample_index));
        B = double(validation_data.B_data(:, :, sample_index));
        sigma2 = double(validation_data.sigma2_data(sample_index));

        [G_GS, ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', Y, S, B, sigma2, p_H, 0, maximum_gs_updates, convergence_tolerance, gd_initial_variance, 0);
        G_linear_initial = (Y-abs(B))*(S'/(S*S'));
        G_previous_linear = local_forward(previous_model.best_parameters_saved, Y, S, B, sigma2, p_H, N, K, G_linear_initial);
        G_previous_gs = local_forward(previous_model.best_parameters_saved, Y, S, B, sigma2, p_H, N, K, G_GS);
        G_new_gs = local_forward(new_model.best_parameters_numeric, Y, S, B, sigma2, p_H, N, K, G_GS);

        previous_linear_error(confirmation_position) = sum(abs(G_previous_linear-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
        previous_gs_init_error(confirmation_position) = sum(abs(G_previous_gs-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
        new_gs_init_error(confirmation_position) = sum(abs(G_new_gs-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
        gs_error(confirmation_position) = sum(abs(G_GS-G_true).^2, 'all')/max(sum(abs(G_true).^2, 'all'), eps);
        true_energy(confirmation_position) = sum(abs(G_true).^2, 'all');

        if mod(confirmation_position, 128) == 0 || confirmation_position == numel(confirmation_indices)
            fprintf('P=%d completed %d/%d confirmation samples.\n', P, confirmation_position, numel(confirmation_indices));
        end
    end

    if any(~isfinite(previous_linear_error)) || any(~isfinite(previous_gs_init_error)) || any(~isfinite(new_gs_init_error)) || any(~isfinite(gs_error)) || any(~isfinite(true_energy)) || any(true_energy <= 0)
        error('Nonfinite estimate or invalid true-channel energy for P=%d.', P);
    end

    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        snr_positions = find(double(validation_data.SNR_dB_data(confirmation_indices)) == current_SNR_dB);
        gs_nmse = gs_error(snr_positions);
        previous_linear_nmse = previous_linear_error(snr_positions);
        previous_gs_nmse = previous_gs_init_error(snr_positions);
        new_gs_nmse = new_gs_init_error(snr_positions);

        gs_nmse_dB = 10*log10(mean(gs_nmse));
        previous_linear_nmse_dB = 10*log10(mean(previous_linear_nmse));
        previous_gs_nmse_dB = 10*log10(mean(previous_gs_nmse));
        new_gs_nmse_dB = 10*log10(mean(new_gs_nmse));
        gain_vs_gs = 10*log10(mean(gs_nmse)/mean(new_gs_nmse));
        gain_vs_previous_gs = 10*log10(mean(previous_gs_nmse)/mean(new_gs_nmse));

        bootstrap_gain_vs_gs = zeros(bootstrap_replicates, 1);
        bootstrap_gain_vs_previous_gs = zeros(bootstrap_replicates, 1);
        rng(20261102+100*P+snr_index, 'twister');
        for draw_index = 1:bootstrap_replicates
            resampled_positions = randi(numel(snr_positions), numel(snr_positions), 1);
            bootstrap_gain_vs_gs(draw_index) = 10*log10(mean(gs_nmse(resampled_positions))/mean(new_gs_nmse(resampled_positions)));
            bootstrap_gain_vs_previous_gs(draw_index) = 10*log10(mean(previous_gs_nmse(resampled_positions))/mean(new_gs_nmse(resampled_positions)));
        end
        bootstrap_gain_vs_gs = sort(bootstrap_gain_vs_gs);
        bootstrap_gain_vs_previous_gs = sort(bootstrap_gain_vs_previous_gs);
        ci_gs_low = bootstrap_gain_vs_gs(max(1, floor(0.025*bootstrap_replicates)));
        ci_gs_high = bootstrap_gain_vs_gs(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)));
        ci_previous_low = bootstrap_gain_vs_previous_gs(max(1, floor(0.025*bootstrap_replicates)));
        ci_previous_high = bootstrap_gain_vs_previous_gs(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)));

        row_index = row_index+1;
        result_values(row_index, :) = [P, current_SNR_dB, numel(snr_positions), previous_model.best_epoch, new_model.best_epoch_saved, gs_nmse_dB, previous_linear_nmse_dB, previous_gs_nmse_dB, new_gs_nmse_dB, gain_vs_gs, ci_gs_low, ci_gs_high, gain_vs_previous_gs, ci_previous_low, ci_previous_high];
        fprintf('P=%d SNR=%g dB: GS=%+.3f dB; previous GS-init DL=%+.3f dB; new GS-init DL=%+.3f dB; gain vs GS=%+.3f dB.\n', P, current_SNR_dB, gs_nmse_dB, previous_gs_nmse_dB, new_gs_nmse_dB, gain_vs_gs);
    end
end

result_table = array2table(result_values, 'VariableNames', {'P','SNR_dB','ConfirmationSamples','PreviousModelEpoch','NewModelEpoch','GS_MeanSampleNMSE_dB','PreviousLinearInitDL_MeanSampleNMSE_dB','PreviousGSInitDL_MeanSampleNMSE_dB','NewGSInitDL_MeanSampleNMSE_dB','NewDL_Gain_vs_GS_dB','GainVsGS_CI95_Low_dB','GainVsGS_CI95_High_dB','NewDL_Gain_vs_PreviousGSInitDL_dB','GainVsPreviousGSInitDL_CI95_Low_dB','GainVsPreviousGSInitDL_CI95_High_dB'});
csv_file = fullfile(output_folder, 'N8_16stage_GS_init_confirmation_validation.csv');
mat_file = fullfile(output_folder, 'N8_16stage_GS_init_confirmation_validation.mat');
writetable(result_table, csv_file);
save(mat_file, 'result_table', 'P_list', 'SNR_grid', 'confirmation_samples_per_SNR', 'selection_samples_per_SNR', 'maximum_gs_updates', 'convergence_tolerance', 'bootstrap_replicates');
disp(result_table);
fprintf('\nConfirmation validation finished. No training or test files were loaded.\n');
fprintf('Positive gain favors the new GS-initialized model. A CI fully above zero supports a paired pointwise gain.\n');
fprintf('CSV: %s\nMAT: %s\n', csv_file, mat_file);

function G_estimate = local_forward(parameters, Y, S, B, sigma2, p_H, N, K, initial_estimate)
right_inverse = S'/(S*S');
G_estimate = initial_estimate;
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
hankel_map = local_hankel_map(N, p_H);
hankel_diagonal_counts = sum(hankel_map, 1).';
for stage_index = 1:size(parameters, 2)
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
