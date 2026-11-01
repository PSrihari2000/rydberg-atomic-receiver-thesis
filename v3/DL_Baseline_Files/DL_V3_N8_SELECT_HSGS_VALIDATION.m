%% Select N=8 soft-HS-GS policy using validation data only
% This script never loads training or reserved test data.
% It selects a soft Hankel-shrinkage multiplier on half of validation data
% and accepts it only when the other half confirms positive paired gain.

clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
output_folder = fullfile(v3_folder, 'results', 'N8_CLASSICAL_COMPARISON_RSR25DB');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
policy_file = fullfile(output_folder, 'N8_frozen_HSGS_validation_policy.mat');
policy_csv = fullfile(output_folder, 'N8_HSGS_validation_selection.csv');
if isfile(policy_file) || isfile(policy_csv)
    error('Validation policy outputs already exist. Inspect them before selecting a new policy.');
end

N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
shrink_multiplier_list = [0 0.25 0.5 1 2];
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
bootstrap_replicates = 2000;
validation_samples_for_policy = 250;
validation_samples_for_confirmation = 250;

number_of_rows = numel(P_list)*numel(SNR_grid);
rows = zeros(number_of_rows, 9);
frozen_multiplier = zeros(numel(P_list), numel(SNR_grid));
validation_counts = zeros(numel(P_list), numel(SNR_grid));
row_index = 0;

fprintf('N=8 HS-GS policy selection: validation files only.\n');
fprintf('No training or reserved test data will be loaded.\n');
fprintf('Hankel split dimensions: %d by %d; path-count prior: 3 to 7.\n', N-p_H, p_H+1);

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    if ~isfile(validation_file)
        error('Missing N=8 validation data file: %s', validation_file);
    end
    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'AoA_deg_range', 'data_version');
    v3_n8_validate_dataset(validation_data, N, K, P, SNR_grid, RSR_dB);

    for snr_index = 1:numel(SNR_grid)
        snr_db = SNR_grid(snr_index);
        point_indices = find(abs(double(validation_data.SNR_dB_data)-snr_db) < 1e-6);
        if numel(point_indices) < validation_samples_for_policy+validation_samples_for_confirmation
            error('Not enough validation samples for P=%d, SNR=%g dB.', P, snr_db);
        end
        rng(631000+1000*P+snr_index, 'twister');
        point_indices = point_indices(randperm(numel(point_indices), validation_samples_for_policy+validation_samples_for_confirmation));
        policy_indices = point_indices(1:validation_samples_for_policy);
        confirmation_indices = point_indices(validation_samples_for_policy+1:end);

        policy_errors = zeros(validation_samples_for_policy, numel(shrink_multiplier_list));
        policy_energy = zeros(validation_samples_for_policy, 1);
        fprintf('Validation selection P=%d, SNR=%g dB.\n', P, snr_db);
        for sample_number = 1:validation_samples_for_policy
            sample_index = policy_indices(sample_number);
            Y = double(validation_data.Y_data(:, :, sample_index));
            G_true = double(validation_data.G_data(:, :, sample_index));
            S = double(validation_data.S_data(:, :, sample_index));
            B = double(validation_data.B_data(:, :, sample_index));
            sigma2 = double(validation_data.sigma2_data(sample_index));
            seed = 711000000+1000000*P+10000*snr_index+sample_index;
            [G_em, ~, ~] = V3_CLASSICAL_ESTIMATOR('EM-GS', Y, S, B, sigma2, p_H, 0, maximum_updates, convergence_tolerance, gd_initial_variance, seed);
            policy_errors(sample_number, 1) = norm(G_em-G_true, 'fro')^2;
            policy_energy(sample_number) = norm(G_true, 'fro')^2;
            for candidate_index = 2:numel(shrink_multiplier_list)
                [G_hs, ~, ~] = V3_CLASSICAL_ESTIMATOR('HS-GS', Y, S, B, sigma2, p_H, shrink_multiplier_list(candidate_index), maximum_updates, convergence_tolerance, gd_initial_variance, seed);
                policy_errors(sample_number, candidate_index) = norm(G_hs-G_true, 'fro')^2;
            end
        end

        candidate_total_error = sum(policy_errors, 1);
        [~, selected_candidate] = min(candidate_total_error);
        selected_multiplier = shrink_multiplier_list(selected_candidate);

        confirmation_errors = zeros(validation_samples_for_confirmation, 2);
        confirmation_energy = zeros(validation_samples_for_confirmation, 1);
        for sample_number = 1:validation_samples_for_confirmation
            sample_index = confirmation_indices(sample_number);
            Y = double(validation_data.Y_data(:, :, sample_index));
            G_true = double(validation_data.G_data(:, :, sample_index));
            S = double(validation_data.S_data(:, :, sample_index));
            B = double(validation_data.B_data(:, :, sample_index));
            sigma2 = double(validation_data.sigma2_data(sample_index));
            seed = 721000000+1000000*P+10000*snr_index+sample_index;
            [G_em, ~, ~] = V3_CLASSICAL_ESTIMATOR('EM-GS', Y, S, B, sigma2, p_H, 0, maximum_updates, convergence_tolerance, gd_initial_variance, seed);
            [G_hs, ~, ~] = V3_CLASSICAL_ESTIMATOR('HS-GS', Y, S, B, sigma2, p_H, selected_multiplier, maximum_updates, convergence_tolerance, gd_initial_variance, seed);
            confirmation_errors(sample_number, 1) = norm(G_em-G_true, 'fro')^2;
            confirmation_errors(sample_number, 2) = norm(G_hs-G_true, 'fro')^2;
            confirmation_energy(sample_number) = norm(G_true, 'fro')^2;
        end

        validation_gain_db = 10*log10(sum(confirmation_errors(:, 1))/max(sum(confirmation_errors(:, 2)), realmin));
        rng(741000+1000*P+snr_index, 'twister');
        bootstrap_gain = zeros(bootstrap_replicates, 1);
        for draw_index = 1:bootstrap_replicates
            sample_ids = randi(validation_samples_for_confirmation, validation_samples_for_confirmation, 1);
            bootstrap_gain(draw_index) = 10*log10(sum(confirmation_errors(sample_ids, 1))/max(sum(confirmation_errors(sample_ids, 2)), realmin));
        end
        bootstrap_gain = sort(bootstrap_gain);
        ci_low = bootstrap_gain(max(1, floor(0.025*bootstrap_replicates)));
        ci_high = bootstrap_gain(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)));
        if selected_multiplier > 0 && ci_low <= 0
            selected_multiplier = 0;
        end

        frozen_multiplier(p_index, snr_index) = selected_multiplier;
        validation_counts(p_index, snr_index) = validation_samples_for_policy+validation_samples_for_confirmation;
        row_index = row_index+1;
        rows(row_index, :) = [P, snr_db, validation_counts(p_index, snr_index), shrink_multiplier_list(selected_candidate), selected_multiplier, validation_gain_db, ci_low, ci_high, double(ci_low > 0)];
        fprintf('  candidate=%.3g, frozen=%.3g, held-out validation gain=%.4f dB, CI=[%.4f, %.4f] dB.\n', shrink_multiplier_list(selected_candidate), selected_multiplier, validation_gain_db, ci_low, ci_high);
    end
end

policy_table = array2table(rows, 'VariableNames', {'P','SNR_dB','ValidationSamples','SelectedCandidateMultiplier','FrozenMultiplier','ConfirmationGain_vs_EMGS_dB','ConfirmationCI_Low_dB','ConfirmationCI_High_dB','ConfirmedPositiveGain'});
configuration = struct('N', N, 'K', K, 'P_list', P_list, 'SNR_dB', SNR_grid, 'RSR_dB', RSR_dB, 'p_H', p_H, 'shrink_multiplier_list', shrink_multiplier_list, 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'gd_initial_variance', gd_initial_variance, 'bootstrap_replicates', bootstrap_replicates, 'validation_samples_for_policy', validation_samples_for_policy, 'validation_samples_for_confirmation', validation_samples_for_confirmation, 'data_version', 'V3_AoA_Uminus90to90');
writetable(policy_table, policy_csv);
save(policy_file, 'configuration', 'frozen_multiplier', 'validation_counts', 'policy_table', '-v7.3');
disp(policy_table);
fprintf('Frozen N=8 HS-GS validation policy saved to %s\n', policy_file);
fprintf('No reserved test files were loaded.\n');

function v3_n8_validate_dataset(data, expected_N, expected_K, expected_P, expected_SNR, expected_RSR)
if data.N ~= expected_N || data.K ~= expected_K || data.P ~= expected_P || data.RSR_dB ~= expected_RSR || ~isequal(double(data.SNR_dB), expected_SNR) || ~isequal(double(data.AoA_deg_range), [-90 90]) || ~strcmp(data.data_version, 'V3_AoA_Uminus90to90')
    error('Dataset metadata does not match the required N=8 V3 AoA scenario.');
end
if size(data.Y_data, 1) ~= expected_N || size(data.Y_data, 2) ~= expected_P || size(data.G_data, 1) ~= expected_N || size(data.G_data, 2) ~= expected_K || size(data.S_data, 1) ~= expected_K || size(data.S_data, 2) ~= expected_P || size(data.B_data, 1) ~= expected_N || size(data.B_data, 2) ~= expected_P
    error('Dataset arrays have unexpected N, K, or P dimensions.');
end
if any(~isfinite(data.SNR_dB_data)) || any(~isfinite(data.sigma2_data)) || any(data.sigma2_data <= 0)
    error('Validation file contains invalid SNR or noise variance values.');
end
end
