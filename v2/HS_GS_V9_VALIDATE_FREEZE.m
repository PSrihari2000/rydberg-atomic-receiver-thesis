%% HS_GS_V9_VALIDATE_FREEZE.m
% Use all 500 validation samples per P/SNR point to freeze a robust HS-GS
% setting before generating or scoring a fresh final test dataset.
clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end
checkpoint_file = fullfile(result_folder, 'HS_GS_V9_validation_checkpoint.mat');
freeze_file = fullfile(result_folder, 'HS_GS_V9_frozen_rule.mat');
if exist(freeze_file, 'file')
    error('The V9 rule is already frozen at %s. Preserve it for the fresh test.', freeze_file);
end

P_list = [10 30];
SNR_dB = -5:5:30;
N = 8;
K = 3;
RSR_dB = 25;
d_over_lambda = 0.5;
p_H_list = [2 3];
sweep_list = [0 1 2 4];
setting_p_H = repelem(p_H_list, length(sweep_list));
setting_sweeps = repmat(sweep_list, 1, length(p_H_list));
number_of_settings = length(setting_p_H);
selected_updates = [100 100 100 100 100 100 100 1000; 100 100 100 100 500 100 100 1000];
rank_selection_updates = 50;
training_fraction = 0.70;
rank_seed_base = 910000;
samples_per_snr = 500;
number_of_folds = 5;
minimum_gain_dB = 0.1;
minimum_improving_folds = 4;
fresh_seed_base = 2026102404;

requested_workers = 6;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', requested_workers);
elseif pool.NumWorkers ~= requested_workers
    delete(pool);
    parpool('local', requested_workers);
end
pool = gcp('nocreate');
fprintf('HS-GS V9 validation started with %d workers.\n', pool.NumWorkers);

validation_error_power = nan(length(P_list), length(SNR_dB), number_of_settings, samples_per_snr);
validation_channel_power = nan(length(P_list), length(SNR_dB), samples_per_snr);
completed_settings = false(length(P_list), length(SNR_dB), number_of_settings);
if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    matching_settings = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && ...
        isequal(checkpoint.setting_p_H, setting_p_H) && ...
        isequal(checkpoint.setting_sweeps, setting_sweeps) && ...
        isequal(checkpoint.selected_updates, selected_updates) && ...
        isequal(checkpoint.rank_selection_updates, rank_selection_updates) && ...
        isequal(checkpoint.training_fraction, training_fraction) && ...
        isequal(checkpoint.samples_per_snr, samples_per_snr);
    if ~matching_settings
        error('V9 validation checkpoint has different settings. Inspect it before resuming.');
    end
    validation_error_power = checkpoint.validation_error_power;
    validation_channel_power = checkpoint.validation_channel_power;
    completed_settings = checkpoint.completed_settings;
    fprintf('Resumed %d/%d validation settings.\n', sum(completed_settings, 'all'), numel(completed_settings));
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    source = load(fullfile(dataset_folder, sprintf('validation_P%d.mat', P)));
    for SNR_index = 1:length(SNR_dB)
        selected_samples = find(source.SNR_dB_data(:) == SNR_dB(SNR_index));
        if length(selected_samples) ~= samples_per_snr
            error('Expected %d validation samples for P=%d, SNR=%d dB; found %d.', ...
                samples_per_snr, P, SNR_dB(SNR_index), length(selected_samples));
        end
        update_count = selected_updates(P_index, SNR_index);
        for setting_number = 1:number_of_settings
            if completed_settings(P_index, SNR_index, setting_number)
                continue;
            end
            p_H = setting_p_H(setting_number);
            sweep_count = setting_sweeps(setting_number);
            setting_timer = tic;
            fprintf('Validating P=%d, SNR=%3d dB, p_H=%d, sweeps=%d\n', ...
                P, SNR_dB(SNR_index), p_H, sweep_count);
            if sweep_count == 0 && p_H ~= p_H_list(1)
                validation_error_power(P_index, SNR_index, setting_number, :) = ...
                    validation_error_power(P_index, SNR_index, 1, :);
            else
                for block_start = 1:100:samples_per_snr
                    block_positions = block_start:min(block_start + 99, samples_per_snr);
                    existing_errors = squeeze(validation_error_power(P_index, SNR_index, setting_number, block_positions));
                    pending_positions = block_positions(isnan(existing_errors(:).'));
                    if isempty(pending_positions)
                        continue;
                    end
                    block_error = zeros(length(pending_positions), 1);
                    block_energy = zeros(length(pending_positions), 1);
                    parfor local_number = 1:length(pending_positions)
                        sample_position = pending_positions(local_number);
                        sample_index = selected_samples(sample_position);
                        Y = double(source.Y_data(:, :, sample_index));
                        G = double(source.G_data(:, :, sample_index));
                        S = double(source.S_data(:, :, sample_index));
                        B = double(source.B_data(:, :, sample_index));
                        sigma2 = double(source.sigma2_data(sample_index));
                        G_hat = hs_gs_v9_sample(Y, S, B, sigma2, P, sample_index, ...
                            p_H, sweep_count, update_count, rank_selection_updates, ...
                            training_fraction, rank_seed_base);
                        block_error(local_number) = norm(G_hat - G, 'fro')^2;
                        block_energy(local_number) = norm(G, 'fro')^2;
                    end
                    validation_error_power(P_index, SNR_index, setting_number, pending_positions) = ...
                        reshape(block_error, 1, 1, 1, []);
                    validation_channel_power(P_index, SNR_index, pending_positions) = ...
                        reshape(block_energy, 1, 1, []);
                    save(checkpoint_file, 'validation_error_power', 'validation_channel_power', ...
                        'completed_settings', 'P_list', 'SNR_dB', 'setting_p_H', ...
                        'setting_sweeps', 'selected_updates', 'rank_selection_updates', ...
                        'training_fraction', 'samples_per_snr', 'N', 'K', 'RSR_dB');
                    fprintf('P=%d, SNR=%3d dB, p_H=%d, sweeps=%d: %d/%d validation samples.\n', ...
                        P, SNR_dB(SNR_index), p_H, sweep_count, ...
                        sum(isfinite(validation_error_power(P_index, SNR_index, setting_number, :)), 'all'), samples_per_snr);
                end
            end
            setting_errors = squeeze(validation_error_power(P_index, SNR_index, setting_number, :));
            channel_energy = squeeze(validation_channel_power(P_index, SNR_index, :));
            if ~all(isfinite(setting_errors)) || ~all(isfinite(channel_energy))
                error('Nonfinite validation result for P=%d, SNR=%d dB, setting=%d.', ...
                    P, SNR_dB(SNR_index), setting_number);
            end
            completed_settings(P_index, SNR_index, setting_number) = true;
            validation_nmse_dB = 10 * log10(sum(setting_errors) / sum(channel_energy));
            fprintf('Validation NMSE %.3f dB; elapsed %.1f s.\n', validation_nmse_dB, toc(setting_timer));
            save(checkpoint_file, 'validation_error_power', 'validation_channel_power', ...
                'completed_settings', 'P_list', 'SNR_dB', 'setting_p_H', ...
                'setting_sweeps', 'selected_updates', 'rank_selection_updates', ...
                'training_fraction', 'samples_per_snr', 'N', 'K', 'RSR_dB');
        end
    end
end

fold_size = samples_per_snr / number_of_folds;
if fold_size ~= floor(fold_size)
    error('The validation sample count must divide evenly into folds.');
end
overall_gain_dB = nan(length(P_list), length(SNR_dB), number_of_settings);
fold_gain_dB = nan(length(P_list), length(SNR_dB), number_of_settings, number_of_folds);
selected_p_H = zeros(length(P_list), length(SNR_dB));
selected_sweeps = zeros(length(P_list), length(SNR_dB));
selected_setting_index = zeros(length(P_list), length(SNR_dB));
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        baseline_error = squeeze(validation_error_power(P_index, SNR_index, 1, :));
        eligible = false(number_of_settings, 1);
        for setting_number = 1:number_of_settings
            candidate_error = squeeze(validation_error_power(P_index, SNR_index, setting_number, :));
            overall_gain_dB(P_index, SNR_index, setting_number) = ...
                10 * log10(sum(baseline_error) / sum(candidate_error));
            for fold_index = 1:number_of_folds
                indices = (fold_index - 1) * fold_size + (1:fold_size);
                fold_gain_dB(P_index, SNR_index, setting_number, fold_index) = ...
                    10 * log10(sum(baseline_error(indices)) / sum(candidate_error(indices)));
            end
            fold_gains = squeeze(fold_gain_dB(P_index, SNR_index, setting_number, :));
            eligible(setting_number) = setting_sweeps(setting_number) > 0 && ...
                overall_gain_dB(P_index, SNR_index, setting_number) >= minimum_gain_dB && ...
                sum(fold_gains > 0) >= minimum_improving_folds;
        end
        best_setting = 1;
        if any(eligible)
            eligible_indices = find(eligible);
            [~, best_eligible] = max(overall_gain_dB(P_index, SNR_index, eligible_indices));
            best_setting = eligible_indices(best_eligible);
        end
        selected_setting_index(P_index, SNR_index) = best_setting;
        selected_p_H(P_index, SNR_index) = setting_p_H(best_setting);
        selected_sweeps(P_index, SNR_index) = setting_sweeps(best_setting);
        fprintf('Frozen P=%d, SNR=%3d dB: p_H=%d, sweeps=%d, validation gain=%.3f dB\n', ...
            P_list(P_index), SNR_dB(SNR_index), setting_p_H(best_setting), ...
            setting_sweeps(best_setting), overall_gain_dB(P_index, SNR_index, best_setting));
    end
end
rule_description = 'Use a nonzero Hankel setting only when its aggregate validation gain over zero sweeps is at least 0.1 dB and at least four of five disjoint validation folds improve. Otherwise use zero sweeps.';
save(freeze_file, 'selected_p_H', 'selected_sweeps', 'selected_setting_index', ...
    'selected_updates', 'overall_gain_dB', 'fold_gain_dB', ...
    'validation_error_power', 'validation_channel_power', 'setting_p_H', ...
    'setting_sweeps', 'minimum_gain_dB', 'minimum_improving_folds', ...
    'number_of_folds', 'samples_per_snr', 'rank_selection_updates', ...
    'training_fraction', 'rank_seed_base', 'fresh_seed_base', ...
    'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', 'd_over_lambda', ...
    'rule_description');
fprintf('Frozen rule saved to %s\n', freeze_file);
