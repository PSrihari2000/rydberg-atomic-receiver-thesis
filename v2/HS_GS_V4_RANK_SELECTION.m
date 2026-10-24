%% HS_GS_V4_RANK_SELECTION.m
% HS-GS rank-selection experiment with a larger held-out pilot fraction.
% Validation data selects the update/sweep setting. Independent test data
% is used only after that selection. The serial ablation file is unchanged.

clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
source_dataset_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

P_list = [10 30];
SNR_dB = -5:5:30;
N = 8;
K = 3;
RSR_dB = 25;
p_H = 3;
rank_list = 1:min(N - p_H, p_H + 1);
% V3 keeps the Hankel split fixed initially and expands the convergence grid.
% This isolates whether the v2 high-SNR gap is caused by insufficient updates
% or Cadzow sweeps before changing the Hankel geometry itself.
update_list = [100 500 1000 2000];
sweep_list = [0 1 2 4];
rank_selection_updates = 50;
training_fraction = 0.50;
random_seed_base = 910000;

% Quick verification limits. Set quick_test_mode = false for final results.
quick_test_mode = false;
if quick_test_mode
    maximum_validation_samples_per_snr = 20;
    maximum_test_samples_per_snr = 20;
else
    maximum_validation_samples_per_snr = 100;
    maximum_test_samples_per_snr = inf;
end
if quick_test_mode
    run_tag = 'quick';
else
    run_tag = 'rank_selection';
end
checkpoint_file = fullfile(result_folder, ['HS_GS_V4_' run_tag '_checkpoint.mat']);

requested_workers = 2;
existing_pool = gcp('nocreate');
if isempty(existing_pool)
    parpool('local', requested_workers);
elseif existing_pool.NumWorkers ~= requested_workers
    delete(existing_pool);
    parpool('local', requested_workers);
end
pool = gcp('nocreate');
fprintf('Parallel HS-GS v4 rank-selection experiment started with %d workers. Quick test mode = %d.\n', pool.NumWorkers, quick_test_mode);

number_of_settings = length(update_list) * length(sweep_list);
setting_updates = repelem(update_list, length(sweep_list));
setting_sweeps = repmat(sweep_list, 1, length(update_list));
selected_updates = zeros(length(P_list), length(SNR_dB));
selected_sweeps = zeros(length(P_list), length(SNR_dB));
validation_setting_error = zeros(length(P_list), length(SNR_dB), number_of_settings);
test_NMSE = zeros(length(P_list), length(SNR_dB));
test_NMSE_dB = zeros(length(P_list), length(SNR_dB));
test_mean_rank = zeros(length(P_list), length(SNR_dB));
completed_points = false(length(P_list), length(SNR_dB));

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    if isequal(checkpoint.P_list, P_list) && isequal(checkpoint.SNR_dB, SNR_dB)
        selected_updates = checkpoint.selected_updates;
        selected_sweeps = checkpoint.selected_sweeps;
        validation_setting_error = checkpoint.validation_setting_error;
        test_NMSE = checkpoint.test_NMSE;
        test_NMSE_dB = checkpoint.test_NMSE_dB;
        test_mean_rank = checkpoint.test_mean_rank;
        completed_points = checkpoint.completed_points;
        fprintf('Loaded checkpoint: %d of %d P/SNR points already complete.\n', sum(completed_points, 'all'), numel(completed_points));
    end
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    validation_source = load(fullfile(source_dataset_folder, sprintf('validation_P%d.mat', P)));
    test_source = load(fullfile(source_dataset_folder, sprintf('test_P%d.mat', P)));

    for SNR_index = 1:length(SNR_dB)
        if completed_points(P_index, SNR_index)
            fprintf('Skipping completed point: P = %d, SNR = %3d dB\n', P, SNR_dB(SNR_index));
            continue;
        end
        snr_timer = tic;
        validation_samples = find(validation_source.SNR_dB_data(:) == SNR_dB(SNR_index));
        test_samples = find(test_source.SNR_dB_data(:) == SNR_dB(SNR_index));
        validation_samples = limit_samples(validation_samples, maximum_validation_samples_per_snr);
        test_samples = limit_samples(test_samples, maximum_test_samples_per_snr);
        validation_error_for_setting = zeros(number_of_settings, 1);

        fprintf('Starting validation: P = %d, SNR = %3d dB\n', P, SNR_dB(SNR_index));
        parfor setting_number = 1:number_of_settings
            update_count = setting_updates(setting_number);
            sweep_count = setting_sweeps(setting_number);
            prediction_error_sum = 0;
            for sample_number = 1:length(validation_samples)
                sample_index = validation_samples(sample_number);
                Y = double(validation_source.Y_data(:, :, sample_index));
                S = double(validation_source.S_data(:, :, sample_index));
                B = double(validation_source.B_data(:, :, sample_index));
                sigma2 = double(validation_source.sigma2_data(sample_index));
                [training_pilots, held_out_pilots] = split_pilots(P, training_fraction, random_seed_base, sample_index);
                candidate_error = zeros(length(rank_list), 1);
                for rank_index = 1:length(rank_list)
                    candidate_rank = rank_list(rank_index);
                    G_candidate = run_hs_gs(Y(:, training_pilots), S(:, training_pilots), B(:, training_pilots), sigma2, candidate_rank, rank_selection_updates, sweep_count, p_H);
                    predicted_magnitude = abs(G_candidate * S(:, held_out_pilots) + B(:, held_out_pilots));
                    candidate_error(rank_index) = norm(Y(:, held_out_pilots) - predicted_magnitude, 'fro')^2;
                end
                [~, best_rank_index] = min(candidate_error);
                selected_rank = rank_list(best_rank_index);
                G_candidate = run_hs_gs(Y(:, training_pilots), S(:, training_pilots), B(:, training_pilots), sigma2, selected_rank, update_count, sweep_count, p_H);
                predicted_magnitude = abs(G_candidate * S(:, held_out_pilots) + B(:, held_out_pilots));
                prediction_error_sum = prediction_error_sum + norm(Y(:, held_out_pilots) - predicted_magnitude, 'fro')^2;
            end
            validation_error_for_setting(setting_number) = prediction_error_sum / length(validation_samples);
        end
        validation_setting_error(P_index, SNR_index, :) = validation_error_for_setting;

        [~, best_setting_number] = min(validation_error_for_setting);
        selected_updates(P_index, SNR_index) = setting_updates(best_setting_number);
        selected_sweeps(P_index, SNR_index) = setting_sweeps(best_setting_number);
        fprintf('Selected: updates = %d, sweeps = %d\n', selected_updates(P_index, SNR_index), selected_sweeps(P_index, SNR_index));

        sample_squared_error = zeros(length(test_samples), 1);
        sample_channel_energy = zeros(length(test_samples), 1);
        sample_selected_rank = zeros(length(test_samples), 1);
        parfor sample_number = 1:length(test_samples)
            sample_index = test_samples(sample_number);
            Y = double(test_source.Y_data(:, :, sample_index));
            G = double(test_source.G_data(:, :, sample_index));
            S = double(test_source.S_data(:, :, sample_index));
            B = double(test_source.B_data(:, :, sample_index));
            sigma2 = double(test_source.sigma2_data(sample_index));
            [training_pilots, held_out_pilots] = split_pilots(P, training_fraction, random_seed_base, sample_index);
            candidate_error = zeros(length(rank_list), 1);
            for rank_index = 1:length(rank_list)
                candidate_rank = rank_list(rank_index);
                G_candidate = run_hs_gs(Y(:, training_pilots), S(:, training_pilots), B(:, training_pilots), sigma2, candidate_rank, rank_selection_updates, selected_sweeps(P_index, SNR_index), p_H);
                predicted_magnitude = abs(G_candidate * S(:, held_out_pilots) + B(:, held_out_pilots));
                candidate_error(rank_index) = norm(Y(:, held_out_pilots) - predicted_magnitude, 'fro')^2;
            end
            [~, best_rank_index] = min(candidate_error);
            selected_rank = rank_list(best_rank_index);
            G_hat = run_hs_gs(Y, S, B, sigma2, selected_rank, selected_updates(P_index, SNR_index), selected_sweeps(P_index, SNR_index), p_H);
            sample_squared_error(sample_number) = norm(G_hat - G, 'fro')^2;
            sample_channel_energy(sample_number) = norm(G, 'fro')^2;
            sample_selected_rank(sample_number) = selected_rank;
        end

        test_NMSE(P_index, SNR_index) = sum(sample_squared_error) / max(sum(sample_channel_energy), eps);
        test_NMSE_dB(P_index, SNR_index) = 10 * log10(test_NMSE(P_index, SNR_index));
        test_mean_rank(P_index, SNR_index) = mean(sample_selected_rank);
        fprintf('Finished: P = %d, SNR = %3d dB, NMSE = %8.3f dB\n', P, SNR_dB(SNR_index), test_NMSE_dB(P_index, SNR_index));
        fprintf('Elapsed for this P/SNR point: %.1f seconds.\n', toc(snr_timer));
        completed_points(P_index, SNR_index) = true;
        save(checkpoint_file, 'selected_updates', 'selected_sweeps', 'validation_setting_error', 'test_NMSE', 'test_NMSE_dB', 'test_mean_rank', 'completed_points', 'update_list', 'sweep_list', 'rank_list', 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', 'p_H', 'rank_selection_updates', 'training_fraction', 'maximum_validation_samples_per_snr', 'maximum_test_samples_per_snr');
        drawnow;
    end
end

save(fullfile(result_folder, 'HS_GS_V4_rank_selection_results.mat'), 'selected_updates', 'selected_sweeps', 'validation_setting_error', 'test_NMSE', 'test_NMSE_dB', 'test_mean_rank', 'update_list', 'sweep_list', 'rank_list', 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', 'p_H', 'rank_selection_updates', 'training_fraction', 'maximum_validation_samples_per_snr', 'maximum_test_samples_per_snr');

function samples = limit_samples(samples, maximum_samples)
if isfinite(maximum_samples)
    samples = samples(1:min(length(samples), maximum_samples));
end
end

function [training_pilots, held_out_pilots] = split_pilots(P, training_fraction, seed_base, sample_index)
rng(seed_base + 1000 * P + sample_index, 'twister');
pilot_order = randperm(P);
training_count = round(training_fraction * P);
training_pilots = pilot_order(1:training_count);
held_out_pilots = pilot_order(training_count + 1:P);
end

function G_hat = run_hs_gs(Y, S, B, sigma2, rank_value, em_updates, cadzow_sweeps, p_H)
N = size(Y, 1);
K = size(S, 1);
G_hat = zeros(N, K);
for cell_index = 1:N
    y_i = Y(cell_index, :).';
    b_i = B(cell_index, :).';
    augmented_matrix = [S.', b_i];
    weighted_matrix = augmented_matrix' * diag(y_i) * augmented_matrix;
    [eigenvectors, eigenvalues] = eig(weighted_matrix);
    [~, principal_index] = max(real(diag(eigenvalues)));
    principal_vector = eigenvectors(:, principal_index);
    principal_vector = principal_vector / max(norm(principal_vector), eps);
    scale_factor = (abs(augmented_matrix * principal_vector).' * y_i) / max(norm(augmented_matrix * principal_vector)^2, eps);
    G_hat(cell_index, :) = (scale_factor * exp(-1j * angle(principal_vector(K + 1))) * principal_vector(1:K)).';
end
for update_index = 1:em_updates
    field = G_hat * S + B;
    concentration = (2 / max(sigma2, eps)) * Y .* abs(field);
    ratio = besseli(1, concentration, 1) ./ max(besseli(0, concentration, 1), realmin);
    expected_field = Y .* exp(1j * angle(field)) .* ratio;
    G_tilde = (expected_field - B) * S' / (S * S');
    G_processed = zeros(N, K);
    for user_index = 1:K
        channel_vector = G_tilde(:, user_index);
        for sweep_index = 1:cadzow_sweeps
            H_rows = N - p_H;
            H_columns = p_H + 1;
            H = zeros(H_rows, H_columns);
            for row_index = 1:H_rows
                H(row_index, :) = channel_vector(row_index:row_index + H_columns - 1).';
            end
            [left_vectors, singular_values, right_vectors] = svd(H, 'econ');
            H_rank = left_vectors(:, 1:rank_value) * singular_values(1:rank_value, 1:rank_value) * right_vectors(:, 1:rank_value)';
            reconstructed_channel = zeros(N, 1);
            copy_count = zeros(N, 1);
            for row_index = 1:H_rows
                for column_index = 1:H_columns
                    sample_index = row_index + column_index - 1;
                    reconstructed_channel(sample_index) = reconstructed_channel(sample_index) + H_rank(row_index, column_index);
                    copy_count(sample_index) = copy_count(sample_index) + 1;
                end
            end
            channel_vector = reconstructed_channel ./ max(copy_count, eps);
        end
        G_processed(:, user_index) = channel_vector;
    end
    G_hat = G_processed;
end
end
