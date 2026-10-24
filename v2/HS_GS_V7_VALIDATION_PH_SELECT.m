%% HS_GS_V7_VALIDATION_PH_SELECT.m
% Select p_H using validation pilot prediction, then evaluate the frozen
% choice on independent test samples. p_H=2 and p_H=3 are the two distinct
% Hankel geometries found in V6 for N=8.
clear; clc; close all;

script_folder = fileparts(mfilename('fullpath'));
source_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir'), mkdir(result_folder); end

P_list = [10 30]; SNR_dB = -5:5:30; N = 8; K = 3; RSR_dB = 25;
p_H_list = [2 3];
rank_selection_updates = 50; training_fraction = 0.70; random_seed_base = 910000;
maximum_validation_samples_per_snr = 100;
maximum_test_samples_per_snr = inf;
% Frozen from V5 validation selection; V7 only selects p_H.
selected_updates = [100 100 100 100 100 100 100 1000; 100 100 100 100 500 100 100 1000];
selected_sweeps  = [4 2 1 1 1 1 1 0;       2 2 1 1 1 1 1 1];

requested_workers = 6;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', requested_workers);
elseif pool.NumWorkers ~= requested_workers
    delete(pool);
    parpool('local', requested_workers);
end
pool = gcp('nocreate');
fprintf('HS-GS V7 validation-based p_H selection started with %d workers.\n', pool.NumWorkers);

checkpoint_file = fullfile(result_folder, 'HS_GS_V7_validation_p_H_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V7_validation_p_H_results.mat');
validation_error = nan(length(P_list), length(SNR_dB), length(p_H_list));
selected_p_H = zeros(length(P_list), length(SNR_dB));
test_NMSE = nan(length(P_list), length(SNR_dB));
test_NMSE_dB = nan(length(P_list), length(SNR_dB));
test_mean_rank = nan(length(P_list), length(SNR_dB));
completed_points = false(length(P_list), length(SNR_dB));

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    if isequal(checkpoint.P_list, P_list) && isequal(checkpoint.SNR_dB, SNR_dB) && isequal(checkpoint.p_H_list, p_H_list)
        validation_error = checkpoint.validation_error;
        selected_p_H = checkpoint.selected_p_H;
        test_NMSE = checkpoint.test_NMSE;
        test_NMSE_dB = checkpoint.test_NMSE_dB;
        test_mean_rank = checkpoint.test_mean_rank;
        completed_points = checkpoint.completed_points;
        fprintf('Loaded checkpoint: %d of %d P/SNR points complete.\n', sum(completed_points, 'all'), numel(completed_points));
    end
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    validation_source = load(fullfile(source_folder, sprintf('validation_P%d.mat', P)));
    test_source = load(fullfile(source_folder, sprintf('test_P%d.mat', P)));
    for snr_index = 1:length(SNR_dB)
        if completed_points(P_index, snr_index)
            fprintf('Skipping completed point: P=%d, SNR=%d dB\n', P, SNR_dB(snr_index));
            continue;
        end
        point_timer = tic;
        validation_samples = find(validation_source.SNR_dB_data(:) == SNR_dB(snr_index));
        validation_samples = limit_samples(validation_samples, maximum_validation_samples_per_snr);
        test_samples = find(test_source.SNR_dB_data(:) == SNR_dB(snr_index));
        test_samples = limit_samples(test_samples, maximum_test_samples_per_snr);

        % Compare p_H using held-out pilot prediction on validation samples.
        for p_index = 1:length(p_H_list)
            p_H = p_H_list(p_index);
            rank_list = 1:min(N-p_H, p_H+1);
            sample_prediction_error = zeros(length(validation_samples), 1);
            parfor sample_number = 1:length(validation_samples)
                sample_index = validation_samples(sample_number);
                Y = double(validation_source.Y_data(:, :, sample_index));
                S = double(validation_source.S_data(:, :, sample_index));
                B = double(validation_source.B_data(:, :, sample_index));
                sigma2 = double(validation_source.sigma2_data(sample_index));
                [training_pilots, held_out_pilots] = split_pilots(P, training_fraction, random_seed_base, sample_index);
                candidate_error = zeros(length(rank_list), 1);
                for rank_index = 1:length(rank_list)
                    G_candidate = run_hs_gs(Y(:, training_pilots), S(:, training_pilots), B(:, training_pilots), sigma2, rank_list(rank_index), rank_selection_updates, selected_sweeps(P_index, snr_index), p_H);
                    predicted_magnitude = abs(G_candidate * S(:, held_out_pilots) + B(:, held_out_pilots));
                    candidate_error(rank_index) = norm(Y(:, held_out_pilots) - predicted_magnitude, 'fro')^2;
                end
                [~, best_rank_index] = min(candidate_error);
                G_candidate = run_hs_gs(Y(:, training_pilots), S(:, training_pilots), B(:, training_pilots), sigma2, rank_list(best_rank_index), selected_updates(P_index, snr_index), selected_sweeps(P_index, snr_index), p_H);
                predicted_magnitude = abs(G_candidate * S(:, held_out_pilots) + B(:, held_out_pilots));
                sample_prediction_error(sample_number) = norm(Y(:, held_out_pilots) - predicted_magnitude, 'fro')^2 / numel(held_out_pilots);
            end
            validation_error(P_index, snr_index, p_index) = mean(sample_prediction_error);
        end
        [~, best_p_index] = min(squeeze(validation_error(P_index, snr_index, :)));
        selected_p_H(P_index, snr_index) = p_H_list(best_p_index);
        p_H = selected_p_H(P_index, snr_index);
        rank_list = 1:min(N-p_H, p_H+1);
        fprintf('P=%d, SNR=%3d dB: validation chose p_H=%d (errors %.6g, %.6g)\n', P, SNR_dB(snr_index), p_H, validation_error(P_index, snr_index, 1), validation_error(P_index, snr_index, 2));

        % Evaluate the validation-selected p_H on independent test samples.
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
                G_candidate = run_hs_gs(Y(:, training_pilots), S(:, training_pilots), B(:, training_pilots), sigma2, rank_list(rank_index), rank_selection_updates, selected_sweeps(P_index, snr_index), p_H);
                predicted_magnitude = abs(G_candidate * S(:, held_out_pilots) + B(:, held_out_pilots));
                candidate_error(rank_index) = norm(Y(:, held_out_pilots) - predicted_magnitude, 'fro')^2;
            end
            [~, best_rank_index] = min(candidate_error);
            sample_selected_rank(sample_number) = rank_list(best_rank_index);
            G_hat = run_hs_gs(Y, S, B, sigma2, sample_selected_rank(sample_number), selected_updates(P_index, snr_index), selected_sweeps(P_index, snr_index), p_H);
            sample_squared_error(sample_number) = norm(G_hat - G, 'fro')^2;
            sample_channel_energy(sample_number) = norm(G, 'fro')^2;
        end
        test_NMSE(P_index, snr_index) = sum(sample_squared_error) / max(sum(sample_channel_energy), eps);
        test_NMSE_dB(P_index, snr_index) = 10 * log10(test_NMSE(P_index, snr_index));
        test_mean_rank(P_index, snr_index) = mean(sample_selected_rank);
        completed_points(P_index, snr_index) = true;
        fprintf('Test: P=%d, SNR=%3d dB, p_H=%d, NMSE=%8.3f dB, mean rank=%.3f, elapsed=%.1f s\n', P, SNR_dB(snr_index), p_H, test_NMSE_dB(P_index, snr_index), test_mean_rank(P_index, snr_index), toc(point_timer));
        save(checkpoint_file, 'validation_error', 'selected_p_H', 'test_NMSE', 'test_NMSE_dB', 'test_mean_rank', 'completed_points', 'p_H_list', 'P_list', 'SNR_dB', 'selected_updates', 'selected_sweeps', 'rank_selection_updates', 'training_fraction', 'maximum_validation_samples_per_snr', 'maximum_test_samples_per_snr', 'N', 'K', 'RSR_dB');
    end
end
save(result_file, 'validation_error', 'selected_p_H', 'test_NMSE', 'test_NMSE_dB', 'test_mean_rank', 'p_H_list', 'P_list', 'SNR_dB', 'selected_updates', 'selected_sweeps', 'rank_selection_updates', 'training_fraction', 'maximum_validation_samples_per_snr', 'maximum_test_samples_per_snr', 'N', 'K', 'RSR_dB');
fprintf('Saved V7 results to %s\n', result_file);

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
    A = [S.', b_i];
    M = A' * diag(y_i) * A;
    [V, D] = eig(M);
    [~, principal_index] = max(real(diag(D)));
    v = V(:, principal_index);
    v = v / max(norm(v), eps);
    r = (abs(A * v).' * y_i) / max(norm(A * v)^2, eps);
    G_hat(cell_index, :) = (r * exp(-1j * angle(v(K + 1))) * v(1:K)).';
end
for update_index = 1:em_updates
    field = G_hat * S + B;
    kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
    ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
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
