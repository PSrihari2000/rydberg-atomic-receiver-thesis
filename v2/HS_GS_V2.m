%% HS_GS_V2.m
% Standalone Hankel-structured GS estimator for the RSR = 25 dB dataset.
%
% This v2 script reads the existing independent test datasets but writes all
% new results only inside the v2 folder. The original thesis folders are not
% modified. Candidate Hankel ranks are selected using held-out pilot
% magnitudes; the true channel is used only for the final NMSE calculation.

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
HS_GS_updates = 100;
rank_selection_updates = 25;
cadzow_sweeps = 4;
p_H = 3;
training_fraction = 0.70;
rank_list = 1:min(N - p_H, p_H + 1);
random_seed_base = 910000;

NMSE_HS_GS = zeros(length(P_list), length(SNR_dB));
NMSE_HS_GS_dB = zeros(length(P_list), length(SNR_dB));
selected_rank_mean = zeros(length(P_list), length(SNR_dB));
selected_rank_histogram = zeros(length(P_list), length(SNR_dB), length(rank_list));

for P_index = 1:length(P_list)
    P = P_list(P_index);
    test_file = fullfile(source_dataset_folder, sprintf('test_P%d.mat', P));
    test_source = load(test_file);

    if size(test_source.G_data, 1) ~= N || size(test_source.G_data, 2) ~= K
        error('Unexpected channel dimensions in %s.', test_file);
    end

    for SNR_index = 1:length(SNR_dB)
        selected_samples = find(test_source.SNR_dB_data(:) == SNR_dB(SNR_index));
        if isempty(selected_samples)
            error('No test samples found for P = %d and SNR = %g dB.', P, SNR_dB(SNR_index));
        end

        squared_error_sum = 0;
        channel_energy_sum = 0;
        rank_sum = 0;

        for sample_number = 1:length(selected_samples)
            sample_index = selected_samples(sample_number);
            Y = double(test_source.Y_data(:, :, sample_index));
            G = double(test_source.G_data(:, :, sample_index));
            S = double(test_source.S_data(:, :, sample_index));
            B = double(test_source.B_data(:, :, sample_index));
            sigma2 = double(test_source.sigma2_data(sample_index));

            rng(random_seed_base + 1000 * P + sample_index, 'twister');
            pilot_order = randperm(P);
            training_count = round(training_fraction * P);
            training_pilots = pilot_order(1:training_count);
            validation_pilots = pilot_order(training_count + 1:P);

            validation_error = zeros(length(rank_list), 1);
            for rank_index = 1:length(rank_list)
                candidate_rank = rank_list(rank_index);
                G_candidate = run_hs_gs(Y(:, training_pilots), ...
                    S(:, training_pilots), B(:, training_pilots), sigma2, ...
                    candidate_rank, rank_selection_updates, cadzow_sweeps, p_H);
                predicted_validation_magnitude = abs(G_candidate * S(:, validation_pilots) + B(:, validation_pilots));
                validation_error(rank_index) = norm(Y(:, validation_pilots) - predicted_validation_magnitude, 'fro')^2;
            end

            [minimum_validation_error, best_rank_index] = min(validation_error);
            if ~isfinite(minimum_validation_error)
                error('Nonfinite HS-GS rank-validation error detected.');
            end
            selected_rank = rank_list(best_rank_index);
            selected_rank_histogram(P_index, SNR_index, best_rank_index) = ...
                selected_rank_histogram(P_index, SNR_index, best_rank_index) + 1;

            G_hat = run_hs_gs(Y, S, B, sigma2, selected_rank, HS_GS_updates, cadzow_sweeps, p_H);
            squared_error_sum = squared_error_sum + norm(G_hat - G, 'fro')^2;
            channel_energy_sum = channel_energy_sum + norm(G, 'fro')^2;
            rank_sum = rank_sum + selected_rank;
        end

        NMSE_HS_GS(P_index, SNR_index) = squared_error_sum / max(channel_energy_sum, eps);
        NMSE_HS_GS_dB(P_index, SNR_index) = 10 * log10(NMSE_HS_GS(P_index, SNR_index));
        selected_rank_mean(P_index, SNR_index) = rank_sum / length(selected_samples);
        fprintf('HS-GS v2: P = %d, SNR = %3d dB, NMSE = %8.3f dB, mean rank = %.3f\n', ...
            P, SNR_dB(SNR_index), NMSE_HS_GS_dB(P_index, SNR_index), selected_rank_mean(P_index, SNR_index));
    end
end

save(fullfile(result_folder, 'HS_GS_V2_results.mat'), 'NMSE_HS_GS', ...
    'NMSE_HS_GS_dB', 'selected_rank_mean', 'selected_rank_histogram', ...
    'rank_list', 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', 'p_H', ...
    'HS_GS_updates', 'rank_selection_updates', 'cadzow_sweeps', ...
    'training_fraction');

figure('Color', 'w', 'Position', [100 100 1000 650]);
hold on;
grid on;
box on;
plot(SNR_dB, NMSE_HS_GS_dB(1, :), '--^', 'LineWidth', 1.5, ...
    'DisplayName', 'HS-GS v2, P = 10');
plot(SNR_dB, NMSE_HS_GS_dB(2, :), '-^', 'LineWidth', 1.5, ...
    'DisplayName', 'HS-GS v2, P = 30');
xlabel('SNR (dB)');
ylabel('NMSE (dB)');
title('HS-GS v2, N = 8, K = 3, RSR = 25 dB');
xticks(SNR_dB);
legend('Location', 'best');
exportgraphics(gcf, fullfile(result_folder, 'HS_GS_V2_nmse_vs_snr.png'), 'Resolution', 300);
savefig(gcf, fullfile(result_folder, 'HS_GS_V2_nmse_vs_snr.fig'));

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
