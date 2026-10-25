%% HS_GS_V13_PAPER_ALIGNED_DIAGNOSTIC.m
% Paper-aligned HS-GS diagnostic for the project's N=8, RSR=25 dB data.
% Uses validation samples, held-out pilot magnitudes for rank selection,
% p_H=3 (5x4 Hankel matrices), and interleaved Cadzow projections.
% Reports EM-GS (no projection), HS-GS, biased GS, and linearized GD.
% True-channel Hankel fit is reported only as an oracle diagnostic; it is
% never used to choose the estimator rank.
%
% Run from MATLAB with:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V13_PAPER_ALIGNED_DIAGNOSTIC

clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

checkpoint_file = fullfile(result_folder, 'HS_GS_V13_paper_aligned_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V13_paper_aligned_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V13_paper_aligned_table.csv');
if exist(result_file, 'file')
    error('V13 results already exist at %s. Inspect or archive them before rerunning.', result_file);
end

P_list = [10 30];
SNR_dB = -5:5:30;
N = 8;
K = 3;
RSR_dB = 25;
p_H = 3;                         % Balanced N=8 Hankel: 5 rows x 4 columns
rank_list = 1:min(N-p_H, p_H+1); % Ranks 1 through 4
selection_updates = 25;
final_updates = 100;
cadzow_sweeps = 4;
training_fraction = 0.70;
samples_per_point = 100;
samples_per_checkpoint = 20;
sample_seed_base = 2026102513;
split_seed_base = 1310000;
GS_GD_max_iterations = 1000;
GS_GD_change_tolerance = 1e-10;
G0_variance = 0.1;
workers_if_starting_pool = 4;

if N ~= 8 || K ~= 3 || p_H ~= 3 || ~isequal(rank_list, 1:4)
    error('This V13 script is configured for N=8, K=3, p_H=3, ranks 1:4.');
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('V13 paper-aligned HS-GS diagnostic using %d workers.\n', pool.NumWorkers);
fprintf('Data: RSR=%g dB; N=%d; K=%d; P=%s; SNR=%s dB.\n', ...
    RSR_dB, N, K, mat2str(P_list), mat2str(SNR_dB));
fprintf('Rank selection uses held-out pilot prediction error; truth is used only for scoring/oracle fit.\n');

sample_indices = zeros(length(P_list), length(SNR_dB), samples_per_point);
completed = false(size(sample_indices));
HS_error = nan(size(sample_indices));
EM_error = nan(size(sample_indices));
GS_error = nan(size(sample_indices));
GD_error = nan(size(sample_indices));
channel_energy = nan(size(sample_indices));
selected_rank = nan(size(sample_indices));
heldout_pilot_error = nan(size(sample_indices));
HS_measurement_residual = nan(size(sample_indices));
oracle_error_by_rank = nan(length(P_list), length(SNR_dB), samples_per_point, length(rank_list));

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    settings_match = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && isequal(checkpoint.N, N) && ...
        isequal(checkpoint.K, K) && isequal(checkpoint.RSR_dB, RSR_dB) && ...
        isequal(checkpoint.p_H, p_H) && isequal(checkpoint.rank_list, rank_list) && ...
        isequal(checkpoint.selection_updates, selection_updates) && ...
        isequal(checkpoint.final_updates, final_updates) && ...
        isequal(checkpoint.cadzow_sweeps, cadzow_sweeps) && ...
        isequal(checkpoint.training_fraction, training_fraction) && ...
        isequal(checkpoint.samples_per_point, samples_per_point) && ...
        isequal(checkpoint.sample_seed_base, sample_seed_base) && ...
        isequal(checkpoint.split_seed_base, split_seed_base) && ...
        isequal(checkpoint.GS_GD_max_iterations, GS_GD_max_iterations) && ...
        isequal(checkpoint.GS_GD_change_tolerance, GS_GD_change_tolerance) && ...
        isequal(checkpoint.G0_variance, G0_variance);
    if ~settings_match
        error('V13 checkpoint settings differ from this script. Preserve and inspect the checkpoint before changing settings.');
    end
    sample_indices = checkpoint.sample_indices;
    completed = checkpoint.completed;
    HS_error = checkpoint.HS_error;
    EM_error = checkpoint.EM_error;
    GS_error = checkpoint.GS_error;
    GD_error = checkpoint.GD_error;
    channel_energy = checkpoint.channel_energy;
    selected_rank = checkpoint.selected_rank;
    heldout_pilot_error = checkpoint.heldout_pilot_error;
    HS_measurement_residual = checkpoint.HS_measurement_residual;
    oracle_error_by_rank = checkpoint.oracle_error_by_rank;
    fprintf('Resumed %d/%d completed samples from checkpoint.\n', sum(completed, 'all'), numel(completed));
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    source_file = fullfile(dataset_folder, sprintf('validation_P%d.mat', P));
    if ~exist(source_file, 'file')
        error('Missing validation data file: %s', source_file);
    end
    source = load(source_file);
    if source.N ~= N || source.K ~= K || source.P ~= P || source.RSR_dB ~= RSR_dB
        error('Dataset metadata mismatch in %s.', source_file);
    end

    for SNR_index = 1:length(SNR_dB)
        candidate_indices = find(source.SNR_dB_data(:) == SNR_dB(SNR_index));
        if length(candidate_indices) < samples_per_point
            error('Not enough validation samples for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        stored_indices = squeeze(sample_indices(P_index, SNR_index, :));
        if all(stored_indices == 0)
            rng(sample_seed_base + 100 * P + SNR_dB(SNR_index), 'twister');
            selected_indices = candidate_indices(randperm(length(candidate_indices), samples_per_point));
            sample_indices(P_index, SNR_index, :) = reshape(selected_indices, 1, 1, []);
        elseif any(stored_indices == 0) || ~all(ismember(stored_indices, candidate_indices))
            error('Saved sample indices are invalid for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end

        point_indices = squeeze(sample_indices(P_index, SNR_index, :));
        point_timer = tic;
        fprintf('Starting V13: P=%d, SNR=%3d dB.\n', P, SNR_dB(SNR_index));
        for block_start = 1:samples_per_checkpoint:samples_per_point
            block_positions = block_start:min(block_start + samples_per_checkpoint - 1, samples_per_point);
            completion_mask = squeeze(completed(P_index, SNR_index, block_positions));
            pending_positions = block_positions(~completion_mask(:).');
            if isempty(pending_positions)
                continue;
            end

            block_sample_indices = point_indices(pending_positions);
            block_Y_data = source.Y_data(:, :, block_sample_indices);
            block_G_data = source.G_data(:, :, block_sample_indices);
            block_S_data = source.S_data(:, :, block_sample_indices);
            block_B_data = source.B_data(:, :, block_sample_indices);
            block_sigma2_data = source.sigma2_data(block_sample_indices);
            block_split_seeds = split_seed_base + 100000 * P + block_sample_indices;

            block_HS_error = zeros(length(pending_positions), 1);
            block_EM_error = zeros(length(pending_positions), 1);
            block_GS_error = zeros(length(pending_positions), 1);
            block_GD_error = zeros(length(pending_positions), 1);
            block_energy = zeros(length(pending_positions), 1);
            block_rank = zeros(length(pending_positions), 1);
            block_pilot_error = zeros(length(pending_positions), 1);
            block_residual = zeros(length(pending_positions), 1);
            block_oracle_error = zeros(length(pending_positions), length(rank_list));

            parfor local_index = 1:length(pending_positions)
                sample_index = block_sample_indices(local_index);
                Y = double(block_Y_data(:, :, local_index));
                G = double(block_G_data(:, :, local_index));
                S = double(block_S_data(:, :, local_index));
                B = double(block_B_data(:, :, local_index));
                sigma2 = double(block_sigma2_data(local_index));

                split_seed = block_split_seeds(local_index);
                [G_hat_HS, chosen_rank, prediction_error] = v13_select_and_refit( ...
                    Y, S, B, sigma2, p_H, rank_list, selection_updates, ...
                    final_updates, cadzow_sweeps, training_fraction, split_seed);
                G_hat_EM = v13_em_gs(Y, S, B, sigma2, final_updates, 0, 1, p_H);
                G_hat_GS = v13_biased_GS(Y, S, B, GS_GD_max_iterations, GS_GD_change_tolerance);
                G_hat_GD = v13_linearized_GD(Y, S, B, GS_GD_max_iterations, ...
                    GS_GD_change_tolerance, G0_variance, sample_index);

                block_HS_error(local_index) = norm(G_hat_HS - G, 'fro')^2;
                block_EM_error(local_index) = norm(G_hat_EM - G, 'fro')^2;
                block_GS_error(local_index) = norm(G_hat_GS - G, 'fro')^2;
                block_GD_error(local_index) = norm(G_hat_GD - G, 'fro')^2;
                block_energy(local_index) = norm(G, 'fro')^2;
                block_rank(local_index) = chosen_rank;
                block_pilot_error(local_index) = prediction_error;
                block_residual(local_index) = norm(abs(G_hat_HS * S + B) - Y, 'fro')^2;
                sample_oracle_errors = zeros(1, numel(rank_list));
                for oracle_rank_index = 1:numel(rank_list)
                    G_oracle_projection = v13_project_channel(G, p_H, rank_list(oracle_rank_index));
                    sample_oracle_errors(oracle_rank_index) = norm(G_oracle_projection - G, 'fro')^2;
                end
                block_oracle_error(local_index, :) = sample_oracle_errors;
            end

            HS_error(P_index, SNR_index, pending_positions) = reshape(block_HS_error, 1, 1, []);
            EM_error(P_index, SNR_index, pending_positions) = reshape(block_EM_error, 1, 1, []);
            GS_error(P_index, SNR_index, pending_positions) = reshape(block_GS_error, 1, 1, []);
            GD_error(P_index, SNR_index, pending_positions) = reshape(block_GD_error, 1, 1, []);
            channel_energy(P_index, SNR_index, pending_positions) = reshape(block_energy, 1, 1, []);
            selected_rank(P_index, SNR_index, pending_positions) = reshape(block_rank, 1, 1, []);
            heldout_pilot_error(P_index, SNR_index, pending_positions) = reshape(block_pilot_error, 1, 1, []);
            HS_measurement_residual(P_index, SNR_index, pending_positions) = reshape(block_residual, 1, 1, []);
            oracle_error_by_rank(P_index, SNR_index, pending_positions, :) = ...
                reshape(block_oracle_error, 1, 1, length(pending_positions), length(rank_list));
            completed(P_index, SNR_index, pending_positions) = true;

            save(checkpoint_file, 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
                'p_H', 'rank_list', 'selection_updates', 'final_updates', ...
                'cadzow_sweeps', 'training_fraction', 'samples_per_point', ...
                'sample_seed_base', 'split_seed_base', 'GS_GD_max_iterations', ...
                'GS_GD_change_tolerance', 'G0_variance', 'sample_indices', ...
                'completed', 'HS_error', 'EM_error', 'GS_error', 'GD_error', ...
                'channel_energy', 'selected_rank', 'heldout_pilot_error', ...
                'HS_measurement_residual', 'oracle_error_by_rank');
            fprintf('P=%d, SNR=%3d dB: %d/%d samples, %.1f s elapsed.\n', ...
                P, SNR_dB(SNR_index), sum(completed(P_index, SNR_index, :), 'all'), ...
                samples_per_point, toc(point_timer));
        end
    end
end

if ~all(completed, 'all') || ~all(isfinite(HS_error), 'all') || ...
        ~all(isfinite(EM_error), 'all') || ~all(isfinite(GS_error), 'all') || ...
        ~all(isfinite(GD_error), 'all') || ~all(isfinite(channel_energy), 'all') || ...
        ~all(isfinite(selected_rank), 'all') || ~all(isfinite(oracle_error_by_rank), 'all')
    error('V13 has incomplete or nonfinite values. The checkpoint has been preserved.');
end

rows = zeros(length(P_list) * length(SNR_dB), 11);
row_number = 0;
oracle_best_nmse_db = zeros(length(P_list), length(SNR_dB));
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        energy = squeeze(channel_energy(P_index, SNR_index, :));
        oracle_by_rank = squeeze(sum(oracle_error_by_rank(P_index, SNR_index, :, :), 3));
        [best_oracle_error, best_oracle_index] = min(oracle_by_rank);
        oracle_best_nmse_db(P_index, SNR_index) = 10 * log10(best_oracle_error / sum(energy));
        row_number = row_number + 1;
        rows(row_number, :) = [P_list(P_index), SNR_dB(SNR_index), ...
            10 * log10(sum(squeeze(HS_error(P_index, SNR_index, :))) / sum(energy)), ...
            10 * log10(sum(squeeze(EM_error(P_index, SNR_index, :))) / sum(energy)), ...
            10 * log10(sum(squeeze(GS_error(P_index, SNR_index, :))) / sum(energy)), ...
            10 * log10(sum(squeeze(GD_error(P_index, SNR_index, :))) / sum(energy)), ...
            mean(squeeze(selected_rank(P_index, SNR_index, :))), ...
            mean(squeeze(heldout_pilot_error(P_index, SNR_index, :))), ...
            10 * log10(best_oracle_error / sum(energy)), ...
            rank_list(best_oracle_index), ...
            mean(squeeze(HS_measurement_residual(P_index, SNR_index, :)))];
    end
end

result_table = array2table(rows, 'VariableNames', {'P', 'SNR_dB', ...
    'HS_NMSE_dB', 'EM_GS_NMSE_dB', 'biased_GS_NMSE_dB', 'linearized_GD_NMSE_dB', ...
    'mean_selected_rank', 'mean_heldout_pilot_error_raw', ...
    'oracle_best_Hankel_fit_NMSE_dB', 'oracle_best_rank', ...
    'mean_HS_measurement_residual_raw'});
save(result_file, 'result_table', 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
    'p_H', 'rank_list', 'selection_updates', 'final_updates', 'cadzow_sweeps', ...
    'training_fraction', 'samples_per_point', 'sample_indices', 'completed', ...
    'HS_error', 'EM_error', 'GS_error', 'GD_error', 'channel_energy', ...
    'selected_rank', 'heldout_pilot_error', 'HS_measurement_residual', ...
    'oracle_error_by_rank', 'oracle_best_nmse_db');
writetable(result_table, table_file);
disp(result_table);
fprintf('V13 diagnostic results saved to %s\n', result_file);

function [G_hat, selected_rank, best_prediction_error] = v13_select_and_refit( ...
        Y, S, B, sigma2, p_H, rank_list, selection_updates, final_updates, ...
        cadzow_sweeps, training_fraction, split_seed)
P = size(S, 2);
rng(split_seed, 'twister');
pilot_order = randperm(P);
training_count = round(training_fraction * P);
training_pilots = pilot_order(1:training_count);
heldout_pilots = pilot_order(training_count + 1:end);
candidate_errors = inf(length(rank_list), 1);
for rank_index = 1:length(rank_list)
    G_candidate = v13_em_gs(Y(:, training_pilots), S(:, training_pilots), ...
        B(:, training_pilots), sigma2, selection_updates, cadzow_sweeps, ...
        rank_list(rank_index), p_H);
    predicted = abs(G_candidate * S(:, heldout_pilots) + B(:, heldout_pilots));
    candidate_errors(rank_index) = norm(Y(:, heldout_pilots) - predicted, 'fro')^2;
end
[best_prediction_error, best_rank_index] = min(candidate_errors);
selected_rank = rank_list(best_rank_index);
G_hat = v13_em_gs(Y, S, B, sigma2, final_updates, cadzow_sweeps, selected_rank, p_H);
end

function G_hat = v13_em_gs(Y, S, B, sigma2, number_of_updates, sweeps, rank_value, p_H)
G_hat = v13_spectral_initialize(Y, S, B);
right_inverse = S' / (S * S');
for update_index = 1:number_of_updates
    field = G_hat * S + B;
    kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
    bessel_ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
    expected_field = Y .* exp(1j * angle(field)) .* bessel_ratio;
    G_hat = (expected_field - B) * right_inverse;
    if sweeps > 0
        G_hat = v13_project_channel(G_hat, p_H, rank_value, sweeps);
    end
end
end

function G_hat = v13_spectral_initialize(Y, S, B)
N = size(Y, 1);
K = size(S, 1);
G_hat = zeros(N, K);
for cell_index = 1:N
    y_i = Y(cell_index, :).';
    b_i = B(cell_index, :).';
    A_bar_i = [S.', b_i];
    M_i = A_bar_i' * diag(y_i) * A_bar_i;
    [V_i, D_i] = eig(M_i);
    [~, principal_index] = max(real(diag(D_i)));
    v_i = V_i(:, principal_index);
    v_i = v_i / max(norm(v_i), eps);
    amplitude_scale = (abs(A_bar_i * v_i).' * y_i) / max(norm(A_bar_i * v_i)^2, eps);
    G_hat(cell_index, :) = (amplitude_scale * exp(-1j * angle(v_i(K + 1))) * v_i(1:K)).';
end
end

function G_projected = v13_project_channel(G, p_H, rank_value, number_of_sweeps)
if nargin < 4
    number_of_sweeps = 1;
end
G_projected = G;
for sweep_index = 1:number_of_sweeps
    for user_index = 1:size(G_projected, 2)
        channel_vector = G_projected(:, user_index);
        N = length(channel_vector);
        H_rows = N - p_H;
        H_columns = p_H + 1;
        H = zeros(H_rows, H_columns);
        for row_index = 1:H_rows
            H(row_index, :) = channel_vector(row_index:row_index + H_columns - 1).';
        end
        [U, Sigma, V] = svd(H, 'econ');
        H_rank = U(:, 1:rank_value) * Sigma(1:rank_value, 1:rank_value) * V(:, 1:rank_value)';
        reconstructed = zeros(N, 1);
        copy_count = zeros(N, 1);
        for row_index = 1:H_rows
            for column_index = 1:H_columns
                vector_index = row_index + column_index - 1;
                reconstructed(vector_index) = reconstructed(vector_index) + H_rank(row_index, column_index);
                copy_count(vector_index) = copy_count(vector_index) + 1;
            end
        end
        G_projected(:, user_index) = reconstructed ./ copy_count;
    end
end
end

function G_hat = v13_biased_GS(Y, S, B, maximum_iterations, tolerance)
G_hat = v13_spectral_initialize(Y, S, B);
right_inverse = S' / (S * S');
for iteration = 1:maximum_iterations
    previous = G_hat;
    G_hat = (Y .* exp(1j * angle(G_hat * S + B)) - B) * right_inverse;
    if norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps) < tolerance
        break;
    end
end
end

function G_hat = v13_linearized_GD(Y, S, B, maximum_iterations, tolerance, initial_variance, sample_index)
rng(720000 + sample_index, 'twister');
N = size(Y, 1);
K = size(S, 1);
G_hat = sqrt(initial_variance / 2) * (randn(N, K) + 1j * randn(N, K));
Z = exp(-1j * angle(B));
step_size = 1 / max(real(eig(S * S')));
for iteration = 1:maximum_iterations
    previous = G_hat;
    G_hat = G_hat + step_size * ((Y - abs(B) - real(Z .* (G_hat * S))) .* conj(Z)) * S';
    if norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps) < tolerance
        break;
    end
end
end
