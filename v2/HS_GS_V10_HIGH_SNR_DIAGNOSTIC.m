%% HS_GS_V10_HIGH_SNR_DIAGNOSTIC.m
% Exploratory validation-only check of EM update count and one Hankel sweep.
% Do not use the V9 fresh test to choose settings for a future estimator.
clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end
checkpoint_file = fullfile(result_folder, 'HS_GS_V10_high_SNR_diagnostic_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V10_high_SNR_diagnostic_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V10_high_SNR_diagnostic_table.csv');
if exist(result_file, 'file')
    error('V10 results already exist at %s. Inspect them before rerunning.', result_file);
end

P_list = [10 30];
SNR_dB = [20 25 30];
update_list = [100 250 500 1000];
p_H = 2;
samples_per_point = 100;
samples_per_checkpoint = 20;
rank_selection_updates = 50;
training_fraction = 0.70;
rank_seed_base = 910000;
sample_seed_base = 20261025;
GS_max_iterations = 1000;
GS_change_tolerance = 1e-10;
requested_workers = 6;

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', requested_workers);
    pool = gcp('nocreate');
end
fprintf('V10 validation diagnostic using %d workers and %d samples per point.\n', ...
    pool.NumWorkers, samples_per_point);

selected_sample_indices = zeros(length(P_list), length(SNR_dB), samples_per_point);
completed_samples = false(size(selected_sample_indices));
EM_error = nan(length(P_list), length(SNR_dB), length(update_list), 2, samples_per_point);
EM_residual = nan(size(EM_error));
GS_error = nan(size(selected_sample_indices));
GS_residual = nan(size(selected_sample_indices));
channel_energy = nan(size(selected_sample_indices));
selected_rank = nan(size(selected_sample_indices));

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    matching_settings = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && ...
        isequal(checkpoint.update_list, update_list) && ...
        isequal(checkpoint.p_H, p_H) && ...
        isequal(checkpoint.samples_per_point, samples_per_point) && ...
        isequal(checkpoint.rank_selection_updates, rank_selection_updates) && ...
        isequal(checkpoint.training_fraction, training_fraction) && ...
        isequal(checkpoint.rank_seed_base, rank_seed_base) && ...
        isequal(checkpoint.sample_seed_base, sample_seed_base) && ...
        isequal(checkpoint.GS_max_iterations, GS_max_iterations) && ...
        isequal(checkpoint.GS_change_tolerance, GS_change_tolerance);
    if ~matching_settings
        error('V10 checkpoint settings have changed. Inspect the checkpoint before continuing.');
    end
    selected_sample_indices = checkpoint.selected_sample_indices;
    completed_samples = checkpoint.completed_samples;
    EM_error = checkpoint.EM_error;
    EM_residual = checkpoint.EM_residual;
    GS_error = checkpoint.GS_error;
    GS_residual = checkpoint.GS_residual;
    channel_energy = checkpoint.channel_energy;
    selected_rank = checkpoint.selected_rank;
    fprintf('Resumed %d/%d validation samples.\n', sum(completed_samples, 'all'), numel(completed_samples));
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    source = load(fullfile(dataset_folder, sprintf('validation_P%d.mat', P)));
    for SNR_index = 1:length(SNR_dB)
        all_at_SNR = find(source.SNR_dB_data(:) == SNR_dB(SNR_index));
        if length(all_at_SNR) < samples_per_point
            error('Too few validation samples at P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        stored_indices = squeeze(selected_sample_indices(P_index, SNR_index, :));
        if all(stored_indices == 0)
            rng(sample_seed_base + 100 * P + SNR_dB(SNR_index), 'twister');
            draw = randperm(length(all_at_SNR), samples_per_point);
            selected_sample_indices(P_index, SNR_index, :) = reshape(all_at_SNR(draw), 1, 1, []);
        elseif any(stored_indices == 0) || ~all(ismember(stored_indices, all_at_SNR))
            error('Stored V10 sample indices are invalid for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        point_indices = squeeze(selected_sample_indices(P_index, SNR_index, :));
        point_timer = tic;
        fprintf('P=%d, SNR=%d dB: starting or resuming.\n', P, SNR_dB(SNR_index));
        for block_start = 1:samples_per_checkpoint:samples_per_point
            block_positions = block_start:min(block_start + samples_per_checkpoint - 1, samples_per_point);
            done_in_block = squeeze(completed_samples(P_index, SNR_index, block_positions));
            pending_positions = block_positions(~done_in_block(:).');
            if isempty(pending_positions)
                continue;
            end
            block_EM0_error = zeros(length(pending_positions), length(update_list));
            block_EM1_error = zeros(size(block_EM0_error));
            block_EM0_residual = zeros(size(block_EM0_error));
            block_EM1_residual = zeros(size(block_EM0_error));
            block_GS_error = zeros(length(pending_positions), 1);
            block_GS_residual = zeros(length(pending_positions), 1);
            block_energy = zeros(length(pending_positions), 1);
            block_rank = zeros(length(pending_positions), 1);
            parfor local_number = 1:length(pending_positions)
                sample_position = pending_positions(local_number);
                sample_index = point_indices(sample_position);
                Y = double(source.Y_data(:, :, sample_index));
                G = double(source.G_data(:, :, sample_index));
                S = double(source.S_data(:, :, sample_index));
                B = double(source.B_data(:, :, sample_index));
                sigma2 = double(source.sigma2_data(sample_index));
                [~, rank_value] = hs_gs_v9_sample(Y, S, B, sigma2, P, sample_index, ...
                    p_H, 1, 1, rank_selection_updates, training_fraction, rank_seed_base);
                [error0, error1, residual0, residual1] = v10_EM_trajectories( ...
                    Y, G, S, B, sigma2, update_list, p_H, rank_value);
                G_hat_GS = v10_biased_GS(Y, S, B, GS_max_iterations, GS_change_tolerance);
                block_EM0_error(local_number, :) = error0;
                block_EM1_error(local_number, :) = error1;
                block_EM0_residual(local_number, :) = residual0;
                block_EM1_residual(local_number, :) = residual1;
                block_GS_error(local_number) = norm(G_hat_GS - G, 'fro')^2;
                block_GS_residual(local_number) = norm(abs(G_hat_GS * S + B) - Y, 'fro')^2;
                block_energy(local_number) = norm(G, 'fro')^2;
                block_rank(local_number) = rank_value;
            end
            for update_index = 1:length(update_list)
                EM_error(P_index, SNR_index, update_index, 1, pending_positions) = ...
                    reshape(block_EM0_error(:, update_index), 1, 1, 1, 1, []);
                EM_error(P_index, SNR_index, update_index, 2, pending_positions) = ...
                    reshape(block_EM1_error(:, update_index), 1, 1, 1, 1, []);
                EM_residual(P_index, SNR_index, update_index, 1, pending_positions) = ...
                    reshape(block_EM0_residual(:, update_index), 1, 1, 1, 1, []);
                EM_residual(P_index, SNR_index, update_index, 2, pending_positions) = ...
                    reshape(block_EM1_residual(:, update_index), 1, 1, 1, 1, []);
            end
            GS_error(P_index, SNR_index, pending_positions) = reshape(block_GS_error, 1, 1, []);
            GS_residual(P_index, SNR_index, pending_positions) = reshape(block_GS_residual, 1, 1, []);
            channel_energy(P_index, SNR_index, pending_positions) = reshape(block_energy, 1, 1, []);
            selected_rank(P_index, SNR_index, pending_positions) = reshape(block_rank, 1, 1, []);
            completed_samples(P_index, SNR_index, pending_positions) = true;
            save(checkpoint_file, 'P_list', 'SNR_dB', 'update_list', 'p_H', ...
                'samples_per_point', 'rank_selection_updates', 'training_fraction', ...
                'rank_seed_base', 'sample_seed_base', 'GS_max_iterations', ...
                'GS_change_tolerance', 'selected_sample_indices', 'completed_samples', ...
                'EM_error', 'EM_residual', 'GS_error', 'GS_residual', ...
                'channel_energy', 'selected_rank');
            fprintf('P=%d, SNR=%d dB: %d/%d samples, %.1f s elapsed.\n', ...
                P, SNR_dB(SNR_index), ...
                sum(completed_samples(P_index, SNR_index, :), 'all'), ...
                samples_per_point, toc(point_timer));
        end
    end
end

if ~all(completed_samples, 'all') || ~all(isfinite(EM_error), 'all') || ...
        ~all(isfinite(EM_residual), 'all') || ~all(isfinite(GS_error), 'all') || ...
        ~all(isfinite(GS_residual), 'all') || ~all(isfinite(channel_energy), 'all') || ...
        ~all(isfinite(selected_rank), 'all')
    error('V10 validation diagnostic has incomplete or nonfinite values.');
end

rows = zeros(length(P_list) * length(SNR_dB) * length(update_list), 10);
row_number = 0;
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        energy = squeeze(channel_energy(P_index, SNR_index, :));
        gs_errors = squeeze(GS_error(P_index, SNR_index, :));
        GS_NMSE_dB = 10 * log10(sum(gs_errors) / sum(energy));
        for update_index = 1:length(update_list)
            error0 = squeeze(EM_error(P_index, SNR_index, update_index, 1, :));
            error1 = squeeze(EM_error(P_index, SNR_index, update_index, 2, :));
            residual0 = squeeze(EM_residual(P_index, SNR_index, update_index, 1, :));
            residual1 = squeeze(EM_residual(P_index, SNR_index, update_index, 2, :));
            EM0_NMSE_dB = 10 * log10(sum(error0) / sum(energy));
            EM1_NMSE_dB = 10 * log10(sum(error1) / sum(energy));
            row_number = row_number + 1;
            rows(row_number, :) = [P_list(P_index), SNR_dB(SNR_index), ...
                update_list(update_index), GS_NMSE_dB, EM0_NMSE_dB, ...
                EM1_NMSE_dB, EM0_NMSE_dB - EM1_NMSE_dB, ...
                GS_NMSE_dB - EM1_NMSE_dB, ...
                mean(error1 < error0), ...
                10 * log10(sum(residual1) / sum(residual0))];
        end
    end
end
result_table = array2table(rows, 'VariableNames', {'P', 'SNR_dB', 'EM_updates', ...
    'GS_NMSE_dB', 'EM0_NMSE_dB', 'EM1_NMSE_dB', 'Hankel_gain_dB', ...
    'EM1_gain_over_GS_dB', 'fraction_samples_helped', 'residual_change_dB'});
save(result_file, 'result_table', 'P_list', 'SNR_dB', 'update_list', 'p_H', ...
    'samples_per_point', 'rank_selection_updates', 'training_fraction', ...
    'rank_seed_base', 'sample_seed_base', 'GS_max_iterations', ...
    'GS_change_tolerance', 'selected_sample_indices', 'completed_samples', ...
    'EM_error', 'EM_residual', 'GS_error', 'GS_residual', ...
    'channel_energy', 'selected_rank');
writetable(result_table, table_file);
disp(result_table);
fprintf('V10 validation diagnostic saved to %s\n', result_file);

function [error0, error1, residual0, residual1] = v10_EM_trajectories(Y, G, S, B, sigma2, update_list, p_H, rank_value)
N = size(Y, 1);
K = size(S, 1);
G_initial = zeros(N, K);
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
    G_initial(cell_index, :) = (r * exp(-1j * angle(v(K + 1))) * v(1:K)).';
end
G_zero = G_initial;
G_one = G_initial;
right_inverse = S' / (S * S');
error0 = zeros(1, length(update_list));
error1 = zeros(1, length(update_list));
residual0 = zeros(1, length(update_list));
residual1 = zeros(1, length(update_list));
for update_number = 1:update_list(end)
    G_zero = v10_EM_update(G_zero, Y, S, B, sigma2, right_inverse);
    G_one = v10_EM_update(G_one, Y, S, B, sigma2, right_inverse);
    for user_index = 1:K
        G_one(:, user_index) = v10_one_Cadzow_sweep(G_one(:, user_index), p_H, rank_value);
    end
    recorded_index = find(update_list == update_number, 1);
    if ~isempty(recorded_index)
        error0(recorded_index) = norm(G_zero - G, 'fro')^2;
        error1(recorded_index) = norm(G_one - G, 'fro')^2;
        residual0(recorded_index) = norm(abs(G_zero * S + B) - Y, 'fro')^2;
        residual1(recorded_index) = norm(abs(G_one * S + B) - Y, 'fro')^2;
    end
end
end

function G_next = v10_EM_update(G_current, Y, S, B, sigma2, right_inverse)
field = G_current * S + B;
kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
expected_field = Y .* exp(1j * angle(field)) .* ratio;
G_next = (expected_field - B) * right_inverse;
end

function channel_vector = v10_one_Cadzow_sweep(channel_vector, p_H, rank_value)
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
        index = row_index + column_index - 1;
        reconstructed(index) = reconstructed(index) + H_rank(row_index, column_index);
        copy_count(index) = copy_count(index) + 1;
    end
end
channel_vector = reconstructed ./ copy_count;
end

function G_hat = v10_biased_GS(Y, S, B, maximum_iterations, tolerance)
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
    v = v / norm(v);
    r = (abs(A * v).' * y_i) / norm(A * v)^2;
    G_hat(cell_index, :) = (r * exp(-1j * angle(v(K + 1))) * v(1:K)).';
end
right_inverse = S' / (S * S');
for update_number = 1:maximum_iterations
    previous = G_hat;
    G_hat = (Y .* exp(1j * angle(G_hat * S + B)) - B) * right_inverse;
    if norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps) < tolerance
        break;
    end
end
end
