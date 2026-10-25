%% HS_GS_V12_PILOT_SELECTED_PROJECTION.m
% Select rank and final Hankel-projection strength using held-out pilot
% magnitude prediction only, then refit on all pilots and score channels.
% Uses a reserved subset of the original training files for this diagnostic.
clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end
checkpoint_file = fullfile(result_folder, 'HS_GS_V12_pilot_selected_projection_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V12_pilot_selected_projection_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V12_pilot_selected_projection_table.csv');
if exist(result_file, 'file')
    error('V12 results already exist at %s. Inspect them before rerunning.', result_file);
end

P_list = [10 30];
SNR_dB = [20 25 30];
p_H = 2;
rank_list = 1:min(8 - p_H, p_H + 1);
alpha_list = [0 0.10 0.25 0.50 1.00];
em_updates = 250;
training_fraction = 0.70;
samples_per_point = 100;
samples_per_checkpoint = 20;
sample_seed_base = 2026102512;
split_seed_base = 1210000;
GS_max_iterations = 1000;
GS_change_tolerance = 1e-10;
G0_variance = 0.1;
workers_if_starting_pool = 6;

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('V12 pilot-selected projection diagnostic using %d workers.\n', pool.NumWorkers);
fprintf('Selection uses pilot prediction error; channel truth is used only for NMSE scoring.\n');

sample_indices = zeros(length(P_list), length(SNR_dB), samples_per_point);
completed = false(size(sample_indices));
HS_error = nan(size(sample_indices));
GS_error = nan(size(sample_indices));
GD_error = nan(size(sample_indices));
channel_energy = nan(size(sample_indices));
selected_alpha = nan(size(sample_indices));
selected_rank = nan(size(sample_indices));
pilot_prediction_error = nan(size(sample_indices));
HS_measurement_residual = nan(size(sample_indices));

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    matching_settings = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && ...
        isequal(checkpoint.p_H, p_H) && ...
        isequal(checkpoint.rank_list, rank_list) && ...
        isequal(checkpoint.alpha_list, alpha_list) && ...
        isequal(checkpoint.em_updates, em_updates) && ...
        isequal(checkpoint.training_fraction, training_fraction) && ...
        isequal(checkpoint.samples_per_point, samples_per_point) && ...
        isequal(checkpoint.sample_seed_base, sample_seed_base) && ...
        isequal(checkpoint.split_seed_base, split_seed_base) && ...
        isequal(checkpoint.GS_max_iterations, GS_max_iterations) && ...
        isequal(checkpoint.GS_change_tolerance, GS_change_tolerance) && ...
        isequal(checkpoint.G0_variance, G0_variance);
    if ~matching_settings
        error('V12 checkpoint settings differ from this script. Inspect before resuming.');
    end
    sample_indices = checkpoint.sample_indices;
    completed = checkpoint.completed;
    HS_error = checkpoint.HS_error;
    GS_error = checkpoint.GS_error;
    GD_error = checkpoint.GD_error;
    channel_energy = checkpoint.channel_energy;
    selected_alpha = checkpoint.selected_alpha;
    selected_rank = checkpoint.selected_rank;
    pilot_prediction_error = checkpoint.pilot_prediction_error;
    HS_measurement_residual = checkpoint.HS_measurement_residual;
    fprintf('Resumed %d/%d samples.\n', sum(completed, 'all'), numel(completed));
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    source = load(fullfile(dataset_folder, sprintf('train_P%d.mat', P)));
    for SNR_index = 1:length(SNR_dB)
        candidates = find(source.SNR_dB_data(:) == SNR_dB(SNR_index));
        if length(candidates) < samples_per_point
            error('Not enough train split samples for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        stored_indices = squeeze(sample_indices(P_index, SNR_index, :));
        if all(stored_indices == 0)
            rng(sample_seed_base + 100 * P + SNR_dB(SNR_index), 'twister');
            draw = randperm(length(candidates), samples_per_point);
            sample_indices(P_index, SNR_index, :) = reshape(candidates(draw), 1, 1, []);
        elseif any(stored_indices == 0) || ~all(ismember(stored_indices, candidates))
            error('Saved V12 sample indices are invalid for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        point_indices = squeeze(sample_indices(P_index, SNR_index, :));
        point_timer = tic;
        fprintf('P=%d, SNR=%d dB: starting or resuming.\n', P, SNR_dB(SNR_index));
        for block_start = 1:samples_per_checkpoint:samples_per_point
            block_positions = block_start:min(block_start + samples_per_checkpoint - 1, samples_per_point);
            pending_positions = block_positions(~squeeze(completed(P_index, SNR_index, block_positions)).');
            if isempty(pending_positions)
                continue;
            end
            block_HS_error = zeros(length(pending_positions), 1);
            block_GS_error = zeros(length(pending_positions), 1);
            block_GD_error = zeros(length(pending_positions), 1);
            block_energy = zeros(length(pending_positions), 1);
            block_alpha = zeros(length(pending_positions), 1);
            block_rank = zeros(length(pending_positions), 1);
            block_pilot_error = zeros(length(pending_positions), 1);
            block_residual = zeros(length(pending_positions), 1);
            parfor local_number = 1:length(pending_positions)
                sample_position = pending_positions(local_number);
                sample_index = point_indices(sample_position);
                Y = double(source.Y_data(:, :, sample_index));
                G = double(source.G_data(:, :, sample_index));
                S = double(source.S_data(:, :, sample_index));
                B = double(source.B_data(:, :, sample_index));
                sigma2 = double(source.sigma2_data(sample_index));
                split_seed = split_seed_base + P * 100000 + sample_index;
                [G_hat_HS, chosen_alpha, chosen_rank, heldout_error] = ...
                    v12_select_and_refit(Y, S, B, sigma2, p_H, rank_list, ...
                    alpha_list, em_updates, training_fraction, split_seed);
                G_hat_GS = v12_biased_GS(Y, S, B, GS_max_iterations, GS_change_tolerance);
                G_hat_GD = v12_linearized_GD(Y, S, B, GS_max_iterations, ...
                    GS_change_tolerance, G0_variance, sample_index);
                block_HS_error(local_number) = norm(G_hat_HS - G, 'fro')^2;
                block_GS_error(local_number) = norm(G_hat_GS - G, 'fro')^2;
                block_GD_error(local_number) = norm(G_hat_GD - G, 'fro')^2;
                block_energy(local_number) = norm(G, 'fro')^2;
                block_alpha(local_number) = chosen_alpha;
                block_rank(local_number) = chosen_rank;
                block_pilot_error(local_number) = heldout_error;
                block_residual(local_number) = norm(abs(G_hat_HS * S + B) - Y, 'fro')^2;
            end
            HS_error(P_index, SNR_index, pending_positions) = reshape(block_HS_error, 1, 1, []);
            GS_error(P_index, SNR_index, pending_positions) = reshape(block_GS_error, 1, 1, []);
            GD_error(P_index, SNR_index, pending_positions) = reshape(block_GD_error, 1, 1, []);
            channel_energy(P_index, SNR_index, pending_positions) = reshape(block_energy, 1, 1, []);
            selected_alpha(P_index, SNR_index, pending_positions) = reshape(block_alpha, 1, 1, []);
            selected_rank(P_index, SNR_index, pending_positions) = reshape(block_rank, 1, 1, []);
            pilot_prediction_error(P_index, SNR_index, pending_positions) = reshape(block_pilot_error, 1, 1, []);
            HS_measurement_residual(P_index, SNR_index, pending_positions) = reshape(block_residual, 1, 1, []);
            completed(P_index, SNR_index, pending_positions) = true;
            save(checkpoint_file, 'P_list', 'SNR_dB', 'p_H', 'rank_list', ...
                'alpha_list', 'em_updates', 'training_fraction', 'samples_per_point', ...
                'sample_seed_base', 'split_seed_base', 'GS_max_iterations', ...
                'GS_change_tolerance', 'G0_variance', 'sample_indices', 'completed', ...
                'HS_error', 'GS_error', 'GD_error', 'channel_energy', ...
                'selected_alpha', 'selected_rank', 'pilot_prediction_error', ...
                'HS_measurement_residual');
            fprintf('P=%d, SNR=%d dB: %d/%d samples, %.1f s elapsed.\n', ...
                P, SNR_dB(SNR_index), sum(completed(P_index, SNR_index, :), 'all'), ...
                samples_per_point, toc(point_timer));
        end
    end
end

if ~all(completed, 'all') || ~all(isfinite(HS_error), 'all') || ...
        ~all(isfinite(GS_error), 'all') || ~all(isfinite(GD_error), 'all') || ...
        ~all(isfinite(channel_energy), 'all') || ~all(isfinite(selected_alpha), 'all')
    error('V12 has incomplete or nonfinite values.');
end

rows = zeros(length(P_list) * length(SNR_dB), 10);
row_number = 0;
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        energy = squeeze(channel_energy(P_index, SNR_index, :));
        hs_errors = squeeze(HS_error(P_index, SNR_index, :));
        gs_errors = squeeze(GS_error(P_index, SNR_index, :));
        gd_errors = squeeze(GD_error(P_index, SNR_index, :));
        alphas = squeeze(selected_alpha(P_index, SNR_index, :));
        ranks = squeeze(selected_rank(P_index, SNR_index, :));
        row_number = row_number + 1;
        rows(row_number, :) = [P_list(P_index), SNR_dB(SNR_index), ...
            10 * log10(sum(hs_errors) / sum(energy)), ...
            10 * log10(sum(gs_errors) / sum(energy)), ...
            10 * log10(sum(gd_errors) / sum(energy)), ...
            mean(alphas), mean(alphas == 0), mean(alphas == 1), ...
            mean(ranks), mean(pilot_prediction_error(P_index, SNR_index, :))];
    end
end
result_table = array2table(rows, 'VariableNames', {'P', 'SNR_dB', ...
    'HS_NMSE_dB', 'GS_NMSE_dB', 'GD_NMSE_dB', 'mean_selected_alpha', ...
    'fraction_alpha_zero', 'fraction_alpha_one', 'mean_selected_rank', ...
    'mean_heldout_pilot_error'});
save(result_file, 'result_table', 'P_list', 'SNR_dB', 'p_H', 'rank_list', ...
    'alpha_list', 'em_updates', 'training_fraction', 'samples_per_point', ...
    'sample_indices', 'completed', 'HS_error', 'GS_error', 'GD_error', ...
    'channel_energy', 'selected_alpha', 'selected_rank', ...
    'pilot_prediction_error', 'HS_measurement_residual');
writetable(result_table, table_file);
disp(result_table);
fprintf('V12 pilot-selected projection results saved to %s\n', result_file);

function [G_hat, selected_alpha, selected_rank, best_prediction_error] = ...
        v12_select_and_refit(Y, S, B, sigma2, p_H, rank_list, alpha_list, ...
        em_updates, training_fraction, split_seed)
P = size(S, 2);
rng(split_seed, 'twister');
pilot_order = randperm(P);
training_count = round(training_fraction * P);
training_pilots = pilot_order(1:training_count);
heldout_pilots = pilot_order(training_count + 1:end);
G_train = v12_em_estimate(Y(:, training_pilots), S(:, training_pilots), ...
    B(:, training_pilots), sigma2, em_updates);
candidate_errors = inf(length(rank_list), length(alpha_list));
for rank_index = 1:length(rank_list)
    G_projected = v12_project_channel(G_train, p_H, rank_list(rank_index));
    for alpha_index = 1:length(alpha_list)
        G_candidate = (1 - alpha_list(alpha_index)) * G_train + ...
            alpha_list(alpha_index) * G_projected;
        predicted = abs(G_candidate * S(:, heldout_pilots) + B(:, heldout_pilots));
        candidate_errors(rank_index, alpha_index) = ...
            norm(Y(:, heldout_pilots) - predicted, 'fro')^2;
    end
end
[best_prediction_error, best_index] = min(candidate_errors(:));
[best_rank_index, best_alpha_index] = ind2sub(size(candidate_errors), best_index);
selected_rank = rank_list(best_rank_index);
selected_alpha = alpha_list(best_alpha_index);
G_full = v12_em_estimate(Y, S, B, sigma2, em_updates);
G_projected_full = v12_project_channel(G_full, p_H, selected_rank);
G_hat = (1 - selected_alpha) * G_full + selected_alpha * G_projected_full;
end

function G_hat = v12_em_estimate(Y, S, B, sigma2, em_updates)
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
right_inverse = S' / (S * S');
for update_index = 1:em_updates
    field = G_hat * S + B;
    kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
    ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
    expected_field = Y .* exp(1j * angle(field)) .* ratio;
    G_hat = (expected_field - B) * right_inverse;
end
end

function G_projected = v12_project_channel(G, p_H, rank_value)
G_projected = zeros(size(G));
for user_index = 1:size(G, 2)
    channel_vector = G(:, user_index);
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
    G_projected(:, user_index) = reconstructed ./ copy_count;
end
end

function G_hat = v12_biased_GS(Y, S, B, maximum_iterations, tolerance)
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
    v = V(:, principal_index) / norm(V(:, principal_index));
    r = (abs(A * v).' * y_i) / norm(A * v)^2;
    G_hat(cell_index, :) = (r * exp(-1j * angle(v(K + 1))) * v(1:K)).';
end
right_inverse = S' / (S * S');
for iteration = 1:maximum_iterations
    previous = G_hat;
    G_hat = (Y .* exp(1j * angle(G_hat * S + B)) - B) * right_inverse;
    if norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps) < tolerance
        break;
    end
end
end

function G_hat = v12_linearized_GD(Y, S, B, maximum_iterations, tolerance, initial_variance, sample_index)
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
