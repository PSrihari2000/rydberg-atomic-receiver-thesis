%% HS_GS_V11_SOFT_FINAL_PROJECTION.m
% Validation diagnostic: run EM without repeated Hankel projections, then
% apply one final rank projection at several blending strengths.
% Uses the exact validation samples and selected ranks saved by V10.
clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
v10_result_file = fullfile(result_folder, 'HS_GS_V10_high_SNR_diagnostic_results.mat');
checkpoint_file = fullfile(result_folder, 'HS_GS_V11_soft_final_projection_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V11_soft_final_projection_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V11_soft_final_projection_table.csv');
if ~exist(v10_result_file, 'file')
    error('Run HS_GS_V10_HIGH_SNR_DIAGNOSTIC first.');
end
if exist(result_file, 'file')
    error('V11 results already exist at %s. Inspect them before rerunning.', result_file);
end
v10 = load(v10_result_file);
P_list = v10.P_list;
SNR_dB = v10.SNR_dB;
update_list = v10.update_list;
p_H = v10.p_H;
alpha_list = [0 0.10 0.25 0.50 1.00];
samples_per_point = v10.samples_per_point;
selected_sample_indices = v10.selected_sample_indices;
selected_rank = v10.selected_rank;

requested_workers = 6;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', requested_workers);
    pool = gcp('nocreate');
end
fprintf('V11 soft final-projection validation using %d workers.\n', pool.NumWorkers);
fprintf('Alpha values: %s (alpha=0 is EM only; alpha=1 is full projection).\n', ...
    mat2str(alpha_list));

samples_per_checkpoint = 20;
completed_samples = false(length(P_list), length(SNR_dB), samples_per_point);
EM_error = nan(length(P_list), length(SNR_dB), length(update_list), length(alpha_list), samples_per_point);
EM_residual = nan(size(EM_error));
channel_energy = v10.channel_energy;
GS_error = v10.GS_error;
GS_residual = v10.GS_residual;

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    matching_settings = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && ...
        isequal(checkpoint.update_list, update_list) && ...
        isequal(checkpoint.alpha_list, alpha_list) && ...
        isequal(checkpoint.p_H, p_H) && ...
        isequal(checkpoint.samples_per_point, samples_per_point) && ...
        isequal(checkpoint.selected_sample_indices, selected_sample_indices) && ...
        isequal(checkpoint.selected_rank, selected_rank);
    if ~matching_settings
        error('V11 checkpoint settings differ from V10 or the requested alpha values.');
    end
    completed_samples = checkpoint.completed_samples;
    EM_error = checkpoint.EM_error;
    EM_residual = checkpoint.EM_residual;
    fprintf('Resumed %d/%d validation samples.\n', sum(completed_samples, 'all'), numel(completed_samples));
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    source = load(fullfile(dataset_folder, sprintf('validation_P%d.mat', P)));
    for SNR_index = 1:length(SNR_dB)
        point_indices = squeeze(selected_sample_indices(P_index, SNR_index, :));
        if any(source.SNR_dB_data(point_indices) ~= SNR_dB(SNR_index))
            error('V10 sample indices do not match P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        point_timer = tic;
        fprintf('P=%d, SNR=%d dB: starting or resuming.\n', P, SNR_dB(SNR_index));
        for block_start = 1:samples_per_checkpoint:samples_per_point
            block_positions = block_start:min(block_start + samples_per_checkpoint - 1, samples_per_point);
            done_in_block = squeeze(completed_samples(P_index, SNR_index, block_positions));
            pending_positions = block_positions(~done_in_block(:).');
            if isempty(pending_positions)
                continue;
            end
            block_error = zeros(length(pending_positions), length(update_list), length(alpha_list));
            block_residual = zeros(size(block_error));
            parfor local_number = 1:length(pending_positions)
                sample_position = pending_positions(local_number);
                sample_index = point_indices(sample_position);
                Y = double(source.Y_data(:, :, sample_index));
                G = double(source.G_data(:, :, sample_index));
                S = double(source.S_data(:, :, sample_index));
                B = double(source.B_data(:, :, sample_index));
                sigma2 = double(source.sigma2_data(sample_index));
                [sample_error, sample_residual] = v11_em_then_soft_project( ...
                    Y, G, S, B, sigma2, update_list, p_H, ...
                    selected_rank(P_index, SNR_index, sample_position), alpha_list);
                block_error(local_number, :, :) = sample_error;
                block_residual(local_number, :, :) = sample_residual;
            end
            for update_index = 1:length(update_list)
                for alpha_index = 1:length(alpha_list)
                    EM_error(P_index, SNR_index, update_index, alpha_index, pending_positions) = ...
                        reshape(block_error(:, update_index, alpha_index), 1, 1, 1, 1, []);
                    EM_residual(P_index, SNR_index, update_index, alpha_index, pending_positions) = ...
                        reshape(block_residual(:, update_index, alpha_index), 1, 1, 1, 1, []);
                end
            end
            completed_samples(P_index, SNR_index, pending_positions) = true;
            save(checkpoint_file, 'P_list', 'SNR_dB', 'update_list', 'alpha_list', ...
                'p_H', 'samples_per_point', 'selected_sample_indices', 'selected_rank', ...
                'completed_samples', 'EM_error', 'EM_residual', 'channel_energy', ...
                'GS_error', 'GS_residual');
            fprintf('P=%d, SNR=%d dB: %d/%d samples, %.1f s elapsed.\n', ...
                P, SNR_dB(SNR_index), ...
                sum(completed_samples(P_index, SNR_index, :), 'all'), ...
                samples_per_point, toc(point_timer));
        end
    end
end

if ~all(completed_samples, 'all') || ~all(isfinite(EM_error), 'all') || ...
        ~all(isfinite(EM_residual), 'all')
    error('V11 contains incomplete or nonfinite validation results.');
end

rows = zeros(length(P_list) * length(SNR_dB) * length(update_list) * length(alpha_list), 10);
row_number = 0;
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        energy = squeeze(channel_energy(P_index, SNR_index, :));
        gs_errors = squeeze(GS_error(P_index, SNR_index, :));
        GS_NMSE_dB = 10 * log10(sum(gs_errors) / sum(energy));
        for update_index = 1:length(update_list)
            base_error = squeeze(EM_error(P_index, SNR_index, update_index, 1, :));
            for alpha_index = 1:length(alpha_list)
                current_error = squeeze(EM_error(P_index, SNR_index, update_index, alpha_index, :));
                current_residual = squeeze(EM_residual(P_index, SNR_index, update_index, alpha_index, :));
                NMSE_dB = 10 * log10(sum(current_error) / sum(energy));
                base_nmse_dB = 10 * log10(sum(base_error) / sum(energy));
                base_residual = squeeze(EM_residual(P_index, SNR_index, update_index, 1, :));
                row_number = row_number + 1;
                rows(row_number, :) = [P_list(P_index), SNR_dB(SNR_index), ...
                    update_list(update_index), alpha_list(alpha_index), ...
                    GS_NMSE_dB, base_nmse_dB, NMSE_dB, ...
                    base_nmse_dB - NMSE_dB, GS_NMSE_dB - NMSE_dB, ...
                    10 * log10(sum(current_residual) / sum(base_residual))];
            end
        end
    end
end
result_table = array2table(rows, 'VariableNames', {'P', 'SNR_dB', 'EM_updates', ...
    'projection_alpha', 'GS_NMSE_dB', 'EM_only_NMSE_dB', 'soft_projection_NMSE_dB', ...
    'gain_over_EM_only_dB', 'gain_over_GS_dB', 'residual_change_vs_EM_dB'});
save(result_file, 'result_table', 'P_list', 'SNR_dB', 'update_list', ...
    'alpha_list', 'p_H', 'samples_per_point', 'selected_sample_indices', ...
    'selected_rank', 'completed_samples', 'EM_error', 'EM_residual', ...
    'channel_energy', 'GS_error', 'GS_residual');
writetable(result_table, table_file);
disp(result_table);
fprintf('V11 results saved to %s\n', result_file);

function [sample_error, sample_residual] = v11_em_then_soft_project(Y, G, S, B, sigma2, update_list, p_H, rank_value, alpha_list)
N = size(Y, 1);
K = size(S, 1);
G_estimate = zeros(N, K);
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
    G_estimate(cell_index, :) = (r * exp(-1j * angle(v(K + 1))) * v(1:K)).';
end
right_inverse = S' / (S * S');
sample_error = zeros(length(update_list), length(alpha_list));
sample_residual = zeros(size(sample_error));
for update_number = 1:update_list(end)
    field = G_estimate * S + B;
    kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
    ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
    expected_field = Y .* exp(1j * angle(field)) .* ratio;
    G_estimate = (expected_field - B) * right_inverse;
    recorded_index = find(update_list == update_number, 1);
    if isempty(recorded_index)
        continue;
    end
    G_projected = zeros(size(G_estimate));
    for user_index = 1:K
        G_projected(:, user_index) = v11_one_Cadzow_sweep( ...
            G_estimate(:, user_index), p_H, rank_value);
    end
    for alpha_index = 1:length(alpha_list)
        G_candidate = (1 - alpha_list(alpha_index)) * G_estimate + ...
            alpha_list(alpha_index) * G_projected;
        sample_error(recorded_index, alpha_index) = norm(G_candidate - G, 'fro')^2;
        sample_residual(recorded_index, alpha_index) = norm(abs(G_candidate * S + B) - Y, 'fro')^2;
    end
end
end

function channel_vector = v11_one_Cadzow_sweep(channel_vector, p_H, rank_value)
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
