%% HS_GS_V14_FIXED_RANK_DIAGNOSTIC.m
% Compare fixed Hankel ranks 2, 3, and 4 using the exact validation samples
% already scored by V13. This isolates rank choice from V13's per-sample rank
% selector. The script does not select settings for a final test.
%
% Run from MATLAB with:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V14_FIXED_RANK_DIAGNOSTIC

clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

v13_file = fullfile(result_folder, 'HS_GS_V13_paper_aligned_results.mat');
checkpoint_file = fullfile(result_folder, 'HS_GS_V14_fixed_rank_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V14_fixed_rank_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V14_fixed_rank_table.csv');
if ~exist(v13_file, 'file')
    error('V14 needs the completed V13 result file: %s', v13_file);
end
if exist(result_file, 'file')
    error('V14 results already exist at %s. Inspect them before rerunning.', result_file);
end

v13 = load(v13_file, 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
    'p_H', 'final_updates', 'cadzow_sweeps', 'samples_per_point', ...
    'sample_indices', 'completed', 'EM_error', 'GS_error', 'channel_energy');
P_list = v13.P_list;
SNR_dB = v13.SNR_dB;
N = v13.N;
K = v13.K;
RSR_dB = v13.RSR_dB;
p_H = 3;
rank_list = [2 3 4];
final_updates = 100;
cadzow_sweeps = 4;
samples_per_point = v13.samples_per_point;
samples_per_checkpoint = 20;
workers_if_starting_pool = 4;

if N ~= 8 || K ~= 3 || RSR_dB ~= 25 || p_H ~= v13.p_H || ...
        final_updates ~= v13.final_updates || cadzow_sweeps ~= v13.cadzow_sweeps || ...
        ~all(v13.completed, 'all')
    error('V13 results do not match the expected completed N=8, K=3, RSR=25 dB configuration.');
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('V14 fixed-rank diagnostic using %d workers.\n', pool.NumWorkers);
fprintf('Reuses V13 validation samples and saved EM-GS/biased-GS baseline errors.\n');
fprintf('Fixed ranks: %s; p_H=%d; updates=%d; Cadzow sweeps/update=%d.\n', ...
    mat2str(rank_list), p_H, final_updates, cadzow_sweeps);

sample_indices = v13.sample_indices;
completed = false(length(P_list), length(SNR_dB), samples_per_point);
fixed_rank_error = nan(length(P_list), length(SNR_dB), samples_per_point, length(rank_list));

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    matching_settings = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && isequal(checkpoint.p_H, p_H) && ...
        isequal(checkpoint.rank_list, rank_list) && ...
        isequal(checkpoint.final_updates, final_updates) && ...
        isequal(checkpoint.cadzow_sweeps, cadzow_sweeps) && ...
        isequal(checkpoint.samples_per_point, samples_per_point) && ...
        isequal(checkpoint.sample_indices, sample_indices);
    if ~matching_settings
        error('V14 checkpoint settings or sample indices differ. Preserve and inspect the checkpoint.');
    end
    completed = checkpoint.completed;
    fixed_rank_error = checkpoint.fixed_rank_error;
    fprintf('Resumed %d/%d completed samples.\n', sum(completed, 'all'), numel(completed));
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
        point_indices = squeeze(sample_indices(P_index, SNR_index, :));
        if any(point_indices < 1) || ~all(ismember(point_indices, candidate_indices))
            error('V13 sample indices do not match validation P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        point_timer = tic;
        fprintf('Starting fixed ranks: P=%d, SNR=%3d dB.\n', P, SNR_dB(SNR_index));

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
            block_error = zeros(length(pending_positions), length(rank_list));

            parfor local_index = 1:length(pending_positions)
                Y = double(block_Y_data(:, :, local_index));
                G_true = double(block_G_data(:, :, local_index));
                S = double(block_S_data(:, :, local_index));
                B = double(block_B_data(:, :, local_index));
                sigma2 = double(block_sigma2_data(local_index));
                sample_errors = zeros(1, length(rank_list));
                for rank_index = 1:length(rank_list)
                    G_hat = v14_interleaved_em_gs(Y, S, B, sigma2, ...
                        final_updates, cadzow_sweeps, rank_list(rank_index), p_H);
                    sample_errors(rank_index) = norm(G_hat - G_true, 'fro')^2;
                end
                block_error(local_index, :) = sample_errors;
            end

            fixed_rank_error(P_index, SNR_index, pending_positions, :) = ...
                reshape(block_error, 1, 1, length(pending_positions), length(rank_list));
            completed(P_index, SNR_index, pending_positions) = true;
            save(checkpoint_file, 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
                'p_H', 'rank_list', 'final_updates', 'cadzow_sweeps', ...
                'samples_per_point', 'sample_indices', 'completed', 'fixed_rank_error');
            fprintf('P=%d, SNR=%3d dB: %d/%d samples, %.1f s elapsed.\n', ...
                P, SNR_dB(SNR_index), sum(completed(P_index, SNR_index, :), 'all'), ...
                samples_per_point, toc(point_timer));
        end
    end
end

if ~all(completed, 'all') || ~all(isfinite(fixed_rank_error), 'all')
    error('V14 is incomplete or contains nonfinite values. Checkpoint preserved.');
end

number_of_rows = length(P_list) * length(SNR_dB) * length(rank_list);
rows = zeros(number_of_rows, 9);
row_number = 0;
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        energy = squeeze(v13.channel_energy(P_index, SNR_index, :));
        em_error = squeeze(v13.EM_error(P_index, SNR_index, :));
        gs_error = squeeze(v13.GS_error(P_index, SNR_index, :));
        for rank_index = 1:length(rank_list)
            hs_error = squeeze(fixed_rank_error(P_index, SNR_index, :, rank_index));
            row_number = row_number + 1;
            hs_nmse_db = 10 * log10(sum(hs_error) / sum(energy));
            em_nmse_db = 10 * log10(sum(em_error) / sum(energy));
            gs_nmse_db = 10 * log10(sum(gs_error) / sum(energy));
            rows(row_number, :) = [P_list(P_index), SNR_dB(SNR_index), ...
                rank_list(rank_index), hs_nmse_db, em_nmse_db, gs_nmse_db, ...
                em_nmse_db - hs_nmse_db, gs_nmse_db - hs_nmse_db, ...
                mean(hs_error ./ max(energy, realmin))];
        end
    end
end

result_table = array2table(rows, 'VariableNames', {'P', 'SNR_dB', 'fixed_rank', ...
    'HS_NMSE_dB', 'EM_GS_NMSE_dB', 'biased_GS_NMSE_dB', ...
    'HS_gain_over_EM_GS_dB', 'HS_gain_over_biased_GS_dB', ...
    'mean_per_sample_linear_NMSE'});
save(result_file, 'result_table', 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
    'p_H', 'rank_list', 'final_updates', 'cadzow_sweeps', 'samples_per_point', ...
    'sample_indices', 'completed', 'fixed_rank_error');
writetable(result_table, table_file);
disp(result_table);
fprintf('V14 fixed-rank results saved to %s\n', result_file);

function G_hat = v14_interleaved_em_gs(Y, S, B, sigma2, number_of_updates, sweeps, rank_value, p_H)
G_hat = v14_spectral_initialize(Y, S, B);
right_inverse = S' / (S * S');
for update_index = 1:number_of_updates
    field = G_hat * S + B;
    kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
    bessel_ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
    expected_field = Y .* exp(1j * angle(field)) .* bessel_ratio;
    G_hat = (expected_field - B) * right_inverse;
    G_hat = v14_project_channel(G_hat, p_H, rank_value, sweeps);
end
end

function G_hat = v14_spectral_initialize(Y, S, B)
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

function G_projected = v14_project_channel(G, p_H, rank_value, number_of_sweeps)
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
