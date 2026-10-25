%% HS_GS_V15_EM_GS_CONVERGENCE_DIAGNOSTIC.m
% Check whether the 100-update EM-GS baseline in V13 stopped too early.
% EM-GS and biased GS use the same spectral initialization, paired samples,
% relative-change stopping tolerance, and maximum iteration count.
% This is a validation diagnostic; it does not use Hankel projection.
%
% Run from MATLAB with:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V15_EM_GS_CONVERGENCE_DIAGNOSTIC

clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

v13_file = fullfile(result_folder, 'HS_GS_V13_paper_aligned_results.mat');
checkpoint_file = fullfile(result_folder, 'HS_GS_V15_em_gs_convergence_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V15_em_gs_convergence_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V15_em_gs_convergence_table.csv');
if ~exist(v13_file, 'file')
    error('V15 needs the completed V13 result file: %s', v13_file);
end
if exist(result_file, 'file')
    error('V15 results already exist at %s. Inspect them before rerunning.', result_file);
end

v13 = load(v13_file, 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
    'sample_indices', 'completed', 'EM_error', 'GS_error', 'channel_energy');
P_list = v13.P_list;
SNR_dB = v13.SNR_dB;
N = v13.N;
K = v13.K;
RSR_dB = v13.RSR_dB;
samples_per_point = size(v13.sample_indices, 3);
samples_per_checkpoint = 20;
maximum_iterations = 1000;
relative_change_tolerance = 1e-10;
workers_if_starting_pool = 4;

if N ~= 8 || K ~= 3 || RSR_dB ~= 25 || ~all(v13.completed, 'all')
    error('V13 results are incomplete or do not match N=8, K=3, RSR=25 dB.');
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('V15 EM-GS convergence diagnostic using %d workers.\n', pool.NumWorkers);
fprintf('Uses the same V13 paired validation samples and V13 baseline errors.\n');
fprintf('Both methods: spectral initialization, max iterations=%d, relative-change tolerance=%.1e.\n', ...
    maximum_iterations, relative_change_tolerance);

sample_indices = v13.sample_indices;
completed = false(length(P_list), length(SNR_dB), samples_per_point);
EM_converged_error = nan(size(completed));
GS_converged_error = nan(size(completed));
EM_iterations = nan(size(completed));
GS_iterations = nan(size(completed));
EM_converged = false(size(completed));
GS_converged = false(size(completed));

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    matching_settings = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && isequal(checkpoint.N, N) && ...
        isequal(checkpoint.K, K) && isequal(checkpoint.RSR_dB, RSR_dB) && ...
        isequal(checkpoint.maximum_iterations, maximum_iterations) && ...
        isequal(checkpoint.relative_change_tolerance, relative_change_tolerance) && ...
        isequal(checkpoint.samples_per_point, samples_per_point) && ...
        isequal(checkpoint.sample_indices, sample_indices);
    if ~matching_settings
        error('V15 checkpoint settings or sample indices differ. Preserve and inspect the checkpoint.');
    end
    completed = checkpoint.completed;
    EM_converged_error = checkpoint.EM_converged_error;
    GS_converged_error = checkpoint.GS_converged_error;
    EM_iterations = checkpoint.EM_iterations;
    GS_iterations = checkpoint.GS_iterations;
    EM_converged = checkpoint.EM_converged;
    GS_converged = checkpoint.GS_converged;
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
            error('V13 indices do not match validation P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        point_timer = tic;
        fprintf('Starting convergence comparison: P=%d, SNR=%3d dB.\n', P, SNR_dB(SNR_index));

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
            block_error = zeros(length(pending_positions), 2);
            block_iterations = zeros(length(pending_positions), 2);
            block_converged = false(length(pending_positions), 2);

            parfor local_index = 1:length(pending_positions)
                Y = double(block_Y_data(:, :, local_index));
                G_true = double(block_G_data(:, :, local_index));
                S = double(block_S_data(:, :, local_index));
                B = double(block_B_data(:, :, local_index));
                sigma2 = double(block_sigma2_data(local_index));
                [G_hat_EM, number_EM_iterations, did_EM_converge] = v15_run_estimator( ...
                    Y, S, B, sigma2, true, maximum_iterations, relative_change_tolerance);
                [G_hat_GS, number_GS_iterations, did_GS_converge] = v15_run_estimator( ...
                    Y, S, B, sigma2, false, maximum_iterations, relative_change_tolerance);
                sample_errors = [norm(G_hat_EM - G_true, 'fro')^2, ...
                    norm(G_hat_GS - G_true, 'fro')^2];
                block_error(local_index, :) = sample_errors;
                block_iterations(local_index, :) = [number_EM_iterations, number_GS_iterations];
                block_converged(local_index, :) = [did_EM_converge, did_GS_converge];
            end

            EM_converged_error(P_index, SNR_index, pending_positions) = block_error(:, 1);
            GS_converged_error(P_index, SNR_index, pending_positions) = block_error(:, 2);
            EM_iterations(P_index, SNR_index, pending_positions) = block_iterations(:, 1);
            GS_iterations(P_index, SNR_index, pending_positions) = block_iterations(:, 2);
            EM_converged(P_index, SNR_index, pending_positions) = block_converged(:, 1);
            GS_converged(P_index, SNR_index, pending_positions) = block_converged(:, 2);
            completed(P_index, SNR_index, pending_positions) = true;

            save(checkpoint_file, 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
                'maximum_iterations', 'relative_change_tolerance', 'samples_per_point', ...
                'sample_indices', 'completed', 'EM_converged_error', ...
                'GS_converged_error', 'EM_iterations', 'GS_iterations', ...
                'EM_converged', 'GS_converged');
            fprintf('P=%d, SNR=%3d dB: %d/%d samples, %.1f s elapsed.\n', ...
                P, SNR_dB(SNR_index), sum(completed(P_index, SNR_index, :), 'all'), ...
                samples_per_point, toc(point_timer));
        end
    end
end

if ~all(completed, 'all') || ~all(isfinite(EM_converged_error), 'all') || ...
        ~all(isfinite(GS_converged_error), 'all') || ~all(isfinite(EM_iterations), 'all') || ...
        ~all(isfinite(GS_iterations), 'all')
    error('V15 is incomplete or contains nonfinite values. Checkpoint preserved.');
end

rows = zeros(length(P_list) * length(SNR_dB), 11);
row_number = 0;
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        energy = squeeze(v13.channel_energy(P_index, SNR_index, :));
        em100_error = squeeze(v13.EM_error(P_index, SNR_index, :));
        gs_v13_error = squeeze(v13.GS_error(P_index, SNR_index, :));
        em_conv_error = squeeze(EM_converged_error(P_index, SNR_index, :));
        gs_conv_error = squeeze(GS_converged_error(P_index, SNR_index, :));
        row_number = row_number + 1;
        em100_nmse_db = 10 * log10(sum(em100_error) / sum(energy));
        gs_v13_nmse_db = 10 * log10(sum(gs_v13_error) / sum(energy));
        em_conv_nmse_db = 10 * log10(sum(em_conv_error) / sum(energy));
        gs_conv_nmse_db = 10 * log10(sum(gs_conv_error) / sum(energy));
        rows(row_number, :) = [P_list(P_index), SNR_dB(SNR_index), ...
            em100_nmse_db, gs_v13_nmse_db, em_conv_nmse_db, gs_conv_nmse_db, ...
            gs_conv_nmse_db - em_conv_nmse_db, ...
            mean(squeeze(EM_iterations(P_index, SNR_index, :))), ...
            mean(squeeze(GS_iterations(P_index, SNR_index, :))), ...
            mean(squeeze(EM_converged(P_index, SNR_index, :))), ...
            mean(squeeze(GS_converged(P_index, SNR_index, :)))];
    end
end

result_table = array2table(rows, 'VariableNames', {'P', 'SNR_dB', ...
    'EM_GS_100_update_NMSE_dB', 'V13_biased_GS_NMSE_dB', ...
    'converged_EM_GS_NMSE_dB', 'converged_biased_GS_NMSE_dB', ...
    'GS_gain_over_EM_GS_dB', 'mean_EM_GS_iterations', ...
    'mean_GS_iterations', 'fraction_EM_GS_converged', 'fraction_GS_converged'});
save(result_file, 'result_table', 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
    'maximum_iterations', 'relative_change_tolerance', 'samples_per_point', ...
    'sample_indices', 'completed', 'EM_converged_error', 'GS_converged_error', ...
    'EM_iterations', 'GS_iterations', 'EM_converged', 'GS_converged');
writetable(result_table, table_file);
disp(result_table);
fprintf('V15 convergence results saved to %s\n', result_file);

function [G_hat, iteration_count, converged] = v15_run_estimator(Y, S, B, sigma2, use_EM, maximum_iterations, tolerance)
G_hat = v15_spectral_initialize(Y, S, B);
right_inverse = S' / (S * S');
converged = false;
iteration_count = maximum_iterations;
for iteration_index = 1:maximum_iterations
    previous = G_hat;
    field = G_hat * S + B;
    if use_EM
        kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
        bessel_ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
        expected_field = Y .* exp(1j * angle(field)) .* bessel_ratio;
        G_hat = (expected_field - B) * right_inverse;
    else
        G_hat = (Y .* exp(1j * angle(field)) - B) * right_inverse;
    end
    relative_change = norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps);
    if relative_change < tolerance
        iteration_count = iteration_index;
        converged = true;
        break;
    end
end
end

function G_hat = v15_spectral_initialize(Y, S, B)
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
