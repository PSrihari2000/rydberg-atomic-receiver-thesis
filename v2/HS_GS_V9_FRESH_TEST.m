%% HS_GS_V9_FRESH_TEST.m
% Evaluate the frozen V9 rule on newly generated paired test samples.
% Save sample-level errors and paired bootstrap intervals for HS-GS gains.
clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder, 'results');
freeze_file = fullfile(result_folder, 'HS_GS_V9_frozen_rule.mat');
fresh_folder = fullfile(result_folder, 'HS_GS_V9_fresh_test_RSR25dB');
checkpoint_file = fullfile(result_folder, 'HS_GS_V9_fresh_test_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V9_fresh_test_results.mat');
if ~exist(freeze_file, 'file')
    error('Run HS_GS_V9_VALIDATE_FREEZE first.');
end
frozen = load(freeze_file);
P_list = frozen.P_list;
SNR_dB = frozen.SNR_dB;
selected_p_H = frozen.selected_p_H;
selected_sweeps = frozen.selected_sweeps;
selected_updates = frozen.selected_updates;
rank_selection_updates = frozen.rank_selection_updates;
training_fraction = frozen.training_fraction;
rank_seed_base = frozen.rank_seed_base;
samples_per_snr = 500;
samples_per_checkpoint = 100;
bootstrap_repetitions = 2000;
GS_max_iterations = 1000;
GD_max_iterations = 1000;
change_tolerance = 1e-10;
G0_var = 0.1;

requested_workers = 6;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', requested_workers);
elseif pool.NumWorkers ~= requested_workers
    delete(pool);
    parpool('local', requested_workers);
end
pool = gcp('nocreate');
fprintf('HS-GS V9 fresh test started with %d workers.\n', pool.NumWorkers);

HS_sample_error = nan(length(P_list), length(SNR_dB), samples_per_snr);
GS_sample_error = nan(size(HS_sample_error));
GD_sample_error = nan(size(HS_sample_error));
sample_channel_energy = nan(size(HS_sample_error));
sample_selected_rank = nan(size(HS_sample_error));
completed_samples = false(size(HS_sample_error));
completed_points = false(length(P_list), length(SNR_dB));
HS_NMSE_dB = nan(size(completed_points));
GS_NMSE_dB = nan(size(completed_points));
GD_NMSE_dB = nan(size(completed_points));
gain_vs_GS_dB = nan(size(completed_points));
gain_vs_GD_dB = nan(size(completed_points));
gain_vs_GS_CI95_dB = nan(length(P_list), length(SNR_dB), 2);
gain_vs_GD_CI95_dB = nan(length(P_list), length(SNR_dB), 2);

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    matching_settings = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && ...
        isequal(checkpoint.selected_p_H, selected_p_H) && ...
        isequal(checkpoint.selected_sweeps, selected_sweeps) && ...
        isequal(checkpoint.selected_updates, selected_updates) && ...
        isequal(checkpoint.samples_per_snr, samples_per_snr) && ...
        isequal(checkpoint.bootstrap_repetitions, bootstrap_repetitions) && ...
        isequal(checkpoint.rank_selection_updates, rank_selection_updates) && ...
        isequal(checkpoint.training_fraction, training_fraction) && ...
        isequal(checkpoint.rank_seed_base, rank_seed_base) && ...
        isequal(checkpoint.GS_max_iterations, GS_max_iterations) && ...
        isequal(checkpoint.GD_max_iterations, GD_max_iterations) && ...
        isequal(checkpoint.change_tolerance, change_tolerance) && ...
        isequal(checkpoint.G0_var, G0_var);
    if ~matching_settings
        error('V9 fresh-test checkpoint settings differ from the frozen rule.');
    end
    HS_sample_error = checkpoint.HS_sample_error;
    GS_sample_error = checkpoint.GS_sample_error;
    GD_sample_error = checkpoint.GD_sample_error;
    sample_channel_energy = checkpoint.sample_channel_energy;
    sample_selected_rank = checkpoint.sample_selected_rank;
    completed_samples = checkpoint.completed_samples;
    completed_points = checkpoint.completed_points;
    HS_NMSE_dB = checkpoint.HS_NMSE_dB;
    GS_NMSE_dB = checkpoint.GS_NMSE_dB;
    GD_NMSE_dB = checkpoint.GD_NMSE_dB;
    gain_vs_GS_dB = checkpoint.gain_vs_GS_dB;
    gain_vs_GD_dB = checkpoint.gain_vs_GD_dB;
    gain_vs_GS_CI95_dB = checkpoint.gain_vs_GS_CI95_dB;
    gain_vs_GD_CI95_dB = checkpoint.gain_vs_GD_CI95_dB;
    fprintf('Resumed %d/%d samples and %d/%d P/SNR points.\n', ...
        sum(completed_samples, 'all'), numel(completed_samples), ...
        sum(completed_points, 'all'), numel(completed_points));
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    fresh_file = fullfile(fresh_folder, sprintf('fresh_test_P%d.mat', P));
    if ~exist(fresh_file, 'file')
        error('Missing %s. Run HS_GS_V9_GENERATE_FRESH_TEST first.', fresh_file);
    end
    source = load(fresh_file);
    if ~isequal(source.SNR_dB, SNR_dB) || source.RSR_dB ~= frozen.RSR_dB || ...
            source.d_over_lambda ~= frozen.d_over_lambda || ...
            source.generation_seed ~= frozen.fresh_seed_base + P || ...
            source.samples_per_snr ~= samples_per_snr
        error('Fresh test metadata does not match the frozen rule for P=%d.', P);
    end
    for SNR_index = 1:length(SNR_dB)
        if completed_points(P_index, SNR_index)
            fprintf('Skipping completed P=%d, SNR=%d dB.\n', P, SNR_dB(SNR_index));
            continue;
        end
        selected_samples = find(source.SNR_dB_data(:) == SNR_dB(SNR_index));
        if length(selected_samples) ~= samples_per_snr
            error('Expected %d fresh test samples for P=%d, SNR=%d dB.', ...
                samples_per_snr, P, SNR_dB(SNR_index));
        end
        p_H = selected_p_H(P_index, SNR_index);
        sweep_count = selected_sweeps(P_index, SNR_index);
        update_count = selected_updates(P_index, SNR_index);
        for block_start = 1:samples_per_checkpoint:samples_per_snr
            block_end = min(block_start + samples_per_checkpoint - 1, samples_per_snr);
            block_positions = block_start:block_end;
            already_done = squeeze(completed_samples(P_index, SNR_index, block_positions));
            pending_positions = block_positions(~already_done(:).');
            if isempty(pending_positions)
                continue;
            end
            block_HS_error = zeros(length(pending_positions), 1);
            block_GS_error = zeros(length(pending_positions), 1);
            block_GD_error = zeros(length(pending_positions), 1);
            block_energy = zeros(length(pending_positions), 1);
            block_rank = zeros(length(pending_positions), 1);
            parfor local_number = 1:length(pending_positions)
                sample_position = pending_positions(local_number);
                sample_index = selected_samples(sample_position);
                Y = double(source.Y_data(:, :, sample_index));
                G = double(source.G_data(:, :, sample_index));
                S = double(source.S_data(:, :, sample_index));
                B = double(source.B_data(:, :, sample_index));
                sigma2 = double(source.sigma2_data(sample_index));
                [G_hat_HS, selected_rank] = hs_gs_v9_sample(Y, S, B, sigma2, ...
                    P, sample_index, p_H, sweep_count, update_count, ...
                    rank_selection_updates, training_fraction, rank_seed_base);
                G_hat_GS = run_GS(Y, S, B, GS_max_iterations, change_tolerance);
                G_hat_GD = run_GD(Y, S, B, GD_max_iterations, ...
                    change_tolerance, G0_var, sample_index);
                block_HS_error(local_number) = norm(G_hat_HS - G, 'fro')^2;
                block_GS_error(local_number) = norm(G_hat_GS - G, 'fro')^2;
                block_GD_error(local_number) = norm(G_hat_GD - G, 'fro')^2;
                block_energy(local_number) = norm(G, 'fro')^2;
                block_rank(local_number) = selected_rank;
            end
            HS_sample_error(P_index, SNR_index, pending_positions) = reshape(block_HS_error, 1, 1, []);
            GS_sample_error(P_index, SNR_index, pending_positions) = reshape(block_GS_error, 1, 1, []);
            GD_sample_error(P_index, SNR_index, pending_positions) = reshape(block_GD_error, 1, 1, []);
            sample_channel_energy(P_index, SNR_index, pending_positions) = reshape(block_energy, 1, 1, []);
            sample_selected_rank(P_index, SNR_index, pending_positions) = reshape(block_rank, 1, 1, []);
            completed_samples(P_index, SNR_index, pending_positions) = true;
            save(checkpoint_file, 'HS_sample_error', 'GS_sample_error', 'GD_sample_error', ...
                'sample_channel_energy', 'sample_selected_rank', 'completed_samples', ...
                'completed_points', 'HS_NMSE_dB', 'GS_NMSE_dB', 'GD_NMSE_dB', ...
                'gain_vs_GS_dB', 'gain_vs_GD_dB', 'gain_vs_GS_CI95_dB', ...
                'gain_vs_GD_CI95_dB', 'selected_p_H', 'selected_sweeps', ...
                'selected_updates', 'P_list', 'SNR_dB', 'samples_per_snr', ...
                'bootstrap_repetitions', 'rank_selection_updates', ...
                'training_fraction', 'rank_seed_base', 'GS_max_iterations', ...
                'GD_max_iterations', 'change_tolerance', 'G0_var');
            fprintf('P=%d, SNR=%3d dB: completed %d/%d paired samples.\n', ...
                P, SNR_dB(SNR_index), sum(completed_samples(P_index, SNR_index, :), 'all'), samples_per_snr);
        end

        hs_error = squeeze(HS_sample_error(P_index, SNR_index, :));
        gs_error = squeeze(GS_sample_error(P_index, SNR_index, :));
        gd_error = squeeze(GD_sample_error(P_index, SNR_index, :));
        energy = squeeze(sample_channel_energy(P_index, SNR_index, :));
        if ~all(isfinite(hs_error)) || ~all(isfinite(gs_error)) || ...
                ~all(isfinite(gd_error)) || ~all(isfinite(energy))
            error('Nonfinite sample result for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        HS_NMSE_dB(P_index, SNR_index) = 10 * log10(sum(hs_error) / sum(energy));
        GS_NMSE_dB(P_index, SNR_index) = 10 * log10(sum(gs_error) / sum(energy));
        GD_NMSE_dB(P_index, SNR_index) = 10 * log10(sum(gd_error) / sum(energy));
        gain_vs_GS_dB(P_index, SNR_index) = GS_NMSE_dB(P_index, SNR_index) - HS_NMSE_dB(P_index, SNR_index);
        gain_vs_GD_dB(P_index, SNR_index) = GD_NMSE_dB(P_index, SNR_index) - HS_NMSE_dB(P_index, SNR_index);
        rng(2026102499 + 100 * P + SNR_index, 'twister');
        bootstrap_GS_gain = zeros(bootstrap_repetitions, 1);
        bootstrap_GD_gain = zeros(bootstrap_repetitions, 1);
        for repetition = 1:bootstrap_repetitions
            resample = randi(samples_per_snr, samples_per_snr, 1);
            bootstrap_GS_gain(repetition) = 10 * log10(sum(gs_error(resample)) / sum(hs_error(resample)));
            bootstrap_GD_gain(repetition) = 10 * log10(sum(gd_error(resample)) / sum(hs_error(resample)));
        end
        bootstrap_GS_gain = sort(bootstrap_GS_gain);
        bootstrap_GD_gain = sort(bootstrap_GD_gain);
        lower_index = max(1, ceil(0.025 * bootstrap_repetitions));
        upper_index = min(bootstrap_repetitions, ceil(0.975 * bootstrap_repetitions));
        gain_vs_GS_CI95_dB(P_index, SNR_index, :) = ...
            reshape([bootstrap_GS_gain(lower_index), bootstrap_GS_gain(upper_index)], 1, 1, 2);
        gain_vs_GD_CI95_dB(P_index, SNR_index, :) = ...
            reshape([bootstrap_GD_gain(lower_index), bootstrap_GD_gain(upper_index)], 1, 1, 2);
        completed_points(P_index, SNR_index) = true;
        save(checkpoint_file, 'HS_sample_error', 'GS_sample_error', 'GD_sample_error', ...
            'sample_channel_energy', 'sample_selected_rank', 'completed_samples', ...
            'completed_points', 'HS_NMSE_dB', 'GS_NMSE_dB', 'GD_NMSE_dB', ...
            'gain_vs_GS_dB', 'gain_vs_GD_dB', 'gain_vs_GS_CI95_dB', ...
            'gain_vs_GD_CI95_dB', 'selected_p_H', 'selected_sweeps', ...
            'selected_updates', 'P_list', 'SNR_dB', 'samples_per_snr', ...
            'bootstrap_repetitions', 'rank_selection_updates', ...
            'training_fraction', 'rank_seed_base', 'GS_max_iterations', ...
            'GD_max_iterations', 'change_tolerance', 'G0_var');
        fprintf('P=%d, SNR=%3d dB: HS-GS %.3f, GS %.3f, GD %.3f dB; GS gain %.3f dB, 95%% CI [%.3f, %.3f].\n', ...
            P, SNR_dB(SNR_index), HS_NMSE_dB(P_index, SNR_index), ...
            GS_NMSE_dB(P_index, SNR_index), GD_NMSE_dB(P_index, SNR_index), ...
            gain_vs_GS_dB(P_index, SNR_index), ...
            gain_vs_GS_CI95_dB(P_index, SNR_index, 1), ...
            gain_vs_GS_CI95_dB(P_index, SNR_index, 2));
    end
end

save(result_file, 'HS_NMSE_dB', 'GS_NMSE_dB', 'GD_NMSE_dB', ...
    'gain_vs_GS_dB', 'gain_vs_GD_dB', 'gain_vs_GS_CI95_dB', ...
    'gain_vs_GD_CI95_dB', 'HS_sample_error', 'GS_sample_error', ...
    'GD_sample_error', 'sample_channel_energy', 'sample_selected_rank', ...
    'selected_p_H', 'selected_sweeps', 'selected_updates', ...
    'P_list', 'SNR_dB', 'samples_per_snr', 'bootstrap_repetitions', ...
    'frozen');
fprintf('Fresh paired results saved to %s\n', result_file);

function G_hat = run_GS(Y, S, B, maximum_iterations, tolerance)
N = size(Y, 1);
K = size(S, 1);
G_hat = zeros(N, K);
for i = 1:N
    y_i = Y(i, :).';
    b_i = B(i, :).';
    A_bar_i = [S.', b_i];
    M_i = A_bar_i' * diag(y_i) * A_bar_i;
    [V_i, D_i] = eig(M_i);
    [~, principal_index] = max(real(diag(D_i)));
    v_i = V_i(:, principal_index) / norm(V_i(:, principal_index));
    r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2;
    G_hat(i, :) = (r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K)).';
end
for iteration = 1:maximum_iterations
    previous = G_hat;
    G_hat = (Y .* exp(1j * angle(G_hat * S + B)) - B) * S' / (S * S');
    if norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps) < tolerance
        break;
    end
end
end

function G_hat = run_GD(Y, S, B, maximum_iterations, tolerance, initial_variance, sample_index)
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
