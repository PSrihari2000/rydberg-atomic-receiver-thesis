%% HS_GS_V16_FROZEN_POLICY_FRESH_TEST.m
% Freeze the V14 validation-selected Hankel rank rule, then evaluate it on
% newly generated paired samples. No channel truth is used to select a rank
% on these fresh samples; truth is used only for NMSE scoring.
%
% Four estimators use the same fresh observations:
%   HS-GS and EM-GS use 100 updates; HS-GS uses 4 Cadzow sweeps/update.
%   Biased GS and linearized GD use up to 1000 iterations and stop at 1e-10
%   relative change, matching the V13 baseline implementation.
%
% Run from MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V16_FROZEN_POLICY_FRESH_TEST

clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

validation_table_file = fullfile(result_folder, 'HS_GS_V14_fixed_rank_table.csv');
freeze_file = fullfile(result_folder, 'HS_GS_V16_frozen_rank_policy.mat');
checkpoint_file = fullfile(result_folder, 'HS_GS_V16_fresh_test_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V16_fresh_test_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V16_fresh_test_table.csv');
if ~isfile(validation_table_file)
    error('Missing completed V14 validation table: %s', validation_table_file);
end
if isfile(result_file)
    error('V16 results already exist. Inspect them before starting a new run: %s', result_file);
end

P_list = [10 30];
SNR_dB = -5:5:30;
N = 8;
K = 3;
RSR_dB = 25;
p_H = 3;
rank_candidates = [2 3 4];
number_of_updates = 100;
cadzow_sweeps = 4;
samples_per_point = 100;
samples_per_checkpoint = 10;
fresh_seed_base = 2026102516;
bootstrap_replicates = 2000;
workers_if_starting_pool = 4;
pilot_spacing = 0.5;
mu_eg = [0; 1785.9 * 1.602e-19 * 5.292e-11; 0];
hbar = 6.62607015e-34 / (2 * pi);

% Select one fixed rank for each P/SNR using only the V14 validation table.
v14_table = readtable(validation_table_file);
required_columns = {'P','SNR_dB','fixed_rank','HS_NMSE_dB'};
if ~all(ismember(required_columns, v14_table.Properties.VariableNames))
    error('V14 table is missing expected columns. Inspect %s.', validation_table_file);
end
selected_rank = zeros(length(P_list), length(SNR_dB));
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        rows = v14_table.P == P_list(P_index) & v14_table.SNR_dB == SNR_dB(SNR_index);
        candidates = v14_table(rows, :);
        if height(candidates) ~= length(rank_candidates) || ...
                ~all(ismember(candidates.fixed_rank, rank_candidates)) || ...
                any(~isfinite(candidates.HS_NMSE_dB))
            error('V14 validation table is incomplete/invalid at P=%d, SNR=%d dB.', ...
                P_list(P_index), SNR_dB(SNR_index));
        end
        [~, best_index] = min(candidates.HS_NMSE_dB);
        selected_rank(P_index, SNR_index) = candidates.fixed_rank(best_index);
    end
end

if isfile(freeze_file)
    frozen = load(freeze_file);
    if ~isequal(frozen.P_list, P_list) || ~isequal(frozen.SNR_dB, SNR_dB) || ...
            ~isequal(frozen.selected_rank, selected_rank) || ...
            frozen.number_of_updates ~= number_of_updates || ...
            frozen.cadzow_sweeps ~= cadzow_sweeps || ...
            frozen.samples_per_point ~= samples_per_point || ...
            frozen.fresh_seed_base ~= fresh_seed_base
        error('Existing V16 frozen policy differs from this script. Preserve it and inspect before proceeding.');
    end
    fprintf('Loaded the previously frozen validation-only rank policy.\n');
else
    policy_source = 'V14 fixed-rank validation table; minimum ratio-of-sums NMSE per P/SNR';
    save(freeze_file, 'P_list', 'SNR_dB', 'selected_rank', 'rank_candidates', ...
        'number_of_updates', 'cadzow_sweeps', 'samples_per_point', ...
        'fresh_seed_base', 'policy_source', 'validation_table_file');
    fprintf('Frozen V14 validation rank policy to %s\n', freeze_file);
end
disp(array2table([repelem(P_list(:), length(SNR_dB)), ...
    repmat(SNR_dB(:), length(P_list), 1), selected_rank(:)], ...
    'VariableNames', {'P','SNR_dB','Frozen_rank'}));

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('V16 fresh paired test using %d workers; samples per P/SNR = %d.\n', ...
    pool.NumWorkers, samples_per_point);
fprintf('Independent deterministic seed range starts at %u.\n', uint32(fresh_seed_base));

method_names = {'HS_GS','EM_GS','biased_GS','GD'};
completed = false(length(P_list), length(SNR_dB), samples_per_point);
sample_squared_error = nan(length(P_list), length(SNR_dB), samples_per_point, length(method_names));
channel_energy = nan(length(P_list), length(SNR_dB), samples_per_point);

if isfile(checkpoint_file)
    checkpoint = load(checkpoint_file);
    if ~isequal(checkpoint.P_list, P_list) || ~isequal(checkpoint.SNR_dB, SNR_dB) || ...
            ~isequal(checkpoint.selected_rank, selected_rank) || ...
            checkpoint.N ~= N || checkpoint.K ~= K || checkpoint.RSR_dB ~= RSR_dB || ...
            checkpoint.p_H ~= p_H || checkpoint.number_of_updates ~= number_of_updates || ...
            checkpoint.cadzow_sweeps ~= cadzow_sweeps || ...
            checkpoint.samples_per_point ~= samples_per_point || ...
            checkpoint.fresh_seed_base ~= fresh_seed_base
        error('Checkpoint settings do not match this V16 script. Preserve it and inspect.');
    end
    completed = checkpoint.completed;
    sample_squared_error = checkpoint.sample_squared_error;
    channel_energy = checkpoint.channel_energy;
    fprintf('Resuming checkpoint: %d/%d fresh samples are complete.\n', ...
        sum(completed, 'all'), numel(completed));
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    for SNR_index = 1:length(SNR_dB)
        point_timer = tic;
        fprintf('P=%d, SNR=%3d dB: starting or resuming.\n', P, SNR_dB(SNR_index));
        for block_start = 1:samples_per_checkpoint:samples_per_point
            block_positions = block_start:min(block_start + samples_per_checkpoint - 1, samples_per_point);
            block_complete = reshape(completed(P_index, SNR_index, block_positions), 1, []);
            pending_positions = block_positions(~block_complete);
            if isempty(pending_positions)
                continue;
            end
            block_errors = zeros(length(pending_positions), length(method_names));
            block_energy = zeros(length(pending_positions), 1);
            block_rank = selected_rank(P_index, SNR_index);

            parfor local_index = 1:length(pending_positions)
                fresh_index = pending_positions(local_index);
                sample_seed = uint32(fresh_seed_base + 1000000 * P + ...
                    10000 * SNR_index + fresh_index);
                [Y, G_true, S, B, sigma2] = v16_generate_sample(N, K, P, ...
                    SNR_dB(SNR_index), RSR_dB, pilot_spacing, mu_eg, hbar, sample_seed);
                G_hs = v16_em_hs(Y, S, B, sigma2, number_of_updates, ...
                    cadzow_sweeps, block_rank, p_H);
                G_em = v16_iterative(Y, S, B, sigma2, number_of_updates, true);
                G_gs = v16_biased_gs(Y, S, B, 1000, 1e-10);
                G_gd = v16_linearized_gd(Y, S, B, 1000, 1e-10, 0.1, sample_seed);
                block_errors(local_index, :) = [norm(G_hs-G_true,'fro')^2, ...
                    norm(G_em-G_true,'fro')^2, norm(G_gs-G_true,'fro')^2, ...
                    norm(G_gd-G_true,'fro')^2];
                block_energy(local_index) = norm(G_true, 'fro')^2;
            end

            sample_squared_error(P_index, SNR_index, pending_positions, :) = ...
                reshape(block_errors, 1, 1, length(pending_positions), length(method_names));
            channel_energy(P_index, SNR_index, pending_positions) = reshape(block_energy, 1, 1, []);
            completed(P_index, SNR_index, pending_positions) = true;
            save(checkpoint_file, 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
                'p_H', 'selected_rank', 'number_of_updates', 'cadzow_sweeps', ...
                'samples_per_point', 'fresh_seed_base', 'method_names', ...
                'completed', 'sample_squared_error', 'channel_energy');
            fprintf('  %d/%d samples complete; %.1f seconds elapsed for this point.\n', ...
                sum(completed(P_index, SNR_index, :), 'all'), samples_per_point, toc(point_timer));
        end
    end
end

if ~all(completed, 'all') || any(~isfinite(sample_squared_error), 'all') || ...
        any(~isfinite(channel_energy), 'all') || any(channel_energy <= 0, 'all')
    error('V16 fresh test is incomplete or contains invalid values. Checkpoint preserved.');
end

% Aggregate NMSE as a ratio of total squared error to total channel energy.
% Bootstrap resamples paired samples and reports the HS-GS gain over each baseline.
rows = zeros(length(P_list) * length(SNR_dB), 16);
row_number = 0;
rng(2026102517, 'twister');
for P_index = 1:length(P_list)
    for SNR_index = 1:length(SNR_dB)
        row_number = row_number + 1;
        energy = reshape(channel_energy(P_index, SNR_index, :), [], 1);
        errors = reshape(sample_squared_error(P_index, SNR_index, :, :), samples_per_point, length(method_names));
        nmse_db = 10 * log10(sum(errors, 1) ./ sum(energy));
        gains = zeros(1, 3);
        ci_low = zeros(1, 3);
        ci_high = zeros(1, 3);
        for baseline_index = 2:4
            bootstrap_gain = zeros(bootstrap_replicates, 1);
            gains(baseline_index - 1) = nmse_db(baseline_index) - nmse_db(1);
            for bootstrap_index = 1:bootstrap_replicates
                draw = randi(samples_per_point, samples_per_point, 1);
                hs_ratio = sum(errors(draw, 1)) / sum(energy(draw));
                baseline_ratio = sum(errors(draw, baseline_index)) / sum(energy(draw));
                bootstrap_gain(bootstrap_index) = 10 * log10(baseline_ratio / hs_ratio);
            end
            limits = prctile(bootstrap_gain, [2.5 97.5]);
            ci_low(baseline_index - 1) = limits(1);
            ci_high(baseline_index - 1) = limits(2);
        end
        rows(row_number, :) = [P_list(P_index), SNR_dB(SNR_index), ...
            selected_rank(P_index, SNR_index), nmse_db, gains(1), ci_low(1), ci_high(1), ...
            gains(2), ci_low(2), ci_high(2), gains(3), ci_low(3), ci_high(3)];
    end
end

result_table = array2table(rows, 'VariableNames', {'P','SNR_dB','Frozen_rank', ...
    'HS_GS_NMSE_dB','EM_GS_NMSE_dB','biased_GS_NMSE_dB','GD_NMSE_dB', ...
    'HS_gain_over_EM_GS_dB','HS_vs_EM_GS_CI_low_dB','HS_vs_EM_GS_CI_high_dB', ...
    'HS_gain_over_biased_GS_dB','HS_vs_biased_GS_CI_low_dB','HS_vs_biased_GS_CI_high_dB', ...
    'HS_gain_over_GD_dB','HS_vs_GD_CI_low_dB','HS_vs_GD_CI_high_dB'});
save(result_file, 'result_table', 'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB', ...
    'p_H', 'selected_rank', 'number_of_updates', 'cadzow_sweeps', ...
    'samples_per_point', 'fresh_seed_base', 'method_names', 'bootstrap_replicates', ...
    'completed', 'sample_squared_error', 'channel_energy');
writetable(result_table, table_file);
disp(result_table);
fprintf('V16 fresh paired-test results saved to %s\n', result_file);
fprintf('Frozen rank policy: %s\n', freeze_file);

function [Y, G, S, B, sigma2] = v16_generate_sample(N, K, P, current_SNR_dB, ...
    RSR_dB, d_over_lambda, mu_eg, hbar, sample_seed)
rng(double(sample_seed), 'twister');
G = zeros(N, K);
for user_index = 1:K
    number_of_paths = randi([3 7]);
    for path_index = 1:number_of_paths
        alpha = (randn + 1j * randn) / sqrt(2);
        theta = 2 * pi * rand;
        spatial_frequency = 2 * pi * d_over_lambda * cos(theta);
        polarization = sqrt(1 / 3) * randn(3, 1);
        for antenna_index = 1:N
            G(antenna_index, user_index) = G(antenna_index, user_index) + ...
                (mu_eg.' * polarization / hbar) * alpha * ...
                exp(-1j * (antenna_index - 1) * spatial_frequency);
        end
    end
end
S = (randn(K, P) + 1j * randn(K, P)) / sqrt(2);
alpha_b = sqrt(10 / 2) * (randn + 1j * randn);
theta_b = 2 * pi * rand;
spatial_frequency_b = 2 * pi * d_over_lambda * cos(theta_b);
polarization_b = sqrt(1 / 3) * randn(3, 1);
b = zeros(N, 1);
for antenna_index = 1:N
    b(antenna_index) = (mu_eg.' * polarization_b / hbar) * alpha_b * ...
        exp(-1j * (antenna_index - 1) * spatial_frequency_b);
end
B = b * ones(1, P);
average_single_user_power = 0;
for user_index = 1:K
    average_single_user_power = average_single_user_power + ...
        mean(abs(G(:, user_index) * S(user_index, :)).^2, 'all') / K;
end
B = B * sqrt(10^(RSR_dB / 10) * average_single_user_power / mean(abs(B).^2, 'all'));
sigma2 = mean(abs(G * S).^2, 'all') / 10^(current_SNR_dB / 10);
W = sqrt(sigma2 / 2) * (randn(N, P) + 1j * randn(N, P));
Y = abs(G * S + B + W);
end

function G_hat = v16_em_hs(Y, S, B, sigma2, updates, sweeps, rank_value, p_H)
G_hat = v16_spectral_initialize(Y, S, B);
right_inverse = S' / (S * S');
for update_index = 1:updates
    field = G_hat * S + B;
    kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
    bessel_ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
    expected_field = Y .* exp(1j * angle(field)) .* bessel_ratio;
    G_hat = (expected_field - B) * right_inverse;
    if rank_value < 4
        G_hat = v16_hankel_project(G_hat, p_H, rank_value, sweeps);
    end
end
end

function G_hat = v16_iterative(Y, S, B, sigma2, updates, use_EM)
G_hat = v16_spectral_initialize(Y, S, B);
right_inverse = S' / (S * S');
for update_index = 1:updates
    field = G_hat * S + B;
    if use_EM
        kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
        bessel_ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
        expected_field = Y .* exp(1j * angle(field)) .* bessel_ratio;
    else
        expected_field = Y .* exp(1j * angle(field));
    end
    G_hat = (expected_field - B) * right_inverse;
end
end

function G_hat = v16_biased_gs(Y, S, B, maximum_iterations, tolerance)
G_hat = v16_spectral_initialize(Y, S, B);
right_inverse = S' / (S * S');
for iteration_index = 1:maximum_iterations
    previous = G_hat;
    field = G_hat * S + B;
    G_hat = (Y .* exp(1j * angle(field)) - B) * right_inverse;
    relative_change = norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps);
    if relative_change < tolerance
        break;
    end
end
end

function G_hat = v16_linearized_gd(Y, S, B, maximum_iterations, tolerance, initial_variance, sample_seed)
rng(mod(double(sample_seed) + 720000, 2^32 - 1), 'twister');
N = size(Y, 1);
K = size(S, 1);
G_hat = sqrt(initial_variance / 2) * (randn(N, K) + 1j * randn(N, K));
Z = exp(-1j * angle(B));
step_size = 1 / max(real(eig(S * S')));
for iteration_index = 1:maximum_iterations
    previous = G_hat;
    G_hat = G_hat + step_size * ...
        ((Y - abs(B) - real(Z .* (G_hat * S))) .* conj(Z)) * S';
    relative_change = norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps);
    if relative_change < tolerance
        break;
    end
end
end

function G_hat = v16_spectral_initialize(Y, S, B)
N = size(Y, 1);
K = size(S, 1);
G_hat = zeros(N, K);
for cell_index = 1:N
    y_i = Y(cell_index, :).';
    b_i = B(cell_index, :).';
    augmented_pilots = [S.', b_i];
    matrix = augmented_pilots' * diag(y_i) * augmented_pilots;
    [eigenvectors, eigenvalues] = eig(matrix);
    [~, principal_index] = max(real(diag(eigenvalues)));
    vector = eigenvectors(:, principal_index);
    vector = vector / max(norm(vector), eps);
    scale = (abs(augmented_pilots * vector).' * y_i) / ...
        max(norm(augmented_pilots * vector)^2, eps);
    G_hat(cell_index, :) = (scale * exp(-1j * angle(vector(K + 1))) * vector(1:K)).';
end
end

function G_projected = v16_hankel_project(G, p_H, rank_value, sweeps)
G_projected = G;
for sweep_index = 1:sweeps
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
