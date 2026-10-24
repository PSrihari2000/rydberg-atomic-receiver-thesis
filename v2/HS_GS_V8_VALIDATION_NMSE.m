%% HS_GS_V8_VALIDATION_NMSE.m
% Tune Hankel pencil and Cadzow sweeps using channel NMSE on validation data.
% The update counts are frozen from V5. The selected setting is then scored
% on the independent test split. This is a development experiment.
clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
source_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

P_list = [10 30];
SNR_dB = -5:5:30;
N = 8;
K = 3;
RSR_dB = 25;
p_H_list = [2 3];
sweep_list = [0 1 2 4];
setting_p_H = repelem(p_H_list, length(sweep_list));
setting_sweeps = repmat(sweep_list, 1, length(p_H_list));
number_of_settings = length(setting_p_H);
selected_updates = [100 100 100 100 100 100 100 1000; 100 100 100 100 500 100 100 1000];
rank_selection_updates = 50;
training_fraction = 0.70;
random_seed_base = 910000;
maximum_validation_samples_per_snr = 100;
maximum_test_samples_per_snr = inf;

requested_workers = 6;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', requested_workers);
elseif pool.NumWorkers ~= requested_workers
    delete(pool);
    parpool('local', requested_workers);
end
pool = gcp('nocreate');
fprintf('HS-GS V8 validation NMSE experiment started with %d workers.\n', pool.NumWorkers);

checkpoint_file = fullfile(result_folder, 'HS_GS_V8_validation_NMSE_checkpoint.mat');
result_file = fullfile(result_folder, 'HS_GS_V8_validation_NMSE_results.mat');
validation_setting_nmse = nan(length(P_list), length(SNR_dB), number_of_settings);
completed_validation_settings = false(size(validation_setting_nmse));
selected_p_H = zeros(length(P_list), length(SNR_dB));
selected_sweeps = zeros(length(P_list), length(SNR_dB));
test_NMSE = nan(length(P_list), length(SNR_dB));
test_NMSE_dB = nan(length(P_list), length(SNR_dB));
test_mean_rank = nan(length(P_list), length(SNR_dB));
completed_test_points = false(length(P_list), length(SNR_dB));

if exist(checkpoint_file, 'file')
    checkpoint = load(checkpoint_file);
    matching_settings = isequal(checkpoint.P_list, P_list) && ...
        isequal(checkpoint.SNR_dB, SNR_dB) && ...
        isequal(checkpoint.setting_p_H, setting_p_H) && ...
        isequal(checkpoint.setting_sweeps, setting_sweeps) && ...
        isequal(checkpoint.selected_updates, selected_updates) && ...
        isequal(checkpoint.rank_selection_updates, rank_selection_updates) && ...
        isequal(checkpoint.training_fraction, training_fraction) && ...
        isequal(checkpoint.maximum_validation_samples_per_snr, maximum_validation_samples_per_snr) && ...
        isequal(checkpoint.maximum_test_samples_per_snr, maximum_test_samples_per_snr);
    if ~matching_settings
        error('V8 checkpoint settings differ from this script. Preserve the checkpoint and inspect both settings before resuming.');
    end
    validation_setting_nmse = checkpoint.validation_setting_nmse;
    completed_validation_settings = checkpoint.completed_validation_settings;
    selected_p_H = checkpoint.selected_p_H;
    selected_sweeps = checkpoint.selected_sweeps;
    test_NMSE = checkpoint.test_NMSE;
    test_NMSE_dB = checkpoint.test_NMSE_dB;
    test_mean_rank = checkpoint.test_mean_rank;
    completed_test_points = checkpoint.completed_test_points;
    fprintf('Resumed: %d/%d validation settings and %d/%d test points complete.\n', ...
        sum(completed_validation_settings, 'all'), numel(completed_validation_settings), ...
        sum(completed_test_points, 'all'), numel(completed_test_points));
end

for P_index = 1:length(P_list)
    P = P_list(P_index);
    validation_source = load(fullfile(source_folder, sprintf('validation_P%d.mat', P)));
    test_source = load(fullfile(source_folder, sprintf('test_P%d.mat', P)));
    for SNR_index = 1:length(SNR_dB)
        if completed_test_points(P_index, SNR_index)
            fprintf('Skipping completed point: P=%d, SNR=%d dB\n', P, SNR_dB(SNR_index));
            continue;
        end
        point_timer = tic;
        validation_samples = find(validation_source.SNR_dB_data(:) == SNR_dB(SNR_index));
        validation_samples = limit_samples(validation_samples, maximum_validation_samples_per_snr);
        if isempty(validation_samples)
            error('No validation samples for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        update_count = selected_updates(P_index, SNR_index);

        for setting_number = 1:number_of_settings
            if completed_validation_settings(P_index, SNR_index, setting_number)
                continue;
            end
            p_H = setting_p_H(setting_number);
            sweep_count = setting_sweeps(setting_number);
            rank_list = 1:min(N - p_H, p_H + 1);
            fprintf('Validating P=%d, SNR=%3d dB, p_H=%d, sweeps=%d, updates=%d\n', ...
                P, SNR_dB(SNR_index), p_H, sweep_count, update_count);

            % With zero sweeps, p_H does not enter the estimator.
            if sweep_count == 0 && p_H ~= p_H_list(1)
                validation_setting_nmse(P_index, SNR_index, setting_number) = ...
                    validation_setting_nmse(P_index, SNR_index, 1);
            else
                sample_squared_error = zeros(length(validation_samples), 1);
                sample_channel_energy = zeros(length(validation_samples), 1);
                parfor sample_number = 1:length(validation_samples)
                    sample_index = validation_samples(sample_number);
                    Y = double(validation_source.Y_data(:, :, sample_index));
                    G = double(validation_source.G_data(:, :, sample_index));
                    S = double(validation_source.S_data(:, :, sample_index));
                    B = double(validation_source.B_data(:, :, sample_index));
                    sigma2 = double(validation_source.sigma2_data(sample_index));
                    selected_rank = choose_rank(Y, S, B, sigma2, P, sample_index, ...
                        rank_list, rank_selection_updates, sweep_count, p_H, ...
                        training_fraction, random_seed_base);
                    G_hat = run_hs_gs(Y, S, B, sigma2, selected_rank, update_count, sweep_count, p_H);
                    sample_squared_error(sample_number) = norm(G_hat - G, 'fro')^2;
                    sample_channel_energy(sample_number) = norm(G, 'fro')^2;
                end
                validation_setting_nmse(P_index, SNR_index, setting_number) = ...
                    sum(sample_squared_error) / max(sum(sample_channel_energy), eps);
            end
            completed_validation_settings(P_index, SNR_index, setting_number) = true;
            fprintf('Validation NMSE = %.3f dB\n', ...
                10 * log10(validation_setting_nmse(P_index, SNR_index, setting_number)));
            save(checkpoint_file, 'validation_setting_nmse', 'completed_validation_settings', ...
                'selected_p_H', 'selected_sweeps', 'test_NMSE', 'test_NMSE_dB', ...
                'test_mean_rank', 'completed_test_points', 'setting_p_H', 'setting_sweeps', ...
                'selected_updates', 'rank_selection_updates', 'training_fraction', ...
                'maximum_validation_samples_per_snr', 'maximum_test_samples_per_snr', ...
                'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB');
        end

        scores = squeeze(validation_setting_nmse(P_index, SNR_index, :));
        if ~all(isfinite(scores))
            error('Nonfinite V8 validation NMSE for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
        [~, best_setting] = min(scores);
        p_H = setting_p_H(best_setting);
        sweep_count = setting_sweeps(best_setting);
        rank_list = 1:min(N - p_H, p_H + 1);
        selected_p_H(P_index, SNR_index) = p_H;
        selected_sweeps(P_index, SNR_index) = sweep_count;
        fprintf('Selected P=%d, SNR=%3d dB: p_H=%d, sweeps=%d, updates=%d\n', ...
            P, SNR_dB(SNR_index), p_H, sweep_count, update_count);

        test_samples = find(test_source.SNR_dB_data(:) == SNR_dB(SNR_index));
        test_samples = limit_samples(test_samples, maximum_test_samples_per_snr);
        if isempty(test_samples)
            error('No test samples for P=%d, SNR=%d dB.', P, SNR_dB(SNR_index));
        end
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
            selected_rank = choose_rank(Y, S, B, sigma2, P, sample_index, ...
                rank_list, rank_selection_updates, sweep_count, p_H, ...
                training_fraction, random_seed_base);
            G_hat = run_hs_gs(Y, S, B, sigma2, selected_rank, update_count, sweep_count, p_H);
            sample_squared_error(sample_number) = norm(G_hat - G, 'fro')^2;
            sample_channel_energy(sample_number) = norm(G, 'fro')^2;
            sample_selected_rank(sample_number) = selected_rank;
        end
        test_NMSE(P_index, SNR_index) = ...
            sum(sample_squared_error) / max(sum(sample_channel_energy), eps);
        test_NMSE_dB(P_index, SNR_index) = 10 * log10(test_NMSE(P_index, SNR_index));
        test_mean_rank(P_index, SNR_index) = mean(sample_selected_rank);
        completed_test_points(P_index, SNR_index) = true;
        save(checkpoint_file, 'validation_setting_nmse', 'completed_validation_settings', ...
            'selected_p_H', 'selected_sweeps', 'test_NMSE', 'test_NMSE_dB', ...
            'test_mean_rank', 'completed_test_points', 'setting_p_H', 'setting_sweeps', ...
            'selected_updates', 'rank_selection_updates', 'training_fraction', ...
            'maximum_validation_samples_per_snr', 'maximum_test_samples_per_snr', ...
            'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB');
        fprintf('Test P=%d, SNR=%3d dB: NMSE=%8.3f dB, mean rank=%.3f, elapsed=%.1f s\n', ...
            P, SNR_dB(SNR_index), test_NMSE_dB(P_index, SNR_index), ...
            test_mean_rank(P_index, SNR_index), toc(point_timer));
    end
end

save(result_file, 'validation_setting_nmse', 'selected_p_H', 'selected_sweeps', ...
    'test_NMSE', 'test_NMSE_dB', 'test_mean_rank', 'setting_p_H', 'setting_sweeps', ...
    'selected_updates', 'rank_selection_updates', 'training_fraction', ...
    'maximum_validation_samples_per_snr', 'maximum_test_samples_per_snr', ...
    'P_list', 'SNR_dB', 'N', 'K', 'RSR_dB');
fprintf('Saved V8 results to %s\n', result_file);

function selected_rank = choose_rank(Y, S, B, sigma2, P, sample_index, ...
    rank_list, rank_selection_updates, sweep_count, p_H, training_fraction, seed_base)
if sweep_count == 0
    selected_rank = rank_list(1);
    return;
end
rng(seed_base + 1000 * P + sample_index, 'twister');
pilot_order = randperm(P);
training_count = round(training_fraction * P);
training_pilots = pilot_order(1:training_count);
held_out_pilots = pilot_order(training_count + 1:P);
candidate_error = zeros(length(rank_list), 1);
for rank_index = 1:length(rank_list)
    G_candidate = run_hs_gs(Y(:, training_pilots), S(:, training_pilots), ...
        B(:, training_pilots), sigma2, rank_list(rank_index), ...
        rank_selection_updates, sweep_count, p_H);
    predicted_magnitude = abs(G_candidate * S(:, held_out_pilots) + B(:, held_out_pilots));
    candidate_error(rank_index) = norm(Y(:, held_out_pilots) - predicted_magnitude, 'fro')^2;
end
[~, best_rank_index] = min(candidate_error);
selected_rank = rank_list(best_rank_index);
end

function samples = limit_samples(samples, maximum_samples)
if isfinite(maximum_samples)
    samples = samples(1:min(length(samples), maximum_samples));
end
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
            H_rank = left_vectors(:, 1:rank_value) * ...
                singular_values(1:rank_value, 1:rank_value) * ...
                right_vectors(:, 1:rank_value)';
            reconstructed_channel = zeros(N, 1);
            copy_count = zeros(N, 1);
            for row_index = 1:H_rows
                for column_index = 1:H_columns
                    sample_index = row_index + column_index - 1;
                    reconstructed_channel(sample_index) = ...
                        reconstructed_channel(sample_index) + H_rank(row_index, column_index);
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
