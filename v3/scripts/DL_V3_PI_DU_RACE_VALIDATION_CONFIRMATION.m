%% V3 PI-DU-RACE held-out validation comparison
% Uses validation samples excluded from checkpoint selection, plus the frozen
% N=8 HS-GS policy. It does not load training or reserved test files.
% Methods: GS, GD, EM-GS, frozen HS-GS, and selected PI-DU-RACE model.

clear;
clc;
close all;
rng(20261029, 'twister');

%% -------------------- FIXED V3 SCENARIO --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
number_of_stages = 4;
validation_samples_per_SNR_for_checkpoint = 64;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
bootstrap_replicates = 2000;
method_names = {'GS','GD','EM-GS','HS-GS','PI-DU-RACE'};

script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
addpath(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
model_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_EMGS_HANKEL_PARFOR_RSR25DB_RUN2');
policy_file = fullfile(v3_folder, 'results', 'N8_CLASSICAL_COMPARISON_RSR25DB', 'N8_frozen_HSGS_validation_policy.mat');
output_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_VALIDATION_CONFIRMATION_RSR25DB_RUN2');
if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it or choose a new output-folder name before running.');
end
if ~isfile(policy_file)
    error('Frozen N=8 HS-GS validation policy is missing: %s', policy_file);
end
if ~isfile(fullfile(script_folder, 'V3_CLASSICAL_ESTIMATOR.m'))
    error('V3_CLASSICAL_ESTIMATOR.m must be in the scripts folder.');
end
if ~isfolder(model_folder)
    error('Selected V3 model folder was not found: %s', model_folder);
end
mkdir(output_folder);

policy_data = load(policy_file, 'configuration', 'frozen_multiplier');
configuration = policy_data.configuration;
if configuration.N ~= N || configuration.K ~= K || configuration.RSR_dB ~= RSR_dB || ~isequal(configuration.P_list, P_list) || ~isequal(configuration.SNR_dB, SNR_grid) || configuration.p_H ~= p_H || ~strcmp(configuration.data_version, 'V3_AoA_Uminus90to90')
    error('Frozen HS-GS policy does not match this V3 N=8 scenario.');
end

fprintf('V3 PI-DU-RACE held-out validation comparison.\n');
fprintf('N=%d, K=%d, P=[%s], RSR=%g dB, AoA=U(-90,90 deg).\n', N, K, num2str(P_list), RSR_dB);
fprintf('Validation samples used for checkpoint selection are excluded.\n');
fprintf('Methods: GS, GD, EM-GS, frozen HS-GS, and PI-DU-RACE.\n');
fprintf('Reserved test files are not loaded.\n');

%% -------------------- HANKEL MAP FOR THE UNFOLDED MODEL --------------------
number_of_hankel_rows = N-p_H;
number_of_hankel_columns = p_H+1;
hankel_map = zeros(number_of_hankel_rows*number_of_hankel_columns, N);
for row_index = 1:number_of_hankel_rows
    for column_index = 1:number_of_hankel_columns
        linear_index = row_index+(column_index-1)*number_of_hankel_rows;
        antenna_index = row_index+column_index-1;
        hankel_map(linear_index, antenna_index) = 1;
    end
end
hankel_diagonal_counts = sum(hankel_map, 1).';

%% -------------------- EVALUATE EACH PILOT COUNT --------------------
result_rows = zeros(numel(P_list)*numel(SNR_grid), 15);
row_index = 0;
all_errors = cell(numel(P_list), 1);
all_true_energy = cell(numel(P_list), 1);
all_snr_values = cell(numel(P_list), 1);

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    model_file = fullfile(model_folder, sprintf('selected_model_P%d.mat', P));
    if ~isfile(validation_file) || ~isfile(model_file)
        error('Missing validation data or selected model for P=%d.', P);
    end
    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'AoA_deg_range', 'data_version');
    model_data = load(model_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages');
    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ~isequal(double(validation_data.AoA_deg_range), [-90 90]) || ~strcmp(validation_data.data_version, 'V3_AoA_Uminus90to90')
        error('Validation metadata does not match the configured V3 scenario for P=%d.', P);
    end
    if model_data.N ~= N || model_data.K ~= K || model_data.P ~= P || model_data.RSR_dB ~= RSR_dB || model_data.p_H ~= p_H || model_data.number_of_stages ~= number_of_stages
        error('Selected model metadata does not match the configured scenario for P=%d.', P);
    end

    selected_parameters = model_data.best_parameters_numeric;
    sample_count = size(validation_data.Y_data, 3);
    sample_errors = nan(sample_count, numel(method_names));
    true_energy = nan(sample_count, 1);
    confirmation_mask = false(sample_count, 1);
    fprintf('\nP=%d: validation samples=%d; selected checkpoint epoch=%d.\n', P, sample_count, model_data.best_epoch_saved);

    for snr_index = 1:numel(SNR_grid)
        candidates = find(double(validation_data.SNR_dB_data) == SNR_grid(snr_index));
        if numel(candidates) <= validation_samples_per_SNR_for_checkpoint
            error('Not enough validation samples for SNR=%g dB and P=%d.', SNR_grid(snr_index), P);
        end
        selected_positions = round(linspace(1, numel(candidates), validation_samples_per_SNR_for_checkpoint));
        checkpoint_selection_indices = candidates(selected_positions);
        confirmation_mask(candidates) = true;
        confirmation_mask(checkpoint_selection_indices) = false;
    end

    confirmation_indices = find(confirmation_mask);
    fprintf('Held-out validation confirmation samples=%d.\n', numel(confirmation_indices));
    for confirmation_index = 1:numel(confirmation_indices)
        sample_index = confirmation_indices(confirmation_index);
        Y = double(validation_data.Y_data(:, :, sample_index));
        G_true = double(validation_data.G_data(:, :, sample_index));
        S = double(validation_data.S_data(:, :, sample_index));
        B = double(validation_data.B_data(:, :, sample_index));
        sigma2 = double(validation_data.sigma2_data(sample_index));
        snr_db = double(validation_data.SNR_dB_data(sample_index));
        snr_index = find(SNR_grid == snr_db, 1);
        random_seed = 831000000+1000000*P+10000*snr_index+sample_index;

        [G_estimate, ~, ~] = V3_CLASSICAL_ESTIMATOR('GS', Y, S, B, sigma2, p_H, 0, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
        sample_errors(sample_index, 1) = norm(G_estimate-G_true, 'fro')^2;
        [G_estimate, ~, ~] = V3_CLASSICAL_ESTIMATOR('GD', Y, S, B, sigma2, p_H, 0, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
        sample_errors(sample_index, 2) = norm(G_estimate-G_true, 'fro')^2;
        [G_estimate, ~, ~] = V3_CLASSICAL_ESTIMATOR('EM-GS', Y, S, B, sigma2, p_H, 0, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
        sample_errors(sample_index, 3) = norm(G_estimate-G_true, 'fro')^2;
        [G_estimate, ~, ~] = V3_CLASSICAL_ESTIMATOR('HS-GS', Y, S, B, sigma2, p_H, policy_data.frozen_multiplier(p_index, snr_index), maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
        sample_errors(sample_index, 4) = norm(G_estimate-G_true, 'fro')^2;

        right_inverse = S'/(S*S');
        G_estimate = (Y-abs(B))*right_inverse;
        noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
        for stage_index = 1:number_of_stages
            threshold_multiplier = 2/(1+exp(-selected_parameters(1, stage_index)));
            damping = 1/(1+exp(-selected_parameters(2, stage_index)));
            threshold = threshold_multiplier*noise_scale;
            field = G_estimate*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            G_hankel = zeros(N, K);
            for user_index = 1:K
                hankel_matrix = reshape(hankel_map*G_emgs(:, user_index), number_of_hankel_rows, number_of_hankel_columns);
                gram_matrix = hankel_matrix'*hankel_matrix;
                [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
                eigenvalues = real(sum(eigenvalue_matrix, 2));
                singular_values = sqrt(max(eigenvalues, 0));
                safe_singular_values = max(singular_values, 1e-12);
                shrink_factors = max(1-threshold./safe_singular_values, 0);
                scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
                hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
                G_hankel(:, user_index) = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
            end
            G_estimate = damping*G_hankel+(1-damping)*G_emgs;
        end
        sample_errors(sample_index, 5) = norm(G_estimate-G_true, 'fro')^2;
        true_energy(sample_index) = norm(G_true, 'fro')^2;

        if mod(confirmation_index, 100) == 0 || confirmation_index == numel(confirmation_indices)
            fprintf('P=%d validation samples completed: %d/%d.\n', P, confirmation_index, numel(confirmation_indices));
        end
    end

    if any(~isfinite(sample_errors(confirmation_indices, :)), 'all') || any(~isfinite(true_energy(confirmation_indices))) || any(true_energy(confirmation_indices) <= 0)
        error('P=%d validation confirmation produced invalid estimates.', P);
    end
    all_errors{p_index} = sample_errors;
    all_true_energy{p_index} = true_energy;
    all_snr_values{p_index} = double(validation_data.SNR_dB_data);

    for snr_index = 1:numel(SNR_grid)
        snr_db = SNR_grid(snr_index);
        snr_indices = confirmation_indices(double(validation_data.SNR_dB_data(confirmation_indices)) == snr_db);
        errors = sample_errors(snr_indices, :);
        energy = true_energy(snr_indices);
        nmse_db = 10*log10(sum(errors, 1)/sum(energy));
        bootstrap_gain = zeros(bootstrap_replicates, numel(method_names)-1);
        rng(941000+1000*P+snr_index, 'twister');
        for draw_index = 1:bootstrap_replicates
            resampled_indices = randi(numel(snr_indices), numel(snr_indices), 1);
            resampled_errors = errors(resampled_indices, :);
            for method_index = 1:numel(method_names)-1
                bootstrap_gain(draw_index, method_index) = 10*log10(sum(resampled_errors(:, method_index))/max(sum(resampled_errors(:, 5)), realmin));
            end
        end
        bootstrap_gain = sort(bootstrap_gain, 1);
        ci_low = bootstrap_gain(max(1, floor(0.025*bootstrap_replicates)), :);
        ci_high = bootstrap_gain(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)), :);
        row_index = row_index+1;
        result_rows(row_index, :) = [P, snr_db, numel(snr_indices), model_data.best_epoch_saved, nmse_db(1), nmse_db(2), nmse_db(3), nmse_db(4), nmse_db(5), 10*log10(sum(errors(:, 1))/sum(errors(:, 5))), ci_low(1), ci_high(1), 10*log10(sum(errors(:, 3))/sum(errors(:, 5))), ci_low(3), ci_high(3)];
        fprintf('P=%d SNR=%g dB model NMSE=%.3f dB; gain vs GS=%.3f dB; gain vs EM-GS=%.3f dB.\n', P, snr_db, nmse_db(5), result_rows(row_index, 10), result_rows(row_index, 13));
    end
end

result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','ConfirmationSamples','SelectedEpoch','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','HSGS_NMSE_dB','PIDURACE_NMSE_dB','PIDURACE_Gain_vs_GS_dB','CI_Low_Gain_vs_GS_dB','CI_High_Gain_vs_GS_dB','PIDURACE_Gain_vs_EMGS_dB','CI_Low_Gain_vs_EMGS_dB','CI_High_Gain_vs_EMGS_dB'});
csv_file = fullfile(output_folder, 'V3_PI_DU_RACE_validation_confirmation.csv');
mat_file = fullfile(output_folder, 'V3_PI_DU_RACE_validation_confirmation.mat');
writetable(result_table, csv_file);
save(mat_file, 'result_table', 'all_errors', 'all_true_energy', 'all_snr_values', 'P_list', 'SNR_grid', 'method_names', 'maximum_updates', 'convergence_tolerance', 'bootstrap_replicates', 'validation_samples_per_SNR_for_checkpoint', '-v7.3');
disp(result_table);
fprintf('\nValidation confirmation complete. No training or test files were loaded.\n');
fprintf('Results are in: %s\n', output_folder);
