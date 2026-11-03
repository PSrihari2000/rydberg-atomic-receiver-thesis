%% Fresh V4 test comparison: classical estimators and the current frozen DL model
% Methods: GS, GD, EM-GS, validation-frozen hard-rank HS-GS, and the current 16-stage PI-DU-RACE model.
% P=10 uses the selected spectrum-adaptive model; P=30 uses its confirmed continuation.
% The independent V4 test split is used only for final evaluation.
% No training or validation sample files are loaded. The already-frozen
% validation rank table is read; no test-based tuning is performed.

clear;
clc;
close all;
rng(20261108, 'twister');

%% -------------------- FIXED V3 SIMULATION SCENARIO --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
AoA_deg_range = [-90 90];
p_H = 3;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
hard_rank_candidates = 1:4;
bootstrap_replicates = 2000;
number_of_baselines = 4;
method_names = {'GS','GD','EM-GS','Hard-rank HS-GS'};

%% -------------------- INPUT AND OUTPUT PATHS --------------------
script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90_FRESH_TEST_V4');
hard_rank_policy_file = fullfile(v3_folder, 'results', 'N8_FRESH_TEST_SIX_METHOD_GS_INIT_COMPARISON_RSR25DB', 'N8_hard_rank_validation_policy.csv');
spectrum_model_folder_P10 = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_SPECTRUM_ADAPTIVE_HANKEL_RSR25DB');
spectrum_model_folder_P30 = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_P30_SPECTRUM_CONTINUATION_BALANCED_RSR25DB');
output_folder = fullfile(v3_folder, 'results', 'N8_FRESH_TEST_V4_CURRENT_DL_FIVE_METHOD_COMPARISON_RSR25DB_RUN2');

if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it or choose a new name before running.');
end
if ~isfolder(scenario_folder) || ~isfile(hard_rank_policy_file) || ~isfolder(spectrum_model_folder_P10) || ~isfolder(spectrum_model_folder_P30)
    error('The fresh V4 test folder, frozen hard-rank policy, or selected model folder is missing.');
end
if ~isfile(fullfile(scenario_folder, 'test_P10.mat')) || ~isfile(fullfile(scenario_folder, 'test_P30.mat'))
    error('Fresh V4 test files for P=10 and P=30 must be generated first.');
end
if ~isfile(fullfile(spectrum_model_folder_P10, 'selected_model_P10.mat')) || ~isfile(fullfile(spectrum_model_folder_P30, 'selected_model_P30.mat'))
    error('A selected P=10 or P=30 current DL model is missing.');
end
mkdir(output_folder);

%% -------------------- LOAD FROZEN VALIDATION HARD-RANK POLICY --------------------
policy_table = readtable(hard_rank_policy_file);
if ~all(ismember({'P','SNR_dB','FrozenHardRank'}, policy_table.Properties.VariableNames)) || height(policy_table) ~= numel(P_list)*numel(SNR_grid)
    error('The frozen validation hard-rank policy is missing required P/SNR rows.');
end
frozen_hard_rank = zeros(numel(P_list), numel(SNR_grid));
for p_index = 1:numel(P_list)
    for snr_index = 1:numel(SNR_grid)
        policy_row = find(policy_table.P == P_list(p_index) & abs(policy_table.SNR_dB-SNR_grid(snr_index)) < 1e-9);
        if numel(policy_row) ~= 1
            error('Expected one validation-frozen hard rank for P=%d and SNR=%g dB.', P_list(p_index), SNR_grid(snr_index));
        end
        selected_rank = policy_table.FrozenHardRank(policy_row);
        if ~ismember(selected_rank, hard_rank_candidates)
            error('Invalid frozen rank for P=%d and SNR=%g dB.', P_list(p_index), SNR_grid(snr_index));
        end
        frozen_hard_rank(p_index, snr_index) = selected_rank;
    end
end

fprintf('Fresh independent V4 test comparison.\n');
fprintf('N=%d, K=%d, P=[%s], RSR=%g dB; %d test samples per SNR.\n', N, K, num2str(P_list), RSR_dB, 500);
fprintf('Methods: GS, GD, EM-GS, frozen hard-rank HS-GS, current frozen DL.\n');
fprintf('Hard ranks are read from the validation-frozen policy; no validation sample files are loaded.\n');
fprintf('Both selected DL checkpoints are frozen; test results are not used for tuning.\n');
fprintf('Outputs: %s\n', output_folder);

%% -------------------- RESULT STORAGE --------------------
number_of_conditions = numel(P_list)*numel(SNR_grid);
result_rows = zeros(number_of_conditions, 9);
test_error_by_P = cell(numel(P_list), 1);
test_channel_energy_by_P = cell(numel(P_list), 1);
test_snr_by_P = cell(numel(P_list), 1);
condition_indices_by_P_SNR = cell(numel(P_list), numel(SNR_grid));
row_index = 0;

%% -------------------- EVALUATE EACH PILOT COUNT --------------------
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    test_file = fullfile(scenario_folder, sprintf('test_P%d.mat', P));
    if P == 10
        spectrum_model_file = fullfile(spectrum_model_folder_P10, 'selected_model_P10.mat');
    else
        spectrum_model_file = fullfile(spectrum_model_folder_P30, 'selected_model_P30.mat');
    end

    test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'RSR_dB', 'AoA_deg_range', 'data_version', 'dataset_release', 'random_seed', 'samples_this_split_per_snr', 'split_name');
    spectrum_model = load(spectrum_model_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs', 'initialization_method');

    if test_data.N ~= N || test_data.K ~= K || test_data.P ~= P || test_data.RSR_dB ~= RSR_dB || ~isequal(double(test_data.AoA_deg_range), AoA_deg_range) || ~strcmp(test_data.data_version, 'V3_AoA_Uminus90to90') || ~strcmp(test_data.dataset_release, 'V3_FRESH_TEST_V4') || test_data.random_seed ~= 2026110301+P || test_data.samples_this_split_per_snr ~= 500 || ~strcmp(test_data.split_name, 'test') || test_data.number_of_samples ~= 4000
        error('Fresh V4 test metadata mismatch for P=%d.', P);
    end
    if spectrum_model.N ~= N || spectrum_model.K ~= K || spectrum_model.P ~= P || spectrum_model.RSR_dB ~= RSR_dB || spectrum_model.p_H ~= p_H || spectrum_model.number_of_stages ~= 16 || spectrum_model.parameters_per_stage ~= 15 || spectrum_model.number_of_gate_features ~= 3 || spectrum_model.number_of_gate_outputs ~= 3 || ~isequal(size(spectrum_model.best_parameters_numeric), [15 16]) || ~strcmp(spectrum_model.initialization_method, 'GS')
        error('Spectrum-adaptive DL model metadata mismatch for P=%d.', P);
    end

    number_of_test_samples = test_data.number_of_samples;
    test_error = zeros(number_of_test_samples, 5);
    test_channel_energy = zeros(number_of_test_samples, 1);
    fprintf('\nP=%d; V4 test samples=%d; selected DL epoch=%d.\n', P, number_of_test_samples, spectrum_model.best_epoch_saved);

    %% -------------------- PROCESS EVERY TEST OBSERVATION --------------------
    for sample_index = 1:number_of_test_samples
        Y = double(test_data.Y_data(:, :, sample_index));
        G_true = double(test_data.G_data(:, :, sample_index));
        S = double(test_data.S_data(:, :, sample_index));
        B = double(test_data.B_data(:, :, sample_index));
        sigma2 = double(test_data.sigma2_data(sample_index));
        current_SNR_dB = double(test_data.SNR_dB_data(sample_index));
        snr_index = find(SNR_grid == current_SNR_dB, 1);
        if isempty(snr_index)
            error('Unexpected test SNR for P=%d sample=%d.', P, sample_index);
        end
        right_inverse = S'/(S*S');

        %% Shared spectral initialization for GS, EM-GS, and hard-rank HS-GS
        G_spectral = zeros(N, K);
        for antenna_index = 1:N
            y = Y(antenna_index, :).';
            b = B(antenna_index, :).';
            augmented_matrix = [S.', b];
            spectral_matrix = augmented_matrix'*diag(y)*augmented_matrix;
            [spectral_vectors, spectral_values] = eig(spectral_matrix);
            [~, principal_index] = max(real(diag(spectral_values)));
            principal_vector = spectral_vectors(:, principal_index);
            principal_vector = principal_vector/max(norm(principal_vector), eps);
            spectral_scale = (abs(augmented_matrix*principal_vector).'*y)/max(norm(augmented_matrix*principal_vector)^2, eps);
            G_spectral(antenna_index, :) = (spectral_scale*exp(-1i*angle(principal_vector(K+1)))*principal_vector(1:K)).';
        end

        %% Algorithm 1: Biased GS
        G_GS = G_spectral;
        for update_index = 1:maximum_updates
            G_previous = G_GS;
            G_GS = (Y.*exp(1i*angle(G_GS*S+B))-B)*right_inverse;
            if norm(G_GS-G_previous, 'fro')^2/max(norm(G_GS, 'fro')^2, eps) < convergence_tolerance
                break;
            end
        end
        test_error(sample_index, 1) = sum(abs(G_GS-G_true).^2, 'all');

        %% Algorithm 2: Linearized GD with reproducible initialization
        random_seed = 881000000+1000000*P+sample_index;
        rng(mod(random_seed+720000, 2^32-1), 'twister');
        G_GD = sqrt(gd_initial_variance/2)*(randn(N, K)+1i*randn(N, K));
        phase_reference = exp(-1i*angle(B));
        step_size = 1/max(real(eig(S*S')));
        for update_index = 1:maximum_updates
            G_previous = G_GD;
            G_GD = G_GD+step_size*((Y-abs(B)-real(phase_reference.*(G_GD*S))).*conj(phase_reference))*S';
            if norm(G_GD-G_previous, 'fro')^2/max(norm(G_GD, 'fro')^2, eps) < convergence_tolerance
                break;
            end
        end
        test_error(sample_index, 2) = sum(abs(G_GD-G_true).^2, 'all');

        %% Algorithm 3: EM-GS
        G_EMGS = G_spectral;
        for update_index = 1:maximum_updates
            G_previous = G_EMGS;
            field = G_EMGS*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_EMGS = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            if norm(G_EMGS-G_previous, 'fro')^2/max(norm(G_EMGS, 'fro')^2, eps) < convergence_tolerance
                break;
            end
        end
        test_error(sample_index, 3) = sum(abs(G_EMGS-G_true).^2, 'all');

        %% Algorithm 4: Validation-frozen hard-rank HS-GS
        selected_rank = frozen_hard_rank(p_index, snr_index);
        G_HSGS = G_spectral;
        for update_index = 1:maximum_updates
            G_previous = G_HSGS;
            field = G_previous*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_EM_update = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            G_projected = zeros(N, K);
            for user_index = 1:K
                channel_vector = G_EM_update(:, user_index);
                hankel_matrix = zeros(N-p_H, p_H+1);
                for hankel_row = 1:N-p_H
                    hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+p_H).';
                end
                [U, Sigma, V] = svd(hankel_matrix, 'econ');
                hankel_matrix = U(:, 1:selected_rank)*Sigma(1:selected_rank, 1:selected_rank)*V(:, 1:selected_rank)';
                reconstructed_channel = zeros(N, 1);
                diagonal_counts = zeros(N, 1);
                for hankel_row = 1:N-p_H
                    for hankel_column = 1:p_H+1
                        antenna_index = hankel_row+hankel_column-1;
                        reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(hankel_row, hankel_column);
                        diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
                    end
                end
                G_projected(:, user_index) = reconstructed_channel./diagonal_counts;
            end
            G_HSGS = G_projected;
            if norm(G_HSGS-G_previous, 'fro')^2/max(norm(G_HSGS, 'fro')^2, eps) < convergence_tolerance
                break;
            end
        end
        test_error(sample_index, 4) = sum(abs(G_HSGS-G_true).^2, 'all');

        %% Algorithm 5: Current frozen spectrum-adaptive 16-stage PI-DU-RACE
        G_DL = G_GS;
        noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
        noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
        parameters = spectrum_model.best_parameters_numeric;
        for stage_index = 1:16
            G_previous = G_DL;
            field = G_DL*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            G_hankel = zeros(N, K);
            for user_index = 1:K
                channel_vector = G_emgs(:, user_index);
                hankel_matrix = zeros(N-p_H, p_H+1);
                for hankel_row = 1:N-p_H
                    hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+p_H).';
                end
                gram_matrix = hankel_matrix'*hankel_matrix;
                [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
                singular_values = sqrt(max(real(sum(eigenvalue_matrix, 2)), 0));
                safe_singular_values = max(singular_values, 1e-12);
                mean_singular_value = max(mean(safe_singular_values), 1e-12);
                relative_spectrum_feature = tanh(log(safe_singular_values./mean_singular_value));
                threshold_logits = parameters(1, stage_index)+parameters(2, stage_index)*noise_feature+parameters(3, stage_index)*relative_spectrum_feature;
                mode_thresholds = noise_scale*(2./(1+exp(-threshold_logits)));
                shrink_factors = max(1-mode_thresholds./safe_singular_values, 0);
                hankel_shrunk = hankel_matrix*(eigenvectors.*reshape(shrink_factors, 1, []))*eigenvectors';
                reconstructed_channel = zeros(N, 1);
                diagonal_counts = zeros(N, 1);
                for hankel_row = 1:N-p_H
                    for hankel_column = 1:p_H+1
                        antenna_index = hankel_row+hankel_column-1;
                        reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_shrunk(hankel_row, hankel_column);
                        diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
                    end
                end
                G_hankel(:, user_index) = reconstructed_channel./diagonal_counts;
            end
            emgs_change = tanh(log(1+sqrt(sum(abs(G_emgs-G_previous).^2, 'all')/max(sum(abs(G_previous).^2, 'all'), eps)))/3);
            hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_emgs).^2, 'all')/max(sum(abs(G_emgs).^2, 'all'), eps)))/3);
            gate_coefficients = reshape(parameters(4:15, stage_index), 4, 3);
            gate_logits = gate_coefficients(1, :).'*noise_feature+gate_coefficients(2, :).'*emgs_change+gate_coefficients(3, :).'*hankel_change+gate_coefficients(4, :).';
            gate_logits = gate_logits-max(gate_logits);
            gate_weights = exp(gate_logits)./sum(exp(gate_logits));
            G_DL = gate_weights(1)*G_previous+gate_weights(2)*G_emgs+gate_weights(3)*G_hankel;
        end
        test_error(sample_index, 5) = sum(abs(G_DL-G_true).^2, 'all');
        test_channel_energy(sample_index) = sum(abs(G_true).^2, 'all');

        if mod(sample_index, 200) == 0 || sample_index == number_of_test_samples
            fprintf('P=%d test samples processed: %d/%d.\n', P, sample_index, number_of_test_samples);
        end
    end

    if any(~isfinite(test_error), 'all') || any(test_channel_energy <= 0)
        error('Nonfinite test error or invalid true-channel energy for P=%d.', P);
    end
    test_error_by_P{p_index} = test_error;
    test_channel_energy_by_P{p_index} = test_channel_energy;
    test_snr_by_P{p_index} = double(test_data.SNR_dB_data(:));

    %% -------------------- SUMMARIZE NMSE BY SNR --------------------
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        sample_indices = find(test_snr_by_P{p_index} == current_SNR_dB);
        if numel(sample_indices) ~= 500
            error('Expected 500 test samples for P=%d and SNR=%g dB.', P, current_SNR_dB);
        end
        condition_indices_by_P_SNR{p_index, snr_index} = sample_indices;
        nmse_dB = 10*log10(sum(test_error(sample_indices, :), 1)./sum(test_channel_energy(sample_indices)));
        row_index = row_index+1;
        result_rows(row_index, :) = [P, current_SNR_dB, numel(sample_indices), frozen_hard_rank(p_index, snr_index), nmse_dB];
        fprintf('P=%d SNR=%g dB: GS %.3f, GD %.3f, EM-GS %.3f, hard-rank HS-GS %.3f, DL %.3f dB.\n', P, current_SNR_dB, nmse_dB(1), nmse_dB(2), nmse_dB(3), nmse_dB(4), nmse_dB(5));
    end
end

%% -------------------- PAIRED BOOTSTRAP CONFIDENCE INTERVALS --------------------
point_gain = zeros(number_of_conditions, number_of_baselines);
bootstrap_gain = zeros(bootstrap_replicates, number_of_conditions, number_of_baselines);
condition_index = 0;
for p_index = 1:numel(P_list)
    sample_error = test_error_by_P{p_index};
    for snr_index = 1:numel(SNR_grid)
        condition_index = condition_index+1;
        sample_indices = condition_indices_by_P_SNR{p_index, snr_index};
        baseline_sum = sum(sample_error(sample_indices, 1:number_of_baselines), 1);
        candidate_sum = sum(sample_error(sample_indices, 5));
        point_gain(condition_index, :) = 10*log10(baseline_sum./candidate_sum);
        rng(2026110800+10000*P_list(p_index)+100*snr_index, 'twister');
        for draw_index = 1:bootstrap_replicates
            resampled_positions = randi(numel(sample_indices), numel(sample_indices), 1);
            selected_samples = sample_indices(resampled_positions);
            bootstrap_baseline_sum = sum(sample_error(selected_samples, 1:number_of_baselines), 1);
            bootstrap_candidate_sum = sum(sample_error(selected_samples, 5));
            bootstrap_gain(draw_index, condition_index, :) = 10*log10(bootstrap_baseline_sum./bootstrap_candidate_sum);
        end
    end
end

pointwise_low = zeros(size(point_gain));
pointwise_high = zeros(size(point_gain));
for condition_index = 1:number_of_conditions
    for baseline_index = 1:number_of_baselines
        sorted_gain = sort(bootstrap_gain(:, condition_index, baseline_index));
        pointwise_low(condition_index, baseline_index) = sorted_gain(max(1, floor(0.025*bootstrap_replicates)));
        pointwise_high(condition_index, baseline_index) = sorted_gain(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)));
    end
end
maximum_absolute_deviation = zeros(bootstrap_replicates, 1);
for draw_index = 1:bootstrap_replicates
    deviation = abs(squeeze(bootstrap_gain(draw_index, :, :))-point_gain);
    maximum_absolute_deviation(draw_index) = max(deviation, [], 'all');
end
maximum_absolute_deviation = sort(maximum_absolute_deviation);
simultaneous_margin_dB = maximum_absolute_deviation(min(bootstrap_replicates, ceil(0.95*bootstrap_replicates)));

%% -------------------- SAVE RESULTS --------------------
result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','TestSamples','FrozenHardRank','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','HardRank_HSGS_NMSE_dB','PI_DU_RACE_16Stage_NMSE_dB'});
ci_rows = cell(number_of_conditions*number_of_baselines, 1);
ci_values = zeros(number_of_conditions*number_of_baselines, 8);
ci_row = 0;
for p_index = 1:numel(P_list)
    for snr_index = 1:numel(SNR_grid)
        condition_index = (p_index-1)*numel(SNR_grid)+snr_index;
        result_row = condition_index;
        for baseline_index = 1:number_of_baselines
            ci_row = ci_row+1;
            ci_rows{ci_row} = method_names{baseline_index};
            ci_values(ci_row, :) = [P_list(p_index), SNR_grid(snr_index), 500, point_gain(condition_index, baseline_index), pointwise_low(condition_index, baseline_index), pointwise_high(condition_index, baseline_index), point_gain(condition_index, baseline_index)-simultaneous_margin_dB, point_gain(condition_index, baseline_index)+simultaneous_margin_dB];
        end
    end
end
paired_ci_table = table(ci_values(:, 1), ci_values(:, 2), ci_values(:, 3), ci_rows, ci_values(:, 4), ci_values(:, 5), ci_values(:, 6), ci_values(:, 7), ci_values(:, 8), 'VariableNames', {'P','SNR_dB','TestSamples','BaselineMethod','DL_Gain_vs_Baseline_dB','PointwiseCI95_Low_dB','PointwiseCI95_High_dB','SimultaneousCI95_Low_dB','SimultaneousCI95_High_dB'});

result_csv = fullfile(output_folder, 'N8_fresh_test_V4_five_method_nmse.csv');
result_mat = fullfile(output_folder, 'N8_fresh_test_V4_five_method_nmse.mat');
paired_ci_csv = fullfile(output_folder, 'N8_fresh_test_V4_DL_paired_CI.csv');
result_png = fullfile(output_folder, 'N8_fresh_test_V4_five_method_nmse.png');
result_fig = fullfile(output_folder, 'N8_fresh_test_V4_five_method_nmse.fig');
writetable(result_table, result_csv);
writetable(paired_ci_table, paired_ci_csv);
configuration = struct('N', N, 'K', K, 'P_list', P_list, 'SNR_grid', SNR_grid, 'RSR_dB', RSR_dB, 'AoA_deg_range', AoA_deg_range, 'p_H', p_H, 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'gd_initial_variance', gd_initial_variance, 'hard_rank_policy_file', hard_rank_policy_file, 'P10_DL_folder', spectrum_model_folder_P10, 'P30_DL_folder', spectrum_model_folder_P30, 'test_dataset_release', 'V3_FRESH_TEST_V4', 'simultaneous_margin_dB', simultaneous_margin_dB);
save(result_mat, 'result_table', 'paired_ci_table', 'configuration', 'frozen_hard_rank', 'test_error_by_P', 'test_channel_energy_by_P', 'test_snr_by_P', 'bootstrap_replicates', '-v7.3');

figure('Color', 'w', 'Position', [100 100 1250 540]);
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    selected_rows = result_table.P == P;
    subplot(1, 2, p_index);
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows), '-o', 'LineWidth', 1.5, 'DisplayName', 'GS');
    hold on;
    plot(SNR_grid, result_table.GD_NMSE_dB(selected_rows), '-s', 'LineWidth', 1.5, 'DisplayName', 'GD');
    plot(SNR_grid, result_table.EMGS_NMSE_dB(selected_rows), '-d', 'LineWidth', 1.5, 'DisplayName', 'EM-GS');
    plot(SNR_grid, result_table.HardRank_HSGS_NMSE_dB(selected_rows), '-^', 'LineWidth', 1.5, 'DisplayName', 'Hard-rank HS-GS');
    plot(SNR_grid, result_table.PI_DU_RACE_16Stage_NMSE_dB(selected_rows), '-p', 'LineWidth', 1.8, 'DisplayName', 'DL');
    grid on;
    box on;
    xlabel('SNR (dB)');
    ylabel('Test NMSE (dB)');
    title(sprintf('N=%d, K=%d, P=%d, RSR=%g dB', N, K, P, RSR_dB));
    xticks(SNR_grid);
    legend('Location', 'best');
end
sgtitle('Fresh independent V4 test comparison');
exportgraphics(gcf, result_png, 'Resolution', 300);
savefig(gcf, result_fig);
disp(result_table);
disp(paired_ci_table);
fprintf('\nFresh V4 five-method comparison complete; no tuning was performed on test data.\n');
fprintf('Simultaneous 95%% margin across all %d DL-versus-baseline contrasts = %.4f dB.\n', number_of_conditions*number_of_baselines, simultaneous_margin_dB);
fprintf('Positive gain favors DL over each baseline.\n');
fprintf('Simultaneous positive gains = %d of %d contrasts.\n', sum(paired_ci_table.SimultaneousCI95_Low_dB > 0), height(paired_ci_table));
fprintf('NMSE CSV: %s\nPaired CI CSV: %s\nMAT: %s\nFigure: %s\n', result_csv, paired_ci_csv, result_mat, result_png);
