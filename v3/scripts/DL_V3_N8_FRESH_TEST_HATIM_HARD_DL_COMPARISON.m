%% Fresh N=8 comparison: GS, GD, EM-GS, Hatim-style HS-GS, frozen hard-rank HS-GS, and DL
% The Hatim-style rank is selected per observation using held-out pilot magnitudes.
% Our hard rank is frozen from the existing validation-selected policy.
% PI-DU-RACE is the trained GS-initialized 16-stage model.
% All six methods use the same new test observations; truth is used only for scoring.

clear;
clc;
close all;

%% -------------------- FIXED V3 SCENARIO --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
AoA_deg_range = [-90 90];
p_H = 3;
rank_candidates = 1:4;
pilot_training_fraction = 0.70;
rank_selection_updates = 25;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;

%% -------------------- INPUT AND OUTPUT PATHS --------------------
script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90_FRESH_TEST_V2');
frozen_policy_file = fullfile(v3_folder, 'results', 'N8_FINAL_FIVE_METHOD_TEST_COMPARISON_RSR25DB', 'N8_hard_rank_validation_policy.csv');
dl_model_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_GS_INIT_RSR25DB');
output_folder = fullfile(v3_folder, 'results', 'N8_FRESH_TEST_HATIM_HARD_DL_COMPARISON_RSR25DB');

if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it or choose a new output folder before running.');
end
if ~isfolder(scenario_folder) || ~isfile(frozen_policy_file) || ~isfolder(dl_model_folder)
    error('The fresh V3 test data, frozen validation hard-rank policy, or GS-init DL model folder is missing.');
end
if ~isfile(fullfile(scenario_folder, 'test_P10.mat')) || ~isfile(fullfile(scenario_folder, 'test_P30.mat'))
    error('Reserved V3 test files for P=10 and P=30 are required.');
end
if ~isfile(fullfile(dl_model_folder, 'selected_model_P10.mat')) || ~isfile(fullfile(dl_model_folder, 'selected_model_P30.mat'))
    error('Selected GS-init 16-stage DL models for P=10 and P=30 are required.');
end
mkdir(output_folder);

%% -------------------- LOAD OUR FROZEN VALIDATION RANK POLICY --------------------
policy_table = readtable(frozen_policy_file);
if ~all(ismember({'P','SNR_dB','FrozenHardRank'}, policy_table.Properties.VariableNames))
    error('The frozen hard-rank policy CSV does not have the required columns.');
end
if height(policy_table) ~= numel(P_list)*numel(SNR_grid)
    error('The frozen hard-rank policy must contain one row for every P/SNR point.');
end
our_frozen_rank = zeros(numel(P_list), numel(SNR_grid));
for p_index = 1:numel(P_list)
    for snr_index = 1:numel(SNR_grid)
        policy_row = find(policy_table.P == P_list(p_index) & abs(policy_table.SNR_dB-SNR_grid(snr_index)) < 1e-9);
        if numel(policy_row) ~= 1
            error('Expected exactly one frozen rank for P=%d and SNR=%g dB.', P_list(p_index), SNR_grid(snr_index));
        end
        selected_rank = policy_table.FrozenHardRank(policy_row);
        if selected_rank < min(rank_candidates) || selected_rank > max(rank_candidates) || selected_rank ~= round(selected_rank)
            error('Invalid frozen rank in policy for P=%d and SNR=%g dB.', P_list(p_index), SNR_grid(snr_index));
        end
        our_frozen_rank(p_index, snr_index) = selected_rank;
    end
end

%% -------------------- RESULT STORAGE --------------------
number_of_rows = numel(P_list)*numel(SNR_grid);
result_rows = zeros(number_of_rows, 11);
test_error_by_P = cell(numel(P_list), 1);
test_channel_energy_by_P = cell(numel(P_list), 1);
test_snr_by_P = cell(numel(P_list), 1);
hatim_rank_by_P = cell(numel(P_list), 1);
row_index = 0;

fprintf('N=8 HS-GS rank-selection comparison; RSR=%g dB; AoA=Uniform(-90,90 deg).\n', RSR_dB);
fprintf('Methods: GS, GD, EM-GS, pilot-split Hatim-style HS-GS, validation-frozen hard-rank HS-GS, GS-init PI-DU-RACE.\n');
fprintf('Both hard-rank HS-GS methods use the same full-data EM-GS/Hankel update and convergence settings.\n');
fprintf('Hatim-style selection uses 70 percent of pilots for candidate estimates and 30 percent for magnitude prediction.\n');
fprintf('Test-channel truth is used only for final NMSE scoring.\n');

%% -------------------- EVALUATE EACH PILOT COUNT --------------------
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    test_file = fullfile(scenario_folder, sprintf('test_P%d.mat', P));
    model_file = fullfile(dl_model_folder, sprintf('selected_model_P%d.mat', P));
    test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'AoA_deg_range', 'data_version', 'random_seed', 'samples_this_split_per_snr', 'split_name');
    model_data = load(model_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs', 'initialization_method');
    if test_data.N ~= N || test_data.K ~= K || test_data.P ~= P || test_data.RSR_dB ~= RSR_dB || ~isequal(test_data.AoA_deg_range, AoA_deg_range) || ~strcmp(test_data.data_version, 'V3_AoA_Uminus90to90')
        error('Test-data metadata mismatch for P=%d.', P);
    end
    if test_data.random_seed ~= 2026110501+P || test_data.samples_this_split_per_snr ~= 500 || ~strcmp(test_data.split_name, 'test')
        error('Fresh V2 test metadata mismatch for P=%d.', P);
    end
    if model_data.N ~= N || model_data.K ~= K || model_data.P ~= P || model_data.RSR_dB ~= RSR_dB || model_data.p_H ~= p_H || model_data.number_of_stages ~= 16 || model_data.parameters_per_stage ~= 13 || model_data.number_of_gate_features ~= 3 || model_data.number_of_gate_outputs ~= 3 || ~isequal(size(model_data.best_parameters_numeric), [13 16]) || ~strcmp(model_data.initialization_method, 'GS')
        error('Selected GS-init 16-stage DL model metadata mismatch for P=%d.', P);
    end
    number_of_test_samples = size(test_data.Y_data, 3);
    if number_of_test_samples ~= 4000
        error('Expected 4000 reserved test samples for P=%d.', P);
    end

    test_error = zeros(number_of_test_samples, 6);
    test_channel_energy = zeros(number_of_test_samples, 1);
    selected_hatim_rank_by_sample = zeros(number_of_test_samples, 1);
    fprintf('\nP=%d; fresh V2 test samples=%d; GS-init DL epoch=%d.\n', P, number_of_test_samples, model_data.best_epoch_saved);

    %% -------------------- PROCESS EVERY TEST OBSERVATION --------------------
    for sample_index = 1:number_of_test_samples
        Y = zeros(N, P);
        G_true = zeros(N, K);
        S = zeros(K, P);
        B = zeros(N, P);
        Y(:) = test_data.Y_data(:, :, sample_index);
        G_true(:) = test_data.G_data(:, :, sample_index);
        S(:) = test_data.S_data(:, :, sample_index);
        B(:) = test_data.B_data(:, :, sample_index);
        sigma2 = test_data.sigma2_data(sample_index);
        current_SNR_dB = test_data.SNR_dB_data(sample_index);
        snr_index = find(SNR_grid == current_SNR_dB, 1);
        if isempty(snr_index)
            error('Unexpected test SNR value for P=%d, sample=%d.', P, sample_index);
        end
        right_inverse = S'/(S*S');

        %% Shared spectral initialization for full-pilot estimators, N x K
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

        %% Algorithm 1: Biased GS, N x K
        G_GS = G_spectral;
        for update_index = 1:maximum_updates
            G_previous = G_GS;
            G_GS = (Y.*exp(1i*angle(G_GS*S+B))-B)*right_inverse;
            if norm(G_GS-G_previous, 'fro')^2/max(norm(G_GS, 'fro')^2, eps) < convergence_tolerance
                break;
            end
        end
        test_error(sample_index, 1) = norm(G_GS-G_true, 'fro')^2;

        %% Algorithm 2: Linearized GD, reproducible sample-specific initialization
        random_seed = 861000000+1000000*P+sample_index;
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
        test_error(sample_index, 2) = norm(G_GD-G_true, 'fro')^2;

        %% Algorithm 3: EM-GS, N x K
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
        test_error(sample_index, 3) = norm(G_EMGS-G_true, 'fro')^2;

        %% Algorithm 4: Our validation-frozen hard-rank HS-GS, N x K
        our_rank = our_frozen_rank(p_index, snr_index);
        G_OurHSGS = G_spectral;
        for update_index = 1:maximum_updates
            G_previous = G_OurHSGS;
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
                hankel_matrix = U(:, 1:our_rank)*Sigma(1:our_rank, 1:our_rank)*V(:, 1:our_rank)';
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
            G_OurHSGS = G_projected;
            if norm(G_OurHSGS-G_previous, 'fro')^2/max(norm(G_OurHSGS, 'fro')^2, eps) < convergence_tolerance
                break;
            end
        end
        test_error(sample_index, 5) = norm(G_OurHSGS-G_true, 'fro')^2;

        %% Algorithm 5: Hatim-style pilot-split rank selection for this observation
        pilot_seed = 861000000+1000000*P+sample_index+345678;
        rng(mod(pilot_seed, 2^32-1), 'twister');
        pilot_order = randperm(P);
        number_of_training_pilots = round(pilot_training_fraction*P);
        training_pilot_indices = pilot_order(1:number_of_training_pilots);
        heldout_pilot_indices = pilot_order(number_of_training_pilots+1:P);
        Y_T = Y(:, training_pilot_indices);
        S_T = S(:, training_pilot_indices);
        B_T = B(:, training_pilot_indices);
        Y_V = Y(:, heldout_pilot_indices);
        S_V = S(:, heldout_pilot_indices);
        B_V = B(:, heldout_pilot_indices);
        right_inverse_T = S_T'/(S_T*S_T');
        rank_selection_error = zeros(1, numel(rank_candidates));

        for candidate_index = 1:numel(rank_candidates)
            candidate_rank = rank_candidates(candidate_index);
            G_candidate = zeros(N, K);
            for antenna_index = 1:N
                y = Y_T(antenna_index, :).';
                b = B_T(antenna_index, :).';
                augmented_matrix = [S_T.', b];
                spectral_matrix = augmented_matrix'*diag(y)*augmented_matrix;
                [spectral_vectors, spectral_values] = eig(spectral_matrix);
                [~, principal_index] = max(real(diag(spectral_values)));
                principal_vector = spectral_vectors(:, principal_index);
                principal_vector = principal_vector/max(norm(principal_vector), eps);
                spectral_scale = (abs(augmented_matrix*principal_vector).'*y)/max(norm(augmented_matrix*principal_vector)^2, eps);
                G_candidate(antenna_index, :) = (spectral_scale*exp(-1i*angle(principal_vector(K+1)))*principal_vector(1:K)).';
            end
            for rank_update = 1:rank_selection_updates
                field = G_candidate*S_T+B_T;
                kappa = 2/max(sigma2, eps)*Y_T.*abs(field);
                bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
                G_EM_update = (Y_T.*exp(1i*angle(field)).*bessel_ratio-B_T)*right_inverse_T;
                G_projected = zeros(N, K);
                for user_index = 1:K
                    channel_vector = G_EM_update(:, user_index);
                    hankel_matrix = zeros(N-p_H, p_H+1);
                    for hankel_row = 1:N-p_H
                        hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+p_H).';
                    end
                    [U, Sigma, V] = svd(hankel_matrix, 'econ');
                    hankel_matrix = U(:, 1:candidate_rank)*Sigma(1:candidate_rank, 1:candidate_rank)*V(:, 1:candidate_rank)';
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
                G_candidate = G_projected;
            end
            rank_selection_error(candidate_index) = norm(Y_V-abs(G_candidate*S_V+B_V), 'fro')^2;
        end
        if any(~isfinite(rank_selection_error))
            error('Pilot-split rank selection produced a nonfinite score for P=%d, sample=%d.', P, sample_index);
        end
        [~, best_rank_index] = min(rank_selection_error);
        hatim_rank = rank_candidates(best_rank_index);
        selected_hatim_rank_by_sample(sample_index) = hatim_rank;

        %% Rerun the selected Hatim-style rank on all P pilots, N x K
        G_HatimHSGS = G_spectral;
        for update_index = 1:maximum_updates
            G_previous = G_HatimHSGS;
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
                hankel_matrix = U(:, 1:hatim_rank)*Sigma(1:hatim_rank, 1:hatim_rank)*V(:, 1:hatim_rank)';
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
            G_HatimHSGS = G_projected;
            if norm(G_HatimHSGS-G_previous, 'fro')^2/max(norm(G_HatimHSGS, 'fro')^2, eps) < convergence_tolerance
                break;
            end
        end
        test_error(sample_index, 4) = norm(G_HatimHSGS-G_true, 'fro')^2;

        %% Algorithm 6: GS-initialized 16-stage PI-DU-RACE
        parameters = model_data.best_parameters_numeric;
        G_DL = G_GS;
        noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
        noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
        for stage_index = 1:16
            G_previous = G_DL;
            threshold_multiplier = 2/(1+exp(-parameters(1, stage_index)));
            stage_threshold = threshold_multiplier*noise_scale;
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
                eigenvalues = real(sum(eigenvalue_matrix, 2));
                singular_values = sqrt(max(eigenvalues, 0));
                safe_singular_values = max(singular_values, 1e-12);
                shrink_factors = max(1-stage_threshold./safe_singular_values, 0);
                scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
                hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
                hankel_map = zeros((N-p_H)*(p_H+1), N);
                for hankel_row = 1:N-p_H
                    for hankel_column = 1:p_H+1
                        linear_index = hankel_row+(hankel_column-1)*(N-p_H);
                        antenna_index = hankel_row+hankel_column-1;
                        hankel_map(linear_index, antenna_index) = 1;
                    end
                end
                diagonal_counts = sum(hankel_map, 1).';
                G_hankel(:, user_index) = (hankel_map'*hankel_shrunk(:))./diagonal_counts;
            end
            emgs_change = tanh(log(1+sqrt(sum(abs(G_emgs-G_previous).^2, 'all')/max(sum(abs(G_previous).^2, 'all'), eps)))/3);
            hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_emgs).^2, 'all')/max(sum(abs(G_emgs).^2, 'all'), eps)))/3);
            gate_coefficients = reshape(parameters(2:13, stage_index), 4, 3);
            gate_logits = gate_coefficients(1, :).'*noise_feature+gate_coefficients(2, :).'*emgs_change+gate_coefficients(3, :).'*hankel_change+gate_coefficients(4, :).';
            gate_logits = gate_logits-max(gate_logits);
            gate_weights = exp(gate_logits)./sum(exp(gate_logits));
            G_DL = gate_weights(1)*G_previous+gate_weights(2)*G_emgs+gate_weights(3)*G_hankel;
        end
        test_error(sample_index, 6) = norm(G_DL-G_true, 'fro')^2;
        test_channel_energy(sample_index) = norm(G_true, 'fro')^2;

        if mod(sample_index, 200) == 0 || sample_index == number_of_test_samples
            fprintf('P=%d test samples processed: %d/%d.\n', P, sample_index, number_of_test_samples);
        end
    end

    if any(~isfinite(test_error), 'all') || any(test_channel_energy <= 0)
        error('Nonfinite test error or invalid channel energy for P=%d.', P);
    end
    test_error_by_P{p_index} = test_error;
    test_channel_energy_by_P{p_index} = test_channel_energy;
    test_snr_by_P{p_index} = double(test_data.SNR_dB_data(:));
    hatim_rank_by_P{p_index} = selected_hatim_rank_by_sample;

    %% -------------------- SUMMARIZE PAIRED TEST NMSE BY SNR --------------------
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        test_indices = find(test_data.SNR_dB_data == current_SNR_dB);
        if numel(test_indices) ~= 500
            error('Expected 500 test samples at P=%d and SNR=%g dB.', P, current_SNR_dB);
        end
        nmse_dB = 10*log10(sum(test_error(test_indices, :), 1)./sum(test_channel_energy(test_indices)));
        mean_hatim_rank = mean(selected_hatim_rank_by_sample(test_indices));
        row_index = row_index+1;
        result_rows(row_index, :) = [P, current_SNR_dB, numel(test_indices), our_frozen_rank(p_index, snr_index), mean_hatim_rank, nmse_dB];
        fprintf('P=%d SNR=%g dB: GS %.3f, GD %.3f, EM-GS %.3f, Hatim-style HS-GS %.3f, frozen hard-rank HS-GS %.3f, DL %.3f dB; mean Hatim rank %.2f.\n', P, current_SNR_dB, nmse_dB(1), nmse_dB(2), nmse_dB(3), nmse_dB(4), nmse_dB(5), nmse_dB(6), mean_hatim_rank);
    end
end

%% -------------------- PAIRED TEST BOOTSTRAP FOR DL AGAINST EACH BASELINE --------------------
bootstrap_replicates = 2000;
baseline_method_names = {'GS','GD','EM-GS','Hatim-style HS-GS','Frozen hard-rank HS-GS'};
paired_ci_rows = cell(numel(P_list)*numel(SNR_grid)*5, 1);
paired_ci_values = zeros(numel(P_list)*numel(SNR_grid)*5, 6);
ci_row_index = 0;
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    sample_errors = test_error_by_P{p_index};
    sample_snr = test_snr_by_P{p_index};
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        snr_indices = find(sample_snr == current_SNR_dB);
        for baseline_index = 1:5
            bootstrap_gain = zeros(bootstrap_replicates, 1);
            rng(2026110600+10000*P+100*snr_index+baseline_index, 'twister');
            for draw_index = 1:bootstrap_replicates
                resampled_positions = randi(numel(snr_indices), numel(snr_indices), 1);
                selected_samples = snr_indices(resampled_positions);
                bootstrap_gain(draw_index) = 10*log10(sum(sample_errors(selected_samples, baseline_index))/sum(sample_errors(selected_samples, 6)));
            end
            bootstrap_gain = sort(bootstrap_gain);
            ci_row_index = ci_row_index+1;
            paired_ci_rows{ci_row_index} = baseline_method_names{baseline_index};
            paired_ci_values(ci_row_index, :) = [P, current_SNR_dB, numel(snr_indices), 10*log10(sum(sample_errors(snr_indices, baseline_index))/sum(sample_errors(snr_indices, 6))), bootstrap_gain(max(1, floor(0.025*bootstrap_replicates))), bootstrap_gain(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)))];
        end
    end
end
paired_test_ci_table = table(paired_ci_values(:, 1), paired_ci_values(:, 2), paired_ci_values(:, 3), paired_ci_rows, paired_ci_values(:, 4), paired_ci_values(:, 5), paired_ci_values(:, 6), 'VariableNames', {'P','SNR_dB','TestSamples','BaselineMethod','DL_Gain_dB','PairedCI95_Low_dB','PairedCI95_High_dB'});

%% -------------------- SAVE TABLE, MAT FILE, AND FIGURE --------------------
result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','TestSamples','FrozenHardRank','MeanHatimStyleRank','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','HatimStyle_HSGS_NMSE_dB','HardRank_HSGS_NMSE_dB','PI_DU_RACE_16Stage_NMSE_dB'});
result_csv = fullfile(output_folder, 'N8_fresh_test_six_method_nmse.csv');
result_mat = fullfile(output_folder, 'N8_fresh_test_six_method_nmse.mat');
result_png = fullfile(output_folder, 'N8_fresh_test_six_method_nmse.png');
result_fig = fullfile(output_folder, 'N8_fresh_test_six_method_nmse.fig');
paired_ci_csv = fullfile(output_folder, 'N8_fresh_test_paired_DL_gain_CI.csv');
configuration = struct('N', N, 'K', K, 'P_list', P_list, 'SNR_grid', SNR_grid, 'RSR_dB', RSR_dB, 'AoA_deg_range', AoA_deg_range, 'p_H', p_H, 'rank_candidates', rank_candidates, 'pilot_training_fraction', pilot_training_fraction, 'rank_selection_updates', rank_selection_updates, 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'our_rank_policy_file', frozen_policy_file, 'Hatim_style_rank_selection', 'per-observation held-out pilot magnitude residual', 'DL_initialization', 'GS', 'DL_model_folder', dl_model_folder, 'data_version', 'V3_AoA_Uminus90to90_FRESH_TEST_V2');
writetable(result_table, result_csv);
writetable(paired_test_ci_table, paired_ci_csv);
save(result_mat, 'result_table', 'paired_test_ci_table', 'configuration', 'our_frozen_rank', 'test_error_by_P', 'test_channel_energy_by_P', 'test_snr_by_P', 'hatim_rank_by_P', 'bootstrap_replicates', '-v7.3');

figure('Color', 'w', 'Position', [100 100 1200 520]);
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    selected_rows = result_table.P == P;
    subplot(1, 2, p_index);
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows), '-o', 'LineWidth', 1.5, 'DisplayName', 'GS');
    hold on;
    plot(SNR_grid, result_table.GD_NMSE_dB(selected_rows), '-s', 'LineWidth', 1.5, 'DisplayName', 'GD');
    plot(SNR_grid, result_table.EMGS_NMSE_dB(selected_rows), '-d', 'LineWidth', 1.5, 'DisplayName', 'EM-GS');
    plot(SNR_grid, result_table.HatimStyle_HSGS_NMSE_dB(selected_rows), '-v', 'LineWidth', 1.7, 'DisplayName', 'Pilot-split HS-GS');
    plot(SNR_grid, result_table.HardRank_HSGS_NMSE_dB(selected_rows), '-^', 'LineWidth', 1.7, 'DisplayName', 'Validation-frozen hard-rank HS-GS');
    plot(SNR_grid, result_table.PI_DU_RACE_16Stage_NMSE_dB(selected_rows), '-x', 'LineWidth', 1.8, 'DisplayName', 'PI-DU-RACE 16-stage');
    grid on;
    xlabel('SNR (dB)');
    ylabel('Test NMSE (dB)');
    title(sprintf('N=%d, K=%d, P=%d, RSR=%g dB', N, K, P, RSR_dB));
    legend('Location', 'best');
    xticks(SNR_grid);
end
sgtitle('V3 N=8 fresh independent test comparison; 500 samples per SNR');
exportgraphics(gcf, result_png, 'Resolution', 200);
savefig(gcf, result_fig);
disp(result_table);
fprintf('\nFresh-test six-method comparison complete.\n');
fprintf('Hatim-style ranks used held-out pilot magnitudes; frozen hard ranks came from validation. DL used its frozen GS-init model.\n');
fprintf('Paired 95%% confidence intervals are pointwise. Positive gain favors DL.\n');
fprintf('CSV: %s\n', result_csv);
fprintf('Paired DL gain intervals: %s\n', paired_ci_csv);
fprintf('Figure: %s\n', result_png);
