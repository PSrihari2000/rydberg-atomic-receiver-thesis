%% N=8 comparison with test-time pilot-split hard-rank selection
% Hard rank is selected per observation using only that observation's pilot magnitudes.
% The pilot split does not use the true channel. The selected rank is then run on all pilots.
% Soft-HS multipliers are fixed to the previously frozen V3 values.
% The selected 16-stage model is loaded; test data are used only for final scoring.
% All six estimators use the exact same reserved test observations.

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
d_over_lambda = 0.5;
p_H = 3;
hard_rank_candidates = 1:4;
pilot_training_fraction = 0.70;
rank_selection_updates = 25;
hard_rank_final_updates = 100;
cadzow_sweeps = 4;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
method_names = {'GS','GD','EM-GS','Hard-rank HS-GS','Soft-rank HS-GS','PI-DU-RACE 16-stage'};

%% -------------------- INPUT AND OUTPUT PATHS --------------------
script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
dl_model_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_ADAPTIVE_GATE_RSR25DB');
output_folder = fullfile(v3_folder, 'results', 'N8_FINAL_SIX_METHOD_TEST_COMPARISON_PILOT_SPLIT_RANK_RSR25DB');

if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect or rename it before running this final comparison.');
end
if ~isfolder(scenario_folder) || ~isfolder(dl_model_folder)
    error('The V3 test-data folder or selected 16-stage model folder is missing.');
end
if ~isfile(fullfile(scenario_folder, 'test_P10.mat')) || ~isfile(fullfile(scenario_folder, 'test_P30.mat'))
    error('Reserved V3 test files are required for P=10 and P=30.');
end
if ~isfile(fullfile(dl_model_folder, 'selected_model_P10.mat')) || ~isfile(fullfile(dl_model_folder, 'selected_model_P30.mat'))
    error('Selected 16-stage DL model files for P=10 and P=30 are required.');
end
mkdir(output_folder);

%% Fixed soft-rank multipliers from the previously frozen V3 validation policy
frozen_soft_multiplier = [1 0.5 0.5 0.25 0.25 0.25 0.25 0; 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25];

%% -------------------- RESULT STORAGE --------------------
number_of_rows = numel(P_list)*numel(SNR_grid);
result_rows = zeros(number_of_rows, 11);
selected_hard_rank_sum = zeros(numel(P_list), numel(SNR_grid));
selected_hard_rank_count = zeros(numel(P_list), numel(SNR_grid));
test_selected_hard_rank_by_P = cell(numel(P_list), 1);
test_error_by_P = cell(numel(P_list), 1);
test_channel_energy_by_P = cell(numel(P_list), 1);
row_index = 0;

fprintf('N=8 final six-method comparison; RSR=%g dB; AoA=Uniform(-90,90 deg).\n', RSR_dB);
fprintf('Hard rank is selected per test observation using a held-out pilot subset; no true channel is used for rank selection.\n');
fprintf('Soft-HS multipliers are fixed to the previously frozen V3 validation values.\n');
fprintf('Only reserved test data and selected DL checkpoints are loaded; no validation data files are loaded.\n');
fprintf('All methods share each test sample: GS, GD, EM-GS, hard HS-GS, soft HS-GS, 16-stage DL.\n');

fprintf('Starting reserved-test loading and final evaluation now.\n');

%% -------------------- EVALUATE ALL METHODS ON RESERVED TEST DATA --------------------
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    test_file = fullfile(scenario_folder, sprintf('test_P%d.mat', P));
    model_file = fullfile(dl_model_folder, sprintf('selected_model_P%d.mat', P));
    test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'AoA_deg_range', 'data_version');
    model_data = load(model_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs');
    if test_data.N ~= N || test_data.K ~= K || test_data.P ~= P || test_data.RSR_dB ~= RSR_dB || ~isequal(test_data.AoA_deg_range, AoA_deg_range) || ~strcmp(test_data.data_version, 'V3_AoA_Uminus90to90')
        error('Test metadata mismatch for P=%d.', P);
    end
    if model_data.N ~= N || model_data.K ~= K || model_data.P ~= P || model_data.RSR_dB ~= RSR_dB || model_data.p_H ~= p_H || model_data.number_of_stages ~= 16 || model_data.parameters_per_stage ~= 13 || model_data.number_of_gate_features ~= 3 || model_data.number_of_gate_outputs ~= 3 || ~isequal(size(model_data.best_parameters_numeric), [13 16])
        error('Selected 16-stage model metadata mismatch for P=%d.', P);
    end
    number_of_test_samples = size(test_data.Y_data, 3);
    if number_of_test_samples ~= 4000
        error('Expected 4000 reserved test samples for P=%d.', P);
    end
    fprintf('\nP=%d; reserved test samples=%d; selected DL epoch=%d.\n', P, number_of_test_samples, model_data.best_epoch_saved);

    %% -------------------- EVALUATE SIX METHODS ON RESERVED TEST DATA --------------------
    test_error = zeros(number_of_test_samples, 6);
    test_channel_energy = zeros(number_of_test_samples, 1);
    selected_hard_rank_by_sample = zeros(number_of_test_samples, 1);
    fprintf('Scoring all six methods on the same reserved P=%d test observations.\n', P);

    for sample_index = 1:number_of_test_samples
        Y = zeros(N, P);
        G_true = zeros(N, K);
        S = zeros(K, P);
        B = zeros(N, P);
        Y(:) = test_data.Y_data(:, :, sample_index);
        G_true(:) = test_data.G_data(:, :, sample_index);
        S(:) = test_data.S_data(:, :, sample_index);
        B(:) = test_data.B_data(:, :, sample_index);
        sigma2 = 0;
        sigma2 = test_data.sigma2_data(sample_index);
        current_SNR_dB = test_data.SNR_dB_data(sample_index);
        snr_index = find(SNR_grid == current_SNR_dB, 1);
        if isempty(snr_index)
            error('Unexpected test SNR value for P=%d sample=%d.', P, sample_index);
        end
        right_inverse = S'/(S*S');

        %% Select hard Hankel rank from this test observation's pilot magnitudes
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
        rank_selection_error = zeros(1, numel(hard_rank_candidates));

        for candidate_index = 1:numel(hard_rank_candidates)
            candidate_rank = hard_rank_candidates(candidate_index);
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
                G_emgs = (Y_T.*exp(1i*angle(field)).*bessel_ratio-B_T)*right_inverse_T;
                G_rank_projected = zeros(N, K);

                for user_index = 1:K
                    channel_vector = G_emgs(:, user_index);

                    for sweep_index = 1:cadzow_sweeps
                        hankel_rows = N-p_H;
                        hankel_columns = p_H+1;
                        hankel_matrix = zeros(hankel_rows, hankel_columns);

                        for hankel_row = 1:hankel_rows
                            hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+hankel_columns-1).';
                        end

                        [U, Sigma, V_svd] = svd(hankel_matrix, 'econ');
                        hankel_matrix = U(:, 1:candidate_rank)*Sigma(1:candidate_rank, 1:candidate_rank)*V_svd(:, 1:candidate_rank)';
                        reconstructed_channel = zeros(N, 1);
                        diagonal_counts = zeros(N, 1);

                        for hankel_row = 1:hankel_rows
                            for hankel_column = 1:hankel_columns
                                antenna_index = hankel_row+hankel_column-1;
                                reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(hankel_row, hankel_column);
                                diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
                            end
                        end

                        channel_vector = reconstructed_channel./diagonal_counts;
                    end

                    G_rank_projected(:, user_index) = channel_vector;
                end

                G_candidate = G_rank_projected;
            end

            predicted_heldout_magnitudes = abs(G_candidate*S_V+B_V);
            rank_selection_error(candidate_index) = norm(Y_V-predicted_heldout_magnitudes, 'fro')^2;
        end

        if any(~isfinite(rank_selection_error))
            error('Pilot-split hard-rank selection produced a nonfinite score for P=%d, sample=%d.', P, sample_index);
        end
        [~, best_rank_index] = min(rank_selection_error);
        selected_hard_rank = hard_rank_candidates(best_rank_index);
        selected_hard_rank_by_sample(sample_index) = selected_hard_rank;
        selected_hard_rank_sum(p_index, snr_index) = selected_hard_rank_sum(p_index, snr_index)+selected_hard_rank;
        selected_hard_rank_count(p_index, snr_index) = selected_hard_rank_count(p_index, snr_index)+1;

        %% Spectral initialization used by GS, EM-GS, and both HS-GS methods
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

        %% Algorithm 1: Biased GS, converged with the shared relative-change rule
        G_GS = G_spectral;
        GS_converged = false;
        for update_index = 1:maximum_updates
            G_previous = G_GS;
            G_GS = (Y.*exp(1i*angle(G_GS*S+B))-B)*right_inverse;
            if norm(G_GS-G_previous, 'fro')^2/max(norm(G_GS, 'fro')^2, eps) < convergence_tolerance
                GS_converged = true;
                break;
            end
        end
        test_error(sample_index, 1) = norm(G_GS-G_true, 'fro')^2;

        %% Algorithm 2: Linearized GD with a reproducible sample-specific initialization
        random_seed = 861000000+1000000*P+sample_index;
        rng(mod(random_seed+720000, 2^32-1), 'twister');
        G_GD = sqrt(gd_initial_variance/2)*(randn(N, K)+1i*randn(N, K));
        phase_reference = exp(-1i*angle(B));
        step_size = 1/max(real(eig(S*S')));
        GD_converged = false;
        for update_index = 1:maximum_updates
            G_previous = G_GD;
            G_GD = G_GD+step_size*((Y-abs(B)-real(phase_reference.*(G_GD*S))).*conj(phase_reference))*S';
            if norm(G_GD-G_previous, 'fro')^2/max(norm(G_GD, 'fro')^2, eps) < convergence_tolerance
                GD_converged = true;
                break;
            end
        end
        test_error(sample_index, 2) = norm(G_GD-G_true, 'fro')^2;

        %% Algorithm 3: EM-GS
        G_EMGS = G_spectral;
        EMGS_converged = false;
        for update_index = 1:maximum_updates
            G_previous = G_EMGS;
            field = G_EMGS*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_EMGS = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            if norm(G_EMGS-G_previous, 'fro')^2/max(norm(G_EMGS, 'fro')^2, eps) < convergence_tolerance
                EMGS_converged = true;
                break;
            end
        end
        test_error(sample_index, 3) = norm(G_EMGS-G_true, 'fro')^2;

        %% Algorithm 4: Pilot-split selected hard-rank HS-GS
        G_HardHSGS = G_spectral;
        for update_index = 1:hard_rank_final_updates
            field = G_HardHSGS*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            G_projected = zeros(N, K);

            for user_index = 1:K
                channel_vector = G_emgs(:, user_index);

                for sweep_index = 1:cadzow_sweeps
                    hankel_matrix = zeros(N-p_H, p_H+1);
                    for hankel_row = 1:N-p_H
                        hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+p_H).';
                    end
                    [U, Sigma, V] = svd(hankel_matrix, 'econ');
                    hankel_matrix = U(:, 1:selected_hard_rank)*Sigma(1:selected_hard_rank, 1:selected_hard_rank)*V(:, 1:selected_hard_rank)';
                    reconstructed_channel = zeros(N, 1);
                    diagonal_counts = zeros(N, 1);

                    for hankel_row = 1:N-p_H
                        for hankel_column = 1:p_H+1
                            antenna_index = hankel_row+hankel_column-1;
                            reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(hankel_row, hankel_column);
                            diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
                        end
                    end

                    channel_vector = reconstructed_channel./diagonal_counts;
                end

                G_projected(:, user_index) = channel_vector;
            end

            G_HardHSGS = G_projected;
        end
        test_error(sample_index, 4) = norm(G_HardHSGS-G_true, 'fro')^2;

        %% Algorithm 5: Frozen-multiplier soft-rank HS-GS
        soft_multiplier = frozen_soft_multiplier(p_index, snr_index);
        threshold_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
        threshold = soft_multiplier*threshold_scale;
        G_SoftHSGS = G_spectral;
        SoftHSGS_converged = false;
        for update_index = 1:maximum_updates
            G_previous = G_SoftHSGS;
            field = G_SoftHSGS*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            G_projected = zeros(N, K);
            if threshold > 0
                for user_index = 1:K
                    channel_vector = G_emgs(:, user_index);
                    hankel_matrix = zeros(N-p_H, p_H+1);
                    for hankel_row = 1:N-p_H
                        hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+p_H).';
                    end
                    [U, Sigma, V] = svd(hankel_matrix, 'econ');
                    shrunk_singular_values = max(diag(Sigma)-threshold, 0);
                    hankel_matrix = U*diag(shrunk_singular_values)*V';
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
            else
                G_projected = G_emgs;
            end
            G_SoftHSGS = G_projected;
            if norm(G_SoftHSGS-G_previous, 'fro')^2/max(norm(G_SoftHSGS, 'fro')^2, eps) < convergence_tolerance
                SoftHSGS_converged = true;
                break;
            end
        end
        test_error(sample_index, 5) = norm(G_SoftHSGS-G_true, 'fro')^2;

        %% Algorithm 6: Trained 16-stage PI-DU-RACE forward pass
        parameters = model_data.best_parameters_numeric;
        G_DL = (Y-abs(B))*right_inverse;
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
            fprintf('P=%d test observations processed: %d/%d.\n', P, sample_index, number_of_test_samples);
        end
    end

    if any(~isfinite(test_error), 'all') || any(test_channel_energy <= 0)
        error('Nonfinite test error or invalid true-channel energy for P=%d.', P);
    end
    test_error_by_P{p_index} = test_error;
    test_channel_energy_by_P{p_index} = test_channel_energy;
    test_selected_hard_rank_by_P{p_index} = selected_hard_rank_by_sample;

    %% -------------------- SUMMARIZE PAIRED TEST NMSE BY SNR --------------------
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        test_indices = find(test_data.SNR_dB_data == current_SNR_dB);
        if numel(test_indices) ~= 500
            error('Expected 500 test samples at P=%d and SNR=%g dB.', P, current_SNR_dB);
        end
        nmse_dB = 10*log10(sum(test_error(test_indices, :), 1)./sum(test_channel_energy(test_indices)));
        row_index = row_index+1;
        mean_selected_hard_rank = selected_hard_rank_sum(p_index, snr_index)/selected_hard_rank_count(p_index, snr_index);
        result_rows(row_index, :) = [P, current_SNR_dB, numel(test_indices), mean_selected_hard_rank, frozen_soft_multiplier(p_index, snr_index), nmse_dB];
        fprintf('P=%d SNR=%g dB: mean hard rank %.2f; GS %.3f, GD %.3f, EM-GS %.3f, hard HS-GS %.3f, soft HS-GS %.3f, DL16 %.3f dB.\n', P, current_SNR_dB, mean_selected_hard_rank, nmse_dB(1), nmse_dB(2), nmse_dB(3), nmse_dB(4), nmse_dB(5), nmse_dB(6));
    end
end

%% -------------------- SAVE FINAL TABLE AND FIGURE --------------------
result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','TestSamples','MeanTestSelectedHardRank','FrozenSoftMultiplier','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','HardRank_HSGS_NMSE_dB','SoftRank_HSGS_NMSE_dB','PI_DU_RACE_16Stage_NMSE_dB'});
result_csv = fullfile(output_folder, 'N8_pilot_split_rank_six_method_test_nmse.csv');
result_mat = fullfile(output_folder, 'N8_pilot_split_rank_six_method_test_nmse.mat');
result_png = fullfile(output_folder, 'N8_pilot_split_rank_six_method_test_nmse.png');
result_fig = fullfile(output_folder, 'N8_pilot_split_rank_six_method_test_nmse.fig');
configuration = struct('N', N, 'K', K, 'P_list', P_list, 'SNR_grid', SNR_grid, 'RSR_dB', RSR_dB, 'AoA_deg_range', AoA_deg_range, 'd_over_lambda', d_over_lambda, 'p_H', p_H, 'hard_rank_candidates', hard_rank_candidates, 'pilot_training_fraction', pilot_training_fraction, 'rank_selection_updates', rank_selection_updates, 'hard_rank_final_updates', hard_rank_final_updates, 'cadzow_sweeps', cadzow_sweeps, 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'gd_initial_variance', gd_initial_variance, 'method_names', {method_names}, 'frozen_soft_multiplier', frozen_soft_multiplier, 'hard_rank_selection_method', 'per-observation held-out pilot magnitude residual', 'data_version', 'V3_AoA_Uminus90to90');
writetable(result_table, result_csv);
save(result_mat, 'result_table', 'configuration', 'test_error_by_P', 'test_channel_energy_by_P', 'test_selected_hard_rank_by_P', '-v7.3');

figure('Color', 'w', 'Position', [100 100 1200 520]);
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    selected_rows = result_table.P == P;
    subplot(1, 2, p_index);
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows), '-o', 'LineWidth', 1.5, 'DisplayName', 'GS');
    hold on;
    plot(SNR_grid, result_table.GD_NMSE_dB(selected_rows), '-s', 'LineWidth', 1.5, 'DisplayName', 'GD');
    plot(SNR_grid, result_table.EMGS_NMSE_dB(selected_rows), '-d', 'LineWidth', 1.5, 'DisplayName', 'EM-GS');
    plot(SNR_grid, result_table.HardRank_HSGS_NMSE_dB(selected_rows), '-^', 'LineWidth', 1.5, 'DisplayName', 'Hard-rank HS-GS');
    plot(SNR_grid, result_table.SoftRank_HSGS_NMSE_dB(selected_rows), '-v', 'LineWidth', 1.5, 'DisplayName', 'Soft-rank HS-GS');
    plot(SNR_grid, result_table.PI_DU_RACE_16Stage_NMSE_dB(selected_rows), '-x', 'LineWidth', 1.8, 'DisplayName', 'PI-DU-RACE 16-stage');
    grid on;
    xlabel('SNR (dB)');
    ylabel('Test NMSE (dB)');
    title(sprintf('N=%d, K=%d, P=%d, RSR=%g dB', N, K, P, RSR_dB));
    legend('Location', 'best');
end
sgtitle('V3 N=8 test comparison with per-observation pilot-split hard-rank selection');
exportgraphics(gcf, result_png, 'Resolution', 200);
savefig(gcf, result_fig);
disp(result_table);
fprintf('\nFinal six-method test comparison complete.\n');
fprintf('Hard rank was selected per test observation from held-out pilot magnitudes without using G_true.\n');
fprintf('Soft multipliers are fixed to the previously frozen V3 validation values.\n');
fprintf('No external validation data files were loaded.\n');
fprintf('CSV: %s\n', result_csv);
fprintf('Figure: %s\n', result_png);
