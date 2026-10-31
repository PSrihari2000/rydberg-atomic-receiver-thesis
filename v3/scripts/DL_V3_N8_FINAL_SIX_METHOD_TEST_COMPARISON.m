%% Final N=8 test comparison: GS, GD, EM-GS, hard/soft HS-GS, and 16-stage PI-DU-RACE
% Validation data select and confirm the hard Hankel rank only.
% The soft-HS multipliers were selected previously using validation data.
% The saved 16-stage model was trained and selected using training/validation data.
% Reserved test data are used once for the final NMSE curves and are not tuned.
% All six estimators use the exact same test observations for each sample.

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
rank_selection_samples_per_SNR = 250;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
method_names = {'GS','GD','EM-GS','Hard-rank HS-GS','Soft-rank HS-GS','PI-DU-RACE 16-stage'};

%% -------------------- INPUT AND OUTPUT PATHS --------------------
script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
classical_folder = fullfile(v3_folder, 'results', 'N8_CLASSICAL_COMPARISON_RSR25DB');
soft_policy_file = fullfile(classical_folder, 'N8_frozen_HSGS_validation_policy.mat');
dl_model_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_16STAGE_ADAPTIVE_GATE_RSR25DB');
output_folder = fullfile(v3_folder, 'results', 'N8_FINAL_SIX_METHOD_TEST_COMPARISON_RSR25DB');

if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect or rename it before running this final comparison.');
end
if ~isfolder(scenario_folder) || ~isfile(soft_policy_file) || ~isfolder(dl_model_folder)
    error('A required V3 data, soft-HS policy, or selected 16-stage model folder is missing.');
end
if ~isfile(fullfile(scenario_folder, 'validation_P10.mat')) || ~isfile(fullfile(scenario_folder, 'validation_P30.mat')) || ~isfile(fullfile(scenario_folder, 'test_P10.mat')) || ~isfile(fullfile(scenario_folder, 'test_P30.mat'))
    error('Both V3 validation and reserved test files are required for P=10 and P=30.');
end
if ~isfile(fullfile(dl_model_folder, 'selected_model_P10.mat')) || ~isfile(fullfile(dl_model_folder, 'selected_model_P30.mat'))
    error('Selected 16-stage DL model files for P=10 and P=30 are required.');
end
mkdir(output_folder);

policy_data = load(soft_policy_file, 'configuration', 'frozen_multiplier');
policy_configuration = policy_data.configuration;
if policy_configuration.N ~= N || policy_configuration.K ~= K || policy_configuration.RSR_dB ~= RSR_dB || ~isequal(policy_configuration.P_list, P_list) || ~isequal(policy_configuration.SNR_dB, SNR_grid) || ~strcmp(policy_configuration.data_version, 'V3_AoA_Uminus90to90')
    error('The saved soft-HS validation policy does not match this V3 scenario.');
end
if policy_configuration.p_H ~= p_H || policy_configuration.maximum_updates ~= maximum_updates || policy_configuration.convergence_tolerance ~= convergence_tolerance
    error('Saved policy Hankel or convergence settings do not match this script.');
end
frozen_soft_multiplier = policy_data.frozen_multiplier;

%% -------------------- RESULT STORAGE --------------------
number_of_rows = numel(P_list)*numel(SNR_grid);
result_rows = zeros(number_of_rows, 11);
hard_rank_policy = zeros(numel(P_list), numel(SNR_grid));
hard_rank_selection_gain_dB = zeros(numel(P_list), numel(SNR_grid));
test_error_by_P = cell(numel(P_list), 1);
test_channel_energy_by_P = cell(numel(P_list), 1);
row_index = 0;

fprintf('N=8 final six-method comparison; RSR=%g dB; AoA=Uniform(-90,90 deg).\n', RSR_dB);
fprintf('Hard rank is selected and confirmed on validation data only.\n');
fprintf('Soft-HS multipliers are loaded from the frozen V3 validation policy.\n');
fprintf('The reserved test split is used only for final scoring. No training data are loaded.\n');
fprintf('All methods share each test sample: GS, GD, EM-GS, hard HS-GS, soft HS-GS, 16-stage DL.\n');

%% -------------------- PROCESS EACH PILOT COUNT --------------------
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    test_file = fullfile(scenario_folder, sprintf('test_P%d.mat', P));
    model_file = fullfile(dl_model_folder, sprintf('selected_model_P%d.mat', P));

    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'AoA_deg_range', 'data_version');
    model_data = load(model_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs');

    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ~isequal(validation_data.AoA_deg_range, AoA_deg_range) || ~strcmp(validation_data.data_version, 'V3_AoA_Uminus90to90')
        error('Validation metadata mismatch for P=%d.', P);
    end
    if model_data.N ~= N || model_data.K ~= K || model_data.P ~= P || model_data.RSR_dB ~= RSR_dB || model_data.p_H ~= p_H || model_data.number_of_stages ~= 16 || model_data.parameters_per_stage ~= 13 || model_data.number_of_gate_features ~= 3 || model_data.number_of_gate_outputs ~= 3 || ~isequal(size(model_data.best_parameters_numeric), [13 16])
        error('Selected 16-stage model metadata mismatch for P=%d.', P);
    end

    number_of_validation_samples = size(validation_data.Y_data, 3);
    if number_of_validation_samples ~= 4000
        error('Expected 4000 V3 validation samples for P=%d.', P);
    end
    fprintf('\nP=%d; validation samples=%d; selected DL epoch=%d.\n', P, number_of_validation_samples, model_data.best_epoch_saved);

    %% -------------------- SELECT HARD HANKEL RANK ON VALIDATION --------------------
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        validation_indices = find(validation_data.SNR_dB_data == current_SNR_dB);
        if numel(validation_indices) ~= 500
            error('Expected 500 validation samples at P=%d and SNR=%g dB.', P, current_SNR_dB);
        end
        rng(511000+1000*P+snr_index, 'twister');
        shuffled_validation_indices = validation_indices(randperm(numel(validation_indices)));
        rank_selection_indices = shuffled_validation_indices(1:rank_selection_samples_per_SNR);
        rank_confirmation_indices = shuffled_validation_indices(rank_selection_samples_per_SNR+1:end);
        validation_rank_error = zeros(1, numel(hard_rank_candidates));
        validation_rank_confirmation_error = zeros(1, numel(hard_rank_candidates));

        fprintf('Selecting hard Hankel rank on validation: P=%d, SNR=%g dB.\n', P, current_SNR_dB);
        for validation_position = 1:numel(rank_selection_indices)
            sample_index = rank_selection_indices(validation_position);
            Y = zeros(N, P);
            G_true = zeros(N, K);
            S = zeros(K, P);
            B = zeros(N, P);
            Y(:) = validation_data.Y_data(:, :, sample_index);
            G_true(:) = validation_data.G_data(:, :, sample_index);
            S(:) = validation_data.S_data(:, :, sample_index);
            B(:) = validation_data.B_data(:, :, sample_index);
            sigma2 = 0;
            sigma2 = validation_data.sigma2_data(sample_index);
            right_inverse = S'/(S*S');

            %% Spectral initialization shared by the four rank candidates
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

            rank_candidate_estimates = zeros(N, K, numel(hard_rank_candidates));
            rank_candidate_converged = false(1, numel(hard_rank_candidates));
            rank_candidate_updates = zeros(1, numel(hard_rank_candidates));
            for candidate_index = 1:numel(hard_rank_candidates)
                rank_candidate_estimates(:, :, candidate_index) = G_spectral;
            end

            %% Run each hard-rank EM-GS candidate to convergence
            for update_index = 1:maximum_updates
                for candidate_index = 1:numel(hard_rank_candidates)
                    if rank_candidate_converged(candidate_index)
                        continue;
                    end
                    G_previous = rank_candidate_estimates(:, :, candidate_index);
                    field = G_previous*S+B;
                    kappa = 2/max(sigma2, eps)*Y.*abs(field);
                    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
                    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
                    G_rank_projected = zeros(N, K);
                    hankel_rows = N-p_H;
                    hankel_columns = p_H+1;
                    for user_index = 1:K
                        channel_vector = G_emgs(:, user_index);
                        hankel_matrix = zeros(hankel_rows, hankel_columns);
                        for hankel_row = 1:hankel_rows
                            hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+hankel_columns-1).';
                        end
                        [U, Sigma, V] = svd(hankel_matrix, 'econ');
                        selected_rank = hard_rank_candidates(candidate_index);
                        hankel_matrix = U(:, 1:selected_rank)*Sigma(1:selected_rank, 1:selected_rank)*V(:, 1:selected_rank)';
                        reconstructed_channel = zeros(N, 1);
                        diagonal_counts = zeros(N, 1);
                        for hankel_row = 1:hankel_rows
                            for hankel_column = 1:hankel_columns
                                antenna_index = hankel_row+hankel_column-1;
                                reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(hankel_row, hankel_column);
                                diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
                            end
                        end
                        G_rank_projected(:, user_index) = reconstructed_channel./diagonal_counts;
                    end
                    rank_candidate_estimates(:, :, candidate_index) = G_rank_projected;
                    relative_update = norm(G_rank_projected-G_previous, 'fro')^2/max(norm(G_rank_projected, 'fro')^2, eps);
                    if relative_update < convergence_tolerance
                        rank_candidate_converged(candidate_index) = true;
                        rank_candidate_updates(candidate_index) = update_index;
                    end
                end
                if all(rank_candidate_converged)
                    break;
                end
            end
            for candidate_index = 1:numel(hard_rank_candidates)
                validation_rank_error(candidate_index) = validation_rank_error(candidate_index)+norm(rank_candidate_estimates(:, :, candidate_index)-G_true, 'fro')^2;
            end
        end

        [~, best_rank_candidate_index] = min(validation_rank_error);
        selected_hard_rank = hard_rank_candidates(best_rank_candidate_index);

        %% Confirm the selected rank against full-rank Hankelization on held-out validation samples
        for validation_position = 1:numel(rank_confirmation_indices)
            sample_index = rank_confirmation_indices(validation_position);
            Y = zeros(N, P);
            G_true = zeros(N, K);
            S = zeros(K, P);
            B = zeros(N, P);
            Y(:) = validation_data.Y_data(:, :, sample_index);
            G_true(:) = validation_data.G_data(:, :, sample_index);
            S(:) = validation_data.S_data(:, :, sample_index);
            B(:) = validation_data.B_data(:, :, sample_index);
            sigma2 = 0;
            sigma2 = validation_data.sigma2_data(sample_index);
            right_inverse = S'/(S*S');
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
            G_selected_rank = G_spectral;
            G_full_rank = G_spectral;
            selected_rank_converged = false;
            full_rank_converged = false;
            for update_index = 1:maximum_updates
                if ~selected_rank_converged
                    G_previous = G_selected_rank;
                    field = G_previous*S+B;
                    kappa = 2/max(sigma2, eps)*Y.*abs(field);
                    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
                    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
                    G_projected = zeros(N, K);
                    for user_index = 1:K
                        channel_vector = G_emgs(:, user_index);
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
                        G_projected(:, user_index) = reconstructed_channel./diagonal_counts;
                    end
                    G_selected_rank = G_projected;
                    if norm(G_projected-G_previous, 'fro')^2/max(norm(G_projected, 'fro')^2, eps) < convergence_tolerance
                        selected_rank_converged = true;
                    end
                end
                if ~full_rank_converged
                    G_previous = G_full_rank;
                    field = G_previous*S+B;
                    kappa = 2/max(sigma2, eps)*Y.*abs(field);
                    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
                    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
                    G_projected = zeros(N, K);
                    for user_index = 1:K
                        channel_vector = G_emgs(:, user_index);
                        hankel_matrix = zeros(N-p_H, p_H+1);
                        for hankel_row = 1:N-p_H
                            hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+p_H).';
                        end
                        [U, Sigma, V] = svd(hankel_matrix, 'econ');
                        hankel_matrix = U*Sigma*V';
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
                    G_full_rank = G_projected;
                    if norm(G_projected-G_previous, 'fro')^2/max(norm(G_projected, 'fro')^2, eps) < convergence_tolerance
                        full_rank_converged = true;
                    end
                end
                if selected_rank_converged && full_rank_converged
                    break;
                end
            end
            validation_rank_confirmation_error(1) = validation_rank_confirmation_error(1)+norm(G_selected_rank-G_true, 'fro')^2;
            validation_rank_confirmation_error(2) = validation_rank_confirmation_error(2)+norm(G_full_rank-G_true, 'fro')^2;
        end
        confirmation_gain = 10*log10(max(validation_rank_confirmation_error(2), realmin)/max(validation_rank_confirmation_error(1), realmin));
        if selected_hard_rank < max(hard_rank_candidates) && confirmation_gain <= 0
            selected_hard_rank = max(hard_rank_candidates);
        end
        hard_rank_policy(p_index, snr_index) = selected_hard_rank;
        hard_rank_selection_gain_dB(p_index, snr_index) = confirmation_gain;
        fprintf('  Frozen hard rank=%d; held-out validation gain vs rank 4=%+.3f dB.\n', selected_hard_rank, confirmation_gain);
    end

end

fprintf('\nHard-rank policy is now frozen for both pilot counts using validation only.\n');
fprintf('The saved soft-HS policy and selected 16-stage models are also frozen.\n');
hard_policy_rows = zeros(number_of_rows, 5);
policy_row_index = 0;
for p_index = 1:numel(P_list)
    for snr_index = 1:numel(SNR_grid)
        policy_row_index = policy_row_index+1;
        hard_policy_rows(policy_row_index, :) = [P_list(p_index), SNR_grid(snr_index), hard_rank_policy(p_index, snr_index), hard_rank_selection_gain_dB(p_index, snr_index), frozen_soft_multiplier(p_index, snr_index)];
    end
end
hard_rank_validation_policy = array2table(hard_policy_rows, 'VariableNames', {'P','SNR_dB','FrozenHardRank','HardRankValidationGain_vs_Rank4_dB','FrozenSoftMultiplier'});
hard_rank_policy_csv = fullfile(output_folder, 'N8_hard_rank_validation_policy.csv');
hard_rank_policy_mat = fullfile(output_folder, 'N8_hard_rank_validation_policy.mat');
writetable(hard_rank_validation_policy, hard_rank_policy_csv);
save(hard_rank_policy_mat, 'hard_rank_validation_policy', 'hard_rank_policy', 'hard_rank_selection_gain_dB', 'frozen_soft_multiplier', 'hard_rank_candidates', 'rank_selection_samples_per_SNR', 'P_list', 'SNR_grid', 'N', 'K', 'p_H', 'maximum_updates', 'convergence_tolerance');
fprintf('Saved the frozen validation policy before loading test data.\n');
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
    number_of_test_samples = size(test_data.Y_data, 3);
    if number_of_test_samples ~= 4000
        error('Expected 4000 reserved test samples for P=%d.', P);
    end
    fprintf('\nP=%d; reserved test samples=%d; selected DL epoch=%d.\n', P, number_of_test_samples, model_data.best_epoch_saved);

    %% -------------------- EVALUATE SIX METHODS ON RESERVED TEST DATA --------------------
    test_error = zeros(number_of_test_samples, 6);
    test_channel_energy = zeros(number_of_test_samples, 1);
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

        %% Algorithm 4: Validation-selected hard-rank HS-GS
        selected_hard_rank = hard_rank_policy(p_index, snr_index);
        G_HardHSGS = G_spectral;
        HardHSGS_converged = false;
        for update_index = 1:maximum_updates
            G_previous = G_HardHSGS;
            field = G_HardHSGS*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            G_projected = zeros(N, K);
            for user_index = 1:K
                channel_vector = G_emgs(:, user_index);
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
                G_projected(:, user_index) = reconstructed_channel./diagonal_counts;
            end
            G_HardHSGS = G_projected;
            if norm(G_HardHSGS-G_previous, 'fro')^2/max(norm(G_HardHSGS, 'fro')^2, eps) < convergence_tolerance
                HardHSGS_converged = true;
                break;
            end
        end
        test_error(sample_index, 4) = norm(G_HardHSGS-G_true, 'fro')^2;

        %% Algorithm 5: Validation-frozen soft-rank HS-GS
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

    %% -------------------- SUMMARIZE PAIRED TEST NMSE BY SNR --------------------
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        test_indices = find(test_data.SNR_dB_data == current_SNR_dB);
        if numel(test_indices) ~= 500
            error('Expected 500 test samples at P=%d and SNR=%g dB.', P, current_SNR_dB);
        end
        nmse_dB = 10*log10(sum(test_error(test_indices, :), 1)./sum(test_channel_energy(test_indices)));
        row_index = row_index+1;
        result_rows(row_index, :) = [P, current_SNR_dB, numel(test_indices), hard_rank_policy(p_index, snr_index), frozen_soft_multiplier(p_index, snr_index), nmse_dB];
        fprintf('P=%d SNR=%g dB: GS %.3f, GD %.3f, EM-GS %.3f, hard HS-GS %.3f, soft HS-GS %.3f, DL16 %.3f dB.\n', P, current_SNR_dB, nmse_dB(1), nmse_dB(2), nmse_dB(3), nmse_dB(4), nmse_dB(5), nmse_dB(6));
    end
end

%% -------------------- SAVE FINAL TABLE AND FIGURE --------------------
result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','TestSamples','FrozenHardRank','FrozenSoftMultiplier','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','HardRank_HSGS_NMSE_dB','SoftRank_HSGS_NMSE_dB','PI_DU_RACE_16Stage_NMSE_dB'});
result_csv = fullfile(output_folder, 'N8_final_six_method_test_nmse.csv');
result_mat = fullfile(output_folder, 'N8_final_six_method_test_nmse.mat');
result_png = fullfile(output_folder, 'N8_final_six_method_test_nmse.png');
result_fig = fullfile(output_folder, 'N8_final_six_method_test_nmse.fig');
configuration = struct('N', N, 'K', K, 'P_list', P_list, 'SNR_grid', SNR_grid, 'RSR_dB', RSR_dB, 'AoA_deg_range', AoA_deg_range, 'd_over_lambda', d_over_lambda, 'p_H', p_H, 'hard_rank_candidates', hard_rank_candidates, 'rank_selection_samples_per_SNR', rank_selection_samples_per_SNR, 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'gd_initial_variance', gd_initial_variance, 'method_names', {method_names}, 'hard_rank_policy', hard_rank_policy, 'hard_rank_selection_gain_dB', hard_rank_selection_gain_dB, 'frozen_soft_multiplier', frozen_soft_multiplier, 'data_version', 'V3_AoA_Uminus90to90');
writetable(result_table, result_csv);
save(result_mat, 'result_table', 'configuration', 'test_error_by_P', 'test_channel_energy_by_P', '-v7.3');

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
sgtitle('V3 N=8 final paired test comparison; 500 samples per SNR');
exportgraphics(gcf, result_png, 'Resolution', 200);
savefig(gcf, result_fig);
disp(result_table);
fprintf('\nFinal six-method test comparison complete.\n');
fprintf('Validation selected hard rank and soft multiplier; test data were used only for final evaluation.\n');
fprintf('CSV: %s\n', result_csv);
fprintf('Figure: %s\n', result_png);
