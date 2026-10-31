%% N=8 comparison: GS, GD, EM-GS, our hard-rank HS-GS, and pilot-split HS-GS
% No data or policy files are loaded.
% Validation and test observations are generated independently in this script.
% Our hard ranks are selected and confirmed using validation channel errors.
% Hatim-style ranks are selected per test observation using held-out pilot magnitudes.
% Test-channel truth is used only to score the final estimates.
% Both HS-GS methods use the same final EM-GS/Hankel update and stopping rule.

clear;
clc;
close all;

%% -------------------- SIMULATION SETTINGS --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
AoA_deg_range = [-90 90];
d_over_lambda = 0.5;
p_H = 3;
rank_candidates = 1:4;
validation_samples_per_SNR = 500;
rank_selection_samples = 250;
pilot_training_fraction = 0.70;
rank_selection_updates = 25;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
test_samples_per_SNR = 500;

number_of_validation_samples = validation_samples_per_SNR*numel(SNR_grid);
number_of_test_samples = test_samples_per_SNR*numel(SNR_grid);

%% -------------------- PHYSICAL CONSTANTS --------------------
hbar = 6.62607015e-34/(2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];

%% -------------------- OUTPUT FOLDER --------------------
script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
output_folder = fullfile(v3_folder, 'results', 'N8_OURS_VS_HATIMSTYLE_HSGS_INTERNAL_VALIDATION_RSR25DB');

if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it or choose a new output folder before running.');
end
mkdir(output_folder);

%% -------------------- RESULT STORAGE --------------------
number_of_result_rows = numel(P_list)*numel(SNR_grid);
result_rows = zeros(number_of_result_rows, 10);
our_frozen_rank = zeros(numel(P_list), numel(SNR_grid));
rank_validation_gain_dB = zeros(numel(P_list), numel(SNR_grid));
test_error_by_P = cell(numel(P_list), 1);
test_channel_energy_by_P = cell(numel(P_list), 1);
hatim_rank_by_P = cell(numel(P_list), 1);
result_row_index = 0;

fprintf('N=8; K=3; P=[%s]; RSR=%g dB; AoA=Uniform(-90,90 deg).\n', num2str(P_list), RSR_dB);
fprintf('Each P/SNR point uses %d validation and %d independent test observations.\n', validation_samples_per_SNR, test_samples_per_SNR);
fprintf('Validation selects our fixed rank; test data are used only for estimator evaluation.\n');

%% -------------------- PROCESS EACH PILOT COUNT --------------------
for p_index = 1:numel(P_list)
    P = P_list(p_index);

    %% Generate independent validation and test observations in memory
    validation_Y_data = zeros(N, P, number_of_validation_samples);
    validation_G_data = complex(zeros(N, K, number_of_validation_samples));
    validation_S_data = complex(zeros(K, P, number_of_validation_samples));
    validation_B_data = complex(zeros(N, P, number_of_validation_samples));
    validation_sigma2_data = zeros(number_of_validation_samples, 1);
    validation_SNR_data = zeros(number_of_validation_samples, 1);

    test_Y_data = zeros(N, P, number_of_test_samples);
    test_G_data = complex(zeros(N, K, number_of_test_samples));
    test_S_data = complex(zeros(K, P, number_of_test_samples));
    test_B_data = complex(zeros(N, P, number_of_test_samples));
    test_sigma2_data = zeros(number_of_test_samples, 1);
    test_SNR_data = zeros(number_of_test_samples, 1);

    split_seed_base = [2026110201 2026110301];

    for split_index = 1:2
        rng(split_seed_base(split_index)+P, 'twister');

        if split_index == 1
            number_of_samples = number_of_validation_samples;
        else
            number_of_samples = number_of_test_samples;
        end

        for sample_index = 1:number_of_samples
            %% Generate the true N x K channel
            G = zeros(N, K);

            for user_index = 1:K
                number_of_paths = randi([3 7]);

                for path_index = 1:number_of_paths
                    alpha = (randn+1i*randn)/sqrt(2);
                    theta_deg = AoA_deg_range(1)+diff(AoA_deg_range)*rand;
                    theta_rad = deg2rad(theta_deg);
                    spatial_phase = 2*pi*d_over_lambda*cos(theta_rad);
                    polarization = sqrt(1/3)*randn(3, 1);

                    for antenna_index = 1:N
                        G(antenna_index, user_index) = G(antenna_index, user_index)+(mu_eg.'*polarization/hbar)*alpha*exp(-1i*(antenna_index-1)*spatial_phase);
                    end
                end
            end

            %% Generate the K x P pilot matrix
            S = (randn(K, P)+1i*randn(K, P))/sqrt(2);

            %% Generate and scale the known N x P reference matrix
            alpha_reference = sqrt(10/2)*(randn+1i*randn);
            theta_reference_deg = AoA_deg_range(1)+diff(AoA_deg_range)*rand;
            theta_reference_rad = deg2rad(theta_reference_deg);
            reference_phase = 2*pi*d_over_lambda*cos(theta_reference_rad);
            polarization_reference = sqrt(1/3)*randn(3, 1);
            b = zeros(N, 1);

            for antenna_index = 1:N
                b(antenna_index) = (mu_eg.'*polarization_reference/hbar)*alpha_reference*exp(-1i*(antenna_index-1)*reference_phase);
            end

            B = b*ones(1, P);
            average_single_user_power = 0;

            for user_index = 1:K
                average_single_user_power = average_single_user_power+mean(abs(G(:, user_index)*S(user_index, :)).^2, 'all')/K;
            end

            B = B*sqrt(10^(RSR_dB/10)*average_single_user_power/mean(abs(B).^2, 'all'));

            %% Generate the magnitude-only measurement for the current SNR
            snr_index = mod(sample_index-1, numel(SNR_grid))+1;
            current_SNR_dB = SNR_grid(snr_index);
            signal_power = mean(abs(G*S).^2, 'all');
            sigma2 = signal_power/10^(current_SNR_dB/10);
            W = sqrt(sigma2/2)*(randn(N, P)+1i*randn(N, P));
            Y = abs(G*S+B+W);

            %% Store the observation in the corresponding in-memory split
            if split_index == 1
                validation_Y_data(:, :, sample_index) = Y;
                validation_G_data(:, :, sample_index) = G;
                validation_S_data(:, :, sample_index) = S;
                validation_B_data(:, :, sample_index) = B;
                validation_sigma2_data(sample_index) = sigma2;
                validation_SNR_data(sample_index) = current_SNR_dB;
            else
                test_Y_data(:, :, sample_index) = Y;
                test_G_data(:, :, sample_index) = G;
                test_S_data(:, :, sample_index) = S;
                test_B_data(:, :, sample_index) = B;
                test_sigma2_data(sample_index) = sigma2;
                test_SNR_data(sample_index) = current_SNR_dB;
            end
        end
    end

    %% -------------------- SELECT OUR HARD RANK USING VALIDATION ONLY --------------------
    fprintf('\nSelecting validation ranks for P=%d.\n', P);

    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        validation_indices = find(validation_SNR_data == current_SNR_dB);

        if numel(validation_indices) ~= validation_samples_per_SNR
            error('Unexpected validation sample count for P=%d and SNR=%g dB.', P, current_SNR_dB);
        end

        rng(511000+1000*P+snr_index, 'twister');
        shuffled_validation_indices = validation_indices(randperm(numel(validation_indices)));
        selection_indices = shuffled_validation_indices(1:rank_selection_samples);
        confirmation_indices = shuffled_validation_indices(rank_selection_samples+1:end);
        validation_rank_error = zeros(1, numel(rank_candidates));

        %% Choose a candidate rank by total channel error on the selection subset
        for selection_position = 1:numel(selection_indices)
            sample_index = selection_indices(selection_position);
            Y = validation_Y_data(:, :, sample_index);
            G_true = validation_G_data(:, :, sample_index);
            S = validation_S_data(:, :, sample_index);
            B = validation_B_data(:, :, sample_index);
            sigma2 = validation_sigma2_data(sample_index);
            right_inverse = S'/(S*S');

            %% Spectral initialization shared by all rank candidates, N x K
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

            for candidate_index = 1:numel(rank_candidates)
                candidate_rank = rank_candidates(candidate_index);
                G_candidate = G_spectral;

                for update_index = 1:maximum_updates
                    G_previous = G_candidate;
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

                    if norm(G_candidate-G_previous, 'fro')^2/max(norm(G_candidate, 'fro')^2, eps) < convergence_tolerance
                        break;
                    end
                end

                validation_rank_error(candidate_index) = validation_rank_error(candidate_index)+norm(G_candidate-G_true, 'fro')^2;
            end
        end

        [~, best_rank_index] = min(validation_rank_error);
        selected_rank = rank_candidates(best_rank_index);

        %% Confirm the selected rank against full rank on separate validation samples
        confirmation_error = zeros(1, 2);

        for confirmation_position = 1:numel(confirmation_indices)
            sample_index = confirmation_indices(confirmation_position);
            Y = validation_Y_data(:, :, sample_index);
            G_true = validation_G_data(:, :, sample_index);
            S = validation_S_data(:, :, sample_index);
            B = validation_B_data(:, :, sample_index);
            sigma2 = validation_sigma2_data(sample_index);
            right_inverse = S'/(S*S');

            %% Spectral initialization, N x K
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

            for confirmation_case = 1:2
                if confirmation_case == 1
                    confirmation_rank = selected_rank;
                else
                    confirmation_rank = max(rank_candidates);
                end

                G_candidate = G_spectral;

                for update_index = 1:maximum_updates
                    G_previous = G_candidate;
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
                        hankel_matrix = U(:, 1:confirmation_rank)*Sigma(1:confirmation_rank, 1:confirmation_rank)*V(:, 1:confirmation_rank)';
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

                    if norm(G_candidate-G_previous, 'fro')^2/max(norm(G_candidate, 'fro')^2, eps) < convergence_tolerance
                        break;
                    end
                end

                confirmation_error(confirmation_case) = confirmation_error(confirmation_case)+norm(G_candidate-G_true, 'fro')^2;
            end
        end

        confirmation_gain_dB = 10*log10(max(confirmation_error(2), realmin)/max(confirmation_error(1), realmin));

        if selected_rank < max(rank_candidates) && confirmation_gain_dB <= 0
            selected_rank = max(rank_candidates);
        end

        our_frozen_rank(p_index, snr_index) = selected_rank;
        rank_validation_gain_dB(p_index, snr_index) = confirmation_gain_dB;

        fprintf('P=%d, SNR=%g dB: selected rank=%d; confirmation gain vs rank 4=%+.3f dB.\n', P, current_SNR_dB, selected_rank, confirmation_gain_dB);
    end

    %% -------------------- EVALUATE INDEPENDENT TEST OBSERVATIONS --------------------
    fprintf('Validation ranks frozen for P=%d. Starting test evaluation.\n', P);

    test_error = zeros(number_of_test_samples, 5);
    test_channel_energy = zeros(number_of_test_samples, 1);
    selected_hatim_rank_by_sample = zeros(number_of_test_samples, 1);

    for sample_index = 1:number_of_test_samples
        Y = test_Y_data(:, :, sample_index);
        G_true = test_G_data(:, :, sample_index);
        S = test_S_data(:, :, sample_index);
        B = test_B_data(:, :, sample_index);
        sigma2 = test_sigma2_data(sample_index);
        current_SNR_dB = test_SNR_data(sample_index);
        snr_index = find(SNR_grid == current_SNR_dB, 1);
        right_inverse = S'/(S*S');

        %% Shared full-pilot spectral initialization, N x K
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

        %% Algorithm 1: biased GS
        G_GS = G_spectral;

        for update_index = 1:maximum_updates
            G_previous = G_GS;
            G_GS = (Y.*exp(1i*angle(G_GS*S+B))-B)*right_inverse;

            if norm(G_GS-G_previous, 'fro')^2/max(norm(G_GS, 'fro')^2, eps) < convergence_tolerance
                break;
            end
        end

        test_error(sample_index, 1) = norm(G_GS-G_true, 'fro')^2;

        %% Algorithm 2: linearized GD
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

        test_error(sample_index, 3) = norm(G_EMGS-G_true, 'fro')^2;

        %% Algorithm 4: our validation-frozen hard-rank HS-GS
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

        test_error(sample_index, 4) = norm(G_OurHSGS-G_true, 'fro')^2;

        %% Algorithm 5: pilot-split rank selection using this observation's magnitudes
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

        %% Rerun the selected rank using all P pilots
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

        test_error(sample_index, 5) = norm(G_HatimHSGS-G_true, 'fro')^2;
        test_channel_energy(sample_index) = norm(G_true, 'fro')^2;

        if mod(sample_index, 200) == 0 || sample_index == number_of_test_samples
            fprintf('P=%d test samples processed: %d/%d.\n', P, sample_index, number_of_test_samples);
        end
    end

    test_error_by_P{p_index} = test_error;
    test_channel_energy_by_P{p_index} = test_channel_energy;
    hatim_rank_by_P{p_index} = selected_hatim_rank_by_sample;

    %% -------------------- SUMMARIZE PAIRED NMSE BY SNR --------------------
    for snr_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(snr_index);
        test_indices = find(test_SNR_data == current_SNR_dB);

        if numel(test_indices) ~= test_samples_per_SNR
            error('Unexpected test sample count for P=%d and SNR=%g dB.', P, current_SNR_dB);
        end

        nmse_dB = 10*log10(sum(test_error(test_indices, :), 1)./sum(test_channel_energy(test_indices)));
        mean_hatim_rank = mean(selected_hatim_rank_by_sample(test_indices));
        result_row_index = result_row_index+1;
        result_rows(result_row_index, :) = [P, current_SNR_dB, numel(test_indices), our_frozen_rank(p_index, snr_index), mean_hatim_rank, nmse_dB];

        fprintf('P=%d, SNR=%g dB: GS %.3f, GD %.3f, EM-GS %.3f, ours %.3f, pilot-split %.3f dB; mean pilot-split rank %.2f.\n', P, current_SNR_dB, nmse_dB(1), nmse_dB(2), nmse_dB(3), nmse_dB(4), nmse_dB(5), mean_hatim_rank);
    end
end

%% -------------------- SAVE RESULTS --------------------
result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','TestSamples','OurFrozenRank','MeanPilotSplitRank','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','OurHardRank_HSGS_NMSE_dB','PilotSplit_HSGS_NMSE_dB'});
result_csv = fullfile(output_folder, 'N8_ours_vs_pilot_split_hsgs_test_nmse.csv');
result_mat = fullfile(output_folder, 'N8_ours_vs_pilot_split_hsgs_test_nmse.mat');
result_png = fullfile(output_folder, 'N8_ours_vs_pilot_split_hsgs_test_nmse.png');
result_fig = fullfile(output_folder, 'N8_ours_vs_pilot_split_hsgs_test_nmse.fig');
configuration = struct('N', N, 'K', K, 'P_list', P_list, 'SNR_grid', SNR_grid, 'RSR_dB', RSR_dB, 'AoA_deg_range', AoA_deg_range, 'p_H', p_H, 'rank_candidates', rank_candidates, 'validation_samples_per_SNR', validation_samples_per_SNR, 'rank_selection_samples', rank_selection_samples, 'pilot_training_fraction', pilot_training_fraction, 'rank_selection_updates', rank_selection_updates, 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'our_frozen_rank', our_frozen_rank, 'rank_validation_gain_dB', rank_validation_gain_dB, 'data_source', 'Generated in memory; no input files loaded');

writetable(result_table, result_csv);
save(result_mat, 'result_table', 'configuration', 'our_frozen_rank', 'rank_validation_gain_dB', 'test_error_by_P', 'test_channel_energy_by_P', 'hatim_rank_by_P', '-v7.3');

figure('Color', 'w', 'Position', [100 100 1200 520]);

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    selected_rows = result_table.P == P;
    subplot(1, 2, p_index);
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows), '-o', 'LineWidth', 1.5, 'DisplayName', 'GS');
    hold on;
    plot(SNR_grid, result_table.GD_NMSE_dB(selected_rows), '-s', 'LineWidth', 1.5, 'DisplayName', 'GD');
    plot(SNR_grid, result_table.EMGS_NMSE_dB(selected_rows), '-d', 'LineWidth', 1.5, 'DisplayName', 'EM-GS');
    plot(SNR_grid, result_table.OurHardRank_HSGS_NMSE_dB(selected_rows), '-^', 'LineWidth', 1.7, 'DisplayName', 'Our validation-selected HS-GS');
    plot(SNR_grid, result_table.PilotSplit_HSGS_NMSE_dB(selected_rows), '-v', 'LineWidth', 1.7, 'DisplayName', 'Pilot-split HS-GS');
    grid on;
    xlabel('SNR (dB)');
    ylabel('Test NMSE (dB)');
    title(sprintf('N=%d, K=%d, P=%d, RSR=%g dB', N, K, P, RSR_dB));
    legend('Location', 'best');
    xticks(SNR_grid);
end

sgtitle('N=8 paired comparison: classical estimators and two hard-rank HS-GS selectors');
exportgraphics(gcf, result_png, 'Resolution', 200);
savefig(gcf, result_fig);

disp(result_table);
fprintf('\nComparison complete. Rank selection used validation only for our frozen-rank method.\n');
fprintf('Pilot-split rank selection used held-out magnitudes from each test observation, not its true channel.\n');
fprintf('Results saved under: %s\n', output_folder);