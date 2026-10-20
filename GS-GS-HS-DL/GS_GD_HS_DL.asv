%% GS_GD_HS_DL.m
clear;
clc;
close all;
rng(1);
%% -------------------- SIMULATION PARAMETERS --------------------
MC = 500;
N = 8;
K = 3;
P_list = [10 30];                             % 1 x 2: tested pilot counts
num_P = length(P_list);
SNR_dB = -5:5:30;                          % 1 x 8: tested SNR values in dB
num_SNR = length(SNR_dB);
RSR_dB = 25;                                  % Local choice because Xu assumes a strong reference but gives no RSR value
G0_var = 0.1;                                 % Xu gives G^(0) entries ~ CN(0,0.1) for PGD
d_over_lambda = 0.5;                          % Local half-wavelength array-spacing choice
GS_max_iterations = 1000;
GD_max_iterations = 1000;
change_tolerance = 1e-10;
rank_list = 1:4;                                % 1 x 4: possible ranks of the 5 x 4 Hankel matrix
training_fraction = 0.70;
rank_selection_updates = 25;
HS_GS_updates = 100;
cadzow_sweeps = 4;
p_H = 3;                                      % Produces an (N-p_H) x (p_H+1) = 5 x 4 Hankel matrix
%% -------------------- PHYSICAL CONSTANTS --------------------
hbar = 6.62607015e-34 / (2 * pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9 * q_charge * a0_bohr; 0]; % 3 x 1: transition dipole-moment vector
%% -------------------- DISPLAY THE SETTINGS --------------------
fprintf('\n');
fprintf('============================================================\n');
fprintf(' GS, GD, HS-GS AND DL FOR A 1D ATOMIC ARRAY\n');
fprintf('============================================================\n');
fprintf('Monte Carlo trials                     = %d\n', MC);
fprintf('Number of cells N                       = %d\n', N);
fprintf('Number of users K                       = %d\n', K);
fprintf('Pilot counts P                          = [%s]\n', num2str(P_list));
fprintf('RSR relative to average one-user power = %d dB\n', RSR_dB);
fprintf('============================================================\n');
%% -------------------- ARRAYS USED TO STORE ERRORS --------------------
GS_error_sum = zeros(num_P, num_SNR);       % 2 x 8: accumulated GS squared errors
GD_error_sum = zeros(num_P, num_SNR);       % 2 x 8: accumulated GD squared errors
HS_GS_error_sum = zeros(num_P, num_SNR);    % 2 x 8: accumulated HS-GS squared errors
DL_error_sum = zeros(num_P, num_SNR);        % 2 x 8: accumulated DL squared errors
G_energy_sum = zeros(num_P, num_SNR);       % 2 x 8: accumulated true-channel energies
selected_rank_sum = zeros(num_P, num_SNR);  % 2 x 8: accumulated selected Hankel ranks
%% -------------------- LOOP OVER P = 10 AND P = 30 --------------------
for P_index = 1:num_P
    P = P_list(P_index);
    fprintf('\n');
    fprintf('Starting P = %d\n', P);
    %% -------------------- MONTE CARLO LOOP --------------------
    for mc = 1:MC
        % -------------------- GENERATE TRUE CHANNEL G --------------------
        G = zeros(N, K);                              % N x K = 8 x 3: true channel matrix
        for k = 1:K
            L_k = randi([3 7]);                       % Xu gives discrete L_k ~ Uniform{3,4,5,6,7}
            for l = 1:L_k
                alpha_lk_real = randn;
                alpha_lk_imag = randn;
                alpha_lk = (alpha_lk_real + 1j * alpha_lk_imag) / sqrt(2); % Xu gives alpha_lk ~ CN(0,1)
                theta_lk = 2 * pi * rand;              % Local theta_lk ~ Uniform(0,2*pi); Xu does not specify this distribution
                phi_lk = 2 * pi * d_over_lambda * cos(theta_lk);
                epsilon_kl = sqrt(1 / 3) * randn(3, 1); % 3 x 1: Xu gives components ~ N(0,1/3); one vector per path is the HS spatial-model choice
                mu_projection_lk = mu_eg.' * epsilon_kl;
                for i = 1:N
                    spatial_phase = exp(-1j * (i - 1) * phi_lk);
                    g_path = (1 / hbar) * mu_projection_lk * alpha_lk * spatial_phase;
                    G(i, k) = G(i, k) + g_path;
                end
            end
        end
        % -------------------- GENERATE PILOT MATRIX S --------------------
        S_real = randn(K, P);                     % K x P: real part of pilot matrix
        S_imag = randn(K, P);                     % K x P: imaginary part of pilot matrix
        S = (S_real + 1j * S_imag) / sqrt(2); % K x P: Xu gives iid s_kp ~ CN(0,1)
        % -------------------- CALCULATE USER FIELD G*S --------------------
        G_times_S = G * S;                         % (N x K)(K x P) = N x P: combined user field
        % -------------------- GENERATE KNOWN REFERENCE B --------------------
        alpha_b_real = randn;
        alpha_b_imag = randn;
        alpha_b = sqrt(10 / 2) * (alpha_b_real + 1j * alpha_b_imag); % Xu gives alpha_b ~ CN(0,10)
        theta_b = 2 * pi * rand;                  % Local theta_b ~ Uniform(0,2*pi); Xu does not specify this distribution
        phi_b = 2 * pi * d_over_lambda * cos(theta_b);
        epsilon_b = sqrt(1 / 3) * randn(3, 1);  % 3 x 1: Xu gives components ~ N(0,1/3); one vector is shared across cells
        mu_projection_b = mu_eg.' * epsilon_b;
        b = zeros(N, 1);                              % N x 1: spatial reference vector
        for i = 1:N
            reference_spatial_phase = exp(-1j * (i - 1) * phi_b);
            b(i) = (1 / hbar) * mu_projection_b * alpha_b * reference_spatial_phase;
        end
        s_b = ones(1, P);                           % 1 x P: local constant reference pilot; Xu does not specify its sequence
        B = b * s_b;                                  % (N x 1)(1 x P) = N x P: known reference matrix
        %% -------------------- SCALE B TO 25 dB SINGLE-USER RSR --------------------
        single_user_power_sum = 0;
        for k = 1:K
            g_k = G(:, k);                                % N x 1: channel vector of user k
            s_k = S(k, :);                                % 1 x P: pilot row of user k
            gk_times_sk = g_k * s_k;                  % (N x 1)(1 x P) = N x P: field from user k
            single_user_power = mean(abs(gk_times_sk(:)).^2);
            single_user_power_sum = single_user_power_sum + single_user_power;
        end
        average_single_user_power = single_user_power_sum / K;
        target_reference_power = 10^(RSR_dB / 10) * average_single_user_power;
        original_reference_power = mean(abs(B(:)).^2);
        reference_scale = sqrt(target_reference_power / original_reference_power);
        B = reference_scale * B;
        %% -------------------- CREATE TRAINING AND VALIDATION PILOT INDICES --------------------
        random_pilot_indices = randperm(P);
        P_training = round(training_fraction * P);
        T = random_pilot_indices(1:P_training);    % 1 x P_T: training-pilot index set
        V = random_pilot_indices(P_training + 1:P); % 1 x P_V: validation-pilot index set
        %% -------------------- LOOP OVER SNR --------------------
        aggregate_signal_power = mean(abs(G_times_S(:)).^2);
        for SNR_index = 1:num_SNR
            current_SNR_dB = SNR_dB(SNR_index);
            current_SNR_linear = 10^(current_SNR_dB / 10);
            sigma2 = aggregate_signal_power / current_SNR_linear;
            W_real = randn(N, P);                     % N x P: real Gaussian-noise samples
            W_imag = randn(N, P);                     % N x P: imaginary Gaussian-noise samples
            W = sqrt(sigma2 / 2) * (W_real + 1j * W_imag); % N x P: Xu gives w_ip ~ CN(0,sigma2)
            Y = abs(G_times_S + B + W);              % N x P: magnitude-only observations
            %% ============================================================
            % ALGORITHM 1: GS
            G_hat_GS = zeros(N, K);                     % N x K: GS channel estimate
            for i = 1:N
                y_i = Y(i, :).';                            % P x 1: measurements at cell i
                b_i = B(i, :).';                            % P x 1: known reference at cell i
                A_bar_i = [S.', b_i];                     % [P x K, P x 1] = P x (K+1): augmented matrix for cell i
                M_i = A_bar_i' * diag(y_i) * A_bar_i;       % ((K+1) x P)(P x P)(P x (K+1)) = (K+1) x (K+1)
                [V_i, D_i] = eig(M_i);                      % V_i and D_i are (K+1) x (K+1)
                [~, principal_index] = max(real(diag(D_i)));
                v_i = V_i(:, principal_index);              % (K+1) x 1: principal eigenvector
                v_i = v_i / norm(v_i);
                r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2;
                g_i = r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K); % K x 1: initialized channel of cell i after reference-phase alignment
                G_hat_GS(i, :) = g_i.';                    % 1 x K: store the spectral initialization for cell i
            end
            for gs_iteration = 1:GS_max_iterations
                G_hat_GS_previous = G_hat_GS;             % N x K: complete channel estimate from the preceding GS update
                G_hat_GS = (Y .* exp(1j * angle(G_hat_GS * S + B)) - B) * S' / (S * S'); % (N x P)(P x K)/(K x K) = N x K
                if norm(G_hat_GS - G_hat_GS_previous, 'fro')^2 / max(norm(G_hat_GS, 'fro')^2, eps) < change_tolerance
                    break;
                end
            end
            %% ============================================================
            % ALGORITHM 2: GD
            G_hat_GD = sqrt(G0_var / 2) * (randn(N, K) + 1j * randn(N, K)); % N x K: Xu gives G^(0) ~ CN(0,0.1) for PGD; adopted here for GD
            Z = exp(-1j * angle(B));                    % N x P: known reference-phase matrix
            for gd_iteration = 1:GD_max_iterations
                G_hat_GD_previous = G_hat_GD;
                G_hat_GD = G_hat_GD + (1 / max(real(eig(S * S')))) * ((Y - abs(B) - real(Z .* (G_hat_GD * S))) .* conj(Z)) * S'; % (N x P)(P x K) = N x K
                if norm(G_hat_GD - G_hat_GD_previous, 'fro')^2 / max(norm(G_hat_GD, 'fro')^2, eps) < change_tolerance
                    break;
                end
            end
            %% ============================================================
            %% ALGORITHM 3: HS-GS
            %% ============================================================
            Y_T = Y(:, T);                                % N x P_T: HS-GS training magnitudes
            S_T = S(:, T);                                % K x P_T: HS-GS training pilots
            B_T = B(:, T);                                % N x P_T: HS-GS training reference
            Y_V = Y(:, V);                                % N x P_V: held-out validation magnitudes
            S_V = S(:, V);                                % K x P_V: held-out validation pilots
            B_V = B(:, V);                                % N x P_V: held-out validation reference
            validation_error = zeros(length(rank_list), 1);
            for r_index = 1:length(rank_list)
                r = rank_list(r_index);
                G_candidate = zeros(N, K);
                for i = 1:N
                    y_i = Y_T(i, :).';
                    b_i = B_T(i, :).';
                    A_bar_i = [S_T.', b_i];
                    M_i = A_bar_i' * diag(y_i) * A_bar_i;
                    [V_i, D_i] = eig(M_i);
                    [~, principal_index] = max(real(diag(D_i)));
                    v_i = V_i(:, principal_index);
                    v_i = v_i / norm(v_i);
                    r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2;
                    G_candidate(i, :) = (r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K)).';
                end
                for hs_update = 1:rank_selection_updates
                    kappa = (2 / sigma2) * Y_T .* abs(G_candidate * S_T + B_T);
                    R_kappa = besseli(1, kappa, 1) ./ besseli(0, kappa, 1);
                    U = Y_T .* exp(1j * angle(G_candidate * S_T + B_T)) .* R_kappa;
                    G_tilde = (U - B_T) * S_T' / (S_T * S_T');
                    G_processed = zeros(N, K);
                    for k = 1:K
                        g_k_tilde = G_tilde(:, k);
                        for sweep = 1:cadzow_sweeps
                            H_rows = N - p_H;
                            H_columns = p_H + 1;
                            H = zeros(H_rows, H_columns);
                            for row = 1:H_rows
                                H(row, :) = g_k_tilde(row:row + H_columns - 1).';
                            end
                            [U_H, Sigma_H, V_H] = svd(H, 'econ');
                            H_rank_r = U_H(:, 1:r) * Sigma_H(1:r, 1:r) * V_H(:, 1:r)';
                            g_k_reconstructed = zeros(N, 1);
                            copy_count = zeros(N, 1);
                            for row = 1:H_rows
                                for column = 1:H_columns
                                    g_k_reconstructed(row + column - 1) = g_k_reconstructed(row + column - 1) + H_rank_r(row, column);
                                    copy_count(row + column - 1) = copy_count(row + column - 1) + 1;
                                end
                            end
                            g_k_tilde = g_k_reconstructed ./ copy_count;
                        end
                        G_processed(:, k) = g_k_tilde;
                    end
                    G_candidate = G_processed;
                end
                Y_hat_V = abs(G_candidate * S_V + B_V); % N x P_V: predicted validation magnitudes
                validation_error(r_index) = norm(Y_V - Y_hat_V, 'fro')^2;
            end
            [minimum_validation_error, best_rank_index] = min(validation_error);
            if ~isfinite(minimum_validation_error)
                error('HS-GS rank validation produced a nonfinite error.');
            end
            selected_rank = rank_list(best_rank_index);
            G_hat_HS_GS = zeros(N, K);
            for i = 1:N
                y_i = Y(i, :).';
                b_i = B(i, :).';
                A_bar_i = [S.', b_i];
                M_i = A_bar_i' * diag(y_i) * A_bar_i;
                [V_i, D_i] = eig(M_i);
                [~, principal_index] = max(real(diag(D_i)));
                v_i = V_i(:, principal_index);
                v_i = v_i / norm(v_i);
                r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2;
                G_hat_HS_GS(i, :) = (r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K)).';
            end
            for hs_update = 1:HS_GS_updates
                kappa = (2 / sigma2) * Y .* abs(G_hat_HS_GS * S + B);
                R_kappa = besseli(1, kappa, 1) ./ besseli(0, kappa, 1);
                U = Y .* exp(1j * angle(G_hat_HS_GS * S + B)) .* R_kappa;
                G_tilde = (U - B) * S' / (S * S');
                G_processed = zeros(N, K);
                for k = 1:K
                    g_k_tilde = G_tilde(:, k);
                    for sweep = 1:cadzow_sweeps
                        H_rows = N - p_H;
                        H_columns = p_H + 1;
                        H = zeros(H_rows, H_columns);
                        for row = 1:H_rows
                            H(row, :) = g_k_tilde(row:row + H_columns - 1).';
                        end
                        [U_H, Sigma_H, V_H] = svd(H, 'econ');
                        H_rank_r = U_H(:, 1:selected_rank) * Sigma_H(1:selected_rank, 1:selected_rank) * V_H(:, 1:selected_rank)';
                        g_k_reconstructed = zeros(N, 1);
                        copy_count = zeros(N, 1);
                        for row = 1:H_rows
                            for column = 1:H_columns
                                g_k_reconstructed(row + column - 1) = g_k_reconstructed(row + column - 1) + H_rank_r(row, column);
                                copy_count(row + column - 1) = copy_count(row + column - 1) + 1;
                            end
                        end
                        g_k_tilde = g_k_reconstructed ./ copy_count;
                    end
                    G_processed(:, k) = g_k_tilde;
                end
                G_hat_HS_GS = G_processed;
            end
            %% ============================================================
            % ALGORITHM 4: DL
            if ~exist('DL_parameters', 'var')
                model_folder = fullfile(fileparts(mfilename('fullpath')), 'unfolded_dl_results_RSR25dB');
                model_P10 = load(fullfile(model_folder, 'model_P10.mat'));
                model_P30 = load(fullfile(model_folder, 'model_P30.mat'));
                DL_parameters = {double(extractdata(model_P10.best_parameters)), double(extractdata(model_P30.best_parameters))};
                DL_channel_scales = [double(model_P10.training_channel_scale), double(model_P30.training_channel_scale)];
                fprintf('DL models loaded for Algorithm 4.\n');
            end
            Y_normalized = single(Y / DL_channel_scales(P_index));
            S_real = single(real(S));
            S_imag = single(imag(S));
            B_real = single(real(B) / DL_channel_scales(P_index));
            B_imag = single(imag(B) / DL_channel_scales(P_index));
            noise_ratio = single(sigma2 / DL_channel_scales(P_index)^2);
            pilot_bound = single(max(real(eig(S * S'))));
            real_channel = zeros(N, K, 1, 'single');
            imag_channel = zeros(N, K, 1, 'single');
            previous_real_channel = real_channel;
            previous_imag_channel = imag_channel;
            noise_level = log(1 + noise_ratio / max(pilot_bound, single(1e-6)));
            for layer_index = 1:size(DL_parameters{P_index}, 1)
                parameters = DL_parameters{P_index};
                step_size = 1.9 / (1 + exp(-(parameters(layer_index, 1) + parameters(layer_index, 4) * noise_level)));
                momentum = 0.95 / (1 + exp(-(parameters(layer_index, 2) + parameters(layer_index, 5) * noise_level)));
                ridge_weight = log(1 + exp(parameters(layer_index, 3)));
                extrapolated_real = real_channel + momentum * (real_channel - previous_real_channel);
                extrapolated_imag = imag_channel + momentum * (imag_channel - previous_imag_channel);
                field_real = pagemtimes(extrapolated_real, reshape(S_real, K, P, 1)) - pagemtimes(extrapolated_imag, reshape(S_imag, K, P, 1)) + reshape(B_real, N, P, 1);
                field_imag = pagemtimes(extrapolated_real, reshape(S_imag, K, P, 1)) + pagemtimes(extrapolated_imag, reshape(S_real, K, P, 1)) + reshape(B_imag, N, P, 1);
                amplitude = sqrt(field_real.^2 + field_imag.^2 + single(1e-12));
                residual_factor = (amplitude - reshape(Y_normalized, N, P, 1)) ./ amplitude;
                error_real = residual_factor .* field_real;
                error_imag = residual_factor .* field_imag;
                pilot_real_transpose = reshape(S_real.', P, K, 1);
                pilot_imag_transpose = reshape(S_imag.', P, K, 1);
                gradient_real = pagemtimes(error_real, pilot_real_transpose) + pagemtimes(error_imag, pilot_imag_transpose);
                gradient_imag = -pagemtimes(error_real, pilot_imag_transpose) + pagemtimes(error_imag, pilot_real_transpose);
                previous_real_channel = real_channel;
                previous_imag_channel = imag_channel;
                denominator = 1 + step_size * ridge_weight * noise_ratio / max(pilot_bound, single(1e-6));
                real_channel = (extrapolated_real - step_size * gradient_real / max(pilot_bound, single(1e-6))) ./ denominator;
                imag_channel = (extrapolated_imag - step_size * gradient_imag / max(pilot_bound, single(1e-6))) ./ denominator;
            end
            G_hat_DL = DL_channel_scales(P_index) * (double(real_channel(:, :, 1)) + 1j * double(imag_channel(:, :, 1)));
            %% -------------------- ACCUMULATE SQUARED ERRORS --------------------
            GS_error_sum(P_index, SNR_index) = GS_error_sum(P_index, SNR_index) + norm(G_hat_GS - G, 'fro')^2;
            GD_error_sum(P_index, SNR_index) = GD_error_sum(P_index, SNR_index) + norm(G_hat_GD - G, 'fro')^2;
            HS_GS_error_sum(P_index, SNR_index) = HS_GS_error_sum(P_index, SNR_index) + norm(G_hat_HS_GS - G, 'fro')^2;
            DL_error_sum(P_index, SNR_index) = DL_error_sum(P_index, SNR_index) + norm(G_hat_DL - G, 'fro')^2;
            G_energy_sum(P_index, SNR_index) = G_energy_sum(P_index, SNR_index) + norm(G, 'fro')^2;
            selected_rank_sum(P_index, SNR_index) = selected_rank_sum(P_index, SNR_index) + selected_rank;
        end
        if mod(mc, 10) == 0 || mc == MC
            fprintf('P = %d, completed trial %d of %d\n', P, mc, MC);
        end
    end
end
%% -------------------- CALCULATE FINAL NMSE --------------------
NMSE_GS = GS_error_sum ./ G_energy_sum;
NMSE_GD = GD_error_sum ./ G_energy_sum;
NMSE_HS_GS = HS_GS_error_sum ./ G_energy_sum;
NMSE_DL = DL_error_sum ./ G_energy_sum;
NMSE_GS_dB = 10 * log10(NMSE_GS);
NMSE_GD_dB = 10 * log10(NMSE_GD);
NMSE_HS_GS_dB = 10 * log10(NMSE_HS_GS);
NMSE_DL_dB = 10 * log10(NMSE_DL);
mean_selected_rank = selected_rank_sum / MC;
%% -------------------- SAVE NUMERICAL RESULTS --------------------
save('GS_GD_HS_DL_fresh_results.mat', 'SNR_dB', 'P_list', 'NMSE_GS', 'NMSE_GD', 'NMSE_HS_GS', 'NMSE_DL', 'NMSE_GS_dB', 'NMSE_GD_dB', 'NMSE_HS_GS_dB', 'NMSE_DL_dB', 'mean_selected_rank', 'MC', 'N', 'K', 'RSR_dB');
%% -------------------- ONE FIGURE FOR P = 10 AND P = 30 --------------------
figure('Color', 'w', 'WindowState', 'maximized');
hold on;
grid on;
box on;
GS_color = [0.00 0.45 0.74];
GD_color = [0.85 0.33 0.10];
HS_GS_color = [0.47 0.67 0.19];
DL_color = [0.49 0.18 0.56];
line_width = 2;
marker_size = 7;
plot(SNR_dB, NMSE_GS_dB(1, :), '--o', 'Color', GS_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'GS, P = 10');
plot(SNR_dB, NMSE_GD_dB(1, :), '--s', 'Color', GD_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'GD, P = 10');
plot(SNR_dB, NMSE_HS_GS_dB(1, :), '--^', 'Color', HS_GS_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'HS-GS, P = 10');
plot(SNR_dB, NMSE_DL_dB(1, :), '--d', 'Color', DL_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'DL, P = 10');
plot(SNR_dB, NMSE_GS_dB(2, :), '-o', 'Color', GS_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'GS, P = 30');
plot(SNR_dB, NMSE_GD_dB(2, :), '-s', 'Color', GD_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'GD, P = 30');
plot(SNR_dB, NMSE_HS_GS_dB(2, :), '-^', 'Color', HS_GS_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'HS-GS, P = 30');
plot(SNR_dB, NMSE_DL_dB(2, :), '-d', 'Color', DL_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'DL, P = 30');
xlabel('SNR (dB)');
ylabel('NMSE (dB)');
title('1D magnitude-only channel estimation, N = 8, K = 3, RSR = 25 dB');
legend('Location', 'best');
xticks(SNR_dB);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
exportgraphics(gcf, 'GS_GD_HS_DL_fresh_nmse_vs_snr.png', 'Resolution', 300);
savefig(gcf, 'GS_GD_HS_DL_fresh_nmse_vs_snr.fig');
fprintf('\n');
fprintf('Simulation completed.\n');
fprintf('Numerical results saved in GS_GD_HS_DL_fresh_results.mat\n');
fprintf('Figure saved in GS_GD_HS_DL_fresh_nmse_vs_snr.png\n');
