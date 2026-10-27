%% GS_GD_HS_GS_FINAL_N8_RSR12.m
clear;
clc;
close all;
rng(1);
workers = 6;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir'), mkdir(result_folder); end
%% -------------------- SIMULATION PARAMETERS --------------------
MC = 500;
N = 8;
K = 3;
P_list = [10 30];                             % 1 x 2: tested pilot counts
num_P = length(P_list);
SNR_dB = -5:5:30;                          % 1 x 8: tested SNR values in dB
num_SNR = length(SNR_dB);
RSR_dB = 12;                                  
G0_var = 0.1;                                 % Xu gives G^(0) entries ~ CN(0,0.1)
d_over_lambda = 0.5;                          % Local half-wavelength array-spacing choice
GS_max_iterations = 1000;
GD_max_iterations = 1000;
change_tolerance = 1e-10;
rank_list = 1:4;                                % 1 x 4: possible ranks of the 5 x 4 Hankel matrix
training_fraction = 0.70;
rank_selection_updates = 25;
HS_GS_updates = 100;
cadzow_sweeps = 4;
p_H = 3;                                      % Hnakel parameter, Produces an (N-p_H) x (p_H+1) = 5 x 4 Hankel matrix
%% -------------------- PHYSICAL CONSTANTS --------------------
hbar = 6.62607015e-34 / (2 * pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9 * q_charge * a0_bohr; 0];    % 3 x 1: transition dipole-moment vector
%% -------------------- DISPLAY THE SETTINGS --------------------
fprintf('\n');
fprintf('============================================================\n');
fprintf(' GS, GD, AND HS-GS FOR A 1D ATOMIC ARRAY\n');
fprintf('============================================================\n');
fprintf('Monte Carlo trials                     = %d\n', MC);
fprintf('Number of cells N                       = %d\n', N);
fprintf('Number of users K                       = %d\n', K);
fprintf('Pilot counts P                          = [%s]\n', num2str(P_list));
fprintf('RSR relative to average one-user power = %d dB\n', RSR_dB);
fprintf('HS-GS rank selection uses held-out pilots from each sample.\n');
fprintf('At maximum Hankel rank, HS-GS uses the EM-GS update without Hankel projection.\n');
fprintf('============================================================\n');
parallel_pool = gcp('nocreate');
if isempty(parallel_pool), parallel_pool = parpool('local', workers); end
fprintf('Parallel pool is active with %d workers.\n', parallel_pool.NumWorkers);
random_stream = parallel.pool.Constant(RandStream('Threefry', 'Seed', 1));
%% -------------------- ARRAYS USED TO STORE ERRORS --------------------
GS_error_sum = zeros(num_P, num_SNR);       % 2 x 8: accumulated GS squared errors
GD_error_sum = zeros(num_P, num_SNR);       % 2 x 8: accumulated GD squared errors
HS_GS_error_sum = zeros(num_P, num_SNR);    % 2 x 8: accumulated HS-GS squared errors
G_energy_sum = zeros(num_P, num_SNR);       % 2 x 8: accumulated true-channel energies
selected_rank_sum = zeros(num_P, num_SNR);  % 2 x 8: accumulated selected Hankel ranks
selected_maximum_rank_count = zeros(num_P, num_SNR);
%% -------------------- LOOP OVER P = 10 AND P = 30 --------------------
for P_index = 1:num_P
    P = P_list(P_index);
    fprintf('\n');
    fprintf('Starting P = %d\n', P);
    trial_GS_error = zeros(MC, num_SNR);
    trial_GD_error = zeros(MC, num_SNR);
    trial_HS_GS_error = zeros(MC, num_SNR);
    trial_G_energy = zeros(MC, num_SNR);
    trial_selected_rank = zeros(MC, num_SNR);
    trial_selected_maximum_rank = zeros(MC, num_SNR);
    %% -------------------- MONTE CARLO LOOP --------------------
    parfor mc = 1:MC
        sample_stream = random_stream.Value;
        sample_stream.Substream = mc + (P_index - 1) * MC;
        previous_global_stream = RandStream.setGlobalStream(sample_stream);
        % -------------------- GENERATE TRUE CHANNEL G --------------------
        G = zeros(N, K);                              % N x K = 8 x 3: true channel matrix
        for k = 1:K
            L_k = randi([3 7]);                       % Xu gives discrete L_k ~ Uniform{3,4,5,6,7}
            for l = 1:L_k
                alpha_lk_real = randn;
                alpha_lk_imag = randn;
                alpha_lk = (alpha_lk_real + 1j * alpha_lk_imag) / sqrt(2); % Xu gives alpha_lk ~ CN(0,1)
                theta_lk = 2 * pi * rand;              % Local theta_lk ~ Uniform(0,2*pi)
                phi_lk = 2 * pi * d_over_lambda * cos(theta_lk);
                epsilon_kl = sqrt(1 / 3) * randn(3, 1); % 3 x 1: Xu gives components ~ N(0,1/3)
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
        S = (S_real + 1j * S_imag) / sqrt(2);     % K x P: s_kp ~ CN(0,1)
        % -------------------- CALCULATE USER FIELD G*S --------------------
        G_times_S = G * S;                         % (N x K)(K x P) = N x P: combined user field
        % -------------------- GENERATE KNOWN REFERENCE B --------------------
        alpha_b_real = randn;
        alpha_b_imag = randn;
        alpha_b = sqrt(10 / 2) * (alpha_b_real + 1j * alpha_b_imag); % alpha_b ~ CN(0,10)
        theta_b = 2 * pi * rand;                  % theta_b ~ Uniform(0,2*pi); 
        phi_b = 2 * pi * d_over_lambda * cos(theta_b);
        epsilon_b = sqrt(1 / 3) * randn(3, 1);  % 3 x 1: Xu gives components ~ N(0,1/3); one vector is shared across cells
        mu_projection_b = mu_eg.' * epsilon_b;
        b = zeros(N, 1);                              % N x 1: spatial reference vector
        for i = 1:N
            reference_spatial_phase = exp(-1j * (i - 1) * phi_b);
            b(i) = (1 / hbar) * mu_projection_b * alpha_b * reference_spatial_phase;
        end
        s_b = ones(1, P);                           % 1 x P: local constant reference pilot
        B = b * s_b;                                  % (N x 1)(1 x P) = N x P: known reference matrix
        %% -------------------- SCALE B TO 12 dB SINGLE-USER RSR --------------------
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
        random_pilot_indices = randperm(P);     % creates a row vector containing every integer from 1 to P, in random order.
        P_training = round(training_fraction * P);     % Round off to nearest inteeger
        T = random_pilot_indices(1:P_training);    % 1 x P_T: training-pilot index set
        V = random_pilot_indices(P_training + 1:P); % 1 x P_V: validation-pilot index set
        %% -------------------- LOOP OVER SNR --------------------
        aggregate_signal_power = mean(abs(G_times_S(:)).^2);
        for SNR_index = 1:num_SNR
            current_SNR_dB = SNR_dB(SNR_index);
            current_SNR_linear = 10^(current_SNR_dB / 10);
            sigma2 = aggregate_signal_power / current_SNR_linear;
            W_real = randn(N, P);                     
            W_imag = randn(N, P);                    
            W = sqrt(sigma2 / 2) * (W_real + 1j * W_imag); % N x P: w_ip ~ CN(0,sigma2)
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
                v_i = V_i(:, principal_index);                         % (K+1) x 1: principal eigenvector
                r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2;
                g_i = r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K);     % K x 1: initialized channel of cell i after reference-phase alignment
                G_hat_GS(i, :) = g_i.';                    % 1 x K: store the spectral initialization for cell i
            end
            for gs_iteration = 1:GS_max_iterations
                G_hat_GS_previous = G_hat_GS;             % N x K: complete channel estimate from the preceding GS update
                G_hat_GS = (Y .* exp(1j * angle(G_hat_GS * S + B)) - B) * S' / (S * S');   % (N x P)(P x K)/(K x K) = N x K
                if norm(G_hat_GS - G_hat_GS_previous, 'fro')^2 / max(norm(G_hat_GS, 'fro')^2, eps) < change_tolerance
                    break;
                end
            end
            %% ============================================================
            % ALGORITHM 2: GD
            G_hat_GD = sqrt(G0_var / 2) * (randn(N, K) + 1j * randn(N, K)); % N x K: G^(0) ~ CN(0,0.1) for PGD; adopted here for GD
            Z = exp(-1j * angle(B));                    % N x P: known reference-phase matrix
            for gd_iteration = 1:GD_max_iterations
                G_hat_GD_previous = G_hat_GD;
                G_hat_GD = G_hat_GD + (1 / max(real(eig(S * S')))) * ((Y - abs(B) - real(Z .* (G_hat_GD * S))) .* conj(Z)) * S'; % (N x P)(P x K) = N x K
                if norm(G_hat_GD - G_hat_GD_previous, 'fro')^2 / max(norm(G_hat_GD, 'fro')^2, eps) < change_tolerance
                    break;
                end
            end
            %% ============================================================
            % ALGORITHM 3: HS-GS
            
            Y_T = Y(:, T);                                % N x P_T: HS-GS training magnitudes
            S_T = S(:, T);                                % K x P_T: HS-GS training pilots
            B_T = B(:, T);                                % N x P_T: HS-GS training reference
            Y_V = Y(:, V);                                % N x P_V: held-out validation magnitudes
            S_V = S(:, V);                                % K x P_V: held-out validation pilots
            B_V = B(:, V);                                % N x P_V: held-out validation reference
            validation_error = zeros(length(rank_list), 1);
            for r_index = 1:length(rank_list)
                r = rank_list(r_index);       % scalar: candidate Hankel rank
                G_candidate = zeros(N, K);    % N x K: candidate channel estimate
                for i = 1:N
                    y_i = Y_T(i, :).';      % P_T x 1: measurements at cell i
                    b_i = B_T(i, :).';       % P_T x 1: reference at cell i
                    A_bar_i = [S_T.', b_i];     % P_T x (K+1): augmented pilot-reference matrix
                    M_i = A_bar_i' * diag(y_i) * A_bar_i;   % ((K+1) x P_T)(P_T x P_T)(P_T x (K+1)) = (K+1)x(K+1)
                    [V_i, D_i] = eig(M_i);
                    [~, principal_index] = max(real(diag(D_i)));
                    v_i = V_i(:, principal_index);
                    r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2; % r_i: scalar scaling factor
                    G_candidate(i, :) = (r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K)).';   % G_candidate: N x K
                end
                for hs_update = 1:rank_selection_updates
                    kappa = (2 / sigma2) * Y_T .* abs(G_candidate * S_T + B_T);  % kappa: N x P_T
                    R_kappa = besseli(1, kappa, 1) ./ besseli(0, kappa, 1);      % R_kappa: N x P_T
                    U = Y_T .* exp(1j * angle(G_candidate * S_T + B_T)) .* R_kappa;  % U: N x P_T Reconstruected Complex signal matrix
                    G_tilde = (U - B_T) * S_T' / (S_T * S_T'); % N x K : Intermediate channel before hankel processing

                    if r < max(rank_list)
                        G_processed = zeros(N, K);   % N x K
                        for k = 1:K
                            g_k_tilde = G_tilde(:, k);   % N x 1: channel column for user k
                            for sweep = 1:cadzow_sweeps
                                H_rows = N - p_H;  % 5
                                H_columns = p_H + 1; % 4
                                H = zeros(H_rows, H_columns);  % H: (N - p_H) x (p_H + 1) = 5x4
                                for row = 1:H_rows
                                    H(row, :) = g_k_tilde(row:row + H_columns - 1).';
                                end
                                [U_H, Sigma_H, V_H] = svd(H);
                                H_rank_r = U_H(:, 1:r) * Sigma_H(1:r, 1:r) * V_H(:, 1:r)';  % Rank-r approximation of H
    
                                g_k_reconstructed = zeros(N, 1);
                                copy_count = zeros(N, 1);
                                for row = 1:H_rows
                                    for column = 1:H_columns
                                        g_k_reconstructed(row + column - 1) = g_k_reconstructed(row + column - 1) + H_rank_r(row, column);
                                        copy_count(row + column - 1) = copy_count(row + column - 1) + 1;
                                    end
                                end
                                g_k_tilde = g_k_reconstructed ./ copy_count;  % N x 1: anti-diagonal averaging reconstructs the channel vector
                            end
                            G_processed(:, k) = g_k_tilde;  % N x 1 inserted into column k of G_processed: N x K
                        end
                        G_candidate = G_processed;  % N x K: updated HS-GS candidate channel
                    else
                        G_candidate = G_tilde;  % Maximum rank: perform the EM-GS update without Hankel projection
                    end
                end
                Y_hat_V = abs(G_candidate * S_V + B_V); % N x P_V: predicted validation magnitudes
                validation_error(r_index) = norm(Y_V - Y_hat_V, 'fro')^2;  % Validation error : 4x1
            end
            [minimum_validation_error, best_rank_index] = min(validation_error);
            if ~isfinite(minimum_validation_error)
                error('HS-GS rank validation produced a nonfinite error.');
            end
            selected_rank = rank_list(best_rank_index);
            G_hat_HS_GS = zeros(N, K);  % G_hat_HS_GS: N x K final HS-GS channel estimate to evaluate
            for i = 1:N
                y_i = Y(i, :).';     % y_i: P x 1 measurement vector for cell i
                b_i = B(i, :).';     % b_i: P x 1 reference vector for cell i
                A_bar_i = [S.', b_i];     % A_bar_i: P x (K+1)
                M_i = A_bar_i' * diag(y_i) * A_bar_i;
                [V_i, D_i] = eig(M_i);
                [~, principal_index] = max(real(diag(D_i)));
                v_i = V_i(:, principal_index);
                r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2;
                G_hat_HS_GS(i, :) = (r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K)).';
            end
            for hs_update = 1:HS_GS_updates
                kappa = (2 / sigma2) * Y .* abs(G_hat_HS_GS * S + B);   % kappa: N x P
                R_kappa = besseli(1, kappa, 1) ./ besseli(0, kappa, 1);  % R_kappa: N x P
                U = Y .* exp(1j * angle(G_hat_HS_GS * S + B)) .* R_kappa;  % U: N x P
                G_tilde = (U - B) * S' / (S * S');     % G_tilde: N x K

                if selected_rank < max(rank_list)
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
                            [U_H, Sigma_H, V_H] = svd(H);
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
                else
                    G_hat_HS_GS = G_tilde;  % Maximum rank: this update is exactly EM-GS
                end
            end
            %% -------------------- ACCUMULATE SQUARED ERRORS --------------------
            trial_GS_error(mc, SNR_index) = norm(G_hat_GS - G, 'fro')^2;
            trial_GD_error(mc, SNR_index) = norm(G_hat_GD - G, 'fro')^2;
            trial_HS_GS_error(mc, SNR_index) = norm(G_hat_HS_GS - G, 'fro')^2;
            trial_G_energy(mc, SNR_index) = norm(G, 'fro')^2;
            trial_selected_rank(mc, SNR_index) = selected_rank;
            trial_selected_maximum_rank(mc, SNR_index) = (selected_rank == max(rank_list));
        end
        RandStream.setGlobalStream(previous_global_stream);
        if mod(mc, 50) == 0 || mc == MC
            fprintf('P = %d, completed trial %d of %d\n', P, mc, MC);
        end
    end
    GS_error_sum(P_index, :) = sum(trial_GS_error, 1);
    GD_error_sum(P_index, :) = sum(trial_GD_error, 1);
    HS_GS_error_sum(P_index, :) = sum(trial_HS_GS_error, 1);
    G_energy_sum(P_index, :) = sum(trial_G_energy, 1);
    selected_rank_sum(P_index, :) = sum(trial_selected_rank, 1);
    selected_maximum_rank_count(P_index, :) = sum(trial_selected_maximum_rank, 1);
end
%% -------------------- CALCULATE FINAL NMSE --------------------
NMSE_GS = GS_error_sum ./ G_energy_sum;
NMSE_GD = GD_error_sum ./ G_energy_sum;
NMSE_HS_GS = HS_GS_error_sum ./ G_energy_sum;
NMSE_GS_dB = 10 * log10(NMSE_GS);
NMSE_GD_dB = 10 * log10(NMSE_GD);
NMSE_HS_GS_dB = 10 * log10(NMSE_HS_GS);
mean_selected_rank = selected_rank_sum / MC;
fraction_maximum_rank = selected_maximum_rank_count / MC;

%% -------------------- DISPLAY RESULTS --------------------
for P_index = 1:num_P
    for SNR_index = 1:num_SNR
        fprintf('P=%2d, SNR=%3d dB: GS=%8.3f dB, GD=%8.3f dB, HS-GS=%8.3f dB, mean rank=%5.2f, max-rank fraction=%5.3f\n', P_list(P_index), SNR_dB(SNR_index), NMSE_GS_dB(P_index, SNR_index), NMSE_GD_dB(P_index, SNR_index), NMSE_HS_GS_dB(P_index, SNR_index), mean_selected_rank(P_index, SNR_index), fraction_maximum_rank(P_index, SNR_index));
    end
end

%% -------------------- SAVE NUMERICAL RESULTS --------------------
result_file = fullfile(result_folder, 'GS_GD_HS_GS_FINAL_N8_RSR12_results.mat');
table_file = fullfile(result_folder, 'GS_GD_HS_GS_FINAL_N8_RSR12_table.csv');
figure_file = fullfile(result_folder, 'GS_GD_HS_GS_FINAL_N8_RSR12.png');
figure_editable_file = fullfile(result_folder, 'GS_GD_HS_GS_FINAL_N8_RSR12.fig');
save(result_file, 'SNR_dB', 'P_list', 'NMSE_GS', 'NMSE_GD', 'NMSE_HS_GS', 'NMSE_GS_dB', 'NMSE_GD_dB', 'NMSE_HS_GS_dB', 'mean_selected_rank', 'fraction_maximum_rank', 'MC', 'N', 'K', 'RSR_dB', 'rank_list', 'rank_selection_updates', 'HS_GS_updates', 'cadzow_sweeps', 'training_fraction', 'workers');
P_column = zeros(num_P * num_SNR, 1);
SNR_column = zeros(num_P * num_SNR, 1);
GS_column = zeros(num_P * num_SNR, 1);
GD_column = zeros(num_P * num_SNR, 1);
HS_GS_column = zeros(num_P * num_SNR, 1);
Mean_rank_column = zeros(num_P * num_SNR, 1);
Maximum_rank_fraction_column = zeros(num_P * num_SNR, 1);
row_number = 0;
for P_index = 1:num_P
    for SNR_index = 1:num_SNR
        row_number = row_number + 1;
        P_column(row_number) = P_list(P_index);
        SNR_column(row_number) = SNR_dB(SNR_index);
        GS_column(row_number) = NMSE_GS_dB(P_index, SNR_index);
        GD_column(row_number) = NMSE_GD_dB(P_index, SNR_index);
        HS_GS_column(row_number) = NMSE_HS_GS_dB(P_index, SNR_index);
        Mean_rank_column(row_number) = mean_selected_rank(P_index, SNR_index);
        Maximum_rank_fraction_column(row_number) = fraction_maximum_rank(P_index, SNR_index);
    end
end
results_table = table(P_column, SNR_column, GS_column, GD_column, HS_GS_column, Mean_rank_column, Maximum_rank_fraction_column, 'VariableNames', {'P', 'SNR_dB', 'GS_NMSE_dB', 'GD_NMSE_dB', 'HS_GS_NMSE_dB', 'Mean_selected_rank', 'Fraction_maximum_rank'});
writetable(results_table, table_file);

%% -------------------- PLOT THREE ALGORITHMS --------------------
figure('Color', 'w', 'WindowState', 'maximized');
hold on;
grid on;
box on;
GS_color = [0.00 0.45 0.74];
GD_color = [0.85 0.33 0.10];
HS_GS_color = [0.47 0.67 0.19];
line_width = 2;
marker_size = 7;
plot(SNR_dB, NMSE_GS_dB(1, :), '--o', 'Color', GS_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'GS, P = 10');
plot(SNR_dB, NMSE_GD_dB(1, :), '--s', 'Color', GD_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'GD, P = 10');
plot(SNR_dB, NMSE_HS_GS_dB(1, :), '--^', 'Color', HS_GS_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'HS-GS, P = 10');
plot(SNR_dB, NMSE_GS_dB(2, :), '-o', 'Color', GS_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'GS, P = 30');
plot(SNR_dB, NMSE_GD_dB(2, :), '-s', 'Color', GD_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'GD, P = 30');
plot(SNR_dB, NMSE_HS_GS_dB(2, :), '-^', 'Color', HS_GS_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'HS-GS, P = 30');
xlabel('SNR (dB)');
ylabel('NMSE (dB)');
title('GS, GD, and HS-GS, N = 8, K = 3, RSR = 12 dB');
legend('Location', 'best');
xticks(SNR_dB);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
exportgraphics(gcf, figure_file, 'Resolution', 300);
savefig(gcf, figure_editable_file);
fprintf('\nSimulation completed.\n');
fprintf('Results saved to: %s\n', result_file);
fprintf('Table saved to: %s\n', table_file);
fprintf('Figure saved to: %s\n', figure_file);
