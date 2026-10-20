%% GS_GD_HS_DL_final.m
% Final paired comparison using the saved test data and trained DL models.

clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, 'DL_dataset_RSR25dB');
model_folder = fullfile(script_folder, 'unfolded_dl_results_RSR25dB');
result_folder = fullfile(script_folder, 'final_four_method_results_RSR25dB');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

P_list = [10 30];
SNR_dB = -5:5:30;
N = 8;
K = 3;
RSR_dB = 25;
GS_max_iterations = 1000;
GD_max_iterations = 1000;
HS_GS_updates = 100;
rank_list = 1:4;
rank_selection_updates = 25;
cadzow_sweeps = 4;
p_H = 3;
change_tolerance = 1e-10;
G0_var = 0.1;
training_fraction = 0.70;

NMSE_GS = zeros(length(P_list), length(SNR_dB));
NMSE_GD = zeros(length(P_list), length(SNR_dB));
NMSE_HS_GS = zeros(length(P_list), length(SNR_dB));
NMSE_DL = zeros(length(P_list), length(SNR_dB));
selected_rank_mean = zeros(length(P_list), length(SNR_dB));

for P_index = 1:length(P_list)
    P = P_list(P_index);
    test_source = load(fullfile(dataset_folder, sprintf('test_P%d.mat', P)));
    model_source = load(fullfile(model_folder, sprintf('model_P%d.mat', P)));
    model_parameters = double(extractdata(model_source.best_parameters));
    training_channel_scale = double(model_source.training_channel_scale);
    for SNR_index = 1:length(SNR_dB)
        selected = find(test_source.SNR_dB_data(:) == SNR_dB(SNR_index));
        GS_error = 0;
        GD_error = 0;
        HS_GS_error = 0;
        DL_error = 0;
        channel_energy = 0;
        rank_sum = 0;
        for sample_number = 1:length(selected)
            sample_index = selected(sample_number);
            Y = double(test_source.Y_data(:, :, sample_index));
            G = double(test_source.G_data(:, :, sample_index));
            S = double(test_source.S_data(:, :, sample_index));
            B = double(test_source.B_data(:, :, sample_index));
            sigma2 = double(test_source.sigma2_data(sample_index));
            G_hat_GS = run_GS(Y, S, B, GS_max_iterations, change_tolerance);
            G_hat_GD = run_GD(Y, S, B, GD_max_iterations, change_tolerance, G0_var, sample_index);
            rng(910000 + 1000 * P + sample_index, 'twister');
            pilot_order = randperm(P);
            training_count = round(training_fraction * P);
            T = pilot_order(1:training_count);
            V = pilot_order(training_count + 1:P);
            validation_error = zeros(length(rank_list), 1);
            for rank_index = 1:length(rank_list)
                candidate_rank = rank_list(rank_index);
                G_candidate = run_HS_GS(Y(:, T), S(:, T), B(:, T), sigma2, candidate_rank, rank_selection_updates, cadzow_sweeps, p_H);
                validation_error(rank_index) = norm(Y(:, V) - abs(G_candidate * S(:, V) + B(:, V)), 'fro')^2;
            end
            [~, best_rank_index] = min(validation_error);
            selected_rank = rank_list(best_rank_index);
            G_hat_HS_GS = run_HS_GS(Y, S, B, sigma2, selected_rank, HS_GS_updates, cadzow_sweeps, p_H);
            G_hat_DL = run_DL(Y, S, B, sigma2, model_parameters, training_channel_scale);
            GS_error = GS_error + norm(G_hat_GS - G, 'fro')^2;
            GD_error = GD_error + norm(G_hat_GD - G, 'fro')^2;
            HS_GS_error = HS_GS_error + norm(G_hat_HS_GS - G, 'fro')^2;
            DL_error = DL_error + norm(G_hat_DL - G, 'fro')^2;
            channel_energy = channel_energy + norm(G, 'fro')^2;
            rank_sum = rank_sum + selected_rank;
        end
        NMSE_GS(P_index, SNR_index) = GS_error / channel_energy;
        NMSE_GD(P_index, SNR_index) = GD_error / channel_energy;
        NMSE_HS_GS(P_index, SNR_index) = HS_GS_error / channel_energy;
        NMSE_DL(P_index, SNR_index) = DL_error / channel_energy;
        selected_rank_mean(P_index, SNR_index) = rank_sum / length(selected);
    end
end

NMSE_GS_dB = 10 * log10(NMSE_GS);
NMSE_GD_dB = 10 * log10(NMSE_GD);
NMSE_HS_GS_dB = 10 * log10(NMSE_HS_GS);
NMSE_DL_dB = 10 * log10(NMSE_DL);
save(fullfile(result_folder, 'final_four_method_results.mat'), 'SNR_dB', 'P_list', 'NMSE_GS', 'NMSE_GD', 'NMSE_HS_GS', 'NMSE_DL', 'NMSE_GS_dB', 'NMSE_GD_dB', 'NMSE_HS_GS_dB', 'NMSE_DL_dB', 'selected_rank_mean', 'RSR_dB');

figure('Color', 'w', 'Position', [100 100 1200 700]);
hold on;
grid on;
box on;
plot(SNR_dB, NMSE_GS_dB(1, :), '--o', 'DisplayName', 'GS, P = 10');
plot(SNR_dB, NMSE_GD_dB(1, :), '--s', 'DisplayName', 'GD, P = 10');
plot(SNR_dB, NMSE_HS_GS_dB(1, :), '--^', 'DisplayName', 'HS-GS, P = 10');
plot(SNR_dB, NMSE_DL_dB(1, :), '--d', 'DisplayName', 'DL, P = 10');
plot(SNR_dB, NMSE_GS_dB(2, :), '-o', 'DisplayName', 'GS, P = 30');
plot(SNR_dB, NMSE_GD_dB(2, :), '-s', 'DisplayName', 'GD, P = 30');
plot(SNR_dB, NMSE_HS_GS_dB(2, :), '-^', 'DisplayName', 'HS-GS, P = 30');
plot(SNR_dB, NMSE_DL_dB(2, :), '-d', 'DisplayName', 'DL, P = 30');
xlabel('SNR (dB)');
ylabel('NMSE (dB)');
title('Magnitude-only channel estimation, N = 8, K = 3, RSR = 25 dB');
legend('Location', 'eastoutside');
xticks(SNR_dB);
exportgraphics(gcf, fullfile(result_folder, 'final_four_method_nmse_vs_snr.png'), 'Resolution', 300);
savefig(gcf, fullfile(result_folder, 'final_four_method_nmse_vs_snr.fig'));

function G_hat = run_GS(Y, S, B, maximum_iterations, tolerance)
N = size(Y, 1);
K = size(S, 1);
G_hat = zeros(N, K);
for i = 1:N
    y_i = Y(i, :).';
    b_i = B(i, :).';
    A_bar_i = [S.', b_i];
    M_i = A_bar_i' * diag(y_i) * A_bar_i;
    [V_i, D_i] = eig(M_i);
    [~, principal_index] = max(real(diag(D_i)));
    v_i = V_i(:, principal_index) / norm(V_i(:, principal_index));
    r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2;
    G_hat(i, :) = (r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K)).';
end
for iteration = 1:maximum_iterations
    previous = G_hat;
    G_hat = (Y .* exp(1j * angle(G_hat * S + B)) - B) * S' / (S * S');
    if norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps) < tolerance
        break;
    end
end
end

function G_hat = run_GD(Y, S, B, maximum_iterations, tolerance, initial_variance, sample_index)
rng(720000 + sample_index, 'twister');
N = size(Y, 1);
K = size(S, 1);
G_hat = sqrt(initial_variance / 2) * (randn(N, K) + 1j * randn(N, K));
Z = exp(-1j * angle(B));
step_size = 1 / max(real(eig(S * S')));
for iteration = 1:maximum_iterations
    previous = G_hat;
    G_hat = G_hat + step_size * ((Y - abs(B) - real(Z .* (G_hat * S))) .* conj(Z)) * S';
    if norm(G_hat - previous, 'fro')^2 / max(norm(G_hat, 'fro')^2, eps) < tolerance
        break;
    end
end
end

function G_hat = run_DL(Y, S, B, sigma2, parameters, channel_scale)
N = size(Y, 1);
K = size(S, 1);
P = size(S, 2);
Y_normalized = single(Y / channel_scale);
S_real = single(real(S));
S_imag = single(imag(S));
B_real = single(real(B) / channel_scale);
B_imag = single(imag(B) / channel_scale);
noise_ratio = single(sigma2 / channel_scale^2);
pilot_bound = single(max(real(eig(S * S'))));
real_channel = zeros(N, K, 1, 'single');
imag_channel = zeros(N, K, 1, 'single');
previous_real_channel = real_channel;
previous_imag_channel = imag_channel;
noise_level = log(1 + noise_ratio / max(pilot_bound, single(1e-6)));
for layer_index = 1:size(parameters, 1)
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
G_hat = channel_scale * (double(real_channel(:, :, 1)) + 1j * double(imag_channel(:, :, 1)));
end

function G_hat = run_HS_GS(Y, S, B, sigma2, rank_value, EM_updates, cadzow_sweeps, p_H)
N = size(Y, 1);
K = size(S, 1);
G_hat = zeros(N, K);
for i = 1:N
    y_i = Y(i, :).';
    b_i = B(i, :).';
    A_bar_i = [S.', b_i];
    M_i = A_bar_i' * diag(y_i) * A_bar_i;
    [V_i, D_i] = eig(M_i);
    [~, principal_index] = max(real(diag(D_i)));
    v_i = V_i(:, principal_index) / norm(V_i(:, principal_index));
    r_i = (abs(A_bar_i * v_i).' * y_i) / norm(A_bar_i * v_i)^2;
    G_hat(i, :) = (r_i * exp(-1j * angle(v_i(K + 1))) * v_i(1:K)).';
end
for update = 1:EM_updates
    kappa = (2 / sigma2) * Y .* abs(G_hat * S + B);
    R_kappa = besseli(1, kappa, 1) ./ besseli(0, kappa, 1);
    U = Y .* exp(1j * angle(G_hat * S + B)) .* R_kappa;
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
            H_rank_r = U_H(:, 1:rank_value) * Sigma_H(1:rank_value, 1:rank_value) * V_H(:, 1:rank_value)';
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
    G_hat = G_processed;
end
end
