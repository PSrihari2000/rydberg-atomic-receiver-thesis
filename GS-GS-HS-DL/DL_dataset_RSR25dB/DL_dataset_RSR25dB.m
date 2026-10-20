%% DL_dataset.m
clear;
clc;
rng(1);
%% -------------------- DATASET SETTINGS --------------------
N = 8;
K = 3;
P_list = [10 30];                                      % 1 x 2: pilot counts
SNR_dB = -5:5:30;                                      % 1 x 8: SNR values
RSR_dB = 25;                                           % Reference power is 25 dB above average single-user power
split_names = {'train', 'validation', 'test'};
samples_per_SNR = [2500 500 500];                      % Gives 2500*8 = 20000 training, 4000 validation and 4000 test channels per P
split_seeds = [2026102001 2026102002 2026102003];
d_over_lambda = 0.5;                                   % Local half-wavelength array-spacing choice
output_folder = fullfile(fileparts(mfilename('fullpath')), 'DL_dataset_RSR25dB');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
%% -------------------- PHYSICAL CONSTANTS --------------------
hbar = 6.62607015e-34 / (2 * pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9 * q_charge * a0_bohr; 0];            % 3 x 1: transition dipole-moment vector
%% -------------------- CREATE P = 10 AND P = 30 DATASETS --------------------
for P_index = 1:length(P_list)
    P = P_list(P_index);
    for split_index = 1:length(split_names)
        split_name = split_names{split_index};
        rng(split_seeds(split_index) + P, 'twister');
        number_of_samples = samples_per_SNR(split_index) * length(SNR_dB);
        Y_data = zeros(N, P, number_of_samples, 'single');                    % N x P x number_of_samples
        G_data = complex(zeros(N, K, number_of_samples, 'single'));           % N x K x number_of_samples
        S_data = complex(zeros(K, P, number_of_samples, 'single'));           % K x P x number_of_samples
        B_data = complex(zeros(N, P, number_of_samples, 'single'));           % N x P x number_of_samples
        sigma2_data = zeros(number_of_samples, 1, 'single');                  % number_of_samples x 1
        SNR_dB_data = zeros(number_of_samples, 1, 'single');                  % number_of_samples x 1
        for sample_index = 1:number_of_samples
            %% -------------------- TRUE CHANNEL G --------------------
            G = zeros(N, K);                                                  % N x K: true channel matrix
            for k = 1:K
                L_k = randi([3 7]);                                           % Xu gives L_k ~ Uniform{3,4,5,6,7}
                for l = 1:L_k
                    alpha_lk = (randn + 1j * randn) / sqrt(2);                 % Xu gives alpha_lk ~ CN(0,1)
                    theta_lk = 2 * pi * rand;                                  % Local theta_lk ~ Uniform(0,2*pi)
                    phi_lk = 2 * pi * d_over_lambda * cos(theta_lk);
                    epsilon_kl = sqrt(1 / 3) * randn(3, 1);                    % 3 x 1: Xu gives polarization components ~ N(0,1/3)
                    for i = 1:N
                        G(i, k) = G(i, k) + (mu_eg.' * epsilon_kl / hbar) * alpha_lk * exp(-1j * (i - 1) * phi_lk);
                    end
                end
            end
            %% -------------------- PILOT MATRIX S --------------------
            S = (randn(K, P) + 1j * randn(K, P)) / sqrt(2);                   % K x P: Xu gives iid s_kp ~ CN(0,1)
            %% -------------------- KNOWN REFERENCE MATRIX B --------------------
            alpha_b = sqrt(10 / 2) * (randn + 1j * randn);                    % Xu gives alpha_b ~ CN(0,10)
            theta_b = 2 * pi * rand;                                          % Local theta_b ~ Uniform(0,2*pi)
            phi_b = 2 * pi * d_over_lambda * cos(theta_b);
            epsilon_b = sqrt(1 / 3) * randn(3, 1);                            % 3 x 1: Xu gives polarization components ~ N(0,1/3)
            b = zeros(N, 1);                                                  % N x 1: reference channel vector
            for i = 1:N
                b(i) = (mu_eg.' * epsilon_b / hbar) * alpha_b * exp(-1j * (i - 1) * phi_b);
            end
            B = b * ones(1, P);                                               % (N x 1)(1 x P) = N x P
            %% -------------------- SCALE B TO RSR = 25 dB --------------------
            average_single_user_power = 0;
            for k = 1:K
                average_single_user_power = average_single_user_power + mean(abs(G(:, k) * S(k, :)).^2, 'all') / K;
            end
            B = B * sqrt(10^(RSR_dB / 10) * average_single_user_power / mean(abs(B).^2, 'all'));
            %% -------------------- ADD NOISE AND FORM Y --------------------
            current_SNR_index = mod(sample_index - 1, length(SNR_dB)) + 1;
            current_SNR_dB = SNR_dB(current_SNR_index);
            sigma2 = mean(abs(G * S).^2, 'all') / 10^(current_SNR_dB / 10);
            W = sqrt(sigma2 / 2) * (randn(N, P) + 1j * randn(N, P));           % N x P: w_ip ~ CN(0,sigma2)
            Y = abs(G * S + B + W);                                           % N x P: magnitude-only observations
            %% -------------------- STORE ONE INDEPENDENT SAMPLE --------------------
            Y_data(:, :, sample_index) = single(Y);
            G_data(:, :, sample_index) = single(G);
            S_data(:, :, sample_index) = single(S);
            B_data(:, :, sample_index) = single(B);
            sigma2_data(sample_index) = single(sigma2);
            SNR_dB_data(sample_index) = single(current_SNR_dB);
            if mod(sample_index, 1000) == 0 || sample_index == number_of_samples
                fprintf('%s, P = %d: created %d of %d channel samples\n', split_name, P, sample_index, number_of_samples);
            end
        end
        measurement_model = 'Y = abs(G*S + B + W)';
        RSR_definition = 'reference power divided by average single-user power';
        output_file = fullfile(output_folder, sprintf('%s_P%d.mat', split_name, P));
        save(output_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'measurement_model', 'RSR_definition', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'number_of_samples', 'split_name', '-v7.3');
        fprintf('Saved %s\n', output_file);
    end
end
fprintf('All DL training, validation and test datasets were created.\n');
