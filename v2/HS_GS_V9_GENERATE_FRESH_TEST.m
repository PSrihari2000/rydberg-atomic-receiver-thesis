%% HS_GS_V9_GENERATE_FRESH_TEST.m
% Generate new channels and observations only after the V9 rule is frozen.
% The model and distributions match DL_dataset_RSR25dB.m; the seed is new.
clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
freeze_file = fullfile(script_folder, 'results', 'HS_GS_V9_frozen_rule.mat');
if ~exist(freeze_file, 'file')
    error('Run HS_GS_V9_VALIDATE_FREEZE before generating the fresh test set.');
end
frozen = load(freeze_file);
output_folder = fullfile(script_folder, 'results', 'HS_GS_V9_fresh_test_RSR25dB');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

N = frozen.N;
K = frozen.K;
P_list = frozen.P_list;
SNR_dB = frozen.SNR_dB;
RSR_dB = frozen.RSR_dB;
fresh_seed_base = frozen.fresh_seed_base;
samples_per_snr = 500;
d_over_lambda = frozen.d_over_lambda;
hbar = 6.62607015e-34 / (2 * pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9 * q_charge * a0_bohr; 0];

for P_index = 1:length(P_list)
    P = P_list(P_index);
    output_file = fullfile(output_folder, sprintf('fresh_test_P%d.mat', P));
    checkpoint_file = fullfile(output_folder, sprintf('generation_P%d_checkpoint.mat', P));
    if exist(output_file, 'file')
        fprintf('Keeping existing fresh test file: %s\n', output_file);
        continue;
    end
    generation_seed = fresh_seed_base + P;
    number_of_samples = samples_per_snr * length(SNR_dB);
    if exist(checkpoint_file, 'file')
        checkpoint = load(checkpoint_file);
        if checkpoint.P ~= P || checkpoint.N ~= N || checkpoint.K ~= K || ...
                checkpoint.RSR_dB ~= RSR_dB || ...
                checkpoint.d_over_lambda ~= d_over_lambda || ...
                checkpoint.generation_seed ~= generation_seed || ...
                checkpoint.number_of_samples ~= number_of_samples || ...
                ~isequal(checkpoint.SNR_dB, SNR_dB)
            error('Fresh generation checkpoint does not match the frozen settings for P=%d.', P);
        end
        Y_data = checkpoint.Y_data;
        G_data = checkpoint.G_data;
        S_data = checkpoint.S_data;
        B_data = checkpoint.B_data;
        sigma2_data = checkpoint.sigma2_data;
        SNR_dB_data = checkpoint.SNR_dB_data;
        next_sample_index = checkpoint.next_sample_index;
        rng(checkpoint.rng_state);
        fprintf('Resuming P=%d at sample %d/%d.\n', P, next_sample_index, number_of_samples);
    else
        rng(generation_seed, 'twister');
        Y_data = zeros(N, P, number_of_samples, 'single');
        G_data = complex(zeros(N, K, number_of_samples, 'single'));
        S_data = complex(zeros(K, P, number_of_samples, 'single'));
        B_data = complex(zeros(N, P, number_of_samples, 'single'));
        sigma2_data = zeros(number_of_samples, 1, 'single');
        SNR_dB_data = zeros(number_of_samples, 1, 'single');
        next_sample_index = 1;
    end
    for sample_index = next_sample_index:number_of_samples
        G = zeros(N, K);
        for k = 1:K
            L_k = randi([3 7]);
            for l = 1:L_k
                alpha_lk = (randn + 1j * randn) / sqrt(2);
                theta_lk = 2 * pi * rand;
                phi_lk = 2 * pi * d_over_lambda * cos(theta_lk);
                epsilon_kl = sqrt(1 / 3) * randn(3, 1);
                for i = 1:N
                    G(i, k) = G(i, k) + (mu_eg.' * epsilon_kl / hbar) * alpha_lk * exp(-1j * (i - 1) * phi_lk);
                end
            end
        end
        S = (randn(K, P) + 1j * randn(K, P)) / sqrt(2);
        alpha_b = sqrt(10 / 2) * (randn + 1j * randn);
        theta_b = 2 * pi * rand;
        phi_b = 2 * pi * d_over_lambda * cos(theta_b);
        epsilon_b = sqrt(1 / 3) * randn(3, 1);
        b = zeros(N, 1);
        for i = 1:N
            b(i) = (mu_eg.' * epsilon_b / hbar) * alpha_b * exp(-1j * (i - 1) * phi_b);
        end
        B = b * ones(1, P);
        average_single_user_power = 0;
        for k = 1:K
            average_single_user_power = average_single_user_power + mean(abs(G(:, k) * S(k, :)).^2, 'all') / K;
        end
        B = B * sqrt(10^(RSR_dB / 10) * average_single_user_power / mean(abs(B).^2, 'all'));
        current_SNR_index = mod(sample_index - 1, length(SNR_dB)) + 1;
        current_SNR_dB = SNR_dB(current_SNR_index);
        sigma2 = mean(abs(G * S).^2, 'all') / 10^(current_SNR_dB / 10);
        W = sqrt(sigma2 / 2) * (randn(N, P) + 1j * randn(N, P));
        Y = abs(G * S + B + W);
        Y_data(:, :, sample_index) = single(Y);
        G_data(:, :, sample_index) = single(G);
        S_data(:, :, sample_index) = single(S);
        B_data(:, :, sample_index) = single(B);
        sigma2_data(sample_index) = single(sigma2);
        SNR_dB_data(sample_index) = single(current_SNR_dB);
        if mod(sample_index, 500) == 0
            next_sample_index = sample_index + 1;
            rng_state = rng;
            save(checkpoint_file, 'Y_data', 'G_data', 'S_data', 'B_data', ...
                'sigma2_data', 'SNR_dB_data', 'P', 'SNR_dB', ...
                'generation_seed', 'number_of_samples', 'next_sample_index', ...
                'N', 'K', 'RSR_dB', 'd_over_lambda', ...
                'rng_state', '-v7.3');
        end
        if mod(sample_index, 1000) == 0 || sample_index == number_of_samples
            fprintf('Fresh P=%d: created %d/%d samples.\n', P, sample_index, number_of_samples);
        end
    end
    measurement_model = 'Y = abs(G*S + B + W)';
    RSR_definition = 'reference power divided by average single-user power';
    split_name = 'fresh_final_test';
    save(output_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', ...
        'SNR_dB_data', 'measurement_model', 'RSR_definition', 'N', 'K', 'P', ...
        'SNR_dB', 'RSR_dB', 'samples_per_snr', 'number_of_samples', ...
        'generation_seed', 'd_over_lambda', 'split_name', '-v7.3');
    fprintf('Saved %s\n', output_file);
end
