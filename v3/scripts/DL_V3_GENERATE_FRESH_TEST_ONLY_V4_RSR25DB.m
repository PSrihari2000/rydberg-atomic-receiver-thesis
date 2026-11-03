%% Generate a fourth independent V3 test-only dataset
% Model: Y = abs(G*S + B + W).
% The channel and measurement model are unchanged from the V3 release.
% This script writes only a new test split and refuses to overwrite files.
% It does not train or evaluate an estimator.

clear;
clc;

%% -------------------- FIXED SCENARIO --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_dB = -5:5:30;
RSR_dB = 25;
split_names = {'test'};
samples_per_SNR = 500;
split_seed_base = 2026110301;
d_over_lambda = 0.5;
AoA_deg_range = [-90 90];

script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_name = 'N8_K3_RSR25dB_AoA_Uminus90to90_FRESH_TEST_V4';
output_folder = fullfile(v3_folder, 'data', scenario_name);
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Check every destination before generating any data.
for P = P_list
    for split_index = 1:numel(split_names)
        output_file = fullfile(output_folder, sprintf('%s_P%d.mat', split_names{split_index}, P));
        if isfile(output_file)
            error('Refusing to overwrite existing V4 split: %s', output_file);
        end
    end
end

%% -------------------- PHYSICAL CONSTANTS --------------------
hbar = 6.62607015e-34 / (2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];

fprintf('V4 fresh test-only generation.\n');
fprintf('N=%d, K=%d, P=[%s], RSR=%g dB, AoA=U([%g,%g] deg).\n', N, K, num2str(P_list), RSR_dB, AoA_deg_range(1), AoA_deg_range(2));
fprintf('Fresh test samples per SNR = %d.\n', samples_per_SNR);
fprintf('Output: %s\n', output_folder);

%% -------------------- GENERATE NEW TEST SPLIT --------------------
for P_index = 1:numel(P_list)
    P = P_list(P_index);

    for split_index = 1:numel(split_names)
        split_name = split_names{split_index};
        random_seed = split_seed_base + P;
        rng(random_seed, 'twister');

        samples_this_split_per_snr = samples_per_SNR;
        number_of_samples = samples_this_split_per_snr*numel(SNR_dB);

        Y_data = zeros(N, P, number_of_samples, 'single');
        G_data = complex(zeros(N, K, number_of_samples, 'single'));
        S_data = complex(zeros(K, P, number_of_samples, 'single'));
        B_data = complex(zeros(N, P, number_of_samples, 'single'));
        sigma2_data = zeros(number_of_samples, 1, 'single');
        SNR_dB_data = zeros(number_of_samples, 1, 'single');

        for sample_index = 1:number_of_samples
            %% -------------------- TRUE CHANNEL --------------------
            G = zeros(N, K);
            for user_index = 1:K
                number_of_paths = randi([3 7]);
                for path_index = 1:number_of_paths
                    alpha = (randn + 1i*randn)/sqrt(2);
                    theta_deg = AoA_deg_range(1) + diff(AoA_deg_range)*rand;
                    theta_rad = deg2rad(theta_deg);
                    spatial_phase = 2*pi*d_over_lambda*cos(theta_rad);
                    polarization = sqrt(1/3)*randn(3, 1);

                    for antenna_index = 1:N
                        G(antenna_index, user_index) = G(antenna_index, user_index) + (mu_eg.'*polarization/hbar)*alpha*exp(-1i*(antenna_index - 1)*spatial_phase);
                    end
                end
            end

            %% -------------------- PILOTS AND REFERENCE --------------------
            S = (randn(K, P) + 1i*randn(K, P))/sqrt(2);

            alpha_reference = sqrt(10/2)*(randn + 1i*randn);
            theta_reference_deg = AoA_deg_range(1) + diff(AoA_deg_range)*rand;
            theta_reference_rad = deg2rad(theta_reference_deg);
            reference_phase = 2*pi*d_over_lambda*cos(theta_reference_rad);
            polarization_reference = sqrt(1/3)*randn(3, 1);
            b = zeros(N, 1);
            for antenna_index = 1:N
                b(antenna_index) = (mu_eg.'*polarization_reference/hbar)*alpha_reference*exp(-1i*(antenna_index - 1)*reference_phase);
            end
            B = b*ones(1, P);

            average_single_user_power = 0;
            for user_index = 1:K
                average_single_user_power = average_single_user_power + mean(abs(G(:, user_index)*S(user_index, :)).^2, 'all')/K;
            end
            B = B*sqrt(10^(RSR_dB/10)*average_single_user_power/mean(abs(B).^2, 'all'));

            %% -------------------- NOISE AND MAGNITUDE MEASUREMENT --------------------
            snr_index = mod(sample_index - 1, numel(SNR_dB)) + 1;
            current_SNR_dB = SNR_dB(snr_index);
            signal_power = mean(abs(G*S).^2, 'all');
            sigma2 = signal_power/10^(current_SNR_dB/10);
            W = sqrt(sigma2/2)*(randn(N, P) + 1i*randn(N, P));
            Y = abs(G*S + B + W);

            %% -------------------- STORE SAMPLE --------------------
            Y_data(:, :, sample_index) = single(Y);
            G_data(:, :, sample_index) = single(G);
            S_data(:, :, sample_index) = single(S);
            B_data(:, :, sample_index) = single(B);
            sigma2_data(sample_index) = single(sigma2);
            SNR_dB_data(sample_index) = single(current_SNR_dB);

            if mod(sample_index, 1000) == 0 || sample_index == number_of_samples
                fprintf('%s, P=%d: %d/%d samples created.\n', split_name, P, sample_index, number_of_samples);
            end
        end

        %% -------------------- SAVE REPRODUCTION METADATA --------------------
        data_version = 'V3_AoA_Uminus90to90';
        dataset_release = 'V3_FRESH_TEST_V4';
        measurement_model = 'Y = abs(G*S + B + W)';
        RSR_definition = 'reference power divided by average single-user received signal power';
        AoA_distribution = 'Uniform incident angle';
        spatial_phase_convention = 'phi = 2*pi*(d/lambda)*cos(theta), theta in radians';

        output_file = fullfile(output_folder, sprintf('%s_P%d.mat', split_name, P));
        save(output_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'measurement_model', 'RSR_definition', 'AoA_distribution', 'AoA_deg_range', 'spatial_phase_convention', 'data_version', 'dataset_release', 'random_seed', 'samples_this_split_per_snr', 'd_over_lambda', 'N', 'K', 'P', 'P_list', 'SNR_dB', 'RSR_dB', 'number_of_samples', 'split_name', '-v7.3');
        fprintf('Saved %s\n', output_file);
    end
end

fprintf('Fresh V4 test-only dataset generation complete.\n');
fprintf('No training or validation data were generated or changed.\n');
