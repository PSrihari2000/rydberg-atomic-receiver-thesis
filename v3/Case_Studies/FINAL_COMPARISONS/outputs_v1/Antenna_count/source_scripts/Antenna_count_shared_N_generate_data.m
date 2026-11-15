%% Generate fresh validation and test splits for the shared-N DL study
% Existing antenna-sweep training data are reused. This script generates
% fresh validation and test data only, under a separate release folder.
% It does not load the existing or newly generated test split.

clear;
clc;

N_list = [8 16 24 32];
K = 3;
P_list = [10 30];
SNR_dB = [-10 0 10 20 30];
RSR_dB = 25;
AoA_deg_range = [-90 90];
d_over_lambda = 0.5;
split_names = {'validation','test'};
samples_per_SNR = 500;
split_seed_base = [2026112001 2026113001];
data_version = 'V3_ANTENNA_COUNT_SHARED_N_FRESH_V1';

script_folder = fileparts(mfilename('fullpath'));
fresh_release_folder = fullfile(script_folder, 'data', 'ANTENNA_COUNT_SHARED_N_FRESH');

for N = N_list
    dimension_folder = fullfile(fresh_release_folder, sprintf('N%d_K%d_RSR25dB', N, K));
    for P = P_list
        for split_index = 1:numel(split_names)
            output_file = fullfile(dimension_folder, sprintf('%s_P%d.mat', split_names{split_index}, P));
            if isfile(output_file)
                error('Refusing to overwrite fresh shared-N split: %s', output_file);
            end
        end
    end
end

if ~isfolder(fresh_release_folder)
    mkdir(fresh_release_folder);
end

hbar = 6.62607015e-34/(2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];

fprintf('Fresh shared-N validation/test data generation only.\n');
fprintf('N=%s, K=%d, P=%s, SNR=%s dB, RSR=%g dB.\n', mat2str(N_list), K, mat2str(P_list), mat2str(SNR_dB), RSR_dB);
fprintf('Samples per SNR per split=%d. Test outputs remain reserved.\n', samples_per_SNR);
fprintf('Fresh data folder: %s\n', fresh_release_folder);

for N_index = 1:numel(N_list)
    N = N_list(N_index);
    dimension_folder = fullfile(fresh_release_folder, sprintf('N%d_K%d_RSR25dB', N, K));
    if ~isfolder(dimension_folder)
        mkdir(dimension_folder);
    end

    for P_index = 1:numel(P_list)
        P = P_list(P_index);

        for split_index = 1:numel(split_names)
            split_name = split_names{split_index};
            random_seed = split_seed_base(split_index)+10000*N+100*K+P;
            rng(random_seed, 'twister');
            number_of_samples = samples_per_SNR*numel(SNR_dB);

            Y_data = zeros(N, P, number_of_samples, 'single');
            G_data = complex(zeros(N, K, number_of_samples, 'single'));
            S_data = complex(zeros(K, P, number_of_samples, 'single'));
            B_data = complex(zeros(N, P, number_of_samples, 'single'));
            sigma2_data = zeros(number_of_samples, 1, 'single');
            SNR_dB_data = zeros(number_of_samples, 1, 'single');

            for sample_index = 1:number_of_samples
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

                S = (randn(K, P)+1i*randn(K, P))/sqrt(2);
                alpha_reference = sqrt(10/2)*(randn+1i*randn);
                theta_reference_deg = AoA_deg_range(1)+diff(AoA_deg_range)*rand;
                theta_reference_rad = deg2rad(theta_reference_deg);
                reference_phase = 2*pi*d_over_lambda*cos(theta_reference_rad);
                polarization_reference = sqrt(1/3)*randn(3, 1);
                reference_vector = zeros(N, 1);
                for antenna_index = 1:N
                    reference_vector(antenna_index) = (mu_eg.'*polarization_reference/hbar)*alpha_reference*exp(-1i*(antenna_index-1)*reference_phase);
                end
                B = reference_vector*ones(1, P);

                average_single_user_power = 0;
                for user_index = 1:K
                    average_single_user_power = average_single_user_power+mean(abs(G(:, user_index)*S(user_index, :)).^2, 'all')/K;
                end
                B = B*sqrt(10^(RSR_dB/10)*average_single_user_power/mean(abs(B).^2, 'all'));

                snr_index = mod(sample_index-1, numel(SNR_dB))+1;
                current_SNR_dB = SNR_dB(snr_index);
                signal_power = mean(abs(G*S).^2, 'all');
                sigma2 = signal_power/10^(current_SNR_dB/10);
                W = sqrt(sigma2/2)*(randn(N, P)+1i*randn(N, P));
                Y = abs(G*S+B+W);

                Y_data(:, :, sample_index) = single(Y);
                G_data(:, :, sample_index) = single(G);
                S_data(:, :, sample_index) = single(S);
                B_data(:, :, sample_index) = single(B);
                sigma2_data(sample_index) = single(sigma2);
                SNR_dB_data(sample_index) = single(current_SNR_dB);

                if mod(sample_index, 1000) == 0 || sample_index == number_of_samples
                    fprintf('N=%d, P=%d, %s: %d/%d generated.\n', N, P, split_name, sample_index, number_of_samples);
                end
            end

            measurement_model = 'Y = abs(G*S + B + W)';
            RSR_definition = 'reference power divided by average single-user received signal power';
            AoA_distribution = 'Uniform incident angle';
            spatial_phase_convention = 'phi = 2*pi*(d/lambda)*cos(theta), theta in radians';
            output_file = fullfile(dimension_folder, sprintf('%s_P%d.mat', split_name, P));
            save(output_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'measurement_model', 'RSR_definition', 'AoA_distribution', 'AoA_deg_range', 'spatial_phase_convention', 'data_version', 'random_seed', 'samples_per_SNR', 'd_over_lambda', 'N', 'K', 'P', 'P_list', 'SNR_dB', 'RSR_dB', 'number_of_samples', 'split_name', '-v7.3');
            fprintf('Saved %s\n', output_file);
        end
    end
end

fprintf('Fresh validation/test generation complete. No test file was loaded or scored.\n');
