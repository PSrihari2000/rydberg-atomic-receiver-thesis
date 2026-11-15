%% Generate an independent confirmatory test split for the frozen six-method study
% This script creates test data only. It does not load validation or test data,
% train a model, or select a policy. It refuses to overwrite existing files.

clear;
clc;

N = 8;
K = 3;
P_list = [10 30];
SNR_dB = -5:5:30;
RSR_dB = 25;
samples_this_split_per_snr = 500;
d_over_lambda = 0.5;
AoA_deg_range = [-90 90];
split_name = 'test';
dataset_release = 'PI_DU_RACE_SIX_METHOD_CONFIRMATORY_TEST_V1';
data_version = dataset_release;
seed_base = 2026122500;

script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
output_folder = fullfile(case_studies_folder, 'data', dataset_release);

if isfolder(output_folder) || isfile(output_folder)
    error('Refusing to overwrite existing test-release folder: %s', output_folder);
end
mkdir(output_folder);

hbar = 6.62607015e-34/(2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];

fprintf('Generating independent confirmatory test data only.\n');
fprintf('Release=%s; N=%d; K=%d; P=[%s]; RSR=%g dB; SNR=[%s] dB.\n', ...
    dataset_release, N, K, num2str(P_list), RSR_dB, num2str(SNR_dB));
fprintf('%d samples per SNR and P; test seed=seed_base+P.\n', samples_this_split_per_snr);
fprintf('Output: %s\n', output_folder);

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    random_seed = seed_base + P;
    rng(random_seed, 'twister');
    number_of_samples = samples_this_split_per_snr*numel(SNR_dB);

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
                spatial_phase = 2*pi*d_over_lambda*cos(deg2rad(theta_deg));
                polarization = sqrt(1/3)*randn(3,1);
                for antenna_index = 1:N
                    G(antenna_index,user_index) = G(antenna_index,user_index) + ...
                        (mu_eg.'*polarization/hbar)*alpha*exp(-1i*(antenna_index-1)*spatial_phase);
                end
            end
        end

        S = (randn(K,P)+1i*randn(K,P))/sqrt(2);
        alpha_reference = sqrt(10/2)*(randn+1i*randn);
        theta_reference = AoA_deg_range(1)+diff(AoA_deg_range)*rand;
        reference_phase = 2*pi*d_over_lambda*cos(deg2rad(theta_reference));
        polarization_reference = sqrt(1/3)*randn(3,1);
        b = zeros(N,1);
        for antenna_index = 1:N
            b(antenna_index) = (mu_eg.'*polarization_reference/hbar)*alpha_reference* ...
                exp(-1i*(antenna_index-1)*reference_phase);
        end
        B = b*ones(1,P);

        average_single_user_power = 0;
        for user_index = 1:K
            average_single_user_power = average_single_user_power + ...
                mean(abs(G(:,user_index)*S(user_index,:)).^2,'all')/K;
        end
        B = B*sqrt(10^(RSR_dB/10)*average_single_user_power/mean(abs(B).^2,'all'));

        snr_index = mod(sample_index-1,numel(SNR_dB))+1;
        current_SNR_dB = SNR_dB(snr_index);
        signal_power = mean(abs(G*S).^2,'all');
        sigma2 = signal_power/10^(current_SNR_dB/10);
        W = sqrt(sigma2/2)*(randn(N,P)+1i*randn(N,P));
        Y = abs(G*S+B+W);

        Y_data(:,:,sample_index) = single(Y);
        G_data(:,:,sample_index) = single(G);
        S_data(:,:,sample_index) = single(S);
        B_data(:,:,sample_index) = single(B);
        sigma2_data(sample_index) = single(sigma2);
        SNR_dB_data(sample_index) = single(current_SNR_dB);

        if mod(sample_index,500) == 0 || sample_index == number_of_samples
            fprintf('P=%d test: %d/%d generated.\n',P,sample_index,number_of_samples);
        end
    end

    measurement_model = 'Y = abs(G*S + B + W)';
    RSR_definition = 'reference power divided by average single-user received signal power';
    AoA_distribution = 'Uniform incident angle';
    spatial_phase_convention = 'phi = 2*pi*(d/lambda)*cos(theta), theta in radians';
    output_file = fullfile(output_folder,sprintf('test_P%d.mat',P));
    save(output_file,'Y_data','G_data','S_data','B_data','sigma2_data','SNR_dB_data', ...
        'measurement_model','RSR_definition','AoA_distribution','AoA_deg_range', ...
        'spatial_phase_convention','data_version','dataset_release','random_seed', ...
        'samples_this_split_per_snr','d_over_lambda','N','K','P','P_list','SNR_dB', ...
        'RSR_dB','number_of_samples','split_name','-v7.3');
    fprintf('Saved %s\n',output_file);
end

fprintf('Independent confirmatory test generation complete. No estimator was run.\n');
