%% Standalone N=8 GS, GD, EM-GS, and HS-GS comparison
% This script generates a new reproducible V3 test set in memory.
% No training, validation, test, checkpoint, or multiplier MAT files are loaded.
% HS-GS multipliers were frozen using the earlier V3 validation experiment.
% The generated observations are shared by all four estimators.
% All algorithm steps are written directly in this script without local functions.
% This script has no MATLAB continuation marks.

clear;
clc;
close all;

%% -------------------- SIMULATION SETTINGS --------------------
N = 8;
K = 3;
P_list = [10 30];
number_of_pilot_counts = numel(P_list);
SNR_grid = -5:5:30;
number_of_SNR_points = numel(SNR_grid);
RSR_dB = 25;
AoA_deg_range = [-90 90];
d_over_lambda = 0.5;
samples_per_SNR = 500;
data_seed_base = 2026110400;
maximum_updates = 500;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
p_H = 3;
bootstrap_replicates = 2000;
method_names = {'GS','GD','EM-GS','HS-GS'};
number_of_methods = numel(method_names);
frozen_HS_multiplier = [1 0.5 0.5 0.25 0.25 0.25 0.25 0; 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25];

%% -------------------- OUTPUT SETTINGS --------------------
script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
output_folder = fullfile(v3_folder, 'results', 'N8_STANDALONE_GS_GD_EMGS_HSGS_RSR25DB');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
result_csv = fullfile(output_folder, 'N8_standalone_classical_test_results.csv');
result_mat = fullfile(output_folder, 'N8_standalone_classical_test_results.mat');
result_figure = fullfile(output_folder, 'N8_standalone_classical_test_nmse.png');

%% -------------------- PHYSICAL CONSTANTS --------------------
hbar = 6.62607015e-34/(2*pi);
q_charge = 1.602e-19;
a0_bohr = 5.292e-11;
mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];

%% -------------------- ARRAYS FOR RESULT STORAGE --------------------
number_of_rows = number_of_pilot_counts*number_of_SNR_points;
result_rows = zeros(number_of_rows, 32);
all_sample_errors = cell(number_of_pilot_counts, 1);
all_true_energy = cell(number_of_pilot_counts, 1);
all_iteration_counts = cell(number_of_pilot_counts, 1);
all_convergence_status = cell(number_of_pilot_counts, 1);
all_runtime_seconds = cell(number_of_pilot_counts, 1);
row_index = 0;

fprintf('Standalone V3 classical comparison: N=%d, K=%d, RSR=%g dB.\n', N, K, RSR_dB);
fprintf('Methods: GS, GD, EM-GS, and validation-frozen soft HS-GS.\n');
fprintf('AoA distribution: Uniform(-90,90 deg).\n');
fprintf('Samples per SNR and pilot count: %d.\n', samples_per_SNR);
fprintf('No dataset or multiplier MAT files are loaded.\n');
fprintf('Maximum updates=%d; relative-change tolerance=%.1e.\n', maximum_updates, convergence_tolerance);
fprintf('HS-GS Hankel split p_H=%d.\n', p_H);
fprintf('Frozen HS-GS multipliers by pilot count and SNR order:\n');
disp(array2table(frozen_HS_multiplier, 'VariableNames', {'SNRm5','SNR0','SNR5','SNR10','SNR15','SNR20','SNR25','SNR30'}, 'RowNames', {'P10','P30'}));

%% -------------------- LOOP OVER PILOT COUNTS --------------------
for p_index = 1:number_of_pilot_counts
    P = P_list(p_index);
    number_of_samples = samples_per_SNR*number_of_SNR_points;
    dataset_seed = data_seed_base+P;
    rng(dataset_seed, 'twister');
    Y_data = zeros(N, P, number_of_samples);
    G_data = complex(zeros(N, K, number_of_samples));
    S_data = complex(zeros(K, P, number_of_samples));
    B_data = complex(zeros(N, P, number_of_samples));
    sigma2_data = zeros(number_of_samples, 1);
    SNR_dB_data = zeros(number_of_samples, 1);

    fprintf('\nGenerating fresh P=%d test observations. Seed=%d.\n', P, dataset_seed);

    %% -------------------- GENERATE CHANNELS AND OBSERVATIONS --------------------
    for snr_index = 1:number_of_SNR_points
        current_SNR_dB = SNR_grid(snr_index);
        for sample_within_snr = 1:samples_per_SNR
            sample_index = (snr_index-1)*samples_per_SNR+sample_within_snr;

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
            signal_power = mean(abs(G*S).^2, 'all');
            sigma2 = signal_power/10^(current_SNR_dB/10);
            W = sqrt(sigma2/2)*(randn(N, P)+1i*randn(N, P));
            Y = abs(G*S+B+W);

            Y_data(:, :, sample_index) = Y;
            G_data(:, :, sample_index) = G;
            S_data(:, :, sample_index) = S;
            B_data(:, :, sample_index) = B;
            sigma2_data(sample_index) = sigma2;
            SNR_dB_data(sample_index) = current_SNR_dB;
        end
        fprintf('  Generated P=%d, SNR=%g dB: %d/%d samples.\n', P, current_SNR_dB, sample_within_snr, samples_per_SNR);
    end

    %% -------------------- ARRAYS FOR CURRENT PILOT COUNT --------------------
    sample_error = zeros(number_of_samples, number_of_methods);
    true_energy = zeros(number_of_samples, 1);
    iteration_count = zeros(number_of_samples, number_of_methods);
    convergence_status = false(number_of_samples, number_of_methods);
    runtime_seconds = zeros(number_of_samples, number_of_methods);

    %% -------------------- ESTIMATE EACH GENERATED CHANNEL --------------------
    fprintf('Estimating channels for P=%d using the shared observations.\n', P);
    for sample_index = 1:number_of_samples
        Y = Y_data(:, :, sample_index);             % N x P = 8 x P: magnitude-only observation
        G_true = G_data(:, :, sample_index);        % N x K = 8 x 3: channel used only for final NMSE
        S = S_data(:, :, sample_index);             % K x P = 3 x P: known pilot matrix
        B = B_data(:, :, sample_index);             % N x P = 8 x P: known reference matrix
        sigma2 = sigma2_data(sample_index);          % scalar: complex noise variance
        snr_db = SNR_dB_data(sample_index);          % scalar: SNR for this observation
        snr_index = find(abs(SNR_grid-snr_db) < 1e-6, 1);
        if isempty(snr_index)
            error('Unexpected SNR value in generated P=%d data: %g.', P, snr_db);
        end
        selected_multiplier = frozen_HS_multiplier(p_index, snr_index); % scalar: frozen validation multiplier
        random_seed = 821000000+1000000*P+10000*snr_index+sample_index;

        %% -------------------- ALGORITHM 1: GS --------------------
        timer = tic;
        G_GS = zeros(N, K);                          % N x K = 8 x 3: spectral initialization
        for antenna_index = 1:N
            y = Y(antenna_index, :).';               % P x 1: magnitudes for one antenna
            b = B(antenna_index, :).';               % P x 1: reference for one antenna
            A = [S.', b];                            % P x (K+1): augmented pilot-reference matrix
            M = A'*diag(y)*A;                        % (K+1) x (K+1): spectral initialization matrix
            [eigenvectors, eigenvalues] = eig(M);
            [~, principal_index] = max(real(diag(eigenvalues)));
            v = eigenvectors(:, principal_index);    % (K+1) x 1: principal eigenvector
            v = v/max(norm(v), eps);
            scale = (abs(A*v).'*y)/max(norm(A*v)^2, eps);
            G_GS(antenna_index, :) = (scale*exp(-1i*angle(v(K+1)))*v(1:K)).';
        end
        right_inverse = S'/(S*S');                  % P x K: pilot right inverse
        GS_converged = false;
        GS_updates = maximum_updates;
        for update_index = 1:maximum_updates
            G_GS_previous = G_GS;                   % N x K: previous GS estimate
            G_GS = (Y.*exp(1i*angle(G_GS*S+B))-B)*right_inverse; % N x K: GS phase replacement update
            if norm(G_GS-G_GS_previous, 'fro')^2/max(norm(G_GS, 'fro')^2, eps) < convergence_tolerance
                GS_updates = update_index;
                GS_converged = true;
                break;
            end
        end
        runtime_seconds(sample_index, 1) = toc(timer);
        iteration_count(sample_index, 1) = GS_updates;
        convergence_status(sample_index, 1) = GS_converged;
        sample_error(sample_index, 1) = norm(G_GS-G_true, 'fro')^2;

        %% -------------------- ALGORITHM 2: GD --------------------
        timer = tic;
        rng(mod(random_seed+720000, 2^32-1), 'twister');
        G_GD = sqrt(gd_initial_variance/2)*(randn(N, K)+1i*randn(N, K)); % N x K: complex-Gaussian initialization
        phase_reference = exp(-1i*angle(B));        % N x P: reference phase rotation
        step_size = 1/max(real(eig(S*S')));         % scalar: GD step size
        GD_converged = false;
        GD_updates = maximum_updates;
        for update_index = 1:maximum_updates
            G_GD_previous = G_GD;                   % N x K: previous GD estimate
            G_GD = G_GD+step_size*((Y-abs(B)-real(phase_reference.*(G_GD*S))).*conj(phase_reference))*S'; % N x K: GD update
            if norm(G_GD-G_GD_previous, 'fro')^2/max(norm(G_GD, 'fro')^2, eps) < convergence_tolerance
                GD_updates = update_index;
                GD_converged = true;
                break;
            end
        end
        runtime_seconds(sample_index, 2) = toc(timer);
        iteration_count(sample_index, 2) = GD_updates;
        convergence_status(sample_index, 2) = GD_converged;
        sample_error(sample_index, 2) = norm(G_GD-G_true, 'fro')^2;

        %% -------------------- ALGORITHM 3: EM-GS --------------------
        timer = tic;
        G_EMGS = zeros(N, K);                        % N x K = 8 x 3: spectral initialization
        for antenna_index = 1:N
            y = Y(antenna_index, :).';               % P x 1: magnitudes for one antenna
            b = B(antenna_index, :).';               % P x 1: reference for one antenna
            A = [S.', b];                            % P x (K+1): augmented pilot-reference matrix
            M = A'*diag(y)*A;                        % (K+1) x (K+1): spectral initialization matrix
            [eigenvectors, eigenvalues] = eig(M);
            [~, principal_index] = max(real(diag(eigenvalues)));
            v = eigenvectors(:, principal_index);    % (K+1) x 1: principal eigenvector
            v = v/max(norm(v), eps);
            scale = (abs(A*v).'*y)/max(norm(A*v)^2, eps);
            G_EMGS(antenna_index, :) = (scale*exp(-1i*angle(v(K+1)))*v(1:K)).';
        end
        right_inverse = S'/(S*S');                  % P x K: pilot right inverse
        EMGS_converged = false;
        EMGS_updates = maximum_updates;
        for update_index = 1:maximum_updates
            G_EMGS_previous = G_EMGS;               % N x K: previous EM-GS estimate
            field = G_EMGS*S+B;                     % N x P: predicted complex field
            kappa = 2/max(sigma2, eps)*Y.*abs(field); % N x P: magnitude-noise concentration
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin); % N x P: Bessel ratio
            G_EMGS = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse; % N x K: EM-GS update
            if norm(G_EMGS-G_EMGS_previous, 'fro')^2/max(norm(G_EMGS, 'fro')^2, eps) < convergence_tolerance
                EMGS_updates = update_index;
                EMGS_converged = true;
                break;
            end
        end
        runtime_seconds(sample_index, 3) = toc(timer);
        iteration_count(sample_index, 3) = EMGS_updates;
        convergence_status(sample_index, 3) = EMGS_converged;
        sample_error(sample_index, 3) = norm(G_EMGS-G_true, 'fro')^2;

        %% -------------------- ALGORITHM 4: SOFT HS-GS --------------------
        timer = tic;
        G_HSGS = zeros(N, K);                        % N x K = 8 x 3: spectral initialization
        for antenna_index = 1:N
            y = Y(antenna_index, :).';               % P x 1: magnitudes for one antenna
            b = B(antenna_index, :).';               % P x 1: reference for one antenna
            A = [S.', b];                            % P x (K+1): augmented pilot-reference matrix
            M = A'*diag(y)*A;                        % (K+1) x (K+1): spectral initialization matrix
            [eigenvectors, eigenvalues] = eig(M);
            [~, principal_index] = max(real(diag(eigenvalues)));
            v = eigenvectors(:, principal_index);    % (K+1) x 1: principal eigenvector
            v = v/max(norm(v), eps);
            scale = (abs(A*v).'*y)/max(norm(A*v)^2, eps);
            G_HSGS(antenna_index, :) = (scale*exp(-1i*angle(v(K+1)))*v(1:K)).';
        end
        right_inverse = S'/(S*S');                  % P x K: pilot right inverse
        threshold_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K); % scalar: noise/pilot threshold scale
        threshold = selected_multiplier*threshold_scale; % scalar: frozen singular-value threshold
        HSGS_converged = false;
        HSGS_updates = maximum_updates;
        for update_index = 1:maximum_updates
            G_HSGS_previous = G_HSGS;               % N x K: previous HS-GS estimate
            field = G_HSGS*S+B;                     % N x P: predicted complex field
            kappa = 2/max(sigma2, eps)*Y.*abs(field); % N x P: magnitude-noise concentration
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin); % N x P: Bessel ratio
            G_HSGS = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse; % N x K: EM-GS update
            if threshold > 0
                number_of_hankel_rows = N-p_H;       % scalar: 5 rows for N=8 and p_H=3
                number_of_hankel_columns = p_H+1;    % scalar: 4 columns for N=8 and p_H=3
                for user_index = 1:K
                    channel_vector = G_HSGS(:, user_index); % N x 1: one user's channel
                    hankel_matrix = zeros(number_of_hankel_rows, number_of_hankel_columns); % 5 x 4 complex Hankel embedding
                    for hankel_row = 1:number_of_hankel_rows
                        hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+number_of_hankel_columns-1).';
                    end
                    [U, Sigma, V] = svd(hankel_matrix, 'econ'); % 5 x 4 Hankel SVD
                    shrunk_singular_values = max(diag(Sigma)-threshold, 0); % at most 4 singular values
                    hankel_matrix = U*diag(shrunk_singular_values)*V'; % 5 x 4 soft-thresholded Hankel matrix
                    reconstructed_channel = zeros(N, 1); % N x 1: channel reconstructed by anti-diagonal averaging
                    diagonal_counts = zeros(N, 1);      % N x 1: number of entries per anti-diagonal
                    for hankel_row = 1:number_of_hankel_rows
                        for hankel_column = 1:number_of_hankel_columns
                            antenna_index = hankel_row+hankel_column-1;
                            reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(hankel_row, hankel_column);
                            diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
                        end
                    end
                    G_HSGS(:, user_index) = reconstructed_channel./diagonal_counts;
                end
            end
            if norm(G_HSGS-G_HSGS_previous, 'fro')^2/max(norm(G_HSGS, 'fro')^2, eps) < convergence_tolerance
                HSGS_updates = update_index;
                HSGS_converged = true;
                break;
            end
        end
        runtime_seconds(sample_index, 4) = toc(timer);
        iteration_count(sample_index, 4) = HSGS_updates;
        convergence_status(sample_index, 4) = HSGS_converged;
        sample_error(sample_index, 4) = norm(G_HSGS-G_true, 'fro')^2;

        true_energy(sample_index) = norm(G_true, 'fro')^2;
        if mod(sample_index, 100) == 0 || sample_index == number_of_samples
            fprintf('  P=%d estimates completed: %d/%d.\n', P, sample_index, number_of_samples);
        end
    end
    %% -------------------- SUMMARIZE RESULTS BY SNR --------------------
    for snr_index = 1:number_of_SNR_points
        snr_db = SNR_grid(snr_index);
        sample_indices = find(abs(SNR_dB_data-snr_db) < 1e-6);
        if numel(sample_indices) ~= samples_per_SNR
            error('P=%d has %d samples at SNR=%g dB; expected %d.', P, numel(sample_indices), snr_db, samples_per_SNR);
        end
        errors = sample_error(sample_indices, :);
        energy = true_energy(sample_indices);
        nmse_db = 10*log10(sum(errors, 1)/sum(energy));
        rng(931000+1000*P+snr_index, 'twister');
        bootstrap_gain = zeros(bootstrap_replicates, number_of_methods-1);
        bootstrap_hs_vs_em = zeros(bootstrap_replicates, 1);
        for draw_index = 1:bootstrap_replicates
            resampled_indices = randi(numel(sample_indices), numel(sample_indices), 1);
            resampled_errors = errors(resampled_indices, :);
            bootstrap_gain(draw_index, 1) = 10*log10(sum(resampled_errors(:, 1))/max(sum(resampled_errors(:, 2)), realmin));
            bootstrap_gain(draw_index, 2) = 10*log10(sum(resampled_errors(:, 1))/max(sum(resampled_errors(:, 3)), realmin));
            bootstrap_gain(draw_index, 3) = 10*log10(sum(resampled_errors(:, 1))/max(sum(resampled_errors(:, 4)), realmin));
            bootstrap_hs_vs_em(draw_index) = 10*log10(sum(resampled_errors(:, 3))/max(sum(resampled_errors(:, 4)), realmin));
        end
        bootstrap_gain = sort(bootstrap_gain, 1);
        ci_low = bootstrap_gain(max(1, floor(0.025*bootstrap_replicates)), :);
        ci_high = bootstrap_gain(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)), :);
        bootstrap_hs_vs_em = sort(bootstrap_hs_vs_em);
        hs_em_ci_low = bootstrap_hs_vs_em(max(1, floor(0.025*bootstrap_replicates)));
        hs_em_ci_high = bootstrap_hs_vs_em(min(bootstrap_replicates, ceil(0.975*bootstrap_replicates)));
        mean_iterations = mean(iteration_count(sample_indices, :), 1);
        convergence_fraction = mean(convergence_status(sample_indices, :), 1);
        mean_runtime = mean(runtime_seconds(sample_indices, :), 1);
        row_index = row_index+1;
        result_rows(row_index, :) = [P, snr_db, numel(sample_indices), frozen_HS_multiplier(p_index, snr_index), nmse_db, nmse_db(1)-nmse_db(2), nmse_db(1)-nmse_db(3), nmse_db(1)-nmse_db(4), ci_low, ci_high, nmse_db(3)-nmse_db(4), hs_em_ci_low, hs_em_ci_high, mean_iterations, convergence_fraction, mean_runtime];
    end
end

%% -------------------- SAVE TABLE AND FIGURE --------------------
result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','Samples','Frozen_HS_multiplier','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','HSGS_NMSE_dB','GD_gain_vs_GS_dB','EMGS_gain_vs_GS_dB','HSGS_gain_vs_GS_dB','GD_CI_low_vs_GS_dB','EMGS_CI_low_vs_GS_dB','HSGS_CI_low_vs_GS_dB','GD_CI_high_vs_GS_dB','EMGS_CI_high_vs_GS_dB','HSGS_CI_high_vs_GS_dB','HSGS_gain_vs_EMGS_dB','HSGS_CI_low_vs_EMGS_dB','HSGS_CI_high_vs_EMGS_dB','GS_mean_updates','GD_mean_updates','EMGS_mean_updates','HSGS_mean_updates','GS_convergence_fraction','GD_convergence_fraction','EMGS_convergence_fraction','HSGS_convergence_fraction','GS_mean_runtime_s','GD_mean_runtime_s','EMGS_mean_runtime_s','HSGS_mean_runtime_s'});
configuration = struct('N', N, 'K', K, 'P_list', P_list, 'SNR_dB', SNR_grid, 'RSR_dB', RSR_dB, 'AoA_deg_range', AoA_deg_range, 'd_over_lambda', d_over_lambda, 'samples_per_SNR', samples_per_SNR, 'data_seed_base', data_seed_base, 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'gd_initial_variance', gd_initial_variance, 'p_H', p_H, 'frozen_HS_multiplier', frozen_HS_multiplier, 'bootstrap_replicates', bootstrap_replicates, 'data_version', 'V3_AoA_Uminus90to90_fresh_seed');
writetable(result_table, result_csv);
save(result_mat, 'result_table', 'configuration', 'all_sample_errors', 'all_true_energy', 'all_iteration_counts', 'all_convergence_status', 'all_runtime_seconds', 'method_names', '-v7.3');

figure_handle = figure('Color', 'w', 'Name', 'N8 standalone classical comparison');
for p_index = 1:number_of_pilot_counts
    subplot(1, number_of_pilot_counts, p_index);
    selected_rows = result_table.P == P_list(p_index);
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows), '-o', 'LineWidth', 1.4, 'DisplayName', 'GS');
    hold on;
    plot(SNR_grid, result_table.GD_NMSE_dB(selected_rows), '-s', 'LineWidth', 1.4, 'DisplayName', 'GD');
    plot(SNR_grid, result_table.EMGS_NMSE_dB(selected_rows), '-d', 'LineWidth', 1.4, 'DisplayName', 'EM-GS');
    plot(SNR_grid, result_table.HSGS_NMSE_dB(selected_rows), '-^', 'LineWidth', 1.4, 'DisplayName', 'Soft HS-GS');
    grid on;
    xticks(SNR_grid);
    xlabel('SNR (dB)');
    ylabel('NMSE (dB)');
    title(sprintf('N=8, K=3, P=%d, RSR=25 dB', P_list(p_index)));
    legend('Location', 'best');
end
saveas(figure_handle, result_figure);
disp(result_table);
fprintf('\nStandalone comparison complete.\n');
fprintf('CSV results: %s\n', result_csv);
fprintf('MAT results: %s\n', result_mat);
fprintf('Figure: %s\n', result_figure);
fprintf('No external data or multiplier files were loaded.\n');
fprintf('The generated observations use a fixed seed and are shared by all four methods.\n');



