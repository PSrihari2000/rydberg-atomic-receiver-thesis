%% Antenna-count sweep: data, validation policy, and classical test workflow
% This is the main file for the antenna-count study.
% Select one workflow phase below and run this same file.
% Classical comparison methods are GS, GD, EM-GS, and hard-rank HS-GS.
% Soft Hankel singular-value shrinkage is reserved for the learned DL method.
% Existing files are never overwritten. MATLAB is run by the user.

clear;
clc;

%% -------------------- SELECT ONE WORKFLOW PHASE --------------------
workflow_phase = 'PLOT_SUMMARY';
% GENERATE creates fresh train/validation/test files.
% DIAGNOSTIC checks hard-HS-GS behavior on a small validation-only subset.
% VALIDATE selects and confirms hard-HS-GS ranks using validation files only.
% TEST scores the frozen classical methods on reserved test files only.
% PLOT_SUMMARY makes one test-NMSE curve per classical method, averaged equally over P and SNR.

%% -------------------- FIXED ANTENNA SWEEP --------------------
study_name = 'ANTENNA_COUNT_SWEEP';
N_list = [8 16 24 32];
K_list = 3;
data_version = 'V3_ANTENNA_COUNT_SWEEP';
scenario_name = 'ANTENNA_COUNT_N8_16_24_32_K3_RSR25DB_SNR_m10_0_10_20_30';

P_list = [10 30];
SNR_dB = [-10 0 10 20 30];
RSR_dB = 25;
split_names = {'train', 'validation', 'test'};
samples_per_SNR = [2000 500 500];
split_seed_base = [2026111101 2026111201 2026111301];
d_over_lambda = 0.5;
AoA_deg_range = [-90 90];
maximum_parallel_workers = 6;

script_folder = fileparts(mfilename('fullpath'));
output_root = fullfile(script_folder, 'data', scenario_name);
results_root = fullfile(script_folder, 'results', 'ANTENNA_COUNT_CLASSICAL');

switch upper(workflow_phase)
    case 'GENERATE'
        %% Check all output destinations before creating any data
        for N_index = 1:numel(N_list)
            N = N_list(N_index);
            for K_index = 1:numel(K_list)
                K = K_list(K_index);
                dimension_folder = fullfile(output_root, sprintf('N%d_K%d_RSR25dB', N, K));
                for P = P_list
                    for split_index = 1:numel(split_names)
                        output_file = fullfile(dimension_folder, sprintf('%s_P%d.mat', split_names{split_index}, P));
                        if isfile(output_file)
                            error('Refusing to overwrite existing sweep dataset: %s', output_file);
                        end
                    end
                end
            end
        end

        if ~isfolder(output_root)
            mkdir(output_root);
        end

        %% -------------------- PHYSICAL CONSTANTS --------------------
        hbar = 6.62607015e-34/(2*pi);
        q_charge = 1.602e-19;
        a0_bohr = 5.292e-11;
        mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];

        fprintf('Antenna sweep dataset generation only.\n');
        fprintf('Study=%s; N=[%s]; K=[%s]; P=[%s]; SNR=[%s] dB; RSR=%g dB.\n', study_name, num2str(N_list), num2str(K_list), num2str(P_list), num2str(SNR_dB), RSR_dB);
        fprintf('Samples per SNR: train=%d, validation=%d, test=%d.\n', samples_per_SNR(1), samples_per_SNR(2), samples_per_SNR(3));
        fprintf('Output root: %s\n', output_root);

        %% -------------------- GENERATE EACH DIMENSION AND PILOT COUNT --------------------
        for N_index = 1:numel(N_list)
            N = N_list(N_index);

            for K_index = 1:numel(K_list)
                K = K_list(K_index);
                dimension_folder = fullfile(output_root, sprintf('N%d_K%d_RSR25dB', N, K));
                if ~isfolder(dimension_folder)
                    mkdir(dimension_folder);
                end

                for P_index = 1:numel(P_list)
                    P = P_list(P_index);

                    for split_index = 1:numel(split_names)
                        split_name = split_names{split_index};
                        random_seed = split_seed_base(split_index) + 10000*N + 100*K + P;
                        rng(random_seed, 'twister');
                        samples_this_split_per_snr = samples_per_SNR(split_index);
                        number_of_samples = samples_this_split_per_snr*numel(SNR_dB);

                        Y_data = zeros(N, P, number_of_samples, 'single');
                        G_data = complex(zeros(N, K, number_of_samples, 'single'));
                        S_data = complex(zeros(K, P, number_of_samples, 'single'));
                        B_data = complex(zeros(N, P, number_of_samples, 'single'));
                        sigma2_data = zeros(number_of_samples, 1, 'single');
                        SNR_dB_data = zeros(number_of_samples, 1, 'single');

                        for sample_index = 1:number_of_samples
                            %% Generate channel G (N x K), with 3 to 7 paths per user
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
                                        G(antenna_index, user_index) = G(antenna_index, user_index) + (mu_eg.'*polarization/hbar)*alpha*exp(-1i*(antenna_index-1)*spatial_phase);
                                    end
                                end
                            end

                            %% Generate pilots S and rank-one known reference B
                            S = (randn(K, P) + 1i*randn(K, P))/sqrt(2);
                            alpha_reference = sqrt(10/2)*(randn + 1i*randn);
                            theta_reference_deg = AoA_deg_range(1) + diff(AoA_deg_range)*rand;
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
                                average_single_user_power = average_single_user_power + mean(abs(G(:, user_index)*S(user_index, :)).^2, 'all')/K;
                            end
                            B = B*sqrt(10^(RSR_dB/10)*average_single_user_power/mean(abs(B).^2, 'all'));

                            %% Generate Y = |G*S + B + W| at the assigned SNR
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
                                fprintf('N=%d, K=%d, P=%d, %s: %d/%d samples.\n', N, K, P, split_name, sample_index, number_of_samples);
                            end
                        end

                        %% Save dimensions and generation provenance with each split
                        measurement_model = 'Y = abs(G*S + B + W)';
                        RSR_definition = 'reference power divided by average single-user received signal power';
                        AoA_distribution = 'Uniform incident angle';
                        spatial_phase_convention = 'phi = 2*pi*(d/lambda)*cos(theta), theta in radians';
                        output_file = fullfile(dimension_folder, sprintf('%s_P%d.mat', split_name, P));
                        save(output_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'measurement_model', 'RSR_definition', 'AoA_distribution', 'AoA_deg_range', 'spatial_phase_convention', 'data_version', 'study_name', 'random_seed', 'samples_this_split_per_snr', 'd_over_lambda', 'N', 'K', 'P', 'P_list', 'SNR_dB', 'RSR_dB', 'number_of_samples', 'split_name', '-v7.3');
                        fprintf('Saved %s\n', output_file);
                    end
                end
            end
        end

        fprintf('Dataset generation complete. No model or estimator was trained or evaluated.\n');
        fprintf('Keep each dimension-specific test split reserved until all settings are frozen.\n');

    case 'DIAGNOSTIC'
        % Small validation-only convergence and iteration-cap diagnostic
        diagnostic_folder = fullfile(results_root, 'convergence_diagnostic_all_ranks');
        diagnostic_file = fullfile(diagnostic_folder, 'hard_rank_hsgs_convergence_diagnostic.mat');
        diagnostic_csv = fullfile(diagnostic_folder, 'hard_rank_hsgs_convergence_diagnostic.csv');
        if isfile(diagnostic_file) || isfile(diagnostic_csv)
            error('Diagnostic output already exists. Inspect it before starting another diagnostic.');
        end
        if ~isfolder(diagnostic_folder)
            mkdir(diagnostic_folder);
        end

        diagnostic_samples_per_snr = 5;
        diagnostic_maximum_updates = 10000;
        diagnostic_tolerance = 1e-10;
        checkpoint_updates = [1000 2500 5000 10000];
        diagnostic_rank_rows = zeros(numel(N_list)*numel(P_list)*numel(SNR_dB)*diagnostic_samples_per_snr*7, 16);
        diagnostic_row_index = 0;

        fprintf('Validation-only HS-GS convergence diagnostic; no training or test files are loaded.\n');
        fprintf('Samples per N/P/SNR=%d; all candidate ranks; iteration checkpoints=[%s].\n', diagnostic_samples_per_snr, num2str(checkpoint_updates));
        for N_index = 1:numel(N_list)
            N = N_list(N_index);
            K = 3;
            p_H = floor((N-1)/2);
            maximum_candidate_rank = min(7, min(N-p_H, p_H+1));
            diagnostic_ranks = 1:maximum_candidate_rank;
            dimension_folder = fullfile(output_root, sprintf('N%d_K%d_RSR25dB', N, K));

            for P_index = 1:numel(P_list)
                P = P_list(P_index);
                validation_file = fullfile(dimension_folder, sprintf('validation_P%d.mat', P));
                if ~isfile(validation_file)
                    error('Missing validation file: %s', validation_file);
                end
                validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version');
                local_validate_split(validation_data, N, K, P, SNR_dB, RSR_dB, data_version, 500);

                for snr_index = 1:numel(SNR_dB)
                    current_SNR_dB = SNR_dB(snr_index);
                    point_indices = find(double(validation_data.SNR_dB_data) == current_SNR_dB);
                    rng(2026112200 + 10000*N + 100*P + snr_index, 'twister');
                    point_indices = point_indices(randperm(numel(point_indices), diagnostic_samples_per_snr));
                    fprintf('Diagnostic: N=%d, P=%d, SNR=%g dB.\n', N, P, current_SNR_dB);

                    for sample_position = 1:numel(point_indices)
                        sample_index = point_indices(sample_position);
                        Y = double(validation_data.Y_data(:, :, sample_index));
                        G_true = double(validation_data.G_data(:, :, sample_index));
                        S = double(validation_data.S_data(:, :, sample_index));
                        B = double(validation_data.B_data(:, :, sample_index));
                        sigma2 = double(validation_data.sigma2_data(sample_index));

                        for rank_index = 1:numel(diagnostic_ranks)
                            selected_rank = diagnostic_ranks(rank_index);
                            [iteration_count, converged, relative_update_at_checkpoints, nmse_dB_at_checkpoints] = local_hard_rank_hsgs_diagnostic(Y, G_true, S, B, sigma2, p_H, selected_rank, diagnostic_maximum_updates, diagnostic_tolerance, checkpoint_updates);
                            diagnostic_row_index = diagnostic_row_index+1;
                            diagnostic_rank_rows(diagnostic_row_index, :) = [N K P current_SNR_dB sample_index selected_rank double(converged) iteration_count relative_update_at_checkpoints nmse_dB_at_checkpoints];
                        end
                    end
                end
            end
        end

        diagnostic_rank_rows = diagnostic_rank_rows(1:diagnostic_row_index, :);
        diagnostic_table = array2table(diagnostic_rank_rows, 'VariableNames', {'N','K','P','SNR_dB','ValidationSampleIndex','HardRank','Converged','Updates','RelUpdate_1000','RelUpdate_2500','RelUpdate_5000','RelUpdate_10000','NMSE_dB_1000','NMSE_dB_2500','NMSE_dB_5000','NMSE_dB_10000'});
        writetable(diagnostic_table, diagnostic_csv);
        save(diagnostic_file, 'diagnostic_table', 'N_list', 'K_list', 'P_list', 'SNR_dB', 'RSR_dB', 'diagnostic_samples_per_snr', 'diagnostic_maximum_updates', 'diagnostic_tolerance', 'checkpoint_updates', 'data_version');
        fprintf('Saved validation-only convergence diagnostic: %s\n', diagnostic_file);
        fprintf('Use this diagnostic to decide whether more iterations improve estimates before running VALIDATE.\n');

    case 'VALIDATE'
        %% Validation-only selection and confirmation of hard-HS-GS rank
        validation_output_folder = fullfile(results_root, 'validation_policy');
        policy_file = fullfile(validation_output_folder, 'antenna_sweep_hard_rank_policy.mat');
        policy_csv = fullfile(validation_output_folder, 'antenna_sweep_hard_rank_policy.csv');
        if isfile(policy_file) || isfile(policy_csv)
            error('Validation policy already exists. Inspect it before selecting another policy.');
        end
        if ~isfolder(validation_output_folder)
            mkdir(validation_output_folder);
        end

        maximum_updates = 10000;
        convergence_tolerance = 1e-10;
        rank_selection_samples = 250;
        number_of_snr_points = numel(SNR_dB);
        hard_rank_policy = zeros(numel(N_list), numel(P_list), number_of_snr_points);
        selected_rank_validation_gain_dB = zeros(size(hard_rank_policy));
        confirmation_reference_rank = zeros(size(hard_rank_policy));
        validation_rows = zeros(numel(N_list)*numel(P_list)*number_of_snr_points, 12);
        validation_row_index = 0;

        fprintf('Validation-only hard-rank HS-GS selection. No test files are loaded.\n');
        for N_index = 1:numel(N_list)
            N = N_list(N_index);
            K = 3;
            p_H = floor((N-1)/2);
            maximum_candidate_rank = min(7, min(N-p_H, p_H+1));
            hard_rank_candidates = 1:maximum_candidate_rank;
            dimension_folder = fullfile(output_root, sprintf('N%d_K%d_RSR25dB', N, K));

            for P_index = 1:numel(P_list)
                P = P_list(P_index);
                validation_file = fullfile(dimension_folder, sprintf('validation_P%d.mat', P));
                if ~isfile(validation_file)
                    error('Missing validation file: %s', validation_file);
                end
                validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version');
                local_validate_split(validation_data, N, K, P, SNR_dB, RSR_dB, data_version, 500);

                for snr_index = 1:number_of_snr_points
                    current_SNR_dB = SNR_dB(snr_index);
                    point_indices = find(double(validation_data.SNR_dB_data) == current_SNR_dB);
                    rng(2026112100 + 10000*N + 100*P + snr_index, 'twister');
                    point_indices = point_indices(randperm(numel(point_indices)));
                    rank_indices = point_indices(1:rank_selection_samples);
                    confirmation_indices = point_indices(rank_selection_samples+1:end);
                    selection_error_by_sample = zeros(numel(rank_indices), numel(hard_rank_candidates));
                    selection_nonconverged_by_sample = false(numel(rank_indices), numel(hard_rank_candidates));

                    fprintf('Select rank in parallel (up to %d workers): N=%d, P=%d, SNR=%g dB.\n', maximum_parallel_workers, N, P, current_SNR_dB);
                    parfor (sample_position = 1:numel(rank_indices), maximum_parallel_workers)
                        sample_index = rank_indices(sample_position);
                        Y = double(validation_data.Y_data(:, :, sample_index));
                        G_true = double(validation_data.G_data(:, :, sample_index));
                        S = double(validation_data.S_data(:, :, sample_index));
                        B = double(validation_data.B_data(:, :, sample_index));
                        sigma2 = double(validation_data.sigma2_data(sample_index));
                        sample_rank_error = zeros(1, numel(hard_rank_candidates));
                        sample_rank_nonconverged = false(1, numel(hard_rank_candidates));
                        for rank_index = 1:numel(hard_rank_candidates)
                            [G_estimate, estimate_converged] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, hard_rank_candidates(rank_index), maximum_updates, convergence_tolerance);
                            sample_rank_error(rank_index) = norm(G_estimate-G_true, 'fro')^2;
                            sample_rank_nonconverged(rank_index) = ~estimate_converged;
                        end
                        selection_error_by_sample(sample_position, :) = sample_rank_error;
                        selection_nonconverged_by_sample(sample_position, :) = sample_rank_nonconverged;
                    end
                    rank_error = sum(selection_error_by_sample, 1);
                    selection_call_count = numel(selection_nonconverged_by_sample);
                    selection_nonconverged_count = nnz(selection_nonconverged_by_sample);

                    [~, best_rank_index] = min(rank_error);
                    selected_rank = hard_rank_candidates(best_rank_index);
                    confirmation_error_by_sample = zeros(numel(confirmation_indices), 2);
                    confirmation_nonconverged_by_sample = false(numel(confirmation_indices), 2);
                    fprintf('Confirm rank in parallel (up to %d workers): N=%d, P=%d, SNR=%g dB.\n', maximum_parallel_workers, N, P, current_SNR_dB);
                    parfor (sample_position = 1:numel(confirmation_indices), maximum_parallel_workers)
                        sample_index = confirmation_indices(sample_position);
                        Y = double(validation_data.Y_data(:, :, sample_index));
                        G_true = double(validation_data.G_data(:, :, sample_index));
                        S = double(validation_data.S_data(:, :, sample_index));
                        B = double(validation_data.B_data(:, :, sample_index));
                        sigma2 = double(validation_data.sigma2_data(sample_index));
                        [G_selected, selected_converged] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, selected_rank, maximum_updates, convergence_tolerance);
                        [G_maximum, reference_converged] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, maximum_candidate_rank, maximum_updates, convergence_tolerance);
                        sample_confirmation_error = [norm(G_selected-G_true, 'fro')^2 norm(G_maximum-G_true, 'fro')^2];
                        sample_confirmation_nonconverged = [~selected_converged ~reference_converged];
                        confirmation_error_by_sample(sample_position, :) = sample_confirmation_error;
                        confirmation_nonconverged_by_sample(sample_position, :) = sample_confirmation_nonconverged;
                    end
                    selected_confirmation_error = sum(confirmation_error_by_sample(:, 1));
                    maximum_rank_confirmation_error = sum(confirmation_error_by_sample(:, 2));
                    selected_confirmation_nonconverged_count = nnz(confirmation_nonconverged_by_sample(:, 1));
                    reference_confirmation_nonconverged_count = nnz(confirmation_nonconverged_by_sample(:, 2));

                    confirmation_gain_dB = 10*log10(max(maximum_rank_confirmation_error, realmin)/max(selected_confirmation_error, realmin));
                    if selected_rank < maximum_candidate_rank && confirmation_gain_dB <= 0
                        selected_rank = maximum_candidate_rank;
                    end
                    hard_rank_policy(N_index, P_index, snr_index) = selected_rank;
                    selected_rank_validation_gain_dB(N_index, P_index, snr_index) = confirmation_gain_dB;
                    confirmation_reference_rank(N_index, P_index, snr_index) = maximum_candidate_rank;
                    validation_row_index = validation_row_index+1;
                    validation_rows(validation_row_index, :) = [N K P current_SNR_dB p_H selected_rank maximum_candidate_rank confirmation_gain_dB selection_call_count selection_nonconverged_count selected_confirmation_nonconverged_count reference_confirmation_nonconverged_count];
                    fprintf('  Frozen rank=%d (reference rank=%d); held-out validation gain=%+.4f dB; capped estimates: selection %d/%d, confirmation selected/reference %d/%d.\n', selected_rank, maximum_candidate_rank, confirmation_gain_dB, selection_nonconverged_count, selection_call_count, selected_confirmation_nonconverged_count, reference_confirmation_nonconverged_count);
                end
            end
        end

        hard_rank_validation_policy = array2table(validation_rows, 'VariableNames', {'N','K','P','SNR_dB','HankelSplit_p_H','FrozenHardRank','ReferenceRank','ConfirmationGain_dB','SelectionEstimateCount','SelectionNonconvergedCount','ConfirmationSelectedNonconvergedCount','ConfirmationReferenceNonconvergedCount'});
        writetable(hard_rank_validation_policy, policy_csv);
        save(policy_file, 'hard_rank_validation_policy', 'hard_rank_policy', 'selected_rank_validation_gain_dB', 'confirmation_reference_rank', 'N_list', 'K_list', 'P_list', 'SNR_dB', 'RSR_dB', 'maximum_updates', 'convergence_tolerance', 'rank_selection_samples', 'data_version');
        fprintf('Saved validation-frozen hard-rank policy: %s\n', policy_file);
        fprintf('Validation phase did not load or inspect any test data.\n');

    case 'TEST'
        %% Frozen classical estimator evaluation on reserved test data
        validation_output_folder = fullfile(results_root, 'validation_policy');
        policy_file = fullfile(validation_output_folder, 'antenna_sweep_hard_rank_policy.mat');
        test_output_folder = fullfile(results_root, 'test_evaluation');
        result_file = fullfile(test_output_folder, 'antenna_sweep_classical_test_results.mat');
        result_csv = fullfile(test_output_folder, 'antenna_sweep_classical_test_nmse.csv');
        figure_file = fullfile(test_output_folder, 'antenna_sweep_classical_test_nmse.png');
        if ~isfile(policy_file)
            error('Run the VALIDATE phase first. No frozen rank policy was found.');
        end
        if isfile(result_file) || isfile(result_csv) || isfile(figure_file)
            error('Classical test outputs already exist. Inspect them before evaluating again.');
        end
        if ~isfolder(test_output_folder)
            mkdir(test_output_folder);
        end

        policy_data = load(policy_file, 'hard_rank_policy', 'N_list', 'K_list', 'P_list', 'SNR_dB', 'RSR_dB', 'maximum_updates', 'convergence_tolerance', 'data_version');
        if ~isequal(policy_data.N_list, N_list) || ~isequal(policy_data.K_list, K_list) || ~isequal(policy_data.P_list, P_list) || ~isequal(policy_data.SNR_dB, SNR_dB) || policy_data.RSR_dB ~= RSR_dB || ~strcmp(policy_data.data_version, data_version)
            error('Frozen validation policy metadata does not match the antenna sweep.');
        end
        maximum_updates = policy_data.maximum_updates;
        convergence_tolerance = policy_data.convergence_tolerance;
        hard_rank_policy = policy_data.hard_rank_policy;
        gd_initial_variance = 0.1;
        method_names = {'GS','GD','EM-GS','Hard-rank HS-GS'};
        number_of_methods = numel(method_names);
        result_rows = zeros(numel(N_list)*numel(P_list)*numel(SNR_dB)*number_of_methods, 8);
        result_row_index = 0;
        test_error_by_dimension = cell(numel(N_list), numel(P_list));
        test_energy_by_dimension = cell(numel(N_list), numel(P_list));
        test_convergence_by_dimension = cell(numel(N_list), numel(P_list));
        test_updates_by_dimension = cell(numel(N_list), numel(P_list));

        fprintf('Frozen classical test evaluation. Test data are used only for final scoring.\n');
        for N_index = 1:numel(N_list)
            N = N_list(N_index);
            K = 3;
            p_H = floor((N-1)/2);
            dimension_folder = fullfile(output_root, sprintf('N%d_K%d_RSR25dB', N, K));
            for P_index = 1:numel(P_list)
                P = P_list(P_index);
                test_file = fullfile(dimension_folder, sprintf('test_P%d.mat', P));
                if ~isfile(test_file)
                    error('Missing reserved test file: %s', test_file);
                end
                test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version');
                local_validate_split(test_data, N, K, P, SNR_dB, RSR_dB, data_version, 500);
                number_of_samples = size(test_data.Y_data, 3);
                sample_error = zeros(number_of_samples, number_of_methods);
                true_energy = zeros(number_of_samples, 1);
                sample_converged = false(number_of_samples, number_of_methods);
                sample_updates = zeros(number_of_samples, number_of_methods);

                fprintf('Test scoring: N=%d, P=%d, samples=%d.\n', N, P, number_of_samples);
                fprintf('Parallel test scoring (up to %d workers): N=%d, P=%d, samples=%d.\n', maximum_parallel_workers, N, P, number_of_samples);
                parfor (sample_index = 1:number_of_samples, maximum_parallel_workers)
                    Y = double(test_data.Y_data(:, :, sample_index));
                    G_true = double(test_data.G_data(:, :, sample_index));
                    S = double(test_data.S_data(:, :, sample_index));
                    B = double(test_data.B_data(:, :, sample_index));
                    sigma2 = double(test_data.sigma2_data(sample_index));
                    snr_db = double(test_data.SNR_dB_data(sample_index));
                    snr_index = find(SNR_dB == snr_db, 1);
                    if isempty(snr_index)
                        error('Unexpected SNR value in test sample: %g dB.', snr_db);
                    end
                    selected_rank = policy_data.hard_rank_policy(N_index, P_index, snr_index);
                    random_seed = 2026113000 + 100000*N + 1000*P + sample_index;
                    [G_GS, updates_GS, converged_GS] = local_classical_estimator('GS', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
                    [G_GD, updates_GD, converged_GD] = local_classical_estimator('GD', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
                    [G_EMGS, updates_EMGS, converged_EMGS] = local_classical_estimator('EM-GS', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
                    [G_HS_GS, converged_HS_GS, updates_HS_GS] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, selected_rank, maximum_updates, convergence_tolerance);
                    sample_method_updates = [updates_GS updates_GD updates_EMGS updates_HS_GS];
                    sample_method_converged = [converged_GS converged_GD converged_EMGS converged_HS_GS];
                    estimates = {G_GS, G_GD, G_EMGS, G_HS_GS};
                    true_energy(sample_index) = norm(G_true, 'fro')^2;
                    sample_error_row = zeros(1, number_of_methods);
                    for method_index = 1:number_of_methods
                        sample_error_row(method_index) = norm(estimates{method_index}-G_true, 'fro')^2;
                    end
                    sample_error(sample_index, :) = sample_error_row;
                    sample_updates(sample_index, :) = sample_method_updates;
                    sample_converged(sample_index, :) = sample_method_converged;
                end

                test_error_by_dimension{N_index, P_index} = sample_error;
                test_energy_by_dimension{N_index, P_index} = true_energy;
                test_convergence_by_dimension{N_index, P_index} = sample_converged;
                test_updates_by_dimension{N_index, P_index} = sample_updates;
                for snr_index = 1:numel(SNR_dB)
                    snr_indices = find(double(test_data.SNR_dB_data) == SNR_dB(snr_index));
                    if numel(snr_indices) ~= 500
                        error('Expected 500 test samples at N=%d, P=%d, SNR=%g dB.', N, P, SNR_dB(snr_index));
                    end
                    for method_index = 1:number_of_methods
                        nmse_linear = sum(sample_error(snr_indices, method_index))/max(sum(true_energy(snr_indices)), realmin);
                        result_row_index = result_row_index+1;
                        convergence_fraction = mean(sample_converged(snr_indices, method_index));
                        mean_updates = mean(sample_updates(snr_indices, method_index));
                        result_rows(result_row_index, :) = [N K P SNR_dB(snr_index) method_index 10*log10(max(nmse_linear, realmin)) convergence_fraction mean_updates];
                    end
                end
            end
        end

        result_table = array2table(result_rows, 'VariableNames', {'N','K','P','SNR_dB','MethodIndex','NMSE_dB','ConvergedFraction','MeanUpdates'});
        method_labels = strings(height(result_table), 1);
        for row_index = 1:height(result_table)
            method_labels(row_index) = string(method_names{result_table.MethodIndex(row_index)});
        end
        result_table.Method = method_labels;
        result_table = movevars(result_table, 'Method', 'After', 'SNR_dB');
        writetable(result_table, result_csv);
        save(result_file, 'result_table', 'test_error_by_dimension', 'test_energy_by_dimension', 'test_convergence_by_dimension', 'test_updates_by_dimension', 'method_names', 'N_list', 'K_list', 'P_list', 'SNR_dB', 'RSR_dB', 'hard_rank_policy', 'p_H', 'maximum_updates', 'convergence_tolerance', 'gd_initial_variance', 'data_version');

        figure('Color', 'w', 'Name', 'Antenna-count sweep classical estimators');
        tiledlayout(numel(P_list), numel(SNR_dB), 'TileSpacing', 'compact', 'Padding', 'compact');
        colors = lines(number_of_methods);
        for P_index = 1:numel(P_list)
            for snr_index = 1:numel(SNR_dB)
                nexttile;
                hold on;
                for method_index = 1:number_of_methods
                    nmse_curve = zeros(size(N_list));
                    for N_index = 1:numel(N_list)
                        row = result_table.N == N_list(N_index) & result_table.P == P_list(P_index) & result_table.SNR_dB == SNR_dB(snr_index) & result_table.MethodIndex == method_index;
                        nmse_curve(N_index) = result_table.NMSE_dB(row);
                    end
                    plot(N_list, nmse_curve, '-o', 'Color', colors(method_index, :), 'LineWidth', 1.2, 'MarkerSize', 4);
                end
                grid on;
                title(sprintf('P=%d, SNR=%g dB', P_list(P_index), SNR_dB(snr_index)));
                xlabel('Antenna count N');
                ylabel('NMSE (dB)');
            end
        end
        legend(method_names, 'Location', 'southoutside', 'Orientation', 'horizontal');
        exportgraphics(gcf, figure_file, 'Resolution', 200);
        fprintf('Saved frozen classical test results: %s\n', result_file);
        fprintf('Saved test NMSE table: %s\n', result_csv);
        fprintf('Saved test NMSE figure: %s\n', figure_file);

    case 'PLOT_SUMMARY'
        %% Separate P=10 and P=30 figures, each averaged across SNR only
        test_output_folder = fullfile(results_root, 'test_evaluation');
        result_csv = fullfile(test_output_folder, 'antenna_sweep_classical_test_nmse.csv');
        if ~isfile(result_csv)
            error('Run TEST first. The per-condition NMSE table is missing: %s', result_csv);
        end

        per_condition_table = readtable(result_csv);
        method_names = {'GS','GD','EM-GS','Hard-rank HS-GS'};
        for P_index = 1:numel(P_list)
            P = P_list(P_index);
            summary_csv = fullfile(test_output_folder, sprintf('antenna_sweep_classical_test_nmse_P%d.csv', P));
            summary_figure = fullfile(test_output_folder, sprintf('antenna_sweep_classical_test_nmse_P%d.png', P));
            if isfile(summary_csv) || isfile(summary_figure)
                error('P=%d summary output already exists. Inspect or rename it before creating another.', P);
            end

            average_nmse_rows = zeros(numel(N_list)*numel(method_names), 4);
            summary_row_index = 0;
            for N_index = 1:numel(N_list)
                N = N_list(N_index);
                for method_index = 1:numel(method_names)
                    selected_rows = per_condition_table.N == N & per_condition_table.P == P & per_condition_table.MethodIndex == method_index;
                    condition_nmse_dB = per_condition_table.NMSE_dB(selected_rows);
                    if numel(condition_nmse_dB) ~= numel(SNR_dB)
                        error('Expected one NMSE value per SNR for P=%d, N=%d, method=%s.', P, N, method_names{method_index});
                    end
                    mean_linear_nmse = mean(10.^(condition_nmse_dB/10));
                    summary_row_index = summary_row_index+1;
                    average_nmse_rows(summary_row_index, :) = [N method_index mean_linear_nmse 10*log10(mean_linear_nmse)];
                end
            end

            average_nmse_table = array2table(average_nmse_rows, 'VariableNames', {'N','MethodIndex','MeanLinearNMSE','MeanNMSE_dB'});
            method_labels = strings(height(average_nmse_table), 1);
            for row_index = 1:height(average_nmse_table)
                method_labels(row_index) = string(method_names{average_nmse_table.MethodIndex(row_index)});
            end
            average_nmse_table.Method = method_labels;
            average_nmse_table = movevars(average_nmse_table, 'Method', 'After', 'N');
            writetable(average_nmse_table, summary_csv);

            figure('Color', 'w', 'Name', sprintf('Antenna-count sweep classical NMSE, P=%d', P));
            hold on;
            colors = lines(numel(method_names));
            for method_index = 1:numel(method_names)
                method_rows = average_nmse_table.MethodIndex == method_index;
                plot(average_nmse_table.N(method_rows), average_nmse_table.MeanNMSE_dB(method_rows), '-o', 'Color', colors(method_index, :), 'LineWidth', 1.7, 'MarkerSize', 6);
            end
            xticks(N_list);
            grid on;
            xlabel('Antenna count N');
            ylabel('Mean NMSE (dB)');
            title(sprintf('Test NMSE at P=%d, averaged over SNR=%s dB', P, mat2str(SNR_dB)));
            legend(method_names, 'Location', 'best');
            exportgraphics(gcf, summary_figure, 'Resolution', 200);
            fprintf('P=%d figure averages linear NMSE over the %d SNR values only, then converts the mean to dB.\n', P, numel(SNR_dB));
            fprintf('Saved P=%d summary table: %s\n', P, summary_csv);
            fprintf('Saved P=%d figure: %s\n', P, summary_figure);
        end

    otherwise
        error('workflow_phase must be GENERATE, DIAGNOSTIC, VALIDATE, TEST, or PLOT_SUMMARY.');
end

function [iteration_count, converged, relative_update_at_checkpoints, nmse_dB_at_checkpoints] = local_hard_rank_hsgs_diagnostic(Y, G_true, S, B, sigma2, p_H, selected_rank, maximum_updates, convergence_tolerance, checkpoint_updates)
N = size(Y, 1);
K = size(S, 1);
right_inverse = S'/(S*S');
G = local_spectral_initialize(Y, S, B);
relative_update_at_checkpoints = nan(1, numel(checkpoint_updates));
nmse_dB_at_checkpoints = nan(1, numel(checkpoint_updates));
converged = false;
iteration_count = maximum_updates;
checkpoint_index = 1;

for update_index = 1:maximum_updates
    G_previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    G_projected = zeros(N, K);
    hankel_rows = N-p_H;
    hankel_columns = p_H+1;
    for user_index = 1:K
        channel_vector = G_emgs(:, user_index);
        hankel_matrix = zeros(hankel_rows, hankel_columns);
        for hankel_row = 1:hankel_rows
            hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+hankel_columns-1).';
        end
        [U, Sigma, V] = svd(hankel_matrix, 'econ');
        hankel_matrix = U(:, 1:selected_rank)*Sigma(1:selected_rank, 1:selected_rank)*V(:, 1:selected_rank)';
        reconstructed_channel = zeros(N, 1);
        diagonal_counts = zeros(N, 1);
        for hankel_row = 1:hankel_rows
            for hankel_column = 1:hankel_columns
                antenna_index = hankel_row+hankel_column-1;
                reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(hankel_row, hankel_column);
                diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
            end
        end
        G_projected(:, user_index) = reconstructed_channel./diagonal_counts;
    end
    G = G_projected;
    relative_update = norm(G-G_previous, 'fro')^2/max(norm(G, 'fro')^2, eps);

    if checkpoint_index <= numel(checkpoint_updates) && update_index == checkpoint_updates(checkpoint_index)
        relative_update_at_checkpoints(checkpoint_index) = relative_update;
        nmse_linear = norm(G-G_true, 'fro')^2/max(norm(G_true, 'fro')^2, realmin);
        nmse_dB_at_checkpoints(checkpoint_index) = 10*log10(max(nmse_linear, realmin));
        checkpoint_index = checkpoint_index+1;
    end

    if relative_update < convergence_tolerance
        converged = true;
        iteration_count = update_index;
        break;
    end
end

if converged && checkpoint_index <= numel(checkpoint_updates)
    nmse_linear = norm(G-G_true, 'fro')^2/max(norm(G_true, 'fro')^2, realmin);
    for remaining_checkpoint = checkpoint_index:numel(checkpoint_updates)
        relative_update_at_checkpoints(remaining_checkpoint) = relative_update;
        nmse_dB_at_checkpoints(remaining_checkpoint) = 10*log10(max(nmse_linear, realmin));
    end
end
end

function local_validate_split(data, N, K, P, SNR_dB, RSR_dB, expected_version, samples_per_snr)
if data.N ~= N || data.K ~= K || data.P ~= P || data.RSR_dB ~= RSR_dB || ~isequal(double(data.SNR_dB), double(SNR_dB)) || ~strcmp(data.data_version, expected_version)
    error('Dataset metadata mismatch for N=%d, K=%d, P=%d.', N, K, P);
end
expected_count = samples_per_snr*numel(SNR_dB);
if size(data.Y_data, 3) ~= expected_count || size(data.G_data, 3) ~= expected_count || size(data.S_data, 3) ~= expected_count || size(data.B_data, 3) ~= expected_count
    error('Unexpected sample-array size for N=%d, K=%d, P=%d.', N, K, P);
end
for snr_index = 1:numel(SNR_dB)
    if sum(double(data.SNR_dB_data) == SNR_dB(snr_index)) ~= samples_per_snr
        error('Unexpected SNR sample count for N=%d, P=%d, SNR=%g dB.', N, P, SNR_dB(snr_index));
    end
end
end

function [G, iteration_count, converged] = local_classical_estimator(method, Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed)
method = upper(string(method));
if method == "GS"
    G = local_spectral_initialize(Y, S, B);
    right_inverse = S'/(S*S');
elseif method == "GD"
    rng(mod(double(random_seed)+720000, 2^32-1), 'twister');
    N = size(Y, 1);
    K = size(S, 1);
    G = sqrt(gd_initial_variance/2)*(randn(N, K)+1i*randn(N, K));
    phase_reference = exp(-1i*angle(B));
    step_size = 1/max(real(eig(S*S')));
elseif method == "EM-GS"
    G = local_spectral_initialize(Y, S, B);
    right_inverse = S'/(S*S');
else
    error('Unknown local classical estimator method: %s', method);
end

converged = false;
iteration_count = maximum_updates;
for update_index = 1:maximum_updates
    G_previous = G;
    if method == "GS"
        G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    elseif method == "GD"
        G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    else
        field = G*S+B;
        kappa = 2/max(sigma2, eps)*Y.*abs(field);
        bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
        G = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    end
    if norm(G-G_previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

function [G, converged, iteration_count] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, selected_rank, maximum_updates, convergence_tolerance)
N = size(Y, 1);
K = size(S, 1);
right_inverse = S'/(S*S');
G = local_spectral_initialize(Y, S, B);
converged = false;
iteration_count = maximum_updates;
for update_index = 1:maximum_updates
    G_previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    G_projected = zeros(N, K);
    hankel_rows = N-p_H;
    hankel_columns = p_H+1;
    for user_index = 1:K
        channel_vector = G_emgs(:, user_index);
        hankel_matrix = zeros(hankel_rows, hankel_columns);
        for hankel_row = 1:hankel_rows
            hankel_matrix(hankel_row, :) = channel_vector(hankel_row:hankel_row+hankel_columns-1).';
        end
        [U, Sigma, V] = svd(hankel_matrix, 'econ');
        hankel_matrix = U(:, 1:selected_rank)*Sigma(1:selected_rank, 1:selected_rank)*V(:, 1:selected_rank)';
        reconstructed_channel = zeros(N, 1);
        diagonal_counts = zeros(N, 1);
        for hankel_row = 1:hankel_rows
            for hankel_column = 1:hankel_columns
                antenna_index = hankel_row+hankel_column-1;
                reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(hankel_row, hankel_column);
                diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
            end
        end
        G_projected(:, user_index) = reconstructed_channel./diagonal_counts;
    end
    G = G_projected;
    relative_update = norm(G-G_previous, 'fro')^2/max(norm(G, 'fro')^2, eps);
    if relative_update < convergence_tolerance
        converged = true;
        iteration_count = update_index;
        break;
    end
end
end

function G = local_spectral_initialize(Y, S, B)
N = size(Y, 1);
K = size(S, 1);
G = zeros(N, K);
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
    G(antenna_index, :) = (spectral_scale*exp(-1i*angle(principal_vector(K+1)))*principal_vector(1:K)).';
end
end
