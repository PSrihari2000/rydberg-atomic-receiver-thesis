%% Recommended classical user-count sweep with independent data splits
% P=10 uses K=1:5 (K=5 is a labeled measurement-boundary stress point).
% P=30 uses K=1:7 (respecting K<N for N=8).
% Run phases in order: GENERATE, VALIDATE, TEST. Do not inspect test data
% until the validation policy is saved and all estimator settings are frozen.

clear;
clc;

%% -------------------- SELECT PHASE --------------------
workflow_phase = 'PLOT_SUMMARY';
% GENERATE creates train, validation, and test splits for each configured cell.
% VALIDATE selects the hard-rank policy using validation data only.
% TEST evaluates all four frozen classical estimators on the new test splits.
% PLOT_SUMMARY averages the saved per-SNR NMSE values in linear units.

%% -------------------- FROZEN STUDY SETTINGS --------------------
N = 8;
P_list = [10 30];
K_by_P = {[1 2 3 4 5], [1 2 3 4 5 6 7]};
SNR_dB = [-10 0 10 20 30];
RSR_dB = 25;
split_names = {'train','validation','test'};
samples_per_SNR = [2000 500 500];
split_seed_base = [2026111501 2026111601 2026111701];
data_version = 'V3_USER_COUNT_RECOMMENDED_SWEEP_V1';
study_name = 'USER_COUNT_RECOMMENDED_SWEEP';
scenario_name = 'USER_COUNT_RECOMMENDED_P10_K1_2_3_4_5_P30_K1_2_3_4_5_6_7_RSR25_SNR_m10_0_10_20_30';
d_over_lambda = 0.5;
AoA_deg_range = [-90 90];
maximum_parallel_workers = 6;
maximum_updates = 20000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
rank_selection_samples = 250;

script_folder = fileparts(mfilename('fullpath'));
data_root = fullfile(script_folder, 'data', scenario_name);
results_root = fullfile(script_folder, 'results', 'USER_COUNT_RECOMMENDED_SWEEP');
policy_folder = fullfile(results_root, 'validation_policy');
policy_file = fullfile(policy_folder, 'recommended_sweep_hard_rank_policy.csv');
test_folder = fullfile(results_root, 'test_evaluation');

%% -------------------- WORKFLOW --------------------
switch upper(workflow_phase)
    case 'GENERATE'
        if isfolder(data_root)
            error('Refusing to overwrite existing recommended-sweep data: %s', data_root);
        end
        mkdir(data_root);

        hbar = 6.62607015e-34/(2*pi);
        q_charge = 1.602e-19;
        a0_bohr = 5.292e-11;
        mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];

        fprintf('Generating recommended sweep data. N=%d; P=%s; SNR=%s dB; RSR=%g dB.\n', N, mat2str(P_list), mat2str(SNR_dB), RSR_dB);
        fprintf('P=10 K=%s; P=30 K=%s.\n', mat2str(K_by_P{1}), mat2str(K_by_P{2}));
        fprintf('Samples/SNR: train=%d, validation=%d, test=%d.\n', samples_per_SNR(1), samples_per_SNR(2), samples_per_SNR(3));

        for P_index = 1:numel(P_list)
            P = P_list(P_index);
            K_list = K_by_P{P_index};
            for K_index = 1:numel(K_list)
                K = K_list(K_index);
                dimension_folder = fullfile(data_root, sprintf('N%d_K%d_P%d_RSR25dB', N, K, P));
                mkdir(dimension_folder);

                for split_index = 1:numel(split_names)
                    split_name = split_names{split_index};
                    random_seed = split_seed_base(split_index)+10000*N+1000*K+P;
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
                        G = zeros(N, K);
                        for user_index = 1:K
                            number_of_paths = randi([3 7]);
                            for path_index = 1:number_of_paths
                                alpha = (randn+1i*randn)/sqrt(2);
                                theta_deg = AoA_deg_range(1)+diff(AoA_deg_range)*rand;
                                spatial_phase = 2*pi*d_over_lambda*cos(deg2rad(theta_deg));
                                polarization = sqrt(1/3)*randn(3, 1);
                                for antenna_index = 1:N
                                    G(antenna_index, user_index) = G(antenna_index, user_index)+(mu_eg.'*polarization/hbar)*alpha*exp(-1i*(antenna_index-1)*spatial_phase);
                                end
                            end
                        end

                        S = (randn(K, P)+1i*randn(K, P))/sqrt(2);
                        alpha_reference = sqrt(10/2)*(randn+1i*randn);
                        theta_reference_deg = AoA_deg_range(1)+diff(AoA_deg_range)*rand;
                        reference_phase = 2*pi*d_over_lambda*cos(deg2rad(theta_reference_deg));
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
                    end

                    measurement_model = 'Y = abs(G*S + B + W)';
                    RSR_definition = 'reference power divided by average single-user received signal power';
                    AoA_distribution = 'Uniform incident angle';
                    spatial_phase_convention = 'phi = 2*pi*(d/lambda)*cos(theta), theta in radians';
                    output_file = fullfile(dimension_folder, sprintf('%s.mat', split_name));
                    save(output_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'measurement_model', 'RSR_definition', 'AoA_distribution', 'AoA_deg_range', 'spatial_phase_convention', 'data_version', 'study_name', 'random_seed', 'samples_this_split_per_snr', 'd_over_lambda', 'N', 'K', 'P', 'P_list', 'SNR_dB', 'RSR_dB', 'number_of_samples', 'split_name', '-v7.3');
                    fprintf('Saved N=%d, K=%d, P=%d, %s.\n', N, K, P, split_name);
                end
            end
        end
        fprintf('Recommended-sweep data generation complete: %s\n', data_root);
        fprintf('No estimators were run and no test data were scored.\n');

    case 'VALIDATE'
        if ~isfolder(data_root)
            error('Data folder not found. Run GENERATE first: %s', data_root);
        end
        if isfile(policy_file)
            error('Validation policy already exists. Refusing to overwrite: %s', policy_file);
        end
        if ~isfolder(policy_folder)
            mkdir(policy_folder);
        end

        p_H = floor((N-1)/2);
        candidate_ranks = 1:min(7, min(N-p_H, p_H+1));
        policy_rows = zeros(sum(cellfun(@numel, K_by_P))*numel(SNR_dB), 13);
        row_index = 0;
        fprintf('Validation-only hard-rank selection; no test files are opened.\n');

        for P_index = 1:numel(P_list)
            P = P_list(P_index);
            K_list = K_by_P{P_index};
            for K_index = 1:numel(K_list)
                K = K_list(K_index);
                dimension_folder = fullfile(data_root, sprintf('N%d_K%d_P%d_RSR25dB', N, K, P));
                validation_file = fullfile(dimension_folder, 'validation.mat');
                validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version');
                local_validate_split(validation_data, N, K, P, SNR_dB, RSR_dB, data_version, samples_per_SNR(2));

                for snr_index = 1:numel(SNR_dB)
                    current_SNR_dB = SNR_dB(snr_index);
                    sample_indices = find(double(validation_data.SNR_dB_data) == current_SNR_dB);
                    rng(2026111800+10000*K+100*P+snr_index, 'twister');
                    sample_indices = sample_indices(randperm(numel(sample_indices)));
                    selection_indices = sample_indices(1:rank_selection_samples);
                    confirmation_indices = sample_indices(rank_selection_samples+1:end);
                    selection_error = zeros(numel(selection_indices), numel(candidate_ranks));
                    selection_nonconverged = false(numel(selection_indices), numel(candidate_ranks));

                    fprintf('Rank validation: K=%d, P=%d, SNR=%g dB.\n', K, P, current_SNR_dB);
                    parfor (sample_position = 1:numel(selection_indices), maximum_parallel_workers)
                        sample_index = selection_indices(sample_position);
                        Y = double(validation_data.Y_data(:, :, sample_index));
                        G_true = double(validation_data.G_data(:, :, sample_index));
                        S = double(validation_data.S_data(:, :, sample_index));
                        B = double(validation_data.B_data(:, :, sample_index));
                        sigma2 = double(validation_data.sigma2_data(sample_index));
                        sample_errors = zeros(1, numel(candidate_ranks));
                        sample_nonconverged = false(1, numel(candidate_ranks));
                        for rank_index = 1:numel(candidate_ranks)
                            [estimate, estimate_converged] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, candidate_ranks(rank_index), maximum_updates, convergence_tolerance);
                            sample_errors(rank_index) = norm(estimate-G_true, 'fro')^2;
                            sample_nonconverged(rank_index) = ~estimate_converged;
                        end
                        selection_error(sample_position, :) = sample_errors;
                        selection_nonconverged(sample_position, :) = sample_nonconverged;
                    end
                    [~, selected_index] = min(sum(selection_error, 1));
                    selected_rank = candidate_ranks(selected_index);

                    confirmation_error = zeros(numel(confirmation_indices), 2);
                    confirmation_nonconverged = false(numel(confirmation_indices), 2);
                    parfor (sample_position = 1:numel(confirmation_indices), maximum_parallel_workers)
                        sample_index = confirmation_indices(sample_position);
                        Y = double(validation_data.Y_data(:, :, sample_index));
                        G_true = double(validation_data.G_data(:, :, sample_index));
                        S = double(validation_data.S_data(:, :, sample_index));
                        B = double(validation_data.B_data(:, :, sample_index));
                        sigma2 = double(validation_data.sigma2_data(sample_index));
                        [estimate_selected, selected_converged] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, selected_rank, maximum_updates, convergence_tolerance);
                        [estimate_maximum, maximum_converged] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, candidate_ranks(end), maximum_updates, convergence_tolerance);
                        confirmation_error(sample_position, :) = [norm(estimate_selected-G_true, 'fro')^2 norm(estimate_maximum-G_true, 'fro')^2];
                        confirmation_nonconverged(sample_position, :) = [~selected_converged ~maximum_converged];
                    end
                    confirmation_gain_dB = 10*log10(max(sum(confirmation_error(:, 2)), realmin)/max(sum(confirmation_error(:, 1)), realmin));
                    if selected_rank < candidate_ranks(end) && confirmation_gain_dB <= 0
                        selected_rank = candidate_ranks(end);
                    end
                    row_index = row_index+1;
                    selected_confirmation_nonconverged_count = nnz(confirmation_nonconverged(:, 1));
                    maximum_confirmation_nonconverged_count = nnz(confirmation_nonconverged(:, 2));
                    policy_rows(row_index, :) = [N K P current_SNR_dB p_H selected_rank candidate_ranks(end) confirmation_gain_dB numel(selection_indices) nnz(selection_nonconverged) numel(confirmation_indices) selected_confirmation_nonconverged_count maximum_confirmation_nonconverged_count];
                    fprintf('  Frozen rank=%d of %d; confirmation gain=%+.3f dB.\n', selected_rank, candidate_ranks(end), confirmation_gain_dB);
                end
            end
        end

        hard_rank_validation_policy = array2table(policy_rows, 'VariableNames', {'N','K','P','SNR_dB','HankelSplit_p_H','FrozenHardRank','ReferenceRank','ConfirmationGain_dB','SelectionSamples','SelectionNonconvergedCount','ConfirmationSamples','SelectedConfirmationNonconvergedCount','ReferenceConfirmationNonconvergedCount'});
        writetable(hard_rank_validation_policy, policy_file);
        save(fullfile(policy_folder, 'recommended_sweep_hard_rank_policy.mat'), 'hard_rank_validation_policy', 'N', 'K_by_P', 'P_list', 'SNR_dB', 'RSR_dB', 'maximum_updates', 'convergence_tolerance', 'data_version');
        fprintf('Saved validation-frozen policy: %s\n', policy_file);
        fprintf('Validation did not load or inspect test data.\n');

    case 'TEST'
        if ~isfile(policy_file)
            error('Frozen validation policy is missing. Run VALIDATE first: %s', policy_file);
        end
        if ~isfolder(data_root)
            error('Data folder is missing. Run GENERATE first: %s', data_root);
        end
        if isfolder(test_folder)
            error('Test output folder already exists. Refusing to overwrite: %s', test_folder);
        end
        mkdir(test_folder);
        policy_table = readtable(policy_file);
        p_H = floor((N-1)/2);
        method_names = {'GS','GD','EM-GS','Hard-rank HS-GS'};
        result_rows = zeros(sum(cellfun(@numel, K_by_P))*numel(SNR_dB)*numel(method_names), 9);
        row_index = 0;
        test_error_by_cell = cell(sum(cellfun(@numel, K_by_P)), numel(P_list));
        test_energy_by_cell = cell(sum(cellfun(@numel, K_by_P)), numel(P_list));
        test_convergence_by_cell = cell(sum(cellfun(@numel, K_by_P)), numel(P_list));
        test_updates_by_cell = cell(sum(cellfun(@numel, K_by_P)), numel(P_list));
        cell_counter = 0;

        fprintf('Scoring the frozen recommended sweep on its reserved test split.\n');
        for P_index = 1:numel(P_list)
            P = P_list(P_index);
            K_list = K_by_P{P_index};
            for K_index = 1:numel(K_list)
                K = K_list(K_index);
                cell_counter = cell_counter+1;
                dimension_folder = fullfile(data_root, sprintf('N%d_K%d_P%d_RSR25dB', N, K, P));
                test_data = load(fullfile(dimension_folder, 'test.mat'), 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version');
                local_validate_split(test_data, N, K, P, SNR_dB, RSR_dB, data_version, samples_per_SNR(3));
                number_of_samples = size(test_data.Y_data, 3);
                sample_errors = zeros(number_of_samples, numel(method_names));
                sample_energy = zeros(number_of_samples, 1);
                sample_converged = false(number_of_samples, numel(method_names));
                sample_updates = zeros(number_of_samples, numel(method_names));

                fprintf('Test scoring: K=%d, P=%d, samples=%d.\n', K, P, number_of_samples);
                parfor (sample_index = 1:number_of_samples, maximum_parallel_workers)
                    Y = double(test_data.Y_data(:, :, sample_index));
                    G_true = double(test_data.G_data(:, :, sample_index));
                    S = double(test_data.S_data(:, :, sample_index));
                    B = double(test_data.B_data(:, :, sample_index));
                    sigma2 = double(test_data.sigma2_data(sample_index));
                    snr_value = double(test_data.SNR_dB_data(sample_index));
                    snr_index = find(SNR_dB == snr_value, 1);
                    policy_row = policy_table.K == K & policy_table.P == P & policy_table.SNR_dB == snr_value;
                    if isempty(snr_index) || nnz(policy_row) ~= 1
                        error('Missing unique frozen policy for K=%d, P=%d, SNR=%g.', K, P, snr_value);
                    end
                    selected_rank = policy_table.FrozenHardRank(policy_row);
                    estimator_seed = 2026113000+100000*N+10000*K+1000*P+sample_index;
                    [G_GS, updates_GS, converged_GS] = local_classical_estimator('GS', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, estimator_seed);
                    [G_GD, updates_GD, converged_GD] = local_classical_estimator('GD', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, estimator_seed);
                    [G_EMGS, updates_EMGS, converged_EMGS] = local_classical_estimator('EM-GS', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, estimator_seed);
                    [G_HS_GS, converged_HS_GS, updates_HS_GS] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, selected_rank, maximum_updates, convergence_tolerance);
                    sample_error_row = [norm(G_GS-G_true, 'fro')^2 norm(G_GD-G_true, 'fro')^2 norm(G_EMGS-G_true, 'fro')^2 norm(G_HS_GS-G_true, 'fro')^2];
                    sample_errors(sample_index, :) = sample_error_row;
                    sample_energy(sample_index) = norm(G_true, 'fro')^2;
                    sample_converged(sample_index, :) = [converged_GS converged_GD converged_EMGS converged_HS_GS];
                    sample_updates(sample_index, :) = [updates_GS updates_GD updates_EMGS updates_HS_GS];
                end

                test_error_by_cell{cell_counter, P_index} = sample_errors;
                test_energy_by_cell{cell_counter, P_index} = sample_energy;
                test_convergence_by_cell{cell_counter, P_index} = sample_converged;
                test_updates_by_cell{cell_counter, P_index} = sample_updates;
                for snr_index = 1:numel(SNR_dB)
                    selected_samples = find(double(test_data.SNR_dB_data) == SNR_dB(snr_index));
                    if numel(selected_samples) ~= samples_per_SNR(3)
                        error('Unexpected test count for K=%d, P=%d, SNR=%g.', K, P, SNR_dB(snr_index));
                    end
                    policy_row = policy_table.K == K & policy_table.P == P & policy_table.SNR_dB == SNR_dB(snr_index);
                    frozen_rank = policy_table.FrozenHardRank(policy_row);
                    for method_index = 1:numel(method_names)
                        nmse_linear = sum(sample_errors(selected_samples, method_index))/max(sum(sample_energy(selected_samples)), realmin);
                        row_index = row_index+1;
                        result_rows(row_index, :) = [N K P SNR_dB(snr_index) method_index frozen_rank 10*log10(max(nmse_linear, realmin)) mean(sample_converged(selected_samples, method_index)) mean(sample_updates(selected_samples, method_index))];
                    end
                end
            end
        end

        result_table = array2table(result_rows, 'VariableNames', {'N','K','P','SNR_dB','MethodIndex','FrozenHardRank','NMSE_dB','ConvergedFraction','MeanUpdates'});
        method_labels = strings(height(result_table), 1);
        for result_index = 1:height(result_table)
            method_labels(result_index) = string(method_names{result_table.MethodIndex(result_index)});
        end
        result_table.Method = method_labels;
        result_table = movevars(result_table, 'Method', 'After', 'SNR_dB');
        result_csv = fullfile(test_folder, 'recommended_user_sweep_test_nmse.csv');
        result_mat = fullfile(test_folder, 'recommended_user_sweep_test_results.mat');
        writetable(result_table, result_csv);
        save(result_mat, 'result_table', 'test_error_by_cell', 'test_energy_by_cell', 'test_convergence_by_cell', 'test_updates_by_cell', 'method_names', 'N', 'K_by_P', 'P_list', 'SNR_dB', 'RSR_dB', 'maximum_updates', 'convergence_tolerance', 'gd_initial_variance', 'data_version');

        colors = [0.00 0.35 0.75; 0.85 0.25 0.05; 0.95 0.60 0.00; 0.55 0.10 0.80];
        method_markers = {'o','s','d','^'};
        method_line_styles = {'-','--','-.',':'};
        for P_index = 1:numel(P_list)
            P = P_list(P_index);
            K_list = K_by_P{P_index};
            figure('Color', 'w', 'Name', sprintf('Recommended user sweep, P=%d', P));
            tiledlayout(1, numel(SNR_dB), 'TileSpacing', 'compact', 'Padding', 'compact');
            for snr_index = 1:numel(SNR_dB)
                nexttile;
                hold on;
                for method_index = 1:numel(method_names)
                    curve = nan(size(K_list));
                    for K_index = 1:numel(K_list)
                        row = result_table.K == K_list(K_index) & result_table.P == P & result_table.SNR_dB == SNR_dB(snr_index) & result_table.MethodIndex == method_index;
                        curve(K_index) = result_table.NMSE_dB(row);
                    end
                    plot(K_list, curve, 'LineStyle', method_line_styles{method_index}, 'Marker', method_markers{method_index}, 'Color', colors(method_index, :), 'LineWidth', 1.25, 'MarkerSize', 5);
                end
                grid on;
                title(sprintf('SNR=%g dB', SNR_dB(snr_index)));
                xlabel('User count K');
                ylabel('NMSE (dB)');
            end
            legend(method_names, 'Location', 'southoutside', 'Orientation', 'horizontal');
            figure_file = fullfile(test_folder, sprintf('recommended_user_sweep_test_nmse_P%d.png', P));
            exportgraphics(gcf, figure_file, 'Resolution', 220);
            fprintf('Saved P=%d test figure: %s\n', P, figure_file);
        end
        fprintf('Saved fresh-split test NMSE table: %s\n', result_csv);
        fprintf('Saved frozen test results: %s\n', result_mat);
        fprintf('No settings were tuned on the test split.\n');

    case 'PLOT_SUMMARY'
        %% Separate P=10 and P=30 summaries averaged across the SNR grid
        result_csv = fullfile(test_folder, 'recommended_user_sweep_test_nmse.csv');
        summary_folder = fullfile(test_folder, 'summary_over_SNR');
        method_names = {'GS','GD','EM-GS','Hard-rank HS-GS'};
        if ~isfile(result_csv)
            error('Frozen test NMSE table is missing: %s', result_csv);
        end
        summary_output_files = cell(1, 2*numel(P_list));
        summary_output_index = 0;
        for output_P = P_list
            summary_output_index = summary_output_index+1;
            summary_output_files{summary_output_index} = fullfile(summary_folder, sprintf('recommended_user_sweep_test_nmse_SNRavg_P%d.csv', output_P));
            summary_output_index = summary_output_index+1;
            summary_output_files{summary_output_index} = fullfile(summary_folder, sprintf('recommended_user_sweep_test_nmse_SNRavg_P%d.png', output_P));
        end
        if any(cellfun(@isfile, summary_output_files))
            error('One or more SNR-summary outputs already exist. Inspect them before rerunning.');
        end
        if ~isfolder(summary_folder)
            mkdir(summary_folder);
        end
        result_table = readtable(result_csv);
        required_columns = {'N','K','P','SNR_dB','Method','NMSE_dB','ConvergedFraction'};
        if ~all(ismember(required_columns, result_table.Properties.VariableNames))
            error('Test NMSE table does not contain the expected columns.');
        end
        summary_method_markers = {'o','s','d','^'};
        summary_method_styles = {'-','--','-.',':'};
        summary_colors = [0.00 0.35 0.75; 0.85 0.25 0.05; 0.95 0.60 0.00; 0.55 0.10 0.80];
        fprintf('Summary convention: arithmetic mean of linear NMSE over SNR, converted to dB afterward.\n');

        for P_index = 1:numel(P_list)
            P = P_list(P_index);
            K_list = K_by_P{P_index};
            summary_rows = zeros(numel(K_list)*numel(method_names), 7);
            summary_row_index = 0;
            for K_index = 1:numel(K_list)
                K = K_list(K_index);
                for method_index = 1:numel(method_names)
                    selected_rows = result_table.K == K & result_table.P == P & string(result_table.Method) == method_names{method_index};
                    method_snr_rows = result_table(selected_rows, :);
                    if height(method_snr_rows) ~= numel(SNR_dB) || ~all(ismember(SNR_dB, method_snr_rows.SNR_dB))
                        error('Expected one saved result for each SNR at K=%d, P=%d, method=%s.', K, P, method_names{method_index});
                    end
                    mean_nmse_linear = mean(10.^(double(method_snr_rows.NMSE_dB)/10));
                    mean_nmse_dB = 10*log10(max(mean_nmse_linear, realmin));
                    mean_convergence = mean(double(method_snr_rows.ConvergedFraction));
                    summary_row_index = summary_row_index+1;
                    summary_rows(summary_row_index, :) = [N K P method_index mean_nmse_linear mean_nmse_dB mean_convergence];
                end
            end

            summary_table = array2table(summary_rows, 'VariableNames', {'N','K','P','MethodIndex','MeanNMSE_linear','MeanNMSE_dB','MeanConvergedFraction'});
            summary_method_labels = strings(height(summary_table), 1);
            for summary_index = 1:height(summary_table)
                summary_method_labels(summary_index) = string(method_names{summary_table.MethodIndex(summary_index)});
            end
            summary_table.Method = summary_method_labels;
            summary_table = movevars(summary_table, 'Method', 'After', 'P');
            summary_csv = fullfile(summary_folder, sprintf('recommended_user_sweep_test_nmse_SNRavg_P%d.csv', P));
            writetable(summary_table, summary_csv);

            figure('Color', 'w', 'Name', sprintf('Recommended user sweep mean NMSE, P=%d', P));
            hold on;
            for method_index = 1:numel(method_names)
                method_rows = summary_table.MethodIndex == method_index;
                method_summary = summary_table(method_rows, :);
                [~, sort_indices] = sort(method_summary.K);
                method_summary = method_summary(sort_indices, :);
                plot(method_summary.K, method_summary.MeanNMSE_dB, 'LineStyle', summary_method_styles{method_index}, 'Marker', summary_method_markers{method_index}, 'Color', summary_colors(method_index, :), 'LineWidth', 1.35, 'MarkerSize', 6, 'DisplayName', method_names{method_index});
            end
            grid on;
            xlabel('User count K');
            ylabel('Mean NMSE over SNR (dB)');
            title(sprintf('Test mean NMSE over SNR=%s dB, P=%d', mat2str(SNR_dB), P));
            legend('Location', 'best');
            summary_figure = fullfile(summary_folder, sprintf('recommended_user_sweep_test_nmse_SNRavg_P%d.png', P));
            exportgraphics(gcf, summary_figure, 'Resolution', 220);
            fprintf('Saved P=%d SNR-averaged summary table: %s\n', P, summary_csv);
            fprintf('Saved P=%d SNR-averaged figure: %s\n', P, summary_figure);
        end
        fprintf('Summary phase read only the saved test NMSE CSV; no samples were loaded or scored.\n');

    otherwise
        error('workflow_phase must be GENERATE, VALIDATE, TEST, or PLOT_SUMMARY.');
end

%% -------------------- LOCAL FUNCTIONS --------------------
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
        error('Unexpected sample count for K=%d, P=%d, SNR=%g.', K, P, SNR_dB(snr_index));
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
    if norm(G-G_previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
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
    A = [S.', b];
    M = A'*diag(y)*A;
    [eigenvectors, eigenvalues] = eig(M);
    [~, principal_index] = max(real(diag(eigenvalues)));
    v = eigenvectors(:, principal_index);
    v = v/max(norm(v), eps);
    scale = (abs(A*v).'*y)/max(norm(A*v)^2, eps);
    G(antenna_index, :) = (scale*exp(-1i*angle(v(K+1)))*v(1:K)).';
end
end
