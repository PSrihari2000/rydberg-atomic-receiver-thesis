%% Frozen N=16 classical comparison on reserved test data only
% Requires the validation-frozen N=16 HS-GS policy.
% Methods: GS, GD, EM-GS, and soft HS-GS.
% No training or validation observations are loaded by this script.

clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N16_K3_RSR25dB_AoA_Uminus90to90');
output_folder = fullfile(v3_folder, 'results', 'N16_CLASSICAL_COMPARISON_RSR25DB');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
policy_file = fullfile(output_folder, 'N16_frozen_HSGS_validation_policy.mat');
result_file = fullfile(output_folder, 'N16_frozen_classical_test_results.mat');
table_file = fullfile(output_folder, 'N16_frozen_classical_test_table.csv');
figure_file = fullfile(output_folder, 'N16_frozen_classical_test_nmse.png');
if ~isfile(policy_file)
    error('Run DL_V3_N16_SELECT_HSGS_VALIDATION first. Missing policy: %s', policy_file);
end
if isfile(result_file) || isfile(table_file) || isfile(figure_file)
    error('Final N=16 test outputs already exist. Inspect them before starting a new evaluation.');
end

policy_data = load(policy_file, 'configuration', 'frozen_multiplier', 'policy_table');
configuration = policy_data.configuration;
N = 16;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
expected_multiplier_size = [numel(P_list), numel(SNR_grid)];
if configuration.N ~= N || configuration.K ~= K || configuration.RSR_dB ~= RSR_dB || ~isequal(configuration.P_list, P_list) || ~isequal(configuration.SNR_dB, SNR_grid) || ~isequal(size(policy_data.frozen_multiplier), expected_multiplier_size) || ~strcmp(configuration.data_version, 'V3_N16_AoA_Uminus90to90')
    error('Frozen validation policy does not match this N=16 V3 comparison.');
end

p_H = configuration.p_H;
maximum_updates = configuration.maximum_updates;
convergence_tolerance = configuration.convergence_tolerance;
gd_initial_variance = configuration.gd_initial_variance;
bootstrap_replicates = 2000;
method_names = {'GS','GD','EM-GS','HS-GS'};
number_of_methods = numel(method_names);
all_results = cell(numel(P_list), 1);
result_rows = zeros(numel(P_list)*numel(SNR_grid), 32);
row_index = 0;

fprintf('Frozen N=16 classical evaluation on reserved test observations.\n');
fprintf('Methods: GS, Xu-form 1D GD, EM-GS, and validation-frozen soft HS-GS.\n');
fprintf('Xu 2D rank-projection PGD is not a separate 1D method.\n');
fprintf('Maximum updates=%d; relative-change tolerance=%.1e.\n', maximum_updates, convergence_tolerance);
fprintf('No training or validation data files will be loaded.\n');

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    test_file = fullfile(scenario_folder, sprintf('test_P%d.mat', P));
    if ~isfile(test_file)
        error('Missing N=16 reserved test file: %s', test_file);
    end
    test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'AoA_deg_range', 'data_version');
    v3_n16_validate_test_dataset(test_data, N, K, P, SNR_grid, RSR_dB);
    sample_count = size(test_data.Y_data, 3);
    checkpoint_file = fullfile(output_folder, sprintf('N16_test_P%d_checkpoint.mat', P));
    checkpoint_configuration = struct('N', N, 'K', K, 'P', P, 'SNR_dB', SNR_grid, 'RSR_dB', RSR_dB, 'p_H', p_H, 'frozen_multiplier', policy_data.frozen_multiplier(p_index, :), 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'gd_initial_variance', gd_initial_variance, 'data_version', configuration.data_version, 'sample_count', sample_count);

    if isfile(checkpoint_file)
        checkpoint = load(checkpoint_file, 'checkpoint_configuration', 'sample_error', 'true_energy', 'iteration_count', 'convergence_status', 'runtime_seconds', 'sample_completed');
        if ~isequaln(checkpoint.checkpoint_configuration, checkpoint_configuration)
            error('Existing P=%d checkpoint uses different settings. Inspect it before continuing.', P);
        end
        sample_error = checkpoint.sample_error;
        true_energy = checkpoint.true_energy;
        iteration_count = checkpoint.iteration_count;
        convergence_status = checkpoint.convergence_status;
        runtime_seconds = checkpoint.runtime_seconds;
        sample_completed = checkpoint.sample_completed;
    else
        sample_error = nan(sample_count, number_of_methods);
        true_energy = nan(sample_count, 1);
        iteration_count = zeros(sample_count, number_of_methods);
        convergence_status = false(sample_count, number_of_methods);
        runtime_seconds = zeros(sample_count, number_of_methods);
        sample_completed = false(sample_count, 1);
    end

    fprintf('\nN=16 test P=%d: %d samples; completed checkpoint samples=%d.\n', P, sample_count, nnz(sample_completed));
    for sample_index = 1:sample_count
        if sample_completed(sample_index)
            continue;
        end
        Y = double(test_data.Y_data(:, :, sample_index));
        G_true = double(test_data.G_data(:, :, sample_index));
        S = double(test_data.S_data(:, :, sample_index));
        B = double(test_data.B_data(:, :, sample_index));
        sigma2 = double(test_data.sigma2_data(sample_index));
        snr_db = double(test_data.SNR_dB_data(sample_index));
        snr_index = find(abs(SNR_grid-snr_db) < 1e-6, 1);
        if isempty(snr_index)
            error('Unexpected SNR value in P=%d test data: %g.', P, snr_db);
        end
        selected_multiplier = policy_data.frozen_multiplier(p_index, snr_index);
        random_seed = 821000000+1000000*P+10000*snr_index+sample_index;
        for method_index = 1:number_of_methods
            timer = tic;
            [G_estimate, iteration_count(sample_index, method_index), convergence_status(sample_index, method_index)] = V3_CLASSICAL_ESTIMATOR(method_names{method_index}, Y, S, B, sigma2, p_H, selected_multiplier, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
            runtime_seconds(sample_index, method_index) = toc(timer);
            sample_error(sample_index, method_index) = norm(G_estimate-G_true, 'fro')^2;
        end
        true_energy(sample_index) = norm(G_true, 'fro')^2;
        sample_completed(sample_index) = true;
        if mod(sample_index, 20) == 0 || sample_index == sample_count
            save(checkpoint_file, 'checkpoint_configuration', 'sample_error', 'true_energy', 'iteration_count', 'convergence_status', 'runtime_seconds', 'sample_completed', '-v7.3');
            fprintf('P=%d test samples completed: %d/%d.\n', P, sample_index, sample_count);
        end
    end
    if ~all(sample_completed) || any(~isfinite(sample_error), 'all') || any(~isfinite(true_energy)) || any(true_energy <= 0)
        error('P=%d test evaluation is incomplete or invalid. Checkpoint retained.', P);
    end
    all_results{p_index} = struct('P', P, 'sample_error', sample_error, 'true_energy', true_energy, 'iteration_count', iteration_count, 'convergence_status', convergence_status, 'runtime_seconds', runtime_seconds, 'SNR_dB_data', double(test_data.SNR_dB_data), 'selected_multiplier_by_SNR', policy_data.frozen_multiplier(p_index, :));

    for snr_index = 1:numel(SNR_grid)
        snr_db = SNR_grid(snr_index);
        sample_indices = find(abs(double(test_data.SNR_dB_data)-snr_db) < 1e-6);
        if isempty(sample_indices)
            error('No test samples for P=%d, SNR=%g dB.', P, snr_db);
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
            for method_index = 2:number_of_methods
                bootstrap_gain(draw_index, method_index-1) = 10*log10(sum(resampled_errors(:, 1))/max(sum(resampled_errors(:, method_index)), realmin));
            end
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
        result_rows(row_index, :) = [P, snr_db, numel(sample_indices), policy_data.frozen_multiplier(p_index, snr_index), nmse_db, nmse_db(1)-nmse_db(2), nmse_db(1)-nmse_db(3), nmse_db(1)-nmse_db(4), ci_low, ci_high, nmse_db(3)-nmse_db(4), hs_em_ci_low, hs_em_ci_high, mean_iterations, convergence_fraction, mean_runtime];
    end
end

result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','Samples','Frozen_HS_multiplier','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB','HSGS_NMSE_dB','GD_gain_vs_GS_dB','EMGS_gain_vs_GS_dB','HSGS_gain_vs_GS_dB','GD_CI_low_vs_GS_dB','EMGS_CI_low_vs_GS_dB','HSGS_CI_low_vs_GS_dB','GD_CI_high_vs_GS_dB','EMGS_CI_high_vs_GS_dB','HSGS_CI_high_vs_GS_dB','HSGS_gain_vs_EMGS_dB','HSGS_CI_low_vs_EMGS_dB','HSGS_CI_high_vs_EMGS_dB','GS_mean_updates','GD_mean_updates','EMGS_mean_updates','HSGS_mean_updates','GS_convergence_fraction','GD_convergence_fraction','EMGS_convergence_fraction','HSGS_convergence_fraction','GS_mean_runtime_s','GD_mean_runtime_s','EMGS_mean_runtime_s','HSGS_mean_runtime_s'});

figure_handle = figure('Color', 'w', 'Name', 'N16 frozen classical comparison');
for p_index = 1:numel(P_list)
    subplot(1, numel(P_list), p_index);
    selected_rows = result_table.P == P_list(p_index);
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows), '-o', 'LineWidth', 1.4, 'DisplayName', 'GS');
    hold on;
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows)-result_table.GD_gain_vs_GS_dB(selected_rows), '-s', 'LineWidth', 1.4, 'DisplayName', 'GD');
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows)-result_table.EMGS_gain_vs_GS_dB(selected_rows), '-d', 'LineWidth', 1.4, 'DisplayName', 'EM-GS');
    plot(SNR_grid, result_table.GS_NMSE_dB(selected_rows)-result_table.HSGS_gain_vs_GS_dB(selected_rows), '-^', 'LineWidth', 1.4, 'DisplayName', 'Soft HS-GS');
    grid on;
    xlabel('SNR (dB)');
    ylabel('NMSE (dB)');
    title(sprintf('N=16, K=3, P=%d, RSR=25 dB', P_list(p_index)));
    legend('Location', 'best');
end

writetable(result_table, table_file);
save(result_file, 'configuration', 'result_table', 'all_results', 'P_list', 'SNR_grid', 'N', 'K', 'RSR_dB', 'method_names', '-v7.3');
saveas(figure_handle, figure_file);
disp(result_table);
fprintf('Frozen N=16 test comparison saved to %s\n', output_folder);
fprintf('The reserved test observations were used only after loading the frozen validation policy.\n');

function v3_n16_validate_test_dataset(data, expected_N, expected_K, expected_P, expected_SNR, expected_RSR)
if data.N ~= expected_N || data.K ~= expected_K || data.P ~= expected_P || data.RSR_dB ~= expected_RSR || ~isequal(double(data.SNR_dB), expected_SNR) || ~isequal(double(data.AoA_deg_range), [-90 90]) || ~strcmp(data.data_version, 'V3_N16_AoA_Uminus90to90')
    error('Test dataset metadata does not match the required N=16 V3 AoA scenario.');
end
if size(data.Y_data, 1) ~= expected_N || size(data.Y_data, 2) ~= expected_P || size(data.G_data, 1) ~= expected_N || size(data.G_data, 2) ~= expected_K || size(data.S_data, 1) ~= expected_K || size(data.S_data, 2) ~= expected_P || size(data.B_data, 1) ~= expected_N || size(data.B_data, 2) ~= expected_P
    error('Test dataset arrays have unexpected N, K, or P dimensions.');
end
if any(~isfinite(data.SNR_dB_data)) || any(~isfinite(data.sigma2_data)) || any(data.sigma2_data <= 0)
    error('Test file contains invalid SNR or noise variance values.');
end
end
