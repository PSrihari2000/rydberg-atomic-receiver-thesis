%% Paired uncertainty analysis for the frozen confirmatory V2 shared-K test
% Reads the already-saved paired sample error energies only. It does not
% load channel data, rerun estimators, select models, or modify test outputs.

clear;
clc;
close all;

%% Frozen analysis settings
N = 8;
P_list = [10 30];
K_by_P = {[1 2 3 4 5], [1 2 3 4 5 6 7]};
SNR_grid = [-10 0 10 20 30];
expected_samples_per_condition = 500;
bootstrap_replicates = 5000;
confidence_level = 0.95;
random_seed = 2026111101;

%% Input and isolated output paths
script_folder = fileparts(mfilename('fullpath'));
test_output_folder = fullfile(script_folder, 'results', 'USER_COUNT_SHARED_PI_DU_RACE_K_SWEEP_CONFIRMATORY_V2', 'test_evaluation');
sample_file = fullfile(test_output_folder, 'confirmatory_v2_shared_K_DL_test_sample_errors.csv');
output_folder = fullfile(test_output_folder, 'paired_uncertainty_v1');

if ~isfile(sample_file)
    error('Saved paired sample-error table was not found: %s', sample_file);
end
if isfolder(output_folder)
    error('Paired uncertainty output already exists. Inspect it and choose a new versioned folder: %s', output_folder);
end

sample_table = readtable(sample_file);
required_columns = {'N','K','P','SNR_dB','SampleIndex','SharedErrorEnergy','V4ErrorEnergy','TrueChannelEnergy'};
if ~all(ismember(required_columns, sample_table.Properties.VariableNames))
    error('Paired sample table is missing one or more expected columns.');
end

if any(sample_table.N ~= N) || any(~isfinite(sample_table.SharedErrorEnergy)) || any(~isfinite(sample_table.V4ErrorEnergy)) || any(~isfinite(sample_table.TrueChannelEnergy)) || any(sample_table.TrueChannelEnergy <= 0)
    error('Sample table contains invalid dimensions, error energies, or truth energies.');
end

mkdir(output_folder);
rng(random_seed, 'twister');
tail_probability = (1-confidence_level)/2;
condition_rows = cell(0, 10);
snr_average_rows = cell(0, 9);

fprintf('Paired bootstrap uncertainty analysis of frozen shared-K test scores.\n');
fprintf('Bootstrap replicates=%d; confidence level=%.1f%%; seed=%d.\n', bootstrap_replicates, 100*confidence_level, random_seed);
fprintf('Positive gain means lower NMSE for Shared-K DL than frozen V4.\n');

%% Bootstrap matched samples within each K/P/SNR condition
for P_index = 1:numel(P_list)
    P = P_list(P_index);
    K_list = K_by_P{P_index};

    for K = K_list
        snr_mean_v4 = zeros(numel(SNR_grid), 1);
        snr_mean_shared = zeros(numel(SNR_grid), 1);
        snr_boot_v4 = zeros(numel(SNR_grid), bootstrap_replicates);
        snr_boot_shared = zeros(numel(SNR_grid), bootstrap_replicates);

        for SNR_index = 1:numel(SNR_grid)
            SNR_dB = SNR_grid(SNR_index);
            selected = sample_table.N == N & sample_table.K == K & sample_table.P == P & sample_table.SNR_dB == SNR_dB;
            rows = sample_table(selected, :);
            if height(rows) ~= expected_samples_per_condition || numel(unique(rows.SampleIndex)) ~= expected_samples_per_condition
                error('Expected %d unique samples at N=%d,K=%d,P=%d,SNR=%g.', expected_samples_per_condition, N, K, P, SNR_dB);
            end

            nmse_v4 = rows.V4ErrorEnergy ./ rows.TrueChannelEnergy;
            nmse_shared = rows.SharedErrorEnergy ./ rows.TrueChannelEnergy;
            snr_mean_v4(SNR_index) = mean(nmse_v4);
            snr_mean_shared(SNR_index) = mean(nmse_shared);

            sample_count = height(rows);
            bootstrap_indices = randi(sample_count, sample_count, bootstrap_replicates);
            snr_boot_v4(SNR_index, :) = mean(nmse_v4(bootstrap_indices), 1);
            snr_boot_shared(SNR_index, :) = mean(nmse_shared(bootstrap_indices), 1);

            point_gain_db = 10*log10(snr_mean_v4(SNR_index) / snr_mean_shared(SNR_index));
            bootstrap_gain_db = 10*log10(snr_boot_v4(SNR_index, :) ./ snr_boot_shared(SNR_index, :));
            bootstrap_gain_db = sort(bootstrap_gain_db);
            lower_index = max(1, floor(tail_probability*bootstrap_replicates));
            upper_index = min(bootstrap_replicates, ceil((1-tail_probability)*bootstrap_replicates));
            gain_ci_low = bootstrap_gain_db(lower_index);
            gain_ci_high = bootstrap_gain_db(upper_index);

            condition_rows(end+1, :) = {N, K, P, SNR_dB, sample_count, snr_mean_v4(SNR_index), snr_mean_shared(SNR_index), point_gain_db, gain_ci_low, gain_ci_high};
        end

        % Match the sweep's aggregation: average linear NMSE across SNR,
        % then convert the ratio to dB. Bootstrap independently within each
        % SNR stratum while retaining the paired V4/shared observations.
        average_v4 = mean(snr_mean_v4);
        average_shared = mean(snr_mean_shared);
        aggregate_gain_db = 10*log10(average_v4 / average_shared);
        bootstrap_average_v4 = mean(snr_boot_v4, 1);
        bootstrap_average_shared = mean(snr_boot_shared, 1);
        aggregate_bootstrap_gain_db = sort(10*log10(bootstrap_average_v4 ./ bootstrap_average_shared));
        aggregate_ci_low = aggregate_bootstrap_gain_db(lower_index);
        aggregate_ci_high = aggregate_bootstrap_gain_db(upper_index);

        snr_average_rows(end+1, :) = {N, K, P, numel(SNR_grid), average_v4, average_shared, aggregate_gain_db, aggregate_ci_low, aggregate_ci_high};
    end
end

condition_table = cell2table(condition_rows, 'VariableNames', {'N','K','P','SNR_dB','Samples','V4MeanNMSE_linear','SharedMeanNMSE_linear','SharedGain_dB','CI95_Lower_dB','CI95_Upper_dB'});
snr_average_table = cell2table(snr_average_rows, 'VariableNames', {'N','K','P','SNRCount','V4MeanNMSE_linear_over_SNR','SharedMeanNMSE_linear_over_SNR','SharedGain_dB','CI95_Lower_dB','CI95_Upper_dB'});

condition_csv = fullfile(output_folder, 'confirmatory_v2_shared_K_paired_CI_by_condition.csv');
snr_average_csv = fullfile(output_folder, 'confirmatory_v2_shared_K_paired_CI_SNRavg.csv');
writetable(condition_table, condition_csv);
writetable(snr_average_table, snr_average_csv);

for P = P_list
    selected = snr_average_table.P == P;
    figure('Color', 'w', 'Name', sprintf('Shared-K paired CI, P=%d', P));
    errorbar(snr_average_table.K(selected), snr_average_table.SharedGain_dB(selected), snr_average_table.SharedGain_dB(selected)-snr_average_table.CI95_Lower_dB(selected), snr_average_table.CI95_Upper_dB(selected)-snr_average_table.SharedGain_dB(selected), 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
    hold on;
    yline(0, 'k--', 'V4 parity', 'LineWidth', 1.0);
    grid on;
    xlabel('User count K');
    ylabel('Shared-K gain over V4 (dB)');
    title(sprintf('Paired test gain with 95%% bootstrap CI, P=%d', P));
    K_plot = K_by_P{P_index_for(P, P_list)};
    xlim([min(K_plot)-0.25, max(K_plot)+0.25]);
    figure_file = fullfile(output_folder, sprintf('confirmatory_v2_shared_K_paired_CI_SNRavg_P%d.png', P));
    exportgraphics(gcf, figure_file, 'Resolution', 200);
end

fprintf('Saved per-condition paired intervals: %s\n', condition_csv);
fprintf('Saved SNR-averaged paired intervals: %s\n', snr_average_csv);
fprintf('Saved separate P=10 and P=30 confidence-interval figures in: %s\n', output_folder);
fprintf('This script only analyzed the already-used test scores; it did not tune models.\n');

function index = P_index_for(P, P_list)
index = find(P_list == P, 1);
end
