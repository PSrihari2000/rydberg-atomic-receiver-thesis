%% Paired bootstrap confidence intervals for the two HS-GS methods
clear;
clc;

result_folder = 'D:\MTech_Thesis_Results\results\N8_OURS_VS_HATIMSTYLE_HSGS_INTERNAL_VALIDATION_RSR25DB';
result_file = fullfile(result_folder, 'N8_ours_vs_pilot_split_hsgs_test_nmse.mat');
saved_results = load(result_file, 'test_error_by_P', 'test_channel_energy_by_P');

P_list = [10 30];
SNR_grid = -5:5:30;
samples_per_SNR = 500;
number_of_bootstrap_samples = 10000;
rng(20261031, 'twister');

bootstrap_results = zeros(numel(P_list)*numel(SNR_grid), 7);
result_row = 0;

for p_index = 1:numel(P_list)
    test_error = saved_results.test_error_by_P{p_index};
    test_channel_energy = saved_results.test_channel_energy_by_P{p_index};

    if size(test_error, 1) ~= samples_per_SNR*numel(SNR_grid) || size(test_error, 2) < 5
        error('Unexpected per-sample test-error array size for P=%d.', P_list(p_index));
    end

    if numel(test_channel_energy) ~= size(test_error, 1)
        error('Test channel-energy array does not match test-error array for P=%d.', P_list(p_index));
    end

    for snr_index = 1:numel(SNR_grid)
        sample_indices = snr_index:numel(SNR_grid):size(test_error, 1);

        if numel(sample_indices) ~= samples_per_SNR
            error('Expected 500 samples for P=%d and SNR=%g dB.', P_list(p_index), SNR_grid(snr_index));
        end

        our_error = test_error(sample_indices, 4);
        pilot_split_error = test_error(sample_indices, 5);
        paired_gain_dB = 10*log10(sum(our_error)/sum(pilot_split_error));

        bootstrap_gain_dB = zeros(number_of_bootstrap_samples, 1);

        for bootstrap_index = 1:number_of_bootstrap_samples
            resampled_positions = randi(samples_per_SNR, samples_per_SNR, 1);
            resampled_indices = sample_indices(resampled_positions);
            bootstrap_gain_dB(bootstrap_index) = 10*log10(sum(test_error(resampled_indices, 4))/sum(test_error(resampled_indices, 5)));
        end

        sorted_bootstrap_gain = sort(bootstrap_gain_dB);
        lower_index = max(1, round(0.025*number_of_bootstrap_samples));
        upper_index = min(number_of_bootstrap_samples, round(0.975*number_of_bootstrap_samples));

        result_row = result_row + 1;
        bootstrap_results(result_row, :) = [P_list(p_index), SNR_grid(snr_index), samples_per_SNR, paired_gain_dB, sorted_bootstrap_gain(lower_index), sorted_bootstrap_gain(upper_index), mean(test_channel_energy(sample_indices))];
    end
end

bootstrap_table = array2table(bootstrap_results, 'VariableNames', {'P','SNR_dB','TestSamples','OurMinusPilotSplit_dB','CI95_Lower_dB','CI95_Upper_dB','MeanTestChannelEnergy'});
disp(bootstrap_table);

csv_file = fullfile(result_folder, 'N8_ours_vs_pilot_split_paired_bootstrap_CI.csv');
writetable(bootstrap_table, csv_file);
fprintf('Paired bootstrap intervals saved to:\n%s\n', csv_file);
fprintf('Negative OurMinusPilotSplit_dB favors our hard-rank method.\n');
fprintf('These are pointwise intervals; they are not simultaneous across all 16 P/SNR points.\n');