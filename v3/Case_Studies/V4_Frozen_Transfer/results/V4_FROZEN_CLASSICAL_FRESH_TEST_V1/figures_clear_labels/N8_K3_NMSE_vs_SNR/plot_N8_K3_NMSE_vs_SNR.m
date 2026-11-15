%% Plot the frozen sweep scores at the in-domain N=8, K=3 setting
% Reads the already saved fresh_test_scores.mat only; does not load raw test
% files, rerun estimators, tune parameters, or modify the original outputs.
% Creates separate P=10 and P=30 NMSE-versus-SNR figures and CSV tables.

clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
results_root = fullfile(script_folder, 'results', 'V4_FROZEN_CLASSICAL_FRESH_TEST_V1');
score_file = fullfile(results_root, 'fresh_test_scores.mat');
output_folder = fullfile(results_root, 'figures_clear_labels', 'N8_K3_NMSE_vs_SNR');

if ~isfile(score_file)
    error('Saved sweep scores were not found: %s', score_file);
end
if isfolder(output_folder)
    existing_outputs = dir(fullfile(output_folder, '*'));
    existing_outputs = existing_outputs(~ismember({existing_outputs.name}, {'.','..'}));
    if ~isempty(existing_outputs)
        error('Output folder already exists and is not empty; inspect it or choose a new output folder: %s', output_folder);
    end
end
if ~isfolder(output_folder)
    mkdir(output_folder);
end

loaded = load(score_file, 'result_table', 'SNR_grid', 'P_list');
result_table = loaded.result_table;
SNR_grid = loaded.SNR_grid(:).';
P_list = loaded.P_list(:).';

required_columns = {'StudyIndex','N','K','P','SNR_dB','GS_NMSE','GD_NMSE', ...
    'EMGS_NMSE','HardRank_HSGS_NMSE','FrozenV4_NMSE'};
if ~all(ismember(required_columns, result_table.Properties.VariableNames))
    error('Saved score table is missing one or more required fields.');
end

% In the sweep results StudyIndex=1 denotes the antenna sweep. Its N=8,K=3
% cell supplies one score per P and SNR for the requested fixed-dimension plot.
selected_rows = result_table.StudyIndex == 1 & result_table.N == 8 & result_table.K == 3;
if nnz(selected_rows) ~= numel(P_list)*numel(SNR_grid)
    error('Expected exactly one N=8,K=3 antenna-sweep score for each P/SNR cell.');
end

method_columns = {'GS_NMSE','GD_NMSE','EMGS_NMSE','HardRank_HSGS_NMSE','FrozenV4_NMSE'};
method_labels = {'GS','GD','EM-GS','Hard-rank HS-GS','DL'};
colors = [0.00 0.35 0.75; 0.85 0.25 0.05; 0.95 0.60 0.00; ...
    0.55 0.10 0.80; 0.10 0.60 0.35];
markers = {'o','s','d','^','v'};

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    linear_nmse = nan(numel(SNR_grid), numel(method_columns));
    for snr_index = 1:numel(SNR_grid)
        row = find(selected_rows & result_table.P == P & result_table.SNR_dB == SNR_grid(snr_index));
        if numel(row) ~= 1
            error('Expected one score row for P=%d and SNR=%g dB.', P, SNR_grid(snr_index));
        end
        for method_index = 1:numel(method_columns)
            value = result_table.(method_columns{method_index})(row);
            if ~isfinite(value) || value < 0
                error('Invalid linear NMSE for %s, P=%d, SNR=%g dB.', ...
                    method_labels{method_index}, P, SNR_grid(snr_index));
            end
            linear_nmse(snr_index, method_index) = value;
        end
    end

    nmse_db = 10*log10(max(linear_nmse, realmin));
    summary_table = table(SNR_grid(:), 'VariableNames', {'SNR_dB'});
    summary_table.P = repmat(P, numel(SNR_grid), 1);
    for method_index = 1:numel(method_labels)
        summary_table.([matlab.lang.makeValidName(method_labels{method_index}) '_NMSE_dB']) = nmse_db(:,method_index);
    end
    csv_file = fullfile(output_folder, sprintf('N8_K3_NMSE_vs_SNR_P%d.csv', P));
    writetable(summary_table, csv_file);

    figure('Color','w','Name',sprintf('NMSE vs SNR, P=%d',P));
    hold on;
    for method_index = 1:numel(method_labels)
        plot(SNR_grid, nmse_db(:,method_index), ['-' markers{method_index}], ...
            'Color', colors(method_index,:), 'LineWidth', 1.5, 'MarkerSize', 7, ...
            'DisplayName', method_labels{method_index});
    end
    grid on;
    xlabel('SNR (dB)');
    ylabel('NMSE (dB)');
    title(sprintf('NMSE vs SNR, N=8, K=3, P=%d, RSR=25 dB',P));
    legend('Location','best');
    figure_file = fullfile(output_folder, sprintf('N8_K3_NMSE_vs_SNR_P%d.png', P));
    exportgraphics(gcf, figure_file, 'Resolution', 220);

    fprintf('Saved P=%d figure: %s\n', P, figure_file);
    fprintf('Saved P=%d table: %s\n', P, csv_file);
end

fprintf('Plot-only extraction complete. Read saved sweep scores; no estimators were rerun and no model or policy was tuned.\n');
