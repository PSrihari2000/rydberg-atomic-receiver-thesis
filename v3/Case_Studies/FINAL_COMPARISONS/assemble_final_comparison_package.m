%% Assemble the existing final comparisons into one reporting package
% Reads saved CSV summaries only. Does not load raw samples, train, tune, or
% rerun any estimator. Existing source scripts and result files are untouched.

clear;
clc;
close all;

%% Locate sources and create a new, non-overwriting output release
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
project_folder = fileparts(case_studies_folder);
output_folder = fullfile(script_folder, 'outputs_v1');

if isfolder(output_folder)
    error('Output folder already exists. Inspect it or choose a new release name: %s', output_folder);
end
mkdir(output_folder);

baseline_source = fullfile(project_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'evidence');
antenna_source = fullfile(case_studies_folder, 'Antenna_sweep', 'results', 'ANTENNA_COUNT_SHARED_N_TEST_EVALUATION');
user_source = fullfile(case_studies_folder, 'Users_sweep', 'results', 'USER_COUNT_SHARED_PI_DU_RACE_K_SWEEP_FROM_V4', 'test_evaluation');

baseline_output = fullfile(output_folder, 'N8_K3_SNR');
antenna_output = fullfile(output_folder, 'Antenna_count');
user_output = fullfile(output_folder, 'User_count');
mkdir(baseline_output);
mkdir(antenna_output);
mkdir(user_output);

%% Copy frozen N=8, K=3 V4 test evidence
copy_required(baseline_source, baseline_output, 'N8_fresh_test_V4_five_method_nmse.csv');
copy_required(baseline_source, baseline_output, 'N8_fresh_test_V4_DL_paired_CI.csv');
copy_required(baseline_source, baseline_output, 'N8_fresh_test_V4_five_method_nmse.png');
copy_optional(baseline_source, baseline_output, 'N8_fresh_test_V4_five_method_nmse.fig');

% Copy the complete frozen V4 package: selected models, hard-rank policy,
% exact test inputs, validation-selection evidence, and runnable comparison.
copy_tree(fullfile(project_folder, 'baselines', 'N8_FROZEN_V4_BASELINE'), ...
    fullfile(baseline_output, 'frozen_V4_package'));

%% Assemble antenna-count results: classical methods plus shared-N DL
antenna_table = readtable(fullfile(antenna_source, 'shared_N_test_nmse_by_condition.csv'));
antenna_methods = {'GS','GD','EM-GS','Hard-rank HS-GS','Shared-N DL'};
required_columns = {'N','K','P','SNR_dB','Method','Samples','MeanNMSE_linear','MeanNMSE_dB'};
require_columns(antenna_table, required_columns, 'shared-N per-condition table');
antenna_table = antenna_table(ismember(string(antenna_table.Method), antenna_methods), :);
antenna_table = sortrows(antenna_table, {'P','SNR_dB','N','MethodIndex'});
writetable(antenna_table, fullfile(antenna_output, 'antenna_test_nmse_by_N_SNR_five_methods.csv'));

% Archive the exact shared-N training, validation, and test inputs and model
% provenance next to the report figures. Only training files are copied from
% the original sweep release; fresh validation/test files come from the
% shared-N release.
antenna_data_output = fullfile(antenna_output, 'data');
mkdir(antenna_data_output);
copy_matching_tree(fullfile(case_studies_folder, 'Antenna_sweep', 'data', ...
    'ANTENNA_COUNT_N8_16_24_32_K3_RSR25DB_SNR_m10_0_10_20_30'), ...
    fullfile(antenna_data_output, 'train'), 'train_P*.mat');
copy_matching_tree(fullfile(case_studies_folder, 'Antenna_sweep', 'data', ...
    'ANTENNA_COUNT_SHARED_N_FRESH'), fullfile(antenna_data_output, 'validation_test'), '*.mat');
copy_tree(fullfile(case_studies_folder, 'Antenna_sweep', 'results', ...
    'ANTENNA_COUNT_SHARED_N_PI_DU_RACE_FROM_V4'), ...
    fullfile(antenna_output, 'model_training_validation'));
copy_tree(antenna_source, fullfile(antenna_output, 'test_evaluation_source'));
copy_tree(fullfile(case_studies_folder, 'Antenna_sweep', 'results', ...
    'ANTENNA_COUNT_CLASSICAL', 'validation_policy'), ...
    fullfile(antenna_output, 'hard_rank_validation_policy'));
copy_tree(fullfile(case_studies_folder, 'Antenna_sweep', 'results', ...
    'ANTENNA_COUNT_CLASSICAL', 'convergence_diagnostic_all_ranks'), ...
    fullfile(antenna_output, 'hard_rank_convergence_diagnostic'));
copy_source_scripts(fullfile(case_studies_folder, 'Antenna_sweep'), ...
    fullfile(antenna_output, 'source_scripts'));

antenna_average = average_over_snr(antenna_table, 'N', 'MeanNMSE_linear', 'MeanNMSE_dB');
writetable(antenna_average, fullfile(antenna_output, 'antenna_test_nmse_SNRavg_five_methods.csv'));

P_list = unique(antenna_table.P).';
SNR_grid = unique(antenna_table.SNR_dB).';
for P = P_list
    p_folder = fullfile(antenna_output, sprintf('P%d', P));
    mkdir(p_folder);
    for SNR_dB = SNR_grid
        condition_table = antenna_table(antenna_table.P == P & antenna_table.SNR_dB == SNR_dB, :);
        fig = figure('Visible','off','Color','w');
        plot_methods(condition_table.N, condition_table.MeanNMSE_dB, condition_table.Method, ...
            'Antenna count N', 'Test NMSE (dB)', ...
            sprintf('Antenna sweep, P=%d, SNR=%g dB, K=3, RSR=25 dB', P, SNR_dB));
        saveas(fig, fullfile(p_folder, sprintf('antenna_nmse_vs_N_SNR_%g_dB.png', SNR_dB)));
        close(fig);
    end
    avg_table = antenna_average(antenna_average.P == P, :);
    fig = figure('Visible','off','Color','w');
    plot_methods(avg_table.N, avg_table.MeanNMSE_over_SNR_dB, avg_table.Method, ...
        'Antenna count N', 'Mean NMSE over SNR (dB)', ...
        sprintf('Antenna sweep, SNR-averaged, P=%d, K=3, RSR=25 dB', P));
    saveas(fig, fullfile(p_folder, 'antenna_nmse_vs_N_SNRavg.png'));
    close(fig);
end

%% Assemble user-count results: classical methods plus shared-K DL
user_table = readtable(fullfile(user_source, 'shared_K_DL_test_nmse.csv'));
user_methods = {'GS','GD','EM-GS','Hard-rank HS-GS','Shared-K DL'};
required_columns = {'N','K','P','SNR_dB','Method','Samples','NMSE_linear','NMSE_dB'};
require_columns(user_table, required_columns, 'shared-K per-condition table');
user_table = user_table(ismember(string(user_table.Method), user_methods), :);
user_table = sortrows(user_table, {'P','SNR_dB','K','MethodIndex'});
writetable(user_table, fullfile(user_output, 'user_test_nmse_by_K_SNR_five_methods.csv'));

% Archive the exact user-sweep data, shared-K checkpoints/training history/test
% scores, and validation-frozen hard-rank policy beside the summary results.
copy_tree(fullfile(case_studies_folder, 'Users_sweep', 'data', ...
    'USER_COUNT_RECOMMENDED_P10_K1_2_3_4_5_P30_K1_2_3_4_5_6_7_RSR25_SNR_m10_0_10_20_30'), ...
    fullfile(user_output, 'data', 'train_validation_test'));
copy_tree(fullfile(case_studies_folder, 'Users_sweep', 'results', ...
    'USER_COUNT_SHARED_PI_DU_RACE_K_SWEEP_FROM_V4'), ...
    fullfile(user_output, 'model_training_validation_test'));
copy_tree(fullfile(case_studies_folder, 'Users_sweep', 'results', ...
    'USER_COUNT_RECOMMENDED_SWEEP', 'validation_policy'), ...
    fullfile(user_output, 'hard_rank_validation_policy'));
copy_source_scripts(fullfile(case_studies_folder, 'Users_sweep'), ...
    fullfile(user_output, 'source_scripts'));

user_average = average_over_snr(user_table, 'K', 'NMSE_linear', 'NMSE_dB');
writetable(user_average, fullfile(user_output, 'user_test_nmse_SNRavg_five_methods.csv'));

P_list = unique(user_table.P).';
SNR_grid = unique(user_table.SNR_dB).';
for P = P_list
    p_folder = fullfile(user_output, sprintf('P%d', P));
    mkdir(p_folder);
    for SNR_dB = SNR_grid
        condition_table = user_table(user_table.P == P & user_table.SNR_dB == SNR_dB, :);
        fig = figure('Visible','off','Color','w');
        plot_methods(condition_table.K, condition_table.NMSE_dB, condition_table.Method, ...
            'User count K', 'Test NMSE (dB)', ...
            sprintf('User sweep, P=%d, SNR=%g dB, N=8, RSR=25 dB', P, SNR_dB));
        saveas(fig, fullfile(p_folder, sprintf('user_nmse_vs_K_SNR_%g_dB.png', SNR_dB)));
        close(fig);
    end
    avg_table = user_average(user_average.P == P, :);
    fig = figure('Visible','off','Color','w');
    plot_methods(avg_table.K, avg_table.MeanNMSE_over_SNR_dB, avg_table.Method, ...
        'User count K', 'Mean NMSE over SNR (dB)', ...
        sprintf('User sweep, SNR-averaged, P=%d, N=8, RSR=25 dB', P));
    saveas(fig, fullfile(p_folder, 'user_nmse_vs_K_SNRavg.png'));
    close(fig);
end

fprintf('Final comparison package assembled from existing test summaries.\n');
fprintf('No raw test samples were loaded and no estimator was rerun.\n');
fprintf('Output: %s\n', output_folder);

%% Local helper functions
function average_table = average_over_snr(source_table, x_name, linear_name, dB_name)
    grouping_names = {'P', x_name, 'Method'};
    [group_index, group_table] = findgroups(source_table(:, grouping_names));
    mean_linear = splitapply(@mean, source_table.(linear_name), group_index);
    sample_count = splitapply(@sum, source_table.Samples, group_index);
    average_table = group_table;
    average_table.SamplesAcrossSNR = sample_count;
    average_table.MeanLinearNMSE_over_SNR = mean_linear;
    average_table.MeanNMSE_over_SNR_dB = 10*log10(mean_linear);
    average_table = sortrows(average_table, {'P', x_name, 'Method'});
end

function plot_methods(x_values, y_values, method_values, x_label, y_label, plot_title)
    method_order = {'GS','GD','EM-GS','Hard-rank HS-GS','Shared-N DL','Shared-K DL'};
    colors = lines(numel(method_order));
    hold on;
    handles = gobjects(0);
    labels = {};
    for method_index = 1:numel(method_order)
        selected = string(method_values) == method_order{method_index};
        if any(selected)
            [x_sorted, order] = sort(x_values(selected));
            y_selected = y_values(selected);
            y_sorted = y_selected(order);
            h = plot(x_sorted, y_sorted, '-o', 'LineWidth', 1.5, ...
                'MarkerSize', 6, 'Color', colors(method_index,:));
            handles(end+1,1) = h; %#ok<AGROW>
            labels{end+1,1} = method_order{method_index}; %#ok<AGROW>
        end
    end
    grid on;
    xlabel(x_label);
    ylabel(y_label);
    title(plot_title, 'Interpreter','none');
    legend(handles, labels, 'Location','best');
    set(gca, 'FontSize', 11);
    hold off;
end

function require_columns(input_table, required_names, description)
    if ~all(ismember(required_names, input_table.Properties.VariableNames))
        error('The %s is missing required columns: %s', description, strjoin(required_names, ', '));
    end
end

function copy_required(source_folder, destination_folder, file_name)
    source_file = fullfile(source_folder, file_name);
    if ~isfile(source_file)
        error('Required saved result is missing: %s', source_file);
    end
    copyfile(source_file, fullfile(destination_folder, file_name));
end

function copy_optional(source_folder, destination_folder, file_name)
    source_file = fullfile(source_folder, file_name);
    if isfile(source_file)
        copyfile(source_file, fullfile(destination_folder, file_name));
    end
end

function copy_tree(source_folder, destination_folder)
    if ~isfolder(source_folder)
        error('Required source folder is missing: %s', source_folder);
    end
    if ~isfolder(destination_folder)
        mkdir(destination_folder);
    end
    source_entries = dir(source_folder);
    source_entries = source_entries(~ismember({source_entries.name}, {'.','..'}));
    for entry_index = 1:numel(source_entries)
        source_path = fullfile(source_folder, source_entries(entry_index).name);
        destination_path = fullfile(destination_folder, source_entries(entry_index).name);
        [success, message] = copyfile(source_path, destination_path);
        if ~success
            error('Copy failed: %s', message);
        end
    end
end

function copy_matching_tree(source_folder, destination_folder, file_pattern)
    if ~isfolder(source_folder)
        error('Required data source folder is missing: %s', source_folder);
    end
    matched_files = dir(fullfile(source_folder, '**', file_pattern));
    matched_files = matched_files(~[matched_files.isdir]);
    if isempty(matched_files)
        error('No files matching %s found under %s.', file_pattern, source_folder);
    end
    for file_index = 1:numel(matched_files)
        source_path = fullfile(matched_files(file_index).folder, matched_files(file_index).name);
        relative_folder = erase(matched_files(file_index).folder, [source_folder filesep]);
        destination_subfolder = fullfile(destination_folder, relative_folder);
        if ~isfolder(destination_subfolder)
            mkdir(destination_subfolder);
        end
        [success, message] = copyfile(source_path, fullfile(destination_subfolder, matched_files(file_index).name));
        if ~success
            error('Copy failed: %s', message);
        end
    end
end

function copy_source_scripts(source_folder, destination_folder)
    if ~isfolder(destination_folder)
        mkdir(destination_folder);
    end
    script_files = dir(fullfile(source_folder, '*.m'));
    for script_index = 1:numel(script_files)
        source_path = fullfile(source_folder, script_files(script_index).name);
        [success, message] = copyfile(source_path, fullfile(destination_folder, script_files(script_index).name));
        if ~success
            error('Script copy failed: %s', message);
        end
    end
end
