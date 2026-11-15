%% Frozen shared-K PI-DU-RACE test evaluation
% Evaluates the validation-selected shared-K models once on the reserved
% recommended user-sweep test splits. It evaluates the matching frozen V4
% warm-start model on the same observations and joins the already-frozen
% classical test table for a complete comparison. No model choices are made
% from test results.

clear;
clc;
close all;

%% -------------------- FROZEN EVALUATION CONFIGURATION --------------------
N = 8;
P_list = [10 30];
K_by_P = {[1 2 3 4 5], [1 2 3 4 5 6 7]};
SNR_grid = [-10 0 10 20 30];
RSR_dB = 25;
number_of_stages = 16;
parameters_per_stage = 15;
number_of_gate_features = 3;
number_of_gate_outputs = 3;
number_of_shrinkage_parameters = 3;
initialization_method = 'GS';
maximum_gs_updates = 5000;
gs_convergence_tolerance = 1e-10;
samples_per_SNR = 500;
data_version = 'V3_USER_COUNT_RECOMMENDED_SWEEP_V1';

%% -------------------- PATHS AND OUTPUT SAFETY --------------------
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
repository_folder = fileparts(case_studies_folder);
scenario_name = 'USER_COUNT_RECOMMENDED_P10_K1_2_3_4_5_P30_K1_2_3_4_5_6_7_RSR25_SNR_m10_0_10_20_30';
scenario_folder = fullfile(script_folder, 'data', scenario_name);
shared_model_folder = fullfile(script_folder, 'results', 'USER_COUNT_SHARED_PI_DU_RACE_K_SWEEP_FROM_V4');
v4_model_folder = fullfile(repository_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
classical_results_file = fullfile(script_folder, 'results', 'USER_COUNT_RECOMMENDED_SWEEP', 'test_evaluation', 'recommended_user_sweep_test_nmse.csv');
output_folder = fullfile(shared_model_folder, 'test_evaluation');
checkpoint_folder = fullfile(output_folder, 'checkpoints');
results_mat = fullfile(output_folder, 'shared_K_DL_test_results.mat');

if ~isfolder(scenario_folder)
    error('User-sweep data folder not found: %s', scenario_folder);
end
if ~isfolder(shared_model_folder)
    error('Shared-model folder not found: %s', shared_model_folder);
end
if ~isfolder(v4_model_folder)
    error('Frozen V4 model folder not found: %s', v4_model_folder);
end
if ~isfile(classical_results_file)
    error('Previously scored classical test table not found: %s', classical_results_file);
end
if isfile(results_mat)
    error('Frozen test outputs already exist. Do not rescore or overwrite them: %s', results_mat);
end
if isfolder(output_folder)
    existing_entries = dir(output_folder);
    existing_entries = existing_entries(~ismember({existing_entries.name}, {'.','..'}));
    unexpected_entries = existing_entries(~strcmp({existing_entries.name}, 'checkpoints') | ~[existing_entries.isdir]);
    if ~isempty(unexpected_entries)
        error('Test output folder contains unexpected files. Inspect them before continuing: %s', output_folder);
    end
end

shared_model_by_P = cell(1, numel(P_list));
v4_model_by_P = cell(1, numel(P_list));
for P_index = 1:numel(P_list)
    P = P_list(P_index);
    K_list = K_by_P{P_index};
    shared_model_file = fullfile(shared_model_folder, sprintf('selected_shared_model_P%d.mat', P));
    v4_model_file = fullfile(v4_model_folder, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    if ~isfile(shared_model_file)
        error('Validation-selected shared model not found: %s', shared_model_file);
    end
    if ~isfile(v4_model_file)
        error('Frozen V4 warm-start model not found: %s', v4_model_file);
    end

    shared_model = load(shared_model_file);
    v4_model = load(v4_model_file);
    local_validate_model(shared_model, N, K_list, P, RSR_dB, number_of_stages, parameters_per_stage, number_of_gate_features, number_of_gate_outputs, number_of_shrinkage_parameters, initialization_method, SNR_grid);
    local_validate_v4_model(v4_model, N, P, RSR_dB, number_of_stages, parameters_per_stage, number_of_gate_features, number_of_gate_outputs, initialization_method);
    shared_model_by_P{P_index} = shared_model;
    v4_model_by_P{P_index} = v4_model;
end

classical_table = readtable(classical_results_file);
classical_method_names = {'GS','GD','EM-GS','Hard-rank HS-GS'};
required_classical_columns = {'N','K','P','SNR_dB','Method','NMSE_dB','ConvergedFraction','MeanUpdates'};
if ~all(ismember(required_classical_columns, classical_table.Properties.VariableNames))
    error('Existing classical test table is missing expected columns: %s', classical_results_file);
end
expected_classical_rows = sum(cellfun(@numel, K_by_P))*numel(SNR_grid)*numel(classical_method_names);
if height(classical_table) ~= expected_classical_rows
    error('Expected %d frozen classical test rows, found %d.', expected_classical_rows, height(classical_table));
end
for P_index = 1:numel(P_list)
    P = P_list(P_index);
    for K = K_by_P{P_index}
        for SNR_index = 1:numel(SNR_grid)
            for method_index = 1:numel(classical_method_names)
                matching_row = classical_table.N == N & classical_table.K == K & classical_table.P == P & classical_table.SNR_dB == SNR_grid(SNR_index) & strcmp(string(classical_table.Method), classical_method_names{method_index});
                if nnz(matching_row) ~= 1
                    error('Existing classical table must have exactly one row for N=%d,K=%d,P=%d,SNR=%g,method=%s.', N, K, P, SNR_grid(SNR_index), classical_method_names{method_index});
                end
            end
        end
    end
end

if ~isfolder(output_folder)
    mkdir(output_folder);
end
if ~isfolder(checkpoint_folder)
    mkdir(checkpoint_folder);
end
fprintf('Frozen shared-K DL test evaluation. No model selection or tuning uses test data.\n');
fprintf('N=%d; P=%s; K(P10)=%s; K(P30)=%s; SNR=%s dB; RSR=%g dB.\n', N, mat2str(P_list), mat2str(K_by_P{1}), mat2str(K_by_P{2}), mat2str(SNR_grid), RSR_dB);
fprintf('Scoring validation-selected shared models against matching frozen V4 warm starts.\n');
fprintf('GS initialization cap=%d; convergence tolerance=%.1e; samples/SNR=%d.\n', maximum_gs_updates, gs_convergence_tolerance, samples_per_SNR);
fprintf('Output folder: %s\n', output_folder);

%% -------------------- SCORE THE RESERVED TEST SPLITS --------------------
method_names = {'V4 warm start', 'Shared-K DL', 'GS', 'GD', 'EM-GS', 'Hard-rank HS-GS'};
number_of_result_rows = sum(cellfun(@numel, K_by_P))*numel(SNR_grid)*2;
result_rows = zeros(number_of_result_rows, 9);
row_index = 0;
sample_result_rows = cell(sum(cellfun(@numel, K_by_P)), numel(P_list));
configuration_index = 0;

for P_index = 1:numel(P_list)
    P = P_list(P_index);
    K_list = K_by_P{P_index};
    shared_model = shared_model_by_P{P_index};
    v4_model = v4_model_by_P{P_index};

    for K_index = 1:numel(K_list)
        K = K_list(K_index);
        configuration_index = configuration_index+1;
        dimension_folder = fullfile(scenario_folder, sprintf('N%d_K%d_P%d_RSR25dB', N, K, P));
        test_file = fullfile(dimension_folder, 'test.mat');
        if ~isfile(test_file)
            error('Reserved test split not found: %s', test_file);
        end
        test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version');
        local_validate_test_data(test_data, N, K, P, SNR_grid, RSR_dB, data_version, samples_per_SNR);

        number_of_samples = size(test_data.Y_data, 3);
        sample_snr = double(test_data.SNR_dB_data(:));
        checkpoint_file = fullfile(checkpoint_folder, sprintf('N%d_K%d_P%d_scores.mat', N, K, P));
        if isfile(checkpoint_file)
            checkpoint = load(checkpoint_file, 'N','K','P','sample_snr','shared_error_energy','v4_error_energy','true_energy','gs_converged','gs_updates');
            if checkpoint.N ~= N || checkpoint.K ~= K || checkpoint.P ~= P || ~isequal(checkpoint.sample_snr, sample_snr) || numel(checkpoint.shared_error_energy) ~= number_of_samples || numel(checkpoint.v4_error_energy) ~= number_of_samples || numel(checkpoint.true_energy) ~= number_of_samples || numel(checkpoint.gs_converged) ~= number_of_samples || numel(checkpoint.gs_updates) ~= number_of_samples
                error('Intermediate checkpoint metadata mismatch: %s', checkpoint_file);
            end
            shared_error_energy = checkpoint.shared_error_energy;
            v4_error_energy = checkpoint.v4_error_energy;
            true_energy = checkpoint.true_energy;
            gs_converged = logical(checkpoint.gs_converged);
            gs_updates = checkpoint.gs_updates;
            fprintf('Resuming from completed test scoring: K=%d, P=%d.\n', K, P);
        else
            shared_error_energy = zeros(number_of_samples, 1);
            v4_error_energy = zeros(number_of_samples, 1);
            true_energy = zeros(number_of_samples, 1);
            gs_converged = false(number_of_samples, 1);
            gs_updates = zeros(number_of_samples, 1);

            fprintf('Scoring test samples: K=%d, P=%d, count=%d.\n', K, P, number_of_samples);
            for sample_index = 1:number_of_samples
                Y = double(test_data.Y_data(:, :, sample_index));
                G_true = double(test_data.G_data(:, :, sample_index));
                S = double(test_data.S_data(:, :, sample_index));
                B = double(test_data.B_data(:, :, sample_index));
                sigma2 = double(test_data.sigma2_data(sample_index));

                [G_initial, gs_updates(sample_index), gs_converged(sample_index)] = local_gs_initialization(Y, S, B, maximum_gs_updates, gs_convergence_tolerance);
                G_shared = local_unfolded_forward_numeric(shared_model.best_parameters_numeric, Y, S, B, sigma2, G_initial, shared_model.p_H, N, K, number_of_stages);
                G_v4 = local_unfolded_forward_numeric(v4_model.best_parameters_numeric, Y, S, B, sigma2, G_initial, v4_model.p_H, N, K, number_of_stages);

                shared_error_energy(sample_index) = sum(abs(G_shared-G_true).^2, 'all');
                v4_error_energy(sample_index) = sum(abs(G_v4-G_true).^2, 'all');
                true_energy(sample_index) = sum(abs(G_true).^2, 'all');
                if mod(sample_index, 250) == 0
                    fprintf('  K=%d, P=%d: scored %d/%d samples.\n', K, P, sample_index, number_of_samples);
                end
            end
            save(checkpoint_file, 'N','K','P','sample_snr','shared_error_energy','v4_error_energy','true_energy','gs_converged','gs_updates');
            fprintf('Saved completed test scoring checkpoint: %s\n', checkpoint_file);
        end

        sample_result_rows{configuration_index, P_index} = [repmat([N K P], number_of_samples, 1), sample_snr, (1:number_of_samples).', shared_error_energy, v4_error_energy, true_energy, double(gs_converged), gs_updates];

        for SNR_index = 1:numel(SNR_grid)
            selected_samples = sample_snr == SNR_grid(SNR_index);
            if nnz(selected_samples) ~= samples_per_SNR
                error('Expected %d test samples at K=%d, P=%d, SNR=%g; found %d.', samples_per_SNR, K, P, SNR_grid(SNR_index), nnz(selected_samples));
            end
            denominator = max(sum(true_energy(selected_samples)), realmin);
            shared_nmse_linear = sum(shared_error_energy(selected_samples))/denominator;
            v4_nmse_linear = sum(v4_error_energy(selected_samples))/denominator;
            convergence_fraction = mean(double(gs_converged(selected_samples)));
            mean_gs_updates = mean(gs_updates(selected_samples));

            row_index = row_index+1;
            result_rows(row_index, :) = [N K P SNR_grid(SNR_index) 1 samples_per_SNR v4_nmse_linear convergence_fraction mean_gs_updates];
            row_index = row_index+1;
            result_rows(row_index, :) = [N K P SNR_grid(SNR_index) 2 samples_per_SNR shared_nmse_linear convergence_fraction mean_gs_updates];
        end
    end
end

result_table = array2table(result_rows(1:row_index, :), 'VariableNames', {'N','K','P','SNR_dB','MethodIndex','Samples','NMSE_linear','ConvergedGSFraction','MeanGSUpdates'});
result_table.Method = method_names(result_table.MethodIndex).';
result_table.NMSE_dB = 10*log10(max(result_table.NMSE_linear, realmin));
result_table = movevars(result_table, {'Method','NMSE_dB'}, 'After', 'SNR_dB');

classical_result_rows = zeros(height(classical_table), 9);
for classical_row_index = 1:height(classical_table)
    method_index = find(strcmp(classical_method_names, char(string(classical_table.Method(classical_row_index)))), 1);
    combined_method_index = method_index+2;
    nmse_linear = 10^(double(classical_table.NMSE_dB(classical_row_index))/10);
    classical_result_rows(classical_row_index, :) = [double(classical_table.N(classical_row_index)), double(classical_table.K(classical_row_index)), double(classical_table.P(classical_row_index)), double(classical_table.SNR_dB(classical_row_index)), combined_method_index, samples_per_SNR, nmse_linear, double(classical_table.ConvergedFraction(classical_row_index)), double(classical_table.MeanUpdates(classical_row_index))];
end
classical_result_table = array2table(classical_result_rows, 'VariableNames', result_table.Properties.VariableNames(1:9));
classical_result_table.Properties.VariableNames = {'N','K','P','SNR_dB','MethodIndex','Samples','NMSE_linear','ConvergedGSFraction','MeanGSUpdates'};
classical_result_table.Method = method_names(classical_result_table.MethodIndex).';
classical_result_table.NMSE_dB = double(classical_table.NMSE_dB);
classical_result_table = movevars(classical_result_table, {'Method','NMSE_dB'}, 'After', 'SNR_dB');
result_table = [result_table; classical_result_table];

sample_table = array2table(vertcat(sample_result_rows{:}), 'VariableNames', {'N','K','P','SNR_dB','SampleIndex','SharedErrorEnergy','V4ErrorEnergy','TrueChannelEnergy','GSConverged','GSUpdates'});
results_csv = fullfile(output_folder, 'shared_K_DL_test_nmse.csv');
sample_csv = fullfile(output_folder, 'shared_K_DL_test_sample_errors.csv');
save(results_mat, 'result_table', 'sample_table', 'N', 'K_by_P', 'P_list', 'SNR_grid', 'RSR_dB', 'samples_per_SNR', 'method_names', 'maximum_gs_updates', 'gs_convergence_tolerance', 'data_version');
writetable(result_table, results_csv);
writetable(sample_table, sample_csv);

%% -------------------- SNR-AVERAGED SUMMARY BY PILOT COUNT --------------------
summary_rows = zeros(sum(cellfun(@numel, K_by_P))*numel(method_names), 8);
summary_row_index = 0;
for P_index = 1:numel(P_list)
    P = P_list(P_index);
    K_list = K_by_P{P_index};
    for K_index = 1:numel(K_list)
        K = K_list(K_index);
        for method_index = 1:numel(method_names)
            selected_rows = result_table.K == K & result_table.P == P & result_table.MethodIndex == method_index;
            method_snr_rows = result_table(selected_rows, :);
            if height(method_snr_rows) ~= numel(SNR_grid) || ~all(ismember(SNR_grid, method_snr_rows.SNR_dB))
                error('Missing test SNR result for K=%d, P=%d, method=%s.', K, P, method_names{method_index});
            end
            mean_nmse_linear = mean(method_snr_rows.NMSE_linear);
            summary_row_index = summary_row_index+1;
            summary_rows(summary_row_index, :) = [N K P method_index mean_nmse_linear 10*log10(max(mean_nmse_linear, realmin)) mean(method_snr_rows.ConvergedGSFraction) mean(method_snr_rows.MeanGSUpdates)];
        end
    end
end
summary_table = array2table(summary_rows(1:summary_row_index, :), 'VariableNames', {'N','K','P','MethodIndex','MeanNMSE_linear','MeanNMSE_dB','MeanConvergedGSFraction','MeanGSUpdates'});
summary_table.Method = method_names(summary_table.MethodIndex).';
summary_table = movevars(summary_table, 'Method', 'After', 'P');
summary_csv = fullfile(output_folder, 'shared_K_DL_test_nmse_SNRavg.csv');
writetable(summary_table, summary_csv);

%% -------------------- P-SPECIFIC TEST FIGURES --------------------
colors = lines(numel(method_names));
markers = {'o','s','^','d','v','>'};
for P_index = 1:numel(P_list)
    P = P_list(P_index);
    K_list = K_by_P{P_index};
    figure('Color', 'w', 'Name', sprintf('Shared-K DL test NMSE, P=%d', P));
    tiledlayout(1, numel(SNR_grid), 'TileSpacing', 'compact', 'Padding', 'compact');
    for SNR_index = 1:numel(SNR_grid)
        nexttile;
        hold on;
        for method_index = 1:numel(method_names)
            curve = nan(size(K_list));
            for K_index = 1:numel(K_list)
                selected = result_table.K == K_list(K_index) & result_table.P == P & result_table.SNR_dB == SNR_grid(SNR_index) & result_table.MethodIndex == method_index;
                curve(K_index) = result_table.NMSE_dB(selected);
            end
            plot(K_list, curve, 'Color', colors(method_index, :), 'Marker', markers{method_index}, 'LineWidth', 1.3, 'DisplayName', method_names{method_index});
        end
        grid on;
        xlabel('User count K');
        ylabel('NMSE (dB)');
        title(sprintf('SNR=%g dB', SNR_grid(SNR_index)));
        hold off;
    end
    legend('Location', 'best');
    sgtitle(sprintf('Frozen shared-K DL test NMSE, P=%d', P));
    figure_file = fullfile(output_folder, sprintf('shared_K_DL_test_nmse_P%d.png', P));
    exportgraphics(gcf, figure_file, 'Resolution', 200);

    figure('Color', 'w', 'Name', sprintf('Shared-K DL SNR-averaged test NMSE, P=%d', P));
    hold on;
    for method_index = 1:numel(method_names)
        selected = summary_table.P == P & summary_table.MethodIndex == method_index;
        method_summary = sortrows(summary_table(selected, :), 'K');
        plot(method_summary.K, method_summary.MeanNMSE_dB, 'Color', colors(method_index, :), 'Marker', markers{method_index}, 'LineWidth', 1.5, 'DisplayName', method_names{method_index});
    end
    grid on;
    xlabel('User count K');
    ylabel('Mean NMSE over SNR (dB)');
    title(sprintf('Frozen shared-K DL test average, P=%d', P));
    legend('Location', 'best');
    hold off;
    summary_figure = fullfile(output_folder, sprintf('shared_K_DL_test_nmse_SNRavg_P%d.png', P));
    exportgraphics(gcf, summary_figure, 'Resolution', 200);
end

fprintf('\nFrozen shared-K DL test evaluation complete.\n');
fprintf('Saved per-SNR test NMSE table: %s\n', results_csv);
fprintf('Saved per-sample paired errors: %s\n', sample_csv);
fprintf('Saved SNR-averaged test NMSE table: %s\n', summary_csv);
fprintf('Saved test outputs and figures in: %s\n', output_folder);
fprintf('No test result was used to tune or select a model.\n');

%% -------------------- LOCAL VALIDATION FUNCTIONS --------------------
function local_validate_model(model, N, K_list, P, RSR_dB, number_of_stages, parameters_per_stage, number_of_gate_features, number_of_gate_outputs, number_of_shrinkage_parameters, initialization_method, SNR_grid)
required_fields = {'best_parameters_numeric','N_supported','K_supported','P','RSR_dB','p_H','number_of_stages','parameters_per_stage','number_of_gate_features','number_of_gate_outputs','number_of_shrinkage_parameters','initialization_method','SNR_grid','data_version'};
for field_index = 1:numel(required_fields)
    if ~isfield(model, required_fields{field_index})
        error('Selected shared model is missing required metadata: %s.', required_fields{field_index});
    end
end
expected_p_H = floor((N-1)/2);
if model.N_supported ~= N || ~isequal(double(model.K_supported), double(K_list)) || model.P ~= P || model.RSR_dB ~= RSR_dB || model.p_H ~= expected_p_H || model.number_of_stages ~= number_of_stages || model.parameters_per_stage ~= parameters_per_stage || model.number_of_gate_features ~= number_of_gate_features || model.number_of_gate_outputs ~= number_of_gate_outputs || model.number_of_shrinkage_parameters ~= number_of_shrinkage_parameters || ~strcmp(model.initialization_method, initialization_method) || ~isequal(double(model.SNR_grid), double(SNR_grid)) || ~strcmp(model.data_version, 'V3_USER_COUNT_RECOMMENDED_SWEEP_V1') || ~isequal(size(model.best_parameters_numeric), [parameters_per_stage number_of_stages])
    error('Selected shared model metadata does not match the frozen test-evaluation configuration for P=%d.', P);
end
end

function local_validate_v4_model(model, N, P, RSR_dB, number_of_stages, parameters_per_stage, number_of_gate_features, number_of_gate_outputs, initialization_method)
required_fields = {'best_parameters_numeric','N','K','P','RSR_dB','p_H','number_of_stages','parameters_per_stage','number_of_gate_features','number_of_gate_outputs','initialization_method'};
for field_index = 1:numel(required_fields)
    if ~isfield(model, required_fields{field_index})
        error('Frozen V4 warm-start model is missing required metadata: %s.', required_fields{field_index});
    end
end
expected_p_H = floor((N-1)/2);
if model.N ~= N || model.K ~= 3 || model.P ~= P || model.RSR_dB ~= RSR_dB || model.p_H ~= expected_p_H || model.number_of_stages ~= number_of_stages || model.parameters_per_stage ~= parameters_per_stage || model.number_of_gate_features ~= number_of_gate_features || model.number_of_gate_outputs ~= number_of_gate_outputs || ~strcmp(model.initialization_method, initialization_method) || ~isequal(size(model.best_parameters_numeric), [parameters_per_stage number_of_stages])
    error('Frozen V4 model metadata does not match the expected warm-start configuration for P=%d.', P);
end
end

function local_validate_test_data(data, N, K, P, SNR_grid, RSR_dB, expected_version, samples_per_snr)
if data.N ~= N || data.K ~= K || data.P ~= P || data.RSR_dB ~= RSR_dB || ~isequal(double(data.SNR_dB), double(SNR_grid)) || ~strcmp(data.data_version, expected_version)
    error('Test dataset metadata mismatch for N=%d,K=%d,P=%d.', N, K, P);
end
expected_count = samples_per_snr*numel(SNR_grid);
if data.number_of_samples ~= expected_count || size(data.Y_data, 1) ~= N || size(data.Y_data, 2) ~= P || size(data.Y_data, 3) ~= expected_count || size(data.G_data, 1) ~= N || size(data.G_data, 2) ~= K || size(data.G_data, 3) ~= expected_count || size(data.S_data, 1) ~= K || size(data.S_data, 2) ~= P || size(data.S_data, 3) ~= expected_count || size(data.B_data, 1) ~= N || size(data.B_data, 2) ~= P || size(data.B_data, 3) ~= expected_count || numel(data.sigma2_data) ~= expected_count || numel(data.SNR_dB_data) ~= expected_count
    error('Unexpected test-data array dimensions for N=%d,K=%d,P=%d.', N, K, P);
end
for snr_index = 1:numel(SNR_grid)
    if sum(double(data.SNR_dB_data) == SNR_grid(snr_index)) ~= samples_per_snr
        error('Unexpected number of test samples for K=%d,P=%d,SNR=%g.', K, P, SNR_grid(snr_index));
    end
end
end

function [G, iteration_count, converged] = local_gs_initialization(Y, S, B, maximum_updates, convergence_tolerance)
G = local_spectral_initialize(Y, S, B);
right_inverse = S'/(S*S');
converged = false;
iteration_count = maximum_updates;
for update_index = 1:maximum_updates
    G_previous = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-G_previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
        iteration_count = update_index;
        converged = true;
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

function G_estimate = local_unfolded_forward_numeric(parameters, Y, S, B, sigma2, G_initial, p_H, N, K, number_of_stages)
right_inverse = S'/(S*S');
G_estimate = G_initial;
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
hankel_map = local_hankel_map(N, p_H);
hankel_diagonal_counts = sum(hankel_map, 1).';
for stage_index = 1:number_of_stages
    G_previous = G_estimate;
    field = G_estimate*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    G_hankel = local_hankel_shrink_numeric(G_emgs, parameters(1:3, stage_index), noise_feature, noise_scale, hankel_map, hankel_diagonal_counts, N, K, p_H);
    emgs_change = tanh(log(1+sqrt(sum(abs(G_emgs-G_previous).^2, 'all')/max(sum(abs(G_previous).^2, 'all'), eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_emgs).^2, 'all')/max(sum(abs(G_emgs).^2, 'all'), eps)))/3);
    gate_coefficients = reshape(parameters(4:15, stage_index), 4, 3);
    gate_logits = gate_coefficients(1, :).'*noise_feature+gate_coefficients(2, :).'*emgs_change+gate_coefficients(3, :).'*hankel_change+gate_coefficients(4, :).';
    gate_logits = gate_logits-max(gate_logits);
    gate_weights = exp(gate_logits)./sum(exp(gate_logits));
    G_estimate = gate_weights(1)*G_previous+gate_weights(2)*G_emgs+gate_weights(3)*G_hankel;
end
end

function G_output = local_hankel_shrink_numeric(G_input, shrinkage_parameters, noise_feature, noise_scale, hankel_map, hankel_diagonal_counts, N, K, p_H)
G_output = zeros(N, K);
for user_index = 1:K
    hankel_matrix = reshape(hankel_map*G_input(:, user_index), N-p_H, p_H+1);
    gram_matrix = hankel_matrix'*hankel_matrix;
    [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
    eigenvalues = real(sum(eigenvalue_matrix, 2));
    singular_values = sqrt(max(eigenvalues, 0));
    safe_singular_values = max(singular_values, 1e-12);
    mean_singular_value = max(mean(safe_singular_values), 1e-12);
    relative_spectrum_feature = tanh(log(safe_singular_values./mean_singular_value));
    threshold_logits = shrinkage_parameters(1)+shrinkage_parameters(2)*noise_feature+shrinkage_parameters(3)*relative_spectrum_feature;
    mode_thresholds = noise_scale*(2./(1+exp(-threshold_logits)));
    shrink_factors = max(1-mode_thresholds./safe_singular_values, 0);
    scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
    hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
    G_output(:, user_index) = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
end
end

function hankel_map = local_hankel_map(N, p_H)
number_of_rows = N-p_H;
number_of_columns = p_H+1;
hankel_map = zeros(number_of_rows*number_of_columns, N);
for row_index = 1:number_of_rows
    for column_index = 1:number_of_columns
        linear_index = sub2ind([number_of_rows number_of_columns], row_index, column_index);
        antenna_index = row_index+column_index-1;
        hankel_map(linear_index, antenna_index) = 1;
    end
end
end
