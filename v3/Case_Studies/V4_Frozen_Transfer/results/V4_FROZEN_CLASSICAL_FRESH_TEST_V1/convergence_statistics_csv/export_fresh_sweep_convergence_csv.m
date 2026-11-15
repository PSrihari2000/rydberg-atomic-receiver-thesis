%% Export saved fresh-sweep convergence fields to CSV
% Reads only fresh_test_scores.mat. Does not load raw data, rerun estimators,
% tune policies, or alter existing score/figure files.
% Classical methods have convergence flags and update counts. PI-DU-RACE is
% a fixed 16-stage network, so iterative convergence is not applicable.

clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
% This exporter is stored beside its CSV outputs under the result release.
results_root = fileparts(script_folder);
score_file = fullfile(results_root, 'fresh_test_scores.mat');
output_folder = script_folder;

if ~isfile(score_file)
    error('Saved sweep score MAT file was not found: %s', score_file);
end
if isfolder(output_folder)
    existing_outputs = dir(fullfile(output_folder, '*'));
    existing_outputs = existing_outputs(~ismember({existing_outputs.name}, {'.','..'}));
    if ~isempty(existing_outputs)
        error('Convergence CSV output folder already has files; inspect it or choose a new folder: %s', output_folder);
    end
end
if ~isfolder(output_folder)
    mkdir(output_folder);
end

loaded = load(score_file, 'score_cells', 'cell_metadata', 'SNR_grid');
score_cells = loaded.score_cells;
cell_metadata = loaded.cell_metadata;
SNR_grid = loaded.SNR_grid(:).';

if numel(score_cells) ~= size(cell_metadata,1)
    error('score_cells and cell_metadata have inconsistent lengths.');
end

classical_names = {'GS','GD','EM-GS','Hard-rank HS-GS'};
% Wide per-sample rows keep the four iterative-method records together.
sample_rows = cell(0, 16);
summary_rows = cell(0, 16);

for cell_index = 1:numel(score_cells)
    item = score_cells{cell_index};
    metadata = cell_metadata(cell_index,:);
    if metadata(1) == 1
        study_name = 'Antenna_sweep';
    elseif metadata(1) == 2
        study_name = 'User_sweep';
    else
        error('Unexpected study index in cell_metadata at row %d.', cell_index);
    end

    if item.N ~= metadata(2) || item.K ~= metadata(3) || item.P ~= metadata(4)
        error('Cell metadata does not match score_cells entry %d.', cell_index);
    end
    if ~isfield(item,'sample_converged') || ~isfield(item,'sample_updates') || ~isfield(item,'sample_snr')
        error('Convergence fields are missing from score_cells entry %d.', cell_index);
    end

    converged = logical(item.sample_converged);
    updates = double(item.sample_updates);
    sample_snr = double(item.sample_snr(:));
    sample_count = numel(sample_snr);
    if ~isequal(size(converged),[sample_count 4]) || ~isequal(size(updates),[sample_count 4])
        error('Unexpected convergence array dimensions in score_cells entry %d.', cell_index);
    end
    if any(~ismember(sample_snr,SNR_grid)) || any(~isfinite(updates),'all') || any(updates < 1,'all')
        error('Invalid sample SNR or iteration count in score_cells entry %d.', cell_index);
    end

    for sample_index = 1:sample_count
        sample_rows(end+1,:) = [{study_name,item.N,item.K,item.P,sample_snr(sample_index),sample_index}, ...
            num2cell(double(converged(sample_index,:))), num2cell(updates(sample_index,:)), ...
            {false,16}];
    end

    for snr_index = 1:numel(SNR_grid)
        selected = sample_snr == SNR_grid(snr_index);
        if ~any(selected)
            error('No samples found for %s N=%d K=%d P=%d SNR=%g dB.', ...
                study_name,item.N,item.K,item.P,SNR_grid(snr_index));
        end
        for method_index = 1:4
            method_converged = converged(selected,method_index);
            method_updates = updates(selected,method_index);
            summary_rows(end+1,:) = {study_name,item.N,item.K,item.P,SNR_grid(snr_index), ...
                classical_names{method_index},nnz(selected),true,nnz(method_converged), ...
                100*mean(method_converged),nnz(~method_converged),mean(method_updates), ...
                median(method_updates),max(method_updates),NaN,NaN};
        end
        summary_rows(end+1,:) = {study_name,item.N,item.K,item.P,SNR_grid(snr_index), ...
            'PI-DU-RACE (DL)',nnz(selected),false,NaN,NaN,NaN,NaN,NaN,NaN,16, ...
            'Fixed-stage network; iterative convergence is not applicable'};
    end
end

sample_table = cell2table(sample_rows,'VariableNames', ...
    {'Study','N','K','P','SNR_dB','SampleIndex', ...
    'GS_Converged','GD_Converged','EMGS_Converged','HardRankHSGS_Converged', ...
    'GS_Updates','GD_Updates','EMGS_Updates','HardRankHSGS_Updates', ...
    'DL_ConvergenceApplicable','DL_FixedStages'});

summary_table = cell2table(summary_rows,'VariableNames', ...
    {'Study','N','K','P','SNR_dB','Method','SampleCount','ConvergenceApplicable', ...
    'ConvergedCount','ConvergenceRate_percent','NotConvergedCount', ...
    'MeanUpdates','MedianUpdates','MaximumUpdates','DL_FixedStages','Note'});

sample_file = fullfile(output_folder,'fresh_sweep_convergence_by_sample.csv');
summary_file = fullfile(output_folder,'fresh_sweep_convergence_summary_by_condition.csv');
writetable(sample_table,sample_file);
writetable(summary_table,summary_file);

fprintf('Saved per-sample convergence CSV: %s\n',sample_file);
fprintf('Saved per-condition convergence summary CSV: %s\n',summary_file);
fprintf('Export only: loaded saved score MAT; no raw test files or estimators were used.\n');
