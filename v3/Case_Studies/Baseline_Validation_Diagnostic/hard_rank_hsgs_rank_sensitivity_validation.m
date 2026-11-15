%% Validation-only hard-rank HS-GS rank-sensitivity diagnostic
% Evaluates ranks 1:4 on fresh validation indices at SNR=20 and 25 dB.
% The frozen V4 model and hard-rank policy are read-only. This script does not
% load test data, select a replacement rank, or alter any project checkpoint.

clear;
clc;
close all;

%% Study settings
N = 8;
K = 3;
P = 10;
RSR_dB = 25;
p_H = 3;
SNR_diagnostic = [20 25];
rank_candidates = 1:4;
samples_per_SNR = 12;
maximum_updates = 20000;
convergence_tolerance = 1e-10;
selection_exclusion_per_snr = 64;
sample_selection_seed = 2026111516;
maximum_parallel_workers = 6;

%% Paths and safety checks
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
project_root = fileparts(case_studies_folder);
validation_file = fullfile(project_root, 'data', ...
    'N8_K3_RSR25dB_AoA_Uminus90to90', 'validation_P10.mat');
policy_file = fullfile(project_root, 'baselines', 'N8_FROZEN_V4_BASELINE', ...
    'policy', 'N8_hard_rank_validation_policy.csv');
prior_diagnostic_file = fullfile(script_folder, 'results', ...
    'HARD_RANK_HSGS_TRAJECTORY_VALIDATION_V1', ...
    'hard_rank_hsgs_validation_trajectory.mat');
output_folder = fullfile(script_folder, 'results', ...
    'HARD_RANK_HSGS_RANK_SENSITIVITY_VALIDATION_V1');

sample_csv = fullfile(output_folder, 'validation_rank_sensitivity_samples.csv');
partial_output_exists = false;
if isfolder(output_folder)
    existing_entries = dir(output_folder);
    existing_names = {existing_entries(~[existing_entries.isdir]).name};
    if isequal(existing_names, {'validation_rank_sensitivity_samples.csv'}) && isfile(sample_csv)
        partial_table = readtable(sample_csv);
        expected_columns = {'N','K','P','RSR_dB','SNR_dB','ValidationIndex', ...
            'CandidateRank','FrozenPolicyRank','NMSE_dB','NormalizedMagnitudeResidual', ...
            'UpdateCount','Converged','RuntimeSeconds'};
        if isequal(partial_table.Properties.VariableNames, expected_columns) && ...
                height(partial_table) == numel(SNR_diagnostic)*samples_per_SNR*numel(rank_candidates)
            partial_output_exists = true;
            fprintf('Found only the expected partial sample CSV. It will be checked against a deterministic rerun before completion.\n');
        else
            error('Existing output folder is not the expected partial run; inspect it before proceeding: %s', output_folder);
        end
    else
        error('Output folder already contains unexpected files. Inspect it before proceeding: %s', output_folder);
    end
elseif isfile(output_folder)
    error('Output path exists as a file. Inspect it before proceeding: %s', output_folder);
end
if ~isfile(validation_file) || ~isfile(policy_file) || ~isfile(prior_diagnostic_file)
    error('Required validation data, frozen policy, or prior trajectory diagnostic is missing.');
end

validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', ...
    'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', ...
    'N', 'K', 'P', 'RSR_dB', 'data_version', 'random_seed', 'split_name');
policy = readtable(policy_file);
prior_diagnostic = load(prior_diagnostic_file, 'selected_indices');

if validation_data.N ~= N || validation_data.K ~= K || ...
        validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ...
        ~strcmp(validation_data.split_name, 'validation')
    error('Validation metadata does not match N=8, K=3, P=10, RSR=25 dB.');
end
if ~all(ismember({'P','SNR_dB','FrozenHardRank'}, policy.Properties.VariableNames))
    error('Frozen policy lacks P, SNR_dB, or FrozenHardRank.');
end
if isempty(prior_diagnostic.selected_indices)
    error('Prior trajectory diagnostic contains no selected indices to exclude.');
end

%% Keep checkpoint-selection and prior diagnostic samples out of this audit
all_snr = unique(double(validation_data.SNR_dB_data(:))).';
checkpoint_selection_indices = local_v4_reserved_validation_indices( ...
    validation_data.SNR_dB_data, all_snr, selection_exclusion_per_snr);
prior_indices = double(prior_diagnostic.selected_indices(:));
if any(prior_indices < 1) || any(prior_indices > validation_data.number_of_samples) || ...
        any(prior_indices ~= floor(prior_indices))
    error('Prior diagnostic indices do not fit this validation split.');
end
excluded_indices = unique([checkpoint_selection_indices(:); prior_indices]);

rng(sample_selection_seed, 'twister');
number_of_samples = numel(SNR_diagnostic)*samples_per_SNR;
selected_indices = zeros(number_of_samples, 1);
selected_snr = zeros(number_of_samples, 1);
write_index = 0;
for snr_index = 1:numel(SNR_diagnostic)
    current_snr = SNR_diagnostic(snr_index);
    candidates = find(double(validation_data.SNR_dB_data(:)) == current_snr);
    candidates(ismember(candidates, excluded_indices)) = [];
    if numel(candidates) < samples_per_SNR
        error('Not enough unused validation samples at SNR=%g dB.', current_snr);
    end
    order = randperm(numel(candidates), samples_per_SNR);
    selected = write_index + (1:samples_per_SNR);
    selected_indices(selected) = candidates(order);
    selected_snr(selected) = current_snr;
    write_index = write_index + samples_per_SNR;
end
if numel(unique(selected_indices)) ~= number_of_samples || ...
        any(ismember(selected_indices, excluded_indices))
    error('Selected indices repeat or overlap with excluded validation samples.');
end

frozen_rank_by_sample = zeros(number_of_samples, 1);
for position = 1:number_of_samples
    policy_rows = find(policy.P == P & policy.SNR_dB == selected_snr(position));
    if numel(policy_rows) ~= 1
        error('Expected one frozen policy rank for P=%d, SNR=%g dB.', ...
            P, selected_snr(position));
    end
    frozen_rank_by_sample(position) = policy.FrozenHardRank(policy_rows);
end

fprintf('Fresh-index validation-only hard-rank sensitivity diagnostic.\n');
fprintf('N=%d, K=%d, P=%d, RSR=%g dB; SNR=%s dB; ranks=%s.\n', ...
    N, K, P, RSR_dB, mat2str(SNR_diagnostic), mat2str(rank_candidates));
fprintf('%d samples/SNR; %d total samples; max updates=%d; tolerance=%.1e.\n', ...
    samples_per_SNR, number_of_samples, maximum_updates, convergence_tolerance);
fprintf('Excludes V4 selection indices and all %d prior trajectory indices.\n', ...
    numel(prior_indices));
fprintf('All rank candidates use the same spectral initialization per sample.\n');
fprintf('No test files are loaded. Output: %s\n', output_folder);

%% Run all candidate ranks on the same validation observations
number_of_ranks = numel(rank_candidates);
nmse_db = nan(number_of_samples, number_of_ranks);
normalized_magnitude_residual = nan(number_of_samples, number_of_ranks);
update_count = zeros(number_of_samples, number_of_ranks);
converged = false(number_of_samples, number_of_ranks);
elapsed_seconds = nan(number_of_samples, number_of_ranks);

pool = gcp('nocreate');
if isempty(pool)
    parpool('Processes', maximum_parallel_workers);
end

parfor (position = 1:number_of_samples, maximum_parallel_workers)
    sample_index = selected_indices(position);
    Y = double(validation_data.Y_data(:, :, sample_index));
    G_true = double(validation_data.G_data(:, :, sample_index));
    S = double(validation_data.S_data(:, :, sample_index));
    B = double(validation_data.B_data(:, :, sample_index));
    sigma2 = double(validation_data.sigma2_data(sample_index));
    initial_G = local_spectral_initialize(Y, S, B);
    channel_energy = max(sum(abs(G_true).^2, 'all'), eps);
    local_nmse_db = nan(1, number_of_ranks);
    local_residual = nan(1, number_of_ranks);
    local_updates = zeros(1, number_of_ranks);
    local_converged = false(1, number_of_ranks);
    local_seconds = nan(1, number_of_ranks);

    for rank_index = 1:number_of_ranks
        timer = tic;
        [estimate, iterations, did_converge] = local_hard_hsgs( ...
            Y, S, B, sigma2, p_H, rank_candidates(rank_index), ...
            maximum_updates, convergence_tolerance, initial_G);
        local_seconds(rank_index) = toc(timer);
        error_energy = sum(abs(estimate-G_true).^2, 'all');
        local_nmse_db(rank_index) = 10*log10(max(error_energy/channel_energy, realmin));
        residual_energy = sum((Y-abs(estimate*S+B)).^2, 'all');
        local_residual(rank_index) = residual_energy/max(sum(Y.^2, 'all'), eps);
        local_updates(rank_index) = iterations;
        local_converged(rank_index) = did_converge;
    end

    nmse_db(position, :) = local_nmse_db;
    normalized_magnitude_residual(position, :) = local_residual;
    update_count(position, :) = local_updates;
    converged(position, :) = local_converged;
    elapsed_seconds(position, :) = local_seconds;
end

%% Save per-sample and condition summaries
resume_partial_output = false;
if isfolder(output_folder)
    existing_entries = dir(output_folder);
    existing_names = {existing_entries(~[existing_entries.isdir]).name};
    if isequal(existing_names, {'validation_rank_sensitivity_samples.csv'}) && isfile(sample_csv)
        existing_sample_table = readtable(sample_csv);
        existing_comparable = table2array(existing_sample_table(:, 1:12));
        current_comparable = table2array(sample_table(:, 1:12));
        comparison_tolerance = 1e-10*max(1, abs(current_comparable));
        matching_scores = all(abs(existing_comparable-current_comparable) <= ...
            comparison_tolerance, 'all');
        if ~partial_output_exists || ...
                ~isequal(existing_sample_table.Properties.VariableNames, ...
                sample_table.Properties.VariableNames) || ~matching_scores
            error('Existing partial sample CSV differs from current workspace results; inspect it manually.');
        end
        resume_partial_output = true;
        fprintf('Verified partial sample results. Refreshing runtime values and saving the remaining outputs.\n');
    else
        error('Output folder is not empty in the expected partial-run form; inspect it: %s', output_folder);
    end
elseif isfile(output_folder)
    error('Output path exists as a file; inspect it: %s', output_folder);
else
    [created, message] = mkdir(output_folder);
    if ~created
        error('Could not create output folder: %s (%s)', output_folder, message);
    end
end
sample_rows = cell(number_of_samples*number_of_ranks, 1);
row_index = 0;
for position = 1:number_of_samples
    for rank_index = 1:number_of_ranks
        row_index = row_index+1;
        sample_rows{row_index} = [N K P RSR_dB selected_snr(position) ...
            selected_indices(position) rank_candidates(rank_index) ...
            frozen_rank_by_sample(position) nmse_db(position, rank_index) ...
            normalized_magnitude_residual(position, rank_index) ...
            update_count(position, rank_index) converged(position, rank_index) ...
            elapsed_seconds(position, rank_index)];
    end
end
sample_table = array2table(vertcat(sample_rows{:}), 'VariableNames', ...
    {'N','K','P','RSR_dB','SNR_dB','ValidationIndex','CandidateRank', ...
    'FrozenPolicyRank','NMSE_dB','NormalizedMagnitudeResidual', ...
    'UpdateCount','Converged','RuntimeSeconds'});
if resume_partial_output
    writetable(sample_table, sample_csv);
else
    writetable(sample_table, sample_csv);
end

summary_rows = zeros(numel(SNR_diagnostic)*number_of_ranks, 11);
summary_row = 0;
for snr_index = 1:numel(SNR_diagnostic)
    selected = selected_snr == SNR_diagnostic(snr_index);
    channel_energies = zeros(nnz(selected), 1);
    selected_positions = find(selected);
    for local_position = 1:numel(selected_positions)
        source_index = selected_indices(selected_positions(local_position));
        G_true = double(validation_data.G_data(:, :, source_index));
        channel_energies(local_position) = max(sum(abs(G_true).^2, 'all'), eps);
    end
    for rank_index = 1:number_of_ranks
        summary_row = summary_row+1;
        pooled_linear_nmse = sum(10.^(nmse_db(selected, rank_index)/10).*channel_energies) / ...
            sum(channel_energies);
        frozen_rank = policy.FrozenHardRank( ...
            find(policy.P == P & policy.SNR_dB == SNR_diagnostic(snr_index), 1));
        summary_rows(summary_row, :) = [N K P RSR_dB SNR_diagnostic(snr_index) ...
            rank_candidates(rank_index) frozen_rank nnz(selected) ...
            10*log10(max(pooled_linear_nmse, realmin)) ...
            mean(update_count(selected, rank_index)) ...
            nnz(~converged(selected, rank_index))];
    end
end
summary_table = array2table(summary_rows, 'VariableNames', ...
    {'N','K','P','RSR_dB','SNR_dB','CandidateRank','FrozenPolicyRank', ...
    'SampleCount','PooledNMSE_dB','MeanUpdates','NonconvergedCount'});
writetable(summary_table, fullfile(output_folder, 'validation_rank_sensitivity_summary.csv'));

data_version = validation_data.data_version;
save(fullfile(output_folder, 'hard_rank_hsgs_rank_sensitivity_validation.mat'), ...
    'sample_table','summary_table','selected_indices','selected_snr', ...
    'rank_candidates','frozen_rank_by_sample','nmse_db', ...
    'normalized_magnitude_residual','update_count','converged','elapsed_seconds', ...
    'maximum_updates','convergence_tolerance','selection_exclusion_per_snr', ...
    'sample_selection_seed','checkpoint_selection_indices','prior_indices', ...
    'validation_file','policy_file','prior_diagnostic_file','data_version','-v7.3');

fprintf('Saved per-sample rank results and pooled condition summary.\n');
fprintf('Validation-only sensitivity diagnostic complete; no test data were loaded.\n');
fprintf('Frozen policy was not changed and no alternative rank was selected. Results: %s\n', ...
    output_folder);

%% Local helpers: same validation split reservation and estimator convention
function reserved_indices = local_v4_reserved_validation_indices(SNR_dB_data, SNR_grid, samples_per_snr)
reserved_indices = [];
for snr_index = 1:numel(SNR_grid)
    candidates = find(double(SNR_dB_data(:)) == SNR_grid(snr_index));
    if numel(candidates) <= 2*samples_per_snr
        error('Validation split is too small to reproduce checkpoint-selection indices.');
    end
    excluded_positions = round(linspace(1, numel(candidates), samples_per_snr));
    fresh_candidates = candidates;
    fresh_candidates(excluded_positions) = [];
    fresh_positions = round(linspace(1, numel(fresh_candidates), samples_per_snr));
    reserved_indices = [reserved_indices; candidates(excluded_positions(:)); ...
        fresh_candidates(fresh_positions(:))]; %#ok<AGROW>
end
end

function G = local_spectral_initialize(Y, S, B)
N_local = size(Y, 1);
K_local = size(S, 1);
G = zeros(N_local, K_local);
for antenna = 1:N_local
    y = Y(antenna, :).';
    b = B(antenna, :).';
    augmented = [S.' b];
    matrix = augmented'*diag(y)*augmented;
    [vectors, values] = eig(matrix);
    [~, index] = max(real(diag(values)));
    vector = vectors(:, index);
    vector = vector/max(norm(vector), eps);
    scale = (abs(augmented*vector).'*y)/max(norm(augmented*vector)^2, eps);
    G(antenna, :) = (scale*exp(-1i*angle(vector(K_local+1)))*vector(1:K_local)).';
end
end

function [G, iteration_count, did_converge] = local_hard_hsgs( ...
        Y, S, B, sigma2, p_H_local, rank_value, maximum_updates_local, tolerance, initial_G)
N_local = size(Y, 1);
K_local = size(S, 1);
right_inverse = S'/(S*S');
G = initial_G;
did_converge = false;
iteration_count = maximum_updates_local;
for iteration = 1:maximum_updates_local
    previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    projected = zeros(N_local, K_local);
    hankel_rows = N_local-p_H_local;
    hankel_columns = p_H_local+1;
    for user_index = 1:K_local
        channel = emgs(:, user_index);
        hankel = zeros(hankel_rows, hankel_columns);
        for hankel_row = 1:hankel_rows
            hankel(hankel_row, :) = channel(hankel_row:hankel_row+hankel_columns-1).';
        end
        [U, Sigma, V] = svd(hankel, 'econ');
        use_rank = min([rank_value, size(Sigma, 1), size(Sigma, 2)]);
        hankel = U(:, 1:use_rank)*Sigma(1:use_rank, 1:use_rank)*V(:, 1:use_rank)';
        reconstructed = zeros(N_local, 1);
        counts = zeros(N_local, 1);
        for hankel_row = 1:hankel_rows
            for hankel_column = 1:hankel_columns
                antenna = hankel_row+hankel_column-1;
                reconstructed(antenna) = reconstructed(antenna)+hankel(hankel_row,hankel_column);
                counts(antenna) = counts(antenna)+1;
            end
        end
        projected(:, user_index) = reconstructed./counts;
    end
    G = projected;
    relative_update = norm(G-previous, 'fro')^2/max(norm(G, 'fro')^2, eps);
    if relative_update < tolerance
        did_converge = true;
        iteration_count = iteration;
        break;
    end
end
end
