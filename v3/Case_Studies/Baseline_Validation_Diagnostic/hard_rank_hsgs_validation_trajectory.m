%% Validation-only trajectory diagnostic for hard-rank HS-GS
% Diagnoses the P=10 anomaly near SNR=20/25 dB using validation samples only.
% The validation-frozen rank policy and the estimator implementation are
% read-only. No test file is loaded and no rank/threshold is selected here.
%
% For each sample, the script records the deployed hard-rank HS-GS estimate
% at selected update checkpoints, its magnitude-data residual, and the exact
% relative-update stopping statistic. A same-sample GS score is also recorded.

clear;
clc;
close all;

%% Frozen diagnostic settings
N = 8;
K = 3;
P = 10;
RSR_dB = 25;
p_H = 3;
SNR_diagnostic = [15 20 25 30];
samples_per_SNR = 12;
maximum_updates = 20000;
convergence_tolerance = 1e-10;
checkpoint_updates = [1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000];
selection_exclusion_per_snr = 64;
sample_selection_seed = 2026111515;
maximum_parallel_workers = 6;

%% Paths and no-overwrite guard
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
project_root = fileparts(case_studies_folder);
validation_file = fullfile(project_root, 'data', ...
    'N8_K3_RSR25dB_AoA_Uminus90to90', 'validation_P10.mat');
policy_file = fullfile(project_root, 'baselines', 'N8_FROZEN_V4_BASELINE', ...
    'policy', 'N8_hard_rank_validation_policy.csv');
output_folder = fullfile(script_folder, 'results', ...
    'HARD_RANK_HSGS_TRAJECTORY_VALIDATION_V1');

if isfolder(output_folder) || isfile(output_folder)
    error('Output folder already exists. Inspect it and set a new folder name: %s', output_folder);
end
if ~isfile(validation_file) || ~isfile(policy_file)
    error('Required validation data or frozen rank policy is missing.');
end

validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', ...
    'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', ...
    'N', 'K', 'P', 'RSR_dB', 'data_version', 'random_seed', ...
    'samples_this_split_per_snr', 'split_name');
policy = readtable(policy_file);

if validation_data.N ~= N || validation_data.K ~= K || ...
        validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ...
        ~strcmp(validation_data.split_name, 'validation')
    error('Validation split metadata does not match N=8, K=3, P=10, RSR=25 dB.');
end
if ~all(ismember({'P','SNR_dB','FrozenHardRank'}, policy.Properties.VariableNames))
    error('Frozen hard-rank policy does not contain the expected columns.');
end

%% Exclude V4 checkpoint-selection validation samples
all_snr = unique(double(validation_data.SNR_dB_data(:))).';
excluded_indices = local_v4_reserved_validation_indices( ...
    validation_data.SNR_dB_data, all_snr, selection_exclusion_per_snr);
rng(sample_selection_seed, 'twister');
selected_indices = zeros(numel(SNR_diagnostic)*samples_per_SNR, 1);
selected_snr = zeros(size(selected_indices));
write_index = 0;
for snr_index = 1:numel(SNR_diagnostic)
    current_snr = SNR_diagnostic(snr_index);
    candidates = find(double(validation_data.SNR_dB_data(:)) == current_snr);
    candidates(ismember(candidates, excluded_indices)) = [];
    if numel(candidates) < samples_per_SNR
        error('Not enough disjoint validation samples at SNR=%g dB.', current_snr);
    end
    order = randperm(numel(candidates), samples_per_SNR);
    chosen = candidates(order);
    selected = write_index + (1:samples_per_SNR);
    selected_indices(selected) = chosen;
    selected_snr(selected) = current_snr;
    write_index = write_index + samples_per_SNR;
end
if numel(unique(selected_indices)) ~= numel(selected_indices) || ...
        any(ismember(selected_indices, excluded_indices))
    error('Diagnostic validation sample indices overlap or repeat.');
end

fprintf('Validation-only hard-rank HS-GS trajectory diagnostic.\n');
fprintf('P=%d; SNR=%s dB; %d samples/condition; cap=%d; tolerance=%.1e.\n', ...
    P, mat2str(SNR_diagnostic), samples_per_SNR, maximum_updates, convergence_tolerance);
fprintf('Using the frozen rank policy; %d checkpoint-selection samples/SNR are excluded.\n', ...
    selection_exclusion_per_snr);
fprintf('No test files are loaded. Output: %s\n', output_folder);

%% Per-sample trajectory computation
number_of_samples = numel(selected_indices);
number_of_checkpoints = numel(checkpoint_updates);
trajectory_nmse_db = nan(number_of_samples, number_of_checkpoints);
trajectory_magnitude_residual = nan(number_of_samples, number_of_checkpoints);
trajectory_relative_update = nan(number_of_samples, number_of_checkpoints);
final_hard_nmse_db = nan(number_of_samples, 1);
gs_nmse_db = nan(number_of_samples, 1);
gs_error_energy = nan(number_of_samples, 1);
hard_error_energy = nan(number_of_samples, 1);
channel_energy_by_sample = nan(number_of_samples, 1);
hard_updates = zeros(number_of_samples, 1);
hard_converged = false(number_of_samples, 1);
gs_updates = zeros(number_of_samples, 1);
gs_converged = false(number_of_samples, 1);
frozen_rank = zeros(number_of_samples, 1);
for position = 1:number_of_samples
    policy_rows = find(policy.P == P & policy.SNR_dB == selected_snr(position));
    if numel(policy_rows) ~= 1
        error('Expected one frozen rank for P=%d, SNR=%g.', P, selected_snr(position));
    end
    frozen_rank(position) = policy.FrozenHardRank(policy_rows);
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('Processes', maximum_parallel_workers);
end

trajectory_actual_update = zeros(number_of_samples, number_of_checkpoints);
parfor (position = 1:number_of_samples, maximum_parallel_workers)
    sample_index = selected_indices(position);
    Y = double(validation_data.Y_data(:, :, sample_index));
    G_true = double(validation_data.G_data(:, :, sample_index));
    S = double(validation_data.S_data(:, :, sample_index));
    B = double(validation_data.B_data(:, :, sample_index));
    sigma2 = double(validation_data.sigma2_data(sample_index));

    [G_hard, trace] = local_hard_hsgs_trajectory(Y, G_true, S, B, sigma2, p_H, ...
        frozen_rank(position), maximum_updates, convergence_tolerance, checkpoint_updates);
    [G_gs, gs_update_count, gs_did_converge] = ...
        local_classical_gs(Y, S, B, maximum_updates, convergence_tolerance);
    gs_updates(position) = gs_update_count;
    gs_converged(position) = gs_did_converge;
    channel_energy = max(sum(abs(G_true).^2, 'all'), eps);
    gs_error = sum(abs(G_gs-G_true).^2, 'all');
    hard_error_energy(position) = sum(abs(G_hard-G_true).^2, 'all');
    gs_error_energy(position) = gs_error;
    channel_energy_by_sample(position) = channel_energy;
    gs_nmse_db(position) = 10*log10(max(gs_error/channel_energy, realmin));
    final_hard_nmse_db(position) = 10*log10(max(hard_error_energy(position)/channel_energy, realmin));
    hard_updates(position) = trace.iteration_count;
    hard_converged(position) = trace.converged;
    trajectory_nmse_db(position, :) = 10*log10(max(trace.error_energy(:).', realmin));
    trajectory_magnitude_residual(position, :) = trace.magnitude_residual(:).';
    trajectory_relative_update(position, :) = trace.relative_update(:).';
    trajectory_actual_update(position, :) = trace.actual_update(:).';
    fprintf('Finished diagnostic sample %d/%d (SNR=%g dB).\n', ...
        position, number_of_samples, selected_snr(position));
end

%% Save validation-only outputs
mkdir(output_folder);
sample_table = table(repmat(P, number_of_samples, 1), selected_snr, selected_indices, ...
    frozen_rank, gs_nmse_db, final_hard_nmse_db, final_hard_nmse_db-gs_nmse_db, ...
    gs_updates, gs_converged, hard_updates, hard_converged, ...
    'VariableNames', {'P','SNR_dB','ValidationIndex','FrozenHardRank', ...
    'GS_NMSE_dB','HardRank_HSGS_NMSE_dB','HardMinusGS_dB', ...
    'GS_Updates','GS_Converged','HardRank_Updates','HardRank_Converged'});
writetable(sample_table, fullfile(output_folder, 'validation_sample_summary.csv'));

trajectory_rows = cell(number_of_samples*number_of_checkpoints, 1);
row_index = 0;
for position = 1:number_of_samples
    for checkpoint_index = 1:number_of_checkpoints
        row_index = row_index + 1;
        trajectory_rows{row_index} = [P selected_snr(position) selected_indices(position) ...
            frozen_rank(position) checkpoint_updates(checkpoint_index) ...
            trajectory_actual_update(position, checkpoint_index) ...
            trajectory_nmse_db(position, checkpoint_index) ...
            trajectory_magnitude_residual(position, checkpoint_index) ...
            trajectory_relative_update(position, checkpoint_index)];
    end
end
trajectory_matrix = vertcat(trajectory_rows{:});
trajectory_table = array2table(trajectory_matrix, 'VariableNames', ...
    {'P','SNR_dB','ValidationIndex','FrozenHardRank','RequestedUpdate','ActualUpdate', ...
    'HardRank_NMSE_dB','NormalizedMagnitudeResidual','RelativeUpdateSquared'});
writetable(trajectory_table, fullfile(output_folder, 'validation_trajectory_checkpoints.csv'));

condition_rows = zeros(numel(SNR_diagnostic), 7);
for snr_index = 1:numel(SNR_diagnostic)
    selected = selected_snr == SNR_diagnostic(snr_index);
    pooled_gs_nmse = sum(gs_error_energy(selected))/sum(channel_energy_by_sample(selected));
    pooled_hard_nmse = sum(hard_error_energy(selected))/sum(channel_energy_by_sample(selected));
    condition_rows(snr_index, :) = [P SNR_diagnostic(snr_index) nnz(selected) ...
        10*log10(max(pooled_gs_nmse, realmin)) 10*log10(max(pooled_hard_nmse, realmin)) ...
        10*log10(max(pooled_hard_nmse, realmin))-10*log10(max(pooled_gs_nmse, realmin)) ...
        sum(~hard_converged(selected))];
end
condition_table = array2table(condition_rows, 'VariableNames', ...
    {'P','SNR_dB','SampleCount','Pooled_GS_NMSE_dB','Pooled_HardRank_HSGS_NMSE_dB', ...
    'Pooled_HardMinusGS_dB','HardRank_NonconvergedCount'});
writetable(condition_table, fullfile(output_folder, 'validation_condition_summary.csv'));

data_version = validation_data.data_version;
save(fullfile(output_folder, 'hard_rank_hsgs_validation_trajectory.mat'), ...
    'sample_table','trajectory_table','condition_table','selected_indices', ...
    'selected_snr','frozen_rank','checkpoint_updates','maximum_updates', ...
    'convergence_tolerance','selection_exclusion_per_snr','sample_selection_seed', ...
    'validation_file','policy_file','data_version','-v7.3');

fprintf('Saved sample summary, trajectory checkpoints, and condition summary.\n');
fprintf('Validation-only diagnostic complete; no test data were loaded or scored.\n');
fprintf('Results: %s\n', output_folder);

%% Local helpers copied from the frozen estimator protocol
function reserved_indices = local_v4_reserved_validation_indices(SNR_dB_data, SNR_grid, samples_per_snr)
reserved_indices = [];
for snr_index = 1:numel(SNR_grid)
    candidates = find(double(SNR_dB_data(:)) == SNR_grid(snr_index));
    if numel(candidates) <= 2*samples_per_snr
        error('Validation split is too small to exclude checkpoint-selection indices.');
    end
    excluded_positions = round(linspace(1, numel(candidates), samples_per_snr));
    fresh_candidates = candidates;
    fresh_candidates(excluded_positions) = [];
    fresh_positions = round(linspace(1, numel(fresh_candidates), samples_per_snr));
    reserved_indices = [reserved_indices; candidates(excluded_positions(:)); ...
        fresh_candidates(fresh_positions(:))]; %#ok<AGROW>
end
end

function [G, trace] = local_hard_hsgs_trajectory(Y, G_true, S, B, sigma2, p_H, ...
        rank_value, maximum_updates, tolerance, checkpoint_updates)
N = size(Y, 1);
K = size(S, 1);
right_inverse = S'/(S*S');
G = local_spectral_initialize(Y, S, B);
number_of_checkpoints = numel(checkpoint_updates);
trace.error_energy = zeros(number_of_checkpoints, 1);
trace.magnitude_residual = zeros(number_of_checkpoints, 1);
trace.relative_update = zeros(number_of_checkpoints, 1);
trace.actual_update = zeros(number_of_checkpoints, 1);
trace.converged = false;
trace.iteration_count = maximum_updates;
checkpoint_index = 1;
channel_energy = max(sum(abs(G_true).^2, 'all'), eps);
for iteration = 1:maximum_updates
    previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    projected = zeros(N, K);
    rows = N-p_H;
    columns = p_H+1;
    for user_index = 1:K
        channel = emgs(:, user_index);
        hankel = zeros(rows, columns);
        for row = 1:rows
            hankel(row, :) = channel(row:row+columns-1).';
        end
        [U, Sigma, V] = svd(hankel, 'econ');
        use_rank = min([rank_value, size(Sigma,1), size(Sigma,2)]);
        hankel = U(:,1:use_rank)*Sigma(1:use_rank,1:use_rank)*V(:,1:use_rank)';
        reconstructed = zeros(N, 1);
        counts = zeros(N, 1);
        for row = 1:rows
            for column = 1:columns
                antenna = row+column-1;
                reconstructed(antenna) = reconstructed(antenna)+hankel(row,column);
                counts(antenna) = counts(antenna)+1;
            end
        end
        projected(:, user_index) = reconstructed./counts;
    end
    G = projected;
    relative_update = norm(G-previous, 'fro')^2/max(norm(G, 'fro')^2, eps);
    if checkpoint_index <= number_of_checkpoints && iteration == checkpoint_updates(checkpoint_index)
        trace.error_energy(checkpoint_index) = sum(abs(G-G_true).^2, 'all');
        trace.magnitude_residual(checkpoint_index) = ...
            sum((Y-abs(G*S+B)).^2, 'all')/max(sum(Y.^2, 'all'), eps);
        trace.relative_update(checkpoint_index) = relative_update;
        trace.actual_update(checkpoint_index) = iteration;
        checkpoint_index = checkpoint_index+1;
    end
    if relative_update < tolerance
        trace.converged = true;
        trace.iteration_count = iteration;
        while checkpoint_index <= number_of_checkpoints
            trace.error_energy(checkpoint_index) = sum(abs(G-G_true).^2, 'all');
            trace.magnitude_residual(checkpoint_index) = ...
                sum((Y-abs(G*S+B)).^2, 'all')/max(sum(Y.^2, 'all'), eps);
            trace.relative_update(checkpoint_index) = relative_update;
            trace.actual_update(checkpoint_index) = iteration;
            checkpoint_index = checkpoint_index+1;
        end
        break;
    end
end
if checkpoint_index <= number_of_checkpoints
    error('Estimator stopped before all requested checkpoints were recorded.');
end
trace.error_energy = trace.error_energy/channel_energy;
end

function G = local_spectral_initialize(Y, S, B)
N = size(Y, 1);
K = size(S, 1);
G = zeros(N, K);
for antenna = 1:N
    y = Y(antenna, :).';
    b = B(antenna, :).';
    augmented = [S.' b];
    matrix = augmented'*diag(y)*augmented;
    [vectors, values] = eig(matrix);
    [~, index] = max(real(diag(values)));
    vector = vectors(:, index);
    vector = vector/max(norm(vector), eps);
    scale = (abs(augmented*vector).'*y)/max(norm(augmented*vector)^2, eps);
    G(antenna, :) = (scale*exp(-1i*angle(vector(K+1)))*vector(1:K)).';
end
end

function [G, iteration_count, converged] = local_classical_gs(Y, S, B, maximum_updates, tolerance)
N = size(Y, 1);
K = size(S, 1);
G = local_spectral_initialize(Y, S, B);
right_inverse = S'/(S*S');
converged = false;
iteration_count = maximum_updates;
for iteration = 1:maximum_updates
    previous = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < tolerance
        iteration_count = iteration;
        converged = true;
        break;
    end
end
end
