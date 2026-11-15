%% Validation-only matched comparison: frozen V4 PI-DU-RACE vs soft-rank HS-GS
% Uses the current 240-parameter P-specific V4 checkpoints without modifying them.
% Loads only the original N=8, K=3, RSR=25 dB validation files.
% It reconstructs the deterministic V4 checkpoint-selection indices, tunes a
% fixed soft-threshold multiplier on disjoint validation samples, then scores
% both methods on another disjoint validation subset. No test files are loaded.

clear;
clc;
close all;

%% Fixed protocol
N = 8;
K = 3;
P_list = [10 30];
RSR_dB = 25;
p_H = 3;
number_of_stages = 16;
parameters_per_stage = 15;
maximum_gs_updates = 5000;
gs_convergence_tolerance = 1e-10;
validation_samples_per_snr_for_checkpoint_selection = 64;
validation_partition_seed = 2026111201;
bootstrap_seed = 2026111202;
bootstrap_replicates = 5000;
threshold_multipliers = 0:0.1:2;

%% Paths and overwrite protection
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
project_root = fileparts(case_studies_folder);
scenario_folder = fullfile(project_root, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
model_root = fullfile(project_root, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
output_folder = fullfile(script_folder, 'results', 'PI_DU_RACE_VS_SOFT_HSGS_VALIDATION_V1');

if isfolder(output_folder) || isfile(output_folder)
    error('Output path already exists. Inspect it and set a new output-folder name before rerunning: %s', output_folder);
end
for P = P_list
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    model_file = fullfile(model_root, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    if ~isfile(validation_file) || ~isfile(model_file)
        error('Missing validation data or frozen V4 model for P=%d.', P);
    end
end
mkdir(output_folder);

fprintf('Validation-only comparison: current V4 vs fixed-threshold soft-rank HS-GS.\n');
fprintf('N=%d, K=%d, RSR=%g dB; P=[%s]; both methods use the same GS start and %d updates.\n', ...
    N, K, RSR_dB, num2str(P_list), number_of_stages);
fprintf('Soft threshold is selected on validation tuning samples only.\n');
fprintf('No test file is loaded; V4 checkpoints are read-only.\n');

%% Result storage
condition_rows = zeros(numel(P_list)*8, 9);
runtime_rows = zeros(numel(P_list), 13);
sample_rows = cell(numel(P_list), 1);
policy_rows = zeros(numel(P_list)*8, 3);
condition_row = 0;
policy_row = 0;

%% Evaluate both P-specific models independently
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    model_file = fullfile(model_root, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));

    % Explicit variable list: no test data or unrelated files are loaded.
    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', ...
        'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', ...
        'RSR_dB', 'AoA_deg_range', 'data_version', 'random_seed', ...
        'samples_this_split_per_snr', 'split_name');
    model = load(model_file, 'best_parameters_numeric', 'best_epoch_saved', ...
        'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', ...
        'parameters_per_stage', 'initialization_method', ...
        'maximum_gs_updates', 'gs_convergence_tolerance');

    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || ...
            validation_data.RSR_dB ~= RSR_dB || ~strcmp(validation_data.split_name, 'validation')
        error('Validation metadata mismatch for P=%d.', P);
    end
    if ~isequal(size(model.best_parameters_numeric), [parameters_per_stage number_of_stages]) || ...
            model.N ~= N || model.K ~= K || model.P ~= P || model.RSR_dB ~= RSR_dB || ...
            model.p_H ~= p_H || model.number_of_stages ~= number_of_stages || ...
            model.parameters_per_stage ~= parameters_per_stage || ~strcmp(model.initialization_method, 'GS')
        error('Frozen V4 checkpoint metadata mismatch for P=%d.', P);
    end
    if isfield(model, 'maximum_gs_updates') && model.maximum_gs_updates ~= maximum_gs_updates
        error('V4 GS update cap mismatch for P=%d.', P);
    end
    if isfield(model, 'gs_convergence_tolerance') && model.gs_convergence_tolerance ~= gs_convergence_tolerance
        error('V4 GS tolerance mismatch for P=%d.', P);
    end

    SNR_grid = unique(double(validation_data.SNR_dB_data(:))).';
    expected_snr_grid = -5:5:30;
    if ~isequal(SNR_grid, expected_snr_grid)
        error('Unexpected validation SNR grid for P=%d.', P);
    end

    % V4 trainers selected 64 deterministic validation samples per SNR using
    % this source-defined rule. Keep those samples out of this comparison.
    checkpoint_selection_indices = local_v4_reserved_validation_indices( ...
        validation_data.SNR_dB_data, SNR_grid, validation_samples_per_snr_for_checkpoint_selection);
    rng(validation_partition_seed + P, 'twister');
    tuning_indices = [];
    diagnostic_indices = [];
    for snr_index = 1:numel(SNR_grid)
        condition_candidates = find(double(validation_data.SNR_dB_data(:)) == SNR_grid(snr_index));
        condition_candidates(ismember(condition_candidates, checkpoint_selection_indices)) = [];
        if numel(condition_candidates) < 4
            error('Too few non-selection validation samples for P=%d, SNR=%g.', P, SNR_grid(snr_index));
        end
        shuffled = condition_candidates(randperm(numel(condition_candidates)));
        midpoint = floor(numel(shuffled)/2);
        tuning_indices = [tuning_indices; shuffled(1:midpoint)]; %#ok<AGROW>
        diagnostic_indices = [diagnostic_indices; shuffled(midpoint+1:end)]; %#ok<AGROW>
    end
    if ~isempty(intersect(checkpoint_selection_indices, tuning_indices)) || ...
            ~isempty(intersect(checkpoint_selection_indices, diagnostic_indices)) || ...
            ~isempty(intersect(tuning_indices, diagnostic_indices))
        error('Validation partitions overlap for P=%d.', P);
    end

    fprintf('\nP=%d: validation=%d; V4 selection excluded=%d; threshold tuning=%d; paired diagnostic=%d.\n', ...
        P, validation_data.number_of_samples, numel(checkpoint_selection_indices), ...
        numel(tuning_indices), numel(diagnostic_indices));

    %% Tune one fixed soft threshold per P/SNR on the tuning subset
    selected_multiplier = zeros(1, numel(SNR_grid));
    for snr_index = 1:numel(SNR_grid)
        current_snr = SNR_grid(snr_index);
        sample_indices = tuning_indices(double(validation_data.SNR_dB_data(tuning_indices)) == current_snr);
        candidate_error = zeros(numel(threshold_multipliers), 1);
        candidate_channel_energy = 0;
        for sample_position = 1:numel(sample_indices)
            sample_index = sample_indices(sample_position);
            [Y, G_true, S, B, sigma2] = local_read_sample(validation_data, sample_index);
            G_initial = local_biased_gs(Y, S, B, maximum_gs_updates, gs_convergence_tolerance);
            channel_energy = sum(abs(G_true).^2, 'all');
            candidate_channel_energy = candidate_channel_energy + channel_energy;
            for candidate_index = 1:numel(threshold_multipliers)
                estimate = local_soft_hsgs_16(G_initial, Y, S, B, sigma2, ...
                    threshold_multipliers(candidate_index), p_H, number_of_stages);
                candidate_error(candidate_index) = candidate_error(candidate_index) + ...
                    sum(abs(estimate-G_true).^2, 'all');
            end
        end
        candidate_nmse = candidate_error/max(candidate_channel_energy, eps);
        [~, selected_index] = min(candidate_nmse);
        selected_multiplier(snr_index) = threshold_multipliers(selected_index);
        policy_row = policy_row + 1;
        policy_rows(policy_row, :) = [P current_snr selected_multiplier(snr_index)];
        fprintf('P=%d SNR=%g dB: validation-selected soft multiplier=%.3f.\n', ...
            P, current_snr, selected_multiplier(snr_index));
    end

    %% Warm up numerical paths, then paired validation diagnostic and timing
    warm_sample = diagnostic_indices(1);
    [Yw, ~, Sw, Bw, sigma2w] = local_read_sample(validation_data, warm_sample);
    G0w = local_biased_gs(Yw, Sw, Bw, maximum_gs_updates, gs_convergence_tolerance);
    local_v4_forward(model.best_parameters_numeric, G0w, Yw, Sw, Bw, sigma2w, p_H, N, K);
    local_soft_hsgs_16(G0w, Yw, Sw, Bw, sigma2w, selected_multiplier(1), p_H, number_of_stages);

    diagnostic_count = numel(diagnostic_indices);
    dl_error = zeros(diagnostic_count, 1);
    soft_error = zeros(diagnostic_count, 1);
    channel_energy = zeros(diagnostic_count, 1);
    gs_seconds = zeros(diagnostic_count, 1);
    gs_iterations = zeros(diagnostic_count, 1);
    gs_converged = false(diagnostic_count, 1);
    dl_forward_seconds = zeros(diagnostic_count, 1);
    soft_forward_seconds = zeros(diagnostic_count, 1);
    diagnostic_snr = zeros(diagnostic_count, 1);
    for position = 1:diagnostic_count
        sample_index = diagnostic_indices(position);
        [Y, G_true, S, B, sigma2] = local_read_sample(validation_data, sample_index);
        diagnostic_snr(position) = double(validation_data.SNR_dB_data(sample_index));

        timer = tic;
        [G_initial, gs_iterations(position), gs_converged(position)] = ...
            local_biased_gs(Y, S, B, maximum_gs_updates, gs_convergence_tolerance);
        gs_seconds(position) = toc(timer);

        multiplier_index = find(SNR_grid == diagnostic_snr(position), 1);
        if mod(position, 2) == 1
            timer = tic;
            G_dl = local_v4_forward(model.best_parameters_numeric, G_initial, Y, S, B, sigma2, p_H, N, K);
            dl_forward_seconds(position) = toc(timer);
            timer = tic;
            G_soft = local_soft_hsgs_16(G_initial, Y, S, B, sigma2, ...
                selected_multiplier(multiplier_index), p_H, number_of_stages);
            soft_forward_seconds(position) = toc(timer);
        else
            timer = tic;
            G_soft = local_soft_hsgs_16(G_initial, Y, S, B, sigma2, ...
                selected_multiplier(multiplier_index), p_H, number_of_stages);
            soft_forward_seconds(position) = toc(timer);
            timer = tic;
            G_dl = local_v4_forward(model.best_parameters_numeric, G_initial, Y, S, B, sigma2, p_H, N, K);
            dl_forward_seconds(position) = toc(timer);
        end
        dl_error(position) = sum(abs(G_dl-G_true).^2, 'all');
        soft_error(position) = sum(abs(G_soft-G_true).^2, 'all');
        channel_energy(position) = sum(abs(G_true).^2, 'all');

        if mod(position, 100) == 0 || position == diagnostic_count
            fprintf('P=%d paired validation diagnostics: %d/%d.\n', P, position, diagnostic_count);
        end
    end
    if any(~isfinite(dl_error)) || any(~isfinite(soft_error)) || any(channel_energy <= 0)
        error('Invalid validation result for P=%d.', P);
    end

    %% Per-SNR paired NMSE and paired bootstrap confidence intervals
    for snr_index = 1:numel(SNR_grid)
        current_snr = SNR_grid(snr_index);
        selected = diagnostic_snr == current_snr;
        if nnz(selected) < 20
            error('Insufficient paired diagnostic samples for P=%d, SNR=%g.', P, current_snr);
        end
        dl_nmse_db = 10*log10(sum(dl_error(selected))/sum(channel_energy(selected)));
        soft_nmse_db = 10*log10(sum(soft_error(selected))/sum(channel_energy(selected)));
        point_gain = soft_nmse_db-dl_nmse_db;
        [gain_low, gain_high] = local_paired_bootstrap_ci( ...
            dl_error(selected), soft_error(selected), channel_energy(selected), ...
            bootstrap_replicates, bootstrap_seed + 100*P + snr_index);
        condition_row = condition_row + 1;
        condition_rows(condition_row, :) = [P current_snr nnz(selected) ...
            selected_multiplier(snr_index) dl_nmse_db soft_nmse_db point_gain gain_low gain_high];
        fprintf('P=%d SNR=%g: DL %.3f dB, soft-HS-GS %.3f dB, soft-minus-DL %.3f dB [%.3f, %.3f].\n', ...
            P, current_snr, dl_nmse_db, soft_nmse_db, point_gain, gain_low, gain_high);
    end

    %% Runtime totals include one shared GS initialization for each method
    gs_mean = mean(gs_seconds);
    dl_forward_mean = mean(dl_forward_seconds);
    soft_forward_mean = mean(soft_forward_seconds);
    runtime_rows(p_index, :) = [P diagnostic_count gs_mean dl_forward_mean ...
        soft_forward_mean gs_mean+dl_forward_mean gs_mean+soft_forward_mean ...
        (gs_mean+dl_forward_mean)/(gs_mean+soft_forward_mean) gs_convergence_tolerance ...
        mean(gs_iterations) mean(gs_converged) ...
        sum(gs_seconds)+sum(dl_forward_seconds) sum(gs_seconds)+sum(soft_forward_seconds)];
    fprintf('P=%d mean/sample seconds: GS %.6g; DL total %.6g; soft-HS-GS total %.6g.\n', ...
        P, gs_mean, gs_mean+dl_forward_mean, gs_mean+soft_forward_mean);

    sample_table = table(repmat(P, diagnostic_count, 1), diagnostic_snr, ...
        diagnostic_indices(:), dl_error./channel_energy, soft_error./channel_energy, ...
        gs_seconds, gs_iterations, gs_converged, dl_forward_seconds, soft_forward_seconds, ...
        'VariableNames', {'P','SNR_dB','ValidationIndex','DL_NMSE_linear', ...
        'SoftHSGS_NMSE_linear','GS_seconds','GS_iterations','GS_converged', ...
        'DL_forward_seconds','SoftHSGS_forward_seconds'});
    sample_rows{p_index} = sample_table;
end

%% Save validation-only outputs
condition_table = array2table(condition_rows, 'VariableNames', {'P','SNR_dB', ...
    'DiagnosticSamples','SoftThresholdMultiplier','V4_NMSE_dB','SoftHSGS_NMSE_dB', ...
    'SoftMinusV4_Gain_dB','GainCI95_Low_dB','GainCI95_High_dB'});
runtime_table = array2table(runtime_rows, 'VariableNames', {'P','DiagnosticSamples', ...
    'GS_seconds_per_sample','V4_forward_seconds_per_sample', ...
    'SoftHSGS_forward_seconds_per_sample','V4_total_seconds_per_sample', ...
    'SoftHSGS_total_seconds_per_sample','V4_to_SoftRuntimeRatio','GS_tolerance', ...
    'GS_iterations_mean','GS_converged_fraction','V4_total_seconds_for_diagnostic_set', ...
    'SoftHSGS_total_seconds_for_diagnostic_set'});
policy_table = array2table(policy_rows, 'VariableNames', {'P','SNR_dB','FrozenSoftThresholdMultiplier'});
sample_table = vertcat(sample_rows{:});
writetable(condition_table, fullfile(output_folder, 'validation_nmse_and_paired_CI_by_condition.csv'));
writetable(runtime_table, fullfile(output_folder, 'validation_runtime_including_GS.csv'));
writetable(policy_table, fullfile(output_folder, 'validation_selected_soft_threshold_policy.csv'));
writetable(sample_table, fullfile(output_folder, 'validation_paired_sample_diagnostics.csv'));
save(fullfile(output_folder, 'validation_comparison.mat'), 'condition_table', ...
    'runtime_table', 'policy_table', 'sample_table', 'threshold_multipliers', ...
    'P_list', 'SNR_grid', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages', ...
    'maximum_gs_updates', 'gs_convergence_tolerance', 'bootstrap_replicates', ...
    'validation_partition_seed', 'bootstrap_seed');

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    selected_rows = condition_table.P == P;
    figure('Color', 'w');
    plot(condition_table.SNR_dB(selected_rows), condition_table.V4_NMSE_dB(selected_rows), '-o', ...
        'LineWidth', 1.6, 'DisplayName', 'PI-DU-RACE V4');
    hold on;
    plot(condition_table.SNR_dB(selected_rows), condition_table.SoftHSGS_NMSE_dB(selected_rows), '-s', ...
        'LineWidth', 1.6, 'DisplayName', '16-update soft-rank HS-GS');
    grid on;
    xlabel('SNR (dB)');
    ylabel('Validation NMSE (dB)');
    title(sprintf('Validation comparison, P=%d; N=%d, K=%d, RSR=%g dB', P, N, K, RSR_dB));
    legend('Location', 'best');
    saveas(gcf, fullfile(output_folder, sprintf('validation_nmse_P%d.png', P)));
end

fprintf('\nValidation-only comparison complete. No test file was loaded.\n');
fprintf('Results: %s\n', output_folder);

%% Local functions
function reserved_indices = local_v4_reserved_validation_indices(SNR_dB_data, SNR_grid, samples_per_snr)
reserved_indices = [];
for snr_index = 1:numel(SNR_grid)
    candidates = find(double(SNR_dB_data(:)) == SNR_grid(snr_index));
    if numel(candidates) <= 2*samples_per_snr
        error('Validation split too small to reproduce V4 checkpoint-selection indices.');
    end
    excluded_positions = round(linspace(1, numel(candidates), samples_per_snr));
    fresh_candidates = candidates;
    fresh_candidates(excluded_positions) = [];
    fresh_positions = round(linspace(1, numel(fresh_candidates), samples_per_snr));
    % Reserve both the prior warm-start selection and current V4 selection.
    reserved_indices = [reserved_indices; candidates(excluded_positions(:)); ...
        fresh_candidates(fresh_positions(:))]; %#ok<AGROW>
end
end

function [Y, G_true, S, B, sigma2] = local_read_sample(data, index)
Y = double(data.Y_data(:, :, index));
G_true = double(data.G_data(:, :, index));
S = double(data.S_data(:, :, index));
B = double(data.B_data(:, :, index));
sigma2 = double(data.sigma2_data(index));
end

function [G, iteration_count, converged] = local_biased_gs(Y, S, B, maximum_updates, tolerance)
N = size(Y, 1);
K = size(S, 1);
G = zeros(N, K);
for antenna_index = 1:N
    y = Y(antenna_index, :).';
    b = B(antenna_index, :).';
    augmented = [S.' b];
    spectral_matrix = augmented'*diag(y)*augmented;
    [vectors, values] = eig(spectral_matrix);
    [~, principal_index] = max(real(diag(values)));
    principal = vectors(:, principal_index);
    principal = principal/max(norm(principal), eps);
    scale = (abs(augmented*principal).'*y)/max(norm(augmented*principal)^2, eps);
    G(antenna_index, :) = (scale*exp(-1i*angle(principal(K+1)))*principal(1:K)).';
end
right_inverse = S'/(S*S');
iteration_count = maximum_updates;
converged = false;
for update_index = 1:maximum_updates
    previous = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

function G = local_emgs_update(G, Y, S, B, sigma2)
right_inverse = S'/(S*S');
field = G*S+B;
kappa = 2/max(sigma2, eps)*Y.*abs(field);
ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
G = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
end

function G = local_soft_hsgs_16(G, Y, S, B, sigma2, multiplier, p_H, stages)
N = size(Y, 1);
K = size(S, 1);
right_inverse = S'/(S*S');
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
threshold = multiplier*noise_scale;
for stage = 1:stages
    G_em = local_emgs_update(G, Y, S, B, sigma2);
    G = local_soft_hankel_projection(G_em, p_H, threshold);
end
end

function G = local_soft_hankel_projection(G, p_H, threshold)
N = size(G, 1);
K = size(G, 2);
rows = N-p_H;
columns = p_H+1;
for user = 1:K
    vector = G(:, user);
    hankel = zeros(rows, columns);
    for row = 1:rows
        hankel(row, :) = vector(row:row+columns-1).';
    end
    [U, Sigma, V] = svd(hankel, 'econ');
    shrunk = max(diag(Sigma)-threshold, 0);
    hankel_shrunk = U*diag(shrunk)*V';
    reconstructed = zeros(N, 1);
    counts = zeros(N, 1);
    for row = 1:rows
        for column = 1:columns
            antenna = row+column-1;
            reconstructed(antenna) = reconstructed(antenna)+hankel_shrunk(row, column);
            counts(antenna) = counts(antenna)+1;
        end
    end
    G(:, user) = reconstructed./counts;
end
end

function G = local_v4_forward(parameters, G, Y, S, B, sigma2, p_H, N, K)
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
right_inverse = S'/(S*S');
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
for stage = 1:16
    previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G_em = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    G_hankel = zeros(N, K);
    for user = 1:K
        vector = G_em(:, user);
        hankel = zeros(N-p_H, p_H+1);
        for row = 1:N-p_H
            hankel(row, :) = vector(row:row+p_H).';
        end
        gram = hankel'*hankel;
        [eigenvectors, eigenvalue_matrix] = eig(gram);
        singular_values = sqrt(max(real(sum(eigenvalue_matrix, 2)), 0));
        safe_values = max(singular_values, 1e-12);
        mean_value = max(mean(safe_values), 1e-12);
        relative_feature = tanh(log(safe_values./mean_value));
        logits = parameters(1, stage)+parameters(2, stage)*noise_feature+ ...
            parameters(3, stage)*relative_feature;
        thresholds = noise_scale*(2./(1+exp(-logits)));
        shrink = max(1-thresholds./safe_values, 0);
        hankel_shrunk = hankel*(eigenvectors.*reshape(shrink, 1, []))*eigenvectors';
        reconstructed = zeros(N, 1);
        counts = zeros(N, 1);
        for row = 1:N-p_H
            for column = 1:p_H+1
                antenna = row+column-1;
                reconstructed(antenna) = reconstructed(antenna)+hankel_shrunk(row, column);
                counts(antenna) = counts(antenna)+1;
            end
        end
        G_hankel(:, user) = reconstructed./counts;
    end
    em_change = tanh(log(1+sqrt(sum(abs(G_em-previous).^2, 'all')/max(sum(abs(previous).^2, 'all'), eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_em).^2, 'all')/max(sum(abs(G_em).^2, 'all'), eps)))/3);
    gate = reshape(parameters(4:15, stage), 4, 3);
    gate_logits = gate(1, :).'*noise_feature+gate(2, :).'*em_change+ ...
        gate(3, :).'*hankel_change+gate(4, :).';
    gate_logits = gate_logits-max(gate_logits);
    weights = exp(gate_logits)./sum(exp(gate_logits));
    G = weights(1)*previous+weights(2)*G_em+weights(3)*G_hankel;
end
end

function [ci_low, ci_high] = local_paired_bootstrap_ci(dl_error, soft_error, energy, replicates, seed)
rng(seed, 'twister');
n = numel(energy);
gain = zeros(replicates, 1);
for replicate = 1:replicates
    selected = randi(n, n, 1);
    dl_nmse = sum(dl_error(selected))/sum(energy(selected));
    soft_nmse = sum(soft_error(selected))/sum(energy(selected));
    gain(replicate) = 10*log10(soft_nmse)-10*log10(dl_nmse);
end
ordered_gain = sort(gain);
ci_low = ordered_gain(max(1, ceil(0.025*replicates)));
ci_high = ordered_gain(min(replicates, ceil(0.975*replicates)));
end
