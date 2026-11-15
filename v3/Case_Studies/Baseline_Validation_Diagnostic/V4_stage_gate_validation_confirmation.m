%% V4 stage and branch confirmation using validation data only
% This script confirms the frozen V4 PI-DU-RACE model on additional V5
% validation samples. It deliberately does not load any test file.
%
% Per SNR, it excludes the 32 reserved indices, the 32 indices used by V5
% screening checkpoint selection, and the 16 indices used by the earlier
% stage diagnostic. It then evaluates 64 additional matched samples.
%
% Outputs include paired sample NMSE, condition-level NMSE, stagewise
% branch NMSE, and mean gate weights. The script does not train or modify a
% model and does not select a new model.

clear;
clc;
close all;

%% -------------------- FIXED CONFIRMATION SETTINGS --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
number_of_stages = 16;
parameters_per_stage = 15;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
reserved_validation_samples_per_SNR = 32;
screening_validation_samples_per_SNR = 32;
previous_diagnostic_samples_per_SNR = 16;
confirmation_samples_per_SNR = 64;

%% -------------------- INPUT AND OUTPUT PATHS --------------------
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
v3_folder = fileparts(case_studies_folder);
validation_folder = fullfile(case_studies_folder, 'Baseline_V5', 'data', 'N8_K3_RSR25dB');
model_folder = fullfile(v3_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
policy_file = fullfile(v3_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'policy', 'N8_hard_rank_validation_policy.csv');
output_folder = fullfile(script_folder, 'results', 'V4_STAGE_GATE_VALIDATION_CONFIRMATION');
previous_diagnostic_file = fullfile(script_folder, 'results', 'V4_STAGE_GATE_VALIDATION', 'V4_validation_paired_sample_diagnostic.csv');

if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it or change the output-folder name before running.');
end
if ~isfolder(validation_folder) || ~isfolder(model_folder) || ~isfile(policy_file) || ~isfile(previous_diagnostic_file)
    error('Required validation data, frozen V4 models, hard-rank policy, or prior diagnostic sample list is missing.');
end

previous_diagnostic_table = readtable(previous_diagnostic_file);
required_previous_columns = {'P','SNR_dB','ValidationSampleIndex'};
if ~all(ismember(required_previous_columns, previous_diagnostic_table.Properties.VariableNames))
    error('The prior diagnostic sample table is missing required index columns.');
end

policy_table = readtable(policy_file);
required_policy_columns = {'P', 'SNR_dB', 'FrozenHardRank'};
if ~all(ismember(required_policy_columns, policy_table.Properties.VariableNames))
    error('The frozen hard-rank policy does not contain the required columns.');
end

fprintf('V4 stage and branch confirmation uses validation data only.\n');
fprintf('No test file will be loaded.\n');
fprintf('N=%d, K=%d, P=[%s], RSR=%g dB, SNR=[%s] dB.\n', N, K, num2str(P_list), RSR_dB, num2str(SNR_grid));
fprintf('Confirmation samples per SNR=%d. Excluding reserved, V5 screening-selection, and prior diagnostic samples.\n', confirmation_samples_per_SNR);
fprintf('Output folder: %s\n', output_folder);

%% -------------------- OUTPUT STORAGE --------------------
sample_rows = cell(0, 15);
condition_rows = cell(0, 13);
stage_rows = cell(0, 12);
diagnostic_config = struct;
diagnostic_config.N = N;
diagnostic_config.K = K;
diagnostic_config.P_list = P_list;
diagnostic_config.SNR_grid = SNR_grid;
diagnostic_config.RSR_dB = RSR_dB;
diagnostic_config.p_H = p_H;
diagnostic_config.number_of_stages = number_of_stages;
diagnostic_config.maximum_updates = maximum_updates;
diagnostic_config.convergence_tolerance = convergence_tolerance;
diagnostic_config.reserved_validation_samples_per_SNR = reserved_validation_samples_per_SNR;
diagnostic_config.screening_validation_samples_per_SNR = screening_validation_samples_per_SNR;
diagnostic_config.previous_diagnostic_samples_per_SNR = previous_diagnostic_samples_per_SNR;
diagnostic_config.confirmation_samples_per_SNR = confirmation_samples_per_SNR;
diagnostic_config.validation_source = validation_folder;
diagnostic_config.split_used = 'validation';
diagnostic_config.test_files_loaded = false;

%% -------------------- EVALUATE BOTH PILOT COUNTS --------------------
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(validation_folder, sprintf('validation_P%d.mat', P));
    model_file = fullfile(model_folder, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    if ~isfile(validation_file) || ~isfile(model_file)
        error('Missing validation data or frozen V4 model for P=%d.', P);
    end

    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'SNR_dB', 'number_of_samples', 'split_name', 'data_version');
    model_data = load(model_file, 'best_parameters_numeric', 'best_epoch_saved', 'N', 'K', 'P', 'RSR_dB', 'p_H', 'number_of_stages', 'parameters_per_stage', 'number_of_gate_features', 'number_of_gate_outputs', 'initialization_method');

    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || validation_data.number_of_samples ~= 4000 || ~strcmp(validation_data.split_name, 'validation') || ~strcmp(validation_data.data_version, 'V5_fresh_validation_selected')
        error('Validation metadata mismatch for P=%d.', P);
    end
    if ~isequal(double(validation_data.SNR_dB), SNR_grid)
        error('Validation SNR grid mismatch for P=%d.', P);
    end
    if model_data.N ~= N || model_data.K ~= K || model_data.P ~= P || model_data.RSR_dB ~= RSR_dB || model_data.p_H ~= p_H || model_data.number_of_stages ~= number_of_stages || model_data.parameters_per_stage ~= parameters_per_stage || ~isequal(size(model_data.best_parameters_numeric), [parameters_per_stage number_of_stages]) || ~strcmp(model_data.initialization_method, 'GS')
        error('Frozen V4 model metadata mismatch for P=%d.', P);
    end

    prior_indices_for_P = previous_diagnostic_table(previous_diagnostic_table.P == P, :);
    selected_indices = local_select_confirmation_validation_indices(double(validation_data.SNR_dB_data(:)), SNR_grid, reserved_validation_samples_per_SNR, screening_validation_samples_per_SNR, previous_diagnostic_samples_per_SNR, confirmation_samples_per_SNR, prior_indices_for_P);
    selected_ranks = zeros(size(SNR_grid));
    for snr_index = 1:numel(SNR_grid)
        policy_rows = find(policy_table.P == P & abs(policy_table.SNR_dB-SNR_grid(snr_index)) < 1e-9);
        if numel(policy_rows) ~= 1
            error('Expected one frozen hard rank for P=%d and SNR=%g dB.', P, SNR_grid(snr_index));
        end
        selected_ranks(snr_index) = policy_table.FrozenHardRank(policy_rows);
        if ~ismember(selected_ranks(snr_index), 1:4)
            error('Invalid frozen hard rank for P=%d and SNR=%g dB.', P, SNR_grid(snr_index));
        end
    end

    fprintf('\nP=%d: selected frozen model epoch=%d; confirmation samples=%d.\n', P, model_data.best_epoch_saved, numel(selected_indices));
    method_nmse = zeros(numel(selected_indices), 4);
    method_nmse_dB = zeros(numel(selected_indices), 4);
    selected_snr = zeros(numel(selected_indices), 1);
    used_sample_index = zeros(numel(selected_indices), 1);
    convergence_flags = false(numel(selected_indices), 3);
    update_counts = zeros(numel(selected_indices), 3);
    stage_nmse_sum = zeros(numel(SNR_grid), number_of_stages, 3);
    stage_gate_sum = zeros(numel(SNR_grid), number_of_stages, 3);
    stage_sample_count = zeros(numel(SNR_grid), 1);

    for selected_position = 1:numel(selected_indices)
        sample_index = selected_indices(selected_position);
        Y = double(validation_data.Y_data(:, :, sample_index));
        G_true = double(validation_data.G_data(:, :, sample_index));
        S = double(validation_data.S_data(:, :, sample_index));
        B = double(validation_data.B_data(:, :, sample_index));
        sigma2 = double(validation_data.sigma2_data(sample_index));
        current_SNR_dB = double(validation_data.SNR_dB_data(sample_index));
        snr_index = find(abs(SNR_grid-current_SNR_dB) < 1e-9, 1);
        if isempty(snr_index)
            error('Unexpected validation SNR at P=%d, sample=%d.', P, sample_index);
        end

        G_initial = local_spectral_initialize(Y, S, B);
        [G_GS, gs_updates, gs_converged] = local_biased_gs(Y, S, B, G_initial, maximum_updates, convergence_tolerance);
        [G_EMGS, emgs_updates, emgs_converged] = local_emgs(Y, S, B, sigma2, G_initial, maximum_updates, convergence_tolerance);
        [G_HS_GS, hsgs_updates, hsgs_converged] = local_hard_rank_hsgs(Y, S, B, sigma2, G_initial, p_H, selected_ranks(snr_index), maximum_updates, convergence_tolerance);
        [G_V4, stage_nmse, stage_branch_nmse, stage_gate_weights] = local_v4_forward(model_data.best_parameters_numeric, Y, G_true, S, B, sigma2, G_GS, N, K, p_H, number_of_stages);

        true_energy = max(sum(abs(G_true).^2, 'all'), eps);
        current_nmse = [sum(abs(G_GS-G_true).^2, 'all'), sum(abs(G_EMGS-G_true).^2, 'all'), sum(abs(G_HS_GS-G_true).^2, 'all'), sum(abs(G_V4-G_true).^2, 'all')]/true_energy;
        method_nmse(selected_position, :) = current_nmse;
        method_nmse_dB(selected_position, :) = 10*log10(max(current_nmse, realmin));
        selected_snr(selected_position) = current_SNR_dB;
        used_sample_index(selected_position) = sample_index;
        convergence_flags(selected_position, :) = [gs_converged emgs_converged hsgs_converged];
        update_counts(selected_position, :) = [gs_updates emgs_updates hsgs_updates];
        stage_sample_count(snr_index) = stage_sample_count(snr_index)+1;
        stage_nmse_sum(snr_index, :, 1) = stage_nmse_sum(snr_index, :, 1)+stage_nmse(:).';
        stage_nmse_sum(snr_index, :, 2) = stage_nmse_sum(snr_index, :, 2)+stage_branch_nmse(:, 1).';
        stage_nmse_sum(snr_index, :, 3) = stage_nmse_sum(snr_index, :, 3)+stage_branch_nmse(:, 2).';
        stage_gate_sum(snr_index, :, :) = stage_gate_sum(snr_index, :, :)+reshape(stage_gate_weights, 1, number_of_stages, 3);

        sample_rows(end+1, :) = {P, current_SNR_dB, sample_index, current_nmse(1), current_nmse(2), current_nmse(3), current_nmse(4), method_nmse_dB(selected_position, 2)-method_nmse_dB(selected_position, 4), method_nmse_dB(selected_position, 3)-method_nmse_dB(selected_position, 4), gs_updates, emgs_updates, hsgs_updates, gs_converged, emgs_converged, hsgs_converged};
        if mod(selected_position, 16) == 0 || selected_position == numel(selected_indices)
            fprintf('P=%d validation samples processed: %d/%d.\n', P, selected_position, numel(selected_indices));
        end
    end

    %% -------------------- SUMMARIZE BY P AND SNR --------------------
    for snr_index = 1:numel(SNR_grid)
        condition_positions = find(selected_snr == SNR_grid(snr_index));
        if numel(condition_positions) ~= confirmation_samples_per_SNR
            error('Expected %d validation confirmation samples for P=%d, SNR=%g dB.', confirmation_samples_per_SNR, P, SNR_grid(snr_index));
        end
        mean_nmse = mean(method_nmse(condition_positions, :), 1);
        mean_nmse_dB = 10*log10(max(mean_nmse, realmin));
        mean_paired_gain_emgs = mean(method_nmse_dB(condition_positions, 2)-method_nmse_dB(condition_positions, 4));
        mean_paired_gain_hsgs = mean(method_nmse_dB(condition_positions, 3)-method_nmse_dB(condition_positions, 4));
        condition_rows(end+1, :) = {P, SNR_grid(snr_index), numel(condition_positions), mean_nmse(1), mean_nmse(2), mean_nmse(3), mean_nmse(4), mean_nmse_dB(1), mean_nmse_dB(2), mean_nmse_dB(3), mean_nmse_dB(4), mean_paired_gain_emgs, mean_paired_gain_hsgs};

        gs_initial_nmse = mean(method_nmse(condition_positions, 1));
        stage_rows(end+1, :) = {P, SNR_grid(snr_index), 0, 10*log10(max(gs_initial_nmse, realmin)), NaN, NaN, NaN, NaN, NaN, selected_ranks(snr_index), numel(condition_positions), model_data.best_epoch_saved};
        for stage_index = 1:number_of_stages
            mean_stage_nmse = squeeze(stage_nmse_sum(snr_index, stage_index, :)).'/stage_sample_count(snr_index);
            mean_gate = squeeze(stage_gate_sum(snr_index, stage_index, :)).'/stage_sample_count(snr_index);
            stage_rows(end+1, :) = {P, SNR_grid(snr_index), stage_index, 10*log10(max(mean_stage_nmse(1), realmin)), 10*log10(max(mean_stage_nmse(2), realmin)), 10*log10(max(mean_stage_nmse(3), realmin)), mean_gate(1), mean_gate(2), mean_gate(3), selected_ranks(snr_index), stage_sample_count(snr_index), model_data.best_epoch_saved};
        end
    end
end

%% -------------------- SAVE VALIDATION DIAGNOSTICS --------------------
sample_table = cell2table(sample_rows, 'VariableNames', {'P','SNR_dB','ValidationSampleIndex','GS_NMSE_linear','EMGS_NMSE_linear','HardRankHSGS_NMSE_linear','V4_NMSE_linear','PairedGain_EMGSminusV4_dB','PairedGain_HSGSminusV4_dB','GS_Updates','EMGS_Updates','HardRankHSGS_Updates','GS_Converged','EMGS_Converged','HardRankHSGS_Converged'});
condition_table = cell2table(condition_rows, 'VariableNames', {'P','SNR_dB','SampleCount','GS_MeanNMSE_linear','EMGS_MeanNMSE_linear','HardRankHSGS_MeanNMSE_linear','V4_MeanNMSE_linear','GS_MeanNMSE_dB','EMGS_MeanNMSE_dB','HardRankHSGS_MeanNMSE_dB','V4_MeanNMSE_dB','MeanPairedGain_EMGSminusV4_dB','MeanPairedGain_HSGSminusV4_dB'});
stage_table = cell2table(stage_rows, 'VariableNames', {'P','SNR_dB','Stage','V4Output_MeanNMSE_dB','EMGSBranch_MeanNMSE_dB','HankelBranch_MeanNMSE_dB','GateWeight_Previous','GateWeight_EMGS','GateWeight_Hankel','FrozenHardRank','SampleCount','V4SelectedEpoch'});

mkdir(output_folder);
sample_csv = fullfile(output_folder, 'V4_validation_confirmation_paired_samples.csv');
condition_csv = fullfile(output_folder, 'V4_validation_confirmation_condition_summary.csv');
stage_csv = fullfile(output_folder, 'V4_validation_confirmation_stage_gate_summary.csv');
mat_file = fullfile(output_folder, 'V4_validation_confirmation_stage_gate.mat');
writetable(sample_table, sample_csv);
writetable(condition_table, condition_csv);
writetable(stage_table, stage_csv);
save(mat_file, 'sample_table', 'condition_table', 'stage_table', 'diagnostic_config', '-v7.3');

fprintf('\nSaved validation-only confirmation sample table: %s\n', sample_csv);
fprintf('Saved validation-only confirmation condition summary: %s\n', condition_csv);
fprintf('Saved validation-only confirmation stage and gate summary: %s\n', stage_csv);
fprintf('Saved validation-only confirmation MATLAB results: %s\n', mat_file);
fprintf('No test data were loaded or scored. No model parameters were changed.\n');

%% -------------------- LOCAL VALIDATION SAMPLE SELECTION --------------------
function selected_indices = local_select_confirmation_validation_indices(SNR_dB_data, SNR_grid, reserved_count, screening_count, previous_diagnostic_count, confirmation_count, previous_diagnostic_table)
selected_indices = [];
for snr_index = 1:numel(SNR_grid)
    candidates = find(abs(SNR_dB_data-SNR_grid(snr_index)) < 1e-9);
    if numel(candidates) ~= 500
        error('Unexpected validation sample count at SNR=%g dB.', SNR_grid(snr_index));
    end
    reserved_positions = round(linspace(1, numel(candidates), reserved_count));
    remaining_after_reserved = setdiff(candidates, candidates(reserved_positions(:)), 'stable');
    screening_positions = round(linspace(1, numel(remaining_after_reserved), screening_count));
    screening_indices = remaining_after_reserved(screening_positions(:));
    available_indices = setdiff(remaining_after_reserved, screening_indices, 'stable');

    prior_rows = previous_diagnostic_table(abs(previous_diagnostic_table.SNR_dB-SNR_grid(snr_index)) < 1e-9, :);
    if height(prior_rows) ~= previous_diagnostic_count
        error('Expected %d prior diagnostic samples at SNR=%g dB.', previous_diagnostic_count, SNR_grid(snr_index));
    end
    prior_diagnostic_indices = prior_rows.ValidationSampleIndex(:).';
    available_indices = setdiff(available_indices, prior_diagnostic_indices, 'stable');
    if numel(available_indices) < confirmation_count
        error('Not enough unused validation samples remain at SNR=%g dB.', SNR_grid(snr_index));
    end

    confirmation_positions = round(linspace(1, numel(available_indices), confirmation_count));
    condition_indices = available_indices(confirmation_positions(:));
    if numel(unique(condition_indices)) ~= confirmation_count || any(ismember(condition_indices, candidates(reserved_positions(:)))) || any(ismember(condition_indices, screening_indices)) || any(ismember(condition_indices, prior_diagnostic_indices))
        error('Confirmation sample selection overlaps a reserved subset at SNR=%g dB.', SNR_grid(snr_index));
    end
    selected_indices = [selected_indices; condition_indices];
end
end

%% -------------------- BIASED-GS SPECTRAL INITIALIZATION --------------------
function G = local_spectral_initialize(Y, S, B)
N = size(Y, 1);
K = size(S, 1);
G = zeros(N, K);
for antenna_index = 1:N
    y = Y(antenna_index, :).';
    b = B(antenna_index, :).';
    augmented_matrix = [S.', b];
    spectral_matrix = augmented_matrix'*diag(y)*augmented_matrix;
    [spectral_vectors, spectral_values] = eig(spectral_matrix);
    [~, principal_index] = max(real(diag(spectral_values)));
    principal_vector = spectral_vectors(:, principal_index);
    principal_vector = principal_vector/max(norm(principal_vector), eps);
    spectral_scale = (abs(augmented_matrix*principal_vector).'*y)/max(norm(augmented_matrix*principal_vector)^2, eps);
    G(antenna_index, :) = (spectral_scale*exp(-1i*angle(principal_vector(K+1)))*principal_vector(1:K)).';
end
end

%% -------------------- CLASSICAL ESTIMATORS --------------------
function [G, iteration_count, converged] = local_biased_gs(Y, S, B, G_initial, maximum_updates, convergence_tolerance)
G = G_initial;
right_inverse = S'/(S*S');
iteration_count = maximum_updates;
converged = false;
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

function [G, iteration_count, converged] = local_emgs(Y, S, B, sigma2, G_initial, maximum_updates, convergence_tolerance)
G = G_initial;
right_inverse = S'/(S*S');
iteration_count = maximum_updates;
converged = false;
for update_index = 1:maximum_updates
    G_previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    if norm(G-G_previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

function [G, iteration_count, converged] = local_hard_rank_hsgs(Y, S, B, sigma2, G_initial, p_H, selected_rank, maximum_updates, convergence_tolerance)
N = size(Y, 1);
K = size(S, 1);
right_inverse = S'/(S*S');
G = G_initial;
iteration_count = maximum_updates;
converged = false;
for update_index = 1:maximum_updates
    G_previous = G;
    field = G_previous*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G_em_update = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    G_projected = zeros(N, K);
    for user_index = 1:K
        channel_vector = G_em_update(:, user_index);
        hankel_matrix = local_hankel_embed(channel_vector, p_H);
        [U, Sigma, V] = svd(hankel_matrix, 'econ');
        hankel_matrix = U(:, 1:selected_rank)*Sigma(1:selected_rank, 1:selected_rank)*V(:, 1:selected_rank)';
        G_projected(:, user_index) = local_hankel_average(hankel_matrix, N, p_H);
    end
    G = G_projected;
    if norm(G-G_previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

%% -------------------- V4 FORWARD PASS AND STAGE LOGGING --------------------
function [G, output_nmse, branch_nmse, gate_weights_by_stage] = local_v4_forward(parameters, Y, G_true, S, B, sigma2, G_initial, N, K, p_H, number_of_stages)
right_inverse = S'/(S*S');
G = G_initial;
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2, 'all'), eps), eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
true_energy = max(sum(abs(G_true).^2, 'all'), eps);
output_nmse = zeros(number_of_stages, 1);
branch_nmse = zeros(number_of_stages, 2);
gate_weights_by_stage = zeros(number_of_stages, 3);
for stage_index = 1:number_of_stages
    G_previous = G;
    field = G_previous*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    G_hankel = zeros(N, K);
    for user_index = 1:K
        hankel_matrix = local_hankel_embed(G_emgs(:, user_index), p_H);
        gram_matrix = hankel_matrix'*hankel_matrix;
        [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
        singular_values = sqrt(max(real(diag(eigenvalue_matrix)), 0));
        safe_singular_values = max(singular_values, 1e-12);
        mean_singular_value = max(mean(safe_singular_values), 1e-12);
        relative_spectrum_feature = tanh(log(safe_singular_values./mean_singular_value));
        threshold_logits = parameters(1, stage_index)+parameters(2, stage_index)*noise_feature+parameters(3, stage_index)*relative_spectrum_feature;
        mode_thresholds = noise_scale*(2./(1+exp(-threshold_logits)));
        shrink_factors = max(1-mode_thresholds./safe_singular_values, 0);
        hankel_shrunk = hankel_matrix*(eigenvectors.*reshape(shrink_factors, 1, []))*eigenvectors';
        G_hankel(:, user_index) = local_hankel_average(hankel_shrunk, N, p_H);
    end
    emgs_change = tanh(log(1+sqrt(sum(abs(G_emgs-G_previous).^2, 'all')/max(sum(abs(G_previous).^2, 'all'), eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_emgs).^2, 'all')/max(sum(abs(G_emgs).^2, 'all'), eps)))/3);
    gate_coefficients = reshape(parameters(4:15, stage_index), 4, 3);
    gate_logits = gate_coefficients(1, :).'*noise_feature+gate_coefficients(2, :).'*emgs_change+gate_coefficients(3, :).'*hankel_change+gate_coefficients(4, :).';
    gate_logits = gate_logits-max(gate_logits);
    current_gate_weights = exp(gate_logits)./sum(exp(gate_logits));
    G = current_gate_weights(1)*G_previous+current_gate_weights(2)*G_emgs+current_gate_weights(3)*G_hankel;
    output_nmse(stage_index) = sum(abs(G-G_true).^2, 'all')/true_energy;
    branch_nmse(stage_index, 1) = sum(abs(G_emgs-G_true).^2, 'all')/true_energy;
    branch_nmse(stage_index, 2) = sum(abs(G_hankel-G_true).^2, 'all')/true_energy;
    gate_weights_by_stage(stage_index, :) = current_gate_weights(:).';
end
end

%% -------------------- HANKEL EMBEDDING AND ANTI-DIAGONAL AVERAGING --------------------
function hankel_matrix = local_hankel_embed(channel_vector, p_H)
N = numel(channel_vector);
number_of_rows = N-p_H;
number_of_columns = p_H+1;
hankel_matrix = zeros(number_of_rows, number_of_columns);
for row_index = 1:number_of_rows
    hankel_matrix(row_index, :) = channel_vector(row_index:row_index+number_of_columns-1).';
end
end

function channel_vector = local_hankel_average(hankel_matrix, N, p_H)
reconstructed_channel = zeros(N, 1);
diagonal_counts = zeros(N, 1);
for row_index = 1:N-p_H
    for column_index = 1:p_H+1
        antenna_index = row_index+column_index-1;
        reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(row_index, column_index);
        diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
    end
end
channel_vector = reconstructed_channel./diagonal_counts;
end
