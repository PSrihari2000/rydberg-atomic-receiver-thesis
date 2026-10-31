%% V25: model-driven unrolled EM-GS with learned filtering, gates, and channel correction
% This is a MATLAB-only, URformer-inspired adaptation for the local model
% Y = abs(G*S + B + W), with N=8, K=3, P=10/30, and RSR=25 dB.
%
% Each unrolled layer performs an EM-GS phase reconstruction, learns a
% kappa-dependent filter and a scalar update gate, applies the LS channel
% update, then predicts a channel-wide residual with a small MLP. The MLP
% sees the complete real/imaginary channel matrix, so it can learn joint
% antenna/user corrections. It replaces the paper's Transformer because
% K=3 and N=8 make a large Transformer disproportionate for this first
% experiment. This is an adaptation, not an exact paper reproduction.
%
% Only training files and V23 fresh validation files are loaded. V20 train
% GS-anchor caches and V24 validation GS-anchor caches are read only. Test
% files are never opened. The reserved V23 test split remains sealed.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2');
%   DL_V25_UNROLLED_EMGS_FILTER_GATE_CHANNEL_MLP_RSR25dB

clear; clc; close all;

script_folder = fileparts(mfilename('fullpath'));
project_folder = fileparts(script_folder);
train_dataset_folder = fullfile(project_folder, 'GS-GS-HS-DL', 'DL_dataset_RSR25dB');
validation_dataset_folder = fullfile(project_folder, 'GS-GS-HS-DL', 'DL_dataset_RSR25dB_V23_valtest');
v20_anchor_root = fullfile(script_folder, 'DL_V20_GS_ANCHORED_RELATIVE_NMSE_MLP_v3_RSR25dB');
v24_anchor_root = fullfile(script_folder, 'DL_V24_L2_REGULARIZED_PER_SNR_EXPERT_MLP_RSR25dB');
output_folder = fullfile(script_folder, 'DL_V25_UNROLLED_EMGS_FILTER_GATE_CHANNEL_MLP_RSR25dB');
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

P_list = [10 30];
SNR_grid = -5:5:30;
N_expected = 8; K_expected = 3; RSR_dB = 25;
GS_max_iterations = 1000; GS_tolerance = 1e-10; anchor_chunk_size = 128;
number_of_unrolled_layers = 4;
filter_hidden_width = 8;
correction_hidden_width = 128;
number_of_epochs = 50;
mini_batch_size = 64;
initial_learning_rate = 1e-3;
early_stopping_patience = 10;
minimum_validation_improvement_dB = 1e-4;
weight_decay = 1e-6;
gradient_norm_limit = 5;
seed_base = 20261028;

fprintf('\n============================================================\n');
fprintf(' V25: UNROLLED EM-GS WITH LEARNED FILTER, GATE, AND CHANNEL MLP\n');
fprintf('============================================================\n');
fprintf('Pilot counts                  = [%s]\n', num2str(P_list));
fprintf('SNR values                    = [%s] dB\n', num2str(SNR_grid));
fprintf('Unrolled layers               = %d\n', number_of_unrolled_layers);
fprintf('FilterNet hidden width        = %d\n', filter_hidden_width);
fprintf('Channel residual MLP width    = %d\n', correction_hidden_width);
fprintf('Training/validation           = separate; independent test not loaded\n');
fprintf('Initial state                 = cached biased-GS estimate\n');
fprintf('GS anchor cache tolerance     = %.1e, maximum %d iterations\n', GS_tolerance, GS_max_iterations);
fprintf('============================================================\n');

summary_table = table();

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    fprintf('\n============================================================\n');
    fprintf('V25 training for P=%d\n', P);
    fprintf('============================================================\n');

    train_file = fullfile(train_dataset_folder, sprintf('train_P%d.mat', P));
    validation_file = fullfile(validation_dataset_folder, sprintf('validation_P%d.mat', P));
    if ~isfile(train_file) || ~isfile(validation_file)
        error('P=%d training or V23 validation file is missing.', P);
    end
    train_source = load(train_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data');
    validation_source = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data');

    validate_split(train_source, N_expected, K_expected, P, SNR_grid, 'training');
    validate_split(validation_source, N_expected, K_expected, P, SNR_grid, 'fresh validation');
    N = size(train_source.Y_data, 1); K = size(train_source.S_data, 1);
    training_channel_scale = sqrt(mean(abs(double(train_source.G_data(:))).^2));
    training_channel_scale = max(training_channel_scale, sqrt(eps));

    train_anchor_folder = fullfile(v20_anchor_root, sprintf('anchor_cache_P%d_train', P));
    validation_anchor_folder = fullfile(v24_anchor_root, sprintf('anchor_cache_P%d_validation', P));
    G_GS_train = read_cached_anchors(train_anchor_folder, size(train_source.Y_data, 3), ...
        P, N, K, GS_max_iterations, GS_tolerance, anchor_chunk_size);
    G_GS_validation = read_cached_anchors(validation_anchor_folder, size(validation_source.Y_data, 3), ...
        P, N, K, GS_max_iterations, GS_tolerance, anchor_chunk_size);

    train_data = prepare_split(train_source, G_GS_train, training_channel_scale);
    validation_data = prepare_split(validation_source, G_GS_validation, training_channel_scale);
    fprintf('Training samples=%d; V23 validation samples=%d; training-only channel RMS=%.7g\n', ...
        size(train_data.Y, 3), size(validation_data.Y, 3), training_channel_scale);

    model_configuration = struct('model_version', 'V25_unrolled_EMGS_filter_gate_channel_MLP', ...
        'N', N, 'K', K, 'P', P, 'RSR_dB', RSR_dB, 'SNR_grid', SNR_grid, ...
        'number_of_unrolled_layers', number_of_unrolled_layers, ...
        'filter_hidden_width', filter_hidden_width, ...
        'correction_hidden_width', correction_hidden_width, ...
        'training_channel_scale', training_channel_scale, ...
        'GS_max_iterations', GS_max_iterations, 'GS_tolerance', GS_tolerance, ...
        'mini_batch_size', mini_batch_size, 'number_of_epochs', number_of_epochs, ...
        'initial_learning_rate', initial_learning_rate, 'weight_decay', weight_decay, ...
        'seed_base', seed_base);

    rng(seed_base + P, 'twister');
    parameter_file = fullfile(output_folder, sprintf('model_P%d_best.mat', P));
    checkpoint_file = fullfile(output_folder, sprintf('checkpoint_P%d.mat', P));
    history_file = fullfile(output_folder, sprintf('history_P%d.mat', P));
    [parameters, history, best_epoch, best_validation_score, best_validation_metrics] = ...
        train_one_pilot_count(train_data, validation_data, SNR_grid, N, K, P, ...
        number_of_unrolled_layers, filter_hidden_width, correction_hidden_width, ...
        number_of_epochs, mini_batch_size, initial_learning_rate, ...
        early_stopping_patience, minimum_validation_improvement_dB, ...
        weight_decay, gradient_norm_limit, model_configuration, ...
        parameter_file, checkpoint_file, history_file);

    final_metrics = evaluate_validation(parameters, validation_data, SNR_grid, N, K, P, ...
        number_of_unrolled_layers, mini_batch_size);
    summary_table = [summary_table; final_metrics]; %#ok<AGROW>
    fprintf('P=%d selected epoch=%d; validation aggregate NMSE=%.4f dB; mean DL-minus-GS=%.4f dB\n', ...
        P, best_epoch, best_validation_score, mean(final_metrics.DL_minus_GS_dB));
    save(parameter_file, 'parameters', 'model_configuration', 'best_epoch', ...
        'best_validation_score', 'best_validation_metrics', 'final_metrics', '-v7.3');
    save(history_file, 'history', 'model_configuration', '-v7.3');
end

summary_file = fullfile(output_folder, 'validation_summary.csv');
writetable(summary_table, summary_file);
save(fullfile(output_folder, 'validation_summary.mat'), 'summary_table', ...
    'P_list', 'SNR_grid', 'RSR_dB', '-v7.3');
plot_validation_summary(summary_table, P_list, SNR_grid, output_folder);
fprintf('\nV25 training and V23 fresh-validation evaluation complete.\n');
fprintf('No test files were loaded.\n');
fprintf('Models, checkpoints, validation table, and figure are in: %s\n', output_folder);

function validate_split(source, N, K, P, SNR_grid, split_name)
if size(source.Y_data,1) ~= N || size(source.Y_data,2) ~= P || ...
        size(source.G_data,1) ~= N || size(source.G_data,2) ~= K || ...
        size(source.S_data,1) ~= K || size(source.S_data,2) ~= P || ...
        size(source.B_data,1) ~= N || size(source.B_data,2) ~= P || ...
        size(source.Y_data,3) ~= size(source.G_data,3) || ...
        size(source.Y_data,3) ~= numel(source.SNR_dB_data) || ...
        size(source.Y_data,3) ~= numel(source.sigma2_data)
    error('%s dimensions are inconsistent.', split_name);
end
available_snr = unique(double(source.SNR_dB_data(:)));
if ~all(ismember(SNR_grid, available_snr))
    error('%s data is missing one or more expected SNR groups.', split_name);
end
end

function G_GS = read_cached_anchors(cache_folder, sample_count, P, N, K, max_iterations, tolerance, chunk_size)
if ~isfolder(cache_folder)
    error('Required read-only GS anchor cache is missing: %s', cache_folder);
end
G_GS = complex(zeros(N, K, sample_count, 'single'));
number_of_chunks = ceil(sample_count / chunk_size);
for chunk_number = 1:number_of_chunks
    first_sample = (chunk_number - 1) * chunk_size + 1;
    last_sample = min(chunk_number * chunk_size, sample_count);
    expected_indices = first_sample:last_sample;
    chunk_file = fullfile(cache_folder, sprintf('gs_anchor_chunk_%05d.mat', chunk_number));
    if ~isfile(chunk_file), error('Required GS anchor chunk is missing: %s', chunk_file); end
    cached = load(chunk_file, 'sample_indices', 'G_GS_chunk', 'P_cached', ...
        'max_iterations_cached', 'tolerance_cached', 'N_cached', 'K_cached');
    cache_matches = isequal(cached.sample_indices, expected_indices) && ...
        cached.P_cached == P && cached.max_iterations_cached == max_iterations && ...
        cached.tolerance_cached == tolerance && cached.N_cached == N && cached.K_cached == K && ...
        size(cached.G_GS_chunk, 3) == numel(expected_indices);
    if ~cache_matches, error('GS anchor cache metadata mismatch: %s', chunk_file); end
    G_GS(:, :, expected_indices) = cached.G_GS_chunk;
end
end

function data = prepare_split(source, G_GS, channel_scale)
data.Y = single(source.Y_data) / single(channel_scale);
data.G = single(source.G_data) / single(channel_scale);
data.S = single(source.S_data);
data.B = single(source.B_data) / single(channel_scale);
data.sigma2 = single(source.sigma2_data(:).') / single(channel_scale)^2;
data.G0 = single(G_GS) / single(channel_scale);
data.SNR_dB = double(source.SNR_dB_data(:));
data.A = precompute_right_inverse(data.S);
end

function A = precompute_right_inverse(S_data)
[K, P, sample_count] = size(S_data);
A = complex(zeros(P, K, sample_count, 'single'));
for sample_index = 1:sample_count
    S = double(S_data(:, :, sample_index));
    A(:, :, sample_index) = single(S' / (S * S'));
end
end

function [parameters, history, best_epoch, best_validation_score, best_validation_metrics] = ...
    train_one_pilot_count(train_data, validation_data, SNR_grid, N, K, P, ...
    number_of_unrolled_layers, filter_hidden_width, correction_hidden_width, ...
    number_of_epochs, mini_batch_size, initial_learning_rate, early_stopping_patience, ...
    minimum_validation_improvement_dB, weight_decay, gradient_norm_limit, ...
    model_configuration, parameter_file, checkpoint_file, history_file)

feature_count = 2 * N * K;
parameters = initialize_parameters(N, K, number_of_unrolled_layers, ...
    filter_hidden_width, correction_hidden_width);
average_gradient = zero_state(parameters);
average_squared_gradient = zero_state(parameters);
iteration = 0; start_epoch = 1; best_epoch = 0;
best_validation_score = inf; best_parameters = parameters;
best_validation_metrics = table(); epochs_without_improvement = 0;
history = table('Size', [0 6], 'VariableTypes', repmat({'double'}, 1, 6), ...
    'VariableNames', {'Epoch','LearningRate','MeanTrainingRelativeNMSE', ...
    'ValidationAggregateNMSE_dB','ValidationMeanDLminusGS_dB','ValidationWorstDLminusGS_dB'});

if isfile(checkpoint_file)
    checkpoint = load(checkpoint_file, 'model_configuration', 'parameters', ...
        'average_gradient', 'average_squared_gradient', 'iteration', 'next_epoch', ...
        'best_epoch', 'best_validation_score', 'best_parameters', ...
        'best_validation_metrics', 'epochs_without_improvement', 'history');
    if ~isequal(checkpoint.model_configuration, model_configuration)
        error('P=%d V25 checkpoint configuration mismatch; preserve it and use a new output folder.', P);
    end
    parameters = checkpoint.parameters;
    average_gradient = checkpoint.average_gradient;
    average_squared_gradient = checkpoint.average_squared_gradient;
    iteration = checkpoint.iteration;
    start_epoch = checkpoint.next_epoch;
    best_epoch = checkpoint.best_epoch;
    best_validation_score = checkpoint.best_validation_score;
    best_parameters = checkpoint.best_parameters;
    best_validation_metrics = checkpoint.best_validation_metrics;
    epochs_without_improvement = checkpoint.epochs_without_improvement;
    history = checkpoint.history;
    fprintf('Resuming P=%d from epoch %d.\n', P, start_epoch);
elseif isfile(parameter_file)
    error('V25 selected model already exists without a checkpoint. Preserve it and use a new output folder.');
end

sample_count = size(train_data.Y, 3);
for epoch = start_epoch:number_of_epochs
    rng(model_configuration.seed_base + 1000 * P + epoch, 'twister');
    shuffled_samples = randperm(sample_count);
    learning_rate = initial_learning_rate * 0.5 * (1 + cos(pi * (epoch - 1) / number_of_epochs));
    batch_losses = zeros(ceil(sample_count / mini_batch_size), 1);
    batch_number = 0;
    for first_index = 1:mini_batch_size:sample_count
        last_index = min(first_index + mini_batch_size - 1, sample_count);
        sample_indices = shuffled_samples(first_index:last_index);
        batch_number = batch_number + 1;
        Y = dlarray(train_data.Y(:, :, sample_indices));
        G = dlarray(train_data.G(:, :, sample_indices));
        S = train_data.S(:, :, sample_indices);
        B = dlarray(train_data.B(:, :, sample_indices));
        sigma2 = dlarray(reshape(train_data.sigma2(sample_indices), 1, 1, []));
        G0 = dlarray(train_data.G0(:, :, sample_indices));
        A = train_data.A(:, :, sample_indices);

        [loss, gradients] = dlfeval(@model_loss, parameters, Y, G, S, B, sigma2, G0, A, ...
            N, K, number_of_unrolled_layers, weight_decay);
        loss_value = double(extractdata(loss));
        if ~isfinite(loss_value), error('P=%d training loss became nonfinite at epoch %d.', P, epoch); end
        gradients = clip_gradient_tree(gradients, gradient_norm_limit);
        iteration = iteration + 1;
        [parameters, average_gradient, average_squared_gradient] = adam_update_tree( ...
            parameters, gradients, average_gradient, average_squared_gradient, iteration, learning_rate);
        batch_losses(batch_number) = loss_value;
    end

    validation_metrics = evaluate_validation(parameters, validation_data, SNR_grid, N, K, P, ...
        number_of_unrolled_layers, mini_batch_size);
    validation_score = validation_metrics.AggregateDL_NMSE_dB(1);
    history(end + 1, :) = {epoch, learning_rate, mean(batch_losses), validation_score, ...
        mean(validation_metrics.DL_minus_GS_dB), max(validation_metrics.DL_minus_GS_dB)}; %#ok<AGROW>
    fprintf('P=%d epoch %3d/%d | train %.6g | val aggregate %.4f dB | mean/worst DL-GS %.4f / %.4f dB\n', ...
        P, epoch, number_of_epochs, mean(batch_losses), validation_score, ...
        mean(validation_metrics.DL_minus_GS_dB), max(validation_metrics.DL_minus_GS_dB));

    if validation_score < best_validation_score - minimum_validation_improvement_dB
        best_validation_score = validation_score;
        best_parameters = parameters;
        best_epoch = epoch;
        best_validation_metrics = validation_metrics;
        epochs_without_improvement = 0;
        save_model_atomic(parameter_file, best_parameters, model_configuration, ...
            best_epoch, best_validation_score, best_validation_metrics);
    else
        epochs_without_improvement = epochs_without_improvement + 1;
    end

    next_epoch = epoch + 1;
    temporary_checkpoint = [checkpoint_file '.tmp.mat'];
    save(temporary_checkpoint, 'model_configuration', 'parameters', ...
        'average_gradient', 'average_squared_gradient', 'iteration', 'next_epoch', ...
        'best_epoch', 'best_validation_score', 'best_parameters', ...
        'best_validation_metrics', 'epochs_without_improvement', 'history', '-v7.3');
    movefile(temporary_checkpoint, checkpoint_file, 'f');
    temporary_history = [history_file '.tmp.mat'];
    save(temporary_history, 'history', 'model_configuration', '-v7.3');
    movefile(temporary_history, history_file, 'f');

    if epochs_without_improvement >= early_stopping_patience
        fprintf('P=%d early stopping after %d epochs without validation improvement.\n', ...
            P, epochs_without_improvement);
        break;
    end
end

if best_epoch == 0
    error('P=%d did not produce a selected validation checkpoint.', P);
end
parameters = best_parameters;
best_validation_score = best_validation_score;
best_validation_metrics = best_validation_metrics;
end

function parameters = initialize_parameters(N, K, number_of_layers, filter_width, correction_width)
feature_count = 2 * N * K;
parameters.filterW1 = cell(1, number_of_layers);
parameters.filterB1 = cell(1, number_of_layers);
parameters.filterW2 = cell(1, number_of_layers);
parameters.filterB2 = cell(1, number_of_layers);
parameters.gateLogit = cell(1, number_of_layers);
parameters.correctionW1 = cell(1, number_of_layers);
parameters.correctionB1 = cell(1, number_of_layers);
parameters.correctionW2 = cell(1, number_of_layers);
parameters.correctionB2 = cell(1, number_of_layers);
for layer = 1:number_of_layers
    parameters.filterW1{layer} = dlarray(single(0.1 * randn(filter_width, 1)));
    parameters.filterB1{layer} = dlarray(zeros(filter_width, 1, 'single'));
    parameters.filterW2{layer} = dlarray(zeros(1, filter_width, 'single'));
    parameters.filterB2{layer} = dlarray(single(2.2));
    parameters.gateLogit{layer} = dlarray(single(0.5));
    parameters.correctionW1{layer} = dlarray(single(randn(correction_width, feature_count) * sqrt(2 / feature_count)));
    parameters.correctionB1{layer} = dlarray(zeros(correction_width, 1, 'single'));
    parameters.correctionW2{layer} = dlarray(zeros(feature_count, correction_width, 'single'));
    parameters.correctionB2{layer} = dlarray(zeros(feature_count, 1, 'single'));
end
end

function [loss, gradients] = model_loss(parameters, Y, G_true, S, B, sigma2, G0, A, N, K, number_of_layers, weight_decay)
G_estimate = unrolled_forward(parameters, Y, S, B, sigma2, G0, A, N, K, number_of_layers);
channel_error = G_estimate - G_true;
sample_error = reshape(sum(sum(real(channel_error).^2 + imag(channel_error).^2, 1), 2), 1, []);
sample_energy = reshape(sum(sum(real(G_true).^2 + imag(G_true).^2, 1), 2), 1, []);
data_loss = mean(sample_error ./ (sample_energy + single(1e-10)), 'all');
penalty = dlarray(single(0));
penalty_fields = {'filterW1','filterW2','correctionW1','correctionW2'};
for field_index = 1:numel(penalty_fields)
    values = parameters.(penalty_fields{field_index});
    for layer = 1:number_of_layers
        penalty = penalty + sum(values{layer}.^2, 'all');
    end
end
loss = real(data_loss + weight_decay * penalty);
gradients = dlgradient(loss, parameters);
end

function G_estimate = unrolled_forward(parameters, Y, S, B, sigma2, G, A, N, K, number_of_layers)
batch_size = size(Y, 3);
for layer = 1:number_of_layers
    current_field = pagemtimes(G, S) + B;
    field_magnitude = abs(current_field);
    phase = current_field ./ (field_magnitude + single(1e-8));
    direct_reconstruction = Y .* phase;
    kappa = 2 * Y .* field_magnitude ./ sigma2;
    kappa_vector = reshape(kappa, 1, []);

    filter_hidden = max(parameters.filterW1{layer} * kappa_vector + parameters.filterB1{layer}, 0);
    learned_filter = 1 ./ (1 + exp(-(parameters.filterW2{layer} * filter_hidden + parameters.filterB2{layer})));
    learned_filter = reshape(learned_filter, size(Y, 1), size(Y, 2), batch_size);
    gate = 1 ./ (1 + exp(-parameters.gateLogit{layer}));
    reconstructed_field = gate .* (direct_reconstruction .* learned_filter) + ...
        (1 - gate) .* direct_reconstruction;

    linear_channel = pagemtimes(reconstructed_field - B, A);
    real_channel_vector = reshape(real(linear_channel), N * K, batch_size);
    imaginary_channel_vector = reshape(imag(linear_channel), N * K, batch_size);
    correction_input = cat(1, real_channel_vector, imaginary_channel_vector);
    correction_hidden = max(parameters.correctionW1{layer} * correction_input + ...
        parameters.correctionB1{layer}, 0);
    correction_vector = parameters.correctionW2{layer} * correction_hidden + parameters.correctionB2{layer};
    correction = reshape(correction_vector, N, 2 * K, batch_size);
    correction_real = real(correction(:, 1:K, :));
    correction_imaginary = real(correction(:, K+1:2*K, :));
    correction_channel = correction_real + 1i * correction_imaginary;
    G = linear_channel + correction_channel;
end
G_estimate = G;
end

function state = zero_state(parameters)
state = struct();
fields = fieldnames(parameters);
for field_index = 1:numel(fields)
    parameter_values = parameters.(fields{field_index});
    state_values = cell(size(parameter_values));
    for layer = 1:numel(parameter_values)
        numeric_value = extractdata(parameter_values{layer});
        state_values{layer} = zeros(size(numeric_value), 'like', numeric_value);
    end
    state.(fields{field_index}) = state_values;
end
end

function gradient_norm = tree_gradient_norm(gradients)
gradient_norm_squared = 0;
fields = fieldnames(gradients);
for field_index = 1:numel(fields)
    values = gradients.(fields{field_index});
    for layer = 1:numel(values)
        numeric_gradient = real(extractdata(values{layer}));
        gradient_norm_squared = gradient_norm_squared + sum(double(numeric_gradient(:)).^2);
    end
end
gradient_norm = sqrt(gradient_norm_squared);
end

function gradients = clip_gradient_tree(gradients, maximum_norm)
current_norm = tree_gradient_norm(gradients);
if current_norm > maximum_norm
    scale = maximum_norm / max(current_norm, eps);
    fields = fieldnames(gradients);
    for field_index = 1:numel(fields)
        values = gradients.(fields{field_index});
        for layer = 1:numel(values), values{layer} = values{layer} .* scale; end
        gradients.(fields{field_index}) = values;
    end
end
end

function [parameters, average_gradient, average_squared_gradient] = ...
    adam_update_tree(parameters, gradients, average_gradient, average_squared_gradient, iteration, learning_rate)
beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
fields = fieldnames(parameters);
for field_index = 1:numel(fields)
    name = fields{field_index};
    parameter_values = parameters.(name);
    gradient_values = gradients.(name);
    first_moment_values = average_gradient.(name);
    second_moment_values = average_squared_gradient.(name);
    for layer = 1:numel(parameter_values)
        gradient = real(extractdata(gradient_values{layer}));
        first_moment = beta1 * first_moment_values{layer} + (1-beta1) * gradient;
        second_moment = beta2 * second_moment_values{layer} + (1-beta2) * gradient.^2;
        first_hat = first_moment / (1-beta1^iteration);
        second_hat = second_moment / (1-beta2^iteration);
        parameter_value = extractdata(parameter_values{layer});
        parameter_value = parameter_value - learning_rate * first_hat ./ (sqrt(second_hat) + epsilon);
        parameter_values{layer} = dlarray(parameter_value);
        first_moment_values{layer} = first_moment;
        second_moment_values{layer} = second_moment;
    end
    parameters.(name) = parameter_values;
    average_gradient.(name) = first_moment_values;
    average_squared_gradient.(name) = second_moment_values;
end
end

function metrics = evaluate_validation(parameters, data, SNR_grid, N, K, P, number_of_layers, batch_size)
total_dl_error = zeros(numel(SNR_grid), 1);
total_gs_error = zeros(numel(SNR_grid), 1);
total_channel_energy = zeros(numel(SNR_grid), 1);
sample_count = zeros(numel(SNR_grid), 1);
for snr_index = 1:numel(SNR_grid)
    indices = find(abs(data.SNR_dB - SNR_grid(snr_index)) < 1e-6);
    sample_count(snr_index) = numel(indices);
    for first_index = 1:batch_size:numel(indices)
        batch_indices = indices(first_index:min(first_index+batch_size-1, numel(indices)));
        Y = dlarray(data.Y(:, :, batch_indices));
        S = data.S(:, :, batch_indices);
        B = dlarray(data.B(:, :, batch_indices));
        sigma2 = dlarray(reshape(data.sigma2(batch_indices), 1, 1, []));
        G0 = dlarray(data.G0(:, :, batch_indices));
        A = data.A(:, :, batch_indices);
        G_hat = unrolled_forward(parameters, Y, S, B, sigma2, G0, A, N, K, number_of_layers);
        G_hat = extractdata(G_hat);
        G_true = data.G(:, :, batch_indices);
        G_GS = data.G0(:, :, batch_indices);
        total_dl_error(snr_index) = total_dl_error(snr_index) + sum(abs(G_hat - G_true).^2, 'all');
        total_gs_error(snr_index) = total_gs_error(snr_index) + sum(abs(G_GS - G_true).^2, 'all');
        total_channel_energy(snr_index) = total_channel_energy(snr_index) + sum(abs(G_true).^2, 'all');
    end
end
GS_NMSE_dB = 10 * log10(max(total_gs_error ./ total_channel_energy, 1e-12));
DL_NMSE_dB = 10 * log10(max(total_dl_error ./ total_channel_energy, 1e-12));
DL_minus_GS_dB = DL_NMSE_dB - GS_NMSE_dB;
AggregateDL_NMSE_dB = 10 * log10(sum(total_dl_error) / sum(total_channel_energy));
metrics = table(repmat(P, numel(SNR_grid), 1), SNR_grid(:), sample_count, ...
    GS_NMSE_dB, DL_NMSE_dB, DL_minus_GS_dB, ...
    repmat(AggregateDL_NMSE_dB, numel(SNR_grid), 1), ...
    'VariableNames', {'P','SNR_dB','Samples','GS_NMSE_dB','DL_NMSE_dB', ...
    'DL_minus_GS_dB','AggregateDL_NMSE_dB'});
end

function save_model_atomic(model_file, parameters, model_configuration, best_epoch, best_validation_score, best_validation_metrics)
temporary_file = [model_file '.tmp.mat'];
save(temporary_file, 'parameters', 'model_configuration', 'best_epoch', ...
    'best_validation_score', 'best_validation_metrics', '-v7.3');
movefile(temporary_file, model_file, 'f');
end

function plot_validation_summary(result_table, P_list, SNR_grid, output_folder)
figure_handle = figure('Color','w','Name','V25 unrolled EM-GS validation');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
for p_index = 1:numel(P_list)
    nexttile; hold on;
    selected = result_table.P == P_list(p_index);
    plot(SNR_grid, result_table.GS_NMSE_dB(selected), '-o', 'LineWidth', 1.5, ...
        'DisplayName', 'Biased GS anchor');
    plot(SNR_grid, result_table.DL_NMSE_dB(selected), '-s', 'LineWidth', 1.5, ...
        'DisplayName', 'V25 unrolled EM-GS');
    grid on; xlabel('SNR (dB)'); ylabel('Validation NMSE (dB)');
    title(sprintf('N=8, K=3, P=%d, RSR=25 dB', P_list(p_index)));
    if p_index == 1, legend('Location','best'); end
end
sgtitle('V25 model-driven DL: training and V23 validation only');
saveas(figure_handle, fullfile(output_folder, 'validation_GS_vs_V25.png'));
savefig(figure_handle, fullfile(output_folder, 'validation_GS_vs_V25.fig'));
end
