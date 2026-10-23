%% train_unfolded_dl_RSR25dB.m
% Train exact-magnitude, noise-aware unfolded DL models for P = 10 and P = 30.

clear;
clc;
close all;

% Locate all input and output folders relative to this script itself.
script_folder = fileparts(mfilename('fullpath'));
dataset_folder = fullfile(script_folder, 'DL_dataset_RSR25dB');
output_folder = fullfile(script_folder, 'unfolded_dl_results_RSR25dB');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

P_list = [10 30];
SNR_dB = -5:5:30;
number_of_layers = 30;
number_of_epochs = 20;
mini_batch_size = 256;
learning_rate = 0.01;

summary_table = table;

fprintf('\n============================================================\n');
fprintf(' EXACT-MAGNITUDE NOISE-AWARE UNFOLDED DL TRAINING\n');
fprintf('============================================================\n');
fprintf('Pilot counts                  = [%s]\n', num2str(P_list));
fprintf('SNR values                    = [%s] dB\n', num2str(SNR_dB));
fprintf('Layers                        = %d\n', number_of_layers);
fprintf('Epochs                        = %d\n', number_of_epochs);
fprintf('Mini-batch size               = %d\n', mini_batch_size);
fprintf('Learning rate                 = %.4g\n', learning_rate);
fprintf('Test files are not loaded by this script.\n');
fprintf('============================================================\n');

for P_index = 1:length(P_list)
    P = P_list(P_index);
    fprintf('\n============================================================\n');
    fprintf('Preparing exact-magnitude unfolded DL for P = %d\n', P);
    fprintf('============================================================\n');

    % Select the training and validation files corresponding to this P.
    % The test files are deliberately not loaded during model training.
    train_file = fullfile(dataset_folder, sprintf('train_P%d.mat', P));
    validation_file = fullfile(dataset_folder, sprintf('validation_P%d.mat', P));
    train_source = load(train_file);
    validation_source = load(validation_file);

    % This scale is estimated from training labels only and then frozen.
    % It keeps the physical channel values near unit scale for stable ADAM updates.
    training_channel_scale = sqrt(mean(abs(double(train_source.G_data(:))).^2));
    training_channel_scale = max(training_channel_scale, sqrt(eps));

    % Convert the raw saved arrays into normalized fields used by the model.
    % The same scale is used for both sets so validation remains independent.
    train_data = pack_dataset(train_source, training_channel_scale);
    validation_data = pack_dataset(validation_source, training_channel_scale);
    number_of_training_samples = size(train_data.target, 3);
    number_of_validation_samples = size(validation_data.target, 3);

    fprintf('Training samples               = %d\n', number_of_training_samples);
    fprintf('Validation samples             = %d\n', number_of_validation_samples);
    fprintf('Frozen training channel scale  = %.6g\n', training_channel_scale);

    % Five learned scalars per layer:
    % base step, base momentum, raw ridge weight, noise-step slope,
    % noise-momentum slope. The sigmoid/softplus parameterizations keep
    % the update schedule bounded and the ridge weight positive.
    initial_parameters = dlarray(single([ ...
        zeros(number_of_layers, 1), ...
        -ones(number_of_layers, 1), ...
        -3 * ones(number_of_layers, 1), ...
        zeros(number_of_layers, 1), ...
        zeros(number_of_layers, 1)]));
    parameters = initial_parameters;
    % Adam keeps running estimates of the first and second gradient moments.
    % These are initialized empty and filled during the first update.
    average_gradient = [];
    average_squared_gradient = [];
    iteration = 0;

    initial_validation_metrics = evaluate_model(parameters, validation_data, mini_batch_size);
    best_validation_nmse = initial_validation_metrics(1);
    best_parameters = parameters;
    history = zeros(number_of_epochs, 5);

    fprintf('Initial validation NMSE = %.4f dB\n', 10 * log10(best_validation_nmse));
    training_timer = tic;

    for epoch = 1:number_of_epochs
        % Shuffle the training samples at the start of every epoch.
        % The seed makes the order reproducible for this P and epoch.
        rng(2026102100 + 100 * P + epoch, 'twister');
        random_order = randperm(number_of_training_samples);
        accumulated_loss = 0;
        number_of_batches = 0;

        for first_index = 1:mini_batch_size:number_of_training_samples
            % Select one mini-batch. The final batch may contain fewer samples.
            batch_indices = random_order(first_index:min(first_index + mini_batch_size - 1, number_of_training_samples));
            batch = take_batch(train_data, batch_indices);
            iteration = iteration + 1;

            % Evaluate the forward model, calculate the supervised loss,
            % and differentiate that loss with respect to all learned scalars.
            [loss, gradients] = dlfeval(@model_loss, parameters, batch);
            if ~all(isfinite(extractdata(gradients)), 'all')
                error('Nonfinite unfolded-DL gradients detected at P=%d, epoch=%d.', P, epoch);
            end

            % Update the trainable unfolded-layer parameters using Adam.
            [parameters, average_gradient, average_squared_gradient] = adamupdate( ...
                parameters, gradients, average_gradient, average_squared_gradient, ...
                iteration, learning_rate);
            accumulated_loss = accumulated_loss + double(extractdata(loss));
            number_of_batches = number_of_batches + 1;
        end

        % Measure both sets after the epoch. Validation is not used to compute
        % gradients; it is used only to choose the best saved parameter set.
        training_metrics = evaluate_model(parameters, train_data, mini_batch_size);
        validation_metrics = evaluate_model(parameters, validation_data, mini_batch_size);
        validation_nmse = validation_metrics(1);

        % Keep the parameters from the epoch with the lowest validation NMSE.
        if validation_nmse < best_validation_nmse
            best_validation_nmse = validation_nmse;
            best_parameters = parameters;
            best_epoch = epoch;
        end

        history(epoch, :) = [epoch, accumulated_loss / number_of_batches, ...
            10 * log10(training_metrics(1)), 10 * log10(validation_nmse), ...
            validation_metrics(2)];
        fprintf('P = %d, epoch %2d/%d, train NMSE = %8.3f dB, validation NMSE = %8.3f dB\n', ...
            P, epoch, number_of_epochs, history(epoch, 3), history(epoch, 4));
    end

    training_seconds = toc(training_timer);
    if ~exist('best_epoch', 'var')
        best_epoch = 0;
    end

    final_training_metrics = evaluate_model(best_parameters, train_data, mini_batch_size);
    final_validation_metrics = evaluate_model(best_parameters, validation_data, mini_batch_size);
    row = table(P, training_seconds, best_epoch, ...
        10 * log10(final_training_metrics(1)), ...
        10 * log10(final_validation_metrics(1)), ...
        final_validation_metrics(2), training_channel_scale, ...
        'VariableNames', {'P', 'TrainingSeconds', 'BestEpoch', ...
        'TrainNMSE_dB', 'ValidationNMSE_dB', 'ValidationEnergyRatio', ...
        'TrainingChannelScale'});
    summary_table = [summary_table; row]; 
    disp(row);

    % Evaluate the selected frozen model separately at every validation SNR.
    validation_by_snr = evaluate_by_snr(best_parameters, validation_data, SNR_dB, mini_batch_size);
    model_file = fullfile(output_folder, sprintf('model_P%d.mat', P));
    % Save the learned parameters together with the settings needed to reuse
    % the model later in the final four-method test comparison.
    save(model_file, 'best_parameters', 'initial_parameters', 'history', 'row', ...
        'validation_by_snr', 'number_of_layers', 'number_of_epochs', ...
        'mini_batch_size', 'learning_rate', 'training_channel_scale', ...
        'P', 'SNR_dB', '-v7.3');
    writetable(validation_by_snr, fullfile(output_folder, sprintf('validation_by_snr_P%d.csv', P)));
    writetable(summary_table, fullfile(output_folder, 'summary.csv'));

    fprintf('Saved P = %d unfolded model and validation history.\n', P);
    clear train_source validation_source train_data validation_data
    clear best_epoch
end

figure('Color', 'w', 'Position', [100 100 1000 650]);
% Plot validation performance only after both P-specific models are saved.
hold on;
grid on;
box on;
for P_index = 1:length(P_list)
    P = P_list(P_index);
    saved_model = load(fullfile(output_folder, sprintf('model_P%d.mat', P)), ...
        'validation_by_snr');
    if P_index == 1
        line_style = '--';
    else
        line_style = '-';
    end
    plot(saved_model.validation_by_snr.SNR_dB, ...
        saved_model.validation_by_snr.ValidationNMSE_dB, ...
        'LineStyle', line_style, 'Marker', 'o', 'LineWidth', 2, ...
        'DisplayName', sprintf('Unfolded DL, P = %d', P));
end
xlabel('SNR (dB)');
ylabel('Validation NMSE (dB)');
title('Exact-magnitude noise-aware unfolded DL validation, RSR = 25 dB');
legend('Location', 'best');
xticks(SNR_dB);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
set(gcf, 'ToolBar', 'none', 'MenuBar', 'none');
try
    axtoolbar(gca, {});
    disableDefaultInteractivity(gca);
catch
end
exportgraphics(gcf, fullfile(output_folder, 'validation_nmse_vs_snr.png'), 'Resolution', 300);
savefig(gcf, fullfile(output_folder, 'validation_nmse_vs_snr.fig'));

fprintf('\nUnfolded DL training and validation completed.\n');
fprintf('Test files were not loaded.\n');
disp(summary_table);

%% ========================================================================
%% DATA PACKING AND BATCH FUNCTIONS
function data = pack_dataset(source, channel_scale)
% Normalize the physical arrays and split complex quantities into real and
% imaginary parts so that they can be processed by the unfolded model.
Y = single(source.Y_data) / single(channel_scale);
G = single(source.G_data) / single(channel_scale);
S = single(source.S_data);
B = single(source.B_data) / single(channel_scale);
sigma2 = single(source.sigma2_data(:)) / single(channel_scale^2);

N = size(G, 1);
K = size(G, 2);
P = size(S, 2);
number_of_samples = size(G, 3);

data.sr = zeros(K, P, number_of_samples, 'single');
data.si = zeros(K, P, number_of_samples, 'single');
data.br = real(B);
data.bi = imag(B);
data.y = Y;
data.bound = zeros(1, 1, number_of_samples, 'single');
data.noise_ratio = reshape(sigma2, 1, 1, number_of_samples);
data.target = zeros(2 * N, K, number_of_samples, 'single');
data.snr = single(source.SNR_dB_data(:));

data.sr = real(S);
data.si = imag(S);
data.target(1:N, :, :) = real(G);
data.target(N + 1:2 * N, :, :) = imag(G);

for sample_index = 1:number_of_samples
    % The largest eigenvalue of S*S^H is a per-sample pilot conditioning
    % bound used to normalize the gradient update inside the DL layers.
    pilot_gram = double(S(:, :, sample_index) * S(:, :, sample_index)');
    data.bound(1, 1, sample_index) = single(max(real(eig(pilot_gram))));
end
end

function batch = take_batch(data, indices)
% Extract selected samples and convert them to dlarray objects for automatic
% differentiation. The target is flattened to match the model output shape.
batch.sr = dlarray(data.sr(:, :, indices));
batch.si = dlarray(data.si(:, :, indices));
batch.br = dlarray(data.br(:, :, indices));
batch.bi = dlarray(data.bi(:, :, indices));
batch.y = dlarray(data.y(:, :, indices));
batch.bound = dlarray(data.bound(:, :, indices));
batch.noise_ratio = dlarray(data.noise_ratio(:, :, indices));
    selected_target = data.target(:, :, indices);
    batch.target = dlarray(reshape(selected_target, [], numel(indices)));
end

%% ========================================================================
%% EXACT-MAGNITUDE UNFOLDED MODEL
function [loss, gradients] = model_loss(parameters, batch)
% The supervised target is the normalized true channel. Mean squared error
% is used to train the real and imaginary channel components jointly.
prediction = forward_model(parameters, batch);
loss = mean((prediction - batch.target).^2, 'all');
gradients = dlgradient(loss, parameters);
end

function prediction = forward_model(parameters, batch)
% Run the 30 unfolded physics-based channel-update layers. No true channel
% is used here; only measurements, pilots, reference, and noise metadata are
% available to the forward model.
N = size(batch.y, 1);
K = size(batch.sr, 1);
number_of_samples = size(batch.y, 3);

real_channel = dlarray(zeros(N, K, number_of_samples, 'single'));
imag_channel = dlarray(zeros(N, K, number_of_samples, 'single'));
previous_real_channel = real_channel;
previous_imag_channel = imag_channel;

noise_level = log(1 + batch.noise_ratio ./ max(batch.bound, single(1e-6)));

for layer_index = 1:size(parameters, 1)
    % Convert the learned raw parameters into a bounded step size and
    % momentum, and a positive noise-aware ridge weight.
    step_size = 1.9 ./ (1 + exp(-(parameters(layer_index, 1) + ...
        parameters(layer_index, 4) .* noise_level)));
    momentum = 0.95 ./ (1 + exp(-(parameters(layer_index, 2) + ...
        parameters(layer_index, 5) .* noise_level)));
    ridge_weight = log(1 + exp(parameters(layer_index, 3)));

    extrapolated_real = real_channel + momentum .* ...
        (real_channel - previous_real_channel);
    extrapolated_imag = imag_channel + momentum .* ...
        (imag_channel - previous_imag_channel);

    % Reconstruct the predicted complex field G*S+B from separate real and
    % imaginary channel, pilot, and reference components.
    field_real = pagemtimes(extrapolated_real, batch.sr) - ...
        pagemtimes(extrapolated_imag, batch.si) + batch.br;
    field_imag = pagemtimes(extrapolated_real, batch.si) + ...
        pagemtimes(extrapolated_imag, batch.sr) + batch.bi;
    % Apply the exact magnitude model and compare the predicted amplitude with
    % the measured magnitude. The small term prevents division by zero.
    amplitude = sqrt(field_real.^2 + field_imag.^2 + single(1e-12));
    residual_factor = (amplitude - batch.y) ./ amplitude;
    error_real = residual_factor .* field_real;
    error_imag = residual_factor .* field_imag;

    pilot_real_transpose = permute(batch.sr, [2 1 3]);
    pilot_imag_transpose = permute(batch.si, [2 1 3]);
    % Back-propagate the magnitude residual through the known pilot matrix to
    % obtain real and imaginary channel-update directions.
    gradient_real = pagemtimes(error_real, pilot_real_transpose) + ...
        pagemtimes(error_imag, pilot_imag_transpose);
    gradient_imag = -pagemtimes(error_real, pilot_imag_transpose) + ...
        pagemtimes(error_imag, pilot_real_transpose);

    previous_real_channel = real_channel;
    previous_imag_channel = imag_channel;
    % Normalize the gradient using the pilot bound and apply noise-aware
    % ridge regularization before completing this unfolded layer update.
    denominator = 1 + step_size .* ridge_weight .* ...
        batch.noise_ratio ./ max(batch.bound, single(1e-6));
    real_channel = (extrapolated_real - step_size .* gradient_real ./ ...
        max(batch.bound, single(1e-6))) ./ denominator;
    imag_channel = (extrapolated_imag - step_size .* gradient_imag ./ ...
        max(batch.bound, single(1e-6))) ./ denominator;
end

prediction = cat(1, real_channel, imag_channel);
prediction = reshape(prediction, 2 * N * K, number_of_samples);
end

%% ========================================================================
%% VALIDATION METRICS
%% ========================================================================
function metrics = evaluate_model(parameters, data, batch_size)
% Compute ratio-of-sums NMSE and estimated-to-target channel energy ratio
% without changing the learned parameters.
error_energy = 0;
target_energy = 0;
prediction_energy = 0;
number_of_samples = size(data.target, 3);

for first_index = 1:batch_size:number_of_samples
    indices = first_index:min(first_index + batch_size - 1, number_of_samples);
    batch = take_batch(data, indices);
    prediction = double(extractdata(forward_model(parameters, batch)));
    target = double(extractdata(batch.target));
    error_energy = error_energy + sum((prediction - target).^2, 'all');
    target_energy = target_energy + sum(target.^2, 'all');
    prediction_energy = prediction_energy + sum(prediction.^2, 'all');
end

metrics = [error_energy / target_energy, prediction_energy / target_energy];
end

function result_table = evaluate_by_snr(parameters, data, SNR_grid_dB, batch_size)
% Group validation samples by their stored SNR and calculate one NMSE value
% for each SNR point for plotting and reporting.
result_table = table;
for SNR_index = 1:length(SNR_grid_dB)
    selected = find(data.snr == SNR_grid_dB(SNR_index));
    error_energy = 0;
    target_energy = 0;
    for first_index = 1:batch_size:length(selected)
        indices = selected(first_index:min(first_index + batch_size - 1, length(selected)));
        batch = take_batch(data, indices);
        prediction = double(extractdata(forward_model(parameters, batch)));
        target = double(extractdata(batch.target));
        error_energy = error_energy + sum((prediction - target).^2, 'all');
        target_energy = target_energy + sum(target.^2, 'all');
    end
    nmse_dB = 10 * log10(error_energy / target_energy);
    result_table = [result_table; table(SNR_grid_dB(SNR_index), nmse_dB, ...
        'VariableNames', {'SNR_dB', 'ValidationNMSE_dB'})]; %#ok<AGROW>
end
end
