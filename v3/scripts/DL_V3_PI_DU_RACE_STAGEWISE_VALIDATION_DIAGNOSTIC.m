%% V3 PI-DU-RACE stage-wise held-out validation diagnostic
% Measures NMSE at the linear initialization and after each EM-GS, Hankel,
% and damping operation. Validation checkpoint-selection samples are excluded.
% Training and reserved test files are never loaded.

clear;
clc;
close all;

%% -------------------- FIXED V3 SCENARIO --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
number_of_stages = 4;
validation_samples_per_SNR_for_checkpoint = 64;

script_folder = fileparts(mfilename('fullpath'));
v3_folder = fileparts(script_folder);
scenario_folder = fullfile(v3_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
model_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_EMGS_HANKEL_PARFOR_RSR25DB_RUN2');
output_folder = fullfile(v3_folder, 'results', 'N8_PI_DU_RACE_STAGEWISE_VALIDATION_DIAGNOSTIC_RSR25DB_RUN2');
if exist(output_folder, 'dir')
    error('Output folder already exists. Inspect it or choose a new output-folder name before running.');
end
if ~isfolder(model_folder) || ~isfolder(scenario_folder)
    error('Required V3 model or validation-data folder was not found.');
end
mkdir(output_folder);

fprintf('V3 PI-DU-RACE stage-wise validation diagnostic.\n');
fprintf('Validation samples used for checkpoint selection are excluded.\n');
fprintf('Training and reserved test files are not loaded.\n');

%% -------------------- HANKEL MAP --------------------
number_of_hankel_rows = N-p_H;
number_of_hankel_columns = p_H+1;
hankel_map = zeros(number_of_hankel_rows*number_of_hankel_columns, N);
for row_index = 1:number_of_hankel_rows
    for column_index = 1:number_of_hankel_columns
        linear_index = row_index+(column_index-1)*number_of_hankel_rows;
        antenna_index = row_index+column_index-1;
        hankel_map(linear_index, antenna_index) = 1;
    end
end
hankel_diagonal_counts = sum(hankel_map, 1).';

%% -------------------- EVALUATE HELD-OUT VALIDATION --------------------
number_of_states = 1+3*number_of_stages;
state_names = cell(1, number_of_states);
state_names{1} = 'LinearInit';
for stage_index = 1:number_of_stages
    first_column = 2+3*(stage_index-1);
    state_names{first_column} = sprintf('Stage%d_EMGS', stage_index);
    state_names{first_column+1} = sprintf('Stage%d_Hankel', stage_index);
    state_names{first_column+2} = sprintf('Stage%d_Damped', stage_index);
end
result_rows = zeros(numel(P_list)*numel(SNR_grid), 3+number_of_states);
row_index = 0;

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    model_file = fullfile(model_folder, sprintf('selected_model_P%d.mat', P));
    if ~isfile(validation_file) || ~isfile(model_file)
        error('Missing validation data or selected model for P=%d.', P);
    end
    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'AoA_deg_range', 'data_version');
    model_data = load(model_file, 'best_parameters_numeric', 'best_epoch_saved', 'P', 'N', 'K', 'RSR_dB', 'p_H', 'number_of_stages');
    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ~isequal(double(validation_data.AoA_deg_range), [-90 90]) || ~strcmp(validation_data.data_version, 'V3_AoA_Uminus90to90')
        error('Validation data metadata does not match V3 settings for P=%d.', P);
    end
    if model_data.N ~= N || model_data.K ~= K || model_data.P ~= P || model_data.RSR_dB ~= RSR_dB || model_data.p_H ~= p_H || model_data.number_of_stages ~= number_of_stages
        error('Selected model metadata does not match V3 settings for P=%d.', P);
    end

    sample_count = size(validation_data.Y_data, 3);
    selected_parameters = model_data.best_parameters_numeric;
    confirmation_mask = false(sample_count, 1);
    for snr_index = 1:numel(SNR_grid)
        candidates = find(double(validation_data.SNR_dB_data) == SNR_grid(snr_index));
        if numel(candidates) <= validation_samples_per_SNR_for_checkpoint
            error('Not enough validation samples for P=%d at SNR=%g dB.', P, SNR_grid(snr_index));
        end
        selected_positions = round(linspace(1, numel(candidates), validation_samples_per_SNR_for_checkpoint));
        confirmation_mask(candidates) = true;
        confirmation_mask(candidates(selected_positions)) = false;
    end
    confirmation_indices = find(confirmation_mask);
    state_error = zeros(sample_count, number_of_states);
    true_energy = zeros(sample_count, 1);

    fprintf('\nP=%d; selected epoch=%d; held-out validation samples=%d.\n', P, model_data.best_epoch_saved, numel(confirmation_indices));
    for sample_position = 1:numel(confirmation_indices)
        sample_index = confirmation_indices(sample_position);
        Y = double(validation_data.Y_data(:, :, sample_index));
        G_true = double(validation_data.G_data(:, :, sample_index));
        S = double(validation_data.S_data(:, :, sample_index));
        B = double(validation_data.B_data(:, :, sample_index));
        sigma2 = double(validation_data.sigma2_data(sample_index));
        right_inverse = S'/(S*S');
        G_estimate = (Y-abs(B))*right_inverse;
        state_error(sample_index, 1) = norm(G_estimate-G_true, 'fro')^2;
        true_energy(sample_index) = norm(G_true, 'fro')^2;

        noise_scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
        for stage_index = 1:number_of_stages
            threshold_multiplier = 2/(1+exp(-selected_parameters(1, stage_index)));
            damping = 1/(1+exp(-selected_parameters(2, stage_index)));
            threshold = threshold_multiplier*noise_scale;
            field = G_estimate*S+B;
            kappa = 2/max(sigma2, eps)*Y.*abs(field);
            bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
            G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
            state_column = 2+3*(stage_index-1);
            state_error(sample_index, state_column) = norm(G_emgs-G_true, 'fro')^2;

            G_hankel = zeros(N, K);
            for user_index = 1:K
                hankel_matrix = reshape(hankel_map*G_emgs(:, user_index), number_of_hankel_rows, number_of_hankel_columns);
                gram_matrix = hankel_matrix'*hankel_matrix;
                [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
                eigenvalues = real(sum(eigenvalue_matrix, 2));
                singular_values = sqrt(max(eigenvalues, 0));
                safe_singular_values = max(singular_values, 1e-12);
                shrink_factors = max(1-threshold./safe_singular_values, 0);
                scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
                hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
                G_hankel(:, user_index) = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
            end
            state_error(sample_index, state_column+1) = norm(G_hankel-G_true, 'fro')^2;
            G_estimate = damping*G_hankel+(1-damping)*G_emgs;
            state_error(sample_index, state_column+2) = norm(G_estimate-G_true, 'fro')^2;
        end
        if mod(sample_position, 200) == 0 || sample_position == numel(confirmation_indices)
            fprintf('P=%d samples completed: %d/%d.\n', P, sample_position, numel(confirmation_indices));
        end
    end

    for snr_index = 1:numel(SNR_grid)
        snr_indices = confirmation_indices(double(validation_data.SNR_dB_data(confirmation_indices)) == SNR_grid(snr_index));
        nmse_db = 10*log10(sum(state_error(snr_indices, :), 1)/sum(true_energy(snr_indices)));
        row_index = row_index+1;
        result_rows(row_index, :) = [P, SNR_grid(snr_index), numel(snr_indices), nmse_db];
    end
end

variable_names = [{'P','SNR_dB','ConfirmationSamples'}, state_names];
result_table = array2table(result_rows, 'VariableNames', variable_names);
csv_file = fullfile(output_folder, 'V3_PI_DU_RACE_stagewise_validation.csv');
mat_file = fullfile(output_folder, 'V3_PI_DU_RACE_stagewise_validation.mat');
writetable(result_table, csv_file);
save(mat_file, 'result_table', 'state_names', 'P_list', 'SNR_grid', 'validation_samples_per_SNR_for_checkpoint', '-v7.3');
disp(result_table);
fprintf('\nStage-wise validation diagnostic complete. No training or test files were loaded.\n');
fprintf('Results are in: %s\n', output_folder);
