%% Validation-only step-size audit for the 1D linearized magnitude GD update
% New standalone diagnostic. It does not edit or call existing project files.
% It loads only the N=8,K=3 train/validation release's validation_P*.mat
% files, and never searches for or loads a test split.
%
% The code compares three scalar multiples of the existing 1/lambda_max(S*S')
% GD step. Every candidate uses the same per-sample initialization. Validation
% NMSE and convergence are diagnostic outputs; freeze any choice before a new
% independent test is generated. This is not a verbatim reproduction of the
% paper until the paper's gradient convention is checked against its equations.

clear;
clc;
close all;

%% -------------------- FIXED VALIDATION-ONLY CONFIGURATION --------------------
N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
validation_samples_per_SNR = 25;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
step_scale_candidates = [0.5 1 2];
random_seed_base = 2026111001;

%% -------------------- LOCAL INPUT AND NEW OUTPUT PATHS --------------------
script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
repository_folder = fileparts(case_studies_folder);
scenario_folder = fullfile(repository_folder, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
output_folder = fullfile(script_folder, 'results', 'XU_1D_GD_STEP_AUDIT_V1');

if ~isfolder(scenario_folder)
    error('Expected local scenario folder is missing: %s', scenario_folder);
end
for P = P_list
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    if ~isfile(validation_file)
        error('Validation split missing for P=%d: %s', P, validation_file);
    end
end
if isfolder(output_folder)
    existing_entries = dir(output_folder);
    existing_names = string({existing_entries.name});
    existing_names = existing_names(~ismember(existing_names, ["." ".."]));
    if ~isempty(existing_names)
        error('Audit output folder contains files. Inspect them and choose a new folder name: %s', output_folder);
    end
    fprintf('Using existing empty output folder left by an interrupted run.\n');
else
    mkdir(output_folder);
end

fprintf('Validation-only 1D linearized-GD step-size audit.\n');
fprintf('N=%d, K=%d, P=%s, RSR=%g dB, SNR=%s dB.\n', N, K, mat2str(P_list), RSR_dB, mat2str(SNR_grid));
fprintf('Samples per P/SNR=%d; candidate step factors=%s; maximum updates=%d.\n', validation_samples_per_SNR, mat2str(step_scale_candidates), maximum_updates);
fprintf('Only validation_P*.mat files will be loaded. No test data are used.\n');
fprintf('New outputs: %s\n', output_folder);

%% -------------------- SCORE THE GD STEP-SIZE CANDIDATES --------------------
summary_rows = zeros(numel(P_list)*numel(SNR_grid)*numel(step_scale_candidates), 9);
sample_rows = zeros(numel(P_list)*numel(SNR_grid)*validation_samples_per_SNR*numel(step_scale_candidates), 11);
summary_row = 0;
sample_row = 0;

for P_index = 1:numel(P_list)
    P = P_list(P_index);
    validation_file = fullfile(scenario_folder, sprintf('validation_P%d.mat', P));
    validation_data = load(validation_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'SNR_dB_data', 'N', 'K', 'P', 'RSR_dB', 'SNR_dB', 'split_name', 'data_version');

    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || validation_data.RSR_dB ~= RSR_dB || ~strcmpi(validation_data.split_name, 'validation') || ~isequal(double(validation_data.SNR_dB), double(SNR_grid))
        error('Validation metadata mismatch in %s.', validation_file);
    end

    sample_SNR = double(validation_data.SNR_dB_data(:));
    for SNR_index = 1:numel(SNR_grid)
        current_SNR_dB = SNR_grid(SNR_index);
        available_indices = find(sample_SNR == current_SNR_dB);
        if numel(available_indices) < validation_samples_per_SNR
            error('Need %d validation samples at P=%d, SNR=%g dB; found %d.', validation_samples_per_SNR, P, current_SNR_dB, numel(available_indices));
        end

        selection_seed = random_seed_base+1000*P+SNR_index;
        rng(selection_seed, 'twister');
        selected_indices = available_indices(randperm(numel(available_indices), validation_samples_per_SNR));
        fprintf('\nP=%d, SNR=%g dB: scoring %d matched validation samples.\n', P, current_SNR_dB, numel(selected_indices));

        candidate_error_energy = zeros(numel(step_scale_candidates), 1);
        candidate_true_energy = zeros(numel(step_scale_candidates), 1);
        candidate_linear_fit_energy = zeros(numel(step_scale_candidates), 1);
        candidate_magnitude_fit_energy = zeros(numel(step_scale_candidates), 1);
        candidate_updates = zeros(numel(step_scale_candidates), 1);
        candidate_convergence_count = zeros(numel(step_scale_candidates), 1);

        for selected_index = 1:numel(selected_indices)
            sample_index = selected_indices(selected_index);
            Y = double(validation_data.Y_data(:, :, sample_index));
            G_true = double(validation_data.G_data(:, :, sample_index));
            S = double(validation_data.S_data(:, :, sample_index));
            B = double(validation_data.B_data(:, :, sample_index));
            if isequal(size(Y), [P N])
                Y = Y';
            end
            if isequal(size(G_true), [K N])
                G_true = G_true';
            end
            if isequal(size(S), [P K])
                S = S';
            end
            if isequal(size(B), [P N])
                B = B';
            end
            if ~isequal(size(Y), [N P]) || ~isequal(size(G_true), [N K]) || ~isequal(size(S), [K P]) || ~isequal(size(B), [N P])
                error('Array dimensions mismatch in validation file %s, sample %d: Y=%s, G=%s, S=%s, B=%s; expected [%d %d], [%d %d], [%d %d], [%d %d].', validation_file, sample_index, mat2str(size(Y)), mat2str(size(G_true)), mat2str(size(S)), mat2str(size(B)), N, P, N, K, K, P, N, P);
            end
            phase_reference = exp(-1i*angle(B));
            lipschitz_scale = max(real(eig(S*S')), eps);

            initialization_seed = random_seed_base+100000*P+100* SNR_index+sample_index;
            rng(initialization_seed, 'twister');
            initial_estimate = sqrt(gd_initial_variance/2)*(randn(N, K)+1i*randn(N, K));

            if SNR_index == 1 && selected_index == 1
                rng(random_seed_base+900000+P, 'twister');
                direction = randn(N, K)+1i*randn(N, K);
                objective_scale = max(sum(Y.^2, 'all'), 1);
                probe_residual = Y-abs(B)-real(phase_reference.*(initial_estimate*S));
                negative_gradient_direction = local_negative_gradient_direction(probe_residual, phase_reference, S, N, K);
                probe_step = 1e-3/max(norm(direction, 'fro'), 1);
                objective_plus = local_linearized_objective(initial_estimate+probe_step*direction, Y, S, B, phase_reference)/objective_scale;
                objective_minus = local_linearized_objective(initial_estimate-probe_step*direction, Y, S, B, phase_reference)/objective_scale;
                finite_difference = (objective_plus-objective_minus)/(2*probe_step);
                analytic_derivative = -real(sum(conj(negative_gradient_direction).*direction, 'all'))/objective_scale;
                derivative_relative_error = abs(finite_difference-analytic_derivative)/max([abs(finite_difference) abs(analytic_derivative) eps]);
                fprintf('Gradient check P=%d at SNR=%g dB: central difference=%+.6e, analytic=%+.6e, relative error=%.3e.\n', P, current_SNR_dB, finite_difference, analytic_derivative, derivative_relative_error);
            end

            for candidate_index = 1:numel(step_scale_candidates)
                step_size = step_scale_candidates(candidate_index)/lipschitz_scale;
                G_estimate = initial_estimate;
                converged = false;
                update_count = maximum_updates;

                for update_index = 1:maximum_updates
                    previous_estimate = G_estimate;
                    linearized_residual = Y-abs(B)-real(phase_reference.*(G_estimate*S));
                    gradient_direction = local_negative_gradient_direction(linearized_residual, phase_reference, S, N, K);
                    G_estimate = G_estimate+step_size.*gradient_direction;
                    relative_change = norm(G_estimate-previous_estimate, 'fro')^2/max(norm(G_estimate, 'fro')^2, eps);
                    if relative_change < convergence_tolerance
                        converged = true;
                        update_count = update_index;
                        break;
                    end
                end

                linearized_residual = Y-abs(B)-real(phase_reference.*(G_estimate*S));
                magnitude_residual = Y-abs(G_estimate*S+B);
                error_energy = sum(abs(G_estimate-G_true).^2, 'all');
                true_energy = sum(abs(G_true).^2, 'all');
                linear_fit_energy = sum(abs(linearized_residual).^2, 'all');
                magnitude_fit_energy = sum(abs(magnitude_residual).^2, 'all');

                candidate_error_energy(candidate_index) = candidate_error_energy(candidate_index)+error_energy;
                candidate_true_energy(candidate_index) = candidate_true_energy(candidate_index)+true_energy;
                candidate_linear_fit_energy(candidate_index) = candidate_linear_fit_energy(candidate_index)+linear_fit_energy;
                candidate_magnitude_fit_energy(candidate_index) = candidate_magnitude_fit_energy(candidate_index)+magnitude_fit_energy;
                candidate_updates(candidate_index) = candidate_updates(candidate_index)+update_count;
                candidate_convergence_count(candidate_index) = candidate_convergence_count(candidate_index)+double(converged);

                sample_row = sample_row+1;
                sample_rows(sample_row, :) = [N K P current_SNR_dB selected_index step_scale_candidates(candidate_index) 10*log10(max(error_energy/max(true_energy, eps), realmin)) update_count double(converged) linear_fit_energy magnitude_fit_energy];
            end
        end

        for candidate_index = 1:numel(step_scale_candidates)
            summary_row = summary_row+1;
            nmse_linear = candidate_error_energy(candidate_index)/max(candidate_true_energy(candidate_index), eps);
            convergence_fraction = candidate_convergence_count(candidate_index)/validation_samples_per_SNR;
            summary_rows(summary_row, :) = [N K P current_SNR_dB step_scale_candidates(candidate_index) nmse_linear 10*log10(max(nmse_linear, realmin)) mean(candidate_updates(candidate_index)/validation_samples_per_SNR) convergence_fraction];
            fprintf('P=%d SNR=%g factor=%g: pooled NMSE=%+.3f dB, mean updates=%.1f, converged=%.1f%%.\n', P, current_SNR_dB, step_scale_candidates(candidate_index), summary_rows(summary_row, 7), summary_rows(summary_row, 8), 100*summary_rows(summary_row, 9));
        end
    end
end

%% -------------------- SAVE DIAGNOSTIC OUTPUTS --------------------
summary_table = array2table(summary_rows, 'VariableNames', {'N','K','P','SNR_dB','StepScale','PooledNMSE_linear','PooledNMSE_dB','MeanUpdates','ConvergedFraction'});
sample_table = array2table(sample_rows, 'VariableNames', {'N','K','P','SNR_dB','ValidationSubsetIndex','StepScale','SampleNMSE_dB','Updates','Converged','LinearizedFitEnergy','ExactMagnitudeFitEnergy'});
summary_csv = fullfile(output_folder, 'xu_1d_gd_validation_step_summary.csv');
sample_csv = fullfile(output_folder, 'xu_1d_gd_validation_step_samples.csv');
results_mat = fullfile(output_folder, 'xu_1d_gd_validation_step_audit.mat');
writetable(summary_table, summary_csv);
writetable(sample_table, sample_csv);
configuration = struct('N', N, 'K', K, 'P_list', P_list, 'SNR_grid', SNR_grid, 'RSR_dB', RSR_dB, 'validation_samples_per_SNR', validation_samples_per_SNR, 'maximum_updates', maximum_updates, 'convergence_tolerance', convergence_tolerance, 'gd_initial_variance', gd_initial_variance, 'step_scale_candidates', step_scale_candidates, 'random_seed_base', random_seed_base, 'validation_folder', scenario_folder);
save(results_mat, 'summary_table', 'sample_table', 'configuration', '-v7.3');

fprintf('\nValidation-only GD step-size audit complete.\n');
fprintf('Summary CSV: %s\n', summary_csv);
fprintf('Per-sample CSV: %s\n', sample_csv);
fprintf('MAT results: %s\n', results_mat);
fprintf('No test data were loaded, scored, or modified. No existing project file was edited.\n');

function objective = local_linearized_objective(G, Y, S, B, phase_reference)
residual = Y-abs(B)-real(phase_reference.*(G*S));
objective = 0.5*sum(residual.^2, 'all');
end

function gradient_direction = local_negative_gradient_direction(residual, phase_reference, S, N, K)
weighted_residual = residual.*conj(phase_reference);
gradient_direction = zeros(N, K);
for user_index = 1:K
    pilot_weights = repmat(conj(S(user_index, :)), N, 1);
    gradient_direction(:, user_index) = sum(weighted_residual.*pilot_weights, 2);
end
end
