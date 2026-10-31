%% Final paired four-method comparison on the reserved V23 test split
% Methods: biased GS, linearized GD, validation-frozen soft HS-GS, V24 DL.
% DL is the selected V24 GS-anchored L2 residual MLP expert for each P/SNR.
% The V36 soft-Hankel multiplier is validation-frozen and is not tuned here.
%
% GS, GD, and HS-GS use the V36 maximum-update and convergence-tolerance
% settings. The DL input retains the V24 GS anchor settings used in training.
%
% This script loads only V23 test_P10/test_P30, frozen V24 model files, and
% the V36 validation-frozen policy. It does not load training or validation
% observations, train a model, tune a policy, or modify source artifacts.
% Run only after confirming V24 and V36 settings are frozen:
%   cd('D:\MTech_Thesis_Results\v2');
%   FINAL_GS_GD_HS_GS_DL_V24_V36_RSR25dB

clear; clc; close all;

script_folder = fileparts(mfilename('fullpath'));
project_folder = fileparts(script_folder);
test_folder = fullfile(project_folder, 'GS-GS-HS-DL', 'DL_dataset_RSR25dB_V23_valtest');
v24_model_folder = fullfile(script_folder, 'DL_V24_L2_REGULARIZED_PER_SNR_EXPERT_MLP_RSR25dB');
v36_result_file = fullfile(script_folder, 'results', 'HS_GS_V36_N8_soft_results.mat');
output_folder = fullfile(script_folder, 'FINAL_GS_GD_HS_GS_DL_V24_V36_RSR25dB');

if ~exist(output_folder, 'dir'), mkdir(output_folder); end
result_file = fullfile(output_folder, 'final_four_method_test_results.mat');
table_file = fullfile(output_folder, 'final_four_method_test_table.csv');
figure_file = fullfile(output_folder, 'final_four_method_test_nmse.png');
if isfile(result_file) || isfile(table_file) || isfile(figure_file)
    error('Final comparison outputs already exist. Inspect them before choosing a new output folder.');
end
if ~isfile(v36_result_file)
    error('Missing V36 frozen HS-GS policy: %s', v36_result_file);
end

N = 8; K = 3; P_list = [10 30]; SNR_grid = -5:5:30; RSR_dB = 25;
V36 = load(v36_result_file, 'configuration', 'frozen_candidate_index');
configuration = V36.configuration;
if configuration.N ~= N || configuration.K ~= K || configuration.RSR_dB ~= RSR_dB || ...
        ~isequal(configuration.P_list, P_list) || ~isequal(configuration.SNR_dB, SNR_grid)
    error('V36 frozen policy does not match N=8, K=3, P=[10 30], SNR=-5:5:30, RSR=25 dB.');
end
if ~isfield(configuration, 'shrink_multiplier_list') || ...
        ~isequal(size(V36.frozen_candidate_index), [numel(P_list), numel(SNR_grid)])
    error('V36 frozen soft-Hankel policy has unexpected fields or dimensions.');
end

% Use the same convergence settings for the iterative baselines and frozen HS-GS.
maximum_updates = configuration.maximum_updates;
convergence_tolerance = configuration.convergence_tolerance;
p_H = configuration.p_H;
bootstrap_replicates = 2000;
gd_initial_variance = configuration.gd_initial_variance;
number_of_methods = 4; % GS, GD, soft HS-GS, selected V24 DL

fprintf('Paired final comparison on reserved V23 test observations.\n');
fprintf('Train/validation files: not loaded. Test files: loaded by this script when run.\n');
fprintf('N=%d, K=%d, RSR=%g dB, P=[%s], SNR=[%s] dB.\n', ...
    N, K, RSR_dB, num2str(P_list), num2str(SNR_grid));
fprintf('DL method: selected V24 per-SNR GS-anchored L2 residual MLP experts.\n');
fprintf('HS-GS method: V36 validation-frozen soft-Hankel thresholds.\n');
fprintf('GS, GD, and HS-GS stop at relative-change tolerance %.3g, maximum %d updates.\n', ...
    convergence_tolerance, maximum_updates);

all_results = cell(numel(P_list), 1);
result_rows = zeros(numel(P_list) * numel(SNR_grid), 24);
row_number = 0;

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    test_file = fullfile(test_folder, sprintf('test_P%d.mat', P));
    if ~isfile(test_file), error('Missing reserved test file: %s', test_file); end
    test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data');
    validate_test_data(test_data, N, K, P, SNR_grid);
    sample_count = size(test_data.Y_data, 3);
    fprintf('\nP=%d: loaded %d reserved test observations.\n', P, sample_count);

    method_error = nan(sample_count, number_of_methods);
    true_energy = nan(sample_count, 1);
    iteration_count = zeros(sample_count, number_of_methods);
    anchor_gs_iteration_count = zeros(sample_count, 1);
    gs_converged = false(sample_count, 1);
    gd_converged = false(sample_count, 1);
    hs_gs_converged = false(sample_count, 1);
    snr_values = double(test_data.SNR_dB_data(:));
    frozen_hs_multiplier = nan(numel(SNR_grid), 1);

    for snr_index = 1:numel(SNR_grid)
        snr_db = SNR_grid(snr_index);
        sample_indices = find(abs(snr_values - snr_db) < 1e-6);
        if isempty(sample_indices), error('No test observations for P=%d, SNR=%g.', P, snr_db); end

        selected_index = V36.frozen_candidate_index(p_index, snr_index);
        selected_multiplier = configuration.shrink_multiplier_list(selected_index);
        frozen_hs_multiplier(snr_index) = selected_multiplier;
        model_file = fullfile(v24_model_folder, sprintf('model_P%d_SNR_%d_best.mat', P, snr_db));
        if ~isfile(model_file), error('Missing selected V24 model: %s', model_file); end
        model = load(model_file, 'best_net', 'feature_mean', 'feature_std', 'model_config', ...
            'best_epoch', 'best_validation_score');
        if model.model_config.P ~= P || model.model_config.SNR_dB ~= snr_db || ...
                model.model_config.N ~= N || model.model_config.K ~= K
            error('V24 model configuration mismatch for P=%d, SNR=%g.', P, snr_db);
        end
        if ~isfield(model.model_config, 'training_channel_scale')
            error('V24 model is missing its training-only channel scale.');
        end
        channel_scale = double(model.model_config.training_channel_scale);
        fprintf('  SNR=%g dB: V24 selected epoch %d; frozen HS multiplier %.3g.\n', ...
            snr_db, model.best_epoch, selected_multiplier);

        for local_index = 1:numel(sample_indices)
            sample_index = sample_indices(local_index);
            Y = double(test_data.Y_data(:, :, sample_index));
            G_true = double(test_data.G_data(:, :, sample_index));
            S = double(test_data.S_data(:, :, sample_index));
            B = double(test_data.B_data(:, :, sample_index));
            sigma2 = double(test_data.sigma2_data(sample_index));

            % Score converged GS, while reproducing V24's training anchor for DL input.
            [G_gs, gs_steps, gs_ok] = final_gs_converged(Y, S, B, ...
                maximum_updates, convergence_tolerance);
            [G_gs_anchor, anchor_gs_steps, ~] = final_gs_converged(Y, S, B, ...
                model.model_config.GS_max_iterations, model.model_config.GS_tolerance);
            [G_gd, gd_steps, gd_ok] = final_gd_converged(Y, S, B, ...
                maximum_updates, convergence_tolerance, gd_initial_variance, ...
                910000000 + 1000000 * P + 10000 * snr_index + sample_index);
            if selected_multiplier == 0
                [G_hs, hs_steps, hs_ok] = final_em_converged(Y, S, B, sigma2, ...
                    maximum_updates, convergence_tolerance);
            else
                threshold_scale = final_singular_noise_scale(S, sigma2, N, p_H);
                [G_hs, hs_steps, hs_ok] = final_soft_em_converged(Y, S, B, sigma2, ...
                    p_H, selected_multiplier * threshold_scale, maximum_updates, convergence_tolerance);
            end

            feature = final_build_one_sample_features(Y, B, S, G_gs_anchor, sigma2, ...
                snr_db, channel_scale, N, K, P);
            normalized_feature = single((feature - model.feature_mean) ./ model.feature_std);
            correction = predict(model.best_net, dlarray(normalized_feature, 'CB'));
            correction = double(gather(extractdata(correction)));
            if ~isequal(size(correction), [2 * K, N])
                error('V24 prediction has unexpected size for P=%d, SNR=%g.', P, snr_db);
            end
            correction_channel = correction(1:K, :).'+1i*correction(K+1:2*K, :).';
            G_dl = G_gs_anchor + channel_scale * correction_channel;

            method_error(sample_index, :) = [norm(G_gs-G_true,'fro')^2, ...
                norm(G_gd-G_true,'fro')^2, norm(G_hs-G_true,'fro')^2, ...
                norm(G_dl-G_true,'fro')^2];
            true_energy(sample_index) = norm(G_true,'fro')^2;
            iteration_count(sample_index, :) = [gs_steps, gd_steps, hs_steps, anchor_gs_steps];
            anchor_gs_iteration_count(sample_index) = anchor_gs_steps;
            gs_converged(sample_index) = gs_ok;
            gd_converged(sample_index) = gd_ok;
            hs_gs_converged(sample_index) = hs_ok;
        end
    end

    if any(~isfinite(method_error), 'all') || any(~isfinite(true_energy)) || any(true_energy <= 0)
        error('Invalid or missing per-sample result for P=%d.', P);
    end
    all_results{p_index} = struct('P', P, 'method_error', method_error, ...
        'true_energy', true_energy, 'iteration_count', iteration_count, ...
        'anchor_gs_iteration_count', anchor_gs_iteration_count, ...
        'gs_converged', gs_converged, 'gd_converged', gd_converged, ...
        'hs_gs_converged', hs_gs_converged, 'SNR_dB_data', snr_values);

    for snr_index = 1:numel(SNR_grid)
        snr_db = SNR_grid(snr_index);
        sample_indices = find(abs(snr_values - snr_db) < 1e-6);
        errors = method_error(sample_indices, :);
        energy = true_energy(sample_indices);
        nmse_db = 10 * log10(sum(errors, 1) ./ sum(energy));
        % Negative method-minus-GS gain means that method improves on GS.
        paired_difference = nmse_db(2:4) - nmse_db(1);
        ci_low = nan(1, 3); ci_high = nan(1, 3);
        rng(720000 + P * 100 + snr_index, 'twister');
        for method_index = 2:4
            bootstrap_gain = zeros(bootstrap_replicates, 1);
            for draw = 1:bootstrap_replicates
                ids = randi(numel(sample_indices), numel(sample_indices), 1);
                bootstrap_gain(draw) = 10 * log10(sum(errors(ids, method_index)) / sum(errors(ids, 1)));
            end
            bootstrap_gain = sort(bootstrap_gain);
            ci_low(method_index-1) = bootstrap_gain(max(1, floor(0.025 * bootstrap_replicates)));
            ci_high(method_index-1) = bootstrap_gain(min(bootstrap_replicates, ceil(0.975 * bootstrap_replicates)));
        end
        row_number = row_number + 1;
        result_rows(row_number, :) = [P, snr_db, numel(sample_indices), frozen_hs_multiplier(snr_index), ...
            nmse_db, ...
            paired_difference, ci_low, ci_high, ...
            mean(iteration_count(sample_indices, 1)), mean(iteration_count(sample_indices, 2)), ...
            mean(iteration_count(sample_indices, 3)), mean(iteration_count(sample_indices, 4)), ...
            mean(gs_converged(sample_indices)), mean(gd_converged(sample_indices)), ...
            mean(hs_gs_converged(sample_indices))];
    end
end

result_table = array2table(result_rows, 'VariableNames', {'P','SNR_dB','Samples','Frozen_HS_multiplier', ...
    'GS_NMSE_dB','GD_NMSE_dB','HS_GS_NMSE_dB','DL_V24_NMSE_dB', ...
    'GD_minus_GS_dB','HS_GS_minus_GS_dB','DL_minus_GS_dB', ...
    'GD_gain_CI_low','HS_GS_gain_CI_low','DL_gain_CI_low', ...
    'GD_gain_CI_high','HS_GS_gain_CI_high','DL_gain_CI_high', ...
    'Mean_GS_updates','Mean_GD_updates','Mean_HS_GS_updates','Mean_V24_anchor_GS_updates', ...
    'GS_convergence_fraction','GD_convergence_fraction','HS_GS_convergence_fraction'});

figure_handle = figure('Color','w','Name','Final paired GS GD HS-GS DL comparison');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
method_columns = {'GS_NMSE_dB','GD_NMSE_dB','HS_GS_NMSE_dB','DL_V24_NMSE_dB'};
method_names = {'Biased GS','Linearized GD','Soft HS-GS (V36 frozen)','DL (V24 residual MLP)'};
for p_index = 1:numel(P_list)
    nexttile; hold on;
    selected_rows = result_table.P == P_list(p_index);
    for method_index = 1:numel(method_columns)
        plot(SNR_grid, result_table{selected_rows, method_columns{method_index}}, '-o', ...
            'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', method_names{method_index});
    end
    grid on; xlabel('SNR (dB)'); ylabel('NMSE (dB)');
    title(sprintf('N=%d, K=%d, P=%d, RSR=%g dB', N, K, P_list(p_index), RSR_dB));
    if p_index == 1, legend('Location','best'); end
end
sgtitle('Four methods on the same reserved V23 test observations');
writetable(result_table, table_file);
save(result_file, 'result_table', 'all_results', 'P_list', 'SNR_grid', 'N', 'K', ...
    'RSR_dB', 'configuration', 'bootstrap_replicates', '-v7.3');
saveas(figure_handle, figure_file);
disp(result_table);
fprintf('\nSaved paired comparison table: %s\n', table_file);
fprintf('Saved paired comparison figure: %s\n', figure_file);
fprintf('Saved paired comparison data: %s\n', result_file);

function validate_test_data(data, N, K, P, SNR_grid)
required_fields = {'Y_data','G_data','S_data','B_data','sigma2_data','SNR_dB_data'};
for field_index = 1:numel(required_fields)
    if ~isfield(data, required_fields{field_index})
        error('Test MAT file is missing field %s.', required_fields{field_index});
    end
end
if size(data.Y_data,1) ~= N || size(data.Y_data,2) ~= P || ...
        size(data.G_data,1) ~= N || size(data.G_data,2) ~= K || ...
        size(data.S_data,1) ~= K || size(data.S_data,2) ~= P || ...
        size(data.B_data,1) ~= N || size(data.B_data,2) ~= P || ...
        size(data.Y_data,3) ~= size(data.G_data,3) || ...
        size(data.Y_data,3) ~= size(data.B_data,3) || ...
        size(data.Y_data,3) ~= numel(data.SNR_dB_data) || ...
        size(data.Y_data,3) ~= numel(data.sigma2_data)
    error('Test data dimensions are inconsistent for P=%d.', P);
end
if ~all(ismember(SNR_grid, unique(double(data.SNR_dB_data(:)))'))
    error('Test data does not contain the required SNR grid for P=%d.', P);
end
end

function feature = final_build_one_sample_features(Y, B, S, G_gs, sigma2, snr_db, channel_scale, N, K, P)
Y = single(Y) / single(channel_scale);
B = single(B) / single(channel_scale);
S = single(S);
Ggs = single(G_gs) / single(channel_scale);
normalized_sigma2 = single(sigma2) / single(channel_scale)^2;
gram = double(S) * double(S)';
gram_eigenvalues = real(eig(gram));
pilot_bound = max(max(gram_eigenvalues), 1e-6);
positive_eigenvalue = gram_eigenvalues(gram_eigenvalues > max(gram_eigenvalues) * 1e-8);
if isempty(positive_eigenvalue)
    pilot_condition = 0;
else
    pilot_condition = log(max(pilot_bound / max(min(positive_eigenvalue), 1e-6), 1));
end
noise_feature = log1p(double(normalized_sigma2) / pilot_bound);
snr_feature = (double(snr_db) - 12.5) / 17.5;
feature = [Y.'; real(B).'; imag(B).'; repmat([real(S(:)); imag(S(:))], 1, N); ...
    [real(Ggs), imag(Ggs)].'; repmat(single([noise_feature; pilot_condition; snr_feature]), 1, N)];
expected_feature_count = 3 * P + 2 * K * P + 2 * K + 3;
if size(feature,1) ~= expected_feature_count
    error('Constructed V24 feature count %d; expected %d.', size(feature,1), expected_feature_count);
end
end

function scale = final_singular_noise_scale(S, sigma2, N, p_H)
K = size(S,1);
right_inverse = S' / (S*S');
scale = sqrt(sigma2) * norm(right_inverse,'fro') * sqrt(max(N-p_H,p_H+1)/K);
end

function [G,iteration,converged] = final_soft_em_converged(Y,S,B,sigma2,p_H,threshold,max_updates,tolerance)
G = final_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    G = final_soft_hankel_project(G,p_H,threshold);
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function G = final_soft_hankel_project(G,p_H,threshold)
N = size(G,1); number_of_rows = N-p_H; number_of_columns = p_H+1;
for user = 1:size(G,2)
    x = G(:,user); H = zeros(number_of_rows,number_of_columns);
    for row = 1:number_of_rows, H(row,:) = x(row:row+number_of_columns-1).'; end
    [U,Sigma,V] = svd(H,'econ'); shrunk = max(diag(Sigma)-threshold,0);
    H_soft = U*diag(shrunk)*V'; reconstructed = zeros(N,1); counts = zeros(N,1);
    for row = 1:number_of_rows
        for column = 1:number_of_columns
            index = row+column-1; reconstructed(index) = reconstructed(index)+H_soft(row,column);
            counts(index) = counts(index)+1;
        end
    end
    G(:,user) = reconstructed./counts;
end
end

function [G,iteration,converged] = final_em_converged(Y,S,B,sigma2,max_updates,tolerance)
G = final_spectral_initialize(Y,S,B); right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G; field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = final_gs_converged(Y,S,B,max_updates,tolerance)
G = final_spectral_initialize(Y,S,B); right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    G = (Y.*exp(1j*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = final_gd_converged(Y,S,B,max_updates,tolerance,initial_variance,seed)
rng(mod(double(seed)+720000,2^32-1),'twister');
N = size(Y,1); K = size(S,1);
G = sqrt(initial_variance/2)*(randn(N,K)+1j*randn(N,K));
phase_reference = exp(-1j*angle(B)); step_size = 1/max(real(eig(S*S')));
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function G = final_spectral_initialize(Y,S,B)
N = size(Y,1); K = size(S,1); G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).'; b = B(antenna,:).'; A = [S.',b];
    M = A'*diag(y)*A; [V,D] = eig(M); [~,principal_index] = max(real(diag(D)));
    v = V(:,principal_index); v = v/max(norm(v),eps);
    scale = (abs(A*v).'*y)/max(norm(A*v)^2,eps);
    G(antenna,:) = (scale*exp(-1j*angle(v(K+1)))*v(1:K)).';
end
end
