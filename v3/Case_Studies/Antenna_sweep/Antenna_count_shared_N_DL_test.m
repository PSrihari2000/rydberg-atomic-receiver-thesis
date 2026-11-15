%% Fresh test evaluation for the shared-N PI-DU-RACE study
% Run only after both P-specific models have been selected on fresh validation.
% Scores GS, GD, EM-GS, validation-frozen hard-rank HS-GS, N=8 V4 weights,
% and the jointly trained shared-N model on the same fresh test observations.

clear;
clc;
close all;

N_list = [8 16 24 32];
K = 3;
P_list = [10 30];
SNR_grid = [-10 0 10 20 30];
RSR_dB = 25;
maximum_updates = 10000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;

script_folder = fileparts(mfilename('fullpath'));
repository_folder = fileparts(fileparts(script_folder));
fresh_root = fullfile(script_folder, 'data', 'ANTENNA_COUNT_SHARED_N_FRESH');
model_root = fullfile(script_folder, 'results', 'ANTENNA_COUNT_SHARED_N_PI_DU_RACE_FROM_V4');
v4_model_root = fullfile(repository_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
policy_file = fullfile(script_folder, 'results', 'ANTENNA_COUNT_CLASSICAL', 'validation_policy', 'antenna_sweep_hard_rank_policy.mat');
output_folder = fullfile(script_folder, 'results', 'ANTENNA_COUNT_SHARED_N_TEST_EVALUATION');

if ~isfolder(fresh_root) || ~isfolder(model_root) || ~isfolder(v4_model_root) || ~isfile(policy_file)
    error('Missing fresh data, selected shared-N models, frozen V4 models, or frozen validation rank policy.');
end
if isfolder(output_folder)
    error('Test output folder already exists. Inspect it and choose a new output folder before rerunning: %s', output_folder);
end
mkdir(output_folder);

policy_data = load(policy_file, 'hard_rank_policy', 'N_list', 'K_list', 'P_list', 'SNR_dB', 'RSR_dB', 'maximum_updates', 'convergence_tolerance');
if ~isequal(policy_data.N_list, N_list) || policy_data.K_list ~= K || ~isequal(policy_data.P_list, P_list) || ~isequal(policy_data.SNR_dB, SNR_grid) || policy_data.RSR_dB ~= RSR_dB
    error('Frozen rank-policy metadata does not match the shared-N test grid.');
end
maximum_updates = policy_data.maximum_updates;
convergence_tolerance = policy_data.convergence_tolerance;

method_names = {'GS','GD','EM-GS','Hard-rank HS-GS','V4 weights (N=8)','Shared-N DL'};
number_of_methods = numel(method_names);
number_of_samples_per_condition = 500;
result_rows = zeros(numel(N_list)*numel(P_list)*numel(SNR_grid)*number_of_methods, 10);
row_index = 0;

fprintf('Fresh shared-N test evaluation. Test files are loaded only in this script.\n');
fprintf('N=%s; K=%d; P=%s; SNR=%s dB.\n', mat2str(N_list), K, mat2str(P_list), mat2str(SNR_grid));

for P_index = 1:numel(P_list)
    P = P_list(P_index);
    selected_model_file = fullfile(model_root, sprintf('selected_shared_N_model_P%d.mat', P));
    v4_model_file = fullfile(v4_model_root, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P));
    if ~isfile(selected_model_file) || ~isfile(v4_model_file)
        error('Missing selected shared-N or frozen V4 model for P=%d.', P);
    end
    shared_model = load(selected_model_file, 'best_parameters_numeric', 'N_supported', 'K', 'P', 'SNR_grid', 'RSR_dB', 'number_of_stages', 'maximum_gs_updates', 'gs_convergence_tolerance', 'data_version_validation');
    v4_model = load(v4_model_file, 'best_parameters_numeric', 'N', 'K', 'P', 'RSR_dB', 'number_of_stages', 'parameters_per_stage', 'initialization_method');
    if ~isequal(shared_model.N_supported, N_list) || shared_model.K ~= K || shared_model.P ~= P || ~isequal(shared_model.SNR_grid, SNR_grid) || shared_model.RSR_dB ~= RSR_dB || ~strcmp(shared_model.data_version_validation, 'V3_ANTENNA_COUNT_SHARED_N_FRESH_V1')
        error('Selected shared-N model metadata mismatch for P=%d.', P);
    end
    if v4_model.N ~= 8 || v4_model.K ~= K || v4_model.P ~= P || v4_model.RSR_dB ~= RSR_dB || v4_model.number_of_stages ~= shared_model.number_of_stages || ~strcmp(v4_model.initialization_method, 'GS')
        error('Frozen V4 warm-start metadata mismatch for P=%d.', P);
    end

    for N_index = 1:numel(N_list)
        N = N_list(N_index);
        p_H = floor((N-1)/2);
        dimension_folder = fullfile(fresh_root, sprintf('N%d_K%d_RSR25dB', N, K));
        test_file = fullfile(dimension_folder, sprintf('test_P%d.mat', P));
        if ~isfile(test_file)
            error('Missing fresh test file: %s', test_file);
        end
        test_data = load(test_file, 'Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data', 'number_of_samples', 'N', 'K', 'P', 'SNR_dB', 'RSR_dB', 'data_version', 'split_name', 'samples_per_SNR');
        if test_data.N ~= N || test_data.K ~= K || test_data.P ~= P || test_data.RSR_dB ~= RSR_dB || ~isequal(double(test_data.SNR_dB), SNR_grid) || ~strcmp(test_data.data_version, 'V3_ANTENNA_COUNT_SHARED_N_FRESH_V1') || ~strcmp(test_data.split_name, 'test') || test_data.samples_per_SNR ~= number_of_samples_per_condition || test_data.number_of_samples ~= number_of_samples_per_condition*numel(SNR_grid)
            error('Fresh test metadata mismatch for N=%d,P=%d.', N, P);
        end

        fprintf('Scoring N=%d, P=%d on %d fresh samples.\n', N, P, test_data.number_of_samples);
        error_sum = zeros(number_of_samples_per_condition, numel(SNR_grid), number_of_methods);
        true_energy_sum = zeros(number_of_samples_per_condition, numel(SNR_grid));
        convergence_count = zeros(numel(SNR_grid), 4);
        update_sum = zeros(numel(SNR_grid), 4);

        for sample_index = 1:test_data.number_of_samples
            current_snr = double(test_data.SNR_dB_data(sample_index));
            snr_index = find(SNR_grid == current_snr, 1);
            if isempty(snr_index)
                error('Unexpected SNR in fresh test sample %d.', sample_index);
            end
            Y = double(test_data.Y_data(:, :, sample_index));
            G_true = double(test_data.G_data(:, :, sample_index));
            S = double(test_data.S_data(:, :, sample_index));
            B = double(test_data.B_data(:, :, sample_index));
            sigma2 = double(test_data.sigma2_data(sample_index));
            seed = 2026113000+100000*N+1000*P+sample_index;

            [G_GS, updates_GS, converged_GS] = local_classical_estimator('GS', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, seed);
            [G_GD, updates_GD, converged_GD] = local_classical_estimator('GD', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, seed);
            [G_EMGS, updates_EMGS, converged_EMGS] = local_classical_estimator('EM-GS', Y, S, B, sigma2, maximum_updates, convergence_tolerance, gd_initial_variance, seed);
            rank = policy_data.hard_rank_policy(N_index, P_index, snr_index);
            [G_HSGS, converged_HSGS, updates_HSGS] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, rank, maximum_updates, convergence_tolerance);
            G0 = local_biased_gs(Y, S, B, shared_model.maximum_gs_updates, shared_model.gs_convergence_tolerance);
            G_V4 = local_unfolded_forward(v4_model.best_parameters_numeric, Y, S, B, sigma2, G0, N, K, p_H, shared_model.number_of_stages);
            G_shared = local_unfolded_forward(shared_model.best_parameters_numeric, Y, S, B, sigma2, G0, N, K, p_H, shared_model.number_of_stages);
            estimates = {G_GS, G_GD, G_EMGS, G_HSGS, G_V4, G_shared};
            energy = max(sum(abs(G_true).^2, 'all'), eps);
            position_in_snr = sum(double(test_data.SNR_dB_data(1:sample_index)) == current_snr);
            true_energy_sum(position_in_snr, snr_index) = energy;
            for method_index = 1:number_of_methods
                error_sum(position_in_snr, snr_index, method_index) = sum(abs(estimates{method_index}-G_true).^2, 'all');
            end
            convergence_count(snr_index, :) = convergence_count(snr_index, :)+[converged_GS converged_GD converged_EMGS converged_HSGS];
            update_sum(snr_index, :) = update_sum(snr_index, :)+[updates_GS updates_GD updates_EMGS updates_HSGS];
        end

        for snr_index = 1:numel(SNR_grid)
            for method_index = 1:number_of_methods
                sample_nmse = error_sum(:, snr_index, method_index)./max(true_energy_sum(:, snr_index), eps);
                mean_nmse = mean(sample_nmse);
                row_index = row_index+1;
                if method_index <= 4
                    method_converged_count = convergence_count(snr_index, method_index);
                    method_mean_updates = update_sum(snr_index, method_index)/number_of_samples_per_condition;
                else
                    method_converged_count = NaN;
                    method_mean_updates = NaN;
                end
                result_rows(row_index, :) = [N K P SNR_grid(snr_index) method_index number_of_samples_per_condition mean_nmse 10*log10(max(mean_nmse, realmin)) method_converged_count method_mean_updates];
            end
        end
    end
end

result_table = array2table(result_rows, 'VariableNames', {'N','K','P','SNR_dB','MethodIndex','Samples','MeanNMSE_linear','MeanNMSE_dB','ConvergedCount','MeanUpdates'});
result_table.Method = reshape(method_names(result_table.MethodIndex), [], 1);
result_table = movevars(result_table, 'Method', 'After', 'SNR_dB');
per_condition_file = fullfile(output_folder, 'shared_N_test_nmse_by_condition.csv');
writetable(result_table, per_condition_file);
save(fullfile(output_folder, 'shared_N_test_results.mat'), 'result_table', 'method_names', 'N_list', 'K', 'P_list', 'SNR_grid', 'RSR_dB', 'policy_file', 'maximum_updates', 'convergence_tolerance');

summary_rows = zeros(numel(N_list)*numel(P_list)*number_of_methods, 6);
summary_row = 0;
for P_index = 1:numel(P_list)
    P = P_list(P_index);
    for N_index = 1:numel(N_list)
        N = N_list(N_index);
        for method_index = 1:number_of_methods
            condition_rows = result_table.N == N & result_table.P == P & result_table.MethodIndex == method_index;
            mean_linear_over_snr = mean(result_table.MeanNMSE_linear(condition_rows));
            summary_row = summary_row+1;
            summary_rows(summary_row, :) = [N K P method_index mean_linear_over_snr 10*log10(max(mean_linear_over_snr, realmin))];
        end
    end
end
summary_table = array2table(summary_rows, 'VariableNames', {'N','K','P','MethodIndex','MeanLinearNMSE_over_SNR','MeanNMSE_over_SNR_dB'});
summary_table.Method = reshape(method_names(summary_table.MethodIndex), [], 1);
summary_table = movevars(summary_table, 'Method', 'After', 'P');
writetable(summary_table, fullfile(output_folder, 'shared_N_test_nmse_SNRavg.csv'));

for P_index = 1:numel(P_list)
    P = P_list(P_index);
    figure('Color', 'w', 'Name', sprintf('Fresh shared-N test NMSE P=%d', P));
    hold on;
    for method_index = 1:number_of_methods
        selected = summary_table.P == P & summary_table.MethodIndex == method_index;
        plot(summary_table.N(selected), summary_table.MeanNMSE_over_SNR_dB(selected), '-o', 'LineWidth', 1.5, 'DisplayName', method_names{method_index});
    end
    grid on;
    xlabel('Antenna count N');
    ylabel('Mean NMSE over SNR (dB)');
    title(sprintf('Fresh test: shared-N model, P=%d, K=%d, RSR=%g dB', P, K, RSR_dB));
    legend('Location', 'best');
    saveas(gcf, fullfile(output_folder, sprintf('shared_N_test_nmse_SNRavg_P%d.png', P)));
end

fprintf('Fresh shared-N test evaluation complete.\n');
fprintf('Per-condition table: %s\n', per_condition_file);
fprintf('SNR-averaged table and separate P figures: %s\n', output_folder);
fprintf('No settings were tuned on the test split.\n');

function [G, iteration_count, converged] = local_classical_estimator(method, Y, S, B, sigma2, maximum_updates, tolerance, gd_initial_variance, random_seed)
method = upper(string(method));
if method == "GS" || method == "EM-GS"
    G = local_spectral_initialize(Y, S, B);
    right_inverse = S'/(S*S');
elseif method == "GD"
    rng(mod(double(random_seed)+720000, 2^32-1), 'twister');
    G = sqrt(gd_initial_variance/2)*(randn(size(Y,1), size(S,1))+1i*randn(size(Y,1), size(S,1)));
    phase_reference = exp(-1i*angle(B));
    step_size = 1/max(real(eig(S*S')));
else
    error('Unknown estimator.');
end
converged = false;
iteration_count = maximum_updates;
for iteration = 1:maximum_updates
    previous = G;
    if method == "GS"
        G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    elseif method == "GD"
        G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    else
        field = G*S+B;
        kappa = 2/max(sigma2, eps)*Y.*abs(field);
        ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
        G = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    end
    if norm(G-previous, 'fro')^2/max(norm(G, 'fro')^2, eps) < tolerance
        iteration_count = iteration;
        converged = true;
        break;
    end
end
end

function [G, converged, iteration_count] = local_hard_rank_hsgs(Y, S, B, sigma2, p_H, rank, maximum_updates, tolerance)
N = size(Y,1); K = size(S,1); right_inverse = S'/(S*S');
G = local_spectral_initialize(Y,S,B); converged = false; iteration_count = maximum_updates;
for iteration = 1:maximum_updates
    previous = G; field = G*S+B; kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G_emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    projected = zeros(N,K); rows = N-p_H; columns = p_H+1;
    for user = 1:K
        channel = G_emgs(:,user); hankel = zeros(rows,columns);
        for row = 1:rows, hankel(row,:) = channel(row:row+columns-1).'; end
        [U,Sigma,V] = svd(hankel,'econ');
        hankel = U(:,1:rank)*Sigma(1:rank,1:rank)*V(:,1:rank)';
        reconstructed = zeros(N,1); counts = zeros(N,1);
        for row = 1:rows
            for column = 1:columns
                antenna = row+column-1; reconstructed(antenna) = reconstructed(antenna)+hankel(row,column); counts(antenna) = counts(antenna)+1;
            end
        end
        projected(:,user) = reconstructed./counts;
    end
    G = projected;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        converged = true; iteration_count = iteration; break;
    end
end
end

function G = local_biased_gs(Y,S,B,maximum_updates,tolerance)
G = local_spectral_initialize(Y,S,B); right_inverse = S'/(S*S');
for iteration = 1:maximum_updates
    previous = G; G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance, break; end
end
end

function G = local_unfolded_forward(parameters,Y,S,B,sigma2,G0,N,K,p_H,number_of_stages)
right_inverse = S'/(S*S'); G = G0;
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2,'all'),eps),eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
map = local_hankel_map(N,p_H); counts = sum(map,1).';
for stage = 1:number_of_stages
    previous = G; field = G*S+B; kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    hankel = zeros(N,K);
    for user = 1:K
        H = reshape(map*emgs(:,user),N-p_H,p_H+1); gram = H'*H;
        [vectors,values] = eig(gram); singular = sqrt(max(real(sum(values,2)),0)); safe = max(singular,1e-12);
        relative = tanh(log(safe/max(mean(safe),1e-12)));
        shrink_logits = parameters(1,stage)+parameters(2,stage)*noise_feature+parameters(3,stage)*relative;
        thresholds = noise_scale*(2./(1+exp(-shrink_logits))); factors = max(1-thresholds./safe,0);
        H_shrunk = H*(vectors.*reshape(factors,1,[]))*vectors';
        hankel(:,user) = (map'*H_shrunk(:))./counts;
    end
    emgs_change = tanh(log(1+sqrt(sum(abs(emgs-previous).^2,'all')/max(sum(abs(previous).^2,'all'),eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(hankel-emgs).^2,'all')/max(sum(abs(emgs).^2,'all'),eps)))/3);
    coefficients = reshape(parameters(4:15,stage),4,3);
    logits = coefficients(1,:).'*noise_feature+coefficients(2,:).'*emgs_change+coefficients(3,:).'*hankel_change+coefficients(4,:).';
    weights = exp(logits-max(logits)); weights = weights/sum(weights);
    G = weights(1)*previous+weights(2)*emgs+weights(3)*hankel;
end
end

function map = local_hankel_map(N,p_H)
rows = N-p_H; columns = p_H+1; map = zeros(rows*columns,N);
for row = 1:rows
    for column = 1:columns
        map(row+(column-1)*rows,row+column-1) = 1;
    end
end
end

function G = local_spectral_initialize(Y,S,B)
N = size(Y,1); K = size(S,1); G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).'; b = B(antenna,:).'; augmented = [S.',b];
    matrix = augmented'*diag(y)*augmented; [vectors,values] = eig(matrix);
    [~,index] = max(real(diag(values))); vector = vectors(:,index); vector = vector/max(norm(vector),eps);
    scale = (abs(augmented*vector).'*y)/max(norm(augmented*vector)^2,eps);
    G(antenna,:) = (scale*exp(-1i*angle(vector(K+1)))*vector(1:K)).';
end
end
