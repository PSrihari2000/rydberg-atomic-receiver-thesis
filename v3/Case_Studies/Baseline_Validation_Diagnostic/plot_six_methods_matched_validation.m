%% Six-method matched validation plot for frozen V4 vs soft-rank HS-GS diagnostic
% Scores GS, GD, EM-GS, hard-rank HS-GS on the exact validation indices already
% used by compare_v4_soft_hsgs_validation.m. Reuses its saved V4 and soft-HS-GS
% scores. No test data are loaded; no checkpoint, threshold, or rank is changed.
% Classical methods use the established spectral/random initialization and
% convergence settings below. V4 and soft-HS-GS retain their matched biased-GS
% initialization and exactly 16 updates from the original comparison.

clear;
clc;
close all;

%% Frozen settings and paths
N = 8;
K = 3;
P_list = [10 30];
RSR_dB = 25;
p_H = 3;
maximum_classical_updates = 20000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
gd_seed_base = 2026111300;
maximum_parallel_workers = 6;

script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
project_root = fileparts(case_studies_folder);
validation_folder = fullfile(project_root, 'data', 'N8_K3_RSR25dB_AoA_Uminus90to90');
model_folder = fullfile(project_root, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
hard_rank_policy_file = fullfile(project_root, 'baselines', 'N8_FROZEN_V4_BASELINE', ...
    'policy', 'N8_hard_rank_validation_policy.csv');
source_folder = fullfile(script_folder, 'results', 'PI_DU_RACE_VS_SOFT_HSGS_VALIDATION_V1');
source_samples_file = fullfile(source_folder, 'validation_paired_sample_diagnostics.csv');
source_summary_file = fullfile(source_folder, 'validation_nmse_and_paired_CI_by_condition.csv');
source_soft_policy_file = fullfile(source_folder, 'validation_selected_soft_threshold_policy.csv');
output_folder = fullfile(script_folder, 'results', 'PI_DU_RACE_SIX_METHOD_VALIDATION_V1');

if isfolder(output_folder) || isfile(output_folder)
    error('Output folder already exists. Inspect it and select a new versioned folder before rerunning: %s', output_folder);
end
required_files = {source_samples_file, source_summary_file, source_soft_policy_file, hard_rank_policy_file};
for file_index = 1:numel(required_files)
    if ~isfile(required_files{file_index})
        error('Required source file is missing: %s', required_files{file_index});
    end
end
for P = P_list
    if ~isfile(fullfile(validation_folder, sprintf('validation_P%d.mat', P))) || ...
            ~isfile(fullfile(model_folder, sprintf('P%d', P), sprintf('selected_model_P%d.mat', P)))
        error('Missing validation input or frozen V4 checkpoint for P=%d.', P);
    end
end

sample_index_table = readtable(source_samples_file);
v4_soft_summary = readtable(source_summary_file);
soft_policy = readtable(source_soft_policy_file);
hard_policy = readtable(hard_rank_policy_file);
required_sample_columns = {'P','SNR_dB','ValidationIndex'};
if ~all(ismember(required_sample_columns, sample_index_table.Properties.VariableNames))
    error('Saved paired-sample file lacks P, SNR_dB, or ValidationIndex columns.');
end
if ~all(ismember({'P','SNR_dB','FrozenSoftThresholdMultiplier'}, soft_policy.Properties.VariableNames)) || ...
        ~all(ismember({'P','SNR_dB','FrozenHardRank'}, hard_policy.Properties.VariableNames))
    error('A saved validation-frozen threshold or hard-rank policy has an unexpected format.');
end

fprintf('Six-method validation-only scoring on the exact V4/soft-HS-GS diagnostic indices.\n');
fprintf('Classical convergence cap=%d; tolerance=%.1e. No test data are loaded.\n', ...
    maximum_classical_updates, convergence_tolerance);
fprintf('The V4 checkpoint and both validation-frozen policies are read-only.\n');
fprintf('Output folder: %s\n', output_folder);
if isempty(gcp('nocreate'))
    parpool('Processes', maximum_parallel_workers);
end

%% Score the four classical methods on the same samples as saved V4 and soft-HS-GS
method_names = {'GS','GD','EM-GS','Hard-rank HS-GS','Soft-rank HS-GS','PI-DU-RACE V4'};
sample_rows = cell(0, 17);
summary_rows = zeros(numel(P_list)*8, 9);
summary_row = 0;

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    validation_data = load(fullfile(validation_folder, sprintf('validation_P%d.mat', P)), ...
        'Y_data','G_data','S_data','B_data','sigma2_data','SNR_dB_data', ...
        'N','K','P','RSR_dB','split_name');
    if validation_data.N ~= N || validation_data.K ~= K || validation_data.P ~= P || ...
            validation_data.RSR_dB ~= RSR_dB || ~strcmp(validation_data.split_name, 'validation')
        error('Validation metadata mismatch for P=%d.', P);
    end
    selected_samples = sample_index_table(sample_index_table.P == P, :);
    if height(selected_samples) ~= 8*186
        error('Expected the saved 186-sample-per-SNR diagnostic subset for P=%d.', P);
    end
    saved_summary = v4_soft_summary(v4_soft_summary.P == P, :);
    if height(saved_summary) ~= 8
        error('Expected eight saved V4/soft-HS-GS SNR rows for P=%d.', P);
    end

    snr_grid = sort(unique(double(selected_samples.SNR_dB))).';
    if ~isequal(snr_grid, -5:5:30)
        error('Unexpected SNR grid for P=%d.', P);
    end

    % Accumulate errors and truth energies by cell for ratio-of-sums NMSE.
    classical_error = zeros(numel(snr_grid), 4);
    truth_energy = zeros(numel(snr_grid), 1);
    sample_count = zeros(numel(snr_grid), 1);
    sample_indices = double(selected_samples.ValidationIndex(:));
    sample_snrs = double(selected_samples.SNR_dB(:));
    sample_classical_nmse = zeros(numel(sample_indices),4);
    sample_classical_updates = zeros(numel(sample_indices),4);
    sample_classical_converged = false(numel(sample_indices),4);
    sample_ranks = zeros(numel(sample_indices),1);
    rng_seeds = zeros(numel(sample_indices),1);
    sample_truth_energy = zeros(numel(sample_indices),1);
    parfor sample_position = 1:numel(sample_indices)
        sample_index = sample_indices(sample_position);
        current_snr = sample_snrs(sample_position);
        snr_index = find(snr_grid == current_snr, 1);
        if isempty(snr_index) || sample_index < 1 || sample_index > size(validation_data.Y_data, 3) || ...
                double(validation_data.SNR_dB_data(sample_index)) ~= current_snr
            error('Invalid or mismatched validation sample index for P=%d.', P);
        end
        Y = double(validation_data.Y_data(:, :, sample_index));
        G_true = double(validation_data.G_data(:, :, sample_index));
        S = double(validation_data.S_data(:, :, sample_index));
        B = double(validation_data.B_data(:, :, sample_index));
        sigma2 = double(validation_data.sigma2_data(sample_index));
        rng_seed = gd_seed_base + 100000*P + sample_index;
        rank_row = hard_policy(hard_policy.P == P & hard_policy.SNR_dB == current_snr, :);
        if height(rank_row) ~= 1
            error('Expected one frozen hard-rank policy for P=%d, SNR=%g.', P, current_snr);
        end
        [errors,updates,converged] = local_score_classical_sample(Y,G_true,S,B,sigma2, ...
            rank_row.FrozenHardRank,p_H,maximum_classical_updates,convergence_tolerance, ...
            gd_initial_variance,rng_seed);
        if any(~isfinite(errors))
            error('Nonfinite classical estimate for P=%d, SNR=%g, validation index=%d.',P,current_snr,sample_index);
        end
        sample_classical_nmse(sample_position,:) = errors/max(sum(abs(G_true).^2,'all'),eps);
        sample_truth_energy(sample_position) = sum(abs(G_true).^2,'all');
        sample_classical_updates(sample_position,:) = updates;
        sample_classical_converged(sample_position,:) = converged;
        sample_ranks(sample_position) = rank_row.FrozenHardRank;
        rng_seeds(sample_position) = rng_seed;
    end
    for sample_position = 1:numel(sample_indices)
        snr_index = find(snr_grid == sample_snrs(sample_position),1);
        sample_error = sample_classical_nmse(sample_position,:);
        classical_error(snr_index,:) = classical_error(snr_index,:) + sample_error*sample_truth_energy(sample_position);
        truth_energy(snr_index) = truth_energy(snr_index) + sample_truth_energy(sample_position);
        sample_count(snr_index) = sample_count(snr_index)+1;
        sample_rows(end+1,:) = {P,sample_snrs(sample_position),sample_indices(sample_position), ...
            sample_classical_nmse(sample_position,1),sample_classical_nmse(sample_position,2), ...
            sample_classical_nmse(sample_position,3),sample_classical_nmse(sample_position,4), ...
            sample_ranks(sample_position),sample_classical_updates(sample_position,1), ...
            sample_classical_updates(sample_position,2),sample_classical_updates(sample_position,3), ...
            sample_classical_updates(sample_position,4),sample_classical_converged(sample_position,1), ...
            sample_classical_converged(sample_position,2),sample_classical_converged(sample_position,3), ...
            sample_classical_converged(sample_position,4),rng_seeds(sample_position)}; %#ok<AGROW>
    end
    if any(sample_count ~= 186)
        error('Sample count differs from 186 in at least one SNR cell for P=%d.', P);
    end

    %% Join precomputed frozen V4 and soft-HS-GS scores for the same indices
    for snr_index = 1:numel(snr_grid)
        current_snr = snr_grid(snr_index);
        prior_row = saved_summary(saved_summary.SNR_dB == current_snr, :);
        if height(prior_row) ~= 1 || prior_row.DiagnosticSamples ~= sample_count(snr_index)
            error('Saved V4/soft-HS-GS summary does not match the scored validation cell.');
        end
        nmse_db = zeros(1, 6);
        nmse_db(1:4) = 10*log10(max(classical_error(snr_index,:)/truth_energy(snr_index), realmin));
        nmse_db(5) = prior_row.SoftHSGS_NMSE_dB;
        nmse_db(6) = prior_row.V4_NMSE_dB;
        summary_row = summary_row + 1;
        summary_rows(summary_row,:) = [P current_snr sample_count(snr_index) nmse_db];
    end
end

%% Save tables and separate pilot-count plots
sample_table = cell2table(sample_rows, 'VariableNames', ...
    {'P','SNR_dB','ValidationIndex','GS_NMSE_linear','GD_NMSE_linear','EMGS_NMSE_linear', ...
     'HardRankHSGS_NMSE_linear','FrozenHardRank','GS_Updates','GD_Updates','EMGS_Updates', ...
     'HardRankHSGS_Updates','GS_Converged','GD_Converged','EMGS_Converged', ...
     'HardRankHSGS_Converged','GDSeed'});
summary_table = array2table(summary_rows, 'VariableNames', ...
    {'P','SNR_dB','Samples','GS_NMSE_dB','GD_NMSE_dB','EMGS_NMSE_dB', ...
     'HardRankHSGS_NMSE_dB','SoftRankHSGS_NMSE_dB','V4_NMSE_dB'});
mkdir(output_folder);
writetable(sample_table, fullfile(output_folder, 'six_method_matched_validation_samples.csv'));
writetable(summary_table, fullfile(output_folder, 'six_method_matched_validation_nmse.csv'));
save(fullfile(output_folder, 'six_method_matched_validation.mat'), ...
    'sample_table','summary_table','method_names','N','K','P_list','RSR_dB','snr_grid', ...
    'p_H','maximum_classical_updates','convergence_tolerance','gd_initial_variance','gd_seed_base','maximum_parallel_workers');

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    rows = summary_table.P == P;
    figure('Color','w');
    hold on;
    plot(summary_table.SNR_dB(rows), summary_table.GS_NMSE_dB(rows), '-o', 'LineWidth',1.5, 'DisplayName','GS');
    plot(summary_table.SNR_dB(rows), summary_table.GD_NMSE_dB(rows), '-s', 'LineWidth',1.5, 'DisplayName','GD');
    plot(summary_table.SNR_dB(rows), summary_table.EMGS_NMSE_dB(rows), '-^', 'LineWidth',1.5, 'DisplayName','EM-GS');
    plot(summary_table.SNR_dB(rows), summary_table.HardRankHSGS_NMSE_dB(rows), '-v', 'LineWidth',1.5, 'DisplayName','Hard-rank HS-GS');
    plot(summary_table.SNR_dB(rows), summary_table.SoftRankHSGS_NMSE_dB(rows), '-d', 'LineWidth',1.5, 'DisplayName','Soft-rank HS-GS');
    plot(summary_table.SNR_dB(rows), summary_table.V4_NMSE_dB(rows), '-p', 'LineWidth',1.7, 'DisplayName','PI-DU-RACE V4');
    grid on;
    xlabel('SNR (dB)');
    ylabel('Validation NMSE (dB)');
    title(sprintf('Six-method matched validation comparison, P=%d', P));
    legend('Location','best');
    saveas(gcf, fullfile(output_folder, sprintf('six_method_validation_nmse_P%d.png',P)));
end

fprintf('\nSix-method matched validation plots and tables saved in: %s\n', output_folder);
fprintf('Validation-only diagnostic; no test data were loaded and no model was tuned.\n');

%% Local estimator implementations, copied to keep this script standalone
function [errors,updates,converged] = local_score_classical_sample(Y,G_true,S,B,sigma2,rank_value,p_H,max_updates,tolerance,gd_variance,seed)
[G_GS,gs_updates,gs_converged] = local_classical_estimator('GS',Y,S,B,sigma2,max_updates,tolerance,gd_variance,seed);
[G_GD,gd_updates,gd_converged] = local_classical_estimator('GD',Y,S,B,sigma2,max_updates,tolerance,gd_variance,seed);
[G_EMGS,emgs_updates,emgs_converged] = local_classical_estimator('EM-GS',Y,S,B,sigma2,max_updates,tolerance,gd_variance,seed);
[G_HS,hs_updates,hs_converged] = local_hard_rank_hsgs(Y,S,B,sigma2,p_H,rank_value,max_updates,tolerance);
updates = [gs_updates gd_updates emgs_updates hs_updates];
converged = [gs_converged gd_converged emgs_converged hs_converged];
errors = [sum(abs(G_GS-G_true).^2,'all'),sum(abs(G_GD-G_true).^2,'all'), ...
    sum(abs(G_EMGS-G_true).^2,'all'),sum(abs(G_HS-G_true).^2,'all')];
end

function [G, iteration_count, converged] = local_classical_estimator(method,Y,S,B,sigma2,max_updates,tolerance,gd_variance,seed)
method = upper(string(method));
if method == "GD"
    rng(mod(double(seed)+720000,2^32-1),'twister');
    N = size(Y,1);
    K = size(S,1);
    G = sqrt(gd_variance/2)*(randn(N,K)+1i*randn(N,K));
    phase_reference = exp(-1i*angle(B));
    step_size = 1/max(real(eig(S*S')));
else
    G = local_spectral_initialize(Y,S,B);
    right_inverse = S'/(S*S');
end
converged = false;
iteration_count = max_updates;
for update = 1:max_updates
    previous = G;
    if method == "GS"
        G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    elseif method == "GD"
        G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    elseif method == "EM-GS"
        field = G*S+B;
        kappa = 2/max(sigma2,eps)*Y.*abs(field);
        ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
        G = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    else
        error('Unknown classical method: %s',method);
    end
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration_count = update;
        converged = true;
        break;
    end
end
end

function [G,iteration_count,converged] = local_hard_rank_hsgs(Y,S,B,sigma2,p_H,rank_value,max_updates,tolerance)
N = size(Y,1);
K = size(S,1);
right_inverse = S'/(S*S');
G = local_spectral_initialize(Y,S,B);
iteration_count = max_updates;
converged = false;
for update = 1:max_updates
    previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G_em = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    G_projected = zeros(N,K);
    for user = 1:K
        hankel = local_hankel_embed(G_em(:,user),p_H);
        [U,Sigma,V] = svd(hankel,'econ');
        hankel = U(:,1:rank_value)*Sigma(1:rank_value,1:rank_value)*V(:,1:rank_value)';
        G_projected(:,user) = local_hankel_average(hankel,N,p_H);
    end
    G = G_projected;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration_count = update;
        converged = true;
        break;
    end
end
end

function G = local_spectral_initialize(Y,S,B)
N = size(Y,1);
K = size(S,1);
G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:)';
    b = B(antenna,:)';
    augmented = [S.' b];
    matrix = augmented'*diag(y)*augmented;
    [vectors,values] = eig(matrix);
    [~,index] = max(real(diag(values)));
    principal = vectors(:,index)/max(norm(vectors(:,index)),eps);
    scale = (abs(augmented*principal)'*y)/max(norm(augmented*principal)^2,eps);
    G(antenna,:) = (scale*exp(-1i*angle(principal(K+1)))*principal(1:K)).';
end
end

function H = local_hankel_embed(x,p_H)
N = numel(x);
H = zeros(N-p_H,p_H+1);
for row = 1:N-p_H
    H(row,:) = x(row:row+p_H).';
end
end

function x = local_hankel_average(H,N,p_H)
x = zeros(N,1);
counts = zeros(N,1);
for row = 1:N-p_H
    for column = 1:p_H+1
        index = row+column-1;
        x(index) = x(index)+H(row,column);
        counts(index) = counts(index)+1;
    end
end
x = x./counts;
end
