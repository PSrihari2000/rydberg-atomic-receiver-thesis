%% Baseline N=8, K=3 classical-estimator convergence diagnostic
% Re-scores only the already-used frozen V4 baseline test files to record
% convergence flags and update counts. It does not tune ranks or parameters.
% The test release was previously inspected; these outputs are diagnostic,
% not independent confirmatory evidence.
% The PI-DU-RACE model has 16 fixed stages; iterative convergence is N/A.

clear;
clc;

N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
maximum_parallel_workers = 6;

script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
repository_folder = fileparts(case_studies_folder);
baseline_folder = fullfile(repository_folder,'baselines','N8_FROZEN_V4_BASELINE');
test_folder = fullfile(baseline_folder,'test_data');
policy_file = fullfile(baseline_folder,'policy','N8_hard_rank_validation_policy.csv');
output_folder = fullfile(case_studies_folder,'V4_Frozen_Transfer','results', ...
    'V4_FROZEN_CLASSICAL_FRESH_TEST_V1','baseline_convergence_diagnostic');

if ~isfile(policy_file) || ~isfile(fullfile(test_folder,'test_P10.mat')) || ...
        ~isfile(fullfile(test_folder,'test_P30.mat'))
    error('Baseline test data or frozen hard-rank policy is missing.');
end
if isfolder(output_folder)
    existing = dir(fullfile(output_folder,'*'));
    existing = existing(~ismember({existing.name},{'.','..'}));
    if ~isempty(existing)
        error('Baseline convergence output folder already contains files. Inspect them or choose a new output folder: %s',output_folder);
    end
else
    mkdir(output_folder);
end

policy_table = readtable(policy_file);
required_policy = {'P','SNR_dB','FrozenHardRank'};
if ~all(ismember(required_policy,policy_table.Properties.VariableNames))
    error('Frozen rank policy must contain P, SNR_dB, and FrozenHardRank.');
end
for p_index = 1:numel(P_list)
    for snr_index = 1:numel(SNR_grid)
        match = policy_table.P == P_list(p_index) & policy_table.SNR_dB == SNR_grid(snr_index);
        if nnz(match) ~= 1
            error('Expected one frozen rank for P=%d, SNR=%g dB.',P_list(p_index),SNR_grid(snr_index));
        end
        if ~ismember(policy_table.FrozenHardRank(match),1:4)
            error('Invalid frozen hard rank in the validation policy.');
        end
    end
end

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes',maximum_parallel_workers);
end
fprintf('Baseline convergence diagnostic; N=8, K=3, P=[10 30], SNR=-5:5:30 dB.\n');
fprintf('Using up to %d workers. Classical cap=%d, tolerance=%.1e.\n', ...
    min(pool.NumWorkers,maximum_parallel_workers),maximum_updates,convergence_tolerance);
fprintf('Existing test data are used only to record convergence; no tuning or selection.\n');

method_names = {'GS','GD','EM-GS','Hard-rank HS-GS'};
sample_rows = cell(0,15);
summary_rows = cell(0,16);

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    loaded = load(fullfile(test_folder,sprintf('test_P%d.mat',P)), ...
        'Y_data','S_data','B_data','sigma2_data','SNR_dB_data', ...
        'number_of_samples','N','K','P','RSR_dB','split_name','dataset_release', ...
        'data_version','random_seed','samples_this_split_per_snr');
    data = loaded;
    if data.N ~= N || data.K ~= K || data.P ~= P || data.RSR_dB ~= RSR_dB || ...
            data.number_of_samples ~= 4000 || ~strcmp(data.split_name,'test') || ...
            ~strcmp(data.dataset_release,'V3_FRESH_TEST_V4') || ...
            ~strcmp(data.data_version,'V3_AoA_Uminus90to90') || ...
            data.random_seed ~= 2026110301+P || data.samples_this_split_per_snr ~= 500
        error('Baseline test metadata mismatch for P=%d.',P);
    end

    n_samples = data.number_of_samples;
    sample_converged = false(n_samples,4);
    sample_updates = zeros(n_samples,4);
    sample_snr = double(data.SNR_dB_data(:));
    if any(~ismember(sample_snr,SNR_grid))
        error('Unexpected SNR found in P=%d test data.',P);
    end

    parfor sample_index = 1:n_samples
        Y = double(data.Y_data(:,:,sample_index));
        S = double(data.S_data(:,:,sample_index));
        B = double(data.B_data(:,:,sample_index));
        sigma2 = double(data.sigma2_data(sample_index));
        current_snr = sample_snr(sample_index);
        policy_match = policy_table.P == P & policy_table.SNR_dB == current_snr;
        selected_rank = policy_table.FrozenHardRank(policy_match);
        random_seed = 881000000+1000000*P+sample_index;
        [sample_converged(sample_index,:),sample_updates(sample_index,:)] = ...
            local_measure_convergence(Y,S,B,sigma2,selected_rank,p_H, ...
            maximum_updates,convergence_tolerance,gd_initial_variance,random_seed);
    end

    for sample_index = 1:n_samples
        sample_rows(end+1,:) = [{8,3,P,sample_snr(sample_index),sample_index}, ...
            num2cell(double(sample_converged(sample_index,:))), ...
            num2cell(sample_updates(sample_index,:)),{false,16}];
    end

    for snr_index = 1:numel(SNR_grid)
        selected = sample_snr == SNR_grid(snr_index);
        if nnz(selected) ~= 500
            error('Expected 500 test samples at P=%d, SNR=%g dB.',P,SNR_grid(snr_index));
        end
        for method_index = 1:4
            flags = sample_converged(selected,method_index);
            counts = sample_updates(selected,method_index);
            summary_rows(end+1,:) = {8,3,P,SNR_grid(snr_index), ...
                method_names{method_index},nnz(selected),true,nnz(flags), ...
                100*mean(flags),nnz(~flags),mean(counts),median(counts), ...
                max(counts),NaN,NaN,NaN};
        end
        summary_rows(end+1,:) = {8,3,P,SNR_grid(snr_index),'PI-DU-RACE (DL)', ...
            nnz(selected),false,NaN,NaN,NaN,NaN,NaN,NaN,16, ...
            'Fixed-stage network; iterative convergence is not applicable',NaN};
    end
    fprintf('P=%d convergence diagnostics recorded for %d test samples.\n',P,n_samples);
end

sample_table = cell2table(sample_rows,'VariableNames', ...
    {'N','K','P','SNR_dB','SampleIndex','GS_Converged','GD_Converged', ...
    'EMGS_Converged','HardRankHSGS_Converged','GS_Updates','GD_Updates', ...
    'EMGS_Updates','HardRankHSGS_Updates','DL_ConvergenceApplicable', ...
    'DL_FixedStages'});

summary_table = cell2table(summary_rows,'VariableNames', ...
    {'N','K','P','SNR_dB','Method','SampleCount','ConvergenceApplicable', ...
    'ConvergedCount','ConvergenceRate_percent','NotConvergedCount', ...
    'MeanUpdates','MedianUpdates','MaximumUpdates','DL_FixedStages','Note','Reserved'});

sample_file = fullfile(output_folder,'baseline_convergence_by_sample.csv');
summary_file = fullfile(output_folder,'baseline_convergence_summary_by_SNR.csv');
writetable(sample_table,sample_file);
writetable(summary_table,summary_file);
fprintf('Saved per-sample baseline convergence CSV: %s\n',sample_file);
fprintf('Saved baseline convergence summary CSV: %s\n',summary_file);
fprintf('Diagnostic only: this previously viewed test release was not used for tuning.\n');

function [converged,updates] = local_measure_convergence(Y,S,B,sigma2,rank_value,p_H,max_updates,tolerance,gd_variance,random_seed)
    [N,P] = size(Y);
    K = size(S,2);
    right_inverse = S'/(S*S');
    G_spectral = zeros(N,K);
    for antenna_index = 1:N
        y = Y(antenna_index,:)';
        b = B(antenna_index,:)';
        augmented = [S.' b];
        spectral_matrix = augmented'*diag(y)*augmented;
        [vectors,values] = eig(spectral_matrix);
        [~,principal_index] = max(real(diag(values)));
        principal = vectors(:,principal_index);
        principal = principal/max(norm(principal),eps);
        scale = (abs(augmented*principal).'*y)/max(norm(augmented*principal)^2,eps);
        G_spectral(antenna_index,:) = (scale*exp(-1i*angle(principal(K+1)))*principal(1:K)).';
    end

    converged = false(1,4);
    updates = zeros(1,4);

    G = G_spectral;
    for iter = 1:max_updates
        previous = G;
        G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
        if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
            converged(1) = true;
            updates(1) = iter;
            break;
        end
    end
    if updates(1) == 0, updates(1) = max_updates; end

    rng(mod(random_seed+720000,2^32-1),'twister');
    G = sqrt(gd_variance/2)*(randn(N,K)+1i*randn(N,K));
    phase_reference = exp(-1i*angle(B));
    step_size = 1/max(real(eig(S*S')));
    for iter = 1:max_updates
        previous = G;
        G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
        if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
            converged(2) = true;
            updates(2) = iter;
            break;
        end
    end
    if updates(2) == 0, updates(2) = max_updates; end

    G = G_spectral;
    for iter = 1:max_updates
        previous = G;
        field = G*S+B;
        kappa = 2/max(sigma2,eps)*Y.*abs(field);
        ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
        G = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
        if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
            converged(3) = true;
            updates(3) = iter;
            break;
        end
    end
    if updates(3) == 0, updates(3) = max_updates; end

    G = G_spectral;
    for iter = 1:max_updates
        previous = G;
        field = previous*S+B;
        kappa = 2/max(sigma2,eps)*Y.*abs(field);
        ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
        G_em = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
        projected = zeros(N,K);
        for user_index = 1:K
            channel = G_em(:,user_index);
            hankel = zeros(N-p_H,p_H+1);
            for hankel_row = 1:N-p_H
                hankel(hankel_row,:) = channel(hankel_row:hankel_row+p_H).';
            end
            [U,Sigma,V] = svd(hankel,'econ');
            hankel = U(:,1:rank_value)*Sigma(1:rank_value,1:rank_value)*V(:,1:rank_value)';
            reconstruction = zeros(N,1);
            diagonal_counts = zeros(N,1);
            for hankel_row = 1:N-p_H
                for hankel_column = 1:p_H+1
                    antenna_index = hankel_row+hankel_column-1;
                    reconstruction(antenna_index) = reconstruction(antenna_index)+hankel(hankel_row,hankel_column);
                    diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
                end
            end
            projected(:,user_index) = reconstruction./diagonal_counts;
        end
        G = projected;
        if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
            converged(4) = true;
            updates(4) = iter;
            break;
        end
    end
    if updates(4) == 0, updates(4) = max_updates; end
end
