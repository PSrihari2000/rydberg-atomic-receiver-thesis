%% Paired reanalysis of the previously viewed frozen sweep test releases.
% Runs the unchanged V4 inference and the established soft-rank rule on the
% same saved observations. This is exploratory reanalysis, not blind confirmation.
clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
experiment_root = fileparts(script_folder);
case_studies_root = fileparts(experiment_root);
project_root = fileparts(case_studies_root);
data_root = fullfile(case_studies_root,'V4_Frozen_Transfer','data','V4_FROZEN_CLASSICAL_FRESH_TEST_V1');
baseline_result_csv = fullfile(case_studies_root,'V4_Frozen_Transfer','results','V4_FROZEN_CLASSICAL_FRESH_TEST_V1','fresh_test_nmse_by_condition_corrected.csv');
model_root = fullfile(project_root,'baselines','N8_FROZEN_V4_BASELINE','models');
validation_root = fullfile(experiment_root,'Validation_results');
output_root = fullfile(experiment_root,'Test_results');
runtime_root = fullfile(experiment_root,'Runtime_results');
log_root = fullfile(experiment_root,'Experiment_logs');
if ~isfile(fullfile(validation_root,'soft_rank_selected_thresholds.csv'))
    error('Run run_validation_tuning.m first; no test data have been loaded.');
end
if ~isfile(baseline_result_csv)
    error('Corrected original per-condition CSV is missing: %s',baseline_result_csv);
end
if local_has_entries(output_root) || local_has_entries(runtime_root)
    error('Test or runtime output folder is not empty; refusing to overwrite.');
end
if isfile(fullfile(log_root,'test_reanalysis_v4_reproduction_checked.csv'))
    error('Existing reproduction log found; choose a new experiment folder before rerunning.');
end

SNR_grid = [-10 0 10 20 30];
P_list = [10 30];
RSR_dB = 25;
samples_per_SNR = 500;
maximum_gs_updates = 20000;
gs_convergence_tolerance = 1e-10;
number_of_stages = 16;
bootstrap_replicates = 2000;
bootstrap_seed = 2026111302;
maximum_parallel_workers = 6;
policy_names = {'configuration_specific','sweep_wide_per_P','N8K3_transfer'};

fprintf('Paired V4/soft-rank scoring on previously viewed sweep observations.\n');
fprintf('This is a test reanalysis, not a fresh untouched confirmation.\n');
fprintf('V4 checkpoint, validation policies, and input test files are read-only.\n');

% Verify checkpoint content before loading any test release.
models = cell(1,numel(P_list));
checkpoint_paths = cell(1,numel(P_list));
checkpoint_hashes = cell(1,numel(P_list));
expected_checkpoint_hashes = { ...
    'C22CACDE9048AB4305ED2251AFB9B620D7A31A66CFD809157B45098662996C15', ...
    '7B4740BCCAF412A50B283BB2515122956084B559AB5CF34BFC34811B4E3B1A00'};
for p_index = 1:numel(P_list)
    P = P_list(p_index);
    checkpoint = fullfile(model_root,sprintf('P%d',P),sprintf('selected_model_P%d.mat',P));
    if ~isfile(checkpoint)
        error('Frozen V4 checkpoint not found: %s',checkpoint);
    end
    checkpoint_paths{p_index} = checkpoint;
    checkpoint_hashes{p_index} = local_sha256(checkpoint);
    if ~strcmpi(checkpoint_hashes{p_index},expected_checkpoint_hashes{p_index})
        error('Frozen P=%d checkpoint hash differs from the inspected original.',P);
    end
    model = load(checkpoint,'best_parameters_numeric','N','K','P','RSR_dB','p_H', ...
        'number_of_stages','parameters_per_stage','number_of_gate_features', ...
        'number_of_gate_outputs','initialization_method');
    if ~isequal(size(model.best_parameters_numeric),[15 16]) || ...
            numel(model.best_parameters_numeric) ~= 240 || ~isreal(model.best_parameters_numeric) || ...
            any(~isfinite(model.best_parameters_numeric),'all') || ...
            model.N ~= 8 || model.K ~= 3 || model.P ~= P || model.RSR_dB ~= RSR_dB || ...
            model.p_H ~= 3 || model.number_of_stages ~= number_of_stages || ...
            model.parameters_per_stage ~= 15 || model.number_of_gate_features ~= 3 || ...
            model.number_of_gate_outputs ~= 3 || ~strcmp(model.initialization_method,'GS')
        error('Checkpoint metadata or 240 real-parameter invariant failed for P=%d.',P);
    end
    models{p_index} = model;
end

policy_table = readtable(fullfile(validation_root,'soft_rank_selected_thresholds.csv'));
baseline_table = readtable(baseline_result_csv);
test_cells = cell(20,1);
cell_metadata = zeros(20,4);
cell_counter = 0;
configuration_rows = cell(20,1);
configuration_row = 0;
worker_count = min(maximum_parallel_workers,feature('numcores'));
pool = gcp('nocreate');
if isempty(pool)
    parpool('Processes',worker_count);
elseif pool.NumWorkers > maximum_parallel_workers
    warning('Existing parallel pool has %d workers; using at most %d workers per loop.', ...
        pool.NumWorkers,maximum_parallel_workers);
end

for P = P_list
    p_index = find(P_list == P,1);
    model = models{p_index};
    for study_index = 1:2
        if study_index == 1
            study = 'Antenna';
            N_list = [8 16 24 32];
            K_list = 3;
        else
            study = 'User';
            N_list = 8;
            if P == 10
                K_list = 1:5;
            else
                K_list = 1:7;
            end
        end
        for N = N_list
            for K = K_list
                if strcmp(study,'Antenna')
                    input_file = fullfile(data_root,'Antenna_sweep', ...
                        sprintf('N%d_K%d_P%d_RSR25dB',N,K,P),'test.mat');
                else
                    input_file = fullfile(data_root,'User_sweep', ...
                        sprintf('N%d_K%d_P%d_RSR25dB',N,K,P),'test.mat');
                end
                if ~isfile(input_file)
                    error('Required saved test file is missing: %s',input_file);
                end
                data = load(input_file,'Y_data','G_data','S_data','B_data','sigma2_data', ...
                    'SNR_dB_data','N','K','P','RSR_dB','SNR_dB','data_version', ...
                    'split_name','number_of_samples','samples_per_SNR');
                local_validate_test(data,N,K,P,RSR_dB,SNR_grid,samples_per_SNR);
                number_of_samples = data.number_of_samples;
                p_H = floor((N-1)/2);

                sample_error = zeros(number_of_samples,1+numel(policy_names));
                sample_energy = zeros(number_of_samples,1);
                sample_snr = double(data.SNR_dB_data(:));
                sample_runtime = zeros(number_of_samples,2+numel(policy_names));
                sample_gs_iterations = zeros(number_of_samples,1);
                sample_gs_converged = false(number_of_samples,1);
                sample_soft_iterations = repmat(number_of_stages,number_of_samples,numel(policy_names));
                fprintf('Scoring %s N=%d K=%d P=%d: %d paired test samples.\n', ...
                    study,N,K,P,number_of_samples);

                % Warm up both methods on a validation-free first test sample;
                % warm-up time is discarded and the saved data remain unchanged.
                Yw = double(data.Y_data(:,:,1));
                Sw = double(data.S_data(:,:,1));
                Bw = double(data.B_data(:,:,1));
                sigma2w = double(data.sigma2_data(1));
                G0w = local_sweep_gs_initialize(Yw,Sw,Bw,maximum_gs_updates,gs_convergence_tolerance);
                local_v4_forward(model.best_parameters_numeric,G0w,Yw,Sw,Bw,sigma2w,p_H,N,K,number_of_stages);
                for policy_index = 1:numel(policy_names)
                    lambda_w = local_lookup_lambda(policy_table,policy_names{policy_index},study,N,K,P,sample_snr(1));
                    local_soft_rank_hsgs(G0w,Yw,Sw,Bw,sigma2w, ...
                        lambda_w*local_noise_scale(sigma2w,Sw,N,K,p_H),p_H,number_of_stages);
                end

                cell_timer = tic;
                parfor (sample_index = 1:number_of_samples,maximum_parallel_workers)
                    Y = double(data.Y_data(:,:,sample_index));
                    G_true = double(data.G_data(:,:,sample_index));
                    S = double(data.S_data(:,:,sample_index));
                    B = double(data.B_data(:,:,sample_index));
                    sigma2 = double(data.sigma2_data(sample_index));
                    current_snr = double(data.SNR_dB_data(sample_index));
                    energy = sum(abs(G_true).^2,'all');
                    timer = tic;
                    [G0,gs_iterations,gs_converged] = local_sweep_gs_initialize( ...
                        Y,S,B,maximum_gs_updates,gs_convergence_tolerance);
                    gs_seconds = toc(timer);
                    timer = tic;
                    G_v4 = local_v4_forward(model.best_parameters_numeric,G0,Y,S,B, ...
                        sigma2,p_H,N,K,number_of_stages);
                    v4_seconds = toc(timer);
                    local_error = zeros(1,1+numel(policy_names));
                    local_error(1) = sum(abs(G_v4-G_true).^2,'all');
                    local_runtime = zeros(1,2+numel(policy_names));
                    local_runtime(1) = gs_seconds;
                    local_runtime(2) = gs_seconds+v4_seconds;
                    for policy_index = 1:numel(policy_names)
                        lambda = local_lookup_lambda(policy_table,policy_names{policy_index}, ...
                            study,N,K,P,current_snr);
                        tau = lambda*local_noise_scale(sigma2,S,N,K,p_H);
                        timer = tic;
                        G_soft = local_soft_rank_hsgs(G0,Y,S,B,sigma2,tau,p_H,number_of_stages);
                        soft_seconds = toc(timer);
                        local_error(policy_index+1) = sum(abs(G_soft-G_true).^2,'all');
                        local_runtime(policy_index+2) = gs_seconds+soft_seconds;
                    end
                    sample_error(sample_index,:) = local_error;
                    sample_energy(sample_index) = energy;
                    sample_runtime(sample_index,:) = local_runtime;
                    sample_gs_iterations(sample_index) = gs_iterations;
                    sample_gs_converged(sample_index) = gs_converged;
                end

                parallel_wall_seconds = toc(cell_timer);
                if any(~isfinite(sample_error),'all') || any(~isfinite(sample_runtime),'all') || ...
                        any(sample_energy <= 0) || any(~isfinite(sample_energy))
                    error('Nonfinite score/runtime or invalid channel energy at %s N=%d K=%d P=%d.',study,N,K,P);
                end
                cell_counter = cell_counter+1;
                test_cells{cell_counter} = struct('study',study,'N',N,'K',K,'P',P, ...
                    'sample_error',sample_error,'sample_energy',sample_energy, ...
                    'sample_snr',sample_snr,'sample_runtime',sample_runtime, ...
                    'gs_iterations',sample_gs_iterations,'gs_converged',sample_gs_converged, ...
                    'soft_iterations',sample_soft_iterations,'input_file',input_file, ...
                    'parallel_wall_seconds',parallel_wall_seconds);
                cell_metadata(cell_counter,:) = [study_index N K P];
                configuration_row = configuration_row+1;
                configuration_rows{configuration_row} = {string(study),N,K,P,number_of_samples, ...
                    maximum_gs_updates,gs_convergence_tolerance,number_of_stages, ...
                    string(input_file),string(data.data_version), ...
                    string(checkpoint_paths{p_index}),string(checkpoint_hashes{p_index})};
            end
        end
    end
end

% Per-condition pooled NMSE. The same true-channel energy denominator is used
% for every estimator on each paired condition.
condition_rows = cell(20*numel(SNR_grid)*numel(policy_names),1);
condition_row = 0;
runtime_rows = cell(20*(2+numel(policy_names)),1);
runtime_row = 0;
for cell_index = 1:cell_counter
    c = test_cells{cell_index};
    for snr_index = 1:numel(SNR_grid)
        selected = c.sample_snr == SNR_grid(snr_index);
        if nnz(selected) ~= samples_per_SNR
            error('Unexpected test count at %s N=%d K=%d P=%d SNR=%g.', ...
                c.study,c.N,c.K,c.P,SNR_grid(snr_index));
        end
        denominator = sum(c.sample_energy(selected));
        nmse_linear = sum(c.sample_error(selected,:),1)/denominator;
        for policy_index = 1:numel(policy_names)
            v4_db = 10*log10(max(nmse_linear(1),realmin));
            soft_db = 10*log10(max(nmse_linear(policy_index+1),realmin));
            condition_row = condition_row+1;
            condition_rows{condition_row} = {string(c.study),c.N,c.K,c.P,SNR_grid(snr_index), ...
                string(policy_names{policy_index}),nmse_linear(1),v4_db, ...
                nmse_linear(policy_index+1),soft_db,soft_db-v4_db,nnz(selected)};
        end
    end
    for method_index = 1:2+numel(policy_names)
        runtime_row = runtime_row+1;
        if method_index == 1
            method_name = 'GS initialization';
            runtime_col = 1;
        elseif method_index == 2
            method_name = 'Frozen PI-DU-RACE V4';
            runtime_col = 2;
        else
            method_name = ['Soft-rank HS-GS ' policy_names{method_index-2}];
            runtime_col = method_index;
        end
        values = c.sample_runtime(:,runtime_col);
        v4_mean_seconds = mean(c.sample_runtime(:,2));
        runtime_rows{runtime_row} = {string(c.study),c.N,c.K,c.P,string(method_name), ...
            mean(values),median(values),std(values),numel(values), ...
            mean(c.gs_iterations),mean(c.gs_converged), ...
            mean(values)/max(v4_mean_seconds,realmin),c.parallel_wall_seconds};
    end
end
condition_table = cell2table(vertcat(condition_rows{:}),'VariableNames', ...
    {'Study','N','K','P','SNR_dB','SoftThresholdPolicy','V4_LinearNMSE','V4_NMSE_dB', ...
    'SoftRank_LinearNMSE','SoftRank_NMSE_dB','SoftMinusV4_Gain_dB','Samples'});
runtime_table = cell2table(vertcat(runtime_rows{:}),'VariableNames', ...
    {'Study','N','K','P','Method','MeanSecondsPerSample','MedianSecondsPerSample', ...
    'StdSecondsPerSample','Samples','MeanGSUpdates','GSConvergedFraction', ...
    'RuntimeRatioVsFrozenV4','ScoringBatchParallelWallSeconds'});
configuration_table = cell2table(vertcat(configuration_rows{:}),'VariableNames', ...
    {'Study','N','K','P','TestSamples','GSUpdateCap','GSTolerance','RefinementStages', ...
    'InputTestFile','DataVersion','FrozenCheckpointFile','CheckpointSHA256'});
% Reproduce the frozen-V4 per-condition scores from the preserved source CSV.
for row_index = 1:height(condition_table)
    row = condition_table(row_index,:);
    % Original corrected table encodes antenna/user as StudyIndex 1/2.
    study_index = 1 + strcmpi(char(row.Study),'User');
    old = baseline_table(baseline_table.StudyIndex == study_index & ...
        baseline_table.N == row.N & baseline_table.K == row.K & baseline_table.P == row.P & ...
        baseline_table.SNR_dB == row.SNR_dB,:);
    if height(old) ~= 1 || abs(old.FrozenV4_NMSE(1)-row.V4_LinearNMSE) > 1e-8*max(old.FrozenV4_NMSE(1),1)
        error('V4 score reproduction check failed at %s N=%d K=%d P=%d SNR=%g.', ...
            row.Study,row.N,row.K,row.P,row.SNR_dB);
    end
end
% Write outputs only after the full V4 reproduction check passes.
writetable(condition_table,fullfile(output_root,'fresh_test_nmse_by_condition_with_soft_rank.csv'));
writetable(runtime_table,fullfile(runtime_root,'runtime_by_configuration.csv'));
save(fullfile(output_root,'paired_test_sample_errors.mat'),'test_cells','cell_metadata', ...
    'policy_names','SNR_grid','P_list','RSR_dB','maximum_gs_updates', ...
    'gs_convergence_tolerance','number_of_stages','bootstrap_replicates', ...
    'bootstrap_seed','-v7.3');
save(fullfile(output_root,'experiment_configuration.mat'),'configuration_table', ...
    'policy_table','SNR_grid','P_list','RSR_dB','samples_per_SNR', ...
    'maximum_gs_updates','gs_convergence_tolerance','number_of_stages', ...
    'bootstrap_replicates','bootstrap_seed','policy_names','checkpoint_paths', ...
    'checkpoint_hashes','-v7.3');
writetable(condition_table,fullfile(log_root,'test_reanalysis_v4_reproduction_checked.csv'));
fprintf('\nPaired test reanalysis complete for %d configurations and %d sample-method pairs.\n', ...
    cell_counter,sum(cellfun(@(x) size(x.sample_error,1),test_cells(1:cell_counter))));
fprintf('V4 per-condition results matched the corrected original CSV.\n');
fprintf('Test scores: %s\nRuntime table: %s\n', ...
    fullfile(output_root,'fresh_test_nmse_by_condition_with_soft_rank.csv'), ...
    fullfile(runtime_root,'runtime_by_configuration.csv'));

function local_validate_test(data,N,K,P,RSR_dB,SNR_grid,samples_per_SNR)
if data.N ~= N || data.K ~= K || data.P ~= P || data.RSR_dB ~= RSR_dB || ...
        ~isequal(double(data.SNR_dB(:)).',SNR_grid) || ~strcmp(data.split_name,'test') || ...
        ~strcmp(data.data_version,'V3_V4_FROZEN_CLASSICAL_FRESH_TEST_V1') || ...
        data.number_of_samples ~= samples_per_SNR*numel(SNR_grid) || ...
        ~isequal(size(data.Y_data),[N P data.number_of_samples]) || ...
        ~isequal(size(data.G_data),[N K data.number_of_samples]) || ...
        ~isequal(size(data.S_data),[K P data.number_of_samples]) || ...
        ~isequal(size(data.B_data),[N P data.number_of_samples]) || ...
        numel(data.sigma2_data) ~= data.number_of_samples || ...
        numel(data.SNR_dB_data) ~= data.number_of_samples
    error('Test metadata or array dimensions mismatch at N=%d,K=%d,P=%d.',N,K,P);
end
for snr_index = 1:numel(SNR_grid)
    if nnz(double(data.SNR_dB_data(:)) == SNR_grid(snr_index)) ~= samples_per_SNR
        error('Unexpected sample count at SNR=%g.',SNR_grid(snr_index));
    end
end
end

function lambda = local_lookup_lambda(policy_table,policy,study,N,K,P,SNR_dB)
selected = strcmp(string(policy_table.Policy),string(policy)) & ...
    strcmpi(string(policy_table.Study),string(study)) & policy_table.N == N & ...
    policy_table.K == K & policy_table.P == P & policy_table.SNR_dB == SNR_dB;
if nnz(selected) ~= 1
    error('Expected one validation-frozen threshold for policy=%s %s N=%d K=%d P=%d SNR=%g.', ...
        policy,study,N,K,P,SNR_dB);
end
lambda = policy_table.ThresholdMultiplier(selected);
end

function [G,iteration_count,converged] = local_sweep_gs_initialize(Y,S,B,maximum_updates,tolerance)
G = local_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
iteration_count = maximum_updates;
converged = false;
for iteration = 1:maximum_updates
    previous = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration_count = iteration;
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
    y = Y(antenna,:).';
    b = B(antenna,:).';
    augmented = [S.',b];
    matrix = augmented'*diag(y)*augmented;
    [vectors,values] = eig(matrix);
    [~,index] = max(real(diag(values)));
    vector = vectors(:,index);
    vector = vector/max(norm(vector),eps);
    scale = (abs(augmented*vector).'*y)/max(norm(augmented*vector)^2,eps);
    G(antenna,:) = (scale*exp(-1i*angle(vector(K+1)))*vector(1:K)).';
end
end

function scale = local_noise_scale(sigma2,S,N,K,p_H)
right_inverse = S'/(S*S');
scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
end

function estimate = local_emgs_update(G,Y,S,B,sigma2)
right_inverse = S'/(S*S');
field = G*S+B;
kappa = 2/max(sigma2,eps)*Y.*abs(field);
ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
estimate = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
end

function G = local_soft_rank_hsgs(G,Y,S,B,sigma2,tau,p_H,stages)
for stage = 1:stages
    G_em = local_emgs_update(G,Y,S,B,sigma2);
    G = local_soft_hankel_projection(G_em,p_H,tau);
end
end

function G = local_soft_hankel_projection(G,p_H,tau)
[N,K] = size(G);
rows = N-p_H;
columns = p_H+1;
for user = 1:K
    vector = G(:,user);
    hankel = zeros(rows,columns);
    for row = 1:rows
        hankel(row,:) = vector(row:row+columns-1).';
    end
    [U,Sigma,V] = svd(hankel,'econ');
    shrunk = max(diag(Sigma)-tau,0);
    hankel_shrunk = U*diag(shrunk)*V';
    reconstructed = zeros(N,1);
    counts = zeros(N,1);
    for row = 1:rows
        for column = 1:columns
            antenna = row+column-1;
            reconstructed(antenna) = reconstructed(antenna)+hankel_shrunk(row,column);
            counts(antenna) = counts(antenna)+1;
        end
    end
    G(:,user) = reconstructed./counts;
end
end

function G = local_v4_forward(parameters,G,Y,S,B,sigma2,p_H,N,K,stages)
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2,'all'),eps),eps))/10);
right_inverse = S'/(S*S');
noise_scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
map = local_hankel_map(N,p_H);
counts = sum(map,1).';
for stage = 1:stages
    previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    hankel_estimate = zeros(N,K);
    for user = 1:K
        H = reshape(map*emgs(:,user),N-p_H,p_H+1);
        gram = H'*H;
        [vectors,values] = eig(gram);
        singular_values = sqrt(max(real(sum(values,2)),0));
        safe_values = max(singular_values,1e-12);
        relative = tanh(log(safe_values./max(mean(safe_values),1e-12)));
        logits = parameters(1,stage)+parameters(2,stage)*noise_feature+parameters(3,stage)*relative;
        thresholds = noise_scale*(2./(1+exp(-logits)));
        factors = max(1-thresholds./safe_values,0);
        H_shrunk = H*(vectors.*reshape(factors,1,[]))*vectors';
        hankel_estimate(:,user) = (map'*H_shrunk(:))./counts;
    end
    em_change = tanh(log(1+sqrt(sum(abs(emgs-previous).^2,'all')/max(sum(abs(previous).^2,'all'),eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(hankel_estimate-emgs).^2,'all')/max(sum(abs(emgs).^2,'all'),eps)))/3);
    gate = reshape(parameters(4:15,stage),4,3);
    gate_logits = gate(1,:).'*noise_feature+gate(2,:).'*em_change+gate(3,:).'*hankel_change+gate(4,:).';
    gate_logits = gate_logits-max(gate_logits);
    weights = exp(gate_logits)./sum(exp(gate_logits));
    G = weights(1)*previous+weights(2)*emgs+weights(3)*hankel_estimate;
end
end

function map = local_hankel_map(N,p_H)
rows = N-p_H;
columns = p_H+1;
map = zeros(rows*columns,N);
for row = 1:rows
    for column = 1:columns
        map(row+(column-1)*rows,row+column-1) = 1;
    end
end
end

function has_entries = local_has_entries(folder)
entries = dir(fullfile(folder,'*'));
entries = entries(~ismember({entries.name},{'.','..'}));
has_entries = ~isempty(entries);
end

function hash = local_sha256(file_path)
if ~ispc
    error('This provenance helper expects Windows certutil.');
end
[status,output] = system(['certutil -hashfile "' file_path '" SHA256']);
if status ~= 0
    error('Could not compute SHA-256 for checkpoint: %s',file_path);
end
tokens = regexp(output,'(?m)^\s*([A-Fa-f0-9]{64})\s*$','tokens','once');
if isempty(tokens)
    error('Could not parse certutil SHA-256 output for: %s',file_path);
end
hash = upper(tokens{1});
end
