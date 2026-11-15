%% Frozen six-method evaluation on the independent confirmatory test release
% Methods: GS, GD, EM-GS, validation-frozen hard-rank HS-GS,
% validation-frozen soft-rank HS-GS, and frozen PI-DU-RACE V4.
% V4 and soft-HS-GS share the same biased-GS initialization and each perform
% exactly 16 EM-GS/Hankel updates. No model or policy is changed here.

clear;
clc;
close all;

N = 8;
K = 3;
P_list = [10 30];
SNR_grid = -5:5:30;
RSR_dB = 25;
p_H = 3;
number_of_stages = 16;
parameters_per_stage = 15;
maximum_gs_updates = 5000;
maximum_classical_updates = 20000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
gd_seed_base = 2026122600;
bootstrap_seed = 2026122700;
bootstrap_replicates = 5000;
maximum_parallel_workers = 6;
expected_samples_per_snr = 500;
runtime_samples_per_snr = 10;
runtime_repeats_per_method = 3;
dataset_release = 'PI_DU_RACE_SIX_METHOD_CONFIRMATORY_TEST_V1';
expected_test_seed_base = 2026122500;

script_folder = fileparts(mfilename('fullpath'));
case_studies_folder = fileparts(script_folder);
project_root = fileparts(case_studies_folder);
test_folder = fullfile(case_studies_folder,'data',dataset_release);
v4_model_folder = fullfile(project_root,'baselines','N8_FROZEN_V4_BASELINE','models');
hard_policy_file = fullfile(project_root,'baselines','N8_FROZEN_V4_BASELINE', ...
    'policy','N8_hard_rank_validation_policy.csv');
soft_policy_file = fullfile(script_folder,'results', ...
    'PI_DU_RACE_VS_SOFT_HSGS_VALIDATION_V1','validation_selected_soft_threshold_policy.csv');
output_folder = fullfile(script_folder,'results','PI_DU_RACE_SIX_METHOD_CONFIRMATORY_TEST_V1');

if isfile(output_folder)
    error('Output path exists as a file: %s',output_folder);
elseif isfolder(output_folder)
    existing_entries = dir(output_folder);
    existing_entries = existing_entries(~ismember({existing_entries.name},{'.','..'}));
    if ~isempty(existing_entries)
        error('Output folder contains files from the interrupted run. Inspect them and choose a new output name: %s',output_folder);
    end
    fprintf('Reusing the empty output folder left by the interrupted run.\n');
end
required_files = {hard_policy_file,soft_policy_file};
for idx = 1:numel(required_files)
    if ~isfile(required_files{idx})
        error('Required frozen validation policy is missing: %s',required_files{idx});
    end
end
for P = P_list
    if ~isfile(fullfile(test_folder,sprintf('test_P%d.mat',P))) || ...
            ~isfile(fullfile(v4_model_folder,sprintf('P%d',P),sprintf('selected_model_P%d.mat',P)))
        error('Missing test file or frozen V4 checkpoint for P=%d.',P);
    end
end
mkdir(output_folder);

hard_policy = readtable(hard_policy_file);
soft_policy = readtable(soft_policy_file);
if ~all(ismember({'P','SNR_dB','FrozenHardRank'},hard_policy.Properties.VariableNames)) || ...
        ~all(ismember({'P','SNR_dB','FrozenSoftThresholdMultiplier'},soft_policy.Properties.VariableNames))
    error('A validation-frozen policy has unexpected columns.');
end
if height(hard_policy) ~= numel(P_list)*numel(SNR_grid) || ...
        height(soft_policy) ~= numel(P_list)*numel(SNR_grid)
    error('Frozen policy row count does not match P and SNR conditions.');
end

fprintf('Frozen six-method confirmatory test evaluation.\n');
fprintf('New release=%s; N=%d; K=%d; RSR=%g dB; P=[%s].\n', ...
    dataset_release,N,K,RSR_dB,num2str(P_list));
fprintf('500 samples/SNR; classical cap=%d; V4 and soft-HS-GS use shared GS start plus 16 updates.\n', ...
    maximum_classical_updates);
fprintf('Validation-frozen policies/checkpoints are read-only. No tuning is performed.\n');
fprintf('Outputs: %s\n',output_folder);
if isempty(gcp('nocreate'))
    parpool('Processes',maximum_parallel_workers);
end

method_names = {'GS','GD','EM-GS','Hard-rank HS-GS','Soft-rank HS-GS','PI-DU-RACE V4'};
sample_tables = cell(numel(P_list),1);
condition_rows = zeros(numel(P_list)*numel(SNR_grid),11);
ci_rows = zeros(numel(P_list)*numel(SNR_grid)*5,9);
runtime_rows = zeros(numel(P_list)*2,10);
condition_row = 0;
ci_row = 0;
runtime_row = 0;

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    test_file = fullfile(test_folder,sprintf('test_P%d.mat',P));
    test_data = load(test_file,'Y_data','G_data','S_data','B_data','sigma2_data', ...
        'SNR_dB_data','number_of_samples','N','K','P','RSR_dB','AoA_deg_range', ...
        'data_version','dataset_release','random_seed','samples_this_split_per_snr', ...
        'split_name','d_over_lambda');
    if test_data.N ~= N || test_data.K ~= K || test_data.P ~= P || ...
            test_data.RSR_dB ~= RSR_dB || ~strcmp(test_data.split_name,'test') || ...
            ~strcmp(test_data.dataset_release,dataset_release) || ...
            ~strcmp(test_data.data_version,dataset_release) || ...
            test_data.random_seed ~= expected_test_seed_base+P || ...
            test_data.samples_this_split_per_snr ~= expected_samples_per_snr || ...
            test_data.number_of_samples ~= expected_samples_per_snr*numel(SNR_grid) || ...
            ~isequal(double(test_data.AoA_deg_range),[-90 90]) || test_data.d_over_lambda ~= 0.5
        error('Test metadata mismatch or unexpected split for P=%d.',P);
    end
    if ~isequal(sort(unique(double(test_data.SNR_dB_data(:)))).',SNR_grid)
        error('Unexpected SNR grid in P=%d test file.',P);
    end
    model_file = fullfile(v4_model_folder,sprintf('P%d',P),sprintf('selected_model_P%d.mat',P));
    model = load(model_file,'best_parameters_numeric','P','N','K','RSR_dB','p_H', ...
        'number_of_stages','parameters_per_stage','initialization_method', ...
        'maximum_gs_updates','gs_convergence_tolerance');
    if ~isequal(size(model.best_parameters_numeric),[parameters_per_stage number_of_stages]) || ...
            model.N ~= N || model.K ~= K || model.P ~= P || model.RSR_dB ~= RSR_dB || ...
            model.p_H ~= p_H || model.number_of_stages ~= number_of_stages || ...
            model.parameters_per_stage ~= parameters_per_stage || ...
            ~strcmp(model.initialization_method,'GS')
        error('Frozen V4 checkpoint metadata/shape mismatch for P=%d.',P);
    end

    hard_rank = zeros(1,numel(SNR_grid));
    soft_multiplier = zeros(1,numel(SNR_grid));
    for snr_index = 1:numel(SNR_grid)
        current_snr = SNR_grid(snr_index);
        hard_row = hard_policy(hard_policy.P==P & hard_policy.SNR_dB==current_snr,:);
        soft_row = soft_policy(soft_policy.P==P & soft_policy.SNR_dB==current_snr,:);
        if height(hard_row) ~= 1 || height(soft_row) ~= 1
            error('Expected exactly one frozen policy row for P=%d and SNR=%g.',P,current_snr);
        end
        hard_rank(snr_index) = double(hard_row.FrozenHardRank);
        soft_multiplier(snr_index) = double(soft_row.FrozenSoftThresholdMultiplier);
        if hard_rank(snr_index) < 1 || hard_rank(snr_index) > min(N-p_H,p_H+1)
            error('Invalid frozen Hankel rank at P=%d, SNR=%g.',P,current_snr);
        end
    end

    sample_count = test_data.number_of_samples;
    snr_by_sample = double(test_data.SNR_dB_data(:));
    error_energy = zeros(sample_count,6);
    truth_energy = zeros(sample_count,1);
    updates = zeros(sample_count,4);
    converged = false(sample_count,4);
    gs_iterations_shared = zeros(sample_count,1);
    gs_converged_shared = false(sample_count,1);
    gd_seed = zeros(sample_count,1);

    fprintf('\nP=%d: scoring %d samples across %d SNRs.\n',P,sample_count,numel(SNR_grid));
    for snr_index = 1:numel(SNR_grid)
        current_snr = SNR_grid(snr_index);
        indices = find(snr_by_sample==current_snr);
        if numel(indices) ~= expected_samples_per_snr
            error('Expected %d samples for P=%d, SNR=%g; found %d.', ...
                expected_samples_per_snr,P,current_snr,numel(indices));
        end
        fprintf('P=%d, SNR=%g dB: scoring %d samples...\n',P,current_snr,numel(indices));
        progress_queue = parallel.pool.DataQueue;
        afterEach(progress_queue,@local_progress_tick);
        cell_error_energy = zeros(numel(indices),6);
        cell_truth_energy = zeros(numel(indices),1);
        cell_updates = zeros(numel(indices),4);
        cell_converged = false(numel(indices),4);
        cell_gs_iterations = zeros(numel(indices),1);
        cell_gs_converged = false(numel(indices),1);
        cell_gd_seed = zeros(numel(indices),1);
        parfor local_index = 1:numel(indices)
            sample_index = indices(local_index);
            Y = double(test_data.Y_data(:,:,sample_index));
            G_true = double(test_data.G_data(:,:,sample_index));
            S = double(test_data.S_data(:,:,sample_index));
            B = double(test_data.B_data(:,:,sample_index));
            sigma2 = double(test_data.sigma2_data(sample_index));
            seed = gd_seed_base + 100000*P + sample_index;

            % Standard classical algorithms use their established starts.
            [G_gs,updates_gs,conv_gs] = local_classical('GS',Y,S,B,sigma2, ...
                maximum_classical_updates,convergence_tolerance,gd_initial_variance,seed,0);
            [G_gd,updates_gd,conv_gd] = local_classical('GD',Y,S,B,sigma2, ...
                maximum_classical_updates,convergence_tolerance,gd_initial_variance,seed,0);
            [G_em,updates_em,conv_em] = local_classical('EM-GS',Y,S,B,sigma2, ...
                maximum_classical_updates,convergence_tolerance,gd_initial_variance,seed,0);
            [G_hard,updates_hard,conv_hard] = local_hard_hsgs(Y,S,B,sigma2,p_H, ...
                hard_rank(snr_index),maximum_classical_updates,convergence_tolerance);

            % Paired DL/soft-HS comparison: one identical biased-GS estimate.
            [G0,gs_updates,gs_conv] = local_biased_gs(Y,S,B,maximum_gs_updates,convergence_tolerance);
            if mod(sample_index,2) == 0
                G_soft = local_soft_hsgs_16(G0,Y,S,B,sigma2,soft_multiplier(snr_index),p_H,number_of_stages);
                G_v4 = local_v4_forward(model.best_parameters_numeric,G0,Y,S,B,sigma2,p_H,N,K);
            else
                G_v4 = local_v4_forward(model.best_parameters_numeric,G0,Y,S,B,sigma2,p_H,N,K);
                G_soft = local_soft_hsgs_16(G0,Y,S,B,sigma2,soft_multiplier(snr_index),p_H,number_of_stages);
            end

            estimates = {G_gs,G_gd,G_em,G_hard,G_soft,G_v4};
            errors_one = zeros(1,6);
            for method_index = 1:6
                errors_one(method_index) = sum(abs(estimates{method_index}-G_true).^2,'all');
            end
            if any(~isfinite(errors_one)) || ~isfinite(sum(abs(G_true).^2,'all'))
                error('Nonfinite score at P=%d, sample=%d.',P,sample_index);
            end
            cell_error_energy(local_index,:) = errors_one;
            cell_truth_energy(local_index) = sum(abs(G_true).^2,'all');
            cell_updates(local_index,:) = [updates_gs updates_gd updates_em updates_hard];
            cell_converged(local_index,:) = [conv_gs conv_gd conv_em conv_hard];
            cell_gs_iterations(local_index) = gs_updates;
            cell_gs_converged(local_index) = gs_conv;
            cell_gd_seed(local_index) = seed;
            send(progress_queue,1);
        end
        fprintf(' complete.\n');
        error_energy(indices,:) = cell_error_energy;
        truth_energy(indices) = cell_truth_energy;
        updates(indices,:) = cell_updates;
        converged(indices,:) = cell_converged;
        gs_iterations_shared(indices) = cell_gs_iterations;
        gs_converged_shared(indices) = cell_gs_converged;
        gd_seed(indices) = cell_gd_seed;
    end

    condition_sample_table = table(repmat(P,sample_count,1),snr_by_sample,(1:sample_count)', ...
        truth_energy,error_energy(:,1),error_energy(:,2),error_energy(:,3),error_energy(:,4), ...
        error_energy(:,5),error_energy(:,6),updates(:,1),updates(:,2),updates(:,3),updates(:,4), ...
        converged(:,1),converged(:,2),converged(:,3),converged(:,4),gs_iterations_shared, ...
        gs_converged_shared,gd_seed, ...
        'VariableNames',{'P','SNR_dB','TestIndex','TruthEnergy','GS_ErrorEnergy','GD_ErrorEnergy', ...
        'EMGS_ErrorEnergy','HardRankHSGS_ErrorEnergy','SoftRankHSGS_ErrorEnergy','V4_ErrorEnergy', ...
        'GS_Updates','GD_Updates','EMGS_Updates','HardRankHSGS_Updates','GS_Converged', ...
        'GD_Converged','EMGS_Converged','HardRankHSGS_Converged','SharedGS_Updates', ...
        'SharedGS_Converged','GDSeed'});
    sample_tables{p_index} = condition_sample_table;

    for snr_index = 1:numel(SNR_grid)
        current_snr = SNR_grid(snr_index);
        selected = snr_by_sample==current_snr;
        count = nnz(selected);
        nmse_db = 10*log10(max(sum(error_energy(selected,:),1)/sum(truth_energy(selected)),realmin));
        condition_row = condition_row+1;
        condition_rows(condition_row,:) = [P current_snr count hard_rank(snr_index) ...
            soft_multiplier(snr_index) nmse_db];
        for method_index = 1:5
            [ci_low,ci_high] = local_paired_ci(error_energy(selected,method_index), ...
                error_energy(selected,6),truth_energy(selected),bootstrap_replicates, ...
                bootstrap_seed+10000*P+100*snr_index+method_index);
            ci_row = ci_row+1;
            if method_index <= 4
                mean_updates = mean(updates(selected,method_index));
                convergence_fraction = mean(converged(selected,method_index));
            else
                mean_updates = NaN;
                convergence_fraction = NaN;
            end
            ci_rows(ci_row,:) = [P current_snr method_index count ...
                nmse_db(6)-nmse_db(method_index) ci_low ci_high ...
                mean_updates convergence_fraction];
        end
    end

    %% Serial paired runtime benchmark; timing includes the common GS start.
    runtime_indices = [];
    for snr_index = 1:numel(SNR_grid)
        candidates = find(snr_by_sample==SNR_grid(snr_index));
        runtime_indices = [runtime_indices; candidates(1:runtime_samples_per_snr)]; %#ok<AGROW>
    end
    warm_index = runtime_indices(1);
    [Yw,~,Sw,Bw,sigma2w] = local_test_sample(test_data,warm_index);
    G0w = local_biased_gs(Yw,Sw,Bw,maximum_gs_updates,convergence_tolerance);
    local_soft_hsgs_16(G0w,Yw,Sw,Bw,sigma2w,soft_multiplier(1),p_H,number_of_stages);
    local_v4_forward(model.best_parameters_numeric,G0w,Yw,Sw,Bw,sigma2w,p_H,N,K);
    runtime_seconds = zeros(numel(runtime_indices),3);
    for runtime_index = 1:numel(runtime_indices)
        sample_index = runtime_indices(runtime_index);
        snr_index = find(SNR_grid==snr_by_sample(sample_index),1);
        [Yrt,~,Srt,Brt,sigma2rt] = local_test_sample(test_data,sample_index);
        timer = tic;
        G0rt = local_biased_gs(Yrt,Srt,Brt,maximum_gs_updates,convergence_tolerance);
        gs_time = toc(timer);
        soft_forward_total = 0;
        v4_forward_total = 0;
        for repeat_index = 1:runtime_repeats_per_method
            if mod(runtime_index+repeat_index,2) == 0
                timer = tic;
                local_soft_hsgs_16(G0rt,Yrt,Srt,Brt,sigma2rt,soft_multiplier(snr_index),p_H,number_of_stages);
                soft_forward_total = soft_forward_total+toc(timer);
                timer = tic;
                local_v4_forward(model.best_parameters_numeric,G0rt,Yrt,Srt,Brt,sigma2rt,p_H,N,K);
                v4_forward_total = v4_forward_total+toc(timer);
            else
                timer = tic;
                local_v4_forward(model.best_parameters_numeric,G0rt,Yrt,Srt,Brt,sigma2rt,p_H,N,K);
                v4_forward_total = v4_forward_total+toc(timer);
                timer = tic;
                local_soft_hsgs_16(G0rt,Yrt,Srt,Brt,sigma2rt,soft_multiplier(snr_index),p_H,number_of_stages);
                soft_forward_total = soft_forward_total+toc(timer);
            end
        end
        soft_forward_mean = soft_forward_total/runtime_repeats_per_method;
        v4_forward_mean = v4_forward_total/runtime_repeats_per_method;
        runtime_seconds(runtime_index,:) = [gs_time gs_time+soft_forward_mean gs_time+v4_forward_mean];
    end
    for runtime_method = 1:2
        runtime_row = runtime_row+1;
        if runtime_method == 1
            method_index = 5;
        else
            method_index = 6;
        end
        runtime_rows(runtime_row,:) = [P method_index numel(runtime_indices) ...
            runtime_repeats_per_method mean(runtime_seconds(:,1)) ...
            mean(runtime_seconds(:,runtime_method+1)-runtime_seconds(:,1)) ...
            mean(runtime_seconds(:,runtime_method+1)) median(runtime_seconds(:,runtime_method+1)) ...
            sum(runtime_seconds(:,runtime_method+1)) ...
            mean(gs_converged_shared(runtime_indices))];
    end
    fprintf('P=%d: shared biased-GS initialization converged on %.2f%% of samples.\n', ...
        P,100*mean(gs_converged_shared));
end

sample_table = vertcat(sample_tables{:});
condition_table = array2table(condition_rows,'VariableNames',{'P','SNR_dB','TestSamples', ...
    'FrozenHardRank','FrozenSoftThresholdMultiplier','GS_NMSE_dB','GD_NMSE_dB', ...
    'EMGS_NMSE_dB','HardRankHSGS_NMSE_dB','SoftRankHSGS_NMSE_dB','V4_NMSE_dB'});
ci_table = array2table(ci_rows,'VariableNames',{'P','SNR_dB','MethodIndex','TestSamples', ...
    'MethodGain_vs_V4_dB','PairedCI95Low_dB','PairedCI95High_dB', ...
    'MeanClassicalUpdates','ConvergedFraction'});
ci_table.Method = reshape(method_names(ci_table.MethodIndex),[],1);
ci_table = movevars(ci_table,'Method','After','MethodIndex');
runtime_table = array2table(runtime_rows,'VariableNames',{'P','MethodIndex', ...
    'RuntimeSamples','ForwardRepeatsPerMethod','MeanSharedGSInitSeconds', ...
    'MeanForwardSeconds','MeanTotalSecondsPerSample','MedianTotalSecondsPerSample', ...
    'SummedSampleSeconds','SharedGSConvergedFraction'});
runtime_table.Method = reshape(method_names(runtime_table.MethodIndex),[],1);
runtime_table = movevars(runtime_table,'Method','After','MethodIndex');

writetable(sample_table,fullfile(output_folder,'confirmatory_test_per_sample_scores.csv'));
writetable(condition_table,fullfile(output_folder,'confirmatory_test_nmse_by_condition.csv'));
writetable(ci_table,fullfile(output_folder,'confirmatory_test_paired_CI_vs_V4.csv'));
writetable(runtime_table,fullfile(output_folder,'confirmatory_test_total_runtime.csv'));
configuration = struct('dataset_release',dataset_release,'N',N,'K',K,'P_list',P_list, ...
    'SNR_grid',SNR_grid,'RSR_dB',RSR_dB,'p_H',p_H,'number_of_stages',number_of_stages, ...
    'maximum_gs_updates',maximum_gs_updates,'maximum_classical_updates',maximum_classical_updates, ...
    'convergence_tolerance',convergence_tolerance,'expected_samples_per_snr',expected_samples_per_snr, ...
    'runtime_samples_per_snr',runtime_samples_per_snr, ...
    'runtime_repeats_per_method',runtime_repeats_per_method, ...
    'bootstrap_replicates',bootstrap_replicates,'hard_policy_file',hard_policy_file, ...
    'soft_policy_file',soft_policy_file,'V4_checkpoint_folder',v4_model_folder);
save(fullfile(output_folder,'confirmatory_test_results.mat'),'sample_table','condition_table', ...
    'ci_table','runtime_table','configuration','method_names','hard_policy','soft_policy','-v7.3');

for p_index = 1:numel(P_list)
    P = P_list(p_index);
    selected = condition_table.P==P;
    figure('Color','w');
    hold on;
    values = condition_table{selected,6:11};
    markers = {'-o','-s','-^','-v','-d','-*'};
    for method_index = 1:6
        plot(SNR_grid,values(:,method_index),markers{method_index}, ...
            'LineWidth',1.5,'DisplayName',method_names{method_index});
    end
    grid on;
    xlabel('SNR (dB)');
    ylabel('Test NMSE (dB)');
    title(sprintf('Independent confirmatory test, P=%d, N=%d, K=%d, RSR=%g dB',P,N,K,RSR_dB));
    legend('Location','best');
    saveas(gcf,fullfile(output_folder,sprintf('confirmatory_test_six_methods_P%d.png',P)));
end

fprintf('\nFrozen confirmatory test evaluation complete. No model/policy was tuned.\n');
fprintf('Results: %s\n',output_folder);

function [Y,G_true,S,B,sigma2] = local_test_sample(data,index)
Y = double(data.Y_data(:,:,index));
G_true = double(data.G_data(:,:,index));
S = double(data.S_data(:,:,index));
B = double(data.B_data(:,:,index));
sigma2 = double(data.sigma2_data(index));
end

function [G,iteration_count,converged] = local_biased_gs(Y,S,B,max_updates,tolerance)
N = size(Y,1);
K = size(S,1);
G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).';
    b = B(antenna,:).';
    augmented = [S.' b];
    matrix = augmented'*diag(y)*augmented;
    [vectors,values] = eig(matrix);
    [~,index] = max(real(diag(values)));
    principal = vectors(:,index)/max(norm(vectors(:,index)),eps);
    scale = (abs(augmented*principal).'*y)/max(norm(augmented*principal)^2,eps);
    G(antenna,:) = (scale*exp(-1i*angle(principal(K+1)))*principal(1:K)).';
end
right_inverse = S'/(S*S');
iteration_count = max_updates;
converged = false;
for iteration = 1:max_updates
    previous = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration_count = iteration;
        converged = true;
        break;
    end
end
end

function G = local_em_update(G,Y,S,B,sigma2)
right_inverse = S'/(S*S');
field = G*S+B;
kappa = 2/max(sigma2,eps)*Y.*abs(field);
ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
G = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
end

function G = local_soft_hsgs_16(G,Y,S,B,sigma2,multiplier,p_H,stages)
right_inverse = S'/(S*S');
N = size(Y,1);
K = size(S,1);
noise_scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
threshold = multiplier*noise_scale;
for stage = 1:stages
    G = local_soft_hankel(local_em_update(G,Y,S,B,sigma2),p_H,threshold);
end
end

function G = local_soft_hankel(G,p_H,threshold)
N = size(G,1);
K = size(G,2);
rows = N-p_H;
columns = p_H+1;
for user = 1:K
    x = G(:,user);
    H = zeros(rows,columns);
    for row = 1:rows
        H(row,:) = x(row:row+columns-1).';
    end
    [U,Sigma,V] = svd(H,'econ');
    shrunk = max(diag(Sigma)-threshold,0);
    H = U*diag(shrunk)*V';
    G(:,user) = local_hankel_average(H,N,p_H);
end
end

function G = local_v4_forward(parameters,G,Y,S,B,sigma2,p_H,N,K)
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2,'all'),eps),eps))/10);
right_inverse = S'/(S*S');
noise_scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
for stage = 1:16
    previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G_em = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    G_hankel = zeros(N,K);
    for user = 1:K
        x = G_em(:,user);
        H = zeros(N-p_H,p_H+1);
        for row = 1:N-p_H
            H(row,:) = x(row:row+p_H).';
        end
        gram = H'*H;
        [V,D] = eig(gram);
        singular = sqrt(max(real(diag(D)),0));
        safe = max(singular,1e-12);
        relative = tanh(log(safe/max(mean(safe),1e-12)));
        logits = parameters(1,stage)+parameters(2,stage)*noise_feature+ ...
            parameters(3,stage)*relative;
        threshold = noise_scale*(2./(1+exp(-logits)));
        shrink = max(1-threshold./safe,0);
        H_shrunk = H*(V.*reshape(shrink,1,[]))*V';
        G_hankel(:,user) = local_hankel_average(H_shrunk,N,p_H);
    end
    em_change = tanh(log(1+sqrt(sum(abs(G_em-previous).^2,'all')/max(sum(abs(previous).^2,'all'),eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(G_hankel-G_em).^2,'all')/max(sum(abs(G_em).^2,'all'),eps)))/3);
    gate = reshape(parameters(4:15,stage),4,3);
    gate_logits = gate(1,:).'*noise_feature+gate(2,:).'*em_change+ ...
        gate(3,:).'*hankel_change+gate(4,:).';
    gate_logits = gate_logits-max(gate_logits);
    weights = exp(gate_logits)./sum(exp(gate_logits));
    G = weights(1)*previous+weights(2)*G_em+weights(3)*G_hankel;
end
end

function [G,iteration_count,converged] = local_classical(method,Y,S,B,sigma2,max_updates,tolerance,gd_variance,seed,unused)
method = upper(string(method));
N = size(Y,1);
K = size(S,1);
if method == "GD"
    rng(mod(double(seed)+720000,2^32-1),'twister');
    G = sqrt(gd_variance/2)*(randn(N,K)+1i*randn(N,K));
    phase_reference = exp(-1i*angle(B));
    step_size = 1/max(real(eig(S*S')));
else
    G = local_spectral_initialize(Y,S,B);
    right_inverse = S'/(S*S');
end
converged = false;
iteration_count = max_updates;
for iteration = 1:max_updates
    previous = G;
    if method == "GS"
        G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    elseif method == "GD"
        G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    else
        G = local_em_update(G,Y,S,B,sigma2);
    end
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration_count = iteration;
        converged = true;
        break;
    end
end
end

function [G,iteration_count,converged] = local_hard_hsgs(Y,S,B,sigma2,p_H,rank_value,max_updates,tolerance)
N = size(Y,1);
K = size(S,1);
G = local_spectral_initialize(Y,S,B);
iteration_count = max_updates;
converged = false;
for iteration = 1:max_updates
    previous = G;
    G_em = local_em_update(G,Y,S,B,sigma2);
    for user = 1:K
        H = local_hankel_embed(G_em(:,user),p_H);
        [U,Sigma,V] = svd(H,'econ');
        H = U(:,1:rank_value)*Sigma(1:rank_value,1:rank_value)*V(:,1:rank_value)';
        G(:,user) = local_hankel_average(H,N,p_H);
    end
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
    augmented = [S.' B(antenna,:).'];
    matrix = augmented'*diag(y)*augmented;
    [V,D] = eig(matrix);
    [~,idx] = max(real(diag(D)));
    vector = V(:,idx)/max(norm(V(:,idx)),eps);
    scale = (abs(augmented*vector).'*y)/max(norm(augmented*vector)^2,eps);
    G(antenna,:) = (scale*exp(-1i*angle(vector(K+1)))*vector(1:K)).';
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
        idx = row+column-1;
        x(idx) = x(idx)+H(row,column);
        counts(idx) = counts(idx)+1;
    end
end
x = x./counts;
end

function local_progress_tick(~)
persistent completed_sample_count
if isempty(completed_sample_count)
    completed_sample_count = 0;
end
completed_sample_count = completed_sample_count+1;
if mod(completed_sample_count,50) == 0
    fprintf(' %d samples scored.\n',completed_sample_count);
end
end

function [low,high] = local_paired_ci(method_error,v4_error,truth_energy,reps,seed)
rng(seed,'twister');
n = numel(truth_energy);
gain = zeros(reps,1);
for b = 1:reps
    selected = randi(n,n,1);
    method_nmse = sum(method_error(selected))/sum(truth_energy(selected));
    v4_nmse = sum(v4_error(selected))/sum(truth_energy(selected));
    gain(b) = 10*log10(max(v4_nmse,realmin))-10*log10(max(method_nmse,realmin));
end
interval = prctile(gain,[2.5 97.5]);
low = interval(1);
high = interval(2);
end
