%% HS_GS_V19_STOPPING_VALIDATION_FRESH_TEST.m
% Tune HS-GS rank, projection duration, and total EM update budget on a
% generated validation set, freeze one setting per P/SNR, then evaluate it
% on a newly generated paired test set. Test channel truth is used only for
% final scoring. Validation channel truth is used for offline hyperparameter
% selection; therefore the frozen schedule is SNR-conditioned, not a
% per-realization receiver-side adaptation rule.
%
% Test baselines: EM-GS, biased GS, and linearized GD run to the same
% relative-change tolerance used in V18. The selected HS-GS candidate uses
% its frozen finite update budget. All methods receive the same test sample.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V19_STOPPING_VALIDATION_FRESH_TEST
%
% Resume by running the same script. Checkpoints are saved in /results after
% each block. Do not delete an existing checkpoint or frozen-policy file.

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir'), mkdir(result_folder); end

validation_checkpoint_file = fullfile(result_folder, 'HS_GS_V19_validation_checkpoint.mat');
test_checkpoint_file = fullfile(result_folder, 'HS_GS_V19_fresh_test_checkpoint.mat');
freeze_file = fullfile(result_folder, 'HS_GS_V19_frozen_policy.mat');
result_file = fullfile(result_folder, 'HS_GS_V19_fresh_test_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V19_fresh_test_table.csv');
if isfile(result_file)
    error('V19 results already exist; inspect them before starting another run: %s', result_file);
end

% Fixed benchmark and simulation model, matching V17/V18.
P_list = [10 30];
SNR_dB = -5:5:30;
N = 8; K = 3; RSR_dB = 25; p_H = 3;
maximum_updates = 500;
cadzow_sweeps = 4;
samples_per_validation_point = 500;
samples_per_test_point = 500;
samples_per_checkpoint = 10;
validation_seed_base = 3400000000;
test_seed_base = 3600000000;
workers_if_starting_pool = 4;
d_over_lambda = 0.5;
mu_eg = [0; 1785.9 * 1.602e-19 * 5.292e-11; 0];
hbar = 6.62607015e-34 / (2*pi);
convergence_maximum_updates = 5000;
convergence_tolerance = 1e-10;

% Candidate columns: [Hankel rank, number of projected EM updates, total EM updates].
% Rank 4 with zero projections is the unprojected EM-GS control.
update_budgets = [25 50 100 200 500];
candidate_settings = zeros(0, 3);
for budget_index = 1:length(update_budgets)
    total_updates = update_budgets(budget_index);
    candidate_settings(end+1,:) = [4 0 total_updates]; %#ok<SAGROW>
    projection_durations = unique([min(5,total_updates), min(25,total_updates)]);
    for rank_value = [2 3]
        for projection_index = 1:length(projection_durations)
            candidate_settings(end+1,:) = [rank_value projection_durations(projection_index) total_updates]; %#ok<SAGROW>
        end
    end
end
candidate_settings = unique(candidate_settings, 'rows', 'stable');
number_of_candidates = size(candidate_settings, 1);

% Freeze policy if already selected. Otherwise resume validation from checkpoint.
if isfile(freeze_file)
    frozen = load(freeze_file);
    if ~isequal(frozen.P_list,P_list) || ~isequal(frozen.SNR_dB,SNR_dB) || ...
            ~isequal(frozen.candidate_settings,candidate_settings) || ...
            frozen.N~=N || frozen.K~=K || frozen.RSR_dB~=RSR_dB || ...
            frozen.p_H~=p_H || frozen.cadzow_sweeps~=cadzow_sweeps || ...
            frozen.samples_per_validation_point~=samples_per_validation_point || ...
            frozen.validation_seed_base~=validation_seed_base || ...
            frozen.samples_per_test_point~=samples_per_test_point || ...
            frozen.test_seed_base~=test_seed_base
        error('Existing V19 frozen policy settings differ. Preserve and inspect the file.');
    end
    selected_candidate = frozen.selected_candidate;
    selected_rank = frozen.selected_rank;
    selected_projection_updates = frozen.selected_projection_updates;
    selected_total_updates = frozen.selected_total_updates;
    validation_nmse_db = frozen.validation_nmse_db;
    validation_completed = frozen.validation_completed;
    fprintf('Loaded frozen V19 policy selected on validation data.\n');
else
    validation_completed = false(length(P_list),length(SNR_dB),samples_per_validation_point);
    validation_squared_error = nan(length(P_list),length(SNR_dB),samples_per_validation_point,number_of_candidates);
    validation_channel_energy = nan(length(P_list),length(SNR_dB),samples_per_validation_point);
    if isfile(validation_checkpoint_file)
        C = load(validation_checkpoint_file);
        if ~isequal(C.P_list,P_list) || ~isequal(C.SNR_dB,SNR_dB) || ...
                ~isequal(C.candidate_settings,candidate_settings) || C.N~=N || C.K~=K || ...
                C.RSR_dB~=RSR_dB || C.p_H~=p_H || ...
                C.samples_per_validation_point~=samples_per_validation_point || ...
                C.validation_seed_base~=validation_seed_base
            error('V19 validation checkpoint settings mismatch. Preserve and inspect it.');
        end
        validation_completed = C.validation_completed;
        validation_squared_error = C.validation_squared_error;
        validation_channel_energy = C.validation_channel_energy;
        fprintf('Resuming validation: %d/%d samples complete.\n', ...
            sum(validation_completed,'all'),numel(validation_completed));
    end

    pool = gcp('nocreate');
    if isempty(pool), parpool('local',workers_if_starting_pool); pool=gcp('nocreate'); end
    fprintf('V19 validation: %d workers, %d candidates, %d samples per P/SNR.\n', ...
        pool.NumWorkers,number_of_candidates,samples_per_validation_point);
    for P_index = 1:length(P_list)
        P = P_list(P_index);
        for SNR_index = 1:length(SNR_dB)
            point_timer = tic;
            for block_start = 1:samples_per_checkpoint:samples_per_validation_point
                positions = block_start:min(block_start+samples_per_checkpoint-1,samples_per_validation_point);
                is_done = reshape(validation_completed(P_index,SNR_index,positions),1,[]);
                pending = positions(~is_done);
                if isempty(pending), continue; end
                block_errors = zeros(length(pending),number_of_candidates);
                block_energy = zeros(length(pending),1);
                parfor local_index = 1:length(pending)
                    sample_index = pending(local_index);
                    seed = uint32(validation_seed_base + 1000000*P + 10000*SNR_index + sample_index);
                    [Y,G,S,B,sigma2] = v19_generate_sample(N,K,P,SNR_dB(SNR_index), ...
                        RSR_dB,d_over_lambda,mu_eg,hbar,seed);
                    sample_errors = zeros(1,number_of_candidates);
                    for candidate_index = 1:number_of_candidates
                        settings = candidate_settings(candidate_index,:);
                        estimate = v19_hybrid_em(Y,S,B,sigma2,settings(1),settings(2), ...
                            settings(3),p_H,cadzow_sweeps);
                        sample_errors(candidate_index) = norm(estimate-G,'fro')^2;
                    end
                    block_errors(local_index,:) = sample_errors;
                    block_energy(local_index) = norm(G,'fro')^2;
                end
                validation_squared_error(P_index,SNR_index,pending,:) = ...
                    reshape(block_errors,1,1,length(pending),number_of_candidates);
                validation_channel_energy(P_index,SNR_index,pending) = reshape(block_energy,1,1,[]);
                validation_completed(P_index,SNR_index,pending) = true;
                save(validation_checkpoint_file,'P_list','SNR_dB','N','K','RSR_dB','p_H', ...
                    'candidate_settings','cadzow_sweeps','samples_per_validation_point', ...
                    'validation_seed_base','validation_completed','validation_squared_error', ...
                    'validation_channel_energy','-v7.3');
                fprintf('Validation P=%d, SNR=%d: %d/%d samples, %.1f s.\n', ...
                    P,SNR_dB(SNR_index),sum(validation_completed(P_index,SNR_index,:),'all'), ...
                    samples_per_validation_point,toc(point_timer));
            end
        end
    end
    if ~all(validation_completed,'all') || any(~isfinite(validation_squared_error),'all') || ...
            any(~isfinite(validation_channel_energy),'all') || any(validation_channel_energy<=0,'all')
        error('V19 validation incomplete or invalid; checkpoint preserved.');
    end

    validation_nmse_db = zeros(length(P_list),length(SNR_dB),number_of_candidates);
    selected_candidate = zeros(length(P_list),length(SNR_dB));
    selected_rank = zeros(length(P_list),length(SNR_dB));
    selected_projection_updates = zeros(length(P_list),length(SNR_dB));
    selected_total_updates = zeros(length(P_list),length(SNR_dB));
    for P_index = 1:length(P_list)
        for SNR_index = 1:length(SNR_dB)
            channel_energy = sum(reshape(validation_channel_energy(P_index,SNR_index,:),[],1));
            errors = reshape(validation_squared_error(P_index,SNR_index,:,:), ...
                samples_per_validation_point,number_of_candidates);
            point_nmse_db = 10*log10(sum(errors,1)/channel_energy);
            validation_nmse_db(P_index,SNR_index,:) = reshape(point_nmse_db,1,1,[]);
            [~,best_candidate] = min(point_nmse_db);
            selected_candidate(P_index,SNR_index) = best_candidate;
            selected_rank(P_index,SNR_index) = candidate_settings(best_candidate,1);
            selected_projection_updates(P_index,SNR_index) = candidate_settings(best_candidate,2);
            selected_total_updates(P_index,SNR_index) = candidate_settings(best_candidate,3);
        end
    end
    policy_source = ['Per-P/SNR minimum validation ratio-of-sums NMSE; validation uses simulated channel truth ' ...
        'for offline selection only.'];
    save(freeze_file,'P_list','SNR_dB','N','K','RSR_dB','p_H','candidate_settings', ...
        'cadzow_sweeps','samples_per_validation_point','validation_seed_base', ...
        'samples_per_test_point','test_seed_base','selected_candidate','selected_rank', ...
        'selected_projection_updates','selected_total_updates','validation_nmse_db', ...
        'validation_completed','policy_source');
    fprintf('Frozen validation policy at %s\n',freeze_file);
end

policy_rows=zeros(length(P_list)*length(SNR_dB),5);
policy_row=0;
for P_index=1:length(P_list)
    for SNR_index=1:length(SNR_dB)
        policy_row=policy_row+1;
        policy_rows(policy_row,:)=[P_list(P_index),SNR_dB(SNR_index), ...
            selected_rank(P_index,SNR_index),selected_projection_updates(P_index,SNR_index), ...
            selected_total_updates(P_index,SNR_index)];
    end
end
policy_table=array2table(policy_rows,'VariableNames', ...
    {'P','SNR_dB','Selected_rank','Projected_updates','Total_updates'});
disp(policy_table);

% New paired test seeds; the frozen HS-GS rule is never changed on test data.
test_completed = false(length(P_list),length(SNR_dB),samples_per_test_point);
method_names = {'HS_GS_frozen_validation','EM_GS_converged','biased_GS_converged','GD_converged'};
test_squared_error = nan(length(P_list),length(SNR_dB),samples_per_test_point,length(method_names));
test_channel_energy = nan(length(P_list),length(SNR_dB),samples_per_test_point);
test_iterations = zeros(length(P_list),length(SNR_dB),samples_per_test_point,4);
test_converged = false(length(P_list),length(SNR_dB),samples_per_test_point,4);
if isfile(test_checkpoint_file)
    C = load(test_checkpoint_file);
    if ~isequal(C.P_list,P_list) || ~isequal(C.SNR_dB,SNR_dB) || ...
            ~isequal(C.selected_candidate,selected_candidate) || ...
            C.samples_per_test_point~=samples_per_test_point || C.test_seed_base~=test_seed_base || ...
            C.N~=N || C.K~=K || C.RSR_dB~=RSR_dB || C.p_H~=p_H || ...
            C.cadzow_sweeps~=cadzow_sweeps || ...
            C.convergence_maximum_updates~=convergence_maximum_updates || ...
            C.convergence_tolerance~=convergence_tolerance
        error('V19 test checkpoint settings mismatch. Preserve and inspect it.');
    end
    test_completed=C.test_completed; test_squared_error=C.test_squared_error;
    test_channel_energy=C.test_channel_energy; test_iterations=C.test_iterations;
    test_converged=C.test_converged;
    fprintf('Resuming fresh test: %d/%d samples complete.\n',sum(test_completed,'all'),numel(test_completed));
end
pool = gcp('nocreate');
if isempty(pool), parpool('local',workers_if_starting_pool); pool=gcp('nocreate'); end
fprintf('V19 fresh paired test: %d workers, %d samples per P/SNR.\n', ...
    pool.NumWorkers,samples_per_test_point);
for P_index = 1:length(P_list)
    P=P_list(P_index);
    for SNR_index = 1:length(SNR_dB)
        point_timer=tic;
        for block_start=1:samples_per_checkpoint:samples_per_test_point
            positions=block_start:min(block_start+samples_per_checkpoint-1,samples_per_test_point);
            is_done=reshape(test_completed(P_index,SNR_index,positions),1,[]);
            pending=positions(~is_done);
            if isempty(pending), continue; end
            block_errors=zeros(length(pending),4);
            block_energy=zeros(length(pending),1);
            block_iterations=zeros(length(pending),4);
            block_converged=false(length(pending),4);
            r=selected_rank(P_index,SNR_index);
            n_projected=selected_projection_updates(P_index,SNR_index);
            n_total=selected_total_updates(P_index,SNR_index);
            parfor local_index=1:length(pending)
                sample_index=pending(local_index);
                seed=uint32(test_seed_base+1000000*P+10000*SNR_index+sample_index);
                [Y,G,S,B,sigma2]=v19_generate_sample(N,K,P,SNR_dB(SNR_index), ...
                    RSR_dB,d_over_lambda,mu_eg,hbar,seed);
                [G_hs,hs_iterations]=v19_hybrid_em(Y,S,B,sigma2,r,n_projected,n_total,p_H,cadzow_sweeps);
                [G_em,em_iterations,em_ok]=v19_em_converged(Y,S,B,sigma2, ...
                    convergence_maximum_updates,convergence_tolerance);
                [G_gs,gs_iterations,gs_ok]=v19_gs_converged(Y,S,B, ...
                    convergence_maximum_updates,convergence_tolerance);
                [G_gd,gd_iterations,gd_ok]=v19_gd_converged(Y,S,B, ...
                    convergence_maximum_updates,convergence_tolerance,0.1,seed);
                block_errors(local_index,:)=[norm(G_hs-G,'fro')^2,norm(G_em-G,'fro')^2, ...
                    norm(G_gs-G,'fro')^2,norm(G_gd-G,'fro')^2];
                block_energy(local_index)=norm(G,'fro')^2;
                block_iterations(local_index,:)=[hs_iterations,em_iterations,gs_iterations,gd_iterations];
                block_converged(local_index,:)=[true,em_ok,gs_ok,gd_ok];
            end
            test_squared_error(P_index,SNR_index,pending,:)=reshape(block_errors,1,1,length(pending),4);
            test_channel_energy(P_index,SNR_index,pending)=reshape(block_energy,1,1,[]);
            test_iterations(P_index,SNR_index,pending,:)=reshape(block_iterations,1,1,length(pending),4);
            test_converged(P_index,SNR_index,pending,:)=reshape(block_converged,1,1,length(pending),4);
            test_completed(P_index,SNR_index,pending)=true;
            save(test_checkpoint_file,'P_list','SNR_dB','N','K','RSR_dB','p_H', ...
                'cadzow_sweeps', ...
                'selected_candidate','selected_rank','selected_projection_updates','selected_total_updates', ...
                'samples_per_test_point','test_seed_base','convergence_maximum_updates', ...
                'convergence_tolerance','method_names','test_completed','test_squared_error', ...
                'test_channel_energy','test_iterations','test_converged','-v7.3');
            fprintf('Test P=%d, SNR=%d: %d/%d samples, %.1f s.\n',P,SNR_dB(SNR_index), ...
                sum(test_completed(P_index,SNR_index,:),'all'),samples_per_test_point,toc(point_timer));
        end
    end
end
if ~all(test_completed,'all') || any(~isfinite(test_squared_error),'all') || ...
        any(~isfinite(test_channel_energy),'all') || any(test_channel_energy<=0,'all')
    error('V19 fresh test incomplete or invalid; checkpoint preserved.');
end

% Ratio-of-sums NMSE and paired bootstrap intervals for HS-GS minus each baseline.
bootstrap_replicates=2000;
rows=zeros(length(P_list)*length(SNR_dB),26);
rng(2026102519,'twister'); row_number=0;
for P_index=1:length(P_list)
    for SNR_index=1:length(SNR_dB)
        row_number=row_number+1;
        energy=reshape(test_channel_energy(P_index,SNR_index,:),[],1);
        errors=reshape(test_squared_error(P_index,SNR_index,:,:),samples_per_test_point,4);
        nmse_db=10*log10(sum(errors,1)/sum(energy));
        gains=zeros(1,3); ci_low=zeros(1,3); ci_high=zeros(1,3);
        for baseline=2:4
            gains(baseline-1)=nmse_db(baseline)-nmse_db(1); % positive means HS-GS is better
            bootstrap_gain=zeros(bootstrap_replicates,1);
            for draw=1:bootstrap_replicates
                indices=randi(samples_per_test_point,samples_per_test_point,1);
                hs_ratio=sum(errors(indices,1))/sum(energy(indices));
                base_ratio=sum(errors(indices,baseline))/sum(energy(indices));
                bootstrap_gain(draw)=10*log10(base_ratio/hs_ratio);
            end
            ci=prctile(bootstrap_gain,[2.5 97.5]);
            ci_low(baseline-1)=ci(1); ci_high(baseline-1)=ci(2);
        end
        mean_iterations=reshape(mean(test_iterations(P_index,SNR_index,:,:),3),1,4);
        convergence_fraction=reshape(mean(test_converged(P_index,SNR_index,:,:),3),1,4);
        rows(row_number,:)=[P_list(P_index),SNR_dB(SNR_index),selected_rank(P_index,SNR_index), ...
            selected_projection_updates(P_index,SNR_index),selected_total_updates(P_index,SNR_index), ...
            nmse_db,gains,ci_low,ci_high,mean_iterations,convergence_fraction];
    end
end
result_table=array2table(rows,'VariableNames',{'P','SNR_dB','HS_rank','HS_projected_updates', ...
    'HS_total_updates','HS_GS_NMSE_dB','EM_GS_NMSE_dB','biased_GS_NMSE_dB','GD_NMSE_dB', ...
    'gain_over_EM_dB','gain_over_GS_dB','gain_over_GD_dB','CI_low_EM','CI_low_GS','CI_low_GD', ...
    'CI_high_EM','CI_high_GS','CI_high_GD','mean_iter_HS','mean_iter_EM','mean_iter_GS', ...
    'mean_iter_GD','completed_budget_fraction_HS','converged_fraction_EM', ...
    'converged_fraction_GS','converged_fraction_GD'});
save(result_file,'result_table','P_list','SNR_dB','candidate_settings','selected_candidate', ...
    'selected_rank','selected_projection_updates','selected_total_updates','samples_per_validation_point', ...
    'samples_per_test_point','validation_seed_base','test_seed_base','convergence_maximum_updates', ...
    'convergence_tolerance','bootstrap_replicates','validation_nmse_db','test_squared_error', ...
    'test_channel_energy','test_iterations','test_converged','method_names','-v7.3');
writetable(result_table,table_file);
disp(result_table);
fprintf('V19 frozen-policy paired test saved to %s\n',result_file);

function [Y,G,S,B,sigma2]=v19_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
rng(double(seed),'twister'); G=zeros(N,K);
for k=1:K
    path_count=randi([3 7]);
    for path_index=1:path_count
        alpha=(randn+1j*randn)/sqrt(2); theta=2*pi*rand;
        phi=2*pi*d_over_lambda*cos(theta); polarization=sqrt(1/3)*randn(3,1);
        for antenna=1:N
            G(antenna,k)=G(antenna,k)+(mu_eg.'*polarization/hbar)*alpha*exp(-1j*(antenna-1)*phi);
        end
    end
end
S=(randn(K,P)+1j*randn(K,P))/sqrt(2);
alpha_b=sqrt(10/2)*(randn+1j*randn); theta_b=2*pi*rand;
phi_b=2*pi*d_over_lambda*cos(theta_b); polarization_b=sqrt(1/3)*randn(3,1); b=zeros(N,1);
for antenna=1:N
    b(antenna)=(mu_eg.'*polarization_b/hbar)*alpha_b*exp(-1j*(antenna-1)*phi_b);
end
B=b*ones(1,P); average_signal_power=0;
for k=1:K
    average_signal_power=average_signal_power+mean(abs(G(:,k)*S(k,:)).^2,'all')/K;
end
B=B*sqrt(10^(RSR_db/10)*average_signal_power/mean(abs(B).^2,'all'));
sigma2=mean(abs(G*S).^2,'all')/10^(snr_db/10);
W=sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P)); Y=abs(G*S+B+W);
end

function [G,iteration]=v19_hybrid_em(Y,S,B,sigma2,r,n_projected,total_updates,p_H,sweeps)
G=v19_spectral_initialize(Y,S,B); right_inverse=S'/(S*S');
for iteration=1:total_updates
    field=G*S+B; kappa=2/max(sigma2,eps)*Y.*abs(field);
    ratio=besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G=(Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if iteration<=n_projected, G=v19_hankel_project(G,p_H,r,sweeps); end
end
end

function [G,iteration,converged]=v19_em_converged(Y,S,B,sigma2,max_updates,tolerance)
G=v19_spectral_initialize(Y,S,B); right_inverse=S'/(S*S'); converged=false; iteration=max_updates;
for n=1:max_updates
    old=G; field=G*S+B; kappa=2/max(sigma2,eps)*Y.*abs(field);
    ratio=besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G=(Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if norm(G-old,'fro')^2/max(norm(G,'fro')^2,eps)<tolerance
        iteration=n; converged=true; break;
    end
end
end

function [G,iteration,converged]=v19_gs_converged(Y,S,B,max_updates,tolerance)
G=v19_spectral_initialize(Y,S,B); right_inverse=S'/(S*S'); converged=false; iteration=max_updates;
for n=1:max_updates
    old=G; G=(Y.*exp(1j*angle(G*S+B))-B)*right_inverse;
    if norm(G-old,'fro')^2/max(norm(G,'fro')^2,eps)<tolerance
        iteration=n; converged=true; break;
    end
end
end

function [G,iteration,converged]=v19_gd_converged(Y,S,B,max_updates,tolerance,initial_variance,seed)
rng(mod(double(seed)+720000,2^32-1),'twister'); N=size(Y,1); K=size(S,1);
G=sqrt(initial_variance/2)*(randn(N,K)+1j*randn(N,K)); Z=exp(-1j*angle(B));
step=1/max(real(eig(S*S'))); converged=false; iteration=max_updates;
for n=1:max_updates
    old=G; G=G+step*((Y-abs(B)-real(Z.*(G*S))).*conj(Z))*S';
    if norm(G-old,'fro')^2/max(norm(G,'fro')^2,eps)<tolerance
        iteration=n; converged=true; break;
    end
end
end

function G=v19_spectral_initialize(Y,S,B)
N=size(Y,1); K=size(S,1); G=zeros(N,K);
for antenna=1:N
    y=Y(antenna,:).'; b=B(antenna,:).'; A=[S.',b]; M=A'*diag(y)*A;
    [V,D]=eig(M); [~,index]=max(real(diag(D))); v=V(:,index); v=v/max(norm(v),eps);
    scale=(abs(A*v).'*y)/max(norm(A*v)^2,eps);
    G(antenna,:)=(scale*exp(-1j*angle(v(K+1)))*v(1:K)).';
end
end

function G=v19_hankel_project(G,p_H,r,sweeps)
for sweep=1:sweeps
    for user=1:size(G,2)
        x=G(:,user); N=length(x); number_of_rows=N-p_H; number_of_columns=p_H+1;
        H=zeros(number_of_rows,number_of_columns);
        for row=1:number_of_rows, H(row,:)=x(row:row+number_of_columns-1).'; end
        [U,Sigma,V]=svd(H,'econ'); H_r=U(:,1:r)*Sigma(1:r,1:r)*V(:,1:r)';
        reconstructed=zeros(N,1); counts=zeros(N,1);
        for row=1:number_of_rows
            for column=1:number_of_columns
                index=row+column-1; reconstructed(index)=reconstructed(index)+H_r(row,column);
                counts(index)=counts(index)+1;
            end
        end
        G(:,user)=reconstructed./counts;
    end
end
end
