%% HS_GS_V17_EARLY_PROJECTION_VALIDATION_FRESH_TEST.m
% Tune early Hankel projection on a generated validation set, freeze the
% selected schedule, then evaluate it on distinct fresh paired test samples.
% The channel truth from test samples is used only for NMSE scoring.
%
% Candidates are [rank, projected EM updates]. [4,0] is plain EM-GS.
% Later EM updates are unconstrained, testing whether early Hankel denoising
% can help without retaining high-SNR low-rank projection bias.
%
% Run from MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V17_EARLY_PROJECTION_VALIDATION_FRESH_TEST

clear;
clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder, 'results');
if ~exist(result_folder, 'dir'), mkdir(result_folder); end

validation_checkpoint_file = fullfile(result_folder, 'HS_GS_V17_validation_checkpoint.mat');
test_checkpoint_file = fullfile(result_folder, 'HS_GS_V17_fresh_test_checkpoint.mat');
freeze_file = fullfile(result_folder, 'HS_GS_V17_frozen_policy.mat');
result_file = fullfile(result_folder, 'HS_GS_V17_fresh_test_results.mat');
table_file = fullfile(result_folder, 'HS_GS_V17_fresh_test_table.csv');
if isfile(result_file)
    error('V17 results already exist; inspect them before starting another run: %s', result_file);
end

P_list = [10 30]; SNR_dB = -5:5:30; N = 8; K = 3; RSR_dB = 25; p_H = 3;
number_of_updates = 100; cadzow_sweeps = 4; samples_per_point = 100;
samples_per_checkpoint = 10; validation_seed_base = 3000000000;
test_seed_base = 3200000000;
workers_if_starting_pool = 4; pilot_spacing = 0.5;
mu_eg = [0; 1785.9 * 1.602e-19 * 5.292e-11; 0];
hbar = 6.62607015e-34 / (2*pi);
candidate_settings = [4 0; 2 1; 2 5; 2 10; 2 25; 2 50; 2 100; ...
                      3 1; 3 5; 3 10; 3 25; 3 50; 3 100];
number_of_candidates = size(candidate_settings, 1);

% Validation phase. Each generated sample is evaluated with every candidate.
if isfile(freeze_file)
    frozen = load(freeze_file);
    if ~isequal(frozen.P_list, P_list) || ~isequal(frozen.SNR_dB, SNR_dB) || ...
            ~isequal(frozen.candidate_settings, candidate_settings) || ...
            frozen.N ~= N || frozen.K ~= K || frozen.RSR_dB ~= RSR_dB || ...
            frozen.p_H ~= p_H || frozen.number_of_updates ~= number_of_updates || ...
            frozen.cadzow_sweeps ~= cadzow_sweeps || ...
            frozen.samples_per_point ~= samples_per_point || ...
            frozen.validation_seed_base ~= validation_seed_base || ...
            frozen.test_seed_base ~= test_seed_base
        error('Existing V17 policy settings differ. Preserve and inspect the freeze file.');
    end
    selected_candidate = frozen.selected_candidate;
    selected_rank = frozen.selected_rank;
    selected_projection_updates = frozen.selected_projection_updates;
    validation_nmse_db = frozen.validation_nmse_db;
    validation_completed = frozen.validation_completed;
    validation_squared_error = frozen.validation_squared_error;
    validation_channel_energy = frozen.validation_channel_energy;
    fprintf('Loaded the previously frozen V17 validation policy.\n');
else
    validation_completed = false(length(P_list), length(SNR_dB), samples_per_point);
    validation_squared_error = nan(length(P_list), length(SNR_dB), samples_per_point, number_of_candidates);
    validation_channel_energy = nan(length(P_list), length(SNR_dB), samples_per_point);
    if isfile(validation_checkpoint_file)
        checkpoint = load(validation_checkpoint_file);
        if ~isequal(checkpoint.P_list, P_list) || ~isequal(checkpoint.SNR_dB, SNR_dB) || ...
                ~isequal(checkpoint.candidate_settings, candidate_settings) || ...
                checkpoint.N ~= N || checkpoint.K ~= K || checkpoint.RSR_dB ~= RSR_dB || ...
                checkpoint.p_H ~= p_H || checkpoint.number_of_updates ~= number_of_updates || ...
                checkpoint.cadzow_sweeps ~= cadzow_sweeps || ...
                checkpoint.samples_per_point ~= samples_per_point || ...
                checkpoint.validation_seed_base ~= validation_seed_base
            error('V17 validation checkpoint settings mismatch. Preserve and inspect it.');
        end
        validation_completed = checkpoint.validation_completed;
        validation_squared_error = checkpoint.validation_squared_error;
        validation_channel_energy = checkpoint.validation_channel_energy;
        fprintf('Resuming validation: %d/%d samples complete.\n', ...
            sum(validation_completed, 'all'), numel(validation_completed));
    end
    pool = gcp('nocreate');
    if isempty(pool), parpool('local', workers_if_starting_pool); pool = gcp('nocreate'); end
    fprintf('V17 validation: %d workers, %d candidates, %d samples per P/SNR.\n', ...
        pool.NumWorkers, number_of_candidates, samples_per_point);

    for P_index = 1:length(P_list)
        P = P_list(P_index);
        for SNR_index = 1:length(SNR_dB)
            point_timer = tic;
            for block_start = 1:samples_per_checkpoint:samples_per_point
                block_positions = block_start:min(block_start+samples_per_checkpoint-1, samples_per_point);
                block_done = reshape(validation_completed(P_index,SNR_index,block_positions),1,[]);
                pending_positions = block_positions(~block_done);
                if isempty(pending_positions), continue; end
                block_errors = zeros(length(pending_positions), number_of_candidates);
                block_energy = zeros(length(pending_positions),1);
                parfor local_index = 1:length(pending_positions)
                    sample_index = pending_positions(local_index);
                    sample_seed = uint32(validation_seed_base + 1000000*P + 10000*SNR_index + sample_index);
                    [Y,G,S,B,sigma2] = v17_generate_sample(N,K,P,SNR_dB(SNR_index), ...
                        RSR_dB,pilot_spacing,mu_eg,hbar,sample_seed);
                    errors_for_sample = zeros(1,number_of_candidates);
                    for candidate_index = 1:number_of_candidates
                        r = candidate_settings(candidate_index,1);
                        n_projected = candidate_settings(candidate_index,2);
                        estimate = v17_hybrid_em(Y,S,B,sigma2,number_of_updates, ...
                            cadzow_sweeps,r,n_projected,p_H);
                        errors_for_sample(candidate_index) = norm(estimate-G,'fro')^2;
                    end
                    block_errors(local_index,:) = errors_for_sample;
                    block_energy(local_index) = norm(G,'fro')^2;
                end
                validation_squared_error(P_index,SNR_index,pending_positions,:) = ...
                    reshape(block_errors,1,1,length(pending_positions),number_of_candidates);
                validation_channel_energy(P_index,SNR_index,pending_positions) = reshape(block_energy,1,1,[]);
                validation_completed(P_index,SNR_index,pending_positions) = true;
                save(validation_checkpoint_file,'P_list','SNR_dB','N','K','RSR_dB','p_H', ...
                    'candidate_settings','number_of_updates','cadzow_sweeps','samples_per_point', ...
                    'validation_seed_base','validation_completed','validation_squared_error', ...
                    'validation_channel_energy');
                fprintf('Validation P=%d, SNR=%d: %d/%d samples (%.1f s).\n',P,SNR_dB(SNR_index), ...
                    sum(validation_completed(P_index,SNR_index,:),'all'),samples_per_point,toc(point_timer));
            end
        end
    end
    if ~all(validation_completed,'all') || any(~isfinite(validation_squared_error),'all') || ...
            any(~isfinite(validation_channel_energy),'all') || any(validation_channel_energy<=0,'all')
        error('Validation phase incomplete or invalid; checkpoint preserved.');
    end
    validation_nmse_db = zeros(length(P_list),length(SNR_dB),number_of_candidates);
    selected_candidate = zeros(length(P_list),length(SNR_dB));
    selected_rank = zeros(length(P_list),length(SNR_dB));
    selected_projection_updates = zeros(length(P_list),length(SNR_dB));
    for P_index = 1:length(P_list)
        for SNR_index = 1:length(SNR_dB)
            energy = sum(reshape(validation_channel_energy(P_index,SNR_index,:),[],1));
            errors = reshape(validation_squared_error(P_index,SNR_index,:,:),samples_per_point,number_of_candidates);
            validation_nmse_db(P_index,SNR_index,:) = 10*log10(sum(errors,1)/energy);
            [~,best] = min(validation_nmse_db(P_index,SNR_index,:));
            selected_candidate(P_index,SNR_index) = best;
            selected_rank(P_index,SNR_index) = candidate_settings(best,1);
            selected_projection_updates(P_index,SNR_index) = candidate_settings(best,2);
        end
    end
    policy_source = 'Generated validation samples; minimum ratio-of-sums NMSE per P/SNR';
    save(freeze_file,'P_list','SNR_dB','N','K','RSR_dB','p_H','candidate_settings', ...
        'number_of_updates','cadzow_sweeps','samples_per_point','validation_seed_base', ...
        'test_seed_base','selected_candidate','selected_rank','selected_projection_updates', ...
        'validation_nmse_db','validation_completed','validation_squared_error', ...
        'validation_channel_energy','policy_source');
    fprintf('Frozen validation policy at %s\n',freeze_file);
end

policy_table = array2table([repelem(P_list(:),length(SNR_dB)), ...
    repmat(SNR_dB(:),length(P_list),1),selected_rank(:),selected_projection_updates(:)], ...
    'VariableNames',{'P','SNR_dB','Selected_rank','Projected_EM_updates'});
disp(policy_table);

% Fresh paired test phase: the frozen choices are not changed using test data.
test_completed = false(length(P_list),length(SNR_dB),samples_per_point);
method_names = {'HS_GS_hybrid','EM_GS','biased_GS','linearized_GD'};
test_squared_error = nan(length(P_list),length(SNR_dB),samples_per_point,length(method_names));
test_channel_energy = nan(length(P_list),length(SNR_dB),samples_per_point);
if isfile(test_checkpoint_file)
    checkpoint = load(test_checkpoint_file);
    if ~isequal(checkpoint.P_list,P_list) || ~isequal(checkpoint.SNR_dB,SNR_dB) || ...
            ~isequal(checkpoint.selected_rank,selected_rank) || ...
            ~isequal(checkpoint.selected_projection_updates,selected_projection_updates) || ...
            checkpoint.N ~= N || checkpoint.K ~= K || checkpoint.RSR_dB ~= RSR_dB || ...
            checkpoint.p_H ~= p_H || checkpoint.number_of_updates ~= number_of_updates || ...
            checkpoint.cadzow_sweeps ~= cadzow_sweeps || ...
            checkpoint.samples_per_point~=samples_per_point || checkpoint.test_seed_base~=test_seed_base
        error('V17 test checkpoint settings mismatch. Preserve and inspect it.');
    end
    test_completed = checkpoint.test_completed;
    test_squared_error = checkpoint.test_squared_error;
    test_channel_energy = checkpoint.test_channel_energy;
    fprintf('Resuming fresh test: %d/%d samples complete.\n',sum(test_completed,'all'),numel(test_completed));
end
pool = gcp('nocreate');
if isempty(pool), parpool('local',workers_if_starting_pool); pool=gcp('nocreate'); end
fprintf('V17 fresh test using %d workers, %d samples per P/SNR.\n',pool.NumWorkers,samples_per_point);
for P_index = 1:length(P_list)
    P=P_list(P_index);
    for SNR_index = 1:length(SNR_dB)
        point_timer=tic;
        for block_start=1:samples_per_checkpoint:samples_per_point
            block_positions=block_start:min(block_start+samples_per_checkpoint-1,samples_per_point);
            block_done=reshape(test_completed(P_index,SNR_index,block_positions),1,[]);
            pending_positions=block_positions(~block_done);
            if isempty(pending_positions), continue; end
            block_errors=zeros(length(pending_positions),length(method_names));
            block_energy=zeros(length(pending_positions),1);
            block_rank=selected_rank(P_index,SNR_index);
            block_projection_updates=selected_projection_updates(P_index,SNR_index);
            parfor local_index=1:length(pending_positions)
                sample_index=pending_positions(local_index);
                sample_seed=uint32(test_seed_base+1000000*P+10000*SNR_index+sample_index);
                [Y,G,S,B,sigma2]=v17_generate_sample(N,K,P,SNR_dB(SNR_index), ...
                    RSR_dB,pilot_spacing,mu_eg,hbar,sample_seed);
                G_hybrid=v17_hybrid_em(Y,S,B,sigma2,number_of_updates,cadzow_sweeps, ...
                    block_rank,block_projection_updates,p_H);
                G_em=v17_em_gs(Y,S,B,sigma2,number_of_updates);
                G_gs=v17_biased_gs(Y,S,B,1000,1e-10);
                G_gd=v17_linearized_gd(Y,S,B,1000,1e-10,0.1,sample_seed);
                block_errors(local_index,:)=[norm(G_hybrid-G,'fro')^2,norm(G_em-G,'fro')^2, ...
                    norm(G_gs-G,'fro')^2,norm(G_gd-G,'fro')^2];
                block_energy(local_index)=norm(G,'fro')^2;
            end
            test_squared_error(P_index,SNR_index,pending_positions,:)= ...
                reshape(block_errors,1,1,length(pending_positions),length(method_names));
            test_channel_energy(P_index,SNR_index,pending_positions)=reshape(block_energy,1,1,[]);
            test_completed(P_index,SNR_index,pending_positions)=true;
            save(test_checkpoint_file,'P_list','SNR_dB','N','K','RSR_dB','p_H','selected_rank', ...
                'selected_projection_updates','number_of_updates','cadzow_sweeps','samples_per_point', ...
                'test_seed_base','method_names','test_completed','test_squared_error','test_channel_energy');
            fprintf('Test P=%d, SNR=%d: %d/%d samples (%.1f s).\n',P,SNR_dB(SNR_index), ...
                sum(test_completed(P_index,SNR_index,:),'all'),samples_per_point,toc(point_timer));
        end
    end
end
if ~all(test_completed,'all') || any(~isfinite(test_squared_error),'all') || ...
        any(~isfinite(test_channel_energy),'all') || any(test_channel_energy<=0,'all')
    error('Fresh test phase incomplete or invalid; checkpoint preserved.');
end

% Ratio-of-sums NMSE and paired bootstrap intervals. Positive gain favors hybrid.
bootstrap_replicates=2000;
rows=zeros(length(P_list)*length(SNR_dB),17);
rng(2026112517,'twister');
row_number=0;
for P_index=1:length(P_list)
    for SNR_index=1:length(SNR_dB)
        row_number=row_number+1;
        energy=reshape(test_channel_energy(P_index,SNR_index,:),[],1);
        errors=reshape(test_squared_error(P_index,SNR_index,:,:),samples_per_point,length(method_names));
        nmse_db=10*log10(sum(errors,1)/sum(energy));
        gain=zeros(1,3); ci_low=zeros(1,3); ci_high=zeros(1,3);
        for baseline=2:4
            gain(baseline-1)=nmse_db(baseline)-nmse_db(1);
            bootstrap_gain=zeros(bootstrap_replicates,1);
            for draw_index=1:bootstrap_replicates
                draw=randi(samples_per_point,samples_per_point,1);
                hybrid_ratio=sum(errors(draw,1))/sum(energy(draw));
                baseline_ratio=sum(errors(draw,baseline))/sum(energy(draw));
                bootstrap_gain(draw_index)=10*log10(baseline_ratio/hybrid_ratio);
            end
            limits=prctile(bootstrap_gain,[2.5 97.5]);
            ci_low(baseline-1)=limits(1); ci_high(baseline-1)=limits(2);
        end
        rows(row_number,:)=[P_list(P_index),SNR_dB(SNR_index),selected_rank(P_index,SNR_index), ...
            selected_projection_updates(P_index,SNR_index),nmse_db,gain(1),ci_low(1),ci_high(1), ...
            gain(2),ci_low(2),ci_high(2),gain(3),ci_low(3),ci_high(3)];
    end
end
result_table=array2table(rows,'VariableNames',{'P','SNR_dB','Selected_rank','Projected_EM_updates', ...
    'HS_GS_hybrid_NMSE_dB','EM_GS_NMSE_dB','biased_GS_NMSE_dB','linearized_GD_NMSE_dB', ...
    'Hybrid_gain_over_EM_GS_dB','Hybrid_vs_EM_GS_CI_low_dB','Hybrid_vs_EM_GS_CI_high_dB', ...
    'Hybrid_gain_over_biased_GS_dB','Hybrid_vs_biased_GS_CI_low_dB','Hybrid_vs_biased_GS_CI_high_dB', ...
    'Hybrid_gain_over_GD_dB','Hybrid_vs_GD_CI_low_dB','Hybrid_vs_GD_CI_high_dB'});
save(result_file,'result_table','P_list','SNR_dB','N','K','RSR_dB','p_H','candidate_settings', ...
    'selected_candidate','selected_rank','selected_projection_updates','number_of_updates', ...
    'cadzow_sweeps','samples_per_point','validation_seed_base','test_seed_base', ...
    'bootstrap_replicates','method_names','validation_nmse_db','validation_completed', ...
    'validation_squared_error','validation_channel_energy','test_completed','test_squared_error','test_channel_energy');
writetable(result_table,table_file);
disp(result_table);
fprintf('V17 fresh paired-test results saved to %s\n',result_file);
fprintf('Frozen policy saved to %s\n',freeze_file);

function [Y,G,S,B,sigma2]=v17_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,sample_seed)
rng(double(sample_seed),'twister');
G=zeros(N,K);
for k=1:K
    number_of_paths=randi([3 7]);
    for path_index=1:number_of_paths
        alpha=(randn+1j*randn)/sqrt(2); theta=2*pi*rand;
        spatial_frequency=2*pi*d_over_lambda*cos(theta); polarization=sqrt(1/3)*randn(3,1);
        for i=1:N
            G(i,k)=G(i,k)+(mu_eg.'*polarization/hbar)*alpha*exp(-1j*(i-1)*spatial_frequency);
        end
    end
end
S=(randn(K,P)+1j*randn(K,P))/sqrt(2);
alpha_b=sqrt(10/2)*(randn+1j*randn); theta_b=2*pi*rand;
spatial_frequency_b=2*pi*d_over_lambda*cos(theta_b); polarization_b=sqrt(1/3)*randn(3,1);
b=zeros(N,1);
for i=1:N, b(i)=(mu_eg.'*polarization_b/hbar)*alpha_b*exp(-1j*(i-1)*spatial_frequency_b); end
B=b*ones(1,P); average_single_user_power=0;
for k=1:K
    average_single_user_power=average_single_user_power+mean(abs(G(:,k)*S(k,:)).^2,'all')/K;
end
B=B*sqrt(10^(RSR_db/10)*average_single_user_power/mean(abs(B).^2,'all'));
sigma2=mean(abs(G*S).^2,'all')/10^(snr_db/10);
W=sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P)); Y=abs(G*S+B+W);
end

function G_hat=v17_hybrid_em(Y,S,B,sigma2,updates,sweeps,rank_value,projection_updates,p_H)
G_hat=v17_spectral_initialize(Y,S,B); right_inverse=S'/(S*S');
for update=1:updates
    field=G_hat*S+B; kappa=(2/max(sigma2,eps))*Y.*abs(field);
    ratio=besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    expected_field=Y.*exp(1j*angle(field)).*ratio;
    G_hat=(expected_field-B)*right_inverse;
    if update<=projection_updates, G_hat=v17_hankel_project(G_hat,p_H,rank_value,sweeps); end
end
end

function G_hat=v17_em_gs(Y,S,B,sigma2,updates)
G_hat=v17_hybrid_em(Y,S,B,sigma2,updates,0,4,0,3);
end

function G_hat=v17_biased_gs(Y,S,B,maximum_iterations,tolerance)
G_hat=v17_spectral_initialize(Y,S,B); right_inverse=S'/(S*S');
for iteration=1:maximum_iterations
    previous=G_hat; G_hat=(Y.*exp(1j*angle(G_hat*S+B))-B)*right_inverse;
    change=norm(G_hat-previous,'fro')^2/max(norm(G_hat,'fro')^2,eps);
    if change<tolerance, break; end
end
end

function G_hat=v17_linearized_gd(Y,S,B,maximum_iterations,tolerance,initial_variance,sample_seed)
rng(mod(double(sample_seed)+720000,2^32-1),'twister');
N=size(Y,1); K=size(S,1); G_hat=sqrt(initial_variance/2)*(randn(N,K)+1j*randn(N,K));
Z=exp(-1j*angle(B)); step_size=1/max(real(eig(S*S')));
for iteration=1:maximum_iterations
    previous=G_hat;
    G_hat=G_hat+step_size*((Y-abs(B)-real(Z.*(G_hat*S))).*conj(Z))*S';
    change=norm(G_hat-previous,'fro')^2/max(norm(G_hat,'fro')^2,eps);
    if change<tolerance, break; end
end
end

function G_hat=v17_spectral_initialize(Y,S,B)
N=size(Y,1); K=size(S,1); G_hat=zeros(N,K);
for i=1:N
    y_i=Y(i,:).'; b_i=B(i,:).'; augmented=[S.',b_i];
    matrix=augmented'*diag(y_i)*augmented; [vectors,values]=eig(matrix);
    [~,index]=max(real(diag(values))); vector=vectors(:,index); vector=vector/max(norm(vector),eps);
    scale=(abs(augmented*vector).'*y_i)/max(norm(augmented*vector)^2,eps);
    G_hat(i,:)=(scale*exp(-1j*angle(vector(K+1)))*vector(1:K)).';
end
end

function G=v17_hankel_project(G,p_H,rank_value,sweeps)
for sweep=1:sweeps
    for k=1:size(G,2)
        x=G(:,k); N=length(x); H_rows=N-p_H; H_columns=p_H+1; H=zeros(H_rows,H_columns);
        for row=1:H_rows, H(row,:)=x(row:row+H_columns-1).'; end
        [U,Sigma,V]=svd(H,'econ'); H_rank=U(:,1:rank_value)*Sigma(1:rank_value,1:rank_value)*V(:,1:rank_value)';
        reconstructed=zeros(N,1); copy_count=zeros(N,1);
        for row=1:H_rows
            for column=1:H_columns
                index=row+column-1; reconstructed(index)=reconstructed(index)+H_rank(row,column);
                copy_count(index)=copy_count(index)+1;
            end
        end
        G(:,k)=reconstructed./copy_count;
    end
end
end
