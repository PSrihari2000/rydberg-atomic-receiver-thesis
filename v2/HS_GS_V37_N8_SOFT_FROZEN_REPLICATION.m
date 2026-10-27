%% V37: independent replication of the frozen V36 soft Hankel policy
% N=8, K=3, RSR=25 dB, P=10/30, SNR=-5:5:30 dB.
% Read only V36's validation-selected multipliers and its earlier frozen
% V33 hard-rank policy. Evaluate every method on new paired test samples.
% No selection or tuning takes place on the V37 test data.
%
% Run: cd('D:\MTech_Thesis_Results\v2'); HS_GS_V37_N8_SOFT_FROZEN_REPLICATION
% The checkpoint saves every 20 samples and resumes if settings match.

clear; clc;
script_folder=fileparts(mfilename('fullpath'));
result_folder=fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end
source_file=fullfile(result_folder,'HS_GS_V36_N8_soft_results.mat');
checkpoint_file=fullfile(result_folder,'HS_GS_V37_N8_soft_replication_checkpoint.mat');
result_file=fullfile(result_folder,'HS_GS_V37_N8_soft_replication_results.mat');
table_file=fullfile(result_folder,'HS_GS_V37_N8_soft_replication_table.csv');
figure_file=fullfile(result_folder,'HS_GS_V37_N8_soft_replication_nmse.png');
figure_editable_file=fullfile(result_folder,'HS_GS_V37_N8_soft_replication_nmse.fig');
if isfile(result_file)||isfile(table_file)
 error('V37 final outputs already exist. Inspect them before starting a new run.');
end
if ~isfile(source_file)
 error('Run V36 first: its validation-frozen settings are required.');
end
V36=load(source_file,'configuration','frozen_candidate_index');
base=V36.configuration;
N=8; K=3; p_H=3; P_list=[10 30]; SNR_dB=-5:5:30; RSR_dB=25;
rank_list=1:min(N-p_H,p_H+1);
if base.N~=N||base.K~=K||base.p_H~=p_H||base.RSR_dB~=RSR_dB|| ...
   ~isequal(base.P_list,P_list)||~isequal(base.SNR_dB,SNR_dB)|| ...
   ~isequal(base.rank_list,rank_list)
 error('V36 configuration differs from the intended N=8, RSR=25 dB test.');
end
shrink_multiplier_list=base.shrink_multiplier_list;
frozen_candidate_index=V36.frozen_candidate_index;
hard_use_hs=base.hard_use_hs;
if ~isequal(size(frozen_candidate_index),[numel(P_list),numel(SNR_dB)])|| ...
   any(frozen_candidate_index<1,'all')|| ...
   any(frozen_candidate_index>numel(shrink_multiplier_list),'all')|| ...
   ~isequal(size(hard_use_hs),size(frozen_candidate_index))
 error('The frozen V36 soft or V33 hard policy has unexpected dimensions.');
end

test_samples_per_point=500;
samples_per_checkpoint=20;
maximum_updates=base.maximum_updates;
convergence_tolerance=base.convergence_tolerance;
rank_selection_updates=base.rank_selection_updates;
training_fraction=base.training_fraction;
hard_cadzow_sweeps=base.hard_cadzow_sweeps;
gd_initial_variance=base.gd_initial_variance;
d_over_lambda=base.d_over_lambda;
bootstrap_replicates=2000;
workers_if_starting_pool=6;
test_seed_base=3900000000; % independent of V33 and V36 tests
mu_eg=[0;1785.9*1.602e-19*5.292e-11;0];
hbar=6.62607015e-34/(2*pi);
number_of_methods=5; % soft, V33 hard/EM, EM, biased GS, GD
replication_config=struct('N',N,'K',K,'p_H',p_H,'P_list',P_list, ...
 'SNR_dB',SNR_dB,'RSR_dB',RSR_dB,'rank_list',rank_list, ...
 'shrink_multiplier_list',shrink_multiplier_list, ...
 'frozen_candidate_index',frozen_candidate_index,'hard_use_hs',hard_use_hs, ...
 'test_samples_per_point',test_samples_per_point, ...
 'samples_per_checkpoint',samples_per_checkpoint, ...
 'maximum_updates',maximum_updates,'convergence_tolerance',convergence_tolerance, ...
 'rank_selection_updates',rank_selection_updates, ...
 'training_fraction',training_fraction,'hard_cadzow_sweeps',hard_cadzow_sweeps, ...
 'gd_initial_variance',gd_initial_variance,'d_over_lambda',d_over_lambda, ...
 'bootstrap_replicates',bootstrap_replicates,'test_seed_base',test_seed_base);

test_error=nan(numel(P_list),numel(SNR_dB),test_samples_per_point,number_of_methods);
test_energy=nan(numel(P_list),numel(SNR_dB),test_samples_per_point);
test_iterations=zeros(numel(P_list),numel(SNR_dB),test_samples_per_point,number_of_methods);
test_converged=false(numel(P_list),numel(SNR_dB),test_samples_per_point,number_of_methods);
test_completed=false(numel(P_list),numel(SNR_dB),test_samples_per_point);
if isfile(checkpoint_file)
 C=load(checkpoint_file);
 if ~isequal(C.replication_config,replication_config)
  error('V37 checkpoint settings differ from this script. Preserve the checkpoint.');
 end
 test_error=C.test_error; test_energy=C.test_energy;
 test_iterations=C.test_iterations; test_converged=C.test_converged;
 test_completed=C.test_completed;
 fprintf('Resuming V37: %d/%d paired samples already completed.\n', ...
  nnz(test_completed),numel(test_completed));
end

pool=gcp('nocreate');
if isempty(pool), parpool('local',workers_if_starting_pool); pool=gcp('nocreate'); end
fprintf('V37 frozen-policy replication: %d workers, %d fresh samples/point.\n', ...
 pool.NumWorkers,test_samples_per_point);
for pi=1:numel(P_list)
 P=P_list(pi);
 for si=1:numel(SNR_dB)
  snr_db=SNR_dB(si); point_timer=tic;
  selected_index=frozen_candidate_index(pi,si);
  selected_multiplier=shrink_multiplier_list(selected_index);
  hard_enabled=hard_use_hs(pi,si);
  fprintf('P=%d SNR=%d dB: frozen multiplier=%.2f, V33 hard=%d.\n', ...
   P,snr_db,selected_multiplier,hard_enabled);
  for block_start=1:samples_per_checkpoint:test_samples_per_point
   positions=block_start:min(block_start+samples_per_checkpoint-1,test_samples_per_point);
   done=reshape(test_completed(pi,si,positions),1,[]);
   pending=positions(~done); if isempty(pending), continue; end
   block_error=nan(numel(pending),number_of_methods);
   block_energy=nan(numel(pending),1);
   block_iterations=zeros(numel(pending),number_of_methods);
   block_converged=false(numel(pending),number_of_methods);
   parfor q=1:numel(pending)
    sample_index=pending(q);
    seed=test_seed_base+1000000*P+10000*si+sample_index;
    [Y,G,S,B,sigma2]=v37_generate_sample(N,K,P,snr_db,RSR_dB, ...
     d_over_lambda,mu_eg,hbar,seed);
    [G_em,em_steps,em_ok]=v37_em_converged(Y,S,B,sigma2, ...
     maximum_updates,convergence_tolerance);
    if selected_index==1
     G_soft=G_em; soft_steps=em_steps; soft_ok=em_ok;
    else
     singular_noise_scale=v37_singular_noise_scale(S,sigma2,N,p_H);
     threshold=selected_multiplier*singular_noise_scale;
     [G_soft,soft_steps,soft_ok]=v37_soft_em_converged(Y,S,B,sigma2,p_H, ...
      threshold,maximum_updates,convergence_tolerance);
    end
    if hard_enabled
     rank_value=v37_select_min_error_rank(Y,S,B,sigma2,p_H,rank_list, ...
      rank_selection_updates,training_fraction,hard_cadzow_sweeps,seed);
     if rank_value==rank_list(end)
      G_hard=G_em; hard_steps=em_steps; hard_ok=em_ok;
     else
      [G_hard,hard_steps,hard_ok]=v37_hs_converged(Y,S,B,sigma2,p_H, ...
       rank_value,hard_cadzow_sweeps,maximum_updates,convergence_tolerance);
     end
    else
     G_hard=G_em; hard_steps=em_steps; hard_ok=em_ok;
    end
    [G_gs,gs_steps,gs_ok]=v37_gs_converged(Y,S,B, ...
     maximum_updates,convergence_tolerance);
    [G_gd,gd_steps,gd_ok]=v37_gd_converged(Y,S,B, ...
     maximum_updates,convergence_tolerance,gd_initial_variance,seed);
    block_error(q,:)=[norm(G_soft-G,'fro')^2,norm(G_hard-G,'fro')^2, ...
     norm(G_em-G,'fro')^2,norm(G_gs-G,'fro')^2,norm(G_gd-G,'fro')^2];
    block_energy(q)=norm(G,'fro')^2;
    block_iterations(q,:)=[soft_steps,hard_steps,em_steps,gs_steps,gd_steps];
    block_converged(q,:)=[soft_ok,hard_ok,em_ok,gs_ok,gd_ok];
   end
   test_error(pi,si,pending,:)=reshape(block_error,[1 1 numel(pending) number_of_methods]);
   test_energy(pi,si,pending)=block_energy;
   test_iterations(pi,si,pending,:)=reshape(block_iterations,[1 1 numel(pending) number_of_methods]);
   test_converged(pi,si,pending,:)=reshape(block_converged,[1 1 numel(pending) number_of_methods]);
   test_completed(pi,si,pending)=true;
   save(checkpoint_file,'replication_config','test_error','test_energy', ...
    'test_iterations','test_converged','test_completed','-v7.3');
   fprintf('  %d/%d samples; %.1f s.\n', ...
    sum(test_completed(pi,si,:),'all'),test_samples_per_point,toc(point_timer));
  end
 end
end
if ~all(test_completed,'all')||any(~isfinite(test_error),'all')|| ...
   any(~isfinite(test_energy),'all')||any(test_energy<=0,'all')
 error('V37 test incomplete or invalid. Checkpoint retained.');
end

% Ratio-of-sums NMSE and paired bootstrap gains; positive gain favors soft.
rng(3920000000,'twister');
number_of_rows=numel(P_list)*numel(SNR_dB);
rows=zeros(number_of_rows,23); row_number=0;
for pi=1:numel(P_list)
 for si=1:numel(SNR_dB)
  row_number=row_number+1;
  errors=reshape(test_error(pi,si,:,:),test_samples_per_point,number_of_methods);
  energy=reshape(test_energy(pi,si,:),test_samples_per_point,1);
  nmse_db=10*log10(sum(errors,1)/sum(energy));
  gains=nmse_db(2:5)-nmse_db(1);
  ci_low=zeros(1,4); ci_high=zeros(1,4);
  for baseline=2:5
   bootstrap_gain=zeros(bootstrap_replicates,1);
   for draw=1:bootstrap_replicates
    ids=randi(test_samples_per_point,test_samples_per_point,1);
    bootstrap_gain(draw)=10*log10(sum(errors(ids,baseline))/sum(errors(ids,1)));
   end
   bootstrap_gain=sort(bootstrap_gain);
   ci_low(baseline-1)=bootstrap_gain(max(1,round(0.025*bootstrap_replicates)));
   ci_high(baseline-1)=bootstrap_gain(min(bootstrap_replicates,round(0.975*bootstrap_replicates)));
  end
  soft_mean_iterations=mean(test_iterations(pi,si,:,1),'all');
  soft_convergence_fraction=mean(test_converged(pi,si,:,1),'all');
  rows(row_number,:)=[P_list(pi),SNR_dB(si), ...
   shrink_multiplier_list(frozen_candidate_index(pi,si)),double(hard_use_hs(pi,si)), ...
   nmse_db,gains,ci_low,ci_high,soft_mean_iterations,soft_convergence_fraction];
 end
end
result_table=array2table(rows,'VariableNames',{'P','SNR_dB', ...
 'frozen_soft_multiplier','V33_hard_uses_HS', ...
 'soft_NMSE_dB','V33_hard_NMSE_dB','EM_GS_NMSE_dB', ...
 'biased_GS_NMSE_dB','GD_NMSE_dB', ...
 'soft_gain_over_V33_hard_dB','soft_gain_over_EM_dB', ...
 'soft_gain_over_GS_dB','soft_gain_over_GD_dB', ...
 'CI_low_V33_hard','CI_low_EM','CI_low_GS','CI_low_GD', ...
 'CI_high_V33_hard','CI_high_EM','CI_high_GS','CI_high_GD', ...
 'soft_mean_iterations','soft_convergence_fraction'});
figure_handle=figure('Color','w','Name','V37 frozen soft Hankel replication');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
method_columns={'soft_NMSE_dB','V33_hard_NMSE_dB','EM_GS_NMSE_dB', ...
 'biased_GS_NMSE_dB','GD_NMSE_dB'};
method_names={'Soft Hankel/EM','V33 hard Hankel/EM','EM-GS','biased GS','GD'};
for pi=1:numel(P_list)
 nexttile; hold on;
 selected_rows=result_table.P==P_list(pi);
 for method=1:number_of_methods
  plot(SNR_dB,result_table{selected_rows,method_columns{method}},'-o', ...
   'LineWidth',1.4,'MarkerSize',4,'DisplayName',method_names{method});
 end
 grid on; xlabel('SNR (dB)'); ylabel('NMSE (dB)');
 title(sprintf('N = %d, P = %d',N,P_list(pi)));
 if pi==1, legend('Location','best'); end
end
sgtitle(sprintf('V37 frozen V36 soft Hankel replication; RSR = %g dB',RSR_dB));
writetable(result_table,table_file);
save(result_file,'replication_config','result_table','test_error','test_energy', ...
 'test_iterations','test_converged','test_completed','-v7.3');
saveas(figure_handle,figure_file); savefig(figure_handle,figure_editable_file);
disp(result_table);
fprintf('V37 table: %s\nV37 figure: %s\nV37 MAT: %s\n', ...
 table_file,figure_file,result_file);

function scale=v37_singular_noise_scale(S,sigma2,N,p_H)
% Heuristic: map measurement noise through the LS right inverse and
% convert the channel scale to a Hankel singular-value scale.
K=size(S,1);
right_inverse=S'/(S*S');
scale=sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
end

function [G,iteration,converged]=v37_soft_em_converged(Y,S,B,sigma2,p_H,threshold,max_updates,tolerance)
G=v37_spectral_initialize(Y,S,B);
right_inverse=S'/(S*S');
converged=false; iteration=max_updates;
for update_index=1:max_updates
 previous_G=G;
 field=G*S+B;
 kappa=2/max(sigma2,eps)*Y.*abs(field);
 ratio=besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
 G=(Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
 G=v37_soft_hankel_project(G,p_H,threshold);
 if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps)<tolerance
  iteration=update_index; converged=true; break;
 end
end
end

function G=v37_soft_hankel_project(G,p_H,threshold)
N=size(G,1); number_of_rows=N-p_H; number_of_columns=p_H+1;
for user=1:size(G,2)
 x=G(:,user);
 H=zeros(number_of_rows,number_of_columns);
 for row=1:number_of_rows
  H(row,:)=x(row:row+number_of_columns-1).';
 end
 [U,Sigma,V]=svd(H,'econ');
 shrunk=max(diag(Sigma)-threshold,0);
 H_soft=U*diag(shrunk)*V';
 reconstructed=zeros(N,1); counts=zeros(N,1);
 for row=1:number_of_rows
  for column=1:number_of_columns
   index=row+column-1;
   reconstructed(index)=reconstructed(index)+H_soft(row,column);
   counts(index)=counts(index)+1;
  end
 end
 G(:,user)=reconstructed./counts;
end
end

function rank_value=v37_select_min_error_rank(Y,S,B,sigma2,p_H,rank_list,updates,fraction,sweeps,seed)
P=size(S,2); rng(mod(double(seed)+76543,2^32-1),'twister'); order=randperm(P);
count=round(fraction*P); train=order(1:count); held=order(count+1:end); errs=zeros(numel(rank_list),1);
for r=1:numel(rank_list)
 G=v37_em_fixed(Y(:,train),S(:,train),B(:,train),sigma2,updates,p_H,rank_list(r),sweeps);
 pred=abs(G*S(:,held)+B(:,held)); errs(r)=mean((Y(:,held)-pred).^2,'all');
end
[~,idx]=min(errs); rank_value=rank_list(idx);
end

function G = v37_em_fixed(Y,S,B,sigma2,number_of_updates,p_H,rank_value,sweeps)
G = v37_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
for update_index = 1:number_of_updates
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if rank_value < min(size(G,1)-p_H,p_H+1)
        G = v37_hankel_project(G,p_H,rank_value,sweeps);
    end
end
end

function [G,iteration,converged] = v37_hs_converged(Y,S,B,sigma2,p_H,rank_value, ...
        sweeps,max_updates,tolerance)
G = v37_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    G = v37_hankel_project(G,p_H,rank_value,sweeps);
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = v37_em_converged(Y,S,B,sigma2,max_updates,tolerance)
G = v37_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = v37_gs_converged(Y,S,B,max_updates,tolerance)
G = v37_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    G = (Y.*exp(1j*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = v37_gd_converged(Y,S,B,max_updates,tolerance,initial_variance,seed)
rng(mod(double(seed)+720000,2^32-1),'twister');
N = size(Y,1); K = size(S,1);
G = sqrt(initial_variance/2)*(randn(N,K)+1j*randn(N,K));
phase_reference = exp(-1j*angle(B));
step_size = 1/max(real(eig(S*S')));
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).* ...
        conj(phase_reference))*S';
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function G = v37_spectral_initialize(Y,S,B)
N = size(Y,1); K = size(S,1);
G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).';
    b = B(antenna,:).';
    A = [S.',b];
    M = A'*diag(y)*A;
    [V,D] = eig(M);
    [~,principal_index] = max(real(diag(D)));
    v = V(:,principal_index);
    v = v/max(norm(v),eps);
    scale = (abs(A*v).'*y)/max(norm(A*v)^2,eps);
    G(antenna,:) = (scale*exp(-1j*angle(v(K+1)))*v(1:K)).';
end
end

function G = v37_hankel_project(G,p_H,rank_value,sweeps)
for sweep = 1:sweeps
    for user = 1:size(G,2)
        x = G(:,user);
        N = length(x);
        number_of_rows = N-p_H;
        number_of_columns = p_H+1;
        H = zeros(number_of_rows,number_of_columns);
        for row = 1:number_of_rows
            H(row,:) = x(row:row+number_of_columns-1).';
        end
        [U,Sigma,V] = svd(H,'econ');
        H_rank = U(:,1:rank_value)*Sigma(1:rank_value,1:rank_value)*V(:,1:rank_value)';
        reconstructed = zeros(N,1);
        counts = zeros(N,1);
        for row = 1:number_of_rows
            for column = 1:number_of_columns
                index = row+column-1;
                reconstructed(index) = reconstructed(index)+H_rank(row,column);
                counts(index) = counts(index)+1;
            end
        end
        G(:,user) = reconstructed./counts;
    end
end
end

function [Y,G,S,B,sigma2] = v37_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
rng(double(seed),'twister');
G = zeros(N,K);
for user = 1:K
    number_of_paths = randi([3 7]);
    for path_index = 1:number_of_paths
        alpha = (randn+1j*randn)/sqrt(2);
        theta = 2*pi*rand;
        phase_step = 2*pi*d_over_lambda*cos(theta);
        polarization = sqrt(1/3)*randn(3,1);
        for antenna = 1:N
            G(antenna,user) = G(antenna,user)+(mu_eg.'*polarization/hbar)* ...
                alpha*exp(-1j*(antenna-1)*phase_step);
        end
    end
end
S = (randn(K,P)+1j*randn(K,P))/sqrt(2);
alpha_b = sqrt(10/2)*(randn+1j*randn);
theta_b = 2*pi*rand;
phase_step_b = 2*pi*d_over_lambda*cos(theta_b);
polarization_b = sqrt(1/3)*randn(3,1);
b = zeros(N,1);
for antenna = 1:N
    b(antenna) = (mu_eg.'*polarization_b/hbar)*alpha_b* ...
        exp(-1j*(antenna-1)*phase_step_b);
end
B = b*ones(1,P);
average_signal_power = 0;
for user = 1:K
    average_signal_power = average_signal_power+mean(abs(G(:,user)*S(user,:)).^2,'all')/K;
end
B = B*sqrt(10^(RSR_db/10)*average_signal_power/mean(abs(B).^2,'all'));
sigma2 = mean(abs(G*S).^2,'all')/10^(snr_db/10);
W = sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P));
Y = abs(G*S+B+W);
end








