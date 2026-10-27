%% HS-GS V33: validation-frozen EM fallback, followed by one fresh test
% Per N/P/SNR condition, validation chooses HS-GS only if its paired
% channel-NMSE gain over converged EM-GS has a positive 95% bootstrap CI.
% Otherwise the frozen condition-level policy uses EM-GS. Test samples are
% independent, and are never used to alter this policy.
%
% Run: cd('D:\MTech_Thesis_Results\v2'); HS_GS_V33_VALIDATION_FROZEN_EM_FALLBACK_TEST
% Checkpoint saves every 20 samples and resumes when settings match.

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end
checkpoint_file = fullfile(result_folder,'HS_GS_V33_checkpoint.mat');
result_file = fullfile(result_folder,'HS_GS_V33_results.mat');
table_file = fullfile(result_folder,'HS_GS_V33_fresh_test_table.csv');
figure_file = fullfile(result_folder,'HS_GS_V33_fresh_test_nmse.png');
figure_editable_file = fullfile(result_folder,'HS_GS_V33_fresh_test_nmse.fig');
if isfile(result_file) || isfile(table_file)
    error('V33 final outputs already exist. Rename/archive them before a new run.');
end

N_list = [8 16]; p_H_list = [3 7]; P_list = [10 30];
SNR_dB = -5:5:30; K = 3; RSR_dB = 25;
validation_samples_per_point = 200; test_samples_per_point = 500;
samples_per_checkpoint = 20; training_fraction = 0.70;
rank_selection_updates = 25; cadzow_sweeps = 4;
maximum_updates = 5000; convergence_tolerance = 1e-10;
bootstrap_replicates = 2000; workers_if_starting_pool = 6;
validation_seed_base = 1100000000; test_seed_base = 3000000000;
d_over_lambda = 0.5;
mu_eg = [0; 1785.9*1.602e-19*5.292e-11; 0];
hbar = 6.62607015e-34/(2*pi); gd_initial_variance = 0.1;
validation_completed = cell(1,numel(N_list));
validation_errors = cell(1,numel(N_list)); validation_energy = cell(1,numel(N_list));
test_completed = cell(1,numel(N_list)); test_errors = cell(1,numel(N_list)); test_energy = cell(1,numel(N_list));
test_ranks = cell(1,numel(N_list)); test_iterations = cell(1,numel(N_list));
test_converged = cell(1,numel(N_list));
rank_lists = cell(1,numel(N_list));
for ni=1:numel(N_list)
    N=N_list(ni); p_H=p_H_list(ni); rank_lists{ni}=1:min(N-p_H,p_H+1);
    validation_completed{ni}=false(numel(P_list),numel(SNR_dB),validation_samples_per_point);
    validation_errors{ni}=nan(numel(P_list),numel(SNR_dB),validation_samples_per_point,2); % hybrid, EM
    validation_energy{ni}=nan(numel(P_list),numel(SNR_dB),validation_samples_per_point);
    test_completed{ni}=false(numel(P_list),numel(SNR_dB),test_samples_per_point);
    test_errors{ni}=nan(numel(P_list),numel(SNR_dB),test_samples_per_point,5); % policy, HS-min, EM, biased GS, GD
    test_energy{ni}=nan(numel(P_list),numel(SNR_dB),test_samples_per_point);
    test_ranks{ni}=nan(numel(P_list),numel(SNR_dB),test_samples_per_point);
    test_iterations{ni}=zeros(numel(P_list),numel(SNR_dB),test_samples_per_point,5);
    test_converged{ni}=false(numel(P_list),numel(SNR_dB),test_samples_per_point,5);
end
policy_frozen=false; frozen_use_hs=false(numel(N_list),numel(P_list),numel(SNR_dB));
validation_gain=nan(numel(N_list),numel(P_list),numel(SNR_dB));
validation_ci_low=nan(size(validation_gain)); validation_ci_high=nan(size(validation_gain));
if isfile(checkpoint_file)
    C=load(checkpoint_file);
    if ~isequal(C.N_list,N_list)||~isequal(C.P_list,P_list)||~isequal(C.SNR_dB,SNR_dB)|| ...
       ~isequal(C.p_H_list,p_H_list)||C.validation_samples_per_point~=validation_samples_per_point|| ...
       C.test_samples_per_point~=test_samples_per_point||C.validation_seed_base~=validation_seed_base|| ...
       C.test_seed_base~=test_seed_base||C.rank_selection_updates~=rank_selection_updates
        error('V33 checkpoint settings mismatch. Preserve it and use a new checkpoint name.');
    end
    validation_completed=C.validation_completed; validation_errors=C.validation_errors;
    validation_energy=C.validation_energy; test_completed=C.test_completed;
    test_errors=C.test_errors; test_energy=C.test_energy; test_ranks=C.test_ranks;
    test_iterations=C.test_iterations; test_converged=C.test_converged;
    policy_frozen=C.policy_frozen; frozen_use_hs=C.frozen_use_hs;
    validation_gain=C.validation_gain; validation_ci_low=C.validation_ci_low;
    validation_ci_high=C.validation_ci_high;
    fprintf('Resuming V33; policy frozen = %d.\n',policy_frozen);
end
pool=gcp('nocreate'); if isempty(pool), parpool('local',workers_if_starting_pool); pool=gcp('nocreate'); end
fprintf('V33 validation-frozen EM fallback; %d workers. Validation=%d and fresh test=%d samples/condition.\n', ...
    pool.NumWorkers,validation_samples_per_point,test_samples_per_point);

% PHASE A: validation selects the condition-level policy.
if ~policy_frozen
 for ni=1:numel(N_list)
  N=N_list(ni); p_H=p_H_list(ni); ranks=rank_lists{ni};
  for pi=1:numel(P_list)
   P=P_list(pi);
   for si=1:numel(SNR_dB)
    snr=SNR_dB(si); timer=tic;
    fprintf('Validation N=%d P=%d SNR=%d dB\n',N,P,snr);
    for first=1:samples_per_checkpoint:validation_samples_per_point
     ix=first:min(first+samples_per_checkpoint-1,validation_samples_per_point);
     pending=ix(~reshape(validation_completed{ni}(pi,si,ix),1,[])); if isempty(pending), continue; end
     be=nan(numel(pending),2); ben=nan(numel(pending),1);
     parfor q=1:numel(pending)
      sample=pending(q); seed=validation_seed_base+100000000*ni+1000000*P+10000*si+sample;
      [Y,G,S,B,sigma2]=v33_generate_sample(N,K,P,snr,RSR_dB,d_over_lambda,mu_eg,hbar,seed);
      rank_value=v33_select_min_error_rank(Y,S,B,sigma2,p_H,ranks,rank_selection_updates,training_fraction,cadzow_sweeps,seed);
      [Gem,iem,cem]=v33_em_converged(Y,S,B,sigma2,maximum_updates,convergence_tolerance);
      if rank_value==ranks(end), Ghs=Gem; else, Ghs=v33_hs_converged(Y,S,B,sigma2,p_H,rank_value,cadzow_sweeps,maximum_updates,convergence_tolerance); end
      be(q,:)=[norm(Ghs-G,'fro')^2,norm(Gem-G,'fro')^2]; ben(q)=norm(G,'fro')^2;
     end
     validation_errors{ni}(pi,si,pending,:)=reshape(be,[1 1 numel(pending) 2]);
     validation_energy{ni}(pi,si,pending)=ben; validation_completed{ni}(pi,si,pending)=true;
     save(checkpoint_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB','rank_lists', ...
      'validation_samples_per_point','test_samples_per_point','validation_seed_base','test_seed_base', ...
      'rank_selection_updates','cadzow_sweeps','training_fraction','maximum_updates','convergence_tolerance', ...
      'validation_completed','validation_errors','validation_energy','test_completed','test_errors','test_energy', ...
      'test_ranks','test_iterations','test_converged','policy_frozen','frozen_use_hs','validation_gain', ...
      'validation_ci_low','validation_ci_high','-v7.3');
     fprintf('  validation %d/%d; %.1f s\n',sum(validation_completed{ni}(pi,si,:),'all'),validation_samples_per_point,toc(timer));
    end
   end
  end
 end
 if any(~cellfun(@(x)all(x,'all'),validation_completed)), error('Validation incomplete; checkpoint retained.'); end
 rng(3410000000,'twister');
 for ni=1:numel(N_list)
  for pi=1:numel(P_list)
   for si=1:numel(SNR_dB)
    e=squeeze(validation_errors{ni}(pi,si,:,:));
    validation_gain(ni,pi,si)=10*log10(sum(e(:,2))/sum(e(:,1)));
    boot=zeros(bootstrap_replicates,1); n=size(e,1);
    for b=1:bootstrap_replicates
      ids=randi(n,n,1); boot(b)=10*log10(sum(e(ids,2))/sum(e(ids,1)));
    end
    boot=sort(boot); validation_ci_low(ni,pi,si)=boot(max(1,round(.025*bootstrap_replicates)));
    validation_ci_high(ni,pi,si)=boot(min(bootstrap_replicates,round(.975*bootstrap_replicates)));
    frozen_use_hs(ni,pi,si)=validation_ci_low(ni,pi,si)>0;
   end
  end
 end
 policy_frozen=true;
 save(checkpoint_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB','rank_lists', ...
  'validation_samples_per_point','test_samples_per_point','validation_seed_base','test_seed_base', ...
  'rank_selection_updates','cadzow_sweeps','training_fraction','maximum_updates','convergence_tolerance', ...
  'validation_completed','validation_errors','validation_energy','test_completed','test_errors','test_energy', ...
  'test_ranks','test_iterations','test_converged','policy_frozen','frozen_use_hs','validation_gain', ...
  'validation_ci_low','validation_ci_high','-v7.3');
 fprintf('Validation policy frozen. HS conditions: %d/%d.\n',sum(frozen_use_hs,'all'),numel(frozen_use_hs));
end

% PHASE B: one independent test using the frozen per-condition choice.
for ni=1:numel(N_list)
 N=N_list(ni); p_H=p_H_list(ni); ranks=rank_lists{ni};
 for pi=1:numel(P_list)
  P=P_list(pi);
  for si=1:numel(SNR_dB)
   snr=SNR_dB(si);
   use_hs_for_point=frozen_use_hs(ni,pi,si);
   timer=tic;
   fprintf('Test N=%d P=%d SNR=%d dB; frozen method=%s\n', ...
    N,P,snr,choose_name(use_hs_for_point));
   for first=1:samples_per_checkpoint:test_samples_per_point
    ix=first:min(first+samples_per_checkpoint-1,test_samples_per_point);
    pending=ix(~reshape(test_completed{ni}(pi,si,ix),1,[])); if isempty(pending), continue; end
    be=nan(numel(pending),5); ben=nan(numel(pending),1); br=zeros(numel(pending),1);
    bi=zeros(numel(pending),5); bc=false(numel(pending),5);
    parfor q=1:numel(pending)
     sample=pending(q); seed=test_seed_base+100000000*ni+1000000*P+10000*si+sample;
     [Y,G,S,B,sigma2]=v33_generate_sample(N,K,P,snr,RSR_dB,d_over_lambda,mu_eg,hbar,seed);
     rank_value=v33_select_min_error_rank(Y,S,B,sigma2,p_H,ranks,rank_selection_updates,training_fraction,cadzow_sweeps,seed);
     [Gem,iem,cem]=v33_em_converged(Y,S,B,sigma2,maximum_updates,convergence_tolerance);
     if rank_value==ranks(end), Ghs=Gem; ihs=iem; chs=cem; else
      [Ghs,ihs,chs]=v33_hs_converged(Y,S,B,sigma2,p_H,rank_value,cadzow_sweeps,maximum_updates,convergence_tolerance);
     end
     if use_hs_for_point
      Gpolicy=Ghs; ipolicy=ihs; cpolicy=chs;
     else
      Gpolicy=Gem; ipolicy=iem; cpolicy=cem;
     end
     [Ggs,igs,cgs]=v33_gs_converged(Y,S,B,maximum_updates,convergence_tolerance);
     [Ggd,igd,cgd]=v33_gd_converged(Y,S,B,maximum_updates,convergence_tolerance,0.1,seed);
     be(q,:)=[norm(Gpolicy-G,'fro')^2,norm(Ghs-G,'fro')^2,norm(Gem-G,'fro')^2,norm(Ggs-G,'fro')^2,norm(Ggd-G,'fro')^2];
     ben(q)=norm(G,'fro')^2; br(q)=rank_value; bi(q,:)=[ipolicy,ihs,iem,igs,igd]; bc(q,:)=[cpolicy,chs,cem,cgs,cgd];
    end
    test_errors{ni}(pi,si,pending,:)=reshape(be,[1 1 numel(pending) 5]); test_energy{ni}(pi,si,pending)=ben;
    test_ranks{ni}(pi,si,pending)=br; test_iterations{ni}(pi,si,pending,:)=reshape(bi,[1 1 numel(pending) 5]);
    test_converged{ni}(pi,si,pending,:)=reshape(bc,[1 1 numel(pending) 5]); test_completed{ni}(pi,si,pending)=true;
    save(checkpoint_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB','rank_lists', ...
     'validation_samples_per_point','test_samples_per_point','validation_seed_base','test_seed_base', ...
     'rank_selection_updates','cadzow_sweeps','training_fraction','maximum_updates','convergence_tolerance', ...
     'validation_completed','validation_errors','validation_energy','test_completed','test_errors','test_energy', ...
     'test_ranks','test_iterations','test_converged','policy_frozen','frozen_use_hs','validation_gain', ...
     'validation_ci_low','validation_ci_high','-v7.3');
    fprintf('  test %d/%d; %.1f s\n',sum(test_completed{ni}(pi,si,:),'all'),test_samples_per_point,toc(timer));
   end
  end
 end
end
if any(~cellfun(@(x)all(x,'all'),test_completed)), error('Test incomplete; checkpoint retained.'); end

% Aggregate test NMSE and paired percentile-bootstrap confidence intervals.
method_names={'Frozen_HS_or_EM','HS_minError','EM_GS','biased_GS','GD'};
rows=zeros(numel(N_list)*numel(P_list)*numel(SNR_dB),23); rr=0; rng(3420000000,'twister');
for ni=1:numel(N_list)
 for pi=1:numel(P_list)
  for si=1:numel(SNR_dB)
   rr=rr+1; e=squeeze(test_errors{ni}(pi,si,:,:)); energy=reshape(test_energy{ni}(pi,si,:),[],1);
   nmse=10*log10(sum(e,1)/sum(energy)); gains=nmse(3:5)-nmse(1); lo=zeros(1,3); hi=zeros(1,3);
   for m=3:5
    z=zeros(bootstrap_replicates,1);
    for b=1:bootstrap_replicates
     ids=randi(test_samples_per_point,test_samples_per_point,1);
     z(b)=10*log10(sum(e(ids,m))/sum(e(ids,1)));
    end
    z=sort(z);
    lo(m-2)=z(max(1,round(.025*bootstrap_replicates)));
    hi(m-2)=z(min(bootstrap_replicates,round(.975*bootstrap_replicates)));
   end
   rrrow=[N_list(ni),P_list(pi),SNR_dB(si),double(frozen_use_hs(ni,pi,si)), ...
    validation_gain(ni,pi,si),validation_ci_low(ni,pi,si),validation_ci_high(ni,pi,si), ...
    nmse,mean(test_ranks{ni}(pi,si,:)),mean(test_converged{ni}(pi,si,:,1),'all'),gains,lo,hi];
   rows(rr,:)=rrrow;
  end
 end
end
result_table=array2table(rows(:,1:23),'VariableNames',{'N','P','SNR_dB','Frozen_uses_HS','Validation_gain_HS_over_EM_dB','Validation_CI_low','Validation_CI_high','Frozen_hybrid_NMSE_dB','HS_minError_NMSE_dB','EM_GS_NMSE_dB','biased_GS_NMSE_dB','GD_NMSE_dB','mean_selected_rank','frozen_convergence_fraction','gain_over_EM_dB','gain_over_GS_dB','gain_over_GD_dB','CI_low_EM','CI_low_GS','CI_low_GD','CI_high_EM','CI_high_GS','CI_high_GD'});
fig=figure('Color','w','Name','V33 frozen EM fallback fresh test'); tiledlayout(numel(N_list),numel(P_list),'TileSpacing','compact','Padding','compact');
cols={'Frozen_hybrid_NMSE_dB','HS_minError_NMSE_dB','EM_GS_NMSE_dB','biased_GS_NMSE_dB','GD_NMSE_dB'};
for ni=1:numel(N_list)
 for pi=1:numel(P_list)
  nexttile; hold on;
  mask=result_table.N==N_list(ni)&result_table.P==P_list(pi);
  for m=1:numel(cols)
   plot(SNR_dB,result_table{mask,cols{m}},'-o','LineWidth',1.3, ...
    'DisplayName',strrep(method_names{m},'_',' '));
  end
  grid on; xlabel('SNR (dB)'); ylabel('NMSE (dB)');
  title(sprintf('N=%d, P=%d',N_list(ni),P_list(pi)));
  if ni==1&&pi==1, legend('Location','best'); end
 end
end
sgtitle(sprintf('V33 validation-frozen HS-GS/EM fallback; RSR=%g dB',RSR_dB));
writetable(result_table,table_file);
save(result_file,'result_table','frozen_use_hs','validation_gain','validation_ci_low','validation_ci_high', ...
 'N_list','p_H_list','P_list','SNR_dB','RSR_dB','rank_lists','validation_samples_per_point', ...
 'test_samples_per_point','validation_seed_base','test_seed_base','test_errors','test_energy','test_ranks', ...
 'test_iterations','test_converged','-v7.3'); saveas(fig,figure_file); savefig(fig,figure_editable_file);
disp(result_table); fprintf('Saved V33 CSV/MAT/PNG/FIG under %s\n',result_folder);

function name=choose_name(flag), if flag, name='HS-GS'; else, name='EM-GS'; end, end

function rank_value=v33_select_min_error_rank(Y,S,B,sigma2,p_H,rank_list,updates,fraction,sweeps,seed)
P=size(S,2); rng(mod(double(seed)+76543,2^32-1),'twister'); order=randperm(P);
count=round(fraction*P); train=order(1:count); held=order(count+1:end); errs=zeros(numel(rank_list),1);
for r=1:numel(rank_list)
 G=v33_em_fixed(Y(:,train),S(:,train),B(:,train),sigma2,updates,p_H,rank_list(r),sweeps);
 pred=abs(G*S(:,held)+B(:,held)); errs(r)=mean((Y(:,held)-pred).^2,'all');
end
[~,idx]=min(errs); rank_value=rank_list(idx);
end

function G = v33_em_fixed(Y,S,B,sigma2,number_of_updates,p_H,rank_value,sweeps)
G = v33_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
for update_index = 1:number_of_updates
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if rank_value < min(size(G,1)-p_H,p_H+1)
        G = v33_hankel_project(G,p_H,rank_value,sweeps);
    end
end
end

function [G,iteration,converged] = v33_hs_converged(Y,S,B,sigma2,p_H,rank_value, ...
        sweeps,max_updates,tolerance)
G = v33_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    G = v33_hankel_project(G,p_H,rank_value,sweeps);
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = v33_em_converged(Y,S,B,sigma2,max_updates,tolerance)
G = v33_spectral_initialize(Y,S,B);
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

function [G,iteration,converged] = v33_gs_converged(Y,S,B,max_updates,tolerance)
G = v33_spectral_initialize(Y,S,B);
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

function [G,iteration,converged] = v33_gd_converged(Y,S,B,max_updates,tolerance,initial_variance,seed)
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

function G = v33_spectral_initialize(Y,S,B)
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

function G = v33_hankel_project(G,p_H,rank_value,sweeps)
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

function [Y,G,S,B,sigma2] = v33_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
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








