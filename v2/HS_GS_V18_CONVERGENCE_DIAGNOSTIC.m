%% HS_GS_V18_CONVERGENCE_DIAGNOSTIC.m
% Replay V17's paired samples and run the frozen hybrid, EM-GS, biased GS,
% and linearized GD to a common relative-change tolerance (or max iteration).
% This reuses samples already scored by V17, so it is a convergence diagnostic,
% not a new untouched final test. Any later algorithm change needs a new test set.
%
% Run: cd('D:\MTech_Thesis_Results\v2'); HS_GS_V18_CONVERGENCE_DIAGNOSTIC

clear; clc;
folder=fileparts(mfilename('fullpath')); out=fullfile(folder,'results');
policy_file=fullfile(out,'HS_GS_V17_frozen_policy.mat');
checkpoint_file=fullfile(out,'HS_GS_V18_convergence_checkpoint.mat');
result_file=fullfile(out,'HS_GS_V18_convergence_results.mat');
table_file=fullfile(out,'HS_GS_V18_convergence_table.csv');
if ~isfile(policy_file), error('Missing V17 frozen policy: %s',policy_file); end
if isfile(result_file), error('V18 results already exist: %s',result_file); end
F=load(policy_file,'P_list','SNR_dB','N','K','RSR_dB','p_H','selected_rank', ...
    'selected_projection_updates','samples_per_point','test_seed_base');
P_list=F.P_list; SNR_dB=F.SNR_dB; N=F.N; K=F.K; RSR_dB=F.RSR_dB; p_H=F.p_H;
rank_choice=F.selected_rank; projection_updates=F.selected_projection_updates;
samples=F.samples_per_point; test_seed_base=F.test_seed_base;
max_iterations=5000; tolerance=1e-10; block_size=10; workers_if_starting_pool=4;
d_over_lambda=0.5; mu_eg=[0;1785.9*1.602e-19*5.292e-11]; mu_eg=[0;mu_eg(2);0];
hbar=6.62607015e-34/(2*pi); methods={'HS_GS','EM_GS','biased_GS','GD'};

completed=false(length(P_list),length(SNR_dB),samples);
errors=nan(length(P_list),length(SNR_dB),samples,4);
energy=nan(length(P_list),length(SNR_dB),samples);
iterations=zeros(length(P_list),length(SNR_dB),samples,4);
converged=false(length(P_list),length(SNR_dB),samples,4);
if isfile(checkpoint_file)
    C=load(checkpoint_file);
    if ~isequal(C.P_list,P_list)||~isequal(C.SNR_dB,SNR_dB)|| ...
            ~isequal(C.rank_choice,rank_choice)||~isequal(C.projection_updates,projection_updates)|| ...
            C.samples~=samples||C.test_seed_base~=test_seed_base|| ...
            C.max_iterations~=max_iterations||C.tolerance~=tolerance
        error('V18 checkpoint settings mismatch; preserve and inspect it.');
    end
    completed=C.completed; errors=C.errors; energy=C.energy;
    iterations=C.iterations; converged=C.converged;
    fprintf('Resuming V18 checkpoint: %d/%d samples complete.\n',sum(completed,'all'),numel(completed));
end
pool=gcp('nocreate');
if isempty(pool), parpool('local',workers_if_starting_pool); pool=gcp('nocreate'); end
fprintf('V18 convergence diagnostic: %d workers, max=%d, tolerance=%.1e.\n', ...
    pool.NumWorkers,max_iterations,tolerance);
fprintf('Reusing V17 test seeds. This is a diagnostic, not a new final test.\n');

for pi=1:length(P_list)
    P=P_list(pi);
    for si=1:length(SNR_dB)
        timer=tic;
        for first=1:block_size:samples
            pos=first:min(first+block_size-1,samples);
            done=reshape(completed(pi,si,pos),1,[]); pending=pos(~done);
            if isempty(pending), continue; end
            block_error=zeros(length(pending),4); block_energy=zeros(length(pending),1);
            block_it=zeros(length(pending),4); block_ok=false(length(pending),4);
            r=rank_choice(pi,si); nproj=projection_updates(pi,si);
            parfor q=1:length(pending)
                sample_index=pending(q);
                seed=uint32(test_seed_base+1000000*P+10000*si+sample_index);
                [Y,G,S,B,sigma2]=v18_sample(N,K,P,SNR_dB(si),RSR_dB,d_over_lambda,mu_eg,hbar,seed);
                [gh,ih,oh]=v18_hybrid(Y,S,B,sigma2,r,nproj,p_H,4,max_iterations,tolerance);
                [ge,ie,oe]=v18_em(Y,S,B,sigma2,max_iterations,tolerance);
                [gs,is,os]=v18_gs(Y,S,B,max_iterations,tolerance);
                [gd,id,od]=v18_gd(Y,S,B,max_iterations,tolerance,0.1,seed);
                block_error(q,:)=[norm(gh-G,'fro')^2,norm(ge-G,'fro')^2, ...
                    norm(gs-G,'fro')^2,norm(gd-G,'fro')^2];
                block_energy(q)=norm(G,'fro')^2; block_it(q,:)=[ih,ie,is,id];
                block_ok(q,:)=[oh,oe,os,od];
            end
            errors(pi,si,pending,:)=reshape(block_error,1,1,length(pending),4);
            energy(pi,si,pending)=reshape(block_energy,1,1,[]);
            iterations(pi,si,pending,:)=reshape(block_it,1,1,length(pending),4);
            converged(pi,si,pending,:)=reshape(block_ok,1,1,length(pending),4);
            completed(pi,si,pending)=true;
            save(checkpoint_file,'P_list','SNR_dB','rank_choice','projection_updates', ...
                'samples','test_seed_base','max_iterations','tolerance','completed', ...
                'errors','energy','iterations','converged');
            fprintf('P=%d SNR=%d: %d/%d samples, %.1f s.\n',P,SNR_dB(si), ...
                sum(completed(pi,si,:),'all'),samples,toc(timer));
        end
    end
end
if ~all(completed,'all')||any(~isfinite(errors),'all')||any(~isfinite(energy),'all')||any(energy<=0,'all')
    error('V18 incomplete or invalid; checkpoint preserved.');
end

rows=zeros(length(P_list)*length(SNR_dB),16); row=0;
for pi=1:length(P_list)
    for si=1:length(SNR_dB)
        row=row+1; point_energy=reshape(energy(pi,si,:),[],1);
        point_error=reshape(errors(pi,si,:,:),samples,4);
        nmse=10*log10(sum(point_error,1)/sum(point_energy));
        mean_it=reshape(mean(iterations(pi,si,:,:),3),1,4);
        conv_rate=reshape(mean(converged(pi,si,:,:),3),1,4);
        rows(row,:)=[P_list(pi),SNR_dB(si),rank_choice(pi,si),projection_updates(pi,si), ...
            nmse,mean_it,conv_rate];
    end
end
result_table=array2table(rows,'VariableNames',{'P','SNR_dB','Frozen_rank','Projected_updates', ...
    'HS_GS_NMSE_dB','EM_GS_NMSE_dB','biased_GS_NMSE_dB','GD_NMSE_dB', ...
    'mean_iter_HS','mean_iter_EM','mean_iter_GS','mean_iter_GD', ...
    'converged_fraction_HS','converged_fraction_EM','converged_fraction_GS','converged_fraction_GD'});
save(result_file,'result_table','P_list','SNR_dB','rank_choice','projection_updates', ...
    'samples','test_seed_base','max_iterations','tolerance','methods','errors','energy','iterations','converged');
writetable(result_table,table_file); disp(result_table);
fprintf('V18 convergence diagnostic saved to %s\n',result_file);

function [Y,G,S,B,sigma2]=v18_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
rng(double(seed),'twister'); G=zeros(N,K);
for k=1:K
    L=randi([3 7]);
    for ell=1:L
        alpha=(randn+1j*randn)/sqrt(2); theta=2*pi*rand;
        phi=2*pi*d_over_lambda*cos(theta); epsilon=sqrt(1/3)*randn(3,1);
        for i=1:N, G(i,k)=G(i,k)+(mu_eg.'*epsilon/hbar)*alpha*exp(-1j*(i-1)*phi); end
    end
end
S=(randn(K,P)+1j*randn(K,P))/sqrt(2);
alpha_b=sqrt(10/2)*(randn+1j*randn); theta_b=2*pi*rand;
phi_b=2*pi*d_over_lambda*cos(theta_b); eps_b=sqrt(1/3)*randn(3,1); b=zeros(N,1);
for i=1:N, b(i)=(mu_eg.'*eps_b/hbar)*alpha_b*exp(-1j*(i-1)*phi_b); end
B=b*ones(1,P); signal_power=0;
for k=1:K, signal_power=signal_power+mean(abs(G(:,k)*S(k,:)).^2,'all')/K; end
B=B*sqrt(10^(RSR_db/10)*signal_power/mean(abs(B).^2,'all'));
sigma2=mean(abs(G*S).^2,'all')/10^(snr_db/10);
W=sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P)); Y=abs(G*S+B+W);
end

function [G,it,ok]=v18_hybrid(Y,S,B,sigma2,r,nproj,p_H,sweeps,max_it,tol)
G=v18_spectral(Y,S,B); R=S'/(S*S'); it=max_it; ok=false;
for n=1:max_it
    old=G; field=G*S+B; kappa=2/max(sigma2,eps)*Y.*abs(field);
    ratio=besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G=(Y.*exp(1j*angle(field)).*ratio-B)*R;
    if n<=nproj, G=v18_project(G,p_H,r,sweeps); end
    change=norm(G-old,'fro')^2/max(norm(G,'fro')^2,eps);
    if n>nproj&&change<tol, it=n; ok=true; break; end
end
end

function [G,it,ok]=v18_em(Y,S,B,sigma2,max_it,tol)
G=v18_spectral(Y,S,B); R=S'/(S*S'); it=max_it; ok=false;
for n=1:max_it
    old=G; field=G*S+B; kappa=2/max(sigma2,eps)*Y.*abs(field);
    ratio=besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G=(Y.*exp(1j*angle(field)).*ratio-B)*R;
    if norm(G-old,'fro')^2/max(norm(G,'fro')^2,eps)<tol, it=n; ok=true; break; end
end
end

function [G,it,ok]=v18_gs(Y,S,B,max_it,tol)
G=v18_spectral(Y,S,B); R=S'/(S*S'); it=max_it; ok=false;
for n=1:max_it
    old=G; G=(Y.*exp(1j*angle(G*S+B))-B)*R;
    if norm(G-old,'fro')^2/max(norm(G,'fro')^2,eps)<tol, it=n; ok=true; break; end
end
end

function [G,it,ok]=v18_gd(Y,S,B,max_it,tol,variance,seed)
rng(mod(double(seed)+720000,2^32-1),'twister'); N=size(Y,1); K=size(S,1);
G=sqrt(variance/2)*(randn(N,K)+1j*randn(N,K)); Z=exp(-1j*angle(B));
step=1/max(real(eig(S*S'))); it=max_it; ok=false;
for n=1:max_it
    old=G; G=G+step*((Y-abs(B)-real(Z.*(G*S))).*conj(Z))*S';
    if norm(G-old,'fro')^2/max(norm(G,'fro')^2,eps)<tol, it=n; ok=true; break; end
end
end

function G=v18_spectral(Y,S,B)
N=size(Y,1); K=size(S,1); G=zeros(N,K);
for i=1:N
    y=Y(i,:).'; b=B(i,:).'; A=[S.',b]; M=A'*diag(y)*A; [V,D]=eig(M);
    [~,j]=max(real(diag(D))); v=V(:,j); v=v/max(norm(v),eps);
    scale=(abs(A*v).'*y)/max(norm(A*v)^2,eps);
    G(i,:)=(scale*exp(-1j*angle(v(K+1)))*v(1:K)).';
end
end

function G=v18_project(G,p_H,r,sweeps)
for sweep=1:sweeps
    for k=1:size(G,2)
        x=G(:,k); N=length(x); nr=N-p_H; nc=p_H+1; H=zeros(nr,nc);
        for i=1:nr, H(i,:)=x(i:i+nc-1).'; end
        [U,Sigma,V]=svd(H,'econ'); H_r=U(:,1:r)*Sigma(1:r,1:r)*V(:,1:r)';
        xsum=zeros(N,1); count=zeros(N,1);
        for i=1:nr
            for j=1:nc, q=i+j-1; xsum(q)=xsum(q)+H_r(i,j); count(q)=count(q)+1; end
        end
        G(:,k)=xsum./count;
    end
end
end
