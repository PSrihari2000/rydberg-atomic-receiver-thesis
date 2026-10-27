%% HS_GS_V25_FROZEN_POLICY_PAIRED_TEST.m
% Fresh paired test of the V24 validation-selected HS-GS settings.
% Compares frozen HS-GS with 50-update unprojected EM-GS, converged EM-GS,
% converged biased GS, and converged linearized GD. Every method sees the
% same newly generated observations at each test point. Test channel truth
% is used only for final NMSE scoring; it does not alter the frozen policy.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V25_FROZEN_POLICY_PAIRED_TEST
%
% The script resumes from its checkpoint if interrupted. It does not
% overwrite V24 outputs or any earlier versioned results.

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end

policy_file = fullfile(result_folder,'HS_GS_V24_frozen_validation_policy.mat');
checkpoint_file = fullfile(result_folder,'HS_GS_V25_paired_test_checkpoint.mat');
result_file = fullfile(result_folder,'HS_GS_V25_frozen_policy_test_results.mat');
table_file = fullfile(result_folder,'HS_GS_V25_frozen_policy_test_table.csv');
figure_file = fullfile(result_folder,'HS_GS_V25_frozen_policy_test_nmse.png');
figure_editable_file = fullfile(result_folder,'HS_GS_V25_frozen_policy_test_nmse.fig');
if ~isfile(policy_file)
    error('V24 frozen policy is missing. Run V24 validation first: %s',policy_file);
end
if isfile(result_file) || isfile(table_file)
    error('V25 final outputs already exist. Inspect them before starting a new test.');
end

% Load settings chosen before this independent test was generated.
policy = load(policy_file);
N_list = [8 16];
p_H_list = [3 7];
P_list = [10 30];
SNR_dB = -5:5:30;
K = 3;
RSR_dB = 25;
cadzow_sweeps = 4;
if ~isequal(policy.N_list,N_list) || ~isequal(policy.p_H_list,p_H_list) || ...
        ~isequal(policy.P_list,P_list) || ~isequal(policy.SNR_dB,SNR_dB) || ...
        policy.K~=K || policy.RSR_dB~=RSR_dB || policy.cadzow_sweeps~=cadzow_sweeps
    error('V24 policy settings do not match V25 benchmark settings. Preserve and inspect both files.');
end
selected_rank = policy.selected_rank;
selected_projected_updates = policy.selected_projected_updates;
selected_total_updates = policy.selected_total_updates;

% Keep the test seed stream distinct from the V20-V24 validation/test streams.
samples_per_test_point = 500;
samples_per_checkpoint = 20;
test_seed_base = 4200000000;
maximum_convergence_updates = 5000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
bootstrap_replicates = 2000;
workers_if_starting_pool = 6;
d_over_lambda = 0.5;
mu_eg = [0; 1785.9 * 1.602e-19 * 5.292e-11; 0];
hbar = 6.62607015e-34 / (2*pi);

method_names = {'HS_GS_V24_frozen','EM_GS_50_updates','EM_GS_converged', ...
    'biased_GS_converged','GD_converged'};
number_of_methods = numel(method_names);
test_squared_error = cell(1,numel(N_list));
test_channel_energy = cell(1,numel(N_list));
test_iterations = cell(1,numel(N_list));
test_converged = cell(1,numel(N_list));
test_completed = cell(1,numel(N_list));
for n_index = 1:numel(N_list)
    test_squared_error{n_index} = nan(numel(P_list),numel(SNR_dB), ...
        samples_per_test_point,number_of_methods);
    test_channel_energy{n_index} = nan(numel(P_list),numel(SNR_dB),samples_per_test_point);
    test_iterations{n_index} = zeros(numel(P_list),numel(SNR_dB), ...
        samples_per_test_point,number_of_methods);
    test_converged{n_index} = false(numel(P_list),numel(SNR_dB), ...
        samples_per_test_point,number_of_methods);
    test_completed{n_index} = false(numel(P_list),numel(SNR_dB),samples_per_test_point);
end

if isfile(checkpoint_file)
    C = load(checkpoint_file);
    if ~isequal(C.N_list,N_list) || ~isequal(C.P_list,P_list) || ...
            ~isequal(C.SNR_dB,SNR_dB) || ~isequal(C.selected_rank,selected_rank) || ...
            ~isequal(C.selected_projected_updates,selected_projected_updates) || ...
            ~isequal(C.selected_total_updates,selected_total_updates) || ...
            C.samples_per_test_point~=samples_per_test_point || C.test_seed_base~=test_seed_base || ...
            C.maximum_convergence_updates~=maximum_convergence_updates || ...
            C.convergence_tolerance~=convergence_tolerance
        error('V25 checkpoint settings differ. Preserve and inspect it; do not mix runs.');
    end
    test_squared_error = C.test_squared_error;
    test_channel_energy = C.test_channel_energy;
    test_iterations = C.test_iterations;
    test_converged = C.test_converged;
    test_completed = C.test_completed;
    fprintf('Resuming V25: %d/%d samples complete.\n', ...
        sum(cellfun(@(x) sum(x,'all'),test_completed)), ...
        numel(N_list)*numel(P_list)*numel(SNR_dB)*samples_per_test_point);
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('local',workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('V25 fresh paired test: %d workers, %d samples per N/P/SNR point.\n', ...
    pool.NumWorkers,samples_per_test_point);
fprintf('Methods: frozen HS-GS, 50-update EM-GS, converged EM-GS, biased GS, and GD.\n');

for n_index = 1:numel(N_list)
    N = N_list(n_index);
    p_H = p_H_list(n_index);
    for P_index = 1:numel(P_list)
        P = P_list(P_index);
        for SNR_index = 1:numel(SNR_dB)
            snr_db = SNR_dB(SNR_index);
            point_timer = tic;
            rank_value = selected_rank(n_index,P_index,SNR_index);
            projected_updates = selected_projected_updates(n_index,P_index,SNR_index);
            total_updates = selected_total_updates(n_index,P_index,SNR_index);
            for block_start = 1:samples_per_checkpoint:samples_per_test_point
                positions = block_start:min(block_start+samples_per_checkpoint-1,samples_per_test_point);
                completed_here = reshape(test_completed{n_index}(P_index,SNR_index,positions),1,[]);
                pending = positions(~completed_here);
                if isempty(pending), continue; end

                block_error = nan(numel(pending),number_of_methods);
                block_energy = nan(numel(pending),1);
                block_iterations = zeros(numel(pending),number_of_methods);
                block_converged = false(numel(pending),number_of_methods);
                parfor local_index = 1:numel(pending)
                    sample_index = pending(local_index);
                    seed = test_seed_base + 1000000*P + 10000*SNR_index + sample_index;
                    [Y,G,S,B,sigma2] = v25_generate_sample(N,K,P,snr_db,RSR_dB, ...
                        d_over_lambda,mu_eg,hbar,seed);

                    G_hs = v25_hybrid_em(Y,S,B,sigma2,rank_value,projected_updates, ...
                        total_updates,p_H,cadzow_sweeps);
                    G_em_50 = v25_hybrid_em(Y,S,B,sigma2,min(N-p_H,p_H+1), ...
                        0,50,p_H,cadzow_sweeps);
                    [G_em,em_iterations,em_ok] = v25_em_converged(Y,S,B,sigma2, ...
                        maximum_convergence_updates,convergence_tolerance);
                    [G_gs,gs_iterations,gs_ok] = v25_gs_converged(Y,S,B, ...
                        maximum_convergence_updates,convergence_tolerance);
                    [G_gd,gd_iterations,gd_ok] = v25_gd_converged(Y,S,B, ...
                        maximum_convergence_updates,convergence_tolerance,gd_initial_variance,seed);

                    block_error(local_index,:) = [norm(G_hs-G,'fro')^2, ...
                        norm(G_em_50-G,'fro')^2,norm(G_em-G,'fro')^2, ...
                        norm(G_gs-G,'fro')^2,norm(G_gd-G,'fro')^2];
                    block_energy(local_index) = norm(G,'fro')^2;
                    block_iterations(local_index,:) = [total_updates,50, ...
                        em_iterations,gs_iterations,gd_iterations];
                    block_converged(local_index,:) = [false,false,em_ok,gs_ok,gd_ok];
                end

                test_squared_error{n_index}(P_index,SNR_index,pending,:) = ...
                    reshape(block_error,[1 1 numel(pending) number_of_methods]);
                test_channel_energy{n_index}(P_index,SNR_index,pending) = block_energy;
                test_iterations{n_index}(P_index,SNR_index,pending,:) = ...
                    reshape(block_iterations,[1 1 numel(pending) number_of_methods]);
                test_converged{n_index}(P_index,SNR_index,pending,:) = ...
                    reshape(block_converged,[1 1 numel(pending) number_of_methods]);
                test_completed{n_index}(P_index,SNR_index,pending) = true;
                save(checkpoint_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB', ...
                    'selected_rank','selected_projected_updates','selected_total_updates', ...
                    'samples_per_test_point','test_seed_base','maximum_convergence_updates', ...
                    'convergence_tolerance','method_names','test_squared_error', ...
                    'test_channel_energy','test_iterations','test_converged','test_completed','-v7.3');
                fprintf('N=%d, P=%d, SNR=%d dB: %d/%d samples; %.1f s.\n', ...
                    N,P,snr_db,sum(test_completed{n_index}(P_index,SNR_index,:),'all'), ...
                    samples_per_test_point,toc(point_timer));
            end
        end
    end
end

for n_index = 1:numel(N_list)
    if ~all(test_completed{n_index},'all') || ...
            any(~isfinite(test_squared_error{n_index}),'all') || ...
            any(~isfinite(test_channel_energy{n_index}),'all') || ...
            any(test_channel_energy{n_index}<=0,'all')
        error('V25 paired test incomplete or invalid. Checkpoint preserved.');
    end
end

% Ratio-of-sums NMSE and paired bootstrap CIs. Positive gain means HS-GS wins.
rng(2026102525,'twister');
number_of_rows = numel(N_list)*numel(P_list)*numel(SNR_dB);
rows = zeros(number_of_rows,6 + number_of_methods + 3*(number_of_methods-1) + ...
    number_of_methods + 3);
row_number = 0;
for n_index = 1:numel(N_list)
    for P_index = 1:numel(P_list)
        for SNR_index = 1:numel(SNR_dB)
            row_number = row_number+1;
            energy = reshape(test_channel_energy{n_index}(P_index,SNR_index,:),[],1);
            errors = reshape(test_squared_error{n_index}(P_index,SNR_index,:,:), ...
                samples_per_test_point,number_of_methods);
            nmse_db = 10*log10(sum(errors,1)/sum(energy));
            gains = zeros(1,number_of_methods-1);
            ci_low = zeros(1,number_of_methods-1);
            ci_high = zeros(1,number_of_methods-1);
            for baseline = 2:number_of_methods
                gains(baseline-1) = nmse_db(baseline)-nmse_db(1);
                bootstrap_gain = zeros(bootstrap_replicates,1);
                for draw = 1:bootstrap_replicates
                    indices = randi(samples_per_test_point,samples_per_test_point,1);
                    hs_ratio = sum(errors(indices,1))/sum(energy(indices));
                    baseline_ratio = sum(errors(indices,baseline))/sum(energy(indices));
                    bootstrap_gain(draw) = 10*log10(baseline_ratio/hs_ratio);
                end
                interval = prctile(bootstrap_gain,[2.5 97.5]);
                ci_low(baseline-1) = interval(1);
                ci_high(baseline-1) = interval(2);
            end
            mean_iterations = reshape(mean(test_iterations{n_index}(P_index,SNR_index,:,:),3),1,[]);
            convergence_fraction = reshape(mean(test_converged{n_index}(P_index,SNR_index,:,:),3),1,[]);
            rows(row_number,:) = [N_list(n_index),P_list(P_index),SNR_dB(SNR_index), ...
                selected_rank(n_index,P_index,SNR_index), ...
                selected_projected_updates(n_index,P_index,SNR_index), ...
                selected_total_updates(n_index,P_index,SNR_index),nmse_db,gains, ...
                ci_low,ci_high,mean_iterations,convergence_fraction(3:5)];
        end
    end
end

result_table = array2table(rows,'VariableNames',{'N','P','SNR_dB','HS_rank', ...
    'HS_projected_updates','HS_total_updates','HS_GS_NMSE_dB','EM_GS_50_NMSE_dB', ...
    'EM_GS_converged_NMSE_dB','biased_GS_converged_NMSE_dB','GD_converged_NMSE_dB', ...
    'gain_over_EM_GS_50_dB','gain_over_EM_GS_converged_dB','gain_over_biased_GS_dB', ...
    'gain_over_GD_dB','CI_low_EM_GS_50','CI_low_EM_GS_converged','CI_low_biased_GS', ...
    'CI_low_GD','CI_high_EM_GS_50','CI_high_EM_GS_converged','CI_high_biased_GS', ...
    'CI_high_GD','mean_iter_HS','mean_iter_EM_GS_50','mean_iter_EM_GS_converged', ...
    'mean_iter_biased_GS','mean_iter_GD','converged_fraction_EM_GS', ...
    'converged_fraction_biased_GS','converged_fraction_GD'});
writetable(result_table,table_file);

figure_handle = figure('Color','w','Name','V25 frozen-policy paired test');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
line_colors = lines(number_of_methods);
for n_index = 1:numel(N_list)
    for P_index = 1:numel(P_list)
        nexttile;
        selected_rows = result_table.N==N_list(n_index) & result_table.P==P_list(P_index);
        for method_index = 1:number_of_methods
            method_columns = {'HS_GS_NMSE_dB','EM_GS_50_NMSE_dB', ...
                'EM_GS_converged_NMSE_dB','biased_GS_converged_NMSE_dB','GD_converged_NMSE_dB'};
            plot(SNR_dB,result_table{selected_rows,method_columns{method_index}}, ...
                '-o','Color',line_colors(method_index,:),'LineWidth',1.4, ...
                'MarkerSize',4,'DisplayName',strrep(method_names{method_index},'_',' '));
            hold on;
        end
        grid on; xlabel('SNR (dB)'); ylabel('NMSE (dB)');
        title(sprintf('N = %d, P = %d',N_list(n_index),P_list(P_index)));
        if n_index==1 && P_index==1, legend('Location','southwest'); end
    end
end
sgtitle('V25: Frozen V24 HS-GS policy on fresh paired test data, RSR = 25 dB');
saveas(figure_handle,figure_file);
savefig(figure_handle,figure_editable_file);

save(result_file,'result_table','N_list','p_H_list','P_list','SNR_dB','K','RSR_dB', ...
    'selected_rank','selected_projected_updates','selected_total_updates', ...
    'samples_per_test_point','test_seed_base','maximum_convergence_updates', ...
    'convergence_tolerance','bootstrap_replicates','method_names','test_squared_error', ...
    'test_channel_energy','test_iterations','test_converged','-v7.3');
disp(result_table);
fprintf('V25 paired-test table: %s\n',table_file);
fprintf('V25 paired-test plot: %s\n',figure_file);
fprintf('V25 result data: %s\n',result_file);

function [Y,G,S,B,sigma2] = v25_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
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
            G(antenna,user) = G(antenna,user) + ...
                (mu_eg.'*polarization/hbar)*alpha*exp(-1j*(antenna-1)*phase_step);
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
    b(antenna) = (mu_eg.'*polarization_b/hbar)*alpha_b*exp(-1j*(antenna-1)*phase_step_b);
end
B = b*ones(1,P);
average_signal_power = 0;
for user = 1:K
    average_signal_power = average_signal_power + mean(abs(G(:,user)*S(user,:)).^2,'all')/K;
end
B = B*sqrt(10^(RSR_db/10)*average_signal_power/mean(abs(B).^2,'all'));
sigma2 = mean(abs(G*S).^2,'all')/10^(snr_db/10);
W = sqrt(sigma2/2)*(randn(N,P)+1j*randn(N,P));
Y = abs(G*S+B+W);
end

function G = v25_hybrid_em(Y,S,B,sigma2,rank_value,projected_updates,total_updates,p_H,sweeps)
G = v25_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
for iteration = 1:total_updates
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if iteration <= projected_updates
        G = v25_hankel_project(G,p_H,rank_value,sweeps);
    end
end
end

function [G,iteration,converged] = v25_em_converged(Y,S,B,sigma2,max_updates,tolerance)
G = v25_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false;
iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index;
        converged = true;
        break;
    end
end
end

function [G,iteration,converged] = v25_gs_converged(Y,S,B,max_updates,tolerance)
G = v25_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false;
iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    G = (Y.*exp(1j*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index;
        converged = true;
        break;
    end
end
end

function [G,iteration,converged] = v25_gd_converged(Y,S,B,max_updates,tolerance,initial_variance,seed)
rng(mod(double(seed)+720000,2^32-1),'twister');
N = size(Y,1);
K = size(S,1);
G = sqrt(initial_variance/2)*(randn(N,K)+1j*randn(N,K));
phase_reference = exp(-1j*angle(B));
step_size = 1/max(real(eig(S*S')));
converged = false;
iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    G = G + step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index;
        converged = true;
        break;
    end
end
end

function G = v25_spectral_initialize(Y,S,B)
N = size(Y,1);
K = size(S,1);
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

function G = v25_hankel_project(G,p_H,rank_value,sweeps)
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
