%% HS-GS V32: fresh test of a held-out-pilot EM fallback rule
% Fit EM-GS and rank-constrained HS-GS candidates on training pilots.
% Select the best constrained HS rank only when its paired held-out-pilot
% prediction improvement over EM-GS exceeds one standard error; otherwise
% use converged EM-GS. Channel truth is only for final NMSE scoring.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V32_EM_FALLBACK_FRESH_TEST
%
% Checkpoint is saved every block. Keep the checkpoint to resume; do not
% delete or overwrite it unless intentionally starting a new seed/run.

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end

checkpoint_file = fullfile(result_folder,'HS_GS_V32_pilot_em_fallback_checkpoint.mat');
result_file = fullfile(result_folder,'HS_GS_V32_pilot_em_fallback_results.mat');
table_file = fullfile(result_folder,'HS_GS_V32_pilot_em_fallback_table.csv');
figure_file = fullfile(result_folder,'HS_GS_V32_pilot_em_fallback_nmse.png');
figure_editable_file = fullfile(result_folder,'HS_GS_V32_pilot_em_fallback_nmse.fig');
if isfile(result_file) || isfile(table_file)
    error('V32 final outputs already exist. Inspect them before starting another run.');
end

N_list = [8 16];
p_H_list = [3 7];
P_list = [10 30];
SNR_dB = -5:5:30;
K = 3;
RSR_dB = 25;
samples_per_test_point = 500;
samples_per_checkpoint = 20;
sample_seed_base = 2600000000;
training_fraction = 0.70;
rank_selection_updates = 25;
cadzow_sweeps = 4;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
bootstrap_replicates = 2000;
workers_if_starting_pool = 6;
d_over_lambda = 0.5;
mu_eg = [0; 1785.9*1.602e-19*5.292e-11; 0];
hbar = 6.62607015e-34/(2*pi);
gd_initial_variance = 0.1;
method_names = {'HS_EM_pilot_selected','HS_GS_minError','EM_GS_converged', ...
    'biased_GS_converged','GD_converged'};
number_of_methods = numel(method_names);

% Hankel embeddings are 5-by-4 (N=8, p_H=3) and 9-by-8
% (N=16, p_H=7); candidate ranks are 1:4 and 1:8, respectively.
rank_lists = cell(1,numel(N_list));
for n_index = 1:numel(N_list)
    N = N_list(n_index);
    p_H = p_H_list(n_index);
    rank_lists{n_index} = 1:min(N-p_H,p_H+1);
end

test_squared_error = cell(1,numel(N_list));
test_channel_energy = cell(1,numel(N_list));
test_selected_rank = cell(1,numel(N_list));
test_selected_em = cell(1,numel(N_list));
test_rank_validation_error = cell(1,numel(N_list));
test_iterations = cell(1,numel(N_list));
test_converged = cell(1,numel(N_list));
test_completed = cell(1,numel(N_list));
for n_index = 1:numel(N_list)
    test_squared_error{n_index} = nan(numel(P_list),numel(SNR_dB), ...
        samples_per_test_point,number_of_methods);
    test_channel_energy{n_index} = nan(numel(P_list),numel(SNR_dB),samples_per_test_point);
    % Rank columns: hybrid choice (max rank denotes EM fallback), min-error, one-SE.
    test_selected_rank{n_index} = nan(numel(P_list),numel(SNR_dB),samples_per_test_point,3);
    test_selected_em{n_index} = false(numel(P_list),numel(SNR_dB),samples_per_test_point);
    test_rank_validation_error{n_index} = nan(numel(P_list),numel(SNR_dB),samples_per_test_point);
    test_iterations{n_index} = zeros(numel(P_list),numel(SNR_dB), ...
        samples_per_test_point,number_of_methods);
    test_converged{n_index} = false(numel(P_list),numel(SNR_dB), ...
        samples_per_test_point,number_of_methods);
    test_completed{n_index} = false(numel(P_list),numel(SNR_dB),samples_per_test_point);
end

if isfile(checkpoint_file)
    C = load(checkpoint_file);
    if ~isequal(C.N_list,N_list) || ~isequal(C.P_list,P_list) || ...
            ~isequal(C.SNR_dB,SNR_dB) || ~isequal(C.p_H_list,p_H_list) || ...
            ~isequal(C.rank_lists,rank_lists) || ...
            C.samples_per_test_point~=samples_per_test_point || ...
            C.sample_seed_base~=sample_seed_base || ...
            C.rank_selection_updates~=rank_selection_updates || ...
            C.maximum_updates~=maximum_updates || ...
            C.convergence_tolerance~=convergence_tolerance
        error('V32 checkpoint settings do not match this script. Preserve the checkpoint.');
    end
    test_squared_error = C.test_squared_error;
    test_channel_energy = C.test_channel_energy;
    test_selected_rank = C.test_selected_rank;
    test_selected_em = C.test_selected_em;
    test_rank_validation_error = C.test_rank_validation_error;
    test_iterations = C.test_iterations;
    test_converged = C.test_converged;
    test_completed = C.test_completed;
    completed_count = sum(cellfun(@(x)sum(x,'all'),test_completed));
    fprintf('Resuming V32: %d/%d samples complete.\n',completed_count, ...
        numel(N_list)*numel(P_list)*numel(SNR_dB)*samples_per_test_point);
end

pool = gcp('nocreate');
if isempty(pool), parpool('local',workers_if_starting_pool); pool = gcp('nocreate'); end
fprintf(['V32 pilot-selected EM fallback fresh test: %d workers, %d samples per N/P/SNR point.\n' ...
    'Rank selection: 70/30 pilot split, %d EM updates per candidate, ranks 1:rmax.\n'], ...
    pool.NumWorkers,samples_per_test_point,rank_selection_updates);
fprintf('Final HS-GS/EM-GS/biased-GS/GD runs use a common convergence threshold.\n');

for n_index = 1:numel(N_list)
    N = N_list(n_index);
    p_H = p_H_list(n_index);
    rank_list = rank_lists{n_index};
    for P_index = 1:numel(P_list)
        P = P_list(P_index);
        for SNR_index = 1:numel(SNR_dB)
            snr_db = SNR_dB(SNR_index);
            point_timer = tic;
            fprintf('Starting V32: N=%d, P=%d, SNR=%d dB.\n',N,P,snr_db);
            for block_start = 1:samples_per_checkpoint:samples_per_test_point
                positions = block_start:min(block_start+samples_per_checkpoint-1, ...
                    samples_per_test_point);
                was_completed = reshape(test_completed{n_index}(P_index,SNR_index,positions),1,[]);
                pending = positions(~was_completed);
                if isempty(pending), continue; end

                block_error = nan(numel(pending),number_of_methods);
                block_energy = nan(numel(pending),1);
                block_rank = nan(numel(pending),3);
                block_rank_error = nan(numel(pending),1);
                block_selected_em = false(numel(pending),1);
                block_iterations = zeros(numel(pending),number_of_methods);
                block_converged = false(numel(pending),number_of_methods);
                parfor local_index = 1:numel(pending)
                    sample_index = pending(local_index);
                    seed = sample_seed_base + 100000000*n_index + ...
                        1000000*P + 10000*SNR_index + sample_index;
                    [Y,G,S,B,sigma2] = v32_generate_sample(N,K,P,snr_db,RSR_dB, ...
                        d_over_lambda,mu_eg,hbar,seed);

                    [rank_min,rank_one_se,best_hs_rank,use_em_fallback,heldout_error] = v32_select_with_em_fallback(Y,S,B,sigma2, ...
                        p_H,rank_list,rank_selection_updates,training_fraction, ...
                        cadzow_sweeps,seed);
                    if use_em_fallback, selected_rank = rank_list(end);
                    else, selected_rank = best_hs_rank; end
                    [G_em,em_updates,em_ok] = v32_em_converged(Y,S,B,sigma2, ...
                        maximum_updates,convergence_tolerance);
                    if rank_min == rank_list(end)
                        G_min = G_em; min_updates = em_updates; min_ok = em_ok;
                    else
                        [G_min,min_updates,min_ok] = v32_hs_converged(Y,S,B,sigma2, ...
                            p_H,rank_min,cadzow_sweeps,maximum_updates, ...
                            convergence_tolerance);
                    end
                    if use_em_fallback
                        G_frozen = G_em; frozen_updates = em_updates; frozen_ok = em_ok;
                    elseif selected_rank == rank_min
                        G_frozen = G_min; frozen_updates = min_updates; frozen_ok = min_ok;
                    else
                        [G_frozen,frozen_updates,frozen_ok] = v32_hs_converged(Y,S,B,sigma2, ...
                            p_H,selected_rank,cadzow_sweeps,maximum_updates,convergence_tolerance);
                    end
                    [G_gs,gs_updates,gs_ok] = v32_gs_converged(Y,S,B, ...
                        maximum_updates,convergence_tolerance);
                    [G_gd,gd_updates,gd_ok] = v32_gd_converged(Y,S,B, ...
                        maximum_updates,convergence_tolerance,gd_initial_variance,seed);

                    block_error(local_index,:) = [norm(G_frozen-G,'fro')^2, ...
                        norm(G_min-G,'fro')^2,norm(G_em-G,'fro')^2, ...
                        norm(G_gs-G,'fro')^2,norm(G_gd-G,'fro')^2];
                    block_energy(local_index) = norm(G,'fro')^2;
                    block_rank(local_index,:) = [selected_rank,rank_min,rank_one_se];
                    block_rank_error(local_index) = heldout_error;
                    block_selected_em(local_index) = use_em_fallback;
                    block_iterations(local_index,:) = [frozen_updates,min_updates,em_updates,gs_updates,gd_updates];
                    block_converged(local_index,:) = [frozen_ok,min_ok,em_ok,gs_ok,gd_ok];
                end

                test_squared_error{n_index}(P_index,SNR_index,pending,:) = ...
                    reshape(block_error,[1 1 numel(pending) number_of_methods]);
                test_channel_energy{n_index}(P_index,SNR_index,pending) = block_energy;
                test_selected_rank{n_index}(P_index,SNR_index,pending,:) = ...
                    reshape(block_rank,[1 1 numel(pending) 3]);
                test_selected_em{n_index}(P_index,SNR_index,pending) = block_selected_em;
                test_rank_validation_error{n_index}(P_index,SNR_index,pending) = block_rank_error;
                test_iterations{n_index}(P_index,SNR_index,pending,:) = ...
                    reshape(block_iterations,[1 1 numel(pending) number_of_methods]);
                test_converged{n_index}(P_index,SNR_index,pending,:) = ...
                    reshape(block_converged,[1 1 numel(pending) number_of_methods]);
                test_completed{n_index}(P_index,SNR_index,pending) = true;
                save(checkpoint_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB', ...
                    'rank_lists','training_fraction','rank_selection_updates','cadzow_sweeps', ...
                    'samples_per_test_point','sample_seed_base','maximum_updates', ...
                    'convergence_tolerance','method_names','test_selected_em', ...
                    'test_squared_error', ...
                    'test_channel_energy','test_selected_rank','test_rank_validation_error', ...
                    'test_iterations','test_converged','test_completed','-v7.3');
                fprintf('  %d/%d samples done; %.1f s elapsed for this point.\n', ...
                    sum(test_completed{n_index}(P_index,SNR_index,:),'all'), ...
                    samples_per_test_point,toc(point_timer));
            end
        end
    end
end

for n_index = 1:numel(N_list)
    if ~all(test_completed{n_index},'all') || ...
            any(~isfinite(test_squared_error{n_index}),'all') || ...
            any(~isfinite(test_channel_energy{n_index}),'all') || ...
            any(test_channel_energy{n_index}<=0,'all') || ...
            any(~isfinite(test_selected_rank{n_index}),'all')
        error('V32 has incomplete or invalid values. Its checkpoint is preserved.');
    end
end

% Ratio-of-sums NMSE and paired bootstrap CIs. Positive gain favors the
% pilot-selected HS/EM hybrid over the comparator named in the gain column.
rng(2026103131,'twister');
number_of_rows = numel(N_list)*numel(P_list)*numel(SNR_dB);
rows = zeros(number_of_rows,36);
fallback_fraction = zeros(number_of_rows,1);
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
                    sample_indices = randi(samples_per_test_point,samples_per_test_point,1);
                    bootstrap_gain(draw) = 10*log10( ...
                        sum(errors(sample_indices,baseline))/sum(errors(sample_indices,1)));
                end
                bootstrap_gain = sort(bootstrap_gain);
                ci_low(baseline-1) = bootstrap_gain(max(1,round(0.025*bootstrap_replicates)));
                ci_high(baseline-1) = bootstrap_gain(min(bootstrap_replicates, ...
                    round(0.975*bootstrap_replicates)));
            end
            ranks = reshape(test_selected_rank{n_index}(P_index,SNR_index,:,:), ...
                samples_per_test_point,3);
            mean_ranks = mean(ranks,1);
            fraction_max = mean(ranks==rank_lists{n_index}(end),1);
            mean_iterations = reshape(mean(test_iterations{n_index}(P_index,SNR_index,:,:),3),1,[]);
            convergence_fraction = reshape(mean(test_converged{n_index}(P_index,SNR_index,:,:),3),1,[]);
            rows(row_number,:) = [N_list(n_index),P_list(P_index),SNR_dB(SNR_index), ...
                mean_ranks,fraction_max,nmse_db,gains,ci_low,ci_high, ...
                mean_iterations,convergence_fraction];
            fallback_fraction(row_number) = mean( ...
                reshape(test_selected_em{n_index}(P_index,SNR_index,:),[],1));
        end
    end
end

result_table = array2table(rows,'VariableNames',{'N','P','SNR_dB', ...
    'mean_rank_hybrid','mean_rank_minError','mean_rank_oneSE', ...
    'fraction_max_hybrid','fraction_max_minError','fraction_max_oneSE', ...
    'HS_EM_hybrid_NMSE_dB','HS_GS_minError_NMSE_dB','EM_GS_NMSE_dB', ...
    'biased_GS_NMSE_dB','GD_NMSE_dB','gain_over_minError_dB','gain_over_EM_GS_dB', ...
    'gain_over_biased_GS_dB','gain_over_GD_dB','CI_low_minError','CI_low_EM_GS', ...
    'CI_low_biased_GS','CI_low_GD','CI_high_minError','CI_high_EM_GS', ...
    'CI_high_biased_GS','CI_high_GD','mean_iter_hybrid','mean_iter_minError', ...
    'mean_iter_EM_GS','mean_iter_biased_GS','mean_iter_GD','conv_fraction_hybrid', ...
    'conv_fraction_minError','conv_fraction_EM_GS','conv_fraction_biased_GS', ...
    'conv_fraction_GD'});
result_table.EM_fallback_fraction = fallback_fraction;

figure_handle = figure('Color','w','Name','V32 pilot-selected EM fallback fresh test');
tiledlayout(numel(N_list),numel(P_list),'TileSpacing','compact','Padding','compact');
method_columns = {'HS_EM_hybrid_NMSE_dB','HS_GS_minError_NMSE_dB', ...
    'EM_GS_NMSE_dB','biased_GS_NMSE_dB','GD_NMSE_dB'};
for n_index = 1:numel(N_list)
    for P_index = 1:numel(P_list)
        nexttile; hold on;
        selected_rows = result_table.N==N_list(n_index) & result_table.P==P_list(P_index);
        for method_index = 1:number_of_methods
            plot(SNR_dB,result_table{selected_rows,method_columns{method_index}}, ...
                '-o','LineWidth',1.4,'MarkerSize',4,'DisplayName', ...
                strrep(method_names{method_index},'_',' '));
        end
        grid on; xlabel('SNR (dB)'); ylabel('NMSE (dB)');
        title(sprintf('N = %d, P = %d',N_list(n_index),P_list(P_index)));
        if n_index==1 && P_index==1, legend('Location','southwest'); end
    end
end
sgtitle(sprintf('V32: pilot-selected EM fallback rank policy on fresh data; RSR = %g dB',RSR_dB));

writetable(result_table,table_file);
save(result_file,'result_table', ...
    'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB', ...
    'rank_lists','training_fraction','rank_selection_updates','cadzow_sweeps', ...
    'samples_per_test_point','sample_seed_base','maximum_updates', ...
    'convergence_tolerance','bootstrap_replicates','method_names', ...
    'test_squared_error','test_channel_energy','test_selected_rank','test_selected_em', ...
    'test_rank_validation_error','test_iterations','test_converged','-v7.3');
saveas(figure_handle,figure_file);
savefig(figure_handle,figure_editable_file);
disp(result_table);
fprintf('V32 table: %s\nV32 figure: %s\nV32 result data: %s\n', ...
    table_file,figure_file,result_file);

function [rank_min,rank_one_se,best_hs_rank,use_em_fallback,hs_error] = ...
        v32_select_with_em_fallback(Y,S,B,sigma2,p_H,rank_list, ...
        selection_updates,training_fraction,sweeps,sample_seed)
P = size(S,2);
rng(mod(double(sample_seed)+76543,2^32-1),'twister');
pilot_order = randperm(P);
training_count = round(training_fraction*P);
training_pilots = pilot_order(1:training_count);
heldout_pilots = pilot_order(training_count+1:end);
error_by_rank_and_pilot = zeros(numel(heldout_pilots),numel(rank_list));
for rank_index = 1:numel(rank_list)
    G_candidate = v32_em_fixed(Y(:,training_pilots),S(:,training_pilots), ...
        B(:,training_pilots),sigma2,selection_updates,p_H,rank_list(rank_index),sweeps);
    prediction = abs(G_candidate*S(:,heldout_pilots)+B(:,heldout_pilots));
    error_by_rank_and_pilot(:,rank_index) = ...
        mean((Y(:,heldout_pilots)-prediction).^2,1).';
end
mean_error = mean(error_by_rank_and_pilot,1);
[~,minimum_index] = min(mean_error);
rank_min = rank_list(minimum_index);
minimum_standard_error = std(error_by_rank_and_pilot(:,minimum_index),0,1)/ ...
    sqrt(numel(heldout_pilots));
eligible = find(mean_error<=mean_error(minimum_index)+minimum_standard_error);
rank_one_se = rank_list(eligible(end));

% Compare the best constrained rank against the rank-max EM candidate on
% the same held-out pilots. Choose HS-GS only when its paired prediction
% improvement exceeds one standard error; otherwise fall back to EM-GS.
[hs_error,hs_index] = min(mean_error(1:end-1));
best_hs_rank = rank_list(hs_index);
paired_improvement = error_by_rank_and_pilot(:,end)- ...
    error_by_rank_and_pilot(:,hs_index);
improvement_standard_error = std(paired_improvement,0,1)/ ...
    sqrt(numel(heldout_pilots));
use_em_fallback = mean(paired_improvement)<=improvement_standard_error;
end

function G = v32_em_fixed(Y,S,B,sigma2,number_of_updates,p_H,rank_value,sweeps)
G = v32_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
for update_index = 1:number_of_updates
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if rank_value < min(size(G,1)-p_H,p_H+1)
        G = v32_hankel_project(G,p_H,rank_value,sweeps);
    end
end
end

function [G,iteration,converged] = v32_hs_converged(Y,S,B,sigma2,p_H,rank_value, ...
        sweeps,max_updates,tolerance)
G = v32_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    G = v32_hankel_project(G,p_H,rank_value,sweeps);
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = v32_em_converged(Y,S,B,sigma2,max_updates,tolerance)
G = v32_spectral_initialize(Y,S,B);
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

function [G,iteration,converged] = v32_gs_converged(Y,S,B,max_updates,tolerance)
G = v32_spectral_initialize(Y,S,B);
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

function [G,iteration,converged] = v32_gd_converged(Y,S,B,max_updates,tolerance,initial_variance,seed)
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

function G = v32_spectral_initialize(Y,S,B)
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

function G = v32_hankel_project(G,p_H,rank_value,sweeps)
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

function [Y,G,S,B,sigma2] = v32_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
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







