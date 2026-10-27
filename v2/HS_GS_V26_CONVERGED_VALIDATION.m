%% HS_GS_V26_CONVERGED_VALIDATION.m
% Validation-only rank/duration search with convergence-matched EM/GS baselines.
% Every HS-GS candidate applies Hankel projection only for a chosen initial
% number of EM updates, then continues EM until the same relative-change
% tolerance used by the converged baselines. The policy is selected from
% three fresh validation seed groups and compared directly with converged GS.
% This script does not create or score an independent final test set.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V26_CONVERGED_VALIDATION
%
% The script saves a checkpoint after each small block and resumes when
% rerun. Do not remove its checkpoint if you need to continue an interrupted run.

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end

checkpoint_file = fullfile(result_folder,'HS_GS_V26_validation_checkpoint.mat');
policy_file = fullfile(result_folder,'HS_GS_V26_frozen_validation_policy.mat');
table_file = fullfile(result_folder,'HS_GS_V26_validation_candidates.csv');
figure_file = fullfile(result_folder,'HS_GS_V26_validation_gain_over_GS.png');
figure_editable_file = fullfile(result_folder,'HS_GS_V26_validation_gain_over_GS.fig');
if isfile(policy_file) || isfile(table_file)
    error('V26 outputs already exist. Inspect them before starting a new validation run.');
end

% Same channel/pilot/reference model and benchmark grid as v2.
N_list = [8 16];
p_H_list = [3 7];
P_list = [10 30];
SNR_dB = -5:5:30;
K = 3;
RSR_dB = 25;
d_over_lambda = 0.5;
mu_eg = [0; 1785.9 * 1.602e-19 * 5.292e-11; 0];
hbar = 6.62607015e-34 / (2*pi);

% Independent validation seed groups; seeds are below MATLAB's 2^32 limit.
seed_base_by_split = [3300000000 3400000000 3500000000];
number_of_splits = numel(seed_base_by_split);
samples_per_split = 100;
samples_per_checkpoint = 10;
maximum_updates = 5000;
convergence_tolerance = 1e-10;
cadzow_sweeps = 4;
workers_if_starting_pool = 6;

% Search ranks 1:maxRank and projection durations 5, 25, or 50 updates.
% The [maxRank,0] candidate is the unprojected EM-GS convergence control.
candidate_settings_by_N = cell(1,numel(N_list));
for n_index = 1:numel(N_list)
    N = N_list(n_index);
    p_H = p_H_list(n_index);
    maximum_rank = min(N-p_H,p_H+1);
    candidates = [maximum_rank 0];
    for rank_value = 1:maximum_rank
        for projection_duration = [5 25 50]
            candidates(end+1,:) = [rank_value projection_duration]; %#ok<SAGROW>
        end
    end
    candidate_settings_by_N{n_index} = candidates;
end

number_of_candidates_by_N = cellfun(@(x) size(x,1),candidate_settings_by_N);
validation_squared_error = cell(1,numel(N_list));
validation_channel_energy = cell(1,numel(N_list));
validation_GS_squared_error = cell(1,numel(N_list));
validation_iterations = cell(1,numel(N_list));
validation_GS_iterations = cell(1,numel(N_list));
validation_completed = cell(1,numel(N_list));
validation_converged = cell(1,numel(N_list));
validation_GS_converged = cell(1,numel(N_list));
for n_index = 1:numel(N_list)
    number_of_candidates = number_of_candidates_by_N(n_index);
    validation_squared_error{n_index} = nan(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split,number_of_candidates);
    validation_channel_energy{n_index} = nan(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split);
    validation_GS_squared_error{n_index} = nan(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split);
    validation_iterations{n_index} = zeros(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split,number_of_candidates);
    validation_GS_iterations{n_index} = zeros(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split);
    validation_completed{n_index} = false(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split);
    validation_converged{n_index} = false(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split,number_of_candidates);
    validation_GS_converged{n_index} = false(numel(P_list),numel(SNR_dB), ...
        number_of_splits,samples_per_split);
end

if isfile(checkpoint_file)
    C = load(checkpoint_file);
    if ~isequal(C.N_list,N_list) || ~isequal(C.p_H_list,p_H_list) || ...
            ~isequal(C.P_list,P_list) || ~isequal(C.SNR_dB,SNR_dB) || ...
            ~isequal(C.seed_base_by_split,seed_base_by_split) || ...
            C.samples_per_split~=samples_per_split || C.maximum_updates~=maximum_updates || ...
            C.convergence_tolerance~=convergence_tolerance || ...
            ~isequal(C.candidate_settings_by_N,candidate_settings_by_N)
        error('V26 checkpoint settings mismatch. Preserve and inspect it; do not mix runs.');
    end
    validation_squared_error = C.validation_squared_error;
    validation_channel_energy = C.validation_channel_energy;
    validation_GS_squared_error = C.validation_GS_squared_error;
    validation_iterations = C.validation_iterations;
    validation_GS_iterations = C.validation_GS_iterations;
    validation_completed = C.validation_completed;
    validation_converged = C.validation_converged;
    validation_GS_converged = C.validation_GS_converged;
    % Recover finished blocks from the finite checkpoint values. The first
    % V26 draft wrote result arrays but omitted setting validation_completed.
    % A saved block is complete only when every candidate and GS score exists.
    for n_index = 1:numel(N_list)
        candidate_scores_present = all(isfinite(validation_squared_error{n_index}),5);
        recovered_completed = isfinite(validation_channel_energy{n_index}) & ...
            isfinite(validation_GS_squared_error{n_index}) & ...
            validation_channel_energy{n_index}>0 & ...
            reshape(candidate_scores_present,size(validation_completed{n_index}));
        validation_completed{n_index} = validation_completed{n_index} | recovered_completed;
    end
    fprintf('Resuming V26: %d/%d samples complete.\n', ...
        sum(cellfun(@(x) sum(x,'all'),validation_completed)), ...
        numel(N_list)*numel(P_list)*numel(SNR_dB)*number_of_splits*samples_per_split);
end

pool = gcp('nocreate');
if isempty(pool)
    parpool('local',workers_if_starting_pool);
    pool = gcp('nocreate');
end
fprintf('V26 convergence-matched validation: %d workers.\n',pool.NumWorkers);
fprintf('%d validation samples per N/P/SNR point; %d candidates for N=8 and %d for N=16.\n', ...
    number_of_splits*samples_per_split,number_of_candidates_by_N(1),number_of_candidates_by_N(2));

for n_index = 1:numel(N_list)
    N = N_list(n_index);
    p_H = p_H_list(n_index);
    candidate_settings = candidate_settings_by_N{n_index};
    number_of_candidates = size(candidate_settings,1);
    for P_index = 1:numel(P_list)
        P = P_list(P_index);
        for SNR_index = 1:numel(SNR_dB)
            snr_db = SNR_dB(SNR_index);
            point_timer = tic;
            for split_index = 1:number_of_splits
                for block_start = 1:samples_per_checkpoint:samples_per_split
                    positions = block_start:min(block_start+samples_per_checkpoint-1,samples_per_split);
                    already_done = reshape(validation_completed{n_index}( ...
                        P_index,SNR_index,split_index,positions),1,[]);
                    pending = positions(~already_done);
                    if isempty(pending), continue; end

                    block_error = nan(numel(pending),number_of_candidates);
                    block_energy = nan(numel(pending),1);
                    block_GS_error = nan(numel(pending),1);
                    block_iterations = zeros(numel(pending),number_of_candidates);
                    block_GS_iterations = zeros(numel(pending),1);
                    block_candidate_converged = false(numel(pending),number_of_candidates);
                    block_GS_converged = false(numel(pending),1);
                    parfor local_index = 1:numel(pending)
                        sample_index = pending(local_index);
                        seed = seed_base_by_split(split_index) + ...
                            100000000*n_index + 1000000*P + sample_index;
                        [Y,G,S,B,sigma2] = v26_generate_sample(N,K,P,snr_db,RSR_dB, ...
                            d_over_lambda,mu_eg,hbar,seed);
                        [G_GS,gs_updates,gs_ok] = v26_gs_converged(Y,S,B, ...
                            maximum_updates,convergence_tolerance);
                        block_energy(local_index) = norm(G,'fro')^2;
                        block_GS_error(local_index) = norm(G_GS-G,'fro')^2;
                        block_GS_iterations(local_index) = gs_updates;
                        block_GS_converged(local_index) = gs_ok;
                        for candidate_index = 1:number_of_candidates
                            rank_value = candidate_settings(candidate_index,1);
                            projected_updates = candidate_settings(candidate_index,2);
                            [G_candidate,updates,converged] = v26_hybrid_em_converged( ...
                                Y,S,B,sigma2,rank_value,projected_updates,p_H, ...
                                cadzow_sweeps,maximum_updates,convergence_tolerance);
                            block_error(local_index,candidate_index) = norm(G_candidate-G,'fro')^2;
                            block_iterations(local_index,candidate_index) = updates;
                            block_candidate_converged(local_index,candidate_index) = converged;
                        end
                    end

                    validation_squared_error{n_index}(P_index,SNR_index,split_index,pending,:) = ...
                        reshape(block_error,[1 1 1 numel(pending) number_of_candidates]);
                    validation_channel_energy{n_index}(P_index,SNR_index,split_index,pending) = block_energy;
                    validation_GS_squared_error{n_index}(P_index,SNR_index,split_index,pending) = block_GS_error;
                    validation_iterations{n_index}(P_index,SNR_index,split_index,pending,:) = ...
                        reshape(block_iterations,[1 1 1 numel(pending) number_of_candidates]);
                    validation_GS_iterations{n_index}(P_index,SNR_index,split_index,pending) = block_GS_iterations;
                    validation_converged{n_index}(P_index,SNR_index,split_index,pending,:) = ...
                        reshape(block_candidate_converged,[1 1 1 numel(pending) number_of_candidates]);
                    validation_GS_converged{n_index}(P_index,SNR_index,split_index,pending) = block_GS_converged;
                    validation_completed{n_index}(P_index,SNR_index,split_index,pending) = true;

                    save(checkpoint_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB', ...
                        'seed_base_by_split','number_of_splits','samples_per_split', ...
                        'maximum_updates','convergence_tolerance','cadzow_sweeps', ...
                        'candidate_settings_by_N','validation_squared_error', ...
                        'validation_channel_energy','validation_GS_squared_error', ...
                        'validation_iterations','validation_GS_iterations','validation_completed', ...
                        'validation_converged','validation_GS_converged','-v7.3');
                    fprintf('N=%d, P=%d, SNR=%d dB, split %d/%d: %d/%d samples; %.1f s.\n', ...
                        N,P,snr_db,split_index,number_of_splits, ...
                        sum(validation_completed{n_index}(P_index,SNR_index,split_index,:),'all'), ...
                        samples_per_split,toc(point_timer));
                end
            end
            fprintf('Validated N=%d, P=%d, SNR=%d dB.\n',N,P,snr_db);
        end
    end
end

% The GS reference must converge and every score must exist. Some individual
% HS-GS candidates may hit the iteration cap; those candidates are excluded
% below rather than invalidating all completed validation results.
for n_index = 1:numel(N_list)
    if ~all(validation_completed{n_index},'all') || ...
            any(~isfinite(validation_squared_error{n_index}),'all') || ...
            any(~isfinite(validation_GS_squared_error{n_index}),'all') || ...
            any(~validation_GS_converged{n_index},'all')
        error('V26 validation is incomplete or the converged GS reference failed; checkpoint preserved.');
    end
end

% Select by maximin split gain over converged biased GS. Positive gain favors HS-GS.
selected_candidate = zeros(numel(N_list),numel(P_list),numel(SNR_dB));
selected_rank = zeros(size(selected_candidate));
selected_projected_updates = zeros(size(selected_candidate));
selected_validation_gain_db = zeros(size(selected_candidate));
selected_min_split_gain_db = zeros(size(selected_candidate));
selected_split_wins = zeros(size(selected_candidate));
selected_passes_all_splits = false(size(selected_candidate));
candidate_rows = cell(0,14);
for n_index = 1:numel(N_list)
    candidates = candidate_settings_by_N{n_index};
    for P_index = 1:numel(P_list)
        for SNR_index = 1:numel(SNR_dB)
            pooled_energy = sum(validation_channel_energy{n_index}( ...
                P_index,SNR_index,:,:),'all');
            pooled_GS_error = sum(validation_GS_squared_error{n_index}( ...
                P_index,SNR_index,:,:),'all');
            pooled_GS_nmse_db = 10*log10(pooled_GS_error/pooled_energy);
            split_gain_db = nan(number_of_splits,size(candidates,1));
            pooled_gain_db = nan(1,size(candidates,1));
            candidate_nmse_db = nan(1,size(candidates,1));
            candidate_mean_iterations = nan(1,size(candidates,1));
            candidate_convergence_fraction = nan(1,size(candidates,1));
            candidate_eligible = false(1,size(candidates,1));
            for candidate_index = 1:size(candidates,1)
                candidate_error = validation_squared_error{n_index}( ...
                    P_index,SNR_index,:,:,:);
                candidate_error_sum = sum(candidate_error(:,:,:,:,candidate_index),'all');
                candidate_nmse_db(candidate_index) = 10*log10(candidate_error_sum/pooled_energy);
                pooled_gain_db(candidate_index) = pooled_GS_nmse_db-candidate_nmse_db(candidate_index);
                candidate_mean_iterations(candidate_index) = mean(validation_iterations{n_index}( ...
                    P_index,SNR_index,:,:,candidate_index),'all');
                candidate_convergence_fraction(candidate_index) = mean(validation_converged{n_index}( ...
                    P_index,SNR_index,:,:,candidate_index),'all');
                candidate_eligible(candidate_index) = all(validation_converged{n_index}( ...
                    P_index,SNR_index,:,:,candidate_index),'all');
                for split_index = 1:number_of_splits
                    split_energy = sum(validation_channel_energy{n_index}( ...
                        P_index,SNR_index,split_index,:),'all');
                    split_GS_error = sum(validation_GS_squared_error{n_index}( ...
                        P_index,SNR_index,split_index,:),'all');
                    split_candidate_error = sum(validation_squared_error{n_index}( ...
                        P_index,SNR_index,split_index,:,candidate_index),'all');
                    split_GS_nmse_db = 10*log10(split_GS_error/split_energy);
                    split_candidate_nmse_db = 10*log10(split_candidate_error/split_energy);
                    split_gain_db(split_index,candidate_index) = split_GS_nmse_db-split_candidate_nmse_db;
                end
            end
            minimum_split_gain = min(split_gain_db,[],1);
            minimum_split_gain(~candidate_eligible) = -Inf;
            if ~any(candidate_eligible)
                error('No HS-GS candidate converged on every validation sample for N=%d, P=%d, SNR=%d dB.', ...
                    N_list(n_index),P_list(P_index),SNR_dB(SNR_index));
            end
            [~,best_index] = max(minimum_split_gain);
            selected_candidate(n_index,P_index,SNR_index) = best_index;
            selected_rank(n_index,P_index,SNR_index) = candidates(best_index,1);
            selected_projected_updates(n_index,P_index,SNR_index) = candidates(best_index,2);
            selected_validation_gain_db(n_index,P_index,SNR_index) = pooled_gain_db(best_index);
            selected_min_split_gain_db(n_index,P_index,SNR_index) = minimum_split_gain(best_index);
            selected_split_wins(n_index,P_index,SNR_index) = sum(split_gain_db(:,best_index)>0);
            selected_passes_all_splits(n_index,P_index,SNR_index) = minimum_split_gain(best_index)>0;
            for candidate_index = 1:size(candidates,1)
                candidate_rows(end+1,:) = {N_list(n_index),P_list(P_index),SNR_dB(SNR_index), ...
                    candidates(candidate_index,1),candidates(candidate_index,2), ...
                    candidate_nmse_db(candidate_index),pooled_gain_db(candidate_index), ...
                    split_gain_db(1,candidate_index),split_gain_db(2,candidate_index), ...
                    split_gain_db(3,candidate_index),candidate_mean_iterations(candidate_index), ...
                    candidate_convergence_fraction(candidate_index),candidate_eligible(candidate_index), ...
                    candidate_index==best_index}; %#ok<SAGROW>
            end
        end
    end
end

candidate_table = cell2table(candidate_rows,'VariableNames', ...
    {'N','P','SNR_dB','Rank','Projected_EM_updates','HS_Validation_NMSE_dB', ...
    'Pooled_gain_over_GS_dB','Split1_gain_over_GS_dB','Split2_gain_over_GS_dB', ...
    'Split3_gain_over_GS_dB','Mean_EM_updates','Convergence_fraction', ...
    'Eligible_all_samples','Selected_maximin'});
writetable(candidate_table,table_file);
policy_source = ['Per N/P/SNR, selected the candidate that maximizes the minimum of its three ' ...
    'validation-group gains over converged biased GS. Positive gain means HS-GS is better.'];
save(policy_file,'N_list','p_H_list','P_list','SNR_dB','K','RSR_dB', ...
    'seed_base_by_split','number_of_splits','samples_per_split','maximum_updates', ...
    'convergence_tolerance','cadzow_sweeps','candidate_settings_by_N', ...
    'selected_candidate','selected_rank','selected_projected_updates', ...
    'selected_validation_gain_db','selected_min_split_gain_db','selected_split_wins', ...
    'selected_passes_all_splits','policy_source','candidate_table');

% Summarize whether any selected HS-GS candidate beats converged GS in every split.
policy_rows = zeros(numel(N_list)*numel(P_list)*numel(SNR_dB),8);
row_number = 0;
for n_index = 1:numel(N_list)
    for P_index = 1:numel(P_list)
        for SNR_index = 1:numel(SNR_dB)
            row_number = row_number+1;
            policy_rows(row_number,:) = [N_list(n_index),P_list(P_index),SNR_dB(SNR_index), ...
                selected_rank(n_index,P_index,SNR_index), ...
                selected_projected_updates(n_index,P_index,SNR_index), ...
                selected_validation_gain_db(n_index,P_index,SNR_index), ...
                selected_min_split_gain_db(n_index,P_index,SNR_index), ...
                selected_split_wins(n_index,P_index,SNR_index)];
        end
    end
end
policy_table = array2table(policy_rows,'VariableNames',{'N','P','SNR_dB','Rank', ...
    'Projected_EM_updates','Pooled_gain_over_GS_dB','Worst_split_gain_over_GS_dB','Split_wins'});
disp(policy_table);
fprintf('Selected candidates positive in all three splits: %d/%d N/P/SNR points.\n', ...
    sum(selected_passes_all_splits,'all'),numel(selected_passes_all_splits));

figure_handle = figure('Color','w','Name','V26 validation HS-GS gain over converged GS');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
line_colors = lines(numel(N_list)*numel(P_list));
line_index = 0;
for n_index = 1:numel(N_list)
    for P_index = 1:numel(P_list)
        line_index = line_index+1;
        nexttile;
        plot(SNR_dB,reshape(selected_validation_gain_db(n_index,P_index,:),1,[]), ...
            '-o','Color',line_colors(line_index,:),'LineWidth',1.5,'MarkerSize',5);
        hold on; yline(0,'k--','GS parity','LineWidth',1.0);
        grid on; xlabel('SNR (dB)'); ylabel('HS-GS gain over GS (dB)');
        title(sprintf('N = %d, P = %d',N_list(n_index),P_list(P_index)));
    end
end
sgtitle('V26 validation only: converged HS-GS candidates versus converged biased GS');
saveas(figure_handle,figure_file);
savefig(figure_handle,figure_editable_file);
fprintf('Candidate table: %s\nFrozen validation policy: %s\n',table_file,policy_file);
fprintf('Validation gain plot: %s\n',figure_file);

function [Y,G,S,B,sigma2] = v26_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
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

function [G,iteration,converged] = v26_hybrid_em_converged(Y,S,B,sigma2,rank_value, ...
    projected_updates,p_H,sweeps,max_updates,tolerance)
G = v26_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false;
iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if update_index <= projected_updates
        G = v26_hankel_project(G,p_H,rank_value,sweeps);
    end
    if update_index > projected_updates && ...
            norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration = update_index;
        converged = true;
        break;
    end
end
end

function [G,iteration,converged] = v26_gs_converged(Y,S,B,max_updates,tolerance)
G = v26_spectral_initialize(Y,S,B);
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

function G = v26_spectral_initialize(Y,S,B)
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

function G = v26_hankel_project(G,p_H,rank_value,sweeps)
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
