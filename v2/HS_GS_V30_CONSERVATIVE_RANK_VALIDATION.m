%% HS-GS V30: validation-only comparison of two pilot-based rank rules
% V30 compares:
%   A) minimum held-out pilot prediction error (the V28/V29 rule), and
%   B) the largest rank within one standard error of the minimum error.
%
% Rank selection uses pilot measurements only. Channel truth is used only
% on this separate validation set to choose which rank rule to freeze for
% each P/SNR condition. V31 must evaluate that frozen policy on new samples.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V30_CONSERVATIVE_RANK_VALIDATION
%
% Checkpoint is saved every block and reused if this script is rerun.

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end

checkpoint_file = fullfile(result_folder,'HS_GS_V30_validation_checkpoint.mat');
result_file = fullfile(result_folder,'HS_GS_V30_validation_results.mat');
table_file = fullfile(result_folder,'HS_GS_V30_validation_table.csv');
policy_file = fullfile(result_folder,'HS_GS_V30_frozen_rank_policy.mat');
figure_file = fullfile(result_folder,'HS_GS_V30_validation_gain.png');
if isfile(result_file) || isfile(table_file) || isfile(policy_file)
    error('V30 validation outputs already exist. Inspect them before starting another run.');
end

N = 16;
p_H = 7;
P_list = [10 30];
SNR_dB = -5:5:30;
K = 3;
RSR_dB = 25;
rank_list = 1:min(N-p_H,p_H+1); % H is 9-by-8; ranks 1 through 8.
validation_samples_per_point = 200;
samples_per_checkpoint = 10;
validation_seed_base = 2200000000;
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

% Methods stored in the last dimension: min-error HS, one-SE HS, EM-GS.
number_of_methods = 3;
validation_squared_error = nan(numel(P_list),numel(SNR_dB), ...
    validation_samples_per_point,number_of_methods);
validation_channel_energy = nan(numel(P_list),numel(SNR_dB),validation_samples_per_point);
validation_selected_rank = nan(numel(P_list),numel(SNR_dB),validation_samples_per_point,2);
validation_completed = false(numel(P_list),numel(SNR_dB),validation_samples_per_point);

if isfile(checkpoint_file)
    C = load(checkpoint_file);
    if C.N~=N || C.p_H~=p_H || ~isequal(C.P_list,P_list) || ...
            ~isequal(C.SNR_dB,SNR_dB) || ~isequal(C.rank_list,rank_list) || ...
            C.validation_samples_per_point~=validation_samples_per_point || ...
            C.validation_seed_base~=validation_seed_base || ...
            C.rank_selection_updates~=rank_selection_updates || ...
            C.maximum_updates~=maximum_updates || ...
            C.convergence_tolerance~=convergence_tolerance
        error('V30 checkpoint settings do not match. Preserve the existing checkpoint.');
    end
    validation_squared_error = C.validation_squared_error;
    validation_channel_energy = C.validation_channel_energy;
    validation_selected_rank = C.validation_selected_rank;
    validation_completed = C.validation_completed;
    fprintf('Resuming V30: %d/%d validation samples complete.\n', ...
        sum(validation_completed,'all'),numel(validation_completed));
end

pool = gcp('nocreate');
if isempty(pool), parpool('local',workers_if_starting_pool); pool = gcp('nocreate'); end
fprintf(['V30 validation: %d workers, N=%d, P=[10 30], %d samples per P/SNR.\n' ...
    'Compare minimum-error rank with the conservative one-SE rank rule.\n'], ...
    pool.NumWorkers,N,validation_samples_per_point);

for P_index = 1:numel(P_list)
    P = P_list(P_index);
    for SNR_index = 1:numel(SNR_dB)
        snr_db = SNR_dB(SNR_index);
        point_timer = tic;
        fprintf('Starting validation: P=%d, SNR=%d dB.\n',P,snr_db);
        for block_start = 1:samples_per_checkpoint:validation_samples_per_point
            positions = block_start:min(block_start+samples_per_checkpoint-1, ...
                validation_samples_per_point);
            already_done = reshape(validation_completed(P_index,SNR_index,positions),1,[]);
            pending = positions(~already_done);
            if isempty(pending), continue; end

            block_error = nan(numel(pending),number_of_methods);
            block_energy = nan(numel(pending),1);
            block_ranks = nan(numel(pending),2);
            parfor local_index = 1:numel(pending)
                sample_index = pending(local_index);
                seed = validation_seed_base+1000000*P+10000*SNR_index+sample_index;
                [Y,G,S,B,sigma2] = v30_generate_sample(N,K,P,snr_db,RSR_dB, ...
                    d_over_lambda,mu_eg,hbar,seed);
                [rank_min,rank_one_se] = v30_select_two_ranks(Y,S,B,sigma2,p_H, ...
                    rank_list,rank_selection_updates,training_fraction,cadzow_sweeps,seed);

                [G_em,~,~] = v30_em_converged(Y,S,B,sigma2, ...
                    maximum_updates,convergence_tolerance);
                if rank_min==rank_list(end)
                    G_min = G_em;
                else
                    [G_min,~,~] = v30_hs_converged(Y,S,B,sigma2,p_H,rank_min, ...
                        cadzow_sweeps,maximum_updates,convergence_tolerance);
                end
                if rank_one_se==rank_list(end)
                    G_one_se = G_em;
                elseif rank_one_se==rank_min
                    G_one_se = G_min;
                else
                    [G_one_se,~,~] = v30_hs_converged(Y,S,B,sigma2,p_H,rank_one_se, ...
                        cadzow_sweeps,maximum_updates,convergence_tolerance);
                end

                block_error(local_index,:) = [norm(G_min-G,'fro')^2, ...
                    norm(G_one_se-G,'fro')^2,norm(G_em-G,'fro')^2];
                block_energy(local_index) = norm(G,'fro')^2;
                block_ranks(local_index,:) = [rank_min,rank_one_se];
            end

            validation_squared_error(P_index,SNR_index,pending,:) = ...
                reshape(block_error,[1 1 numel(pending) number_of_methods]);
            validation_channel_energy(P_index,SNR_index,pending) = block_energy;
            validation_selected_rank(P_index,SNR_index,pending,:) = ...
                reshape(block_ranks,[1 1 numel(pending) 2]);
            validation_completed(P_index,SNR_index,pending) = true;
            save(checkpoint_file,'N','p_H','P_list','SNR_dB','K','RSR_dB','rank_list', ...
                'validation_samples_per_point','validation_seed_base','training_fraction', ...
                'rank_selection_updates','cadzow_sweeps','maximum_updates', ...
                'convergence_tolerance','validation_squared_error', ...
                'validation_channel_energy','validation_selected_rank', ...
                'validation_completed','-v7.3');
            fprintf('  %d/%d validation samples complete; %.1f s.\n', ...
                sum(validation_completed(P_index,SNR_index,:),'all'), ...
                validation_samples_per_point,toc(point_timer));
        end
    end
end

if ~all(validation_completed,'all') || ...
        any(~isfinite(validation_squared_error),'all') || ...
        any(~isfinite(validation_channel_energy),'all') || ...
        any(~isfinite(validation_selected_rank),'all')
    error('V30 validation is incomplete or contains invalid values. Checkpoint preserved.');
end

% Compare policies on validation truth, then freeze one policy per P/SNR.
% The independent V31 test must not influence this choice.
rng(2026103030,'twister');
rows = zeros(numel(P_list)*numel(SNR_dB),14);
frozen_rank_rule = strings(numel(P_list),numel(SNR_dB));
for P_index = 1:numel(P_list)
    for SNR_index = 1:numel(SNR_dB)
        errors = reshape(validation_squared_error(P_index,SNR_index,:,:), ...
            validation_samples_per_point,number_of_methods);
        energy = reshape(validation_channel_energy(P_index,SNR_index,:),[],1);
        nmse_db = 10*log10(sum(errors,1)/sum(energy));
        bootstrap_gain = zeros(bootstrap_replicates,1);
        for draw = 1:bootstrap_replicates
            indices = randi(validation_samples_per_point,validation_samples_per_point,1);
            bootstrap_gain(draw) = 10*log10(sum(errors(indices,1))/sum(errors(indices,2)));
        end
        bootstrap_gain = sort(bootstrap_gain);
        ci_low = bootstrap_gain(max(1,round(0.025*bootstrap_replicates)));
        ci_high = bootstrap_gain(min(bootstrap_replicates,round(0.975*bootstrap_replicates)));
        % Use the one-SE policy only if it improves validation NMSE with a
        % positive paired 95% lower bound; otherwise keep the V28 min-error rule.
        if ci_low>0
            frozen_rank_rule(P_index,SNR_index) = "oneSE";
        else
            frozen_rank_rule(P_index,SNR_index) = "minError";
        end
        ranks = reshape(validation_selected_rank(P_index,SNR_index,:,:), ...
            validation_samples_per_point,2);
        max_fraction = mean(ranks==rank_list(end),1);
        rows((P_index-1)*numel(SNR_dB)+SNR_index,:) = [P_list(P_index), ...
            SNR_dB(SNR_index),mean(ranks(:,1)),mean(ranks(:,2)), ...
            max_fraction(1),max_fraction(2),nmse_db, ...
            nmse_db(1)-nmse_db(2),ci_low,ci_high, ...
            double(frozen_rank_rule(P_index,SNR_index)=="oneSE"),validation_samples_per_point];
    end
end

validation_table = array2table(rows,'VariableNames',{'P','SNR_dB', ...
    'mean_rank_minError','mean_rank_oneSE','fraction_max_minError', ...
    'fraction_max_oneSE','NMSE_minError_dB','NMSE_oneSE_dB', ...
    'EM_GS_NMSE_dB','gain_oneSE_over_min_dB','CI_low_gain_dB', ...
    'CI_high_gain_dB','frozen_oneSE_flag','validation_samples'});

frozen_policy = struct('N',N,'p_H',p_H,'P_list',P_list,'SNR_dB',SNR_dB, ...
    'K',K,'RSR_dB',RSR_dB,'rank_list',rank_list, ...
    'frozen_rank_rule',frozen_rank_rule,'training_fraction',training_fraction, ...
    'rank_selection_updates',rank_selection_updates,'cadzow_sweeps',cadzow_sweeps, ...
    'maximum_updates',maximum_updates,'convergence_tolerance',convergence_tolerance, ...
    'validation_seed_base',validation_seed_base, ...
    'validation_samples_per_point',validation_samples_per_point, ...
    'policy_rule','oneSE only when validation paired-bootstrap CI lower bound > 0; otherwise minError');

figure_handle = figure('Color','w','Name','V30 validation selector comparison');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
for P_index = 1:numel(P_list)
    nexttile; hold on;
    selected = validation_table.P==P_list(P_index);
    plot(SNR_dB,validation_table.gain_oneSE_over_min_dB(selected),'-o', ...
        'LineWidth',1.5,'DisplayName','one-SE gain over min-error rule');
    yline(0,'k--','DisplayName','equal validation NMSE');
    grid on; xlabel('SNR (dB)'); ylabel('Gain (dB)');
    title(sprintf('N = %d, P = %d',N,P_list(P_index)));
    legend('Location','best');
end
sgtitle('V30 validation only: conservative rank rule versus minimum-error rule');

writetable(validation_table,table_file);
save(policy_file,'frozen_policy','validation_table');
save(result_file,'validation_table','frozen_policy','N','p_H','P_list','SNR_dB', ...
    'K','RSR_dB','rank_list','training_fraction','rank_selection_updates', ...
    'cadzow_sweeps','maximum_updates','convergence_tolerance', ...
    'validation_seed_base','validation_samples_per_point','bootstrap_replicates', ...
    'validation_squared_error','validation_channel_energy', ...
    'validation_selected_rank','frozen_rank_rule','-v7.3');
saveas(figure_handle,figure_file);
disp(validation_table);
fprintf('V30 validation table: %s\nFrozen rank policy: %s\nValidation figure: %s\n', ...
    table_file,policy_file,figure_file);

function [rank_min,rank_one_se] = v30_select_two_ranks(Y,S,B,sigma2,p_H,rank_list, ...
        selection_updates,training_fraction,sweeps,sample_seed)
P = size(S,2);
rng(mod(double(sample_seed)+76543,2^32-1),'twister');
pilot_order = randperm(P);
training_count = round(training_fraction*P);
training_pilots = pilot_order(1:training_count);
heldout_pilots = pilot_order(training_count+1:end);
error_by_rank_and_pilot = zeros(numel(heldout_pilots),numel(rank_list));
for rank_index = 1:numel(rank_list)
    G_candidate = v30_em_fixed(Y(:,training_pilots),S(:,training_pilots), ...
        B(:,training_pilots),sigma2,selection_updates,p_H,rank_list(rank_index),sweeps);
    prediction = abs(G_candidate*S(:,heldout_pilots)+B(:,heldout_pilots));
    error_by_rank_and_pilot(:,rank_index) = ...
        mean((Y(:,heldout_pilots)-prediction).^2,1).';
end
mean_error = mean(error_by_rank_and_pilot,1);
[minimum_error,best_index] = min(mean_error);
standard_error = std(error_by_rank_and_pilot(:,best_index),0,1)/sqrt(numel(heldout_pilots));
eligible = find(mean_error<=minimum_error+standard_error);
rank_min = rank_list(best_index);
rank_one_se = rank_list(eligible(end));
end

function G = v30_em_fixed(Y,S,B,sigma2,number_of_updates,p_H,rank_value,sweeps)
G = v30_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
for update_index = 1:number_of_updates
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if rank_value<min(size(G,1)-p_H,p_H+1)
        G = v30_hankel_project(G,p_H,rank_value,sweeps);
    end
end
end

function [G,iteration,converged] = v30_hs_converged(Y,S,B,sigma2,p_H,rank_value,sweeps,max_updates,tolerance)
G = v30_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    G = v30_hankel_project(G,p_H,rank_value,sweeps);
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps)<tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function [G,iteration,converged] = v30_em_converged(Y,S,B,sigma2,max_updates,tolerance)
G = v30_spectral_initialize(Y,S,B);
right_inverse = S'/(S*S');
converged = false; iteration = max_updates;
for update_index = 1:max_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    G = (Y.*exp(1j*angle(field)).*ratio-B)*right_inverse;
    if norm(G-previous_G,'fro')^2/max(norm(G,'fro')^2,eps)<tolerance
        iteration = update_index; converged = true; break;
    end
end
end

function G = v30_spectral_initialize(Y,S,B)
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

function G = v30_hankel_project(G,p_H,rank_value,sweeps)
for sweep = 1:sweeps
    for user = 1:size(G,2)
        x = G(:,user); N = length(x);
        number_of_rows = N-p_H; number_of_columns = p_H+1;
        H = zeros(number_of_rows,number_of_columns);
        for row = 1:number_of_rows
            H(row,:) = x(row:row+number_of_columns-1).';
        end
        [U,Sigma,V] = svd(H,'econ');
        H_rank = U(:,1:rank_value)*Sigma(1:rank_value,1:rank_value)*V(:,1:rank_value)';
        reconstructed = zeros(N,1); counts = zeros(N,1);
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

function [Y,G,S,B,sigma2] = v30_generate_sample(N,K,P,snr_db,RSR_db,d_over_lambda,mu_eg,hbar,seed)
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
