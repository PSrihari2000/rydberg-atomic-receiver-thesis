%% HS_GS_V23_TRUE_HANKEL_RANK_AUDIT.m
% Audit whether the simulator's true channels are represented well by the
% Hankel ranks used by HS-GS. This is a model diagnostic, not an estimator
% comparison: it uses noiseless simulated channels only and does not select
% settings from any final test set.
%
% For g[n] = sum_l alpha_l*z_l^n, its Hankel matrix has rank at most the
% number of distinct paths, subject to the Hankel matrix dimensions. This
% script measures the actual singular spectra and the channel distortion
% after the same four Cadzow rank-projection sweeps used by HS-GS.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   HS_GS_V23_TRUE_HANKEL_RANK_AUDIT
%
% The audit uses 1,000 independent generated channels per user and N. It
% runs serially and should be much lighter than the paired SNR experiments.

clear; clc;
script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
if ~exist(result_folder,'dir'), mkdir(result_folder); end

result_file = fullfile(result_folder,'HS_GS_V23_true_hankel_rank_audit.mat');
table_file = fullfile(result_folder,'HS_GS_V23_true_hankel_rank_audit.csv');
figure_file = fullfile(result_folder,'HS_GS_V23_true_hankel_rank_audit.png');
if isfile(result_file)
    error('V23 audit results already exist; inspect them before starting again: %s',result_file);
end

N_list = [8 16];
p_H_list = [3 7];
K = 3;
number_of_samples = 1000;
number_of_cadzow_sweeps = 4;
seed_base = 2800000000;
d_over_lambda = 0.5;
mu_eg = [0; 1785.9 * 1.602e-19 * 5.292e-11; 0];
hbar = 6.62607015e-34 / (2*pi);

max_rank_list = zeros(size(N_list));
path_count = zeros(length(N_list),number_of_samples,K);
numerical_rank = zeros(length(N_list),number_of_samples,K);
retained_energy = cell(size(N_list));
projection_squared_error = cell(size(N_list));
channel_squared_energy = cell(size(N_list));
rank_summary = cell(size(N_list));

fprintf('V23 true-channel Hankel audit: %d samples/user, N = 8 and 16.\n',number_of_samples);
for n_index = 1:length(N_list)
    N = N_list(n_index);
    p_H = p_H_list(n_index);
    H_rows = N - p_H;
    H_columns = p_H + 1;
    maximum_hankel_rank = min(H_rows,H_columns);
    max_rank_list(n_index) = maximum_hankel_rank;
    retained_energy{n_index} = zeros(number_of_samples,K,maximum_hankel_rank);
    projection_squared_error{n_index} = zeros(number_of_samples,K,maximum_hankel_rank);
    channel_squared_energy{n_index} = zeros(number_of_samples,K);

    for sample_index = 1:number_of_samples
        % The same seed across N uses the same generated path parameters;
        % the longer array simply observes more antenna elements.
        [G,sample_path_count] = v23_generate_channel( ...
            N,K,d_over_lambda,mu_eg,hbar,seed_base+sample_index);
        for user_index = 1:K
            g = G(:,user_index);
            H_true = v23_hankel_embed(g,H_rows,H_columns);
            singular_values = svd(H_true,'econ');
            path_count(n_index,sample_index,user_index) = sample_path_count(user_index);
            numerical_rank(n_index,sample_index,user_index) = ...
                sum(singular_values > 1e-10*max(singular_values));
            channel_squared_energy{n_index}(sample_index,user_index) = norm(g)^2;

            for rank_value = 1:maximum_hankel_rank
                retained_energy{n_index}(sample_index,user_index,rank_value) = ...
                    sum(singular_values(1:rank_value).^2)/sum(singular_values.^2);
                g_projected = g;
                for sweep = 1:number_of_cadzow_sweeps
                    H = v23_hankel_embed(g_projected,H_rows,H_columns);
                    [U_H,Sigma_H,V_H] = svd(H,'econ');
                    H_rank = U_H(:,1:rank_value)*Sigma_H(1:rank_value,1:rank_value)* ...
                        V_H(:,1:rank_value)';
                    g_projected = v23_antidiagonal_average(H_rank,N);
                end
                projection_squared_error{n_index}(sample_index,user_index,rank_value) = ...
                    norm(g_projected-g)^2;
            end
        end
        if mod(sample_index,200)==0
            fprintf('N=%d: %d/%d samples complete.\n',N,sample_index,number_of_samples);
        end
    end

    rank_rows = zeros(maximum_hankel_rank,10);
    sampled_paths = path_count(n_index,:,:);
    sampled_true_rank = numerical_rank(n_index,:,:);
    for rank_value = 1:maximum_hankel_rank
        signal_energy = sum(channel_squared_energy{n_index},'all');
        projection_error = sum(projection_squared_error{n_index}(:,:,rank_value),'all');
        rank_rows(rank_value,:) = [N,p_H,H_rows,H_columns,rank_value, ...
            mean(sampled_paths,'all'),mean(sampled_true_rank,'all'), ...
            mean(sampled_true_rank<=rank_value,'all'), ...
            mean(retained_energy{n_index}(:,:,rank_value),'all'), ...
            10*log10(max(projection_error/signal_energy,realmin))];
    end
    rank_summary{n_index} = array2table(rank_rows,'VariableNames', ...
        {'N','p_H','H_rows','H_columns','Assumed_rank', ...
         'Mean_paths_per_user','Mean_true_Hankel_numerical_rank', ...
         'Fraction_true_rank_at_most_assumed','Mean_singular_energy_retained', ...
         'Projection_NMSE_dB'});
    fprintf('\nTrue-channel projection summary for N=%d (H is %d-by-%d):\n', ...
        N,H_rows,H_columns);
    disp(rank_summary{n_index});
end

combined_table = vertcat(rank_summary{:});
writetable(combined_table,table_file);
figure('Color','w'); hold on; grid on; box on;
plot_styles = {'o-','s-'};
for n_index = 1:length(N_list)
    plot(rank_summary{n_index}.Assumed_rank, ...
        rank_summary{n_index}.Projection_NMSE_dB,plot_styles{n_index}, ...
        'LineWidth',1.6,'MarkerSize',7,'DisplayName',sprintf('N = %d',N_list(n_index)));
end
xlabel('Assumed Hankel rank');
ylabel('Noiseless channel projection NMSE (dB)');
title('How much the Hankel projection distorts the true simulated channels');
legend('Location','best');
saveas(gcf,figure_file);

save(result_file,'N_list','p_H_list','K','number_of_samples', ...
    'number_of_cadzow_sweeps','seed_base','max_rank_list','path_count', ...
    'numerical_rank','retained_energy','projection_squared_error', ...
    'channel_squared_energy','rank_summary','combined_table','-v7.3');
fprintf('V23 audit table saved to %s\n',table_file);
fprintf('V23 audit figure saved to %s\n',figure_file);
fprintf('V23 audit data saved to %s\n',result_file);

function [G,path_counts] = v23_generate_channel(N,K,d_over_lambda,mu_eg,hbar,seed)
rng(double(seed),'twister');
G = zeros(N,K);
path_counts = zeros(1,K);
for user_index = 1:K
    number_of_paths = randi([3 7]);
    path_counts(user_index) = number_of_paths;
    for path_index = 1:number_of_paths
        alpha = (randn+1j*randn)/sqrt(2);
        theta = 2*pi*rand;
        spatial_phase = 2*pi*d_over_lambda*cos(theta);
        polarization = sqrt(1/3)*randn(3,1);
        path_gain = (mu_eg.'*polarization/hbar)*alpha;
        for antenna = 1:N
            G(antenna,user_index) = G(antenna,user_index) + ...
                path_gain*exp(-1j*(antenna-1)*spatial_phase);
        end
    end
end
end

function H = v23_hankel_embed(g,H_rows,H_columns)
H = zeros(H_rows,H_columns);
for row = 1:H_rows
    H(row,:) = g(row:row+H_columns-1).';
end
end

function g = v23_antidiagonal_average(H,N)
g_sum = zeros(N,1);
count = zeros(N,1);
for row = 1:size(H,1)
    for column = 1:size(H,2)
        index = row+column-1;
        g_sum(index) = g_sum(index)+H(row,column);
        count(index) = count(index)+1;
    end
end
g = g_sum./count;
end
