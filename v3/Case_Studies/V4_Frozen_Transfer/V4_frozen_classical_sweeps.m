%% Frozen V4 transfer against classical estimators on fresh N/K sweeps
% This script is independent of the old antenna/user sweep test releases.
% It applies the original frozen N=8,K=3 V4 weights unchanged at all cells.
% No model training or parameter selection is performed here.
%
% Run phases in order: GENERATE_TEST, SCORE_TEST, PLOT_ONLY.
% PLOT_ONLY also writes the corrected per-condition dB table.
% Hard-rank policies are read-only policies selected from prior validation.
% P=10/P=30 outputs remain separate. All methods use pooled NMSE:
% sum_i ||Ghat_i-G_i||_F^2 / sum_i ||G_i||_F^2 per condition.

clear;
clc;
close all;

%% -------------------- PHASE --------------------
workflow_phase = '';
% GENERATE_TEST creates a fresh independent test release only.
% SCORE_TEST scores the frozen methods on that release.
% PLOT_ONLY regenerates plots from the saved test result MAT file.

%% -------------------- FIXED STUDY GRID --------------------
N_antenna = [8 16 24 32];
K_antenna = 3;
N_user = 8;
K_by_P = {[1 2 3 4 5], [1 2 3 4 5 6 7]};
P_list = [10 30];
SNR_grid = [-10 0 10 20 30];
RSR_dB = 25;
samples_per_SNR = 500;
maximum_updates = 20000;
convergence_tolerance = 1e-10;
gd_initial_variance = 0.1;
maximum_parallel_workers = 6;
bootstrap_replicates = 2000;
bootstrap_seed = 2026121201;

%% -------------------- PATHS AND RELEASE METADATA --------------------
script_folder = fileparts(mfilename('fullpath'));
repository_folder = fileparts(fileparts(script_folder));
data_root = fullfile(script_folder, 'data', 'V4_FROZEN_CLASSICAL_FRESH_TEST_V1');
results_root = fullfile(script_folder, 'results', 'V4_FROZEN_CLASSICAL_FRESH_TEST_V1');
antenna_policy_file = fullfile(repository_folder, 'Case_Studies', 'Antenna_sweep', 'results', 'ANTENNA_COUNT_CLASSICAL', 'validation_policy', 'antenna_sweep_hard_rank_policy.csv');
user_policy_file = fullfile(repository_folder, 'Case_Studies', 'Users_sweep', 'results', 'USER_COUNT_CONFIRMATORY_V2', 'validation_policy', 'confirmatory_v2_hard_rank_policy.csv');
v4_model_root = fullfile(repository_folder, 'baselines', 'N8_FROZEN_V4_BASELINE', 'models');
data_version = 'V3_V4_FROZEN_CLASSICAL_FRESH_TEST_V1';
study_name = 'V4_FROZEN_CLASSICAL_N_K_TRANSFER';
d_over_lambda = 0.5;
AoA_deg_range = [-90 90];

method_names = {'GS','GD','EM-GS','Hard-rank HS-GS','Frozen V4'};
number_of_methods = numel(method_names);

switch upper(workflow_phase)
    case 'GENERATE_TEST'
        if isfolder(data_root)
            error('Fresh-test data folder already exists. Inspect it; do not overwrite: %s', data_root);
        end
        if isfolder(results_root)
            error('Results folder already exists. Inspect it; do not overwrite: %s', results_root);
        end
        if ~isfile(antenna_policy_file) || ~isfile(user_policy_file)
            error('A prior validation-frozen hard-rank policy is missing. No test-based policy selection is allowed.');
        end
        mkdir(data_root);
        hbar = 6.62607015e-34/(2*pi);
        q_charge = 1.602e-19;
        a0_bohr = 5.292e-11;
        mu_eg = [0; 1785.9*q_charge*a0_bohr; 0];
        fprintf('Generating independent test data only. No model is trained or tuned.\n');
        fprintf('N sweep: N=%s,K=%d; user sweep: N=%d,K(P10)=%s,K(P30)=%s.\n', mat2str(N_antenna), K_antenna, N_user, mat2str(K_by_P{1}), mat2str(K_by_P{2}));
        fprintf('P=%s; SNR=%s dB; RSR=%g dB; samples/SNR=%d.\n', mat2str(P_list), mat2str(SNR_grid), RSR_dB, samples_per_SNR);

        for study_index = 1:2
            for P_index = 1:numel(P_list)
                P = P_list(P_index);
                if study_index == 1
                    N_list = N_antenna;
                    K_list = K_antenna;
                    study_folder = fullfile(data_root, 'Antenna_sweep');
                else
                    N_list = N_user;
                    K_list = K_by_P{P_index};
                    study_folder = fullfile(data_root, 'User_sweep');
                end
                for N = N_list
                    for K = K_list
                        cell_folder = fullfile(study_folder, sprintf('N%d_K%d_P%d_RSR25dB', N, K, P));
                        mkdir(cell_folder);
                        random_seed = 2026121201 + study_index*1000000 + N*10000 + K*100 + P;
                        rng(random_seed, 'twister');
                        number_of_samples = samples_per_SNR*numel(SNR_grid);
                        Y_data = zeros(N, P, number_of_samples, 'single');
                        G_data = complex(zeros(N, K, number_of_samples, 'single'));
                        S_data = complex(zeros(K, P, number_of_samples, 'single'));
                        B_data = complex(zeros(N, P, number_of_samples, 'single'));
                        sigma2_data = zeros(number_of_samples, 1, 'single');
                        SNR_dB_data = zeros(number_of_samples, 1, 'single');

                        for sample_index = 1:number_of_samples
                            G = zeros(N, K);
                            for user_index = 1:K
                                number_of_paths = randi([3 7]);
                                for path_index = 1:number_of_paths
                                    alpha = (randn+1i*randn)/sqrt(2);
                                    theta = deg2rad(AoA_deg_range(1)+diff(AoA_deg_range)*rand);
                                    spatial_phase = 2*pi*d_over_lambda*cos(theta);
                                    polarization = sqrt(1/3)*randn(3, 1);
                                    for antenna_index = 1:N
                                        G(antenna_index,user_index) = G(antenna_index,user_index)+(mu_eg.'*polarization/hbar)*alpha*exp(-1i*(antenna_index-1)*spatial_phase);
                                    end
                                end
                            end
                            S = (randn(K,P)+1i*randn(K,P))/sqrt(2);
                            alpha_reference = sqrt(10/2)*(randn+1i*randn);
                            theta_reference = deg2rad(AoA_deg_range(1)+diff(AoA_deg_range)*rand);
                            reference_phase = 2*pi*d_over_lambda*cos(theta_reference);
                            polarization_reference = sqrt(1/3)*randn(3,1);
                            b = zeros(N,1);
                            for antenna_index = 1:N
                                b(antenna_index) = (mu_eg.'*polarization_reference/hbar)*alpha_reference*exp(-1i*(antenna_index-1)*reference_phase);
                            end
                            B = b*ones(1,P);
                            average_single_user_power = 0;
                            for user_index = 1:K
                                average_single_user_power = average_single_user_power+mean(abs(G(:,user_index)*S(user_index,:)).^2,'all')/K;
                            end
                            B = B*sqrt(10^(RSR_dB/10)*average_single_user_power/max(mean(abs(B).^2,'all'),eps));
                            snr_index = mod(sample_index-1,numel(SNR_grid))+1;
                            current_SNR_dB = SNR_grid(snr_index);
                            sigma2 = mean(abs(G*S).^2,'all')/10^(current_SNR_dB/10);
                            W = sqrt(sigma2/2)*(randn(N,P)+1i*randn(N,P));
                            Y = abs(G*S+B+W);
                            Y_data(:,:,sample_index) = single(Y);
                            G_data(:,:,sample_index) = single(G);
                            S_data(:,:,sample_index) = single(S);
                            B_data(:,:,sample_index) = single(B);
                            sigma2_data(sample_index) = single(sigma2);
                            SNR_dB_data(sample_index) = single(current_SNR_dB);
                        end

                        measurement_model = 'Y = abs(G*S + B + W)';
                        RSR_definition = 'reference power divided by average single-user received signal power';
                        AoA_distribution = 'Uniform incident angle';
                        spatial_phase_convention = 'phi = 2*pi*(d/lambda)*cos(theta), theta in radians';
                        split_name = 'test';
                        P_list_saved = P_list;
                        SNR_dB = SNR_grid;
                        output_file = fullfile(cell_folder, 'test.mat');
                        save(output_file, 'Y_data','G_data','S_data','B_data','sigma2_data','SNR_dB_data','measurement_model','RSR_definition','AoA_distribution','AoA_deg_range','spatial_phase_convention','data_version','study_name','random_seed','samples_per_SNR','d_over_lambda','N','K','P','P_list_saved','SNR_dB','RSR_dB','number_of_samples','split_name','-v7.3');
                        fprintf('Saved fresh test: %s\n', output_file);
                    end
                end
            end
        end
        fprintf('Fresh test generation complete. No validation or prior test data were loaded.\n');
        fprintf('After reviewing the generated release metadata, set workflow_phase to SCORE_TEST.\n');

    case 'SCORE_TEST'
        if ~isfolder(data_root) || isfolder(results_root)
            error('Expected fresh data and a new results path; refusing missing/overwritten release.');
        end
        if ~isfile(antenna_policy_file) || ~isfile(user_policy_file)
            error('Validation-frozen hard-rank policy files are required.');
        end
        if ~isfolder(v4_model_root)
            error('Frozen V4 model folder is missing.');
        end
        antenna_policy = readtable(antenna_policy_file);
        user_policy = readtable(user_policy_file);
        required_policy_columns = {'N','K','P','SNR_dB','FrozenHardRank'};
        if ~all(ismember(required_policy_columns, antenna_policy.Properties.VariableNames)) || ~all(ismember(required_policy_columns, user_policy.Properties.VariableNames))
            error('A validation policy is missing required N/K/P/SNR/rank columns.');
        end
        mkdir(results_root);
        result_rows = zeros(0, 11);
        score_cells = cell(0,1);
        cell_metadata = zeros(0,4);
        cell_counter = 0;

        for study_index = 1:2
            for P_index = 1:numel(P_list)
                P = P_list(P_index);
                if study_index == 1
                    N_list = N_antenna;
                    K_list = K_antenna;
                    study_folder = fullfile(data_root, 'Antenna_sweep');
                    policy_table = antenna_policy;
                    study_label = 'Antenna';
                else
                    N_list = N_user;
                    K_list = K_by_P{P_index};
                    study_folder = fullfile(data_root, 'User_sweep');
                    policy_table = user_policy;
                    study_label = 'User';
                end
                model_file = fullfile(v4_model_root, sprintf('P%d',P), sprintf('selected_model_P%d.mat',P));
                if ~isfile(model_file)
                    error('Frozen V4 model missing: %s', model_file);
                end
                model = load(model_file, 'best_parameters_numeric','N','K','P','RSR_dB','p_H','number_of_stages','parameters_per_stage','number_of_gate_features','number_of_gate_outputs','initialization_method');
                if model.N ~= 8 || model.K ~= 3 || model.P ~= P || model.RSR_dB ~= RSR_dB || model.p_H ~= 3 || model.number_of_stages ~= 16 || model.parameters_per_stage ~= 15 || model.number_of_gate_features ~= 3 || model.number_of_gate_outputs ~= 3 || ~isequal(size(model.best_parameters_numeric),[15 16]) || any(~isfinite(model.best_parameters_numeric),'all') || ~strcmp(model.initialization_method,'GS')
                    error('Frozen V4 checkpoint metadata mismatch for P=%d.',P);
                end
                for N = N_list
                    for K = K_list
                        data_file = fullfile(study_folder,sprintf('N%d_K%d_P%d_RSR25dB',N,K,P),'test.mat');
                        if ~isfile(data_file)
                            error('Fresh test file missing: %s',data_file);
                        end
                        data = load(data_file,'Y_data','G_data','S_data','B_data','sigma2_data','SNR_dB_data','N','K','P','RSR_dB','SNR_dB','data_version','split_name','number_of_samples','samples_per_SNR');
                        if data.N ~= N || data.K ~= K || data.P ~= P || data.RSR_dB ~= RSR_dB || ~isequal(double(data.SNR_dB),SNR_grid) || ~strcmp(data.data_version,data_version) || ~strcmp(data.split_name,'test') || data.number_of_samples ~= samples_per_SNR*numel(SNR_grid)
                            error('Fresh-test metadata mismatch at N=%d,K=%d,P=%d.',N,K,P);
                        end
                        sample_error = zeros(data.number_of_samples,number_of_methods);
                        sample_energy = zeros(data.number_of_samples,1);
                        sample_converged = false(data.number_of_samples,4);
                        sample_updates = zeros(data.number_of_samples,4);
                        fprintf('Scoring %s sweep N=%d,K=%d,P=%d on %d fresh samples.\n',study_label,N,K,P,data.number_of_samples);
                        parfor (sample_index = 1:data.number_of_samples,maximum_parallel_workers)
                            Y = double(data.Y_data(:,:,sample_index));
                            G_true = double(data.G_data(:,:,sample_index));
                            S = double(data.S_data(:,:,sample_index));
                            B = double(data.B_data(:,:,sample_index));
                            sigma2 = double(data.sigma2_data(sample_index));
                            current_snr = double(data.SNR_dB_data(sample_index));
                            policy_row = find(policy_table.N == N & policy_table.K == K & policy_table.P == P & policy_table.SNR_dB == current_snr);
                            if numel(policy_row) ~= 1
                                error('Expected one validation-frozen rank for N=%d,K=%d,P=%d,SNR=%g.',N,K,P,current_snr);
                            end
                            selected_rank = policy_table.FrozenHardRank(policy_row);
                            p_H = floor((N-1)/2);
                            seed = 2026121201+study_index*10000000+N*100000+K*1000+P*10+sample_index;
                            [G_GS,updates_GS,converged_GS] = local_classical('GS',Y,S,B,sigma2,maximum_updates,convergence_tolerance,gd_initial_variance,seed);
                            [G_GD,updates_GD,converged_GD] = local_classical('GD',Y,S,B,sigma2,maximum_updates,convergence_tolerance,gd_initial_variance,seed);
                            [G_EMGS,updates_EMGS,converged_EMGS] = local_classical('EM-GS',Y,S,B,sigma2,maximum_updates,convergence_tolerance,gd_initial_variance,seed);
                            [G_HSGS,converged_HSGS,updates_HSGS] = local_hard_hsgs(Y,S,B,sigma2,p_H,selected_rank,maximum_updates,convergence_tolerance);
                            G_V4 = local_v4_forward(model.best_parameters_numeric,Y,S,B,sigma2,G_GS,N,K,p_H,model.number_of_stages);
                            estimates = {G_GS,G_GD,G_EMGS,G_HSGS,G_V4};
                            for method_index = 1:number_of_methods
                                sample_error(sample_index,method_index) = sum(abs(estimates{method_index}-G_true).^2,'all');
                            end
                            sample_energy(sample_index) = sum(abs(G_true).^2,'all');
                            sample_converged(sample_index,:) = [converged_GS converged_GD converged_EMGS converged_HSGS];
                            sample_updates(sample_index,:) = [updates_GS updates_GD updates_EMGS updates_HSGS];
                        end
                        if any(~isfinite(sample_error),'all') || any(sample_energy <= 0)
                            error('Nonfinite method score or invalid channel energy at N=%d,K=%d,P=%d.',N,K,P);
                        end
                        cell_counter = cell_counter+1;
                        score_cells{cell_counter,1} = struct('study',study_label,'N',N,'K',K,'P',P,'sample_error',sample_error,'sample_energy',sample_energy,'sample_snr',double(data.SNR_dB_data(:)),'sample_converged',sample_converged,'sample_updates',sample_updates);
                        cell_metadata(cell_counter,:) = [study_index N K P];
                        for snr_index = 1:numel(SNR_grid)
                            selected = double(data.SNR_dB_data(:)) == SNR_grid(snr_index);
                            if nnz(selected) ~= samples_per_SNR
                                error('Unexpected test sample count at N=%d,K=%d,P=%d,SNR=%g.',N,K,P,SNR_grid(snr_index));
                            end
                            energy_sum = sum(sample_energy(selected));
                            nmse = sum(sample_error(selected,:),1)/max(energy_sum,realmin);
                            ranks = policy_table.FrozenHardRank(policy_table.N == N & policy_table.K == K & policy_table.P == P & policy_table.SNR_dB == SNR_grid(snr_index));
                            result_rows(end+1,:) = [study_index N K P SNR_grid(snr_index) ranks(:).' nmse];
                        end
                    end
                end
            end
        end
        result_table = array2table(result_rows,'VariableNames',{'StudyIndex','N','K','P','SNR_dB','FrozenHardRank','GS_NMSE','GD_NMSE','EMGS_NMSE','HardRank_HSGS_NMSE','FrozenV4_NMSE'});
        raw_columns = {'GS_NMSE','GD_NMSE','EMGS_NMSE','HardRank_HSGS_NMSE','FrozenV4_NMSE'};
        db_columns = {'GS_NMSE_dB','GD_NMSE_dB','EM_GS_NMSE_dB','Hard_rankHS_GS_NMSE_dB','FrozenV4_NMSE_dB'};
        for method_index = 1:number_of_methods
            linear_values = result_table.(raw_columns{method_index});
            result_table.(db_columns{method_index}) = 10*log10(max(linear_values,realmin));
        end
        writetable(result_table,fullfile(results_root,'fresh_test_nmse_by_condition.csv'));
        save(fullfile(results_root,'fresh_test_scores.mat'),'result_table','score_cells','cell_metadata','method_names','N_antenna','K_antenna','N_user','K_by_P','P_list','SNR_grid','RSR_dB','maximum_updates','convergence_tolerance','gd_initial_variance','antenna_policy_file','user_policy_file','data_version','bootstrap_replicates','bootstrap_seed','-v7.3');
        fprintf('Scored all methods on paired fresh test observations. No model or policy was tuned.\n');
        fprintf('Saved results under: %s\n',results_root);
        fprintf('Set workflow_phase to PLOT_ONLY to create per-SNR and SNR-aggregated figures.\n');

    case 'PLOT_ONLY'
        score_file = fullfile(results_root,'fresh_test_scores.mat');
        if ~isfile(score_file)
            error('Fresh test score MAT is missing: %s',score_file);
        end
        loaded = load(score_file,'result_table','method_names','SNR_grid','P_list');
        result_table = loaded.result_table;
        SNR_grid = loaded.SNR_grid;
        P_list = loaded.P_list;
        plot_method_names = {'GS','GD','EM-GS','Hard-rank HS-GS','PI-DU-RACE (DL)'};
        raw_columns = {'GS_NMSE','GD_NMSE','EMGS_NMSE','HardRank_HSGS_NMSE','FrozenV4_NMSE'};
        db_columns = {'GS_NMSE_dB','GD_NMSE_dB','EM_GS_NMSE_dB','Hard_rankHS_GS_NMSE_dB','FrozenV4_NMSE_dB'};
        for method_index = 1:numel(raw_columns)
            if ~ismember(raw_columns{method_index},result_table.Properties.VariableNames)
                error('Required linear NMSE column is missing: %s',raw_columns{method_index});
            end
            linear_values = result_table.(raw_columns{method_index});
            if any(~isfinite(linear_values)) || any(linear_values < 0)
                error('Invalid linear NMSE values in column %s.',raw_columns{method_index});
            end
            result_table.(db_columns{method_index}) = 10*log10(max(linear_values,realmin));
        end
        corrected_table_file = fullfile(results_root,'fresh_test_nmse_by_condition_plot_only_corrected.csv');
        plot_root = fullfile(results_root,'figures_combined');
        if isfile(corrected_table_file) || isfolder(plot_root)
            error('Combined PLOT_ONLY output already exists. Inspect it before rerunning. Table: %s | Figures: %s',corrected_table_file,plot_root);
        end
        writetable(result_table,corrected_table_file);
        mkdir(plot_root);
        colors = [0.00 0.35 0.75;0.85 0.25 0.05;0.95 0.60 0.00;0.55 0.10 0.80;0.10 0.60 0.35];
        markers = {'o','s','d','^','v'};
        study_names = {'Antenna','User'};
        for study_index = 1:2
            study_folder = fullfile(plot_root,study_names{study_index});
            mkdir(study_folder);
            for P = P_list
                subset = result_table.StudyIndex == study_index & result_table.P == P;
                if study_index == 1
                    x_values = unique(result_table.N(subset));
                    x_label = 'Antenna count N';
                    aggregate_name = 'antenna';
                else
                    x_values = unique(result_table.K(subset));
                    x_label = 'User count K';
                    aggregate_name = 'user';
                end
                %% One 1-by-SNR-grid panel figure; each panel is an NMSE-vs-dimension curve.
                figure('Color','w','Name',sprintf('%s sweep per-SNR P=%d',study_names{study_index},P));
                tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
                for snr_index = 1:numel(SNR_grid)
                    nexttile;
                    hold on;
                    rows = subset & result_table.SNR_dB == SNR_grid(snr_index);
                    for method_index = 1:number_of_methods
                        curve = nan(size(x_values));
                        raw_values = result_table.(raw_columns{method_index});
                        for x_index = 1:numel(x_values)
                            if study_index == 1
                                row = rows & result_table.N == x_values(x_index);
                            else
                                row = rows & result_table.K == x_values(x_index);
                            end
                            if nnz(row) == 1
                                curve(x_index) = 10*log10(max(raw_values(row),realmin));
                            end
                        end
                        plot(x_values,curve,'-o','Color',colors(method_index,:),'Marker',markers{method_index},'LineWidth',1.3,'DisplayName',plot_method_names{method_index});
                    end
                    grid on;
                    title(sprintf('SNR=%g dB',SNR_grid(snr_index)));
                    xlabel(x_label);
                    ylabel('Aggregate NMSE (dB)');
                end
                legend(plot_method_names,'Location','southoutside','Orientation','horizontal');
                sgtitle(sprintf('%s sweep, P=%d',study_names{study_index},P));
                exportgraphics(gcf,fullfile(study_folder,sprintf('%s_sweep_per_SNR_P%d.png',aggregate_name,P)),'Resolution',220);

                %% Equal-weight mean of per-SNR pooled linear NMSE, then dB conversion.
                aggregate_table = table(x_values(:),'VariableNames',{'Dimension'});
                aggregate_table.P = repmat(P,numel(x_values),1);
                for method_index = 1:number_of_methods
                    aggregate_linear = nan(numel(x_values),1);
                    raw_values = result_table.(raw_columns{method_index});
                    for x_index = 1:numel(x_values)
                        values = nan(numel(SNR_grid),1);
                        for snr_index = 1:numel(SNR_grid)
                            rows = subset & result_table.SNR_dB == SNR_grid(snr_index);
                            if study_index == 1
                                rows = rows & result_table.N == x_values(x_index);
                            else
                                rows = rows & result_table.K == x_values(x_index);
                            end
                            if nnz(rows) == 1
                                values(snr_index) = raw_values(rows);
                            end
                        end
                        if all(isfinite(values))
                            aggregate_linear(x_index) = mean(values);
                        end
                    end
                    method_prefix = matlab.lang.makeValidName(plot_method_names{method_index});
                    aggregate_table.([method_prefix '_AggregateLinearNMSEOverSNR']) = aggregate_linear;
                    aggregate_table.([method_prefix '_AggregateNMSEOverSNR_dB']) = 10*log10(max(aggregate_linear,realmin));
                end
                writetable(aggregate_table,fullfile(study_folder,sprintf('%s_sweep_SNR_aggregated_P%d.csv',aggregate_name,P)));
                figure('Color','w','Name',sprintf('%s sweep SNR aggregated P=%d',study_names{study_index},P));
                hold on;
                for method_index = 1:number_of_methods
                    column_name = [matlab.lang.makeValidName(plot_method_names{method_index}) '_AggregateNMSEOverSNR_dB'];
                    plot(x_values,aggregate_table.(column_name),'-o','Color',colors(method_index,:),'Marker',markers{method_index},'LineWidth',1.5,'DisplayName',plot_method_names{method_index});
                end
                grid on;
                xlabel(x_label);
                ylabel('Aggregate NMSE over SNR (dB)');
                title(sprintf('%s sweep, P=%d',study_names{study_index},P));
                legend('Location','best');
                exportgraphics(gcf,fullfile(study_folder,sprintf('%s_aggregate_over_SNR_P%d.png',aggregate_name,P)),'Resolution',220);
            end
        end
        fprintf('Corrected per-condition CSV saved: %s\n',corrected_table_file);
        fprintf('Per-SNR figures and SNR-aggregate figures/CSVs saved in: %s\n',plot_root);
        fprintf('PLOT_ONLY used saved scores only; no estimators were rerun or test data rescored.\n');

    otherwise
        error('workflow_phase must be GENERATE_TEST, SCORE_TEST, or PLOT_ONLY.');
end

%% -------------------- LOCAL ESTIMATORS --------------------
function [G,iteration_count,converged] = local_classical(method,Y,S,B,sigma2,maximum_updates,tolerance,gd_initial_variance,random_seed)
method = upper(string(method));
if method == "GS" || method == "EM-GS"
    G = local_spectral_initialize(Y,S,B);
    right_inverse = S'/(S*S');
elseif method == "GD"
    rng(mod(double(random_seed)+720000,2^32-1),'twister');
    G = sqrt(gd_initial_variance/2)*(randn(size(Y,1),size(S,1))+1i*randn(size(Y,1),size(S,1)));
    phase_reference = exp(-1i*angle(B));
    step_size = 1/max(real(eig(S*S')));
else
    error('Unknown classical method.');
end
converged = false;
iteration_count = maximum_updates;
for iteration = 1:maximum_updates
    previous = G;
    if method == "GS"
        G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    elseif method == "GD"
        G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    else
        field = G*S+B;
        kappa = 2/max(sigma2,eps)*Y.*abs(field);
        ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
        G = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    end
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        iteration_count = iteration;
        converged = true;
        break;
    end
end
end

function [G,converged,iteration_count] = local_hard_hsgs(Y,S,B,sigma2,p_H,rank_value,maximum_updates,tolerance)
N = size(Y,1);
K = size(S,1);
right_inverse = S'/(S*S');
G = local_spectral_initialize(Y,S,B);
converged = false;
iteration_count = maximum_updates;
for iteration = 1:maximum_updates
    previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    projected = zeros(N,K);
    rows = N-p_H;
    columns = p_H+1;
    for user_index = 1:K
        channel = emgs(:,user_index);
        hankel = zeros(rows,columns);
        for row = 1:rows
            hankel(row,:) = channel(row:row+columns-1).';
        end
        [U,Sigma,V] = svd(hankel,'econ');
        use_rank = min([rank_value,size(Sigma,1),size(Sigma,2)]);
        hankel = U(:,1:use_rank)*Sigma(1:use_rank,1:use_rank)*V(:,1:use_rank)';
        reconstructed = zeros(N,1);
        counts = zeros(N,1);
        for row = 1:rows
            for column = 1:columns
                antenna = row+column-1;
                reconstructed(antenna) = reconstructed(antenna)+hankel(row,column);
                counts(antenna) = counts(antenna)+1;
            end
        end
        projected(:,user_index) = reconstructed./counts;
    end
    G = projected;
    if norm(G-previous,'fro')^2/max(norm(G,'fro')^2,eps) < tolerance
        converged = true;
        iteration_count = iteration;
        break;
    end
end
end

function G = local_v4_forward(parameters,Y,S,B,sigma2,G,N,K,p_H,number_of_stages)
right_inverse = S'/(S*S');
noise_feature = tanh(log(max(sigma2/max(mean(abs(B).^2,'all'),eps),eps))/10);
noise_scale = sqrt(sigma2)*norm(right_inverse,'fro')*sqrt(max(N-p_H,p_H+1)/K);
map = local_hankel_map(N,p_H);
counts = sum(map,1).';
for stage = 1:number_of_stages
    previous = G;
    field = G*S+B;
    kappa = 2/max(sigma2,eps)*Y.*abs(field);
    ratio = besseli(1,kappa,1)./max(besseli(0,kappa,1),realmin);
    emgs = (Y.*exp(1i*angle(field)).*ratio-B)*right_inverse;
    hankel = zeros(N,K);
    for user_index = 1:K
        H = reshape(map*emgs(:,user_index),N-p_H,p_H+1);
        gram = H'*H;
        [vectors,values] = eig(gram);
        singular = sqrt(max(real(sum(values,2)),0));
        safe = max(singular,1e-12);
        relative = tanh(log(safe/max(mean(safe),1e-12)));
        logits = parameters(1,stage)+parameters(2,stage)*noise_feature+parameters(3,stage)*relative;
        thresholds = noise_scale*(2./(1+exp(-logits)));
        factors = max(1-thresholds./safe,0);
        H_shrunk = H*(vectors.*reshape(factors,1,[]))*vectors';
        hankel(:,user_index) = (map'*H_shrunk(:))./counts;
    end
    emgs_change = tanh(log(1+sqrt(sum(abs(emgs-previous).^2,'all')/max(sum(abs(previous).^2,'all'),eps)))/3);
    hankel_change = tanh(log(1+sqrt(sum(abs(hankel-emgs).^2,'all')/max(sum(abs(emgs).^2,'all'),eps)))/3);
    coefficients = reshape(parameters(4:15,stage),4,3);
    gate_logits = coefficients(1,:).'*noise_feature+coefficients(2,:).'*emgs_change+coefficients(3,:).'*hankel_change+coefficients(4,:).';
    weights = exp(gate_logits-max(gate_logits));
    weights = weights/sum(weights);
    G = weights(1)*previous+weights(2)*emgs+weights(3)*hankel;
end
end

function map = local_hankel_map(N,p_H)
rows = N-p_H;
columns = p_H+1;
map = zeros(rows*columns,N);
for row = 1:rows
    for column = 1:columns
        map(row+(column-1)*rows,row+column-1) = 1;
    end
end
end

function G = local_spectral_initialize(Y,S,B)
N = size(Y,1);
K = size(S,1);
G = zeros(N,K);
for antenna = 1:N
    y = Y(antenna,:).';
    b = B(antenna,:).';
    augmented = [S.',b];
    matrix = augmented'*diag(y)*augmented;
    [vectors,values] = eig(matrix);
    [~,index] = max(real(diag(values)));
    vector = vectors(:,index);
    vector = vector/max(norm(vector),eps);
    scale = (abs(augmented*vector).'*y)/max(norm(augmented*vector)^2,eps);
    G(antenna,:) = (scale*exp(-1i*angle(vector(K+1)))*vector(1:K)).';
end
end
