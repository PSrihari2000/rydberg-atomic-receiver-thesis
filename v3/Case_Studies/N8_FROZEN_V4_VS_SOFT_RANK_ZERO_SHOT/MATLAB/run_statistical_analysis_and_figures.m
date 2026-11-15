%% Paired uncertainty, SNR averaging, figures, and summary for the reanalysis.
% Consumes only outputs from run_test_scoring.m; does not load source test data.
clear;
clc;

script_folder = fileparts(mfilename('fullpath'));
experiment_root = fileparts(script_folder);
test_root = fullfile(experiment_root,'Test_results');
runtime_root = fullfile(experiment_root,'Runtime_results');
figure_root = fullfile(experiment_root,'Figures');
stats_root = fullfile(experiment_root,'Statistical_analysis');
log_root = fullfile(experiment_root,'Experiment_logs');
required_files = {fullfile(test_root,'paired_test_sample_errors.mat'), ...
    fullfile(test_root,'fresh_test_nmse_by_condition_with_soft_rank.csv'), ...
    fullfile(runtime_root,'runtime_by_configuration.csv')};
for file_index = 1:numel(required_files)
    if ~isfile(required_files{file_index})
        error('Missing prerequisite output: %s',required_files{file_index});
    end
end
stats_entries = dir(fullfile(stats_root,'*'));
stats_entries = stats_entries(~ismember({stats_entries.name},{'.','..'}));
figure_entries = dir(fullfile(figure_root,'*'));
figure_entries = figure_entries(~ismember({figure_entries.name},{'.','..'}));
if ~isempty(stats_entries) || ~isempty(figure_entries)
    error('Statistical or figure output folder is not empty; refusing to overwrite.');
end

loaded = load(required_files{1},'test_cells','policy_names','SNR_grid', ...
    'P_list','bootstrap_replicates','bootstrap_seed');
test_cells = loaded.test_cells;
policy_names = loaded.policy_names;
SNR_grid = loaded.SNR_grid;
P_list = loaded.P_list;
bootstrap_replicates = loaded.bootstrap_replicates;
bootstrap_seed = loaded.bootstrap_seed;
condition_table = readtable(required_files{2});
runtime_table = readtable(required_files{3});
if bootstrap_replicates < 2000
    error('At least 2,000 paired bootstrap replicates are required.');
end

% Generate paired percentile intervals from the sample-level squared errors
% and shared true-channel energies, never from aggregate dB values.
rng(bootstrap_seed,'twister');
ci_rows = cell(20*numel(SNR_grid)*numel(policy_names),1);
gain_rows = cell(size(ci_rows));
ci_row = 0;
gain_row = 0;
for cell_index = 1:numel(test_cells)
    c = test_cells{cell_index};
    for snr_index = 1:numel(SNR_grid)
        selected = find(c.sample_snr == SNR_grid(snr_index));
        if numel(selected) ~= 500
            error('Expected 500 samples in condition %s N=%d K=%d P=%d SNR=%g.', ...
                c.study,c.N,c.K,c.P,SNR_grid(snr_index));
        end
        energy = c.sample_energy(selected);
        err = c.sample_error(selected,:);
        for policy_index = 1:numel(policy_names)
            gains = zeros(bootstrap_replicates,1);
            for replicate = 1:bootstrap_replicates
                indices = randi(numel(selected),numel(selected),1);
                denominator = sum(energy(indices));
                nmse = sum(err(indices,[1 policy_index+1]),1)/denominator;
                gains(replicate) = 10*log10(max(nmse(2),realmin))- ...
                    10*log10(max(nmse(1),realmin));
            end
            point = 10*log10(sum(err(:,policy_index+1))/sum(energy))- ...
                10*log10(sum(err(:,1))/sum(energy));
            limits = prctile(gains,[2.5 97.5]);
            excludes_zero = limits(1) > 0 || limits(2) < 0;
            ci_row = ci_row+1;
            ci_rows{ci_row} = {string(c.study),c.N,c.K,c.P,SNR_grid(snr_index), ...
                string(policy_names{policy_index}),point,limits(1),limits(2), ...
                excludes_zero,bootstrap_replicates,numel(selected)};
            gain_row = gain_row+1;
            matching = condition_table(strcmpi(string(condition_table.Study),string(c.study)) & ...
                condition_table.N == c.N & condition_table.K == c.K & ...
                condition_table.P == c.P & condition_table.SNR_dB == SNR_grid(snr_index) & ...
                strcmp(string(condition_table.SoftThresholdPolicy),string(policy_names{policy_index})),:);
            if height(matching) ~= 1
                error('Could not uniquely match condition NMSE output for %s N=%d K=%d P=%d.', ...
                    c.study,c.N,c.K,c.P);
            end
            gain_rows{gain_row} = {string(c.study),c.N,c.K,c.P,SNR_grid(snr_index), ...
                string(policy_names{policy_index}),matching.V4_NMSE_dB(1), ...
                matching.SoftRank_NMSE_dB(1),point,limits(1),limits(2),excludes_zero};
        end
    end
end
ci_table = cell2table(vertcat(ci_rows{:}),'VariableNames', ...
    {'Study','N','K','P','SNR_dB','SoftThresholdPolicy','Gain_dB', ...
    'CI95_Lower_dB','CI95_Upper_dB','CI_Excludes_Zero','BootstrapReplicates','Samples'});
gain_table = cell2table(vertcat(gain_rows{:}),'VariableNames', ...
    {'Study','N','K','P','SNR_dB','SoftThresholdPolicy','FrozenV4_NMSE_dB', ...
    'SoftRank_NMSE_dB','V4_Improvement_dB','CI95_Lower_dB','CI95_Upper_dB', ...
    'CI_Excludes_Zero'});
writetable(ci_table,fullfile(stats_root,'v4_vs_soft_rank_bootstrap_ci.csv'));
writetable(gain_table,fullfile(stats_root,'v4_vs_soft_rank_gain_by_condition.csv'));

% SNR average: equal-weight arithmetic mean of the five pooled linear NMSEs,
% followed by conversion to dB. Bootstrap by resampling within each SNR stratum.
avg_rows = cell(20*(1+numel(policy_names)),1);
avg_ci_rows = cell(20*numel(policy_names),1);
avg_row = 0;
avg_ci_row = 0;
for cell_index = 1:numel(test_cells)
    c = test_cells{cell_index};
    stratum_error = zeros(numel(SNR_grid),size(c.sample_error,2));
    stratum_energy = zeros(numel(SNR_grid),1);
    stratum_indices = cell(numel(SNR_grid),1);
    for snr_index = 1:numel(SNR_grid)
        stratum_indices{snr_index} = find(c.sample_snr == SNR_grid(snr_index));
        idx = stratum_indices{snr_index};
        stratum_error(snr_index,:) = sum(c.sample_error(idx,:),1);
        stratum_energy(snr_index) = sum(c.sample_energy(idx));
    end
    point_nmse = mean(stratum_error./stratum_energy,1);
    avg_row = avg_row+1;
    avg_rows{avg_row} = {string(c.study),c.N,c.K,c.P,string('Frozen PI-DU-RACE V4'), ...
        point_nmse(1),10*log10(max(point_nmse(1),realmin))};
    for policy_index = 1:numel(policy_names)
        soft_col = policy_index+1;
        avg_row = avg_row+1;
        avg_rows{avg_row} = {string(c.study),c.N,c.K,c.P, ...
            string(['Soft-rank HS-GS ' policy_names{policy_index}]), ...
            point_nmse(soft_col),10*log10(max(point_nmse(soft_col),realmin))};
        boot_gain = zeros(bootstrap_replicates,1);
        for replicate = 1:bootstrap_replicates
            bootstrap_nmse = zeros(numel(SNR_grid),2);
            for snr_index = 1:numel(SNR_grid)
                idx = stratum_indices{snr_index};
                draw = idx(randi(numel(idx),numel(idx),1));
                denom = sum(c.sample_energy(draw));
                bootstrap_nmse(snr_index,:) = sum(c.sample_error(draw,[1 soft_col]),1)/denom;
            end
            mean_nmse = mean(bootstrap_nmse,1);
            boot_gain(replicate) = 10*log10(max(mean_nmse(2),realmin))- ...
                10*log10(max(mean_nmse(1),realmin));
        end
        point_gain = 10*log10(max(point_nmse(soft_col),realmin))- ...
            10*log10(max(point_nmse(1),realmin));
        limits = prctile(boot_gain,[2.5 97.5]);
        avg_ci_row = avg_ci_row+1;
        avg_ci_rows{avg_ci_row} = {string(c.study),c.N,c.K,c.P, ...
            string(policy_names{policy_index}),point_gain,limits(1),limits(2), ...
            limits(1)>0 || limits(2)<0,bootstrap_replicates};
    end
end
avg_table = cell2table(vertcat(avg_rows{:}),'VariableNames', ...
    {'Study','N','K','P','Method','SNR_Average_LinearNMSE','SNR_Average_NMSE_dB'});
avg_ci_table = cell2table(vertcat(avg_ci_rows{:}),'VariableNames', ...
    {'Study','N','K','P','SoftThresholdPolicy','V4_Improvement_dB', ...
    'CI95_Lower_dB','CI95_Upper_dB','CI_Excludes_Zero','BootstrapReplicates'});
writetable(avg_table(strcmpi(string(avg_table.Study),'Antenna'),:), ...
    fullfile(test_root,'snr_averaged_nmse_antenna.csv'));
writetable(avg_table(strcmpi(string(avg_table.Study),'User'),:), ...
    fullfile(test_root,'snr_averaged_nmse_users.csv'));
writetable(avg_ci_table,fullfile(stats_root,'v4_vs_soft_rank_snr_average_bootstrap_ci.csv'));

% Figures: per-SNR NMSE, SNR-averaged NMSE, paired gain with intervals, and
% total runtime including GS initialization. P=10/P=30 and studies stay apart.
method_names = [{'Frozen PI-DU-RACE V4'}, ...
    cellfun(@(x) ['Soft-rank HS-GS ' x],policy_names,'UniformOutput',false)];
for P = P_list
    for study_name = {'Antenna','User'}
        study = study_name{1};
        if strcmp(study,'Antenna')
            x_field = 'N';
            x_label = 'Antenna count N';
            x_values = [8 16 24 32];
        else
            x_field = 'K';
            x_label = 'User count K';
            if P == 10
                x_values = 1:5;
            else
                x_values = 1:7;
            end
        end
        figure('Visible','off','Color','w','Position',[100 100 1500 700]);
        tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
        for snr_index = 1:numel(SNR_grid)
            nexttile;
            hold on;
            for method_index = 1:numel(method_names)
                y = nan(size(x_values));
                for x_index = 1:numel(x_values)
                    y(x_index) = local_condition_nmse(condition_table,study,x_field, ...
                        x_values(x_index),P,SNR_grid(snr_index),method_index,policy_names);
                end
                plot(x_values,y,'-o','LineWidth',1.3,'DisplayName',method_names{method_index});
            end
            grid on;
            xlabel(x_label);
            ylabel('NMSE (dB)');
            title(sprintf('SNR = %g dB',SNR_grid(snr_index)));
            if snr_index == 1
                legend('Location','best');
            end
        end
        sgtitle(sprintf('%s sweep, P=%d: per-SNR paired test reanalysis',study,P));
        local_save_figure(gcf,figure_root,sprintf('%s_P%d_per_SNR_NMSE',study,P));
        close(gcf);

        figure('Visible','off','Color','w','Position',[100 100 1050 700]);
        hold on;
        for method_index = 1:numel(method_names)
            y = nan(size(x_values));
            for x_index = 1:numel(x_values)
                selector = strcmpi(string(avg_table.Study),study) & avg_table.P == P & ...
                    avg_table.(x_field) == x_values(x_index) & ...
                    strcmp(string(avg_table.Method),string(method_names{method_index}));
                if nnz(selector) ~= 1
                    error('Missing or duplicate SNR-average row for %s P=%d.',study,P);
                end
                y(x_index) = avg_table.SNR_Average_NMSE_dB(selector);
            end
            plot(x_values,y,'-o','LineWidth',1.5,'DisplayName',method_names{method_index});
        end
        grid on;
        xlabel(x_label);
        ylabel('Aggregate NMSE over SNR (dB)');
        title(sprintf('%s sweep, P=%d: aggregate NMSE over SNR',study,P));
        legend('Location','best');
        local_save_figure(gcf,figure_root,sprintf('%s_P%d_SNR_averaged_NMSE',study,P));
        close(gcf);

        figure('Visible','off','Color','w','Position',[100 100 1050 700]);
        hold on;
        for policy_index = 1:numel(policy_names)
            y = nan(size(x_values));
            lower = nan(size(x_values));
            upper = nan(size(x_values));
            for x_index = 1:numel(x_values)
                selector = strcmpi(string(avg_ci_table.Study),study) & avg_ci_table.P == P & ...
                    avg_ci_table.(x_field) == x_values(x_index) & ...
                    strcmp(string(avg_ci_table.SoftThresholdPolicy),string(policy_names{policy_index}));
                if nnz(selector) ~= 1
                    error('Missing or duplicate aggregate CI for %s P=%d.',study,P);
                end
                y(x_index) = avg_ci_table.V4_Improvement_dB(selector);
                lower(x_index) = y(x_index)-avg_ci_table.CI95_Lower_dB(selector);
                upper(x_index) = avg_ci_table.CI95_Upper_dB(selector)-y(x_index);
            end
            errorbar(x_values,y,lower,upper,'-o','LineWidth',1.4, ...
                'DisplayName',policy_names{policy_index});
        end
        yline(0,'k:');
        grid on;
        xlabel(x_label);
        ylabel('Frozen V4 improvement (dB)');
        title(sprintf('%s sweep, P=%d: paired bootstrap 95%% CI',study,P));
        legend('Location','best');
        local_save_figure(gcf,figure_root,sprintf('%s_P%d_SNR_averaged_gain_CI',study,P));
        close(gcf);

        figure('Visible','off','Color','w','Position',[100 100 1050 700]);
        hold on;
        runtime_methods = unique(string(runtime_table.Method),'stable');
        for method_index = 1:numel(runtime_methods)
            y = nan(size(x_values));
            for x_index = 1:numel(x_values)
                selector = strcmpi(string(runtime_table.Study),study) & runtime_table.P == P & ...
                    runtime_table.(x_field) == x_values(x_index) & ...
                    string(runtime_table.Method) == runtime_methods(method_index);
                if nnz(selector) ~= 1
                    continue;
                end
                y(x_index) = runtime_table.MeanSecondsPerSample(selector);
            end
            plot(x_values,y,'-o','LineWidth',1.4,'DisplayName',runtime_methods(method_index));
        end
        grid on;
        xlabel(x_label);
        ylabel('End-to-end runtime per sample (s)');
        title(sprintf('%s sweep, P=%d: runtime includes GS initialization',study,P));
        legend('Location','best');
        local_save_figure(gcf,figure_root,sprintf('%s_P%d_runtime',study,P));
        close(gcf);

        figure('Visible','off','Color','w','Position',[100 100 1050 700]);
        hold on;
        for method_index = 1:numel(method_names)
            runtime_x = nan(size(x_values));
            nmse_y = nan(size(x_values));
            for x_index = 1:numel(x_values)
                runtime_selector = strcmpi(string(runtime_table.Study),study) & ...
                    runtime_table.P == P & runtime_table.(x_field) == x_values(x_index) & ...
                    string(runtime_table.Method) == string(method_names{method_index});
                nmse_selector = strcmpi(string(avg_table.Study),study) & ...
                    avg_table.P == P & avg_table.(x_field) == x_values(x_index) & ...
                    string(avg_table.Method) == string(method_names{method_index});
                if nnz(runtime_selector) ~= 1 || nnz(nmse_selector) ~= 1
                    error('Missing runtime/accuracy pair for %s P=%d.',study,P);
                end
                runtime_x(x_index) = runtime_table.MeanSecondsPerSample(runtime_selector);
                nmse_y(x_index) = avg_table.SNR_Average_NMSE_dB(nmse_selector);
            end
            scatter(runtime_x,nmse_y,55,'filled','DisplayName',method_names{method_index});
        end
        grid on;
        xlabel('End-to-end runtime per sample (s)');
        ylabel('Aggregate NMSE over SNR (dB)');
        title(sprintf('%s sweep, P=%d: accuracy versus runtime',study,P));
        legend('Location','best');
        local_save_figure(gcf,figure_root,sprintf('%s_P%d_accuracy_vs_runtime',study,P));
        close(gcf);
    end
end

summary_file = fullfile(log_root,'experiment_summary.txt');
if isfile(summary_file)
    error('An experiment summary already exists; refusing to overwrite it.');
end
fid = fopen(summary_file,'w');
if fid < 0
    error('Could not create experiment summary file.');
end
closer = onCleanup(@() fclose(fid));
fprintf(fid,'Frozen PI-DU-RACE V4 vs nonlearned soft-rank HS-GS\n');
fprintf(fid,'Status: paired reanalysis of previously viewed test releases; not blind confirmation.\n');
fprintf(fid,'No test data were used for threshold tuning. No V4 checkpoint was modified.\n');
fprintf(fid,'Bootstrap replicates: %d; seed: %d; CI: percentile 95%%.\n', ...
    bootstrap_replicates,bootstrap_seed);
fprintf(fid,'Positive V4 improvement means soft-rank NMSE is higher (V4 is better).\n');
fprintf(fid,'NMSE: pooled squared error / pooled true-channel energy; dB after pooling.\n');
fprintf(fid,'SNR aggregate: arithmetic mean of five pooled linear NMSE values, then dB.\n');
fprintf(fid,'Full outputs: %s\n',experiment_root);
clear closer;

fprintf('Paired statistical analysis and figures complete from saved scores.\n');
fprintf('This script did not load or rescore source test data.\n');

function value = local_condition_nmse(T,study,x_field,x_value,P,SNR_dB,method_index,policy_names)
if method_index == 1
    policy = 'configuration_specific';
else
    policy = policy_names{method_index-1};
end
selector = strcmpi(string(T.Study),study) & T.P == P & T.(x_field) == x_value & ...
    T.SNR_dB == SNR_dB & strcmp(string(T.SoftThresholdPolicy),policy);
if nnz(selector) ~= 1
    error('Missing or duplicate NMSE row for %s P=%d SNR=%g.',study,P,SNR_dB);
end
if method_index == 1
    value = T.V4_NMSE_dB(selector);
else
    value = T.SoftRank_NMSE_dB(selector);
end
end

function local_save_figure(fig,folder,base_name)
exportgraphics(fig,fullfile(folder,[base_name '.png']),'Resolution',300);
exportgraphics(fig,fullfile(folder,[base_name '.pdf']),'ContentType','vector');
end
