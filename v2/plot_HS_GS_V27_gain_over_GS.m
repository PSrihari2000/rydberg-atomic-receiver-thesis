%% Plot the V27 paired NMSE gain of HS-GS over converged biased GS
% Positive gain means HS-GS has lower NMSE; zero means equal NMSE.
% This is a difference plot, so it makes small changes visible while keeping
% the absolute-NMSE comparison available in the three-method plot.
% It reads saved V27 results only and does not rerun the simulation.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   plot_HS_GS_V27_gain_over_GS

script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
table_file = fullfile(result_folder,'HS_GS_V27_frozen_policy_fresh_test.csv');
png_file = fullfile(result_folder,'HS_GS_V27_gain_over_GS.png');
fig_file = fullfile(result_folder,'HS_GS_V27_gain_over_GS.fig');

if ~isfile(table_file)
    error('V27 result table was not found: %s',table_file);
end
results = readtable(table_file);
required_variables = {'N','P','SNR_dB','gain_over_biased_GS_dB', ...
    'CI_low_biased_GS','CI_high_biased_GS'};
if ~all(ismember(required_variables,results.Properties.VariableNames))
    error('The V27 CSV does not contain gain and confidence interval columns.');
end

N_list = [8 16];
P_list = [10 30];
line_colors = [0.00 0.45 0.74; 0.93 0.40 0.10; ...
    0.49 0.18 0.56; 0.20 0.60 0.30];
figure_handle = figure('Color','w','Name','V27 HS-GS gain over GS');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
for n_index = 1:numel(N_list)
    for p_index = 1:numel(P_list)
        N = N_list(n_index);
        P = P_list(p_index);
        nexttile;
        selected = results(results.N==N & results.P==P,:);
        selected = sortrows(selected,'SNR_dB');
        if height(selected) ~= 8
            error('Expected 8 SNR rows for N=%d, P=%d; found %d.', ...
                N,P,height(selected));
        end
        gain = selected.gain_over_biased_GS_dB;
        ci_low = selected.CI_low_biased_GS;
        ci_high = selected.CI_high_biased_GS;
        lower_error = max(gain-ci_low,0);
        upper_error = max(ci_high-gain,0);
        errorbar(selected.SNR_dB,gain,lower_error,upper_error,'o-', ...
            'Color',line_colors((n_index-1)*numel(P_list)+p_index,:), ...
            'LineWidth',1.5,'MarkerSize',5,'CapSize',5);
        hold on;
        yline(0,'k--','GS parity','LineWidth',1.0);
        grid on;
        xlabel('SNR (dB)');
        ylabel('HS-GS gain over GS (dB)');
        title(sprintf('N = %d, P = %d',N,P));
    end
end
sgtitle('V27 fresh paired test: positive gain favors HS-GS (95% paired bootstrap CI)');
exportgraphics(figure_handle,png_file,'Resolution',200);
savefig(figure_handle,fig_file);
fprintf('HS-GS gain plot saved to:\n%s\n%s\n',png_file,fig_file);
