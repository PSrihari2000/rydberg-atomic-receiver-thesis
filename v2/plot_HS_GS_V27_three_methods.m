%% Plot the V27 three-algorithm comparison from saved test results
% Reads the completed V27 CSV only; it does not rerun simulations.
% Curves shown: frozen V26 HS-GS, converged biased GS, and converged GD.
%
% Run in MATLAB:
%   cd('D:\MTech_Thesis_Results\v2')
%   plot_HS_GS_V27_three_methods

script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder,'results');
table_file = fullfile(result_folder,'HS_GS_V27_frozen_policy_fresh_test.csv');
png_file = fullfile(result_folder,'HS_GS_V27_three_method_comparison.png');
fig_file = fullfile(result_folder,'HS_GS_V27_three_method_comparison.fig');

if ~isfile(table_file)
    error('V27 result table was not found: %s',table_file);
end
results = readtable(table_file);
required_variables = {'N','P','SNR_dB','HS_GS_NMSE_dB', ...
    'biased_GS_converged_NMSE_dB','GD_converged_NMSE_dB'};
if ~all(ismember(required_variables,results.Properties.VariableNames))
    error('The V27 CSV does not contain the expected three-method result columns.');
end

N_list = [8 16];
P_list = [10 30];
method_columns = {'HS_GS_NMSE_dB','biased_GS_converged_NMSE_dB', ...
    'GD_converged_NMSE_dB'};
method_names = {'HS-GS (frozen V26 policy)','GS (converged)','GD (converged)'};
line_colors = [0.00 0.45 0.74; 0.93 0.40 0.10; 0.49 0.18 0.56];
markers = {'o','s','^'};

figure_handle = figure('Color','w','Name','V27: HS-GS, GS, and GD');
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
        for method_index = 1:numel(method_columns)
            plot(selected.SNR_dB,selected.(method_columns{method_index}), ...
                ['-' markers{method_index}],'Color',line_colors(method_index,:), ...
                'LineWidth',1.6,'MarkerSize',5, ...
                'DisplayName',method_names{method_index});
            hold on;
        end
        grid on;
        xlabel('SNR (dB)');
        ylabel('NMSE (dB)');
        title(sprintf('N = %d, P = %d',N,P));
        if n_index==1 && p_index==1
            legend('Location','southwest');
        end
    end
end
sgtitle('V27 fresh paired test: HS-GS, converged GS, and converged GD (RSR = 25 dB)');
exportgraphics(figure_handle,png_file,'Resolution',200);
savefig(figure_handle,fig_file);
fprintf('Three-method graph saved to:\n%s\n%s\n',png_file,fig_file);
