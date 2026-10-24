%% compare_hs_gs_v2_with_baselines.m
% Compare the isolated HS-GS v2 result with the existing GS and GD curves.
% This script only reads previous results and writes a new figure in v2.

clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
v2_result_folder = fullfile(script_folder, 'results');
baseline_result_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', ...
    'final_four_method_results_RSR25dB');

hs_gs_file = fullfile(v2_result_folder, 'HS_GS_V2_results.mat');
baseline_file = fullfile(baseline_result_folder, 'final_four_method_results.mat');

if ~exist(hs_gs_file, 'file')
    error('Run HS_GS_V2.m first: %s was not found.', hs_gs_file);
end
if ~exist(baseline_file, 'file')
    error('Baseline result file was not found: %s', baseline_file);
end

hs_gs = load(hs_gs_file);
baseline = load(baseline_file);

if ~isequal(hs_gs.P_list, baseline.P_list) || ~isequal(hs_gs.SNR_dB, baseline.SNR_dB)
    error('The v2 and baseline results use different P or SNR grids.');
end

figure('Color', 'w', 'Position', [100 100 1200 700]);
for P_index = 1:length(hs_gs.P_list)
    subplot(1, 2, P_index);
    hold on;
    grid on;
    box on;
    plot(hs_gs.SNR_dB, baseline.NMSE_GS_dB(P_index, :), '-o', ...
        'LineWidth', 1.4, 'DisplayName', 'GS');
    plot(hs_gs.SNR_dB, baseline.NMSE_GD_dB(P_index, :), '-s', ...
        'LineWidth', 1.4, 'DisplayName', 'GD');
    plot(hs_gs.SNR_dB, baseline.NMSE_HS_GS_dB(P_index, :), '--^', ...
        'LineWidth', 1.4, 'DisplayName', 'Previous HS-GS');
    plot(hs_gs.SNR_dB, hs_gs.NMSE_HS_GS_dB(P_index, :), '-^', ...
        'LineWidth', 1.8, 'DisplayName', 'HS-GS v2');
    xlabel('SNR (dB)');
    ylabel('NMSE (dB)');
    title(sprintf('P = %d', hs_gs.P_list(P_index)));
    xticks(hs_gs.SNR_dB);
    legend('Location', 'best');
end

sgtitle('HS-GS v2 versus GS and GD, N = 8, K = 3, RSR = 25 dB');
exportgraphics(gcf, fullfile(v2_result_folder, ...
    'HS_GS_V2_with_GS_GD_comparison.png'), 'Resolution', 300);
savefig(gcf, fullfile(v2_result_folder, ...
    'HS_GS_V2_with_GS_GD_comparison.fig'));

comparison_table = table(hs_gs.SNR_dB(:), ...
    hs_gs.NMSE_HS_GS_dB(1, :).', baseline.NMSE_GS_dB(1, :).', ...
    baseline.NMSE_GD_dB(1, :).', hs_gs.NMSE_HS_GS_dB(2, :).', ...
    baseline.NMSE_GS_dB(2, :).', baseline.NMSE_GD_dB(2, :).', ...
    'VariableNames', {'SNR_dB', 'HS_GS_v2_P10', 'GS_P10', 'GD_P10', ...
    'HS_GS_v2_P30', 'GS_P30', 'GD_P30'});
disp(comparison_table);
save(fullfile(v2_result_folder, 'HS_GS_V2_with_GS_GD_comparison.mat'), ...
    'comparison_table');
