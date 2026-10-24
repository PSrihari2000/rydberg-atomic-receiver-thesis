%% compare_four_methods_v2.m
% Compare GS, GD, HS-GS v2, and unfolded-DL on the same SNR grid.
% The DL values below are the current independent Colab test results.
% Replace them after a new DL training run if the values change.

clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
result_folder = fullfile(script_folder, 'results');
baseline_folder = fullfile(script_folder, '..', 'GS-GS-HS-DL', ...
    'final_four_method_results_RSR25dB');

hs_file = fullfile(result_folder, ...
    'HS_GS_V2_ablation_parallel_results.mat');
baseline_file = fullfile(baseline_folder, ...
    'final_four_method_results.mat');

if ~exist(hs_file, 'file')
    error('HS-GS result file not found: %s', hs_file);
end
if ~exist(baseline_file, 'file')
    error('GS/GD baseline file not found: %s', baseline_file);
end

hs = load(hs_file);
baseline = load(baseline_file);

if ~isequal(hs.P_list, baseline.P_list) || ~isequal(hs.SNR_dB, baseline.SNR_dB)
    error('The result files use different P or SNR grids.');
end

% Current Colab test NMSE values, in dB, ordered by SNR = -5:5:30 dB.
DL_NMSE_dB = [
    -1.3410141,  -2.9933491,  -5.4787695,  -8.6203677, ...
   -12.9792176, -17.1527399, -21.6240217, -24.4649915; ...
    -3.6131194,  -6.7274201, -11.1643437, -15.9993294, ...
   -21.0061658, -25.8757588, -30.7970983, -35.6477381];

if ~isequal(size(DL_NMSE_dB), size(hs.test_NMSE_dB))
    error('DL_NMSE_dB must have size [2, 8] for P = 10 and P = 30.');
end

figure('Color', 'w', 'Position', [100 100 1300 650]);
for P_index = 1:numel(hs.P_list)
    subplot(1, 2, P_index);
    hold on;
    grid on;
    box on;
    plot(hs.SNR_dB, baseline.NMSE_GS_dB(P_index, :), '-o', ...
        'LineWidth', 1.4, 'DisplayName', 'GS');
    plot(hs.SNR_dB, baseline.NMSE_GD_dB(P_index, :), '-s', ...
        'LineWidth', 1.4, 'DisplayName', 'GD');
    plot(hs.SNR_dB, hs.test_NMSE_dB(P_index, :), '-^', ...
        'LineWidth', 1.8, 'DisplayName', 'HS-GS v2');
    plot(hs.SNR_dB, DL_NMSE_dB(P_index, :), '-d', ...
        'LineWidth', 1.8, 'DisplayName', 'Unfolded DL');
    xlabel('SNR (dB)');
    ylabel('NMSE (dB)');
    title(sprintf('P = %d', hs.P_list(P_index)));
    xticks(hs.SNR_dB);
    legend('Location', 'best');
end
sgtitle('GS, GD, HS-GS v2, and unfolded DL; RSR = 25 dB');

exportgraphics(gcf, fullfile(result_folder, ...
    'four_method_comparison_v2.png'), 'Resolution', 300);
savefig(gcf, fullfile(result_folder, ...
    'four_method_comparison_v2.fig'));

comparison_table = table(hs.SNR_dB(:), ...
    baseline.NMSE_GS_dB(1, :).', baseline.NMSE_GD_dB(1, :).', ...
    hs.test_NMSE_dB(1, :).', DL_NMSE_dB(1, :).', ...
    baseline.NMSE_GS_dB(2, :).', baseline.NMSE_GD_dB(2, :).', ...
    hs.test_NMSE_dB(2, :).', DL_NMSE_dB(2, :).', ...
    'VariableNames', {'SNR_dB', 'GS_P10', 'GD_P10', 'HS_GS_v2_P10', ...
    'DL_P10', 'GS_P30', 'GD_P30', 'HS_GS_v2_P30', 'DL_P30'});
disp(comparison_table);
save(fullfile(result_folder, 'four_method_comparison_v2.mat'), ...
    'comparison_table', 'DL_NMSE_dB');

fprintf('\nSaved four-method plot and table in: %s\n', result_folder);
