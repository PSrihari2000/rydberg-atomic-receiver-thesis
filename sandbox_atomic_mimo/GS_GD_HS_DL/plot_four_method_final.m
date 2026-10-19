function plot_four_method_final()
%PLOT_FOUR_METHOD_FINAL Plot the frozen 500-trial four-method comparison.

rootDir = fileparts(mfilename('fullpath'));
resultDir = fullfile(rootDir, 'four_method_final');
tablePath = fullfile(resultDir, 'nmse_snr.csv');
T = readtable(tablePath, 'TextType', 'string');

pilotCounts = [10, 30];
methods = ["Biased GS", "Linearized GD", "HS-GS", ...
    "Noise-aware unfolded DL"];
colors = [0.0000 0.4470 0.7410; ...
          0.8500 0.3250 0.0980; ...
          0.4660 0.6740 0.1880; ...
          0.4940 0.1840 0.5560];
markers = {'o', 's', '^', 'd'};

fig = figure('Color', 'w', 'Position', [100 100 1280 500]);
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

for pIndex = 1:numel(pilotCounts)
    ax = nexttile(layout);
    hold(ax, 'on');
    for methodIndex = 1:numel(methods)
        rows = T.P == pilotCounts(pIndex) & T.Method == methods(methodIndex);
        x = T.SNRdB(rows);
        y = T.NMSEdB(rows);
        [x, order] = sort(x);
        y = y(order);
        plot(ax, x, y, '-', 'Color', colors(methodIndex, :), ...
            'Marker', markers{methodIndex}, 'LineWidth', 1.8, ...
            'MarkerSize', 6, 'DisplayName', methods(methodIndex));
    end
    grid(ax, 'on');
    box(ax, 'on');
    xlabel(ax, 'SNR (dB)');
    ylabel(ax, 'NMSE (dB)');
    title(ax, sprintf('P = %d pilots', pilotCounts(pIndex)));
    xticks(ax, -5:5:30);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 12);
end

legend(ax, methods, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'NumColumns', 2);
title(layout, 'Magnitude-only channel estimation, N = 8 and K = 3', ...
    'FontName', 'Times New Roman', 'FontSize', 15, ...
    'FontWeight', 'bold');

exportgraphics(fig, fullfile(resultDir, 'nmse_vs_snr.png'), ...
    'Resolution', 300);
savefig(fig, fullfile(resultDir, 'nmse_vs_snr.fig'));
close(fig);
end
