%% benchmark_accuracy_c3p_boxplot.m
% Boxplot drawing script in the same style as the provided PnP benchmark figures.
%
% This script reads the CSV produced by pem_benchmark_mc and generates
% grouped boxplots versus noise level, with per-solver colors and dashed whiskers.
%
% Expected CSV columns:
%   problem, solver, noise, trial, rot_err_deg, trans_err
%
% Example:
%   ./pem_benchmark_mc --mc 200 --N 20 --noise 0,0.002,0.005,0.01,0.02 --out mc_results.csv
%   Then run this script in MATLAB.
%
% Notes:
%   - X-axis uses noise levels as tick labels, while plotting uses integer positions
%     to avoid compressed axes when noise values are small.
%   - For each problem, two figures are produced: rotation error and translation error.

clear; close all; clc;
warning('off');
format long g

%% User settings
csvFile = '../build/mc_results.csv';     % path to CSV from pem_benchmark_mc
motion_num = 20;               % should match --N used in pem_benchmark_mc
saveFigures = false;           % set true to save PNGs
outDir = '.';                  % output directory for saved figures

%% Read CSV
T = readtable(csvFile);

% Convert to string arrays for robust filtering
problemCol = string(T.problem);
solverCol  = string(T.solver);

% Unique problems in file
problems = unique(problemCol, 'stable');

%% Plot settings, matching the look of the example figure
FontSizeAxis  = 18;
FontSizeLegend = 18;
BoxLineWidth  = 2.0;
LegendLoc     = 'northwest';

% Baseline colors similar to the example figure legend
cGray  = [0.5, 0.5, 0.5];
cBlue  = [0.0, 0.4470, 0.7410];
cOrange= [0.8500, 0.3250, 0.0980];
cPurple= [0.4940, 0.1840, 0.5560];
cCyan  = [0.3010, 0.7450, 0.9330];
cGreen = [0.4660, 0.6740, 0.1880];
cRed   = [1.0, 0.0, 0.0];

%% For each problem, define solver order and colors
for p = 1:numel(problems)
    prob = problems(p);

    if prob == "AXXB"
        methods = ["Park1994", "Horaud1995", "Daniilidis1999", "Zhang2017", "Analytical", "Ours-PEM"];
        DeviceColors = {cGray, cOrange, cBlue, cPurple, cGreen, cRed};
        probTitle = 'AX = XB';
    elseif prob == "AXYB"
        methods = ["Dornaika1998", "Shah2013", "Park2016", "Tabb2017", "Analytical", "Ours-PEM"];
        DeviceColors = {cGray, cOrange, cBlue, cPurple, cGreen, cRed};
        probTitle = 'AX = YB';
    elseif prob == "AXB_YCZ"
        methods = ["Wu2016", "Ma2018", "Sui2023", "Analytical", "Ours-PEM"];
        DeviceColors = {cGray, cBlue, cPurple, cGreen, cRed};
        probTitle = 'AXB = YCZ';
    else
        % Fallback: use whatever solvers appear for this problem
        methods = unique(solverCol(problemCol == prob), 'stable');
        baseCols = {cGray, cOrange, cBlue, cPurple, cCyan, cGreen, cRed};
        DeviceColors = baseCols(1:numel(methods));
        probTitle = char(prob);
    end

    % Filter table for this problem
    Tp = T(problemCol == prob, :);

    noiseLevels = unique(double(Tp.noise));
    noiseLevels = sort(noiseLevels);

    xBase = 1:numel(noiseLevels);
    xTickLabels = arrayfun(@(x) sprintf('%.4g', x), noiseLevels, 'UniformOutput', false);

    % Offsets for grouped boxes at each noise level
    M = numel(methods);
    offsets = linspace(-0.28, 0.28, M);

    % Rotation plot
    figure('Color', 'w'); hold on; set(gca, 'NextPlot', 'add');
    plotGroupedBoxplot(Tp, methods, DeviceColors, noiseLevels, xBase, offsets, ...
        "rot_err_deg", 0.18, BoxLineWidth);

    addLegendLines(methods, DeviceColors);
    xlabel('Noise Levels', 'Interpreter', 'LaTeX', 'FontSize', FontSizeAxis);
    ylabel('Rotation Error (Unit: deg)', 'Interpreter', 'LaTeX', 'FontSize', FontSizeAxis);
    grid on; grid minor;
    set(gca, 'XTick', xBase, 'XTickLabel', xTickLabels, 'FontSize', FontSizeAxis);
    title(sprintf('%s, Number of Measurements = %d', probTitle, motion_num), ...
        'Interpreter', 'LaTeX', 'FontSize', FontSizeAxis);
    legend(methods, 'Interpreter', 'LaTeX', 'FontSize', FontSizeLegend, 'Location', LegendLoc);
    xlim([0.5, numel(noiseLevels)+0.5]);
    ylim([0, 2]);

    if saveFigures
        if ~exist(outDir, 'dir'), mkdir(outDir); end
        saveas(gcf, fullfile(outDir, sprintf('%s_rot_boxplot.png', char(prob))));
    end

    % Translation plot
    figure('Color', 'w'); hold on; set(gca, 'NextPlot', 'add');
    plotGroupedBoxplot(Tp, methods, DeviceColors, noiseLevels, xBase, offsets, ...
        "trans_err", 0.18, BoxLineWidth);

    addLegendLines(methods, DeviceColors);
    xlabel('Noise Levels', 'Interpreter', 'LaTeX', 'FontSize', FontSizeAxis);
    ylabel('Translation Error (Unit: m)', 'Interpreter', 'LaTeX', 'FontSize', FontSizeAxis);
    grid on; grid minor;
    set(gca, 'XTick', xBase, 'XTickLabel', xTickLabels, 'FontSize', FontSizeAxis);
    title(sprintf('%s, Number of Measurements = %d', probTitle, motion_num), ...
        'Interpreter', 'LaTeX', 'FontSize', FontSizeAxis);
    legend(methods, 'Interpreter', 'LaTeX', 'FontSize', FontSizeLegend, 'Location', LegendLoc);
    xlim([0.5, numel(noiseLevels)+0.5]);
    ylim([0, 0.2]);

    if saveFigures
        if ~exist(outDir, 'dir'), mkdir(outDir); end
        saveas(gcf, fullfile(outDir, sprintf('%s_trans_boxplot.png', char(prob))));
    end
end

%% -------- Local helper functions --------

function plotGroupedBoxplot(Tp, methods, DeviceColors, noiseLevels, xBase, offsets, metricName, boxWidth, lineW)
% Draw grouped boxplots for one metric.
% Tp is a table filtered to one problem.

M = numel(methods);
K = numel(noiseLevels);

for m = 1:M
    method = methods(m);
    color = DeviceColors{m};

    % Determine max number of trials for padding
    maxCount = 0;
    for k = 1:K
        idx = string(Tp.solver) == method & abs(double(Tp.noise) - noiseLevels(k)) < 1e-12;
        vals = Tp.(metricName)(idx);
        maxCount = max(maxCount, numel(vals));
    end

    Xmat = NaN(maxCount, K);
    for k = 1:K
        idx = string(Tp.solver) == method & abs(double(Tp.noise) - noiseLevels(k)) < 1e-12;
        vals = Tp.(metricName)(idx);
        if ~isempty(vals)
            Xmat(1:numel(vals), k) = vals(:);
        end
    end

    pos = xBase + offsets(m);

    h = boxplot(Xmat, 'positions', pos, 'colors', color, ...
        'symbol', '', 'widths', boxWidth, 'plotstyle', 'traditional');

    % Style: dashed whiskers, colored lines, consistent linewidth
    try
        set(h, 'LineWidth', lineW);
    catch
    end
    try
        set(h(1:2,:), 'LineStyle', '--');
    catch
    end
    try
        set(h(3:4,:), 'LineStyle', '--');
    catch
    end
end
end

function addLegendLines(methods, DeviceColors)
% Create dummy line objects so the legend shows clean colored lines.

for i = 1:numel(methods)
    plot(NaN, 1, 'color', DeviceColors{i}, 'LineWidth', 1);
end
end
