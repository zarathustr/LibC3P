%% benchmark_c3p_noise_sweep_boxplot_summary.m
% Two-figure summary for three C3P specializations:
%   AX = YB,  AXB = YCZ,  AXBY = ZCWD
%
% Figure 1: Rotation error vs noise, all three problem types in one plot
% Figure 2: Translation error vs noise, all three problem types in one plot
%
% Input CSVs come from pem_c3p_noise_sweep:
%   problem, solver, P, Q, noise, trial, rot_err_deg, trans_err

clear; close all; clc;
warning('off');
format long g

%% User settings
csvFiles = { ...
    '../build/c3p_p1q1.csv', ...
    '../build/c3p_p1q2.csv', ...
    '../build/c3p_p2q2.csv'  ...
};

meas_num = 20;                 % should match --N used in pem_c3p_noise_sweep
solverWanted = "PEM";          % set to "PEM" or "Analytical" if multiple solvers exist
saveFigures = false;           % set true to save PNGs
outDir = '.';                  % output directory for saved figures

%% Read all CSVs and concatenate
T = table();
for i = 1:numel(csvFiles)
    f = csvFiles{i};
    if ~isfile(f)
        warning('File not found: %s (skipped)', f);
        continue;
    end
    Ti = readtable(f);
    if isempty(T)
        T = Ti;
    else
        T = [T; Ti]; 
    end
end

if isempty(T) || height(T) == 0
    error('No data loaded. Check csvFiles paths.');
end

%% Basic checks
vars = string(T.Properties.VariableNames);
needVars = ["problem","solver","noise","trial","rot_err_deg","trans_err"];
for k = 1:numel(needVars)
    if ~any(vars == needVars(k))
        error('CSV data is missing column: %s', needVars(k));
    end
end

T.problem = string(T.problem);
T.solver  = string(T.solver);

%% Filter by solver if requested
if strlength(solverWanted) > 0
    if ~any(T.solver == solverWanted)
        error('solverWanted = %s not found in the loaded CSV data.', solverWanted);
    end
    T = T(T.solver == solverWanted, :);
end

if height(T) == 0
    error('No rows remain after filtering by solverWanted.');
end

%% Desired problem ordering
desiredOrder = ["AXYB","AXB_YCZ","AXBY_ZCWD"]';
problemsAll = unique(T.problem, 'stable');
problemsAll = problemsAll(:);
problems = [desiredOrder(ismember(desiredOrder, problemsAll)); problemsAll(~ismember(problemsAll, desiredOrder))];
problems = unique(problems, 'stable');

% Global noise levels (union)
noiseLevelsAll = unique(double(T.noise));
noiseLevelsAll = sort(noiseLevelsAll);
xBase = 1:numel(noiseLevelsAll);
xTickLabels = arrayfun(@(x) sprintf('%.4g', x), noiseLevelsAll, 'UniformOutput', false);

%% Style settings, matching the provided boxplot style
FontSizeAxis   = 18;
FontSizeLegend = 18;
BoxLineWidth   = 2.0;
LegendLoc      = 'northwest';
boxWidth       = 0.18;

% Colors per problem type
cGray   = [0.5, 0.5, 0.5];
cBlue   = [0.0, 0.4470, 0.7410];
cRed    = [1.0, 0.0, 0.0];

probNames = strings(0,1);
probColors = {};

for i = 1:numel(problems)
    prob = problems(i);
    if prob == "AXYB"
        probNames(end+1,1) = "AX = YB"; 
        probColors{end+1} = cGray; 
    elseif prob == "AXB_YCZ"
        probNames(end+1,1) = "AXB = YCZ"; 
        probColors{end+1} = cBlue; 
    elseif prob == "AXBY_ZCWD"
        probNames(end+1,1) = "AXBY = ZCWD"; 
        probColors{end+1} = cRed; 
    else
        probNames(end+1,1) = prob; 
        probColors{end+1} = cBlue; 
    end
end

% Offsets for three problems in each noise group
Pnum = numel(problems);
if Pnum == 1
    offsets = 0;
else
    offsets = linspace(-0.28, 0.28, Pnum);
end

%% Rotation figure
figure('Color','w'); hold on; set(gca,'NextPlot','add');

for i = 1:Pnum
    prob = problems(i);
    Tp = T(T.problem == prob, :);
    if height(Tp) == 0
        continue;
    end

    color = probColors{i};

    % Noise levels present for this problem
    noiseHere = unique(double(Tp.noise));
    noiseHere = sort(noiseHere);

    % Map noiseHere to global x positions
    [tf, idxPos] = ismember(noiseHere, noiseLevelsAll);
    idxPos = idxPos(tf);
    pos = xBase(idxPos) + offsets(i);

    % Build maxCount x numel(noiseHere) matrix for boxplot
    maxCount = 0;
    for k = 1:numel(noiseHere)
        idx = abs(double(Tp.noise) - noiseHere(k)) < 1e-12;
        vals = Tp.rot_err_deg(idx);
        maxCount = max(maxCount, numel(vals));
    end

    Xmat = NaN(maxCount, numel(noiseHere));
    for k = 1:numel(noiseHere)
        idx = abs(double(Tp.noise) - noiseHere(k)) < 1e-12;
        vals = Tp.rot_err_deg(idx);
        if ~isempty(vals)
            Xmat(1:numel(vals), k) = vals(:);
        end
    end

    h = boxplot(Xmat, 'positions', pos, 'colors', color, ...
        'symbol', '', 'widths', boxWidth, 'plotstyle', 'traditional');

    try
        set(h, 'LineWidth', BoxLineWidth);
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

% Legend
hLeg = gobjects(Pnum,1);
for i = 1:Pnum
    hLeg(i) = plot(NaN, NaN, 'Color', probColors{i}, 'LineWidth', BoxLineWidth);
end

xlabel('Noise Levels', 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
ylabel('Rotation Error (Unit: $^\circ$)', 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
title(sprintf('$N = %d$, Solver = Ours-%s', meas_num, solverWanted), ...
    'Interpreter','LaTeX', 'FontSize', FontSizeAxis);

grid on; grid minor;
set(gca, 'XTick', xBase, 'XTickLabel', xTickLabels, 'FontSize', FontSizeAxis);
xlim([0.5, numel(noiseLevelsAll)+0.5]);
ylim([0, 1.5]);
% ylim(autoYLim(double(T.rot_err_deg)));

legend(hLeg, probNames, 'Interpreter','LaTeX', 'FontSize', FontSizeLegend, 'Location', LegendLoc);

if saveFigures
    if ~exist(outDir,'dir'), mkdir(outDir); end
    saveas(gcf, fullfile(outDir, sprintf('c3p_summary_rot_%s.png', char(solverWanted))));
end

%% Translation figure
figure('Color','w'); hold on; set(gca,'NextPlot','add');

for i = 1:Pnum
    prob = problems(i);
    Tp = T(T.problem == prob, :);
    if height(Tp) == 0
        continue;
    end

    color = probColors{i};

    noiseHere = unique(double(Tp.noise));
    noiseHere = sort(noiseHere);

    [tf, idxPos] = ismember(noiseHere, noiseLevelsAll);
    idxPos = idxPos(tf);
    pos = xBase(idxPos) + offsets(i);

    maxCount = 0;
    for k = 1:numel(noiseHere)
        idx = abs(double(Tp.noise) - noiseHere(k)) < 1e-12;
        vals = Tp.trans_err(idx);
        maxCount = max(maxCount, numel(vals));
    end

    Xmat = NaN(maxCount, numel(noiseHere));
    for k = 1:numel(noiseHere)
        idx = abs(double(Tp.noise) - noiseHere(k)) < 1e-12;
        vals = Tp.trans_err(idx);
        if ~isempty(vals)
            Xmat(1:numel(vals), k) = vals(:);
        end
    end

    h = boxplot(Xmat, 'positions', pos, 'colors', color, ...
        'symbol', '', 'widths', boxWidth, 'plotstyle', 'traditional');

    try
        set(h, 'LineWidth', BoxLineWidth);
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

% Legend
hLeg = gobjects(Pnum,1);
for i = 1:Pnum
    hLeg(i) = plot(NaN, NaN, 'Color', probColors{i}, 'LineWidth', BoxLineWidth);
end

xlabel('Noise Levels', 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
ylabel('Translation Error (Unit: m)', 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
title(sprintf('$N = %d$, Solver = Ours-%s', meas_num, solverWanted), ...
    'Interpreter','LaTeX', 'FontSize', FontSizeAxis);

grid on; grid minor;
set(gca, 'XTick', xBase, 'XTickLabel', xTickLabels, 'FontSize', FontSizeAxis);
xlim([0.5, numel(noiseLevelsAll)+0.5]);
ylim([0, 0.08]);
% ylim(autoYLim(double(T.trans_err)));

legend(hLeg, probNames, 'Interpreter','LaTeX', 'FontSize', FontSizeLegend, 'Location', LegendLoc);

if saveFigures
    if ~exist(outDir,'dir'), mkdir(outDir); end
    saveas(gcf, fullfile(outDir, sprintf('c3p_summary_trans_%s.png', char(solverWanted))));
end

%% ---- helper ----
function yl = autoYLim(vals)
vals = vals(:);
vals = vals(~isnan(vals));
if isempty(vals)
    yl = [0, 1];
    return;
end
mx = max(vals);
if mx <= 0
    yl = [0, 1];
    return;
end
yl = [0, mx * 1.15];
end
