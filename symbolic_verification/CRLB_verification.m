%% demo_cov_crlb_AXBY_ZCWD.m
% Numerical simulation of covariance estimation and CRLB check
% for the C3P specialization AXBY = ZCWD.
%
% This script:
% 1) Generates a noise-free synthetic dataset that exactly satisfies
%       A_i * X * B_i * Y = Z * C_i * W * D_i
% 2) Adds right-invariant SE(3) noise to A_i, B_i, C_i, D_i
% 3) Computes the CRLB of the stacked pose parameters [X,Y,Z,W] using
%    the Lie-algebra linearization r_i = Log( (Z C W D)^{-1} (A X B Y) )
% 4) Runs Monte Carlo to estimate the covariance of the pose error twists
% 5) Visualizes 3-sigma bounds from Monte Carlo covariance vs CRLB
%
% Output convention:
% - Pose error is the right-invariant error twist
%       xi_X = Log( X_gt^{-1} * X_hat )
%   similarly for Y,Z,W.
%
% Visualization:
% - Figure 1: per-pose 3-sigma comparison (MC vs CRLB) in twist coordinates
% - Figure 2: covariance heatmaps (MC vs CRLB)
% - Figure 3: translation 3-sigma ellipsoids for X translation block (optional)

clear; close all; clc; rng(1);

%% User parameters
N  = 30;         % number of measurements
MC = 2000;       % Monte Carlo trials

% Measurement noise on SE(3) factors A,B,C,D
sigmaRotDeg = 0.5;                     % degrees, applied to axis-angle components
sigmaRot    = sigmaRotDeg * pi/180;    % radians
sigmaTrans  = 0.005;                   % meters

% Random motion magnitudes for generating ground truth and measurements
rotMagGT  = 0.8;    % rad
tranMagGT = 1.0;    % m
rotMagMeas  = 1.0;  % rad
tranMagMeas = 1.0;  % m

% Plot options
showEllipsoid = true;   % set false if you only want the bar plots

%% Build ground-truth unknowns X,Y,Z,W in SE(3)
Xgt = randSE3(rotMagGT, tranMagGT);
Ygt = randSE3(rotMagGT, tranMagGT);
Zgt = randSE3(rotMagGT, tranMagGT);
Wgt = randSE3(rotMagGT, tranMagGT);

%% Generate noise-free measurement factors A,B,C and compute D to satisfy closure
A = cell(N,1); B = cell(N,1); C = cell(N,1); D = cell(N,1);

for i = 1:N
    A{i} = randSE3(rotMagMeas, tranMagMeas);
    B{i} = randSE3(rotMagMeas, tranMagMeas);
    C{i} = randSE3(rotMagMeas, tranMagMeas);

    L = A{i} * Xgt * B{i} * Ygt;
    % Enforce A X B Y = Z C W D  =>  D = W^{-1} C^{-1} Z^{-1} (A X B Y)
    D{i} = invSE3(Wgt) * invSE3(C{i}) * invSE3(Zgt) * L;
end

% Sanity check: objective at truth should be ~0 in noise-free case
r0 = residualStack_AXBY_ZCWD(A, B, C, D, Xgt, Ygt, Zgt, Wgt);
fprintf('Noise-free mean residual norm: %.3e\n', mean(vecnorm(reshape(r0,6,[]),2,1)));

%% CRLB computation at the true parameters
SigmaMeas = diag([sigmaRot^2 sigmaRot^2 sigmaRot^2 sigmaTrans^2 sigmaTrans^2 sigmaTrans^2]);

[FIM, CRLB, JthetaCells, SigmaCells] = crlb_AXBY_ZCWD(A, B, C, D, Xgt, Ygt, Zgt, Wgt, SigmaMeas);

fprintf('FIM size: %d x %d\n', size(FIM,1), size(FIM,2));
fprintf('CRLB trace: %.6e\n', trace(CRLB));

%% Monte Carlo covariance estimation
Dparam = 24;
deltaHat = zeros(Dparam, MC);

FIMsym = 0.5 * (FIM + FIM');

for t = 1:MC
    % Add measurement noise
    Ahat = cell(N,1); Bhat = cell(N,1); Chat = cell(N,1); Dhat = cell(N,1);
    for i = 1:N
        Ahat{i} = addNoiseSE3(A{i}, sigmaRot, sigmaTrans);
        Bhat{i} = addNoiseSE3(B{i}, sigmaRot, sigmaTrans);
        Chat{i} = addNoiseSE3(C{i}, sigmaRot, sigmaTrans);
        Dhat{i} = addNoiseSE3(D{i}, sigmaRot, sigmaTrans);
    end

    % Residual evaluated at the true parameters
    r = residualStack_AXBY_ZCWD(Ahat, Bhat, Chat, Dhat, Xgt, Ygt, Zgt, Wgt);

    % One-step optimal WLS estimate in the linearized model
    g = zeros(Dparam,1);
    for i = 1:N
        ri = r(6*(i-1)+1:6*i);
        Ji = JthetaCells{i};
        Si = SigmaCells{i};
        g = g + Ji' * (Si \ ri);
    end

    deltaHat(:,t) = FIMsym \ g;
end

CovMC = cov(deltaHat');                 % 24x24 sample covariance
CovMC = 0.5 * (CovMC + CovMC');         % symmetrize

% Compare Monte Carlo covariance to CRLB
relFro = norm(CovMC - CRLB, 'fro') / max(norm(CRLB, 'fro'), 1e-12);
relDiag = mean(abs(diag(CovMC) - diag(CRLB)) ./ max(abs(diag(CRLB)), 1e-12));
eigMin = min(eig(0.5*(CovMC - CRLB + (CovMC - CRLB)')));

fprintf('MC vs CRLB: relative Frobenius error %.3e\n', relFro);
fprintf('MC vs CRLB: mean relative diagonal error %.3e\n', relDiag);
fprintf('MC vs CRLB: min eigenvalue CovMC-CRLB %.3e\n', eigMin);

%% 3-sigma comparison plots in twist coordinates
sigMC   = 3 * sqrt(max(diag(CovMC), 0));
sigCRLB = 3 * sqrt(max(diag(CRLB), 0));

labels6 = {'$\omega_x$','$\omega_y$','$\omega_z$','$t_x$','$t_y$','$t_z$'};
poseNames = {'X','Y','Z','W'};

figure('Color','w');
tiledlayout(2,2, 'Padding','compact', 'TileSpacing','compact');

for p = 1:4
    idx = (p-1)*6 + (1:6);
    M = [sigCRLB(idx), sigMC(idx)];

    nexttile;
    b = bar(M, 'grouped');
    grid on; grid minor;

    set(gca, 'XTick', 1:6, 'XTickLabel', labels6, 'TickLabelInterpreter','latex');
    xlabel('Twist Component', 'Interpreter','latex');
    ylabel('3$\sigma$ bound', 'Interpreter','latex');
    title(sprintf('%s', poseNames{p}), 'Interpreter','latex');

    if p == 1
        legend({'CRLB 3$\sigma$', 'MC 3$\sigma$'}, 'Interpreter','latex', 'Location','northwest');
    end
end

sgtitle(sprintf('AXBY=ZCWD: 3\\sigma pose uncertainty, N=%d, MC=%d, noise %.2f deg, %.3f m', ...
    N, MC, sigmaRotDeg, sigmaTrans), 'Interpreter','latex');

%% Covariance heatmaps
figure('Color','w');
subplot(1, 2, 1);
imagesc(CovMC);
colormap parula
axis image
colorbar
title('Monte-Carlo Covariance', 'Interpreter', 'LaTeX', 'FontSize', 18);
xlabel('Parameter Index', 'Interpreter', 'LaTeX', 'FontSize', 18); 
ylabel('Parameter Index', 'Interpreter', 'LaTeX', 'FontSize', 18);

subplot(1, 2, 2);
imagesc(CRLB);
axis image
colorbar
title('CRLB', 'Interpreter', 'LaTeX', 'FontSize', 18);
xlabel('Parameter Index', 'Interpreter', 'LaTeX', 'FontSize', 18); 
ylabel('Parameter Index', 'Interpreter', 'LaTeX', 'FontSize', 18);

%% Optional: 3-sigma translation ellipsoid for X translation block
if showEllipsoid
    CovX_t_MC   = CovMC(4:6,4:6);
    CovX_t_CRLB = CRLB(4:6,4:6);

    figure('Color','w'); hold on; grid on; axis equal;
    title('X translation 3$\sigma$ ellipsoids: MC vs CRLB', 'Interpreter','latex');
    xlabel('$t_x$ (m)', 'Interpreter','latex');
    ylabel('$t_y$ (m)', 'Interpreter','latex');
    zlabel('$t_z$ (m)', 'Interpreter','latex');

    plotEllipsoid3Sigma(CovX_t_CRLB, [0;0;0], 25, [0 0 1], 0.15); % CRLB: blue
    plotEllipsoid3Sigma(CovX_t_MC,   [0;0;0], 25, [1 0 0], 0.15); % MC: red
    legend({'CRLB 3$\sigma$', 'MC 3$\sigma$'}, 'Interpreter','latex');
end

%% ------------------ Local functions ------------------

function [FIM, CRLB, JthetaCells, SigmaCells] = crlb_AXBY_ZCWD(A, B, C, D, X, Y, Z, W, SigmaMeas)
% CRLB for AXBY=ZCWD using r_i = Log( D^{-1} W^{-1} C^{-1} Z^{-1} A X B Y )

N = numel(A);
Dparam = 24;
FIM = zeros(Dparam, Dparam);
JthetaCells = cell(N,1);
SigmaCells  = cell(N,1);

AdW = adjointSE3(W);
AdZ = adjointSE3(Z);

for i = 1:N
    % Factors in T = D^{-1} W^{-1} C^{-1} Z^{-1} A X B Y
    F2 = invSE3(W);
    F3 = invSE3(C{i});
    F4 = invSE3(Z);
    F5 = A{i};
    F6 = X;
    F7 = B{i};
    F8 = Y;

    S1 = F2 * F3 * F4 * F5 * F6 * F7 * F8;
    S2 = F3 * F4 * F5 * F6 * F7 * F8;
    S3 = F4 * F5 * F6 * F7 * F8;
    S4 = F5 * F6 * F7 * F8;
    S5 = F6 * F7 * F8;
    S6 = F7 * F8;

    Ad_invS1 = adjointSE3(invSE3(S1));
    Ad_invS2 = adjointSE3(invSE3(S2));
    Ad_invS3 = adjointSE3(invSE3(S3));
    Ad_invS4 = adjointSE3(invSE3(S4));
    Ad_invS5 = adjointSE3(invSE3(S5));
    Ad_invS6 = adjointSE3(invSE3(S6));
    Ad_Yinv  = adjointSE3(invSE3(Y));

    % Jacobian wrt unknowns theta = [x; y; z; w]
    Jx = Ad_invS6;
    Jy = eye(6);
    Jz = -Ad_invS4 * AdZ;
    Jw = -Ad_invS2 * AdW;

    Jtheta = [Jx, Jy, Jz, Jw];
    JthetaCells{i} = Jtheta;

    % Measurement Jacobians for A,B,C,D noises
    JA = Ad_invS5;
    JB = Ad_Yinv;

    JC = -Ad_invS3 * adjointSE3(C{i});
    JD = -Ad_invS1 * adjointSE3(D{i});

    Sigma_i = JA*SigmaMeas*JA' + JB*SigmaMeas*JB' + JC*SigmaMeas*JC' + JD*SigmaMeas*JD';
    Sigma_i = 0.5 * (Sigma_i + Sigma_i');
    SigmaCells{i} = Sigma_i;

    FIM = FIM + Jtheta' * (Sigma_i \ Jtheta);
end

FIM = 0.5 * (FIM + FIM');
CRLB = pinv(FIM);
CRLB = 0.5 * (CRLB + CRLB');
end

function Tn = addNoiseSE3(T, sigmaRot, sigmaTrans)
xi = [sigmaRot*randn(3,1); sigmaTrans*randn(3,1)];
Tn = T * se3Exp(xi);
end

function r = residualStack_AXBY_ZCWD(A, B, C, D, X, Y, Z, W)
N = numel(A);
r = zeros(6*N,1);
for i = 1:N
    L = A{i} * X * B{i} * Y;
    R = Z * C{i} * W * D{i};
    T = invSE3(R) * L;
    r(6*(i-1)+1:6*i) = se3Log(T);
end
end

function T = randSE3(rotMag, tranMag)
axis = randn(3,1);
axis = axis / max(norm(axis), 1e-12);
ang = rotMag * randn(1);
w = axis * ang;
R = so3Exp(w);
t = tranMag * randn(3,1);
T = [R, t; 0 0 0 1];
end

function Tinv = invSE3(T)
R = T(1:3,1:3);
t = T(1:3,4);
Tinv = [R', -R'*t; 0 0 0 1];
end

function AdT = adjointSE3(T)
R = T(1:3,1:3);
p = T(1:3,4);
AdT = [R, zeros(3,3); skew(p)*R, R];
end

function T = se3Exp(xi)
w = xi(1:3);
v = xi(4:6);
th = norm(w);
W = skew(w);

if th < 1e-12
    R = eye(3) + W;
    V = eye(3) + 0.5 * W;
else
    R = eye(3) + (sin(th)/th)*W + ((1-cos(th))/th^2)*(W*W);
    V = eye(3) + ((1-cos(th))/th^2)*W + ((th - sin(th))/th^3)*(W*W);
end

t = V * v;
T = [R, t; 0 0 0 1];
end

function xi = se3Log(T)
R = T(1:3,1:3);
t = T(1:3,4);

w = so3Log(R);
th = norm(w);
W = skew(w);

if th < 1e-12
    % First-order approximation
    Vinv = eye(3) - 0.5*W + (1/12)*(W*W);
else
    A = sin(th)/th;
    B = (1-cos(th))/th^2;
    % Inverse left Jacobian for SO(3) translation part
    Vinv = eye(3) - 0.5*W + (1/th^2) * (1 - A/(2*B)) * (W*W);
end

v = Vinv * t;
xi = [w; v];
end

function R = so3Exp(w)
th = norm(w);
W = skew(w);
if th < 1e-12
    R = eye(3) + W;
else
    R = eye(3) + (sin(th)/th)*W + ((1-cos(th))/th^2)*(W*W);
end
end

function w = so3Log(R)
tr = trace(R);
c = (tr - 1) / 2;
c = min(1, max(-1, c));
th = acos(c);

if th < 1e-12
    w = 0.5 * vee(R - R');
else
    w = (th / (2*sin(th))) * vee(R - R');
end
end

function v = vee(M)
v = [M(3,2); M(1,3); M(2,1)];
end

function S = skew(a)
S = [  0,   -a(3),  a(2);
      a(3),  0,    -a(1);
     -a(2), a(1),   0  ];
end

function plotEllipsoid3Sigma(Cov3, mu, n, rgb, alphaVal)
% Plot 3-sigma ellipsoid for a 3x3 covariance matrix
Cov3 = 0.5*(Cov3 + Cov3');
[V,D] = eig(Cov3);
d = max(diag(D), 0);
r = 3 * sqrt(d);

% Unit sphere
[xe,ye,ze] = ellipsoid(0,0,0,1,1,1,n);
XYZ = [xe(:)'; ye(:)'; ze(:)'];

% Scale and rotate
S = V * diag(r);
XYZ2 = S * XYZ;

xe2 = reshape(XYZ2(1,:) + mu(1), size(xe));
ye2 = reshape(XYZ2(2,:) + mu(2), size(ye));
ze2 = reshape(XYZ2(3,:) + mu(3), size(ze));

h = surf(xe2, ye2, ze2);
set(h, 'FaceColor', rgb, 'FaceAlpha', alphaVal, 'EdgeColor', 'none');
end
