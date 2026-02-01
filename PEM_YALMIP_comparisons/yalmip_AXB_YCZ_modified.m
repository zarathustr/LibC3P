clear all
close all
clc

format short g

addpath('sedumi');
if(~verLessThan('matlab', '8.0.0'))
    run(fullfile('sedumi', 'install_sedumi.m'))
end
p = genpath('YALMIP');
addpath(p);

len = 6;
A = zeros(4, 4, len);
B = zeros(4, 4, len);
C = zeros(4, 4, len);

R = orthonormalize(randn(3, 3));
t = randn(3, 1);
X = [R, t;
     zeros(1, 3), 1];
 
R = orthonormalize(randn(3, 3));
t = randn(3, 1);
Y = [R, t;
     zeros(1, 3), 1];
 
R = orthonormalize(randn(3, 3));
t = randn(3, 1);
Z = [R, t;
     zeros(1, 3), 1];
 
noise = 1e-3;
for i = 1 : len
    A(1 : 3, 1 : 3, i) = orthonormalize(randn(3, 3));
    A(1 : 3, 4, i) = randn(3, 1);
    A(4, 4, i) = 1;
    
    C(1 : 3, 1 : 3, i) = orthonormalize(randn(3, 3));
    C(1 : 3, 4, i) = randn(3, 1);
    C(4, 4, i) = 1;
    
    B(:, :, i) = inv(A(:, :, i) * X) * Y * C(:, :, i) * Z;
    
    A(1 : 3, 1 : 3, i) = perturb_rot(A(1 : 3, 1 : 3, i), noise);
    B(1 : 3, 1 : 3, i) = perturb_rot(B(1 : 3, 1 : 3, i), noise);
    C(1 : 3, 1 : 3, i) = perturb_rot(C(1 : 3, 1 : 3, i), noise);
    A(1 : 3, 4, i) = A(1 : 3, 4, i) + noise * randn(3, 1);
    B(1 : 3, 4, i) = B(1 : 3, 4, i) + noise * randn(3, 1);
    C(1 : 3, 4, i) = C(1 : 3, 4, i) + noise * randn(3, 1);
end

% load AXB_YCZ_bad_case1

% AXB = YCZ
% AXBZ' = YC
% Y'AX = CZB'
J_func_AXB_YCZ(X, Y, Z, A, B, C)
J_func_AXB_YCZ(Z, inv(Y), X, C, invs(B), A)
J_func_AXB_YCZ(inv(X), inv(Z), inv(Y), invs(B), invs(A), invs(C))


for i = 1 : 6
    eval(sprintf('R%d = sdpvar(3, 3, ''full'');', i));
    eval(sprintf('t%d = sdpvar(3, 1);', i));
end
XX = [
    R1, t1;
    zeros(1, 3), 1];
XX_ = [
    R2, t2;
    zeros(1, 3), 1];

YY = [
    R3, t3;
    zeros(1, 3), 1];
YY_ = [
    R4, t4;
    zeros(1, 3), 1];

ZZ = [
    R5, t5;
    zeros(1, 3), 1];
ZZ_ = [
    R6, t6;
    zeros(1, 3), 1];

invXX_ = [
    R2.', - R2.' * t2;
    zeros(1, 3), 1];

invYY_ = [
    R4.', - R4.' * t4;
    zeros(1, 3), 1];

invZZ_ = [
    R6.', - R6.' * t6;
    zeros(1, 3), 1];


tic;
J1 = J_func_AXB_YCZ(XX,  YY,  ZZ,  A,        B,        C);        % A X B = Y C Z
J2 = J_func_AXB_YCZ(XX_, ZZ_, YY_, invs(B),  invs(A),  invs(C));  % B^{-1} X^{-1} A^{-1} = Z^{-1} C^{-1} Y^{-1}

J3 = J_func_AXB_YCZ(ZZ,  YY_, XX,  C,        invs(B),  A);        % C Z B^{-1} = Y^{-1} A X
J4 = J_func_AXB_YCZ(ZZ_, XX_, YY,  B,        invs(C),  invs(A));  % B Z^{-1} C^{-1} = X^{-1} A^{-1} Y

J5 = J_func_AXB_YCZ(YY,  XX,  ZZ_, invs(A),  C,        B);        % A^{-1} Y C = X B Z^{-1}
J6 = J_func_AXB_YCZ(YY_, ZZ,  XX_, invs(C),  A,        invs(B));  % C^{-1} Y^{-1} A = Z B^{-1} X^{-1}

f = J1 + J2 + J3 + J4 + J5 + J6;

rho = noise;
F_ = [
    XX == invXX_;
    YY == invYY_;
    ZZ == invZZ_;
    ];
options = sdpsettings('solver', 'fmincon', 'verbose', 1);
sol = optimize(F_, f, options);
toc

X_true = X
X_sdp1 = value(XX)

Y_true = Y
Y_sdp1 = value(YY)

Z_true = Z
Z_sdp1 = value(ZZ)


function R = q2R(q)
q0 = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
R = [
        q0^2 + q1^2 - q2^2 - q3^2,         2*q0*q3 + 2*q1*q2,         2*q1*q3 - 2*q0*q2;
                2*q1*q2 - 2*q0*q3, q0^2 - q1^2 + q2^2 - q3^2,         2*q0*q1 + 2*q2*q3;
                2*q0*q2 + 2*q1*q3,         2*q2*q3 - 2*q0*q1, q0^2 - q1^2 - q2^2 + q3^2];
end