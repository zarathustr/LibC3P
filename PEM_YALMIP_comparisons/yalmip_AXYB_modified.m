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

len = 3;
A = zeros(4, 4, len);
B = zeros(4, 4, len);

level = 5e-1;

R = orthonormalize(eye(3) + level * randn(3, 3));
t = randn(3, 1);
X = [R, t;
     zeros(1, 3), 1];
 
R = orthonormalize(eye(3) + level * randn(3, 3));
t = randn(3, 1);
Y = [R, t;
     zeros(1, 3), 1];

noise = 1e-4;
for i = 1 : len
    A(1 : 3, 1 : 3, i) = orthonormalize(randn(3, 3));
    A(1 : 3, 4, i) = randn(3, 1);
    A(4, 4, i) = 1;
    B(:, :, i) = inv(Y) * A(:, :, i) * X;
    
    A(1 : 3, 1 : 3, i) = perturb_rot(A(1 : 3, 1 : 3, i), noise);
    B(1 : 3, 1 : 3, i) = perturb_rot(B(1 : 3, 1 : 3, i), noise);
    A(1 : 3, 4, i) = A(1 : 3, 4, i) + noise * randn(3, 1);
    B(1 : 3, 4, i) = B(1 : 3, 4, i) + noise * randn(3, 1);
end


J_func(X, Y, A, B)
RX = sdpvar(3, 3, 'full');
tX = sdpvar(3, 1);
XX = [RX, tX;
     zeros(1, 3), 1];
ineqX = [
    eye(3), RX.'; 
    RX, eye(3);
    ];

RY = sdpvar(3, 3, 'full');
tY = sdpvar(3, 1);
YY = [RY, tY;
     zeros(1, 3), 1];
ineqY = [
    eye(3), RY.'; 
    RY, eye(3);
    ];

F = blkdiag(ineqX, ineqY) >= 0;
f = J_func(XX, YY, A, B);

options = sdpsettings('solver', 'lmilab', 'verbose', 1);
tic;
sol = optimize(F, f, options);
toc


XX_ = value(XX);
YY_ = value(YY);
clear RX tX XX RY tY YY

RX = sdpvar(3, 3, 'full');
tX = sdpvar(3, 1);
XX = [RX, tX;
     zeros(1, 3), 1];
ineqX = [
    eye(3), RX.'; 
    RX, eye(3);
    ];

RY = sdpvar(3, 3, 'full');
tY = sdpvar(3, 1);
YY = [RY, tY;
     zeros(1, 3), 1];
ineqY = [
    eye(3), RY.'; 
    RY, eye(3);
    ];

RZ = sdpvar(3, 3, 'full');
tZ = sdpvar(3, 1);
ZZ = [RZ, tZ;
     zeros(1, 3), 1];
ineqZ = [
    eye(3), RZ.'; 
    RZ, eye(3);
    ];

RW = sdpvar(3, 3, 'full');
tW = sdpvar(3, 1);
WW = [RW, tW;
     zeros(1, 3), 1];
ineqW = [
    eye(3), RW.'; 
    RW, eye(3);
    ];
 
% AX = YB
% B * inv(X) = inv(Y) * A
% inv(A) * Y = X * inv(B)
% inv(B) * inv(Y) = inv(X) * inv(A)
J1 = J_func(XX, YY, A, B);
J2 = J_func(WW, ZZ, B, A);
J3 = J_func(YY, XX, invs(A), invs(B));
J4 = J_func(ZZ, WW, invs(B), invs(A));
f = J1 + J2 + J3 + J4;
invZ = [
    RZ.', - RZ.' * tZ;
    zeros(1, 3), 1];

invW = [
    RW.', - RW.' * tW;
    zeros(1, 3), 1];

rho = 1e-2;
F_ = [
    XX == invW;
    YY == invZ;
    ];
options = sdpsettings('solver', 'fmincon', 'verbose', 1);
tic;
sol = optimize(F_, f, options);
toc

X_sdp_lmi = XX_
X_true = X
X_sdp_fmincon = value(XX)

Y_sdp_lmi = YY_
Y_true = Y
Y_sdp_fmincon = value(YY)



function R = q2R(q)
q0 = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
R = [
        q0^2 + q1^2 - q2^2 - q3^2,         2*q0*q3 + 2*q1*q2,         2*q1*q3 - 2*q0*q2;
                2*q1*q2 - 2*q0*q3, q0^2 - q1^2 + q2^2 - q3^2,         2*q0*q1 + 2*q2*q3;
                2*q0*q2 + 2*q1*q3,         2*q2*q3 - 2*q0*q1, q0^2 - q1^2 - q2^2 + q3^2];
end

function J = J_func(X, Y, A, B)
J = 0;
len = size(A, 3);
for i = 1 : len
    res = A(:, :, i) * X - Y * B(:, :, i);
    J = J + 1 / len * trace(res.' * res);
end
end

function A_ = invs(A)
len = size(A, 3);
A_ = zeros(4, 4, len);
for i = 1 : len
    A_(:, :, i) = inv(A(:, :, i));
end
end