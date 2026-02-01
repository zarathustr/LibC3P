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
R = orthonormalize(randn(3, 3));
t = randn(3, 1);
X = [R, t;
     zeros(1, 3), 1];

noise = 1e-3;
for i = 1 : len
    A(1 : 3, 1 : 3, i) = orthonormalize(randn(3, 3));
    A(1 : 3, 4, i) = randn(3, 1);
    A(4, 4, i) = 1;
    B(:, :, i) = inv(X) * A(:, :, i) * X;
    
    A(1 : 3, 1 : 3, i) = perturb_rot(A(1 : 3, 1 : 3, i), noise);
    B(1 : 3, 1 : 3, i) = perturb_rot(B(1 : 3, 1 : 3, i), noise);
    A(1 : 3, 4, i) = A(1 : 3, 4, i) + noise * randn(3, 1);
    B(1 : 3, 4, i) = B(1 : 3, 4, i) + noise * randn(3, 1);
end


J_func(X, A, B)
RX = sdpvar(3, 3, 'full');
tX = sdpvar(3, 1);
XX = [RX, tX;
     zeros(1, 3), 1];
ineqX = [
    eye(3), RX.'; 
    RX, eye(3);
    ];

F = blkdiag(ineqX) >= 0;
f = J_func(XX, A, B);

options = sdpsettings('solver', 'lmilab', 'verbose', 1, 'debug', 1);
sol = optimize(F, f, options);


XX_ = value(XX);
clear RX tX XX

RX = sdpvar(3, 3, 'full');
tX = sdpvar(3, 1);
XX = [RX, tX;
     zeros(1, 3), 1];
ineqX = [
    eye(3), RX.'; 
    RX, eye(3);
    ];

RZ = sdpvar(3, 3, 'full');
tZ = sdpvar(3, 1);
ZZ = [RZ, tZ;
     zeros(1, 3), 1];
ineqZ = [
    eye(3), RZ.'; 
    RZ, eye(3);
    ];
 
% AX = XB
% B * inv(X) = inv(X) * A
% inv(A) * X = X * inv(B)
% inv(B) * inv(X) = inv(X) * inv(A)
J1 = J_func(XX, A, B);
J2 = J_func(ZZ, B, A);
J3 = J_func(XX, invs(A), invs(B));
J4 = J_func(ZZ, invs(B), invs(A));

f = J1 + J2 + J3 + J4;
tW = sdpvar(3, 1);
tW = - RZ.' * tZ;
invZ = [
    RZ.', tW;
    zeros(1, 3), 1];

rho = noise;
F_ = XX == invZ;
options = sdpsettings('solver', 'fmincon', 'verbose', 1, 'debug', 1);
tic;
sol = optimize(F_, f, options);
toc
X_sdp = XX_
X_true = X
X_sdp1 = inv(value(ZZ))



function R = q2R(q)
q0 = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
R = [
        q0^2 + q1^2 - q2^2 - q3^2,         2*q0*q3 + 2*q1*q2,         2*q1*q3 - 2*q0*q2;
                2*q1*q2 - 2*q0*q3, q0^2 - q1^2 + q2^2 - q3^2,         2*q0*q1 + 2*q2*q3;
                2*q0*q2 + 2*q1*q3,         2*q2*q3 - 2*q0*q1, q0^2 - q1^2 - q2^2 + q3^2];
end

function J = J_func(X, A, B)
J = 0;
len = size(A, 3);
for i = 1 : len
    res = A(:, :, i) * X - X * B(:, :, i);
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