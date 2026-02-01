clear all
close all
clc

format short g

len = 15;
A = zeros(4, 4, len);
B = zeros(4, 4, len);
C = zeros(4, 4, len);
D = zeros(4, 4, len);

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
 
R = orthonormalize(randn(3, 3));
t = randn(3, 1);
W = [R, t;
     zeros(1, 3), 1];
 
noise = 1e-3;
for i = 1 : len
    A(1 : 3, 1 : 3, i) = orthonormalize(randn(3, 3));
    A(1 : 3, 4, i) = randn(3, 1);
    A(4, 4, i) = 1;
    
    B(1 : 3, 1 : 3, i) = orthonormalize(randn(3, 3));
    B(1 : 3, 4, i) = randn(3, 1);
    B(4, 4, i) = 1;
    
    C(1 : 3, 1 : 3, i) = orthonormalize(randn(3, 3));
    C(1 : 3, 4, i) = randn(3, 1);
    C(4, 4, i) = 1;
    
    D(:, :, i) = inv(Z * C(:, :, i) * W) * A(:, :, i) * X * B(:, :, i) * Y;
    
    A(1 : 3, 1 : 3, i) = perturb_rot(A(1 : 3, 1 : 3, i), noise);
    B(1 : 3, 1 : 3, i) = perturb_rot(B(1 : 3, 1 : 3, i), noise);
    C(1 : 3, 1 : 3, i) = perturb_rot(C(1 : 3, 1 : 3, i), noise);
    D(1 : 3, 1 : 3, i) = perturb_rot(D(1 : 3, 1 : 3, i), noise);
    A(1 : 3, 4, i) = A(1 : 3, 4, i) + noise * randn(3, 1);
    B(1 : 3, 4, i) = B(1 : 3, 4, i) + noise * randn(3, 1);
    C(1 : 3, 4, i) = C(1 : 3, 4, i) + noise * randn(3, 1);
    D(1 : 3, 4, i) = D(1 : 3, 4, i) + noise * randn(3, 1);
end

x0 = randn(14 * 4, 1);
J_func_AXBY_ZCWD(X, Y, Z, W, A, B, C, D)
J_func_AXBY_ZCWD(inv(W), inv(Z), inv(Y), inv(X), invs(D), invs(C), invs(B), invs(A))


for i = 1 : 8
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

WW = [
    R7, t7;
    zeros(1, 3), 1];
WW_ = [
    R8, t8;
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


invWW_ = [
    R8.', - R8.' * t8;
    zeros(1, 3), 1];


tic;
J1 = J_func_AXBY_ZCWD(XX,  YY,  ZZ,  WW,  A,        B,        C,        D);        % A X B Y = Z C W D
J2 = J_func_AXBY_ZCWD(WW_, ZZ_, YY_, XX_, invs(D),  invs(C),  invs(B),  invs(A));  % D^{-1} W^{-1} C^{-1} Z^{-1} = Y^{-1} B^{-1} X^{-1} A^{-1}

J3 = J_func_AXBY_ZCWD(ZZ,  WW,  XX,  YY,  invs(A),  C,        B,        invs(D));  % A^{-1} Z C W = X B Y D^{-1}
J4 = J_func_AXBY_ZCWD(YY_, XX_, WW_, ZZ_, D,        invs(B),  invs(C),  A);        % D Y^{-1} B^{-1} X^{-1} = W^{-1} C^{-1} Z^{-1} A

J5 = J_func_AXBY_ZCWD(ZZ_, XX,  WW,  YY_, invs(C),  A,        D,        invs(B));  % C^{-1} Z^{-1} A X = W D Y^{-1} B^{-1}
J6 = J_func_AXBY_ZCWD(YY,  WW_, XX_, ZZ,  B,        invs(D),  invs(A),  C);        % B Y D^{-1} W^{-1} = X^{-1} A^{-1} Z C

J7 = J_func_AXBY_ZCWD(XX_, ZZ,  YY,  WW_, invs(B),  invs(A),  invs(D),  invs(C));  % B^{-1} X^{-1} A^{-1} Z = Y D^{-1} W^{-1} C^{-1}
J8 = J_func_AXBY_ZCWD(WW,  YY_, ZZ_, XX,  C,        D,        A,        B);        % C W D Y^{-1} = Z^{-1} A X B
f = J1 + J2 + J3 + J4 + J5 + J6 + J7 + J8;

rho = noise;
F_ = [
    XX == invXX_;
    YY == invYY_;
    ZZ == invZZ_;
    WW == invWW_;
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

W_true = W
W_sdp1 = value(WW)



function R = q2R(q)
q0 = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
R = [
        q0^2 + q1^2 - q2^2 - q3^2,         2*q0*q3 + 2*q1*q2,         2*q1*q3 - 2*q0*q2;
                2*q1*q2 - 2*q0*q3, q0^2 - q1^2 + q2^2 - q3^2,         2*q0*q1 + 2*q2*q3;
                2*q0*q2 + 2*q1*q3,         2*q2*q3 - 2*q0*q1, q0^2 - q1^2 - q2^2 + q3^2];
end