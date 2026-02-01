% Numerical verification for Appendix II (High-order Lie algebra multiplication derivatives)
%
% This script verifies the analytical gradient formula in Appendix II:
%   d/dtheta tr( A (theta^)^p B (theta^)^q C (theta^)^r ... )
% by comparing it against a central finite-difference approximation.
%
% The implementation is for SO(3): theta in R^3 and theta^ is the 3x3 skew matrix.

close all
clear all
clc

% Choose two test cases: a 3-factor and a 5-factor example
cases = struct([]);

cases(1).name   = '3-factor case: tr(A S^p B S^q C S^r)';
cases(1).mats   = {randn(3,3), randn(3,3), randn(3,3)};
cases(1).powers = [2, 3, 1];   % p,q,r in Eq. (65)

cases(2).name   = '5-factor case: tr(A1 S^p1 A2 S^p2 A3 S^p3 A4 S^p4 A5 S^p5)';
cases(2).mats   = {randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)};
cases(2).powers = [2, 2, 1, 3, 2];

% Random theta (avoid tiny theta to reduce degeneracy)
theta = 0.3 * randn(3,1);

% Finite-difference step sweep for a qualitative check
eps_list = logspace(-2, -9, 8);

for c = 1:numel(cases)
    mats   = cases(c).mats;
    powers = cases(c).powers;

    fprintf('\nCase %d: %s\n', c, cases(c).name);
    fprintf('Powers: [%s]\n', num2str(powers));
    fprintf('theta:  [% .4f  % .4f  % .4f]^T\n', theta(1), theta(2), theta(3));

    % Analytical gradient from Appendix II Eq. (65) generalization
    g_an = grad_trace_highorder(theta, mats, powers);

    % Check FD error trend versus epsilon
    rel_err = zeros(size(eps_list));
    abs_err = zeros(size(eps_list));

    for k = 1:numel(eps_list)
        eps_fd = eps_list(k);
        g_fd = grad_fd(theta, mats, powers, eps_fd);

        abs_err(k) = norm(g_fd - g_an, 2);
        rel_err(k) = abs_err(k) / max(1, norm(g_an,2));
    end

    % Report a representative epsilon (middle of the sweep)
    eps_pick = eps_list(round(end/2));
    g_fd_pick = grad_fd(theta, mats, powers, eps_pick);

    fprintf('Analytical grad (Appendix II):   [% .6e  % .6e  % .6e]^T\n', g_an(1), g_an(2), g_an(3));
    fprintf('Finite-diff grad (eps=%1.1e):    [% .6e  % .6e  % .6e]^T\n', eps_pick, g_fd_pick(1), g_fd_pick(2), g_fd_pick(3));
    fprintf('||g_fd - g_an||_2 at eps=%1.1e:  % .3e\n', eps_pick, norm(g_fd_pick-g_an,2));

    % Plot error vs epsilon
    figure('Name', cases(c).name);
    loglog(eps_list, abs_err, '-o', 'LineWidth', 1.5); hold on;
    loglog(eps_list, rel_err, '-s', 'LineWidth', 1.5);
    grid on;
    xlabel('Finite-difference step \epsilon');
    ylabel('Error');
    legend('Absolute error ||g_{fd}-g_{an}||_2', 'Relative error');
    title(['Derivative verification: ', cases(c).name]);
end

fprintf('\nNumerical Done.\n');

dim = 3;
sym_num = 5;
len = dim * (dim - 1) / 2;

str = 'syms';
for i = 1 : len
    str = strcat(str, sprintf(' r%d', i));
end
strcat(str, ' real');
eval(str);

str = 'r = [';
for i = 1 : len
    str = strcat(str, sprintf('r%d,', i));
end
str = strcat(str, '];');
eval(str);

for j = 1 : dim
    for k = 1 : sym_num
        strr = 'syms';
        for i = 1 : dim
            strr = strcat(strr, sprintf(' %c%d%d', char('a' + k - 1), j, i));
        end
        strcat(strr, ' real');
        eval(strr);
    end
end

for k = 1 : sym_num
    str_top = char('A' + k - 1);
    str_top = strcat(str_top, ' = [');
    for i = 1 : dim
        str = '';
        for j = 1 : dim - 1
            str = strcat(str, sprintf('%c%d%d, ', char('a' + k - 1), i, j));
        end
        str = strcat(str, sprintf('%c%d%d;', char('a' + k - 1), i, dim));
        str_top = strcat(str_top, str);
    end
    str_top = strcat(str_top, '];');
    eval(str_top);
end

S = vpa(zeros(dim, dim));
S_idx = skewdec(dim, 1);
for j = 1 : dim
    for i = j + 1 : dim
        idx = abs(S_idx(i, j)) - 1;
        S(j, i) = (-1)^(i - j) * r(len - idx + 1);
        S(i, j) = - (-1)^(i - j) * r(len - idx + 1);
    end
end

cross = jacobian(trace(A * S), r);
assumeAlso(A, 'real')
simplify(cross - wedge(A' - A, dim)')
P = jacobian(jacobian(trace(A * S * B * S * C), r), r);

assumeAlso(A, 'real')
assumeAlso(B, 'real')
assumeAlso(C, 'real')
assumeAlso(D, 'real')
assumeAlso(E, 'real')
assumeAlso(S, 'real')
vvv = jacobian(trace(A * S * B * S * C), r);
assumeAlso(vvv, 'real')
HHH = - (B' * S' * A' * C' + B * S' * C * A);
assumeAlso(HHH, 'real')
bbb = wedge(HHH' - HHH, dim);
simplify(bbb' - vvv)

GGG = equationsToMatrix(bbb == zeros(len, 1), r);
simplify(jacobian(trace(A * S * B), r) - Z_func(B * A, dim)')


k = 3;
HHHH = jacobian(trace(A * S^k), r);
JJJJ = zeros(len, 1);
for i = 0 : k - 1
    KKK = Z_func(S^i * A * S^(k - i - 1), dim);
    JJJJ = JJJJ + KKK;
end
simplify(HHHH - JJJJ.')


GGGG = jacobian(HHHH, r);


HHHH = jacobian(trace(A * S^k * B), r);
JJJJ = zeros(len, 1);
for i = 0 : k - 1
    KKK = Z_func(S^i * B * A * S^(k - i - 1), dim);
    JJJJ = JJJJ + KKK;
end
simplify(HHHH - JJJJ.')

p = 2;
q = 2;
rr = 2;
rrr = 2;
rrrr = 2;
rrrrr = 2;

r_val = randn(len, 1);
for i = 1 : len
    str = sprintf('r%d = r_val(%d);', i, i);
    eval(str);
end
for k = 1 : sym_num
    str = sprintf('%c_val = randn(dim, dim);', char('A' + k - 1));
    eval(str);
    
    for i = 1 : dim
        for j = 1 : dim
            str = sprintf('%c%d%d = %c_val(%d, %d);', char('a' + k - 1), i, j, char('A' + k - 1), i, j);
            eval(str);
        end
    end
end


HHHH = jacobian(trace(A * S^p * B * S^q), r);
JJJJ1 = zeros(len, 1);
for i = 0 : p - 1
    KKK = Z_func(eval(S^i * B * S^q * A * S^(p - i - 1)), dim);
    JJJJ1 = JJJJ1 + KKK;
end
JJJJ2 = zeros(len, 1);
for i = 0 : q - 1
    KKK = Z_func(eval(S^i * A * S^p * B * S^(q - i - 1)), dim);
    JJJJ2 = JJJJ2 + KKK;
end
eval(HHHH - JJJJ1.' - JJJJ2.')


HHHH = jacobian(trace(A * S^p * B * S^q * C * S^rr), r);
JJJJ1 = zeros(len, 1);
for i = 0 : p - 1
    KKK = Z_func(eval(S^i * B * S^q * C * S^rr * A * S^(p - i - 1)), dim);
    JJJJ1 = JJJJ1 + KKK;
end
JJJJ2 = zeros(len, 1);
for i = 0 : q - 1
    KKK = Z_func(eval(S^i * C * S^rr * A * S^p * B * S^(q - i - 1)), dim);
    JJJJ2 = JJJJ2 + KKK;
end
JJJJ3 = zeros(len, 1);
for i = 0 : rr - 1
    KKK = Z_func(eval(S^i * A * S^p * B * S^q * C * S^(rr - i - 1)), dim);
    JJJJ3 = JJJJ3 + KKK;
end
eval(HHHH - JJJJ1.' - JJJJ2.' - JJJJ3.')



HHHH = jacobian(trace(A * S^p * B * S^q * C * S^rr * D * S^rrr), r);
JJJJ1 = zeros(len, 1);
for i = 0 : p - 1
    KKK = Z_func(eval(S^i * B * S^q * C * S^rr * D * S^rrr * A * S^(p - i - 1)), dim);
    JJJJ1 = JJJJ1 + KKK;
end
JJJJ2 = zeros(len, 1);
for i = 0 : q - 1
    KKK = Z_func(eval(S^i * C * S^rr * D * S^rrr * A * S^p * B * S^(q - i - 1)), dim);
    JJJJ2 = JJJJ2 + KKK;
end
JJJJ3 = zeros(len, 1);
for i = 0 : rr - 1
    KKK = Z_func(eval(S^i * D * S ^rrr * A * S^p * B * S^q * C * S^(rr - i - 1)), dim);
    JJJJ3 = JJJJ3 + KKK;
end
JJJJ4 = zeros(len, 1);
for i = 0 : rrr - 1
    KKK = Z_func(eval(S^i * A * S^p * B * S^q * C * S^rr * D * S^(rrr - i - 1)), dim);
    JJJJ4 = JJJJ4 + KKK;
end
eval(HHHH - JJJJ1.' - JJJJ2.' - JJJJ3.' - JJJJ4.')


HHHH = jacobian(trace(A * S^p * B * S^q * C * S^rr * D * S^rrr * E * S^rrrr), r);
JJJJ1 = zeros(len, 1);
for i = 0 : p - 1
    KKK = Z_func(eval(S^i * B * S^q * C * S^rr * D * S^rrr * E * S^rrrr * A * S^(p - i - 1)), dim);
    JJJJ1 = JJJJ1 + KKK;
end
JJJJ2 = zeros(len, 1);
for i = 0 : q - 1
    KKK = Z_func(eval(S^i * C * S^rr * D * S^rrr * E * S^rrrr * A * S^p * B * S^(q - i - 1)), dim);
    JJJJ2 = JJJJ2 + KKK;
end
JJJJ3 = zeros(len, 1);
for i = 0 : rr - 1
    KKK = Z_func(eval(S^i * D * S ^rrr * E * S^rrrr * A * S^p * B * S^q * C * S^(rr - i - 1)), dim);
    JJJJ3 = JJJJ3 + KKK;
end
JJJJ4 = zeros(len, 1);
for i = 0 : rrr - 1
    KKK = Z_func(eval(S^i * E * S^rrrr * A * S^p * B * S^q * C * S^rr * D * S^(rrr - i - 1)), dim);
    JJJJ4 = JJJJ4 + KKK;
end
JJJJ5 = zeros(len, 1);
for i = 0 : rrrr - 1
    KKK = Z_func(eval(S^i * A * S^p * B * S^q * C * S^rr * D * S^rrr * E * S^(rrrr - i - 1)), dim);
    JJJJ5 = JJJJ5 + KKK;
end
eval(HHHH - JJJJ1.' - JJJJ2.' - JJJJ3.' - JJJJ4.' - JJJJ5.')

fprintf('\n Symbolic Done.\n');


function S = hat3(w)
    S = [   0   -w(3)  w(2);
          w(3)   0    -w(1);
         -w(2)  w(1)   0   ];
end

function v = vee3(S)
    v = [S(3,2); S(1,3); S(2,1)];
end

function z = Z3(M)
    z = vee3(M.' - M);
end

function f = obj_trace_highorder(theta, mats, powers)
    S = hat3(theta);
    m = numel(mats);
    Pmax = max(powers);
    Spow = cell(Pmax+1,1);
    Spow{1} = eye(3);
    for k = 1:Pmax
        Spow{k+1} = Spow{k} * S;
    end
    T = eye(3);
    for j = 1:m
        T = T * mats{j} * Spow{powers(j)+1};
    end
    f = trace(T);
end

function g = grad_trace_highorder(theta, mats, powers)
    S = hat3(theta);
    m = numel(mats);

    Pmax = max(powers);
    Spow = cell(Pmax+1,1);
    Spow{1} = eye(3);
    for k = 1:Pmax
        Spow{k+1} = Spow{k} * S;
    end

    g = zeros(3,1);

    for l = 1:m
        pl = powers(l);
        for i = 0:(pl-1)
            M = Spow{i+1};

            idx = [ (l+1):m, 1:(l-1) ];
            for jj = idx
                M = M * mats{jj} * Spow{powers(jj)+1};
            end

            M = M * mats{l} * Spow{(pl - i - 1) + 1};

            g = g + Z3(M);
        end
    end
end

function g = grad_fd(theta, mats, powers, eps_fd)
    g = zeros(3,1);
    for k = 1:3
        e = zeros(3,1); e(k) = 1;
        fp = obj_trace_highorder(theta + eps_fd*e, mats, powers);
        fm = obj_trace_highorder(theta - eps_fd*e, mats, powers);
        g(k) = (fp - fm) / (2*eps_fd);
    end
end