function J = J_func_AXBY_ZCWD(X, Y, Z, W, A, B, C, D)
J = 0;
len = size(A, 3);
for i = 1 : len
    res = A(:, :, i) * X * B(:, :, i) * Y - Z * C(:, :, i) * W * D(:, :, i);
    J = J + 1 / len * trace(res.' * res);
end
end