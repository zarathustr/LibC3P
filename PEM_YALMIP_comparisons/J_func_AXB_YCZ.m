function J = J_func_AXB_YCZ(X, Y, Z, A, B, C)
J = 0;
len = size(A, 3);
for i = 1 : len
    res = A(:, :, i) * X * B(:, :, i) - Y * C(:, :, i) * Z;
    J = J + 1 / len * trace(res.' * res);
end
end