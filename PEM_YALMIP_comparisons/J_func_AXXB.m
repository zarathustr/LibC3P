function J = J_func_AXXB(X, A, B)
J = 0;
len = size(A, 3);
for j = 1 : len
    XA = A(:, :, j);
    XB = B(:, :, j);
    res = XA * X - X * XB;
    J = J + 1 / len * trace(res' * res);
end
end