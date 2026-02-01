function J = J_func_AXYB(X, Y, A, B)
J = 0;
len = size(A, 3);
for j = 1 : len
    XA = A(:, :, j);
    XB = B(:, :, j);
    res = XA * X - Y * XB;
    J = J + 1 / len * trace(res' * res);
end
end