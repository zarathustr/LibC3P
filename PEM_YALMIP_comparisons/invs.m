function A_ = invs(A)
len = size(A, 3);
A_ = zeros(4, 4, len);
for i = 1 : len
    A_(:, :, i) = inv(A(:, :, i));
end
end