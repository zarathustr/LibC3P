function xx = Poses2vec(Xs, P, dim)
length = dim * (dim - 1) / 2 + dim;
xx = zeros(P * length, 1);
for i = 1 : P
    X = Xs(:, :, i);
    R = X(1 : dim, 1 : dim);
    t = X(1 : dim, dim + 1);
    x = [wedge(logm(R), dim); t];
    xx((i - 1) * length + 1 : i * length, :) = x;
end
end
