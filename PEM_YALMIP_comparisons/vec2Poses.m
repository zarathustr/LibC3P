function Xs = vec2Poses(x, P, dim)
Xs = zeros(dim + 1, dim + 1, P);
xx = x(1 : end, :);
length = dim * (dim - 1) / 2 + dim;
for i = 1 : P
    v = xx((i - 1) * length + 1 : i * length, 1);
    Xs(:, :, i) = vec2Pose(v);
end
end