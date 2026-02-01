function x = vec(X)
x = reshape(X, [size(X, 1) * size(X, 2), 1]);
end