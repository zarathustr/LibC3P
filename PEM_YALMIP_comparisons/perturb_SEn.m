function out = perturb_SEn(A, noiseR, noiseT)
len = size(A, 3);
out = A;
dim = size(A, 1) - 1;
for i = 1 : len
    R = A(1 : dim, 1 : dim, i);
    t = A(1 : dim, dim + 1, i);
    R = orthonormalize(randn(dim, dim) * noiseR + eye(dim)) * R;
    t = randn(dim, 1) * noiseT + t;
    X = [R, t;
         zeros(1, dim), 1];
    out(:, :, i) = X;
    
end
end