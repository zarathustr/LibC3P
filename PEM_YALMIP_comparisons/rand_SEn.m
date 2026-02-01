function X = rand_SEn(n, m)
if(m == 1)
    R = orthonormalize(randn(n, n));
    t = randn(n, 1);
    X = [R, t;
         zeros(1, n), 1];
else
    X = zeros(n + 1, n + 1, m);
    for i = 1 : m
        R = orthonormalize(randn(n, n));
        t = randn(n, 1);
        XX = [R, t;
              zeros(1, n), 1];
        X(:, :, i) = XX;
    end
end
end