function Rp = perturb_rot(R, noise)
dim = size(R, 1);
theta = randn(dim * (dim - 1) / 2, 1) * noise;
thetax = times_(theta, dim);
p = expm(thetax);
Rp = R * p;
end