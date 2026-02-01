function X = vec2Pose(v)
dim = (sqrt(size(v, 1) * 8 + 1) - 1) / 2;
x = v(1 : dim * (dim - 1) / 2, 1);
t = v(end - dim + 1 : end, 1);
X = [expm(times_(x, dim)), t;
     zeros(1, dim), 1];
end