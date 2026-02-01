function [R, t, X] = SEn_extract(A)
dim = size(A, 1);
R = A(1 : dim - 1, 1 : dim - 1);
t = A(1 : dim - 1, dim);
X = [R, t;
     zeros(1, dim - 1), 1];
end