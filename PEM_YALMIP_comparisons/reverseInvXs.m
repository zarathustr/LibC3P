function out = reverseInvXs(Xs)
out = Xs;
num = size(Xs, 3);
for i = 1 : num
    out(:, :, i) = inv(Xs(:, :, num - i + 1));
end
end

