function out = reverseXs(Xs)
out = Xs;
num = size(Xs, 3);
for i = 1 : num
    out(:, :, i) = (Xs(:, :, num - i + 1));
end
end

