function out = reverseInvAs(As)
out = As;
N = size(As, 4);
for i = 1 : N
    out(:, :, :, i) = reverseInvXs(As(:, :, :, i));
end
end
