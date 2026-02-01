function x = Z_func(A, dim)
if(dim == 3)
    x = [A(2, 3) - A(3, 2);
         A(3, 1) - A(1, 3);
         A(1, 2) - A(2, 1)];
elseif(dim == 4)
    x = [A(3, 4) - A(4, 3);
         A(4, 2) - A(2, 4);
         A(1, 4) - A(4, 1); 
         Z_func(A, 3);
         ];
else
    x = wedge(A' - A, dim);
end
end