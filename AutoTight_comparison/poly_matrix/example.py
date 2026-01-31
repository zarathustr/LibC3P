#!/usr/bin/env python3
""" Usage example for PolyMatrix class. """ 

from poly_matrix import PolyMatrix
import numpy as np

Q = PolyMatrix()

Q["x1", "x1"] = np.r_[np.c_[1, 2], np.c_[2, 3]]
Q["x1", "x2"] = np.r_[np.c_[3, 4], np.c_[4, 5]]
Q["x1", "z1"] = np.c_[[7, 8]]
Q["x2", "z2"] = np.c_[[9, 10]]
Q["z1", "z1"] = 3
Q["z2", "z2"] = 2
Q["l", "l"] = 1

print("Set up matrix Q:")
print(Q)

A = PolyMatrix()
A["l", "l"] = 1.0
print("Set up matrix A:")
print(A)
print("Augmented matrix:")
A.print(Q.variable_dict)

# create sparse Q matrix
H = Q + A
H_mat = H.get_matrix()
print("Sparse matrix H:")
print(H_mat.toarray())

f_dict = {"x1": [0, 1], "x2": [2, 3], "z1": 1, "z2": 3, "l": 1}
f = Q.get_vector(**f_dict)
cost = f @ H_mat @ f
print("Cost:", cost)
