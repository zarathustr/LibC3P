import itertools
import sys
from os.path import dirname

import numpy as np
import pytest

sys.path.append(dirname(__file__) + "/../")
print("appended:", sys.path[-1])

from poly_matrix.least_squares_problem import LeastSquaresProblem

plotting = False
if plotting:
    import matplotlib.pylab as plt


def test_range_only():
    """
    For example, we want to create the following cost:

    $f(x) = sum_{i,j \in \mathcal{E}} (r_{ij} - ||t_j - a_i||)^2$

    We first rewrite each residual as $b_{ij}.T @ x$ where x contains all unknowns and $b_{ij}$ is the coefficient vector.

    """
    N = 2
    K = 3
    D = 2
    noise = 1e-3

    mat_noiseless = LeastSquaresProblem()
    mat_noisy = LeastSquaresProblem()

    np.random.seed(1)
    anchors = np.random.rand(K, D)
    points = np.random.rand(N, D)

    edges = list(itertools.product(range(N), range(K), repeat=1))
    for n, k in edges:
        distance_sq = np.linalg.norm(anchors[k] - points[n]) ** 2
        distance_sq_noisy = distance_sq + np.random.normal(loc=0, scale=noise)
        mat_noiseless.add_residual(
            {"l": distance_sq, f"tau{n}": -1, f"alpha{k}": -1, f"e{n}{k}": 2}
        )
        mat_noisy.add_residual(
            {"l": distance_sq_noisy, f"tau{n}": -1, f"alpha{k}": -1, f"e{n}{k}": 2}
        )

    # below may be provided by custom lifter classes.
    var_dict = {
        "l": 1,
    }
    var_dict.update({f"x{n}": D for n in range(N)})
    var_dict.update({f"a{k}": D for k in range(K)})
    var_dict.update({f"tau{n}": 1 for n in range(N)})
    var_dict.update({f"alpha{k}": 1 for k in range(K)})
    var_dict.update({f"e{n}{k}": 1 for n, k in edges})
    x_data = [[1]]
    x_data += [p for p in points]
    x_data += [a for a in anchors]
    x_data += [[np.linalg.norm(p) ** 2] for p in points]
    x_data += [[np.linalg.norm(a) ** 2] for a in anchors]
    x_data += [[anchors[k] @ points[n]] for n, k in edges]
    x = np.concatenate(x_data)

    B = mat_noiseless.get_B_matrix(var_dict)

    B_dense = B.toarray()
    Q_dense = B_dense.T @ B_dense
    Q = mat_noiseless.get_Q()
    Q_test = Q.toarray(var_dict)

    np.testing.assert_allclose(Q_test, Q_dense)

    if plotting:
        Q.matshow(variables=var_dict)
        plt.show()

    Q_dense = Q.toarray(variables=var_dict)
    err = x.T @ Q_dense @ x
    assert abs(err) < 1e-10, err

    Q = mat_noisy.get_Q().toarray(variables=var_dict)
    err = x.T @ Q @ x
    assert abs(err) < noise, err


if __name__ == "__main__":
    test_range_only()
    print("all tests passed")
