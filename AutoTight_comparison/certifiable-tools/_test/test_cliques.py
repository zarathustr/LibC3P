import itertools
import os

import numpy as np
from cert_tools.base_clique import BaseClique

from poly_matrix import PolyMatrix

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def generate_random_X(seed=0):
    np.random.seed(seed)
    n_landmarks = 4
    dim_x = 4
    dim_z = 6
    X = PolyMatrix()
    variables = {"h": 1, "x": dim_x}
    variables.update({f"z_{j}": dim_z for j in range(n_landmarks)})
    for pair in itertools.combinations_with_replacement(variables, 2):
        print(pair)
        fill = np.random.rand(variables[pair[0]], variables[pair[1]])
        if pair[0] == pair[1]:
            fill += fill.T
        X[pair[0], pair[1]] = fill
    return X


def test_overlap():
    X = generate_random_X()
    clique_vars = [
        ["h", "x", "z_0"],
        ["h", "x", "z_1"],
        ["h", "x", "z_0", "z_1"],
        ["h", "x", "z_1", "z_2"],
        ["h", "x", "z_0", "z_2"],
    ]
    clique_list = []
    for vars in clique_vars:
        var_dict = {v: X.variable_dict_i[v] for v in vars}
        X_k = X.get_matrix(variables=var_dict, output_type="dense")
        clique_list.append(BaseClique(var_dict=var_dict, X=X_k, Q=None))

    # sanity check
    c0 = clique_list[0]
    assert isinstance(c0, BaseClique)
    assert c0.get_ranges("h") == [[[0], [0]]]
    assert c0.get_ranges("x") == [
        [[0], list(range(1, X.variable_dict_i["x"] + 1))]
    ], c0.get_ranges("x")

    for cl, ck in itertools.combinations(clique_list, 2):
        overlap = BaseClique.get_overlap(cl, ck)
        for l in overlap:
            print(f"testing overlap {l} of {cl}, {ck}")
            for rl, rk in zip(cl.get_ranges(l), ck.get_ranges(l)):
                np.testing.assert_allclose(cl.X[rl[0], rl[1]], ck.X[rk[0], rk[1]])


if __name__ == "__main__":
    test_overlap()
    print("all tests passed.")
