import itertools

import numpy as np

from poly_matrix import PolyMatrix


class LeastSquaresProblem(object):
    """Least-squares problem class

    This class implements functionalities specific to least-squares problems, such
    as easily setting up a PolyMatrix instance from residuals etc.

    For usage examples, see `_test/test_cost.py`
    """

    def __init__(self):
        self.m = 0
        self.B = PolyMatrix(symmetric=False)
        self.Q = None

    def get_B_matrix(self, variables, output_type="csc"):
        return self.B.get_matrix(
            variables=([m for m in range(self.m)], variables),
            output_type=output_type,
        )

    def get_Q(self):
        if self.Q is None:
            self.Q = self.B.transpose().multiply(self.B)
        return self.Q

    def add_residual(self, res_dict: dict):
        """Incrementally build Q from residuals. See

        :param res_dict: dictionary storing the what each variable (key) is multiplied with in the residual (value).
        """
        for key, val in res_dict.items():
            self.B[self.m, key] += val
        self.m += 1
        return

        # old implementation directly constructs Q.
        for diag, val in res_dict.items():
            # forbid 1-dimensional arrays cause they are ambiguous.
            assert np.ndim(val) in [
                0,
                2,
            ]
            if np.ndim(val) == 0:
                self[diag, diag] += val**2
            else:
                self[diag, diag] += val @ val.T

        for off_diag_pair in itertools.combinations(res_dict.items(), 2):
            dict0, dict1 = off_diag_pair

            if np.ndim(dict1[1]) > 0:
                new_val = dict0[1] * dict1[1].T
            else:
                new_val = dict0[1] * dict1[1]
            # new value is an array:
            self[dict0[0], dict1[0]] += new_val
