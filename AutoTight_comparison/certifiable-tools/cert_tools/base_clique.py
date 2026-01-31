import numpy as np

OVERLAP_ALL = False


class BaseClique(object):
    @staticmethod
    def get_overlap(cl, ck, h="h"):
        return set(cl.var_dict.keys()).intersection(ck.var_dict.keys()).difference(h)

    def __init__(
        self,
        Q,
        A_list=[],
        b_list=[],
        left_slice_start=[],
        left_slice_end=[],
        right_slice_start=[],
        right_slice_end=[],
        var_dict=None,
        X=None,
        index=0,
        x_dim=None,
        hom="h",
    ):
        assert Q is not None or X is not None
        self.Q = Q

        self.A_list = A_list
        self.b_list = b_list
        self.left_slice_start = left_slice_start
        self.left_slice_end = left_slice_end
        self.right_slice_start = right_slice_start
        self.right_slice_end = right_slice_end

        if var_dict is not None:
            assert hom in var_dict, f"Each clique must have a {hom}."
        self.var_dict = var_dict
        self.var_start_index = None
        self.X = X

        # dimension of full clique
        self.X_dim = Q.shape[0] if Q is not None else X.shape[0]

        # dimension of each node inside clique
        self.x_dim = x_dim

        self.index = index
        self.hom = hom

    def get_ranges(self, j_key: str, i_key: str = None):
        """Return the index range of var_key.

        :param var_key: name of overlapping  variable
        """
        if i_key is None:
            i_key = self.hom

        if self.var_start_index is None:
            self.var_start_index = dict(
                zip(self.var_dict.keys(), np.cumsum([0] + list(self.var_dict.values())))
            )

        i_range = list(
            range(
                self.var_start_index[i_key],
                self.var_start_index[i_key] + self.var_dict[i_key],
            )
        )
        j_range = list(
            range(
                self.var_start_index[j_key],
                self.var_start_index[j_key] + self.var_dict[j_key],
            )
        )
        if OVERLAP_ALL:
            return [[i_range, j_range]] + [[i_range, j_range]]
        else:
            return [[i_range, j_range]]

    def get_Qij(self, key_i, key_j):
        ranges = self.get_ranges(j_key=key_j, i_key=key_i)[0]
        return self.Q.toarray()[np.ix_(ranges[0], ranges[1])]

    def get_Aij_agg(self, key_i, key_j):
        ranges = self.get_ranges(j_key=key_j, i_key=key_i)[0]
        Aij_agg = np.zeros((self.var_dict[key_i], self.var_dict[key_j]))
        for A in self.A_list:
            Aij_agg += np.abs(A.toarray()[np.ix_(ranges[0], ranges[1])]) > 1e-10
        return Aij_agg > 0

    def __repr__(self):
        vars_pretty = tuple(self.var_dict.keys()) if self.var_dict is not None else ()
        return f"clique var_dict={vars_pretty}"
