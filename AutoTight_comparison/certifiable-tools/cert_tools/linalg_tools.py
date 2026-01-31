from copy import deepcopy

import numpy as np
import scipy.linalg as la

METHOD = "qrp"
NULL_THRESH = 1e-5


def project_so3(X):
    if X.shape[0] == 4:
        X = deepcopy(X)
        rot = X[:3, :3]
        U, S, Vh = np.linalg.svd(rot)
        rot = U @ Vh
        X[:3, :3] = rot
        return X
    else:
        U, S, Vh = np.linalg.svd(X)
        return U @ Vh


def rank_project(X, p=1, tolerance=1e-10):
    """Project symmetric matrix X to matrix of rank p."""
    assert la.issymmetric(X)
    E, V = np.linalg.eigh(X)
    if p is None:
        p = np.sum(np.abs(E) > tolerance)
    x = V[:, -p:] * np.sqrt(E[-p:])

    X_hat = np.outer(x, x)
    error = np.sum(np.abs(E[:-p]))
    info = {
        "EVR": abs(E[-p]) / abs(E[-p - 1]),
        "error X": np.linalg.norm(X_hat - X),
        "error eigs": error,
        "mean error eigs": error / (X.shape[0] - p),
    }
    return x, info


def find_dependent_columns(A_sparse, tolerance=1e-10, verbose=False, debug=False):
    """
    Returns a list of indices corresponding to the columns of A_sparse that are linearly dependent.
    """
    import sparseqr as sqr

    # Use sparse rank revealing QR
    # We "solve" a least squares problem to get the rank and permutations
    # This is the cheapest way to use sparse QR, since it does not require
    # explicit construction of the Q matrix. We can't do this with qr function
    # because the "just return R" option is not exposed.
    Z, R, E, rank = sqr.rz(
        A_sparse, np.zeros((A_sparse.shape[0], 1)), tolerance=tolerance
    )
    if rank == A_sparse.shape[1]:
        return []

    # Sort the diagonal values. Note that SuiteSparse uses A_sparseMD/METIS ordering
    # to acheive sparsity.
    r_vals = np.abs(R.diagonal())
    sort_inds = np.argsort(r_vals)[::-1]
    if (rank < A_sparse.shape[1]) and verbose:
        print(f"clean_constraints: keeping {rank}/{A_sparse.shape[1]} independent")

    bad_idx = list(range(A_sparse.shape[1]))
    good_idx_list = sorted(E[sort_inds[:rank]])[::-1]
    for good_idx in good_idx_list:
        del bad_idx[good_idx]

    # Sanity check
    if debug:
        Z, R, E, rank_full = sqr.rz(
            A_sparse.tocsc()[:, good_idx_list],
            np.zeros((A_sparse.shape[0], 1)),
            tolerance=tolerance,
        )
        if rank_full != rank:
            print(
                f"Warning: selected constraints did not pass lin. independence check. Rank is {rank_full}, should be {rank}."
            )
    return bad_idx


def get_nullspace(A_dense, method=METHOD, tolerance=NULL_THRESH):
    info = {}

    if method != "qrp":
        print("Warning: method other than qrp is not recommended.")

    if method == "svd":
        U, S, Vh = np.linalg.svd(
            A_dense
        )  # nullspace of A_dense is in last columns of V / last rows of Vh
        rank = np.sum(np.abs(S) > tolerance)
        basis = Vh[rank:, :]
    elif method == "qr":
        # if A_dense.T = QR, the last n-r columns
        # of R make up the nullspace of A_dense.
        Q, R = np.linalg.qr(A_dense.T)
        S = np.abs(np.diag(R))
        sorted_idx = np.argsort(S)[::-1]
        S = S[sorted_idx]
        rank = np.where(S < tolerance)[0][0]
        # decreasing order
        basis = Q[:, sorted_idx[rank:]].T
    elif method == "qrp":
        # Based on Section 5.5.5 "Basic Solutions via QR with Column Pivoting" from Golub and Van Loan.
        # assert A_dense.shape[0] >= A_dense.shape[1], "only tall matrices supported"

        Q, R, P = la.qr(A_dense, pivoting=True, mode="economic")
        if Q.shape[0] < 1e4:
            np.testing.assert_allclose(Q @ R, A_dense[:, P], atol=1e-5)

        S = np.abs(np.diag(R))
        rank = np.sum(S > tolerance)
        R1 = R[:rank, :]
        R11, R12 = R1[:, :rank], R1[:, rank:]
        # [R11  R12]  @  [R11^-1 @ R12] = [R12 - R12]
        # [0    0 ]       [    -I    ]    [0]
        N = np.vstack([la.solve_triangular(R11, R12), -np.eye(R12.shape[1])])

        # Inverse permutation
        Pinv = np.zeros(len(P), int)
        for k, p in enumerate(P):
            Pinv[p] = k
        LHS = R1[:, Pinv]

        info["Q1"] = Q[:, :rank]
        info["LHS"] = LHS

        basis = np.zeros(N.T.shape)
        basis[:, P] = N.T
    else:
        raise ValueError(method)

    # test that it is indeed a null space
    error = A_dense @ basis.T
    info["values"] = S
    info["error"] = error
    return basis, info
