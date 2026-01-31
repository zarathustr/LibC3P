import os
import numpy as np

from cert_tools.eig_tools import get_min_eigpairs

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def rand_sym_mat_test(method="direct", n=5, k=1, tol=1e-8):
    """Test function for testing on random symmetric matrices with distinct
    eigenvalues"""
    # Reset rng
    np.random.seed(0)
    # Define eigenvalues and vectors
    eig_vals = np.random.rand(n) * 2 - 1
    D = np.diag(eig_vals)
    T = np.random.rand(n, n) * 2 - 1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    A = Q @ D @ Q.T
    # Run test
    vals_test, vecs_test = get_min_eigpairs(A, method=method, k=k)
    # Sort actual values and vectors
    sortind = np.argsort(eig_vals)
    vals_true = eig_vals[sortind[:k]]
    vecs_true = Q[:, sortind[:k]]

    # Check eigenvalues
    np.testing.assert_allclose(
        vals_test, vals_true, rtol=0.0, atol=tol, err_msg="Eigenvalues not equal"
    )
    # Flip sign of eigenvector if required
    for i in range(k):
        if not np.sign(vecs_test[0, i]) == np.sign(vecs_true[0, i]):
            vecs_test[:, i] = -vecs_test[:, i]
    # Check eigenvector
    np.testing.assert_allclose(
        vecs_test, vecs_true, rtol=0.0, atol=tol, err_msg="Eigenvectors not equal"
    )


def close_eig_test(method="direct", k=1, tol=1e-8):
    """Test function for testing on random symmetric matrices with distinct
    eigenvalues"""
    # Reset rng
    np.random.seed(0)
    # Define eigenvalues and vectors
    small = np.random.rand(3) * 1e-8
    med = np.random.rand(3) * 1e-3
    large = np.array([1.0, 10.0, 100.0, 500.0])
    eig_vals = np.hstack([0.0, small, large])
    n = len(eig_vals)
    D = np.diag(eig_vals)
    T = np.random.rand(n, n) * 2 - 1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    A = Q @ D @ Q.T
    # Run test
    vals_test, vecs_test = get_min_eigpairs(A, method=method, k=k)
    # Sort actual values and vectors
    sortind = np.argsort(eig_vals)
    vals_true = eig_vals[sortind[:k]]
    vecs_true = Q[:, sortind[:k]]

    # Check eigenvalues
    np.testing.assert_allclose(
        vals_test, vals_true, rtol=0.0, atol=tol, err_msg="Eigenvalues not equal"
    )
    # Flip sign of eigenvector if required
    for i in range(k):
        if not np.sign(vecs_test[0, i]) == np.sign(vecs_true[0, i]):
            vecs_test[:, i] = -vecs_test[:, i]
    # Check eigenvector
    # np.testing.assert_allclose(vecs_test,
    #                            vecs_true,
    #                            rtol = 0.,
    #                            atol=tol,
    #                            err_msg="Eigenvectors not equal"
    #                            )


def test_direct():
    method = "direct"
    rand_sym_mat_test(method=method, n=5, k=2)
    close_eig_test(method=method, k=2)


def test_lanczos():
    method = "lanczos"
    rand_sym_mat_test(method=method, n=5, k=2)
    close_eig_test(method=method, k=2)


def test_lanczos_shifted():
    method = "lanczos-shifted"
    rand_sym_mat_test(method=method, n=5, k=2)
    close_eig_test(method=method, k=2)


if __name__ == "__main__":
    # test_direct()
    test_lanczos()
    test_lanczos_shifted()
