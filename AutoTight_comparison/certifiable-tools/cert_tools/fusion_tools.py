import scipy.sparse as sp
from mosek.fusion import Matrix


def mat_fusion(X):
    """Convert sparse matrix X to fusion format"""
    try:
        X.eliminate_zeros()
    except AttributeError:
        X = sp.csr_array(X)
    I, J = X.nonzero()
    V = X.data
    return Matrix.sparse(*X.shape, I, J, V)


def get_slice(X: Matrix, i: int):
    (N, X_dim, X_dim) = X.getShape()
    return X.slice([i, 0, 0], [i + 1, X_dim, X_dim]).reshape([X_dim, X_dim])
