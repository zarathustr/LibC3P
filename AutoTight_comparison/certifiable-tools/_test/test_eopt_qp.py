import numpy as np


def test_eopt_qp():
    """
    Use test problem from Overton 1988
    """
    from cert_tools.eopt_solvers_qp import get_max_eig
    from cert_tools.eopt_solvers_qp import solve_eopt_qp

    def A2(kappa):
        return np.r_[np.c_[1, kappa], np.c_[kappa, 4]]

    Q = np.eye(2)
    A1 = np.r_[np.c_[1.0, 0.0], np.c_[0.0, -1.0]]

    # for large kappa, problem should be easy.
    # this is the example from Overton 1988, for which we know the exact solution.
    kappa = 3.0
    A_list = [A1, A2(kappa)]
    x_init = np.array([1.0, 2.0])

    A_vec = np.hstack([Ai.reshape((-1, 1)) for Ai in A_list])

    eigs, *_ = get_max_eig(Q, A_vec, x_init)
    l_max = eigs[0]
    assert abs(l_max - 12.32) < 0.01

    x_sol, info = solve_eopt_qp(Q, A_vec, xinit=x_init, verbose=2)
    assert info["success"] is True
    np.testing.assert_almost_equal(x_sol, 0.0)

    print("big kappa test passed")

    # for small kappa, problem is unbounded. Most importantly, as stated in
    # Overton 1988, p. 264, there is a valid descent direction at (0, 0)
    kappa = 2.25
    A_list = [A1, A2(kappa)]
    A_vec = -np.hstack([Ai.reshape((-1, 1)) for Ai in A_list])
    x_init = np.zeros(2)
    x_sol, info = solve_eopt_qp(Q, A_vec, xinit=x_init, verbose=2)
    assert info["success"] is False
    print("small kappa test passed")


if __name__ == "__main__":
    test_eopt_qp()
    print("all tests passed")
