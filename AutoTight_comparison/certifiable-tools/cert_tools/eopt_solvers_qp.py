import cvxpy as cp
import mosek
import numpy as np
from cert_tools.eig_tools import get_min_eigpairs
from cert_tools.eopt_solvers import (
    backtrack_cutoff,
    backtrack_factor,
    backtrack_start,
    get_cert_mat,
)
from cert_tools.sdp_solvers import options_cvxpy

# tolerance for minimum eigevalue: mineig >= -TOL_EIG <=> A >= 0
TOL_EIG = 1e-10


def get_min_multiplicity(eigs, tau):
    "See equations (22), Overton 1992."
    eig_min = eigs[0]
    assert eig_min == np.min(eigs), "eigenvalues not sorted!"

    diff = eigs - eig_min - tau * max(1, abs(eig_min))
    # t is where diff_t <= 0, diff_{t+1} >= 0
    valid_idx = np.argwhere(diff >= 0)
    if len(valid_idx):
        # t_p1 = int(valid_idx[-1])
        t_p1 = int(valid_idx[0])
    else:
        t_p1 = len(eigs)
    assert diff[t_p1 - 1] <= 0
    return t_p1


def solve_d_from_indefinite_U(U, Q_1, A_vec, verbose=False):
    """
    Solve (17) from Overton 1992 to find a descent direciton d
    in case dual matrix U is indefinite.
    """
    m = A_vec.shape[1]
    n, t = Q_1.shape
    eig, vec = get_min_eigpairs(U, method="lanczos", k=1)

    d = cp.Variable(m)
    delta = cp.Variable()
    constraint = [
        cp.sum(
            [
                d[k] * Q_1.T @ A_vec[:, k].reshape((n, n), order="F") @ Q_1
                for k in range(m)
            ]
        )
        - delta * np.eye(t)
        == -vec @ vec.T
    ]
    prob = cp.Problem(cp.Minimize(1), constraints=constraint)

    try:
        prob.solve(solver="MOSEK", verbose=verbose > 2, **options_cvxpy)
    except mosek.Error:
        print("Did not find MOSEK, using different solver.")
        prob.solve(solver="CVXOPT", verbose=verbose > 2)

    success = d.value is not None
    info = {"msg": prob.status, "success": success, "delta": delta.value}
    return d.value, info


def solve_inner_QP(vecs, eigs, A_vec, t, rho, W, verbose=False, lmin=False):
    """
    Solve the direction-finding QP (Overton 1992, equations (24) - (27)).

    vecs and eigs are the eig-pairs at a current estimate, and t is the estimated multiplicity
    of the biggest one.

    """
    Q_1 = vecs[:, :t]
    lambdas = eigs[:t]
    eig_max = lambdas[0]
    n_eig = len(eigs)
    n = vecs.shape[0]
    m = A_vec.shape[1]

    d = cp.Variable(m)
    delta = cp.Variable()

    # create t(t+1)/2 constraints
    constraints = []
    rhs = np.diag(lambdas - eig_max)
    lhs = delta * np.eye(t) - cp.sum(
        [d[k] * Q_1.T @ A_vec[:, k].reshape((n, n), order="F") @ Q_1 for k in range(m)]
    )
    # constraints.append(rhs A_list== lhs)
    for i in range(t):
        for j in range(i, t):
            constraints.append(lhs[i, j] == rhs[i, j])
    constraints += [
        delta
        - cp.sum(
            [
                d[k]
                * vecs[:, i].T
                @ A_vec[:, k].reshape((n, n), order="F")
                @ vecs[:, i]
                for k in range(m)
            ]
        )
        >= eigs[i] - eig_max
        for i in range(t, n_eig)
    ]
    constraints += [cp.norm_inf(d) <= rho]

    prob = cp.Problem(cp.Minimize(delta + d.T @ W @ d), constraints=constraints)
    try:
        prob.solve(solver="MOSEK", verbose=verbose > 2, **options_cvxpy)
    except mosek.Error:
        print("Did not find MOSEK, using different solver.")
        prob.solve(solver="CVXOPT", verbose=verbose > 2)

    success = d.value is not None
    if success:
        # U = constraints[0].dual_value
        U = np.zeros((t, t))
        k = 0
        for i in range(t):
            for j in range(i, t):
                if i == j:
                    U[i, j] = constraints[k].dual_value
                else:
                    U[j, i] = constraints[k].dual_value / 2
                    U[i, j] = constraints[k].dual_value / 2
                k += 1
        eigs = np.linalg.eigvalsh(U)

        if np.all(eigs <= 0):
            U = -U
        d = d.value
        delta = delta.value
    else:
        print("Warning: didn't find feasible direction.")
        U = None
        d = None
        delta = None

    info = {"success": success, "msg": prob.status, "delta": delta}
    return U, d, info


def compute_current_W(vecs, eigs, A_vec, t, w):
    Q_1 = vecs[:, :t]
    n = Q_1.shape[0]
    m = A_vec.shape[1]
    U = cp.Variable((t, t), symmetric=True)
    constraints = [
        U >> 0,
    ]
    obj = cp.Minimize(
        cp.norm2(
            cp.trace(U)
            - 1
            + cp.sum(
                [
                    cp.trace(Q_1.T @ A_vec[:, k].reshape((n, n), order="F") @ Q_1 @ U)
                    for k in range(m)
                ]
            )
        )
    )
    prob = cp.Problem(obj, constraints=constraints)
    sol = prob.solve()

    U_est = U.value

    L_bar = np.diag(eigs[t:])
    Q_1_bar = vecs[:, t:]

    W = np.empty((m, m))
    for j in range(m):
        for k in range(j, m):
            G_jk = (
                2
                * Q_1.T
                @ A_vec[:, k].reshape((n, n), order="F")
                @ Q_1_bar
                @ np.diag([1 / (w - L_bar[i]) for i in range(L_bar.shape[0])])
                @ Q_1_bar.T
                @ A_vec[:, j].reshape((n, n), order="F")
                @ Q_1
            )
            if t == 1:
                W_jk = G_jk * U_est
            else:
                W_jk = np.trace(U_est @ G_jk)
            W[j, k] = W[k, j] = W_jk
    return W


def get_max_eig(Q, A_vec, x_new, tau=1e-8):
    n = Q.shape[0]
    k = min(n, 5)
    method = "direct" if k == n else "lanczos"

    H = get_cert_mat(Q, A_vec, x_new)
    eigs, vecs = get_min_eigpairs(-H, k=k, method=method)
    t = get_min_multiplicity(eigs, tau)
    return -eigs, -vecs, t


def solve_eopt_qp(
    Q,
    A_vec,
    xinit=None,
    max_iters=1000,
    gtol=1e-10,
    verbose=1,
    tau=1e-5,
    l_threshold=None,
):
    """Solve E_OPT: min_x sigma_max (Q + sum_i (x_i * A_i)), using the QP algorithm
                                     ----------H(x)---------
    provided by Overton 1992 (adaptation of Overton 1988).
    """
    # TODO: convert this to max_x sigma_min (H(x))
    m = A_vec.shape[1]
    n = Q.shape[0]

    if xinit is None:
        xinit = np.zeros(m)

    # trust region radios
    rho = 1.0

    # weight matrix for QP
    W = np.zeros((m, m))
    U = None

    x = xinit

    i = 0
    while i <= max_iters:
        eigs, vecs, t = get_max_eig(Q, A_vec, x, tau)

        if i == 0 and verbose > 1:
            print(f"start \t eigs {eigs.round(2)} \t t {t} \t \t lambda {eigs[0]:.4e}")

        Q_1 = vecs[:, :t]
        if t == 1:
            grad = np.concatenate(
                [Q_1.T @ A_i.reshape((n, n), order="F") @ Q_1 for A_i in A_vec.T]
            ).flatten()
            d = -grad / np.linalg.norm(grad)

            # backgracking, using Nocedal Algorithm 3.1
            alpha = backtrack_start

            l_old = eigs[0]
            while np.linalg.norm(alpha * d) > gtol:
                eigs, *_ = get_max_eig(Q, A_vec, x + alpha * d, tau)
                l_new = eigs[0]

                # trying to minimize l_max
                l_test = l_old + backtrack_cutoff * alpha * grad.T @ d
                if l_new <= l_test:
                    break
                alpha *= backtrack_factor

            if np.linalg.norm(alpha * d) <= gtol:
                msg = "Converged in stepsize"
                success = True
                break
            x += alpha * d
        else:
            # w = max(Q_1.T @ H @ Q_1)
            # W = compute_current_W(vecs, eigs, A_list, t, w)
            U, d, info = solve_inner_QP(
                vecs, eigs, A_vec, t, rho=rho, W=W, verbose=verbose
            )

            if not info["success"]:
                raise ValueError("coudn't find feasible direction d")

            eigs_U = np.linalg.eigvalsh(U)

            if eigs_U[0] >= -TOL_EIG:
                l_emp = eigs[0] + info["delta"]

                eigs, *_ = get_max_eig(Q, A_vec, x + d, tau)
                l_new = eigs[0]
                if abs(l_new - l_emp) > 1e-10:
                    print(
                        f"Expected lambda not equal to actual lambda! {l_new:.6e}, {l_emp:.6e}"
                    )

                if np.linalg.norm(d) < gtol:
                    msg = "Converged in stepsize"
                    success = True
                    break
                else:
                    x += d
            else:
                print("Warning: U not p.s.d.")
                d, info = solve_d_from_indefinite_U(U, Q_1, A_vec)
                eigs, *_ = get_max_eig(Q, A_vec, x + d, tau)
                l_new = eigs[0]
                if d is not None:
                    x += d
                else:
                    tau = 0.5 * tau
                    rho = 0.5 * rho

        if verbose > 1:
            print(
                f"it {i} \t eigs {eigs.round(2)} \t t {t} \t d {d.round(2)} \t lambda {l_new:.4e}"
            )

        if l_threshold and (l_new >= l_threshold):
            msg = "Found valid certificate"
            success = True
            break

        i += 1
        if i == max_iters:
            msg = "Reached maximum iterations"
            success = False
    info = {"success": success, "msg": msg, "U": U, "lambda": l_new}
    return x, info
