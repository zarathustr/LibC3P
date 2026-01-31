import itertools
import sys

import cvxpy as cp
import numpy as np
from cert_tools.base_clique import BaseClique
from cert_tools.fusion_tools import get_slice, mat_fusion
from cert_tools.sdp_solvers import options_cvxpy
from mosek.fusion import Domain, Expr, Model, ObjectiveSense, ProblemStatus

CONSTRAIN_ALL_OVERLAP = False

TOL = 1e-5


# TODO(FD) this is extremely hacky but I don't know how to read solution data when the
# status is UNKNOWN...
def read_costs_from_mosek(fname):
    f = open(fname, "r")
    ls = f.readlines()
    primal_line = ls[-2].split(" ")
    assert "Primal." in primal_line
    primal_value = float(primal_line[primal_line.index("obj:") + 1])

    dual_line = ls[-1].split(" ")
    assert "Dual." in dual_line
    dual_value = float(dual_line[dual_line.index("obj:") + 1])
    return primal_value, dual_value


def solve_oneshot_dual_slow(clique_list):
    """Implementation of range-space clique decomposition as in [Zheng 2020]."""
    from cert_tools.sdp_solvers import adjust_Q

    N = len(clique_list) + 1
    A_list = []
    for k, clique in enumerate(clique_list):
        clique.H = cp.Variable(clique.Q.shape, PSD=True)
        if k == 0:
            A_list += [clique.E.T @ A @ clique.E for A in clique.A_list]
        else:
            A_list += [clique.E.T @ clique.A_list[-1] @ clique.E]

    Q = cp.sum([clique.E.T @ clique.Q @ clique.E for clique in clique_list])
    Q_here, scale, offset = adjust_Q(Q)
    sigmas = cp.Variable(len(A_list))
    constraints = [
        cp.sum([clique.E.T @ clique.H @ clique.E for clique in clique_list])
        == Q_here + cp.sum([sigmas[k] * A_list[k] for k in range(len(A_list))])
    ]
    cprob = cp.Problem(cp.Maximize(-sigmas[0]), constraints)
    options_cvxpy["verbose"] = True
    cprob.solve(solver="MOSEK", **options_cvxpy)

    # H_k_list = [clique.H.value for clique in clique_list]
    X_k_list = constraints[0].dual_value
    sigma_dict = {i: sigma.value for i, sigma in enumerate(sigmas)}
    if not np.isinf(cprob.value):
        cost = cprob.value * scale + offset
        info = {"cost": cost, "sigma_dict": sigma_dict}
        info["success"] = True
    else:
        info = {"cost": np.inf, "sigma_dict": sigma_dict}
        info["success"] = False
    return X_k_list, info


def solve_oneshot_dual_cvxpy(clique_list, tol=TOL, verbose=False, adjust=False):
    """Implementation of range-space clique decomposition using auxiliary variables."""
    B_list_left = clique_list[0].get_B_list_left()
    B_list_right = clique_list[0].get_B_list_right()
    N = len(clique_list) + 1
    # raise ValueError("need to implement a fast dual version of this!")
    constraints = []
    sigmas = cp.Variable(N)
    rhos = cp.Variable(N - 1)
    for k, clique in enumerate(clique_list):
        if k == 0:
            z_var_left = None
            z_var_right = cp.Variable(len(B_list_right))

            s = sigmas[k : k + 2]
            A_list = clique.A_list[1:]
        elif k < N - 2:
            z_var_left = z_var_right
            z_var_right = cp.Variable(len(B_list_right))

            A_list = clique.A_list[2:]
            s = sigmas[k + 1 : k + 2]
        else:
            z_var_left = z_var_right  # previous right now becomes left.
            z_var_right = None

            A_list = clique.A_list[2:]
            s = sigmas[k + 1 : k + 2]

        clique.H = (
            clique.Q
            + rhos[k] * clique.A_list[0]
            + cp.sum([s[i] * A_list[i] for i in range(len(A_list))])
        )
        if z_var_left is not None:
            clique.H += cp.sum(
                [z_var_left[i] * B_list_left[i] for i in range(len(B_list_left))]
            )
        if z_var_right is not None:
            clique.H += cp.sum(
                [z_var_right[i] * B_list_right[i] for i in range(len(B_list_right))]
            )
        constraints += [clique.H >> 0]
    cprob = cp.Problem(cp.Maximize(-cp.sum(rhos)), constraints)
    data, *__ = cprob.get_problem_data(cp.SCS)
    options_cvxpy["verbose"] = verbose
    cprob.solve(solver="MOSEK", **options_cvxpy)

    X_k_list = [con.dual_value for con in constraints]
    # H_k_list = [clique.H.value for clique in clique_list]
    sigma_dict = {i: sigma.value for i, sigma in enumerate(sigmas)}
    if not np.isinf(cprob.value):
        cost = cprob.value
        info = {"cost": cost, "sigma_dict": sigma_dict}
        info["success"] = True
    else:
        info = {"cost": np.inf, "sigma_dict": sigma_dict}
        info["success"] = False
    return X_k_list, info


def solve_oneshot_primal_fusion(clique_list, verbose=False, tol=TOL, adjust=False):
    """
    clique_list is a list of objects inheriting from BaseClique.
    """
    if adjust:
        from cert_tools.sdp_solvers import adjust_Q

        raise ValueError("adjust_Q does not work when dealing with cliques")

    assert isinstance(clique_list[0], BaseClique)

    X_dim = clique_list[0].X_dim
    N = len(clique_list)
    with Model("primal") as M:
        # creates (N x X_dim x X_dim) variable
        X = M.variable(Domain.inPSDCone(X_dim, N))

        if adjust:
            Q_scale_offsets = [adjust_Q(c.Q) for c in clique_list]
        else:
            Q_scale_offsets = [(c.Q, 1.0, 0.0) for c in clique_list]

        # objective
        M.objective(
            ObjectiveSense.Minimize,
            Expr.add(
                [
                    Expr.dot(mat_fusion(Q_scale_offsets[i][0]), get_slice(X, i))
                    for i in range(N)
                ]
            ),
        )

        # standard equality constraints
        A_0_constraints = []
        for i, clique in enumerate(clique_list):
            for A, b in zip(clique.A_list, clique.b_list):
                A_fusion = mat_fusion(A)
                con = M.constraint(
                    Expr.dot(A_fusion, get_slice(X, i)), Domain.equalsTo(b)
                )
                if b == 1:
                    A_0_constraints.append(con)

        # for cl, ck in itertools.permutations(clique_list, 2):
        # for cl, ck in itertools.combinations(clique_list, 2):
        for cl, ck in zip(clique_list[:-1], clique_list[1:]):
            overlap = BaseClique.get_overlap(cl, ck, h=cl.hom)
            for l in overlap:
                for rl, rk in zip(cl.get_ranges(l), ck.get_ranges(l)):
                    # cl.X_var[rl[0], rl[1]] == ck.X[rk[0], rk[1]])
                    left_start = [rl[0][0], rl[1][0]]
                    left_end = [rl[0][-1] + 1, rl[1][-1] + 1]
                    right_start = [rk[0][0], rk[1][0]]
                    right_end = [rk[0][-1] + 1, rk[1][-1] + 1]
                    X_left = X.slice([cl.index] + left_start, [cl.index + 1] + left_end)
                    X_right = X.slice(
                        [ck.index] + right_start, [ck.index + 1] + right_end
                    )
                    M.constraint(Expr.sub(X_left, X_right), Domain.equalsTo(0))

                    if cl.X is not None and ck.X is not None:
                        np.testing.assert_allclose(
                            cl.X[
                                left_start[0] : left_end[0], left_start[1] : left_end[1]
                            ],
                            ck.X[
                                right_start[0] : right_end[0],
                                right_start[1] : right_end[1],
                            ],
                        )

        M.setSolverParam("intpntCoTolDfeas", tol)  # default 1e-8
        M.setSolverParam("intpntCoTolPfeas", tol)  # default 1e-8
        M.setSolverParam("intpntCoTolMuRed", tol)  # default 1e-8
        if verbose:
            M.setLogHandler(sys.stdout)
        else:
            f = open("mosek_output.tmp", "a+")
            M.setLogHandler(f)
        M.solve()
        if M.getProblemStatus() is ProblemStatus.Unknown:
            X_list_k = []
            cost = np.inf
            if not verbose:
                f.close()
                primal_value, dual_value = read_costs_from_mosek("mosek_output.tmp")
                if (abs(primal_value) - abs(dual_value)) / abs(primal_value) > 1e-2:
                    print("Warning: solution not good")
                cost = abs(primal_value)
            info = {"success": False, "cost": cost, "msg": "UNKNOWN"}
        elif M.getProblemStatus() is ProblemStatus.PrimalAndDualFeasible:
            X_list_k = [
                np.reshape(get_slice(X, i).level(), (X_dim, X_dim)) for i in range(N)
            ]
            cost_raw = M.primalObjValue()
            costs_per_clique = [con.dual()[0] for con in A_0_constraints]
            cost_test = abs(sum(costs_per_clique))
            if cost_test > 1e-8:
                assert abs((cost_raw - cost_test) / cost_test) < 1e-1
            cost = sum(
                costs_per_clique[i] * Q_scale_offsets[i][1] + Q_scale_offsets[i][2]
                for i in range(N)
            )
            info = {"success": True, "cost": cost}
        return X_list_k, info


def solve_oneshot_primal_cvxpy(clique_list, verbose=False, tol=TOL):
    constraints = []
    for clique in clique_list:
        clique.X_var = cp.Variable((clique.X_dim, clique.X_dim), PSD=True)
        constraints += [
            cp.trace(A @ clique.X_var) == b
            for A, b in zip(clique.A_list, clique.b_list)
        ]

    # add constraints for overlapping regions
    for cl, ck in itertools.combinations(clique_list, 2):
        overlap = BaseClique.get_overlap(cl, ck, h=cl.hom)
        for l in overlap:
            for rl, rk in zip(cl.get_ranges(l), ck.get_ranges(l)):
                constraints.append(cl.X_var[rl[0], rl[1]] == ck.X_var[rk[0], rk[1]])
                np.testing.assert_allclose(cl.X[rl[0], rl[1]], ck.X[rk[0], rk[1]])

    cprob = cp.Problem(
        cp.Minimize(
            cp.sum([cp.trace(clique.Q @ clique.X_var) for clique in clique_list])
        ),
        constraints,
    )

    options_cvxpy["verbose"] = verbose
    cprob.solve(solver="MOSEK", **options_cvxpy)

    X_k_list = [clique.X_var.value for clique in clique_list]
    sigma_dict = {
        i: constraint.dual_value
        for i, constraint in enumerate(constraints[-len(clique_list) :])
    }
    info = {"cost": cprob.value, "sigma_dict": sigma_dict}
    if not np.isinf(cprob.value):
        info["success"] = True
    else:
        info["success"] = False
    return X_k_list, info


def solve_oneshot(
    clique_list, use_primal=True, use_fusion=False, verbose=False, tol=TOL
):
    if not use_primal:
        print("Defaulting to primal because dual cliques not implemented yet.")
    if use_fusion:
        return solve_oneshot_primal_fusion(clique_list, verbose=verbose, tol=tol)
    else:
        return solve_oneshot_primal_cvxpy(clique_list, verbose=verbose, tol=tol)
    # return solve_oneshot_dual_cvxpy(
    #        clique_list, verbose=verbose, tol=tol, adjust=adjust
    #    )
