import sys
from copy import deepcopy
from multiprocessing import Pipe, Process

import cvxpy as cp
import numpy as np
from cert_tools.admm_clique import ADMMClique
from cert_tools.fusion_tools import mat_fusion
from cert_tools.sdp_solvers import options_cvxpy
from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

EARLY_STOP = True
EARLY_STOP_MIN = 1e-2

RHO_START = 1.0

MAX_ITER = 1000

# See [Boyd 2010] for explanations of these.
MU_RHO = 2.0  # how much dual and primal residual may get unbalanced.
TAU_RHO = 2.0  # how much to change rho in each iteration.
EPS_ABS = 0.0  # set to 0 to use relative only
EPS_REL = 1e-4

# Stop ADMM if the last N_ADMM iterations don't have significant change in cost.
N_ADMM = 3


def initialize_Z(clique_list, X0=None):
    """Initialize Z (consensus variable) based on contents of X0 (initial feasible points)"""
    x_dim = clique_list[0].x_dim
    indices = [0] + list(range(1 + x_dim, 1 + 2 * x_dim))
    ii, jj = np.meshgrid(indices, indices)
    for k, clique in enumerate(clique_list[:-1]):
        if X0 is not None:
            clique.Z_new = deepcopy(X0[k][ii, jj])
        else:
            clique.Z_new = np.zeros((1 + clique.x_dim, 1 + clique.x_dim))


def check_convergence(clique_list, primal_err, dual_err):
    """Check convergence using the criteria by [Boyd 2010]."""
    dim_primal = (len(clique_list) - 1) * clique_list[0].X_dim
    dim_dual = len(clique_list) * clique_list[0].X_dim
    eps_max = np.max(
        [
            np.max(np.hstack([clique.X_new.flatten(), clique.Z_new.flatten()]))
            for clique in clique_list[:-1]
        ]
    )
    eps_pri = np.sqrt(dim_primal) * EPS_ABS + EPS_REL * eps_max
    eps_dual = np.sqrt(dim_dual) * EPS_ABS + EPS_REL * np.linalg.norm(
        np.hstack([clique.sigmas for clique in clique_list])
    )
    return (primal_err < eps_pri) and (dual_err < eps_dual)


def update_rho(rho, dual_res, primal_res, mu=MU_RHO, tau=TAU_RHO):
    """Update rho as suggested by [Boyd 2010]."""
    assert tau >= 1.0
    assert mu >= 0
    if primal_res >= mu * dual_res:
        return rho * tau
    elif dual_res >= mu * primal_res:
        return rho / tau
    else:
        return rho


def update_Z(clique_list):
    """Average the overlapping areas of X_new for consensus (stored in Z_new)."""
    x_dim = clique_list[0].x_dim
    for i in range(len(clique_list) - 1):
        # for each Z, average over the neighboring cliques.
        left = clique_list[i].X_new
        right = clique_list[i + 1].X_new

        clique_list[i].Z_prev = deepcopy(clique_list[i].Z_new)

        average = np.zeros((1 + x_dim, 1 + x_dim))
        average[0, 0] = 1.0
        average[0, 1:] = average[1:, 0] = 0.5 * (
            left[1 + x_dim :, 0] + right[1 : 1 + x_dim, 0]
        )
        average[1:, 1:] = 0.5 * (
            left[1 + x_dim :, 1 + x_dim :] + right[1 : 1 + x_dim, 1 : 1 + x_dim]
        )
        clique_list[i].Z_new = average


def update_sigmas(clique_list, rho_k):
    """Update dual variables."""
    primal_res = []
    dual_res = []
    for k, clique in enumerate(clique_list):
        assert isinstance(clique, ADMMClique)
        left = clique_list[k - 1].Z_new if k > 0 else None
        right = clique_list[k].Z_new if k < len(clique_list) - 1 else None
        clique.generate_g(left=left, right=right)
        primal_res_k = clique.evaluate_F(clique.X_new)

        clique.sigmas += rho_k * primal_res_k
        primal_res.append(primal_res_k)

        if k < len(clique_list) - 1:
            dual_res_k = rho_k * clique.get_dual_residual().flatten()
            dual_res.append(dual_res_k)

    primal_res = np.hstack(primal_res)
    dual_res = np.hstack(dual_res)

    primal_err = np.linalg.norm(primal_res)
    dual_err = np.linalg.norm(dual_res)
    return primal_err, dual_err


# TODO(FD): this function is hideous. Can we simplify / remove it somehow?
def wrap_up(
    clique_list,
    cost_history,
    primal_err,
    dual_err,
    rho_k,
    iter,
    early_stop,
    verbose=False,
):
    """Calculate and print statistics, check stopping criteria etc."""
    info = {}
    cost_original = cost_history[-1]
    if verbose:
        with np.printoptions(precision=2, suppress=True, threshold=5):
            if iter % 20 == 0:
                print(
                    "iter     rho_k          prim. error         dual error       cost"
                )
            print(
                f"{iter} \t {rho_k:2.4e} \t {primal_err:2.4e} \t {dual_err:2.4e} \t {cost_original:5.5f}"
            )

    rel_diff = (
        np.max(np.abs(np.diff(cost_history[-N_ADMM:]))) / cost_history[-1]
        if len(cost_history) >= N_ADMM
        else None
    )
    if check_convergence(clique_list, primal_err, dual_err):
        info["success"] = True
        info["msg"] = f"converged after {iter} iterations"
        info["stop"] = True
    elif early_stop and rel_diff and (rel_diff < EARLY_STOP_MIN):
        info["success"] = True
        info["msg"] = f"stopping after {iter} because cost didn't change enough"
        info["stop"] = True
    # All cliques did not solve in the last iteration.
    elif all([c.status < 0 for c in clique_list]):
        info["success"] = False
        info["msg"] = f"all problems infeasible after {iter} iterations"
        info["stop"] = True
    return info


def solve_inner_sdp_fusion(Q, Constraints, F, g, sigmas, rho, verbose=False):
    """Solve X update of ADMM using fusion."""
    with Model("primal") as M:
        # creates (N x X_dim x X_dim) variable
        X = M.variable("X", Domain.inPSDCone(Q.shape[0]))
        if F is not None:
            S = M.variable("S", Domain.inPSDCone(F.shape[0] + 2))
            a = M.variable("a", Domain.inPSDCone(1))

        # standard equality constraints
        for A, b in Constraints:
            M.constraint(Expr.dot(mat_fusion(A), X), Domain.equalsTo(b))

        # interlocking equality constraints
        # objective
        if F is not None:
            assert g is not None
            if F.shape[1] == Q.shape[0]:
                err = Expr.sub(
                    Expr.mul(F, X.slice([0, 0], [Q.shape[0], 1])),
                    Matrix.dense(g.value[:, None]),
                )
            else:
                err = Expr.sub(Expr.mul(F, Expr.flatten(X)), g)

            # doesn't work unforuntately:
            # Expr.mul(0.5 * rho, Expr.sum(Expr.mulElm(err, err))),
            M.objective(
                ObjectiveSense.Minimize,
                Expr.add(
                    [
                        Expr.dot(mat_fusion(Q), X),
                        Expr.sum(
                            Expr.mul(Matrix.dense(sigmas[None, :]), err)
                        ),  # sum is to go from [1,1] to scalar
                        Expr.sum(a),
                    ]
                ),
            )
            M.constraint(Expr.sub(S.index([0, 0]), a), Domain.equalsTo(0.0))
            M.constraint(
                Expr.sub(
                    S.slice([1, 1], [1 + F.shape[0], 1 + F.shape[0]]),
                    Matrix.sparse(
                        F.shape[0],
                        F.shape[0],
                        range(F.shape[0]),
                        range(F.shape[0]),
                        [2 / rho] * F.shape[0],
                    ),
                ),
                Domain.equalsTo(0.0),
            )
            M.constraint(
                Expr.sub(S.slice([1, 0], [1 + F.shape[0], 1]), err),
                Domain.equalsTo(0.0),
            )
        else:
            M.objective(ObjectiveSense.Minimize, Expr.dot(Q, X))

        # M.setSolverParam("intpntCoTolRelGap", 1.0e-7)
        if verbose:
            M.setLogHandler(sys.stdout)
        M.solve()

        X = np.reshape(X.level(), Q.shape)
        info = {"success": True, "cost": M.primalObjValue()}
    return X, info


def solve_inner_sdp(clique: ADMMClique, rho=None, verbose=False, use_fusion=True):
    """Solve the inner SDP of the ADMM algorithm, similar to [Dall'Anese 2013]

    min <Q, X> + y'e(X) + rho/2*||e(X)||^2
    s.t. <Ai, X> = bi
         X >= 0

    where e(X) = F @ vec(X) - b
    """
    if use_fusion:
        Constraints = list(zip(clique.A_list, clique.b_list))
        return solve_inner_sdp_fusion(
            clique.Q, Constraints, clique.F, clique.g, clique.sigmas, rho, verbose
        )
    else:
        objective = clique.get_objective_cvxpy(clique.X_var, rho)
        constraints = clique.get_constraints_cvxpy(clique.X_var)
        cprob = cp.Problem(objective, constraints)
        options_cvxpy["verbose"] = verbose
        try:
            cprob.solve(solver="MOSEK", **options_cvxpy)
            info = {
                "cost": float(cprob.value),
                "success": clique.X_var.value is not None,
            }
        except:
            info = {"cost": np.inf, "success": False}
        return clique.X_var.value, info


def solve_alternating(
    clique_list,
    X0: list[np.ndarray] = None,
    sigmas: dict = None,
    rho_start: float = RHO_START,
    use_fusion: bool = True,
    early_stop: bool = EARLY_STOP,
    verbose: bool = False,
    max_iter=MAX_ITER,
    mu_rho=MU_RHO,
    tau_rho=TAU_RHO,
):
    """Use ADMM to solve decomposed SDP, but without using parallelism."""
    if sigmas is not None:
        for k, sigma in sigmas.items():
            clique_list[k] = sigma

    rho_k = rho_start
    info_here = {"success": False, "msg": "did not converge.", "stop": False}

    cost_history = []

    initialize_Z(clique_list, X0)  # fill Z_new
    for iter in range(max_iter):
        cost_lagrangian = 0
        cost_original = 0

        # ADMM step 1: update X
        for k, clique in enumerate(clique_list):
            # update g with solved value from previous iteration
            left = clique_list[k - 1].Z_new if k > 0 else None
            right = clique_list[k].Z_new if k < len(clique_list) - 1 else None
            clique.generate_g(left=left, right=right)
            if clique.X_var.value is not None:
                pass

            X, info = solve_inner_sdp(
                clique, rho_k, verbose=False, use_fusion=use_fusion
            )
            cost = info["cost"]

            if X is not None:
                clique.X_new = deepcopy(X)
                clique.status = 1
            else:
                print(f"clique {k:02.0f} did not converge!!")
                clique.status = -1

            cost_lagrangian += cost
            cost_original += float(np.trace(clique.Q @ clique.X_new))

        # ADMM step 2: update Z
        update_Z(clique_list)

        # ADMM step 3: update Lagrange multipliers
        primal_err, dual_err = update_sigmas(clique_list, rho_k)

        rho_k = update_rho(rho_k, dual_err, primal_err, mu=mu_rho, tau=tau_rho)

        info_here["cost"] = cost_original
        cost_history.append(cost_original)
        info_here.update(
            wrap_up(
                clique_list,
                cost_history,
                primal_err,
                dual_err,
                rho_k,
                iter,
                early_stop,
                verbose,
            )
        )
        if info_here["stop"]:
            break

    X_k_list = [clique.X_new for clique in clique_list]
    return X_k_list, info_here


def solve_parallel(
    clique_list,
    X0=None,
    rho_start=RHO_START,
    early_stop=False,
):
    """Use ADMM to solve decomposed SDP, with simple parallelism."""

    def run_worker(clique, pipe):
        # ADMM loop.
        while True:
            left, right = pipe.recv()
            clique.generate_g(left=left, right=right)

            X, info = solve_inner_sdp(
                clique, rho=clique.rho_k, verbose=False, use_fusion=True
            )

            if X is not None:
                clique.X_new = X
                clique.counter += 1
                clique.status = 1
            else:
                clique.status = -1
                pass
            pipe.send(clique)

    initialize_Z(clique_list, X0)

    # Setup the workers
    pipes = []
    procs = []
    for i, clique in enumerate(clique_list):
        clique.rho_k = rho_start
        clique.counter = 0
        clique.status = 0
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_worker, args=(clique, remote))]
        procs[-1].start()

    # Run ADMM
    rho_k = rho_start
    info_here = {"success": False, "msg": "did not converge", "stop": False}
    cost_history = []
    for iter in range(MAX_ITER):
        # ADMM step 1: update X varaibles (in parallel)
        for k, pipe in enumerate(pipes):
            left = clique_list[k - 1].Z_new if k > 0 else None
            right = clique_list[k].Z_new if k < len(clique_list) - 1 else None
            pipe.send([left, right])
        clique_list = [pipe.recv() for pipe in pipes]

        # ADMM step 2: update Z variables
        update_Z(clique_list)

        # ADMM step 3: update Lagrange multipliers
        primal_err, dual_err = update_sigmas(clique_list, rho_k)

        # Intermediate steps: update rho and check convergence
        rho_k = update_rho(rho_k, dual_err, primal_err)
        for clique in clique_list:
            clique.rho_k = rho_k

        cost_original = np.sum(
            [np.trace(clique.X_new @ clique.Q) for clique in clique_list]
        )
        cost_history.append(cost_original)
        info_here["cost"] = cost_original
        info_here.update(
            wrap_up(
                clique_list, cost_history, primal_err, dual_err, rho_k, iter, early_stop
            )
        )
        if info_here["stop"]:
            break

    [p.terminate() for p in procs]
    return [clique.X_new for clique in clique_list], info_here
