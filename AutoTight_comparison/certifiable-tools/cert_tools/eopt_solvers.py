# Optimization
import cvxpy as cp
from scipy.optimize import linprog

import mosek

# Maths
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from cert_tools.eig_tools import get_min_eigpairs
from cert_tools.linalg_tools import get_nullspace

# Plotting
import matplotlib.pyplot as plt


# Data storage
import pandas as pd

# Number of eigenvalues to compute
EIG_METHOD = "direct"  # "lobpcg"
N_EIGS_LANCZOS = 1

# Default options for cutting plane method
opts_cut_dflt = dict(
    tol_eig=1e-6,  # Eigenvalue tolerance
    max_iter=200,  # Maximum iterations
    min_eig_ub=1.0,  # Upper bound for cutting plane
    lambda_level=0.9,  # level multiplier (for level method)
    level_method_bound=1e5,  # above this level, default to vanilla cut plane
    tol_null=1e-5,  # null space tolerance for first order KKT constraints
    use_null=True,  # if true, reparameterize problem using null space
    cut_buffer=30,  # number of cuts stored at once, FIFO
    use_hessian=False,  # Flag to select whether to use the Hessian
)

opts_sub_dflt = dict(
    tol_eig=1e-8, max_iters=200, gtol=1e-7, k_max=1, use_null=True, tol_null=1e-5
)

# see Nocedal & Wright, Algorithm 3.1
# rho (how much to decrase alpha)
backtrack_factor = 0.5
#  c (when to stop)
backtrack_cutoff = 0.2
# starting value for alpha
backtrack_start = 10.0


class CutPlaneModel:
    """This class stores all of the information of the cutting plane model.
    This includes all of the cutting planes as well as any equality constraints.
    This class also stores methods for finding the optimum of the model.
    Model in epigraph form is as follows:
    max t
    s.t. t <= values[j] + gradients[j].T @ delta
                    + gradients[j].T @(x_current - eval_pts[j]), forall j
    """

    def __init__(m, n_vars, A_eq=None, b_eq=None, opts=opts_cut_dflt):
        """Initialize cut plane model"""
        # Store shape
        m.n_vars = n_vars
        # Add a single cutting plane
        m.gradients = []
        m.values = []
        m.eval_pts = []
        m.n_cuts = 0
        # Equality constraints
        m.A_eq = A_eq
        m.b_eq = b_eq
        # Cutting plane buffer limit
        m.opts = opts

    def add_cut(m, grad_info: dict, eval_pt):
        """Add a cutting plane to the model

        Args:
            m (CutPlaneModel)
            grad_info (dict): gradient information dictionary
            eval_pt (): point where the gradient was evaluated
        """
        # Add data
        m.gradients += [grad_info["subgrad"]]
        m.values += [grad_info["min_eig"]]
        m.eval_pts += [eval_pt]
        # Update number of cuts
        m.n_cuts += 1
        # Remove cuts if exceeding the max number allowed
        if m.n_cuts > m.opts["cut_buffer"]:
            m.rm_cut(ind=0)

    def rm_cut(m, ind):
        """Remove cut planes at index"""
        # Remove planes
        m.gradients.pop(ind)
        m.values.pop(ind)
        m.eval_pts.pop(ind)
        # Decrement cut counter
        m.n_cuts -= 1

    def evaluate(m, x):
        values = [
            (m.values[i] + m.gradients[i].T @ (x - m.eval_pts[i]))[0, 0]
            for i in range(m.n_cuts)
        ]
        return min(values + [m.opts["min_eig_ub"]])

    def solve_lp_linprog(m):
        # Cost: maximize t
        one = np.array([[1.0]])
        c = -np.hstack([one, np.zeros((1, m.n_vars))])
        # Set bounds on t level only
        bounds = [(None, m.opts["min_eig_ub"])] + [(None, None)] * m.n_vars
        # Solve opt problem
        if m.A_eq is not None:
            A_eq_lp = np.hstack([np.zeros((m.A_eq.shape[0], 1)), m.A_eq])
            b_eq_lp = m.b_eq.squeeze()
        else:
            A_eq_lp = b_eq_lp = None

        # NOTE: t <= grad @ (x-x_j) + f(x_j) <==>
        # [1 -grad] @ [t; x] <= f(x_j) -grad @ x_j
        A_cut = np.hstack((np.ones((m.n_cuts, 1)), -np.hstack(m.gradients).T))
        b_cut = np.array(
            [m.values[i] - m.gradients[i].T @ m.eval_pts[i] for i in range(m.n_cuts)]
        )
        # Run Linprog
        res = linprog(
            c=c.squeeze(),
            A_eq=A_eq_lp,
            b_eq=b_eq_lp,
            A_ub=A_cut,
            b_ub=b_cut,
            method="highs-ds",
            bounds=bounds,
        )
        if res.success:
            t_lp, x_lp = res.x[0], res.x[1:]
            x_lp = np.expand_dims(x_lp, axis=1)
        else:
            raise ValueError("Linear subproblem failed.")
        return t_lp, x_lp

    def solve_level_project(m, x_prox, level):
        """Solve level set projection problem: return the closest point to x_prox
        (the prox-center) such that the acheived model value is above the provided
        level"""

        # VARIABLES
        delta = cp.Variable((m.n_vars, 1), "delta")
        t = cp.Variable(1, "t")
        # CONSTRAINTS
        # cut plane model constraints
        # NOTE: these constraints could be cached
        constraints = [
            t <= m.values[i] + m.gradients[i].T @ (delta + x_prox - m.eval_pts[i])
            for i in range(m.n_cuts)
        ]
        # model value larger than specified level
        constraints += [t >= level]
        # add equality constraints
        if m.A_eq is not None:
            if len(m.b_eq.shape) == 1:
                b_eq = m.b_eq[:, None]
            else:
                b_eq = m.b_eq
            constraints += [m.A_eq @ (delta + x_prox) == b_eq]
        # Solve Quadratic Program:
        prob = cp.Problem(cp.Minimize(cp.norm2(delta)), constraints)
        try:
            prob.solve(solver="MOSEK", verbose=False)
        except mosek.Error:
            print("Did not find MOSEK, using different solver.")
            prob.solve(solver="CVXOPT", verbose=False)
        # Get solution
        return t.value, delta.value + x_prox


def f_eopt(Q, A_vec, x, **kwargs_eig):
    """Objective of E-OPT"""
    H = get_cert_mat(Q, A_vec, x)
    grad_info = get_grad_info(H=H, A_vec=A_vec, k=1, **kwargs_eig)
    return grad_info["min_eig"]


def get_grad_info(H, A_vec, U=None, tau=1e-8, get_hessian=False, **kwargs_eig):
    eig_vals, eig_vecs = get_min_eigpairs(H, **kwargs_eig)
    # get minimum eigenvalue
    min_eig = np.min(eig_vals)
    # split eigenvector sets based on closeness to min (multiplicity could be > 1)
    ind_1 = np.abs(eig_vals - min_eig) < tau
    Q_1 = eig_vecs[:, ind_1]
    # Size of matrix
    n = H.shape[0]
    # Multiplicity
    t = Q_1.shape[1]
    # Scale variable
    if U is None:
        U = np.zeros((t, t))
        i = np.random.choice(range(t))
        U[i, i] = 1.0
        # U = 1 / t * np.eye(t)

    # Compute gradient
    subgrad = A_vec.T @ (Q_1 @ U @ Q_1.T).reshape(-1, 1, order="F")
    # Compute Hessian
    if t == 1 and get_hessian:
        # Get other eigvectors and eigenvalues
        ind_s = np.abs(eig_vals - min_eig) >= tau
        Q_s = eig_vecs[:, ind_s]
        eig_inv_diffs = 1 / (min_eig - eig_vals[ind_s])
        Lambda_s = np.diag(eig_inv_diffs)
        # Construct Hessian
        Q_bar = A_vec.T @ np.kron(np.eye(n), Q_1) @ Q_s
        hessian = Q_bar @ Lambda_s @ Q_bar.T * 2
        # Compute damping term for conditioning of QP
        damp = np.min(eig_inv_diffs)
    else:
        hessian = None
        damp = None
    grad_info = dict(
        subgrad=np.array(subgrad),
        hessian=hessian,
        min_eig=min_eig,
        min_vec=Q_1,
        t=t,
        damp=damp,
    )
    return grad_info


def preprocess_constraints(C, Constraints, x_cand, use_null=False, opts=opts_cut_dflt):
    """Pre-processing steps for certificate optimization.
    Uses cost, constraints, and candidate solution to build the affine, first-order
    conditions for the SDP.
    If option use_null set to True, the constraints are reparameterized using
    a sparse null-space basis for the original optimality conditions

    Args:
        C (_type_): _description_
        Constraints (_type_): _description_
        x_cand (_type_): _description_
        opts (_type_, optional): _description_. Defaults to opts_cut_dflt.
    """
    # Loop through constraints
    A_bar = []
    A_vec = []
    for A, b in Constraints:
        # LHS matrix of affine constraint
        A_bar += [A @ x_cand]
        # Vectorized list of constraints: TODO Change this to the half-vec format
        A_vec += [A.reshape((-1, 1), order="F")]
    A_bar = np.hstack(A_bar)
    A_vec = sp.hstack(A_vec)
    # RHS of affine constraints
    b_bar = -C @ x_cand

    # Perform QR decomposition to characterize and work with null space
    # TODO: we should convert this to a sparse QR decomposition (use sparseqr module)
    basis, info = get_nullspace(A_bar, method="qrp", tolerance=opts["tol_null"])

    # Truncate eigenvalues of A_bar to make nullspace more defined (required due to opt tolerances)
    # Define equality constraints. Column of zeros added for epigraph form variable
    # TODO Consider adding this column later when actually running the LP
    A_eq = info["LHS"]
    b_eq = info["Q1"].T @ b_bar
    # Output
    return A_vec, A_eq, b_eq, basis.T, A_bar


def solve_eopt(
    Q,
    Constraints,
    x_cand,
    opts=opts_cut_dflt,
    verbose=True,
    plot=False,
    exploit_centered=False,
    method="cuts",
    **kwargs,
):
    """Solve the certificate/eigenvalue optimization problem"""
    # Preprocess constraints
    use_null = opts["use_null"]

    constr_info = preprocess_constraints(
        Q,
        Constraints,
        x_cand,
        use_null=use_null,
        opts=opts,
    )
    A_vec, A_eq, b_eq, basis, A_bar = constr_info

    if plot:
        fig1, axs1 = plt.subplots(1, 3)
        axs1[0].matshow(A_bar)
        axs1[0].set_title("L")
        axs1[1].matshow(np.hstack([A_eq, b_eq]).T)
        axs1[1].set_title("range basis (A.T, b.T)")
        axs1[2].matshow(basis)
        axs1[2].set_title("null basis (N)")

        fig2, axs2 = plt.subplots(1, 2)
        eigs = np.linalg.eigvalsh(Q.toarray())[:3]
        axs2[0].matshow(Q.toarray())
        axs2[0].set_title(f"Q \n{eigs}")

    # INITIALIZE
    # Initialize multiplier variables
    x_bar = la.lstsq(A_eq, b_eq, rcond=None)[0]

    if use_null:
        # Update Q matrix to include the fixed lagrange multipliers
        # TODO: might be clearer if we rename Q here to H_bar or something.
        Q = Q + (A_vec @ x_bar).reshape(Q.shape, order="F")
        A_vec = A_vec @ sp.coo_array(basis)
        x = np.zeros((A_vec.shape[1], 1))

    else:
        x = kwargs.get("x_init", x_bar)

    if exploit_centered:
        A_vec = np.hstack(
            [A.reshape(Q.shape, order="F")[1:, 1:].reshape(-1, 1) for A in A_vec.T]
        )
        Q = Q[1:, 1:]
        x_cand = x_cand[1:]

    if plot:
        eigs = np.linalg.eigvalsh(Q)[:3]
        axs2[1].matshow(Q)
        axs2[1].set_title(f"Q used \n{eigs}")

    # Most general form:
    # Q_bar + sum_i(a_i * A_bar_i)
    # s.t. Ca_i = b

    # Get orthogonal vector to x_cand for eigenvalue solver
    x_rand = np.random.rand(*x_cand.shape) - 1
    v0 = x_rand - x_cand.T @ x_rand / (x_cand.T @ x_cand)
    kwargs_eig = {"v0": v0, "method": EIG_METHOD}

    if method == "cuts":
        solver = solve_eopt_cuts
    elif method == "sub":
        solver = solve_eopt_sub
    else:
        raise ValueError(f"Unknown method {method} in solve_eopt")

    if use_null:
        alphas, info = solver(
            Q,
            A_vec,
            A_eq=None,
            b_eq=None,
            xinit=x,
            kwargs_eig=kwargs_eig,
            verbose=verbose,
        )
        mults = x_bar.flatten() + basis @ alphas.flatten()
    else:
        alphas, info = solver(
            Q,
            A_vec,
            A_eq=A_eq,
            b_eq=b_eq,
            xinit=x,
            kwargs_eig=kwargs_eig,
            verbose=verbose,
        )
        mults = alphas.flatten()

    info["mults"] = mults
    info["A_vec"] = A_vec
    info["Q"] = Q
    return alphas, info


def solve_eopt_sub(
    Q,
    A_vec,
    A_eq=None,
    b_eq=None,
    xinit=None,
    opts=opts_sub_dflt,
    verbose=1,
    kwargs_eig={},
):
    """Solve E_OPT using a simple subgradient method with backtracking.

    :param A_vec: n*2xm matrix of vectorized constraints.
    """
    m = A_vec.shape[1]
    n = Q.shape[0]
    assert A_vec.shape[0] == n**2

    if (A_eq is not None) or (b_eq is not None):
        raise NotImplementedError("Can't use equality constraints in QP yet.")

    if xinit is None:
        xinit = np.zeros(m)
    else:
        xinit = xinit.flatten()

    k = min(n, opts["k_max"])
    kwargs_eig = dict(k=N_EIGS_LANCZOS, method=EIG_METHOD)

    i = 0
    x = xinit
    print("it \t alpha \t t \t l_min")
    np.random.seed(1)

    l_new = None
    while i <= opts["max_iters"]:
        H = get_cert_mat(Q, A_vec, x)
        grad_info = get_grad_info(H, A_vec, U=None, **kwargs_eig)
        t = grad_info["t"]
        if t > 1:
            print(f"multiplicity {t}")
        if i == 0 and verbose > 0:
            print(f"start \t ------ \t {t} \t {grad_info['min_eig']:1.4e}")

        grad = grad_info["subgrad"][:, 0]
        d = grad / np.linalg.norm(grad)

        l_old = grad_info["min_eig"]

        # make sure that the gradient is valid
        # assert f_eopt(Q, A_vec, x + d * 1e-8, lmin=lmin) > l_old

        # backgracking, using Nocedal Algorithm 3.1
        alpha = backtrack_start

        H = get_cert_mat(Q, A_vec, x + alpha * d)
        grad_info = get_grad_info(H, A_vec, U=None, **kwargs_eig)
        l_new = grad_info["min_eig"]

        while np.linalg.norm(alpha * d) > opts["gtol"]:
            l_test = l_old + backtrack_cutoff * alpha * grad.T @ d
            if l_new >= l_test:
                break
            alpha *= backtrack_factor
            H = get_cert_mat(Q, A_vec, x + alpha * d)
            grad_info = get_grad_info(H, A_vec, U=None, **kwargs_eig)
            kwargs_eig["v0"] = grad_info["min_vec"]
            l_new = grad_info["min_eig"]

        if np.linalg.norm(alpha * d) <= opts["gtol"]:
            msg = "Converged in stepsize"
            success = True
            break
        x += alpha * d

        if verbose > 0:
            print(f"it {i} \t {alpha:1.4f} \t {t} \t {l_new:1.4e}")

        if (opts["tol_eig"] is not None) and (l_new >= -opts["tol_eig"]):
            msg = "Found valid certificate"
            success = True
            break

        i += 1
        if i == opts["max_iters"]:
            msg = "Reached maximum iterations"
            success = False
    info = {"success": success, "msg": msg, "min_eig": l_new, "H": H}
    return x, info


def solve_eopt_cuts(
    Q,
    A_vec,
    xinit,
    A_eq=None,
    b_eq=None,
    verbose=True,
    opts=opts_cut_dflt,
    kwargs_eig={},
):
    """Solve the certificate/eigenvalue optimization problem using a cutting plane algorithm.
    Current algorithm uses the level method with the target level at a tolerance below zero
    """
    # Initialize cut plane model
    m = CutPlaneModel(xinit.shape[0], A_eq=A_eq, b_eq=b_eq)
    # Intialize status vars for optimization
    status = "RUNNING"
    header_printed = False
    n_iter = 0
    t_max = np.inf
    t_min = -np.inf
    iter_info = []

    x = None
    grad_info = None
    while status == "RUNNING":
        # SOLVE CUT PLANE PROGRAM
        if n_iter > 0:
            if opts["use_hessian"]:
                # Solve QP
                pass
            else:
                # Solve LP
                t_lp, x_lp = m.solve_lp_linprog()
                # Compute level projection
                level = t_min + opts["lambda_level"] * (t_lp - t_min)

                # Condition on level required for now for solving projection
                if (np.abs(level) <= opts["level_method_bound"]) and (
                    opts["lambda_level"] < 1
                ):
                    t_qp, x_new = m.solve_level_project(x_prox=x, level=level)
                else:
                    x_new = x_lp
            # Store previous gradient information
            grad_info_prev = grad_info.copy()
        else:
            # Initialization step
            x_new = xinit.copy()
            t_lp = m.evaluate(x_new)

        # CUT PLANE UPDATE
        # Construct Current Certificate matrix
        H = get_cert_mat(Q, A_vec, x_new)
        # current gradient and minimum eig
        grad_info = get_grad_info(H=H, A_vec=A_vec, **kwargs_eig)

        # Add Cuts
        m.add_cut(grad_info, x_new)

        # STATUS UPDATE
        # update model upper bound
        if t_lp <= t_max:
            t_max = t_lp
        # update model lower bound
        if grad_info["min_eig"] > t_min:
            t_min = grad_info["min_eig"]

        # termination criteria
        if t_min >= -opts["tol_eig"]:  # positive lower bound
            status = "POS_LB"
        elif m.n_vars == 0:  # no variables (i.e. no redundant constraints)
            status = "NO_VAR"
        elif t_max < -2 * opts["tol_eig"]:  # negative upper bound
            status = "NEG_UB"
        elif n_iter >= opts["max_iter"]:  # max iterations
            status = "MAX_ITER"

        # Update vars
        n_iter += 1
        x = x_new
        # plot_along_grad(Q, A_vec, x_new, grad_info["subgrad"], 1)

        # Statistics
        gap = t_max - t_min
        if n_iter > 1:
            delta_grad = grad_info["subgrad"] - grad_info_prev["subgrad"]
        else:
            delta_grad = np.zeros(grad_info["subgrad"].shape)
        delta_x = x_new - x
        delta_norm = la.norm(delta_x)
        if delta_norm > 0:
            curv = (delta_grad.T @ delta_x)[0, 0] / delta_norm
        else:
            curv = 0.0

        # Store data
        info = dict(
            n_iter=n_iter,
            delta_norm=delta_norm,
            x=x,
            min_eig_curr=grad_info["min_eig"],
            t_max=t_max,
            t_min=t_min,
            gap=gap,
            mult=grad_info["t"],
            curv=curv,
        )
        iter_info += [info]
        if verbose:
            if n_iter % 10 == 1:
                header_printed = False
            if header_printed is False:
                print(" N   | delta_nrm |  eig val  |   t_max   |", end="")
                print("   t_min     | gap    |   curv   | mult. |")
                header_printed = True
            print(
                f" {n_iter:3d} | {delta_norm:5.4e} | {grad_info['min_eig']:5.4e} | {t_max:5.4e} |",
                end="",
            )
            print(f"{t_min:5.4e} | {gap:5.4e} | {curv:5.4e} | {grad_info['t']:4d}")
    return x, dict(
        min_eig=grad_info["min_eig"],
        H=H,
        status=status,
        gap=gap,
        t_min=t_min,
        t_max=t_max,
        iter_info=pd.DataFrame(iter_info),
        model=m,
    )


def get_cert_mat(C, A_vec, mults, sparsify=True, exploit_centered=False):
    """Generate certificate matrix from cost, constraints and multipliers
    C is the cost matrix amd A_vec is expected to be a vectorized version
    of the constraint matrices"""
    if sp.issparse(C) and not sparsify:
        H = C.todense()
    elif not sp.issparse(C) and sparsify:
        H = sp.csc_array(C)
    else:
        H = C.copy()
    # Loop through A matrices
    if sp.issparse(A_vec) and not sparsify:
        A_vec = A_vec.todense()
    elif not sp.issparse(A_vec) and sparsify:
        A_vec = sp.csc_array(A_vec)
    # Add in standard constraints
    if exploit_centered:
        size = H.shape[0] + 1
        H += (A_vec @ mults).reshape((size, size), order="F")[1:, 1:]
    else:
        H += (A_vec @ mults).reshape(H.shape, order="F")
    return H


def plot_along_grad(C, A_vec, mults, step, alpha_max):
    alphas = np.linspace(0, alpha_max, 100)
    min_eigs = np.zeros(alphas.shape)
    for i in range(len(alphas)):
        step_alpha = mults + alphas[i] * step
        # Apply step
        H_alpha = get_cert_mat(C, A_vec, step_alpha)
        # Check new minimum eigenvalue
        grad_info = get_grad_info(H_alpha, A_vec, k=1)
        min_eigs[i] = grad_info["min_eig"]

    # Plot min eig
    plt.figure()
    plt.plot(alphas, min_eigs, color="r")
    plt.show()
