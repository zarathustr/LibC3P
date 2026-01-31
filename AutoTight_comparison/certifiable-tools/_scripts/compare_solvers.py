import os
import time

import numpy as np
import matplotlib.pylab as plt

from cert_tools.sdp_solvers import solve_feasibility_sdp
from cert_tools.eopt_solvers import solve_eopt

np.set_printoptions(precision=2)

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def save_all(figname):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    def multipage(filename, figs=None, dpi=200):
        pp = PdfPages(filename)
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.tight_layout()
            fig.savefig(pp, format="pdf")
        pp.close()

    multipage(figname)


if __name__ == "__main__":
    import pickle

    plot = False

    exploit_centered = False

    # 2D-RO with redundant constraints, one position
    # problem_name = f"test_prob_11L"

    # 2D-RO with redundant constraints, 3 positions
    # problem_name = f"test_prob_12G"

    # 2D-RO with redundant constraints, 10 positions
    problem_name = f"test_prob_13Gc"

    fname = os.path.join(root_dir, "_test", f"{problem_name}.pkl")
    with open(fname, "rb") as f:
        data = pickle.load(f)

    # info_qp = solve_eopt_qp(**data, verbose=2)
    t1 = time.time()
    x, info_cuts = solve_eopt(
        **data,
        exploit_centered=exploit_centered,
        plot=plot,
        method="sub",
        use_null=True,
    )  # , x_init=np.array(info_feas["yvals"]))
    print(f"------- time for sub: {(time.time() - t1)*1e3:.0f} ms")

    t1 = time.time()
    H, info_feas = solve_feasibility_sdp(
        **data, adjust=False, soft_epsilon=False, verbose=False
    )
    print(f"------- time for SDP: {(time.time() - t1)*1e3:.0f} ms")
    if plot and (H is not None):
        eigs = np.linalg.eigvalsh(H.toarray())[:3]

        fig, ax = plt.subplots()
        ax.matshow(H.toarray())
        ax.set_title(f"H SDP \n{eigs}")
        print("minimum eigenvalues:", eigs)
    elif H is None:
        print("Warning: SDP didn't solve")

    t1 = time.time()
    x, info_cuts = solve_eopt(
        **data, exploit_centered=exploit_centered, plot=plot, method="cuts"
    )  # , x_init=np.array(info_feas["yvals"]))
    print(f"------- time for cuts: {(time.time() - t1)*1e3:.0f} ms")

    if plot:
        fig, ax = plt.subplots()
        ax.plot(info_cuts["iter_info"].t_min, color="k")
        ax.plot(info_cuts["iter_info"].min_eig_curr, color="C1")
        ax.plot(info_cuts["iter_info"].t_max, color="k")
        ax.axhline(0, color="k", ls="--")
        ax.set_yscale("symlog")
        ax.set_ylim(None, 1.1)
        # ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.grid()

        H_eopt = info_cuts["H"]
        eigs = np.linalg.eigvalsh(H_eopt)[:3]

        fig, ax = plt.subplots()
        ax.matshow(H_eopt)
        ax.set_title(f"H eOPT \n{eigs}")
        print("minimum eigenvalues:", eigs)

        appendix = "_rmv" if exploit_centered else ""
        figname = os.path.join(root_dir, "_plots", f"{problem_name}{appendix}.pdf")
        save_all(figname)
        print(f"saved all as {figname}")
