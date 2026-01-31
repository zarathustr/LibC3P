import os
import pickle
import time
from pathlib import Path

import numpy as np

from auto_template.learner import Learner
from lifters.range_only_lifters import RangeOnlyLocLifter


def main() -> None:
    """
    Quick end-to-end smoke demo that does not require a MOSEK license.

    It runs a tiny Range-Only localization lifter through the Learner once and
    writes a pickle artifact under `_results_test_quick/`.
    """

    # Prefer an open-source solver unless the user explicitly requests MOSEK.
    os.environ.setdefault("CERT_TOOLS_CVXPY_FALLBACK_SOLVER", "CVXOPT")
    os.environ.setdefault("CERT_TOOLS_SCS_MAX_ITERS", "5000")

    out_dir = Path("_results_test_quick")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    lifter = RangeOnlyLocLifter(n_positions=2, n_landmarks=5, d=3, level="no")
    learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)

    t0 = time.time()
    dict_list, success = learner.run(verbose=False, plot=False)
    dt = time.time() - t0

    print(f"lifter: {lifter}")
    print(f"tightness_success: {success}")
    if not success:
        print(
            "note: tightness_success=False usually means the SDP solver could not certify tightness "
            "(common without a MOSEK license / when using fallback solvers). "
            "This demo is still considered a PASS if it runs end-to-end and writes the output pickle."
        )
    print(f"n_steps: {len(dict_list)}")
    print(f"elapsed_s: {dt:.2f}")

    out_path = out_dir / f"learner_{str(lifter).replace('/', '_')}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(learner, f)
        pickle.dump(dict_list, f)
        pickle.dump({"success": success, "elapsed_s": dt}, f)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
