import numpy as np

from auto_template.learner import Learner
from lifters.c3p_se3_axby_zcwd_lifter import C3PSE3AXBYZCWDLifter


def main():
    np.random.seed(0)

    lifter = C3PSE3AXBYZCWDLifter(
        n_measurements=8,
        rot_noise_rad=1e-3,
        trans_noise=1e-3,
        variable_list=None,
    )

    learner = Learner(
        lifter=lifter,
        variable_list=lifter.variable_list,
        apply_templates=False,
        n_inits=3,
        use_known=True,
        use_incremental=True,
    )

    learner.run(verbose=True, plot=False)

    data = {}
    learner.find_global_solution(data_dict=data)

    print("\n===== Results =====")
    if "global theta" in data:
        th = data["global theta"]
        print("Recovered theta (q,t for X,Y,Z,W):")
        print(th)
        errs = lifter.get_error(th)
        print("\nErrors versus ground truth:")
        for k, v in errs.items():
            print(f"{k}: {v:.6g}")
    if "global cost" in data:
        print(f"\nGlobal cost: {data['global cost']:.6e}")


if __name__ == "__main__":
    main()

