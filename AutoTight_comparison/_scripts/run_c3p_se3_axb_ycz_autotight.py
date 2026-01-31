import numpy as np

from auto_template.learner import Learner
from lifters.c3p_se3_axb_ycz_lifter import C3PSE3AXB_YCZLifter


def main():
    np.random.seed(0)

    lifter = C3PSE3AXB_YCZLifter(
        n_measurements=6,
        rot_noise_rad=1e-3,
        trans_noise=1e-3,
    )

    learner = Learner(
        lifter=lifter,
        variable_list=lifter.variable_list,
        apply_templates=True,
        n_inits=3,
        use_known=True,
        use_incremental=True,
    )

    learner.run(verbose=True, plot=False)

    data = {}
    learner.find_global_solution(data_dict=data)

    print("\n===== Results: AXB=YCZ =====")
    if "global theta" in data:
        th = data["global theta"]
        print("Recovered theta [qX,tX,qY,tY,qZ,tZ]:")
        print(th)
        errs = lifter.get_error(th)
        for k, v in errs.items():
            print(f"{k}: {v}")
    if "global cost" in data:
        print("Global cost:", data["global cost"])


if __name__ == "__main__":
    main()

