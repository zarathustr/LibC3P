# Test Problem Descriptions

This file describes the test problems that appear in this test directory. All test problems store dictionaries with the following variables:

| Variable | Description |
| ---------| ------------|
| Q | Quadratic cost matrix|
| Q_norm | Normalized/preconditioned quadratic cost matrix |
| Constraints | List of 2-tuples (A,b) that describe constraints $\left< A, X\right> = b$|
| adjust | 2-tuple with scale and offset for preconditioning Q | 
| X | Solution to SDP|
| x_cand | candidate solution to QCQP |
|cost| optimal SDP cost|


| Filename | Description |
| ---------| ------------|
| test_prob_1.pkl | 1-pose localization problem with isotropic weights. Tight without redundant constraints. No redundant constraints present (Just orthogonality and homogenizing constraints)|
| test_prob_2.pkl | Same as test_prob_1.pkl, but with redundant constraints included|
| test_prob_3.pkl | 1-pose stereo localization (map to euclidean). Not-tight without redundant constraints. 2 m standoff 0.5 m bounding box on 10 landmarks.|
| test_prob_4.pkl | Same as test_prob_3.pkl, but with redundant constraints.|
| test_prob_5.pkl | 10 pose SLAM on dataset 3 with variance 6e-2 on relative-pose measurements. No redundant constraints, not tight|
| test_prob_6.pkl | Same as test_prob_5.pkl, but with redundant constraints. Should be tight.|
| test_prob_7.pkl | Same as test_prob_4, but with 5 poses and 30 landmarks | 
| test_prob_8.pkl* | Sectic polynomial with 1 local minimum, 1 global minimum. Created by `constraint_learning/examples/poly6_lifter.py`  | 
| test_prob_9.pkl* | Sectic polynomial with 2 local minima, 1 global minimum. Created by `constraint_learning/examples/poly6_lifter.py` | 
| test_prob_10.pkl* | Range-only localization with 4 anchors, 1 position, minimal quadratic  substitution (no redundant constraints). Created by `constraint_learning/examples/ro_lifter.py` | 
| test_prob_11.pkl* | Range-only localization with 4 anchors, 1 position, full quadratic substitution (with redundant constraints). Created by `constraint_learning/examples/ro_lifter.py` | 
| test_prob_12.pkl* | Same as test_prob_11, but with 3 positions. | 
| test_prob_13.pkl* | Same as test_prob_11, but with 10 positions.|
| test_prob_14.pkl* | Wahba problem, with 4 landmarks in 3 dimensions, with total 21 constraints (including redundant). Created by `constraint_learning/examples/robust_lifter.py` |
| test_prob_15.pkl* | robust Wahba problem, with 5 landmarks, 1 outlier, in 3 dimensions, with total 1771 constraints (including redundant). Created by `constraint_learning/examples/robust_lifter.py`|
| test_prob_16.pkl* | Quartic polynomial with 1 local minimum, 1 global minimum. Created by `constraint_learning/examples/poly4_lifter.py` |

*these files may come in variants "G/L" for global / local candidate solutions, and some of them have "c" for the centered version.
