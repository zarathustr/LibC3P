import numpy as np
from scipy.optimize import least_squares

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix

from lifters.c3p_se3_dq_utils import (
    quat_normalize,
    dq_L,
    dq_R,
    dq_mul,
    se3_from_qt,
    se3_inv,
    se3_mul,
    rand_quat,
    rand_se3,
    apply_noise_se3,
    se3_to_dq,
    dq_from_qt,
    qt_from_dq,
    rot_err_deg,
    trans_err,
)


class C3PSE3AXXBLifter(StateLifter):
    """
    SE(3) C3P of type AX = XB via unit dual quaternions.

    Unknown:
        X in SE(3) as dqX in R^8.

    Constraint per measurement i:
        dqA_i * dqX = dqX * dqB_i
    Linear residual:
        (L(dqA_i) - R(dqB_i)) dqX = 0
    """

    LEVELS = ["no"]
    PARAM_LEVELS = ["no"]
    TIGHTNESS = "cost"

    def __init__(
        self,
        n_measurements: int = 10,
        rot_noise_rad: float = 0.0,
        trans_noise: float = 0.0,
        level="no",
        param_level="no",
        d=3,
        variable_list=None,
        robust=False,
        n_outliers=0,
    ):
        self.n_measurements = int(n_measurements)
        self.rot_noise_rad = float(rot_noise_rad)
        self.trans_noise = float(trans_noise)

        self._A0 = None
        self._B0 = None

        super().__init__(
            level=level,
            param_level=param_level,
            d=d,
            variable_list=variable_list,
            robust=robust,
            n_outliers=n_outliers,
        )

    @property
    def VARIABLE_LIST(self):
        return [["h", "x"]]

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = {"h": 1, "x": 8}
        return self.var_dict_

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.sample_theta()
        return self.theta_

    def get_p(self, parameters=None, var_subset=None):
        return np.array([1.0])

    def sample_theta(self):
        qX = rand_quat()
        tX = np.random.normal(scale=1.0, size=3)
        return np.hstack([qX, tX])

    def _split_theta(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        assert theta.size == 7
        qX = quat_normalize(theta[0:4])
        tX = theta[4:7]
        return qX, tX

    def _dq_from_theta(self, theta: np.ndarray) -> np.ndarray:
        qX, tX = self._split_theta(theta)
        return dq_from_qt(qX, tX)

    def generate_random_setup(self):
        """
        Sample A_i randomly and set B_i = X^{-1} A_i X so that AX=XB holds exactly.
        """
        qX, tX = self._split_theta(self.theta)
        TX = se3_from_qt(qX, tX)

        A0 = []
        B0 = []
        for _ in range(self.n_measurements):
            TA = rand_se3(trans_sigma=1.0)
            TB = se3_mul(se3_mul(se3_inv(TX), TA), TX)
            A0.append(TA)
            B0.append(TB)

        self._A0 = np.stack(A0, axis=0)
        self._B0 = np.stack(B0, axis=0)
        self.parameters = [1.0]
        self.y_ = None

    def get_Q(self, noise: float = None, output_poly: bool = False, use_cliques: list = []):
        if (self._A0 is None) or (self._B0 is None):
            self.generate_random_setup()

        if noise is not None:
            rot_sigma = float(noise)
            trans_sigma = float(noise)
        else:
            rot_sigma = float(self.rot_noise_rad)
            trans_sigma = float(self.trans_noise)

        A = np.empty((self.n_measurements, 8))
        B = np.empty((self.n_measurements, 8))
        for i in range(self.n_measurements):
            TA = apply_noise_se3(self._A0[i], rot_sigma, trans_sigma)
            TB = apply_noise_se3(self._B0[i], rot_sigma, trans_sigma)
            A[i] = se3_to_dq(TA)
            B[i] = se3_to_dq(TB)

        self.y_ = {"A": A, "B": B}
        Q = self.get_Q_from_y(self.y_, output_poly=output_poly, use_cliques=use_cliques)
        return Q, self.y_

    def get_Q_from_y(self, y, output_poly: bool = False, use_cliques: list = []):
        A = y["A"]
        B = y["B"]

        if len(use_cliques):
            js = use_cliques
        else:
            js = list(range(self.n_measurements))

        Qpoly = PolyMatrix(symmetric=True)
        Q_xx = np.zeros((8, 8), dtype=float)

        for i in js:
            Mi = dq_L(A[i]) - dq_R(B[i])
            Q_xx += Mi.T @ Mi

        Qpoly["x", "x"] = Q_xx
        if output_poly:
            return 0.5 * Qpoly
        return 0.5 * Qpoly.get_matrix(self.var_dict)

    def get_x(self, theta=None, parameters=None, var_subset=None) -> np.ndarray:
        if theta is None:
            theta = self.theta
        _ = parameters

        if var_subset is None:
            var_subset = list(self.var_dict.keys())
        elif isinstance(var_subset, dict):
            var_subset = list(var_subset.keys())

        dqX = self._dq_from_theta(theta)
        blocks = {"h": np.array([1.0]), "x": dqX}
        return np.concatenate([blocks[k].reshape(-1) for k in var_subset], axis=0)

    def get_theta(self, x: np.ndarray):
        x = np.asarray(x, dtype=float).reshape(-1)
        idx = 0
        idx += 1
        dqX = x[idx : idx + 8]
        qX, tX = qt_from_dq(dqX)
        return np.hstack([qX, tX])

    def get_error(self, theta_hat):
        th_hat = np.asarray(theta_hat, dtype=float).reshape(-1)
        if th_hat.size != 7:
            th_hat = self.get_theta(th_hat)

        q_gt, t_gt = self._split_theta(self.theta)
        q, t = self._split_theta(th_hat)
        return {"rot_deg": rot_err_deg(q, q_gt), "trans": trans_err(t, t_gt)}

    def residuals(self, theta, y) -> np.ndarray:
        dqX = self._dq_from_theta(theta)
        A = y["A"]
        B = y["B"]
        res = []
        for i in range(self.n_measurements):
            left = dq_mul(A[i], dqX)
            right = dq_mul(dqX, B[i])
            res.append(left - right)
        return np.concatenate(res, axis=0)

    def local_solver(self, t0, y, verbose=False):
        t0 = np.asarray(t0, dtype=float).reshape(-1)

        def fun(th):
            th = th.copy()
            th[0:4] = quat_normalize(th[0:4])
            return self.residuals(th, y)

        res = least_squares(
            fun,
            t0,
            method="trf",
            max_nfev=200,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            verbose=2 if verbose else 0,
        )
        th = res.x
        th[0:4] = quat_normalize(th[0:4])
        info = {"success": bool(res.success), "cost": float(res.cost), "nfev": int(res.nfev)}
        return th, info, float(res.cost)

    def test_and_add(self, A_list, Ai, var_dict, output_poly: bool):
        x = self.get_x(theta=self.theta, parameters=None, var_subset=var_dict)
        Ai_sparse = Ai.get_matrix(var_dict)
        err = float(x.T @ (Ai_sparse @ x))
        if abs(err) > 1e-8:
            raise ValueError(f"Known constraint check failed: x^T A x = {err}")
        if output_poly:
            A_list.append(Ai)
        else:
            A_list.append(Ai_sparse)

    def get_A_known(self, var_dict=None, output_poly=False):
        if var_dict is None:
            var_dict = self.var_dict

        A_list = []
        if "x" in var_dict:
            E_real = np.diag([1, 1, 1, 1, 0, 0, 0, 0]).astype(float)
            E_ortho = np.zeros((8, 8), dtype=float)
            E_ortho[:4, 4:] = np.eye(4)
            E_ortho[4:, :4] = np.eye(4)

            Ai = PolyMatrix(symmetric=True)
            Ai["x", "x"] = E_real
            Ai["h", "h"] = -1.0
            self.test_and_add(A_list, Ai, var_dict, output_poly)

            Ai = PolyMatrix(symmetric=True)
            Ai["x", "x"] = E_ortho
            self.test_and_add(A_list, Ai, var_dict, output_poly)

        return A_list

    def get_cost(self, theta, y) -> float:
        Q = self.get_Q_from_y(y, output_poly=False)
        x = self.get_x(theta=theta)
        return float(x.T @ (Q @ x))

    def __repr__(self):
        return f"c3p_se3_axxb_N{self.n_measurements}"

