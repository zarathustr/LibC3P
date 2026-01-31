import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_L(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ],
        dtype=float,
    )


def _quat_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, z, -y],
            [y, -z, w, x],
            [z, y, -x, w],
        ],
        dtype=float,
    )


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return _quat_L(q1) @ q2


def _dq_L(dq: np.ndarray) -> np.ndarray:
    dq = np.asarray(dq, dtype=float).reshape(8)
    qr = dq[:4]
    qd = dq[4:]
    Lr = _quat_L(qr)
    Ld = _quat_L(qd)
    top = np.hstack([Lr, np.zeros((4, 4))])
    bottom = np.hstack([Ld, Lr])
    return np.vstack([top, bottom])


def _dq_R(dq: np.ndarray) -> np.ndarray:
    dq = np.asarray(dq, dtype=float).reshape(8)
    qr = dq[:4]
    qd = dq[4:]
    Rr = _quat_R(qr)
    Rd = _quat_R(qd)
    top = np.hstack([Rr, np.zeros((4, 4))])
    bottom = np.hstack([Rd, Rr])
    return np.vstack([top, bottom])


def _dq_mul(dq1: np.ndarray, dq2: np.ndarray) -> np.ndarray:
    return _dq_L(dq1) @ dq2


def _se3_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _se3_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def _se3_from_qt(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    Rm = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def _dq_from_qt(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    t = np.asarray(t, dtype=float).reshape(3)
    t_quat = np.array([0.0, t[0], t[1], t[2]], dtype=float)
    qd = 0.5 * _quat_mul(t_quat, q)
    return np.hstack([q, qd])


def _qt_from_dq(dq: np.ndarray):
    dq = np.asarray(dq, dtype=float).reshape(8)
    qr = _quat_normalize(dq[:4])
    qd = dq[4:]
    # enforce orthogonality numerically
    qd = qd - qr * float(qr @ qd)
    t_quat = 2.0 * _quat_mul(qd, _quat_conj(qr))
    t = t_quat[1:4]
    return qr, t


def _rand_quat() -> np.ndarray:
    q = np.random.normal(size=4)
    return _quat_normalize(q)


def _rand_se3(trans_sigma=1.0) -> np.ndarray:
    q = _rand_quat()
    t = np.random.normal(scale=trans_sigma, size=3)
    return _se3_from_qt(q, t)


def _apply_noise_se3(T: np.ndarray, rot_sigma_rad: float, trans_sigma: float) -> np.ndarray:
    """
    Left-multiplicative noise:
        T_noisy = Exp([w, v]) * T
    Here we use small-angle approximation for rotation via axis-angle.
    """
    if rot_sigma_rad is None:
        rot_sigma_rad = 0.0
    if trans_sigma is None:
        trans_sigma = 0.0

    w = np.random.normal(scale=rot_sigma_rad, size=3)
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        Rn = np.eye(3)
    else:
        axis = w / angle
        Rn = Rot.from_rotvec(axis * angle).as_matrix()
    vn = np.random.normal(scale=trans_sigma, size=3)

    N = np.eye(4)
    N[:3, :3] = Rn
    N[:3, 3] = vn
    return N @ T


class C3PSE3AXBYZCWDLifter(StateLifter):
    """
    SE(3) C3P of type AXBY = ZCWD formulated via unit dual quaternions.

    Unknowns:
        X, Y, Z, W in SE(3) represented as (q, t), with q in R^4 unit quaternion, t in R^3.

    Lifted QCQP variables:
        dqX, dqY, dqZ, dqW in R^8 (dual quaternions)
        uXY = kron(dqY, dqX) in R^64
        uWZ = kron(dqW, dqZ) in R^64

    For each measurement i with known dual quaternions (A_i, B_i, C_i, D_i):
        A_i * X * B_i * Y  -  Z * C_i * W * D_i = 0
    is written as a linear residual in (uXY, uWZ):
        r_i = S_i uXY - R_i uWZ in R^8
    and we solve min 0.5 * sum_i ||r_i||^2 subject to quadratic constraints.
    """

    LEVELS = ["no"]
    PARAM_LEVELS = ["no"]

    # Use cost tightness by default. You may switch to "rank" if desired.
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
        self.rot_noise_rad = rot_noise_rad
        self.trans_noise = trans_noise

        # Precompute the constant bilinear multiplication matrix for dual quaternions:
        # dq1*dq2 = T_dq @ kron(dq2, dq1)
        self.T_dq = self._build_T_dq()

        # Storage for noise-free SE(3) measurement tuples
        self._A0 = None
        self._B0 = None
        self._C0 = None
        self._D0 = None

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
        # Study all variables at once
        return [["h", "x", "y", "z", "w", "uXY", "uWZ"]]

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = {
                "h": 1,
                "x": 8,
                "y": 8,
                "z": 8,
                "w": 8,
                "uXY": 64,
                "uWZ": 64,
            }
        return self.var_dict_

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.sample_theta()
        return self.theta_

    def get_p(self, parameters=None, var_subset=None):
        # No extra parameters
        return np.array([1.0])

    def sample_theta(self):
        # theta is [qX(4), tX(3), qY(4), tY(3), qZ(4), tZ(3), qW(4), tW(3)]
        qX, qY, qZ, qW = _rand_quat(), _rand_quat(), _rand_quat(), _rand_quat()
        tX = np.random.normal(scale=1.0, size=3)
        tY = np.random.normal(scale=1.0, size=3)
        tZ = np.random.normal(scale=1.0, size=3)
        tW = np.random.normal(scale=1.0, size=3)
        return np.hstack([qX, tX, qY, tY, qZ, tZ, qW, tW])

    def get_vec_around_gt(self, delta: float = 0.0):
        if delta == 0:
            return self.theta.copy()
        th = self.theta.copy()
        # Add small perturbations to quaternion components and translations
        for k in [0, 7, 14, 21]:
            q = th[k : k + 4] + np.random.normal(scale=delta, size=4)
            th[k : k + 4] = _quat_normalize(q)
            th[k + 4 : k + 7] = th[k + 4 : k + 7] + np.random.normal(scale=delta, size=3)
        return th

    def generate_random_setup(self):
        """
        Generate a consistent noise-free dataset for the current ground truth (X,Y,Z,W).
        We sample A_i, B_i, C_i randomly and compute D_i so that AXBY=ZCWD holds exactly.
        """
        # Ensure ground truth exists
        th = self.theta

        qX, tX, qY, tY, qZ, tZ, qW, tW = self._split_theta(th)
        TX = _se3_from_qt(qX, tX)
        TY = _se3_from_qt(qY, tY)
        TZ = _se3_from_qt(qZ, tZ)
        TW = _se3_from_qt(qW, tW)

        A0 = []
        B0 = []
        C0 = []
        D0 = []

        for _ in range(self.n_measurements):
            TA = _rand_se3(trans_sigma=1.0)
            TB = _rand_se3(trans_sigma=1.0)
            TC = _rand_se3(trans_sigma=1.0)

            # Solve for TD so that: TA*TX*TB*TY = TZ*TC*TW*TD
            # TD = (TZ*TC*TW)^{-1} * (TA*TX*TB*TY)
            left = _se3_mul(_se3_mul(_se3_mul(TA, TX), TB), TY)
            right_prefix = _se3_mul(_se3_mul(TZ, TC), TW)
            TD = _se3_mul(_se3_inv(right_prefix), left)

            A0.append(TA)
            B0.append(TB)
            C0.append(TC)
            D0.append(TD)

        self._A0 = np.stack(A0, axis=0)
        self._B0 = np.stack(B0, axis=0)
        self._C0 = np.stack(C0, axis=0)
        self._D0 = np.stack(D0, axis=0)

        # no extra parameters
        self.parameters = [1.0]
        self.y_ = None
        return

    def _split_theta(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        assert theta.size == 28
        qX = _quat_normalize(theta[0:4]); tX = theta[4:7]
        qY = _quat_normalize(theta[7:11]); tY = theta[11:14]
        qZ = _quat_normalize(theta[14:18]); tZ = theta[18:21]
        qW = _quat_normalize(theta[21:25]); tW = theta[25:28]
        return qX, tX, qY, tY, qZ, tZ, qW, tW

    def _dq_from_theta(self, theta: np.ndarray):
        qX, tX, qY, tY, qZ, tZ, qW, tW = self._split_theta(theta)
        dqX = _dq_from_qt(qX, tX)
        dqY = _dq_from_qt(qY, tY)
        dqZ = _dq_from_qt(qZ, tZ)
        dqW = _dq_from_qt(qW, tW)
        return dqX, dqY, dqZ, dqW

    def _se3_to_dq(self, T: np.ndarray) -> np.ndarray:
        Rm = T[:3, :3]
        t = T[:3, 3]
        q_xyzw = Rot.from_matrix(Rm).as_quat()  # [x,y,z,w]
        q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        return _dq_from_qt(q, t)

    def _build_T_dq(self) -> np.ndarray:
        T = np.zeros((8, 64), dtype=float)
        for a in range(8):
            P = np.zeros(8); P[a] = 1.0
            for b in range(8):
                Q = np.zeros(8); Q[b] = 1.0
                v = _dq_mul(P, Q)
                T[:, b * 8 + a] = v
        return T

    def _coeff_matrices(self, dqA: np.ndarray, dqB: np.ndarray, dqC: np.ndarray, dqD: np.ndarray):
        """
        Build the linear residual matrices S, R in:
            r = S uXY - R uWZ
        """
        M = _dq_L(dqA) @ _dq_R(dqB)  # maps dqX -> dqA*dqX*dqB
        N = _dq_L(dqC) @ _dq_R(dqD)  # maps dqW -> dqC*dqW*dqD

        S = self.T_dq @ np.kron(np.eye(8), M)          # 8 x 64
        R = self.T_dq @ np.kron(N, np.eye(8))          # 8 x 64
        return S, R

    def get_Q(self, noise: float = None, output_poly: bool = False, use_cliques: list = []):
        """
        Build Q from internally stored SE(3) measurements with noise.
        """
        if (self._A0 is None) or (self._B0 is None) or (self._C0 is None) or (self._D0 is None):
            self.generate_random_setup()

        # allow overriding stored noise if caller provides
        if noise is not None:
            rot_sigma = float(noise)
            trans_sigma = float(noise)
        else:
            rot_sigma = float(self.rot_noise_rad)
            trans_sigma = float(self.trans_noise)

        # Generate noisy measurements once per call
        A = np.empty((self.n_measurements, 8))
        B = np.empty((self.n_measurements, 8))
        C = np.empty((self.n_measurements, 8))
        D = np.empty((self.n_measurements, 8))

        for i in range(self.n_measurements):
            TA = _apply_noise_se3(self._A0[i], rot_sigma, trans_sigma)
            TB = _apply_noise_se3(self._B0[i], rot_sigma, trans_sigma)
            TC = _apply_noise_se3(self._C0[i], rot_sigma, trans_sigma)
            TD = _apply_noise_se3(self._D0[i], rot_sigma, trans_sigma)
            A[i] = self._se3_to_dq(TA)
            B[i] = self._se3_to_dq(TB)
            C[i] = self._se3_to_dq(TC)
            D[i] = self._se3_to_dq(TD)

        self.y_ = {"A": A, "B": B, "C": C, "D": D}

        Q = self.get_Q_from_y(self.y_, output_poly=output_poly, use_cliques=use_cliques)
        return Q, self.y_

    def get_Q_from_y(self, y, output_poly: bool = False, use_cliques: list = []):
        from poly_matrix.poly_matrix import PolyMatrix

        A = y["A"]; B = y["B"]; C = y["C"]; D = y["D"]
        assert A.shape[0] == self.n_measurements

        # Build quadratic cost in uXY and uWZ only
        Qpoly = PolyMatrix(symmetric=True)

        Q_uXY_uXY = np.zeros((64, 64), dtype=float)
        Q_uWZ_uWZ = np.zeros((64, 64), dtype=float)
        Q_uXY_uWZ = np.zeros((64, 64), dtype=float)

        for i in range(self.n_measurements):
            S, R = self._coeff_matrices(A[i], B[i], C[i], D[i])
            Q_uXY_uXY += S.T @ S
            Q_uWZ_uWZ += R.T @ R
            Q_uXY_uWZ += -S.T @ R

        Qpoly["uXY", "uXY"] = Q_uXY_uXY
        Qpoly["uWZ", "uWZ"] = Q_uWZ_uWZ
        Qpoly["uXY", "uWZ"] = Q_uXY_uWZ

        if output_poly:
            return 0.5 * Qpoly
        Q_sparse = 0.5 * Qpoly.get_matrix(variables=self.var_dict)
        return Q_sparse

    def get_x(self, theta, var_subset=None) -> np.ndarray:
        """
        Lift theta -> x = [h; dqX; dqY; dqZ; dqW; uXY; uWZ]
        """
        if var_subset is None:
            var_subset = list(self.var_dict.keys())
        elif isinstance(var_subset, dict):
            var_subset = list(var_subset.keys())

        dqX, dqY, dqZ, dqW = self._dq_from_theta(theta)
        uXY = np.kron(dqY, dqX)
        uWZ = np.kron(dqW, dqZ)

        blocks = {
            "h": np.array([1.0]),
            "x": dqX,
            "y": dqY,
            "z": dqZ,
            "w": dqW,
            "uXY": uXY,
            "uWZ": uWZ,
        }

        x_list = []
        for k in var_subset:
            x_list.append(blocks[k].reshape(-1))
        return np.concatenate(x_list, axis=0)

    def get_theta(self, x: np.ndarray):
        """
        Extract theta = [qX,tX,qY,tY,qZ,tZ,qW,tW] from a lifted x vector.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        # fixed ordering based on var_dict
        idx = 0
        idx += 1  # h
        dqX = x[idx : idx + 8]; idx += 8
        dqY = x[idx : idx + 8]; idx += 8
        dqZ = x[idx : idx + 8]; idx += 8
        dqW = x[idx : idx + 8]; idx += 8
        qX, tX = _qt_from_dq(dqX)
        qY, tY = _qt_from_dq(dqY)
        qZ, tZ = _qt_from_dq(dqZ)
        qW, tW = _qt_from_dq(dqW)
        return np.hstack([qX, tX, qY, tY, qZ, tZ, qW, tW])

    def get_error(self, theta_hat):
        """
        Compute rotation and translation errors for X,Y,Z,W against the stored ground truth.
        """
        th_gt = self.theta
        th_hat = np.asarray(theta_hat, dtype=float).reshape(-1)
        if th_hat.size != 28:
            # If theta_hat is a lifted x vector, convert it.
            th_hat = self.get_theta(th_hat)

        qX_gt, tX_gt, qY_gt, tY_gt, qZ_gt, tZ_gt, qW_gt, tW_gt = self._split_theta(th_gt)
        qX, tX, qY, tY, qZ, tZ, qW, tW = self._split_theta(th_hat)

        def rot_err(q1, q2):
            q1 = _quat_normalize(q1); q2 = _quat_normalize(q2)
            c = abs(float(np.dot(q1, q2)))
            c = min(1.0, max(-1.0, c))
            return 2.0 * np.arccos(c) * 180.0 / np.pi

        def trans_err(t1, t2):
            return float(np.linalg.norm(np.asarray(t1) - np.asarray(t2)))

        rot_errs = np.array([
            rot_err(qX, qX_gt),
            rot_err(qY, qY_gt),
            rot_err(qZ, qZ_gt),
            rot_err(qW, qW_gt),
        ])
        trans_errs = np.array([
            trans_err(tX, tX_gt),
            trans_err(tY, tY_gt),
            trans_err(tZ, tZ_gt),
            trans_err(tW, tW_gt),
        ])

        return {
            "rot_mean_deg": float(rot_errs.mean()),
            "rot_max_deg": float(rot_errs.max()),
            "trans_mean": float(trans_errs.mean()),
            "trans_max": float(trans_errs.max()),
        }

    def get_cost(self, theta, y) -> float:
        Q = self.get_Q_from_y(y, output_poly=False)
        x = self.get_x(theta)
        return float(x.T @ (Q @ x))

    def residuals(self, theta, y) -> np.ndarray:
        """
        Direct residual stacking (8 per measurement), for local NLS.
        """
        dqX, dqY, dqZ, dqW = self._dq_from_theta(theta)
        A = y["A"]; B = y["B"]; C = y["C"]; D = y["D"]
        res_list = []
        for i in range(self.n_measurements):
            left = _dq_mul(_dq_mul(_dq_mul(A[i], dqX), B[i]), dqY)
            right = _dq_mul(_dq_mul(_dq_mul(dqZ, C[i]), dqW), D[i])
            res_list.append(left - right)
        return np.concatenate(res_list, axis=0)

    def local_solver(self, t0, y, verbose=False, method="lsq"):
        """
        Local nonlinear least-squares over theta (q,t for each transform),
        using SciPy's least_squares with quaternion normalization in the residual.
        """
        t0 = np.asarray(t0, dtype=float).reshape(-1)

        def fun(th):
            # enforce quaternion normalization explicitly
            th = th.copy()
            for k in [0, 7, 14, 21]:
                th[k : k + 4] = _quat_normalize(th[k : k + 4])
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
        theta_hat = res.x
        for k in [0, 7, 14, 21]:
            theta_hat[k : k + 4] = _quat_normalize(theta_hat[k : k + 4])

        info = {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "cost": float(res.cost),
            "max res": float(np.max(np.abs(res.fun))) if res.fun.size else 0.0,
        }
        return theta_hat, info, float(res.cost)

    def get_A_known(self, var_dict=None, output_poly=False):
        """
        Known quadratic constraints:
            Unit dual quaternion constraints for x,y,z,w:
                ||qr||^2 = h^2
                qr^T qd = 0
            Bilinear lifting constraints:
                uXY = kron(y, x)
                uWZ = kron(w, z)
        """
        if var_dict is None:
            var_dict = self.var_dict

        A_list = []
        from poly_matrix.poly_matrix import PolyMatrix

        # Helper matrices
        E_real = np.diag([1, 1, 1, 1, 0, 0, 0, 0]).astype(float)
        E_ortho = np.zeros((8, 8), dtype=float)
        E_ortho[:4, 4:] = np.eye(4)
        E_ortho[4:, :4] = np.eye(4)

        for key in ["x", "y", "z", "w"]:
            if key in var_dict:
                # ||qr||^2 - h^2 = 0
                Ai = PolyMatrix(symmetric=True)
                Ai[key, key] = E_real
                Ai["h", "h"] = -1.0
                self.test_and_add(A_list, Ai, output_poly=output_poly)

                # qr^T qd = 0  (scaled by 2, still valid)
                Ai = PolyMatrix(symmetric=True)
                Ai[key, key] = E_ortho
                self.test_and_add(A_list, Ai, output_poly=output_poly)

        # Bilinear constraints uXY_k = y_a * x_b  ->  h*uXY_k - y_a*x_b = 0
        if ("uXY" in var_dict) and ("x" in var_dict) and ("y" in var_dict):
            for a in range(8):
                for b in range(8):
                    k = a * 8 + b
                    Ai = PolyMatrix(symmetric=True)

                    # h * uXY_k
                    e = np.zeros((1, 64))
                    e[0, k] = 1.0
                    Ai["h", "uXY"] = 0.5 * e

                    # - y_a * x_b
                    M = np.zeros((8, 8))
                    M[a, b] = -0.5
                    Ai["y", "x"] = M

                    self.test_and_add(A_list, Ai, output_poly=output_poly)

        # Bilinear constraints uWZ_k = w_a * z_b  ->  h*uWZ_k - w_a*z_b = 0
        if ("uWZ" in var_dict) and ("w" in var_dict) and ("z" in var_dict):
            for a in range(8):
                for b in range(8):
                    k = a * 8 + b
                    Ai = PolyMatrix(symmetric=True)

                    e = np.zeros((1, 64))
                    e[0, k] = 1.0
                    Ai["h", "uWZ"] = 0.5 * e

                    M = np.zeros((8, 8))
                    M[a, b] = -0.5
                    Ai["w", "z"] = M

                    self.test_and_add(A_list, Ai, output_poly=output_poly)

        return A_list

    def __repr__(self):
        return f"c3p_se3_axby_zcwd_N{self.n_measurements}"
