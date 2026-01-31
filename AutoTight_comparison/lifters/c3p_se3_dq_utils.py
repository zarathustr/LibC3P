import numpy as np
from scipy.spatial.transform import Rotation as Rot


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_L(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float).reshape(4)
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ],
        dtype=float,
    )


def quat_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float).reshape(4)
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, z, -y],
            [y, -z, w, x],
            [z, y, -x, w],
        ],
        dtype=float,
    )


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return quat_L(q1) @ np.asarray(q2, dtype=float).reshape(4)


def dq_L(dq: np.ndarray) -> np.ndarray:
    dq = np.asarray(dq, dtype=float).reshape(8)
    qr = dq[:4]
    qd = dq[4:]
    Lr = quat_L(qr)
    Ld = quat_L(qd)
    top = np.hstack([Lr, np.zeros((4, 4))])
    bottom = np.hstack([Ld, Lr])
    return np.vstack([top, bottom])


def dq_R(dq: np.ndarray) -> np.ndarray:
    dq = np.asarray(dq, dtype=float).reshape(8)
    qr = dq[:4]
    qd = dq[4:]
    Rr = quat_R(qr)
    Rd = quat_R(qd)
    top = np.hstack([Rr, np.zeros((4, 4))])
    bottom = np.hstack([Rd, Rr])
    return np.vstack([top, bottom])


def dq_mul(dq1: np.ndarray, dq2: np.ndarray) -> np.ndarray:
    return dq_L(dq1) @ np.asarray(dq2, dtype=float).reshape(8)


def se3_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=float) @ np.asarray(B, dtype=float)


def se3_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    Rm = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t
    return Ti


def se3_from_qt(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    Rm = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T = np.eye(4, dtype=float)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def dq_from_qt(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    t = np.asarray(t, dtype=float).reshape(3)
    t_quat = np.array([0.0, t[0], t[1], t[2]], dtype=float)
    qd = 0.5 * quat_mul(t_quat, q)
    return np.hstack([q, qd])


def qt_from_dq(dq: np.ndarray):
    dq = np.asarray(dq, dtype=float).reshape(8)
    qr = quat_normalize(dq[:4])
    qd = dq[4:]
    qd = qd - qr * float(qr @ qd)
    t_quat = 2.0 * quat_mul(qd, quat_conj(qr))
    t = t_quat[1:4]
    return qr, t


def se3_to_dq(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    Rm = T[:3, :3]
    t = T[:3, 3]
    q_xyzw = Rot.from_matrix(Rm).as_quat()
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    return dq_from_qt(q, t)


def rand_quat() -> np.ndarray:
    q = np.random.normal(size=4)
    return quat_normalize(q)


def rand_se3(trans_sigma: float = 1.0) -> np.ndarray:
    q = rand_quat()
    t = np.random.normal(scale=trans_sigma, size=3)
    return se3_from_qt(q, t)


def apply_noise_se3(T: np.ndarray, rot_sigma_rad: float, trans_sigma: float) -> np.ndarray:
    if rot_sigma_rad is None:
        rot_sigma_rad = 0.0
    if trans_sigma is None:
        trans_sigma = 0.0

    w = np.random.normal(scale=float(rot_sigma_rad), size=3)
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        Rn = np.eye(3, dtype=float)
    else:
        axis = w / angle
        Rn = Rot.from_rotvec(axis * angle).as_matrix()
    vn = np.random.normal(scale=float(trans_sigma), size=3)

    N = np.eye(4, dtype=float)
    N[:3, :3] = Rn
    N[:3, 3] = vn
    return N @ np.asarray(T, dtype=float)


def build_T_dq() -> np.ndarray:
    T = np.zeros((8, 64), dtype=float)
    for a in range(8):
        e1 = np.zeros(8)
        e1[a] = 1.0
        for b in range(8):
            e2 = np.zeros(8)
            e2[b] = 1.0
            v = dq_mul(e1, e2)
            T[:, b * 8 + a] = v
    return T


def rot_err_deg(q_est: np.ndarray, q_gt: np.ndarray) -> float:
    q_est = quat_normalize(q_est)
    q_gt = quat_normalize(q_gt)
    c = abs(float(np.dot(q_est, q_gt)))
    c = min(1.0, max(-1.0, c))
    return float(2.0 * np.arccos(c) * 180.0 / np.pi)


def trans_err(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    t_est = np.asarray(t_est, dtype=float).reshape(3)
    t_gt = np.asarray(t_gt, dtype=float).reshape(3)
    return float(np.linalg.norm(t_est - t_gt))

