#include "solvers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "lm.h"
#include "se3.h"

namespace pem {

namespace {

using Vec3 = arma::vec::fixed<3>;
using Vec4 = arma::vec::fixed<4>;
using Vec6 = arma::vec::fixed<6>;
using Mat3 = arma::mat::fixed<3, 3>;
using Mat4 = arma::mat::fixed<4, 4>;
using Mat8 = arma::mat::fixed<8, 8>;
using Quat = Vec4;  // [w x y z]^T

static inline double clamp01(double x) { return std::max(-1.0, std::min(1.0, x)); }

static inline arma::vec flattenMat4(const Mat4& M) {
    return arma::vectorise(arma::mat(M));
}

static inline Mat3 I3() {
    Mat3 I;
    I.eye();
    return I;
}

static inline Mat3 scaleToSO3(const Mat3& Min) {
    // The homogeneous SVD solution recovers vec(R) only up to a global scale.
    // For an ideal case Min = s * R, det(Min) = s^3 det(R) = s^3, so
    //     R = (sgn(det(Min)) / cbrt(|det(Min)|)) * Min.
    double d = arma::det(Min);
    if (!std::isfinite(d) || std::abs(d) < 1e-12) {
        return Min;
    }
    double s = (d >= 0.0 ? 1.0 : -1.0) / std::cbrt(std::abs(d));
    return s * Min;
}

static inline Mat3 vec9ToRot(const arma::vec& v9) {
    Mat3 M;
    M = arma::reshape(v9, 3, 3);
    M = scaleToSO3(M);
    return orthonormalizeRot(M);
}

static inline bool solveLeastSquares(arma::vec& x, const arma::mat& A, const arma::vec& b) {
    // Armadillo's solve() supports least-squares for non-square systems.
    // If it fails due to rank issues, fall back to pinv.
    bool ok = arma::solve(x, A, b);
    if (!ok) {
        x = arma::pinv(A) * b;
    }
    return ok;
}

static inline Mat3 solveRotationAXXB_SVD(const std::vector<Mat3>& RA, const std::vector<Mat3>& RB) {
    const int N = static_cast<int>(RA.size());
    arma::mat W(9 * N, 9, arma::fill::zeros);

    arma::mat I = arma::mat(I3());
    for (int i = 0; i < N; ++i) {
        arma::mat Wi = arma::kron(I, arma::mat(RA[i])) - arma::kron(arma::mat(RB[i].t()), I);
        W.submat(9 * i, 0, 9 * i + 8, 8) = Wi;
    }

    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, W);
    arma::vec v = V.col(V.n_cols - 1);
    return vec9ToRot(v);
}

static inline std::pair<Mat3, Mat3> solveRotationAXYB_SVD(const std::vector<Mat3>& RA, const std::vector<Mat3>& RB) {
    const int N = static_cast<int>(RA.size());
    arma::mat W(9 * N, 18, arma::fill::zeros);

    arma::mat I = arma::mat(I3());
    for (int i = 0; i < N; ++i) {
        arma::mat L = arma::kron(I, arma::mat(RA[i]));
        arma::mat R = arma::kron(arma::mat(RB[i].t()), I);
        arma::mat Wi(9, 18, arma::fill::zeros);
        Wi.submat(0, 0, 8, 8) = L;
        Wi.submat(0, 9, 8, 17) = -R;
        W.submat(9 * i, 0, 9 * i + 8, 17) = Wi;
    }

    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, W);
    arma::vec v = V.col(V.n_cols - 1);

    Mat3 RX = vec9ToRot(v.subvec(0, 8));
    Mat3 RY = vec9ToRot(v.subvec(9, 17));
    return {RX, RY};
}

template <typename VecLike>
static inline Quat toQuat(const VecLike& v) {
    Quat q;
    q(0) = v(0);
    q(1) = v(1);
    q(2) = v(2);
    q(3) = v(3);
    return q;
}

// -----------------------------
// Quaternion helpers
// -----------------------------

static inline Quat quatNormalize(const Quat& a) {
    Quat q = a;
    double n = arma::norm(q);
    if (n <= 0) {
        q.zeros();
        q(0) = 1.0;
        return q;
    }
    q /= n;
    if (q(0) < 0) q *= -1.0;
    return q;
}

static inline Quat quatConj(const Quat& q) {
    Quat qc = q;
    qc(1) *= -1.0;
    qc(2) *= -1.0;
    qc(3) *= -1.0;
    return qc;
}

static inline Quat quatMul(const Quat& p, const Quat& q) {
    const double pw = p(0), px = p(1), py = p(2), pz = p(3);
    const double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
    Quat r;
    r(0) = pw * qw - px * qx - py * qy - pz * qz;
    r(1) = pw * qx + px * qw + py * qz - pz * qy;
    r(2) = pw * qy - px * qz + py * qw + pz * qx;
    r(3) = pw * qz + px * qy - py * qx + pz * qw;
    return r;
}

static inline Mat3 quatToRot(const Quat& qq) {
    Quat q = quatNormalize(qq);
    double w = q(0), x = q(1), y = q(2), z = q(3);
    Mat3 R;
    R(0, 0) = 1.0 - 2.0 * (y * y + z * z);
    R(0, 1) = 2.0 * (x * y - z * w);
    R(0, 2) = 2.0 * (x * z + y * w);
    R(1, 0) = 2.0 * (x * y + z * w);
    R(1, 1) = 1.0 - 2.0 * (x * x + z * z);
    R(1, 2) = 2.0 * (y * z - x * w);
    R(2, 0) = 2.0 * (x * z - y * w);
    R(2, 1) = 2.0 * (y * z + x * w);
    R(2, 2) = 1.0 - 2.0 * (x * x + y * y);
    return R;
}

static inline Quat rotToQuat(const Mat3& Rin) {
    Mat3 R = orthonormalizeRot(Rin);
    Quat q;
    double tr = R(0, 0) + R(1, 1) + R(2, 2);
    if (tr > 0) {
        double S = std::sqrt(tr + 1.0) * 2.0;
        q(0) = 0.25 * S;
        q(1) = (R(2, 1) - R(1, 2)) / S;
        q(2) = (R(0, 2) - R(2, 0)) / S;
        q(3) = (R(1, 0) - R(0, 1)) / S;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        double S = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
        q(0) = (R(2, 1) - R(1, 2)) / S;
        q(1) = 0.25 * S;
        q(2) = (R(0, 1) + R(1, 0)) / S;
        q(3) = (R(0, 2) + R(2, 0)) / S;
    } else if (R(1, 1) > R(2, 2)) {
        double S = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
        q(0) = (R(0, 2) - R(2, 0)) / S;
        q(1) = (R(0, 1) + R(1, 0)) / S;
        q(2) = 0.25 * S;
        q(3) = (R(1, 2) + R(2, 1)) / S;
    } else {
        double S = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
        q(0) = (R(1, 0) - R(0, 1)) / S;
        q(1) = (R(0, 2) + R(2, 0)) / S;
        q(2) = (R(1, 2) + R(2, 1)) / S;
        q(3) = 0.25 * S;
    }
    return quatNormalize(q);
}

static inline Mat4 RQ(const Quat& qq) {
    Quat q = qq;
    Mat4 M;
    M(0, 0) = q(0);  M(0, 1) = -q(1); M(0, 2) = -q(2); M(0, 3) = -q(3);
    M(1, 0) = q(1);  M(1, 1) = q(0);  M(1, 2) = q(3);  M(1, 3) = -q(2);
    M(2, 0) = q(2);  M(2, 1) = -q(3); M(2, 2) = q(0);  M(2, 3) = q(1);
    M(3, 0) = q(3);  M(3, 1) = q(2);  M(3, 2) = -q(1); M(3, 3) = q(0);
    return M;
}

static inline Mat4 LQ(const Quat& qq) {
    Quat q = qq;
    Mat4 M;
    M(0, 0) = q(0);  M(0, 1) = -q(1); M(0, 2) = -q(2); M(0, 3) = -q(3);
    M(1, 0) = q(1);  M(1, 1) = q(0);  M(1, 2) = -q(3); M(1, 3) = q(2);
    M(2, 0) = q(2);  M(2, 1) = q(3);  M(2, 2) = q(0);  M(2, 3) = -q(1);
    M(3, 0) = q(3);  M(3, 1) = -q(2); M(3, 2) = q(1);  M(3, 3) = q(0);
    return M;
}

// -----------------------------
// Dual quaternion helpers
// -----------------------------

struct DualQuat {
    Quat qr;  // real
    Quat qd;  // dual
};

static inline DualQuat dqMul(const DualQuat& a, const DualQuat& b) {
    DualQuat r;
    r.qr = quatMul(a.qr, b.qr);
    r.qd = quatMul(a.qr, b.qd) + quatMul(a.qd, b.qr);
    return r;
}

static inline DualQuat dqNeg(const DualQuat& d) {
    DualQuat out = d;
    out.qr = -out.qr;
    out.qd = -out.qd;
    return out;
}

static inline double dqDot8(const DualQuat& a, const DualQuat& b) {
    // Inner product in R^8 between stacked [qr; qd].
    return arma::dot(a.qr, b.qr) + arma::dot(a.qd, b.qd);
}

static inline DualQuat dqNormalize(const DualQuat& d) {
    DualQuat out = d;
    double n = arma::norm(out.qr);
    if (n <= 0) {
        out.qr.zeros();
        out.qr(0) = 1.0;
        out.qd.zeros();
        return out;
    }
    // Scale both parts by the same factor to preserve the represented rigid motion.
    out.qr /= n;
    out.qd /= n;
    // Enforce the unit dual quaternion constraint qr^T qd = 0.
    double dotqd = arma::dot(out.qr, out.qd);
    out.qd -= out.qr * dotqd;
    // Re-normalize and fix the global sign consistently on both parts.
    double nr = arma::norm(out.qr);
    if (nr <= 0) {
        out.qr.zeros();
        out.qr(0) = 1.0;
        out.qd.zeros();
        return out;
    }
    out.qr /= nr;
    out.qd /= nr;
    if (out.qr(0) < 0) {
        out.qr *= -1.0;
        out.qd *= -1.0;
    }
    return out;
}

static inline DualQuat se3ToDQ(const SE3& T) {
    DualQuat d;
    d.qr = rotToQuat(T.R);
    Quat t; t.zeros();
    t(1) = T.t(0); t(2) = T.t(1); t(3) = T.t(2);
    d.qd = 0.5 * quatMul(t, d.qr);
    return dqNormalize(d);
}

static inline SE3 dqToSE3(const DualQuat& dqin) {
    DualQuat d = dqNormalize(dqin);
    Mat3 R = orthonormalizeRot(quatToRot(d.qr));
    Quat tquat = quatMul(d.qd, quatConj(d.qr));
    Vec3 t;
    t(0) = 2.0 * tquat(1);
    t(1) = 2.0 * tquat(2);
    t(2) = 2.0 * tquat(3);
    return orthonormalizeSE3(SE3(R, t));
}


// Zhang 2017 dual quaternion convention: qd = 1/2 (qr ⊗ t), t recovered by t = 2 (qr*) ⊗ qd
static inline DualQuat se3ToDQ_Zhang(const SE3& T) {
    DualQuat d;
    d.qr = rotToQuat(T.R);
    Quat t; t.zeros();
    t(1) = T.t(0); t(2) = T.t(1); t(3) = T.t(2);
    d.qd = 0.5 * quatMul(d.qr, t);
    return dqNormalize(d);
}

static inline SE3 dqToSE3_Zhang(const DualQuat& dqin) {
    DualQuat d = dqNormalize(dqin);
    Mat3 R = orthonormalizeRot(quatToRot(d.qr));
    Quat tquat = quatMul(quatConj(d.qr), d.qd);
    Vec3 t;
    t(0) = 2.0 * tquat(1);
    t(1) = 2.0 * tquat(2);
    t(2) = 2.0 * tquat(3);
    return orthonormalizeSE3(SE3(R, t));
}

// Extract the Plücker line parameters (direction l, moment m) of the screw axis
// from a unit dual quaternion q + eps q0 written with Daniilidis' convention q0 = 1/2 t q.
// For a general screw, qr = (cos(theta/2), sin(theta/2) l) and qd contains sin(theta/2) m plus a component along l.
static inline void dqToScrewLine(const DualQuat& dq_in, Vec3& l, Vec3& m) {
    DualQuat dq = dqNormalize(dq_in);
    Quat qr = quatNormalize(dq.qr);
    Quat qd = dq.qd;

    Vec3 v;
    v(0) = qr(1); v(1) = qr(2); v(2) = qr(3);
    double s = arma::norm(v);

    if (s < 1e-12) {
        // Near-zero rotation: axis direction is ill-defined and the screw axis is close to infinity.
        // Fall back to the rotation log to define an axis, and set the moment to zero.
        Mat3 R = quatToRot(qr);
        Vec3 w = so3Log(R);
        double th = arma::norm(w);
        if (th < 1e-12) {
            l.zeros();
            l(0) = 1.0;
            m.zeros();
            return;
        }
        l = w / th;
        s = std::sin(0.5 * th);
        if (std::abs(s) < 1e-12) {
            m.zeros();
            return;
        }
    } else {
        l = v / s;
    }

    Vec3 vd;
    vd(0) = qd(1); vd(1) = qd(2); vd(2) = qd(3);
    double proj = arma::dot(l, vd);
    Vec3 vd_perp = vd - proj * l;
    m = vd_perp / s;
}

static inline Mat8 Ldq(const DualQuat& d) {
    Mat8 M; M.zeros();
    Mat4 Lr = LQ(d.qr);
    Mat4 Ld = LQ(d.qd);
    M.submat(0, 0, 3, 3) = Lr;
    M.submat(4, 0, 7, 3) = Ld;
    M.submat(4, 4, 7, 7) = Lr;
    return M;
}

static inline Mat8 Rdq(const DualQuat& d) {
    Mat8 M; M.zeros();
    Mat4 Rr = RQ(d.qr);
    Mat4 Rd = RQ(d.qd);
    M.submat(0, 0, 3, 3) = Rr;
    M.submat(4, 0, 7, 3) = Rd;
    M.submat(4, 4, 7, 7) = Rr;
    return M;
}

static inline arma::vec flatten8(const DualQuat& d) {
    arma::vec v(8);
    v.subvec(0, 3) = d.qr;
    v.subvec(4, 7) = d.qd;
    return v;
}

static inline DualQuat unflatten8(const arma::vec& v) {
    DualQuat d;
    d.qr = toQuat(v.subvec(0, 3));
    d.qd = toQuat(v.subvec(4, 7));
    return dqNormalize(d);
}

// -----------------------------
// Residuals used for baseline iterative solvers
// -----------------------------

static arma::vec residualAXXB_basic(const AXXBData& data, const std::vector<SE3>& vars) {
    const SE3& X = vars.at(0);
    int N = static_cast<int>(data.A.size());
    double s = std::sqrt(1.0 / static_cast<double>(N));
    arma::vec r(16 * N);
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        Mat4 E = (data.A[i] * X).matrix() - (X * data.B[i]).matrix();
        r.subvec(idx, idx + 15) = s * flattenMat4(E);
        idx += 16;
    }
    return r;
}

static arma::vec residualAXYB_basic(const AXYBData& data, const std::vector<SE3>& vars) {
    const SE3& X = vars.at(0);
    const SE3& Y = vars.at(1);
    int N = static_cast<int>(data.A.size());
    double s = std::sqrt(1.0 / static_cast<double>(N));
    arma::vec r(16 * N);
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        Mat4 E = (data.A[i] * X).matrix() - (Y * data.B[i]).matrix();
        r.subvec(idx, idx + 15) = s * flattenMat4(E);
        idx += 16;
    }
    return r;
}

static arma::vec residualAXB_YCZ_basic(const AXB_YCZData& data, const std::vector<SE3>& vars) {
    const SE3& X = vars.at(0);
    const SE3& Y = vars.at(1);
    const SE3& Z = vars.at(2);
    int N = static_cast<int>(data.A.size());
    double s = std::sqrt(1.0 / static_cast<double>(N));
    arma::vec r(16 * N);
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        Mat4 E = (data.A[i] * X * data.B[i]).matrix() - (Y * data.C[i] * Z).matrix();
        r.subvec(idx, idx + 15) = s * flattenMat4(E);
        idx += 16;
    }
    return r;
}

static arma::mat numericalJacobian(
    const ResidualFunction& fun,
    const std::vector<SE3>& vars,
    const arma::vec& r0,
    double eps
) {
    const int K = static_cast<int>(vars.size());
    const int D = 6 * K;
    const int M = static_cast<int>(r0.n_rows);
    arma::mat J(M, D, arma::fill::zeros);
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < 6; ++d) {
            Vec6 delta; delta.zeros();
            delta(d) = eps;
            std::vector<SE3> vars_pert = vars;
            vars_pert[k] = se3Exp(delta) * vars[k];
            arma::vec r1 = fun(vars_pert);
            J.col(6 * k + d) = (r1 - r0) / eps;
        }
    }
    return J;
}

}  // namespace

// =============================================================
// PEM solvers
// =============================================================

SE3 solveAXXB_PEM(const AXXBData& data, int restarts) {
    ResidualFunction fun = [&](const std::vector<SE3>& vars) { return residualAXXB_PEM(data, vars); };
    LMOptions opts;
    opts.max_iters = 80;
    opts.max_restarts = std::max(1, restarts);
    opts.verbose = false;
    std::vector<SE3> init = {SE3::Identity()};
    return orthonormalizeSE3(solveLMSE3(fun, init, opts, nullptr)[0]);
}

std::pair<SE3, SE3> solveAXYB_PEM(const AXYBData& data, int restarts) {
    ResidualFunction fun = [&](const std::vector<SE3>& vars) { return residualAXYB_PEM(data, vars); };
    LMOptions opts;
    opts.max_iters = 100;
    opts.max_restarts = std::max(1, restarts);
    opts.verbose = false;
    std::vector<SE3> init = {SE3::Identity(), SE3::Identity()};
    auto sol = solveLMSE3(fun, init, opts, nullptr);
    return {orthonormalizeSE3(sol[0]), orthonormalizeSE3(sol[1])};
}

std::tuple<SE3, SE3, SE3> solveAXB_YCZ_PEM(const AXB_YCZData& data, int restarts) {
    ResidualFunction fun = [&](const std::vector<SE3>& vars) { return residualAXB_YCZ_PEM(data, vars); };
    LMOptions opts;
    opts.max_iters = 120;
    opts.max_restarts = std::max(1, restarts);
    opts.verbose = false;
    std::vector<SE3> init = {SE3::Identity(), SE3::Identity(), SE3::Identity()};
    auto sol = solveLMSE3(fun, init, opts, nullptr);
    return {orthonormalizeSE3(sol[0]), orthonormalizeSE3(sol[1]), orthonormalizeSE3(sol[2])};
}

std::tuple<SE3, SE3, SE3, SE3> solveAXBY_ZCWD_PEM(const AXBY_ZCWDData& data, int restarts) {
    ResidualFunction fun = [&](const std::vector<SE3>& vars) { return residualAXBY_ZCWD_PEM(data, vars); };
    LMOptions opts;
    opts.max_iters = 160;
    opts.max_restarts = std::max(1, restarts);
    opts.verbose = false;
    std::vector<SE3> init = {SE3::Identity(), SE3::Identity(), SE3::Identity(), SE3::Identity()};
    auto sol = solveLMSE3(fun, init, opts, nullptr);
    return {orthonormalizeSE3(sol[0]), orthonormalizeSE3(sol[1]), orthonormalizeSE3(sol[2]), orthonormalizeSE3(sol[3])};
}

// =============================================================
// Analytical solvers (Kronecker/SVD, Section "SVD Solution for SO(3) Rotation")
// =============================================================

SE3 solveAXXB_Analytical(const AXXBData& data) {
    const int N = static_cast<int>(data.A.size());
    if (N <= 0) {
        return SE3::Identity();
    }

    std::vector<Mat3> RA(N), RB(N);
    for (int i = 0; i < N; ++i) {
        RA[i] = data.A[i].R;
        RB[i] = data.B[i].R;
    }

    Mat3 RX = solveRotationAXXB_SVD(RA, RB);

    // Translation LS: (R_A - I) t_X = R_X t_B - t_A
    arma::mat J(3 * N, 3, arma::fill::zeros);
    arma::vec rhs(3 * N, arma::fill::zeros);
    arma::mat I = arma::mat(I3());
    for (int i = 0; i < N; ++i) {
        J.submat(3 * i, 0, 3 * i + 2, 2) = arma::mat(data.A[i].R) - I;
        rhs.subvec(3 * i, 3 * i + 2) = arma::mat(RX) * arma::mat(data.B[i].t) - arma::mat(data.A[i].t);
    }
    arma::vec sol;
    solveLeastSquares(sol, J, rhs);
    Vec3 tX = sol.subvec(0, 2);
    return orthonormalizeSE3(SE3(RX, tX));
}

std::pair<SE3, SE3> solveAXYB_Analytical(const AXYBData& data) {
    const int N = static_cast<int>(data.A.size());
    if (N <= 0) {
        return {SE3::Identity(), SE3::Identity()};
    }

    std::vector<Mat3> RA(N), RB(N);
    for (int i = 0; i < N; ++i) {
        RA[i] = data.A[i].R;
        RB[i] = data.B[i].R;
    }
    auto RXY = solveRotationAXYB_SVD(RA, RB);
    Mat3 RX = RXY.first;
    Mat3 RY = RXY.second;

    // Translation LS: R_A t_X - t_Y = R_Y t_B - t_A
    arma::mat J(3 * N, 6, arma::fill::zeros);
    arma::vec rhs(3 * N, arma::fill::zeros);
    arma::mat I = arma::mat(I3());
    for (int i = 0; i < N; ++i) {
        J.submat(3 * i, 0, 3 * i + 2, 2) = arma::mat(data.A[i].R);
        J.submat(3 * i, 3, 3 * i + 2, 5) = -I;
        rhs.subvec(3 * i, 3 * i + 2) = arma::mat(RY) * arma::mat(data.B[i].t) - arma::mat(data.A[i].t);
    }
    arma::vec sol;
    solveLeastSquares(sol, J, rhs);
    Vec3 tX = sol.subvec(0, 2);
    Vec3 tY = sol.subvec(3, 5);

    return {orthonormalizeSE3(SE3(RX, tX)), orthonormalizeSE3(SE3(RY, tY))};
}

std::tuple<SE3, SE3, SE3> solveAXB_YCZ_Analytical(const AXB_YCZData& data) {
    const int N = static_cast<int>(data.A.size());
    if (N <= 0) {
        return {SE3::Identity(), SE3::Identity(), SE3::Identity()};
    }

    // -------------------------------------------------
    // Step 1: solve R_X from a homogeneous Kronecker/SVD system
    //         (R_B^T ⊗ R_A) vec(R_X) = vec(R_Y R_C R_Z)
    //         vec(R_Y R_C R_Z) = (vec(R_C)^T ⊗ I_9) vec(R_Z^T ⊗ R_Y)
    // -------------------------------------------------
    arma::mat W(9 * N, 90, arma::fill::zeros);
    arma::mat I9 = arma::eye<arma::mat>(9, 9);
    for (int i = 0; i < N; ++i) {
        arma::mat L = arma::kron(arma::mat(data.B[i].R.t()), arma::mat(data.A[i].R));
        arma::vec vecRC = arma::vectorise(arma::mat(data.C[i].R));
        arma::mat P = arma::kron(vecRC.t(), I9);  // 9 x 81

        arma::mat Wi(9, 90, arma::fill::zeros);
        Wi.submat(0, 0, 8, 8) = L;
        Wi.submat(0, 9, 8, 89) = -P;
        W.submat(9 * i, 0, 9 * i + 8, 89) = Wi;
    }

    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, W);
    arma::vec v = V.col(V.n_cols - 1);
    Mat3 RX = vec9ToRot(v.subvec(0, 8));

    // -------------------------------------------------
    // Step 2: solve (R_Z^T, R_Y) from an AX = YB rotation system
    //         (R_A R_X R_B) R_Z^T = R_Y R_C
    // -------------------------------------------------
    std::vector<Mat3> RA2(N), RB2(N);
    for (int i = 0; i < N; ++i) {
        RA2[i] = data.A[i].R * RX * data.B[i].R;
        RB2[i] = data.C[i].R;
    }
    auto RZt_RY = solveRotationAXYB_SVD(RA2, RB2);
    Mat3 RZt = RZt_RY.first;
    Mat3 RY = RZt_RY.second;
    Mat3 RZ = orthonormalizeRot(RZt.t());

    // -------------------------------------------------
    // Step 3: translation LS on the full SE(3) equation
    //   R_A t_X - t_Y - R_Y R_C t_Z = -t_A - R_A R_X t_B + R_Y t_C
    // -------------------------------------------------
    arma::mat J(3 * N, 9, arma::fill::zeros);
    arma::vec rhs(3 * N, arma::fill::zeros);
    arma::mat I = arma::mat(I3());
    for (int i = 0; i < N; ++i) {
        const Mat3& RA = data.A[i].R;
        const Mat3& RC = data.C[i].R;
        const Vec3& tA = data.A[i].t;
        const Vec3& tB = data.B[i].t;
        const Vec3& tC = data.C[i].t;

        J.submat(3 * i, 0, 3 * i + 2, 2) = arma::mat(RA);
        J.submat(3 * i, 3, 3 * i + 2, 5) = -I;
        J.submat(3 * i, 6, 3 * i + 2, 8) = -arma::mat(RY * RC);

        rhs.subvec(3 * i, 3 * i + 2) = -arma::mat(tA) - arma::mat(RA * RX * tB) + arma::mat(RY * tC);
    }
    arma::vec sol;
    solveLeastSquares(sol, J, rhs);
    Vec3 tX = sol.subvec(0, 2);
    Vec3 tY = sol.subvec(3, 5);
    Vec3 tZ = sol.subvec(6, 8);

    return {orthonormalizeSE3(SE3(RX, tX)), orthonormalizeSE3(SE3(RY, tY)), orthonormalizeSE3(SE3(RZ, tZ))};
}

// =============================================================
// AX = XB baselines
// =============================================================

SE3 solveAXXB_Park1994(const AXXBData& data) {
    // Park & Martin 1994: rotation from log-map alignment.
    int N = static_cast<int>(data.A.size());
    arma::mat M(3, 3, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Vec3 a = so3Log(data.A[i].R);
        Vec3 b = so3Log(data.B[i].R);
        M += arma::mat(a) * arma::mat(b).t();
    }
    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, M);
    Mat3 RX = orthonormalizeRot(U * V.t());
    if (arma::det(RX) < 0) {
        U.col(2) *= -1.0;
        RX = orthonormalizeRot(U * V.t());
    }

    arma::mat A(3 * N, 3, arma::fill::zeros);
    arma::vec b(3 * N, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Mat3 RA = data.A[i].R;
        Vec3 tA = data.A[i].t;
        Vec3 tB = data.B[i].t;
        A.submat(3 * i, 0, 3 * i + 2, 2) = RA - I3();
        b.subvec(3 * i, 3 * i + 2) = RX * tB - tA;
    }
    Vec3 tX = arma::solve(A, b);
    return orthonormalizeSE3(SE3(RX, tX));
}

SE3 solveAXXB_Horaud1995(const AXXBData& data) {
    // Quaternion least-squares: (L(qA) - R(qB)) qX = 0
    int N = static_cast<int>(data.A.size());
    arma::mat W(4 * N, 4, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Quat qA = rotToQuat(data.A[i].R);
        Quat qB = rotToQuat(data.B[i].R);
        Mat4 Li = LQ(qA);
        Mat4 Ri = RQ(qB);
        W.submat(4 * i, 0, 4 * i + 3, 3) = Li - Ri;
    }
    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, W);
    Quat qX = toQuat(V.col(3));
    qX = quatNormalize(qX);
    Mat3 RX = orthonormalizeRot(quatToRot(qX));

    arma::mat A(3 * N, 3, arma::fill::zeros);
    arma::vec b(3 * N, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Mat3 RA = data.A[i].R;
        Vec3 tA = data.A[i].t;
        Vec3 tB = data.B[i].t;
        A.submat(3 * i, 0, 3 * i + 2, 2) = RA - I3();
        b.subvec(3 * i, 3 * i + 2) = RX * tB - tA;
    }
    Vec3 tX = arma::solve(A, b);
    return orthonormalizeSE3(SE3(RX, tX));
}

SE3 solveAXXB_Daniilidis1999(const AXXBData& data) {
    // Daniilidis 1999 (I. J. Robotics Research): simultaneous hand-eye calibration
    // with unit dual quaternions using SVD.
    //
    // This implementation follows the paper's line-based SVD derivation:
    // build the 6n x 8 matrix T from the screw axis Plücker line parameters
    // and solve for the unit dual quaternion (q, q0) by enforcing the two
    // constraints q^T q = 1 and q^T q0 = 0.
    // See eqs. (31)–(36) in the paper.

    const int n = static_cast<int>(data.A.size());
    if (n < 2) {
        // The minimal requirement is two nonparallel screw axes.
        // Fall back to Park if insufficient motions are provided.
        return solveAXXB_Park1994(data);
    }

    arma::mat T(6 * n, 8, arma::fill::zeros);

    for (int i = 0; i < n; ++i) {
        DualQuat dqA = se3ToDQ(data.A[i]);
        DualQuat dqB = se3ToDQ(data.B[i]);

        Vec3 aE, aE0;
        Vec3 bE, bE0;
        dqToScrewLine(dqA, aE, aE0);
        dqToScrewLine(dqB, bE, bE0);

        arma::mat S(6, 8, arma::fill::zeros);

        Vec3 d = aE - bE;
        Vec3 s = aE + bE;
        Mat3 sx = skew(s);

        // Top 3 rows: [aE - bE, [aE + bE]_x, 0, 0]
        S(0, 0) = d(0); S(1, 0) = d(1); S(2, 0) = d(2);
        S.submat(0, 1, 2, 3) = arma::mat(sx);

        Vec3 d0 = aE0 - bE0;
        Vec3 s0 = aE0 + bE0;
        Mat3 s0x = skew(s0);

        // Bottom 3 rows: [aE0 - bE0, [aE0 + bE0]_x, aE - bE, [aE + bE]_x]
        S(3, 0) = d0(0); S(4, 0) = d0(1); S(5, 0) = d0(2);
        S.submat(3, 1, 5, 3) = arma::mat(s0x);
        S(3, 4) = d(0); S(4, 4) = d(1); S(5, 4) = d(2);
        S.submat(3, 5, 5, 7) = arma::mat(sx);

        T.submat(6 * i, 0, 6 * i + 5, 7) = S;
    }

    arma::mat U, V;
    arma::vec sv;
    arma::svd(U, sv, V, T);

    arma::vec v7 = V.col(6);
    arma::vec v8 = V.col(7);

    arma::vec u1 = v7.subvec(0, 3);
    arma::vec v1 = v7.subvec(4, 7);
    arma::vec u2 = v8.subvec(0, 3);
    arma::vec v2 = v8.subvec(4, 7);

    // Solve eq. (35) with s = lambda1/lambda2:
    // (u1^T v1) s^2 + (u1^T v2 + u2^T v1) s + (u2^T v2) = 0.
    const double qa = arma::dot(u1, v1);
    const double qb = arma::dot(u1, v2) + arma::dot(u2, v1);
    const double qc = arma::dot(u2, v2);

    std::vector<double> s_candidates;
    const double eps = 1e-14;
    if (std::abs(qa) > eps) {
        double disc = qb * qb - 4.0 * qa * qc;
        if (disc < 0.0) disc = 0.0;
        double sd = std::sqrt(disc);
        s_candidates.push_back((-qb + sd) / (2.0 * qa));
        s_candidates.push_back((-qb - sd) / (2.0 * qa));
    } else if (std::abs(qb) > eps) {
        s_candidates.push_back(-qc / qb);
    } else {
        s_candidates.push_back(0.0);
    }

    // Choose the s that maximizes the positive trinomial in eq. (36).
    double best_s = s_candidates[0];
    double best_denom = -1.0;
    for (double s : s_candidates) {
        double denom = s * s * arma::dot(u1, u1) + 2.0 * s * arma::dot(u1, u2) + arma::dot(u2, u2);
        if (denom > best_denom) {
            best_denom = denom;
            best_s = s;
        }
    }

    double denom = std::max(best_denom, 1e-18);
    double lambda2 = 1.0 / std::sqrt(denom);
    double lambda1 = best_s * lambda2;

    arma::vec x = lambda1 * v7 + lambda2 * v8;

    DualQuat dq;
    dq.qr = toQuat(x.subvec(0, 3));
    dq.qd = toQuat(x.subvec(4, 7));
    dq = dqNormalize(dq);

    return dqToSE3(dq);
}

SE3 solveAXXB_Zhang2017(const AXXBData& data, int max_iters) {
    // Zhang, Zhang, Yang 2017 (Int. J. CARS): computationally efficient hand–eye calibration.
    //
    // We implement the paper's two-step iteration in the dual quaternion domain.
    // The method alternates between solving qd,X and qr,X via pseudo-inverses in
    //   Hl * qr,X = Hr * qd,X
    // where Hl and Hr are stacked from each motion pair.
    // See eqs. (24)–(35) and Algorithm 1 in the paper.

    const int J = static_cast<int>(data.A.size());
    if (J < 2) {
        return solveAXXB_Park1994(data);
    }

    std::vector<DualQuat> dqA(J), dqB(J);
    for (int i = 0; i < J; ++i) {
        dqA[i] = se3ToDQ_Zhang(data.A[i]);
        dqB[i] = se3ToDQ_Zhang(data.B[i]);
    }

    // Step 0: initialize qr,X using the rotation-only equation
    // (L(qr,Bi) - R(qr,Ai)) qr,X = 0.
    arma::mat W(4 * J, 4, arma::fill::zeros);
    for (int i = 0; i < J; ++i) {
        Mat4 Mi = LQ(dqB[i].qr) - RQ(dqA[i].qr);
        W.submat(4 * i, 0, 4 * i + 3, 3) = Mi;
    }
    arma::mat U, V;
    arma::vec sv;
    arma::svd(U, sv, V, W);
    Quat qr = toQuat(V.col(3));
    qr = quatNormalize(qr);

    // Build Hl and Hr once (they are constant for the given motion set).
    arma::mat Hl(8 * J, 4, arma::fill::zeros);
    arma::mat Hr(8 * J, 4, arma::fill::zeros);

    for (int i = 0; i < J; ++i) {
        Mat4 M = LQ(dqB[i].qr) - RQ(dqA[i].qr);
        Mat4 N = RQ(dqA[i].qd) - LQ(dqB[i].qd);

        Hl.submat(8 * i, 0, 8 * i + 3, 3) = M;
        Hl.submat(8 * i + 4, 0, 8 * i + 7, 3) = N;

        // Hr has a zero top block.
        Hr.submat(8 * i + 4, 0, 8 * i + 7, 3) = M;
    }

    arma::mat pinvHr = arma::pinv(Hr);
    arma::mat pinvHl = arma::pinv(Hl);

    Quat qr_n = qr;
    Quat qd_n; qd_n.zeros();

    const int iters = std::max(1, max_iters);
    for (int k = 0; k < iters; ++k) {
        // Eq. (32): qd,X^n = Hr^+ Hl qr,X^{n-1}
        arma::vec qd_vec = pinvHr * (Hl * qr_n);
        qd_n = toQuat(qd_vec);

        // Eq. (33): qr,X^n = Hl^+ Hr qd,X^n
        arma::vec qr_vec = pinvHl * (Hr * qd_n);
        Quat qr_next = toQuat(qr_vec);

        // Handle the quaternion double-cover when checking convergence.
        double d1 = arma::norm(qr_next - qr_n);
        double d2 = arma::norm(qr_next + qr_n);
        double diff = std::min(d1, d2);
        qr_n = qr_next;

        if (diff < 1e-12) {
            break;
        }
    }

    // Re-scale both parts by the same magnitude, then enforce unit constraints.
    double s = arma::norm(qr_n);
    if (s <= 0) s = 1.0;
    qr_n /= s;
    qd_n /= s;

    // Enforce qr^T qd = 0.
    qd_n -= qr_n * arma::dot(qr_n, qd_n);

    // Fix the global sign consistently.
    if (qr_n(0) < 0) {
        qr_n *= -1.0;
        qd_n *= -1.0;
    }
    double nr = std::max(1e-18, arma::norm(qr_n));
    qr_n /= nr;
    qd_n /= nr;

    DualQuat dqX;
    dqX.qr = qr_n;
    dqX.qd = qd_n;
    return dqToSE3_Zhang(dqX);
}

// =============================================================
// AX = YB baselines
// =============================================================

std::pair<SE3, SE3> solveAXYB_Dornaika1998(const AXYBData& data) {
    // Dornaika & Horaud 1998 closed-form for rotations.
    int N = static_cast<int>(data.A.size());
    arma::mat C(4, 4, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Quat qA = rotToQuat(data.A[i].R);
        Quat qB = rotToQuat(data.B[i].R);
        C += LQ(qA).t() * RQ(qB);
    }
    arma::mat CtC = C.t() * C;
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, CtC);
    int idx = static_cast<int>(eigval.n_elem) - 1;
    Quat qY = toQuat(eigvec.col(idx));
    qY = quatNormalize(qY);
    double mu = std::max(1e-12, eigval(idx));
    arma::vec tmpX = -(C * qY) / std::sqrt(mu);
    Quat qX = toQuat(tmpX);
    qX = quatNormalize(qX);

    Mat3 RX = orthonormalizeRot(quatToRot(qX));
    Mat3 RY = orthonormalizeRot(quatToRot(qY));

    arma::mat A(3 * N, 6, arma::fill::zeros);
    arma::vec b(3 * N, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Mat3 RA = data.A[i].R;
        Vec3 tA = data.A[i].t;
        Vec3 tB = data.B[i].t;
        A.submat(3 * i, 0, 3 * i + 2, 2) = RA;
        A.submat(3 * i, 3, 3 * i + 2, 5) = -I3();
        b.subvec(3 * i, 3 * i + 2) = RY * tB - tA;
    }
    arma::vec x = arma::solve(A, b);
    Vec3 tX = x.subvec(0, 2);
    Vec3 tY = x.subvec(3, 5);
    return {orthonormalizeSE3(SE3(RX, tX)), orthonormalizeSE3(SE3(RY, tY))};
}

std::pair<SE3, SE3> solveAXYB_Shah2013(const AXYBData& data) {
    // Shah 2013 Kronecker solution for rotations.
    int N = static_cast<int>(data.A.size());
    arma::mat K(9, 9, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        K += arma::kron(data.B[i].R, data.A[i].R);
    }
    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, K);
    arma::vec u1 = U.col(0);
    arma::vec v1 = V.col(0);
    arma::mat VY = arma::reshape(u1, 3, 3);
    arma::mat VX = arma::reshape(v1, 3, 3);

    auto scaleToDet1 = [](const arma::mat& M) -> arma::mat {
        arma::mat out = M;
        double d = arma::det(out);
        if (std::abs(d) < 1e-12) return out;
        double a = (d >= 0 ? 1.0 : -1.0) * std::pow(std::abs(d), -1.0 / 3.0);
        out = a * out;
        return out;
    };

    Mat3 RX = orthonormalizeRot(scaleToDet1(VX));
    Mat3 RY = orthonormalizeRot(scaleToDet1(VY));

    arma::mat A(3 * N, 6, arma::fill::zeros);
    arma::vec b(3 * N, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Mat3 RA = data.A[i].R;
        Vec3 tA = data.A[i].t;
        Vec3 tB = data.B[i].t;
        A.submat(3 * i, 0, 3 * i + 2, 2) = I3();
        A.submat(3 * i, 3, 3 * i + 2, 5) = -RA;
        b.subvec(3 * i, 3 * i + 2) = tA - RY * tB;
    }
    arma::vec x = arma::solve(A, b);
    Vec3 tY = x.subvec(0, 2);
    Vec3 tX = x.subvec(3, 5);
    return {orthonormalizeSE3(SE3(RX, tX)), orthonormalizeSE3(SE3(RY, tY))};
}

std::pair<SE3, SE3> solveAXYB_Park2016TIE(const AXYBData& data, int restarts) {
    // Implemented as multi-start nonlinear refinement on the basic residual.
    ResidualFunction fun = [&](const std::vector<SE3>& vars) { return residualAXYB_basic(data, vars); };
    LMOptions opts;
    opts.max_iters = 120;
    opts.max_restarts = std::max(1, restarts);
    opts.verbose = false;
    std::vector<SE3> init = {SE3::Identity(), SE3::Identity()};
    auto sol = solveLMSE3(fun, init, opts, nullptr);
    return {orthonormalizeSE3(sol[0]), orthonormalizeSE3(sol[1])};
}

std::pair<SE3, SE3> solveAXYB_Tabb2017(const AXYBData& data, int refine_lm_iters) {
    // Iterative refinement from a closed-form initialization (Shah 2013).
    auto init = solveAXYB_Shah2013(data);
    ResidualFunction fun = [&](const std::vector<SE3>& vars) { return residualAXYB_basic(data, vars); };
    LMOptions opts;
    opts.max_iters = std::max(1, refine_lm_iters);
    opts.max_restarts = 1;
    opts.verbose = false;
    std::vector<SE3> seed = {init.first, init.second};
    auto sol = solveLMSE3(fun, seed, opts, nullptr);
    return {orthonormalizeSE3(sol[0]), orthonormalizeSE3(sol[1])};
}

// =============================================================
// AXB = YCZ baselines
// =============================================================

std::tuple<SE3, SE3, SE3> solveAXB_YCZ_Wu2016TRO(const AXB_YCZData& data) {
    // Wu et al. 2016: closed-form quaternion solution for rotations (no LM/iterative refinement here).
    //
    // The paper derives a linear homogeneous system of the form
    //     W_ABC(qA,qB,qC) * qXYZ = 0,
    // where qXYZ stacks the unknown quaternion qX and the 16-vector qYZ that is
    // formed by all products qY_i qZ_j. The smallest singular vector gives a
    // closed-form estimate of (qX, qYZ). The quaternion pair (qY, qZ) is then
    // recovered from a rank-1 factorization of a 4x4 matrix built from qYZ.

    const int N = static_cast<int>(data.A.size());
    const int m0 = std::min(5, N);

    arma::vec best_v;
    double best_sigma_min = std::numeric_limits<double>::infinity();

    auto buildWC = [](const Quat& qC) {
        // Build W_C (4x16) such that
        //   RQ(qZ) * LQ(qY) * qC = W_C * qYZ,
        // where qYZ stacks the bilinear terms qY_i qZ_j in row-major order:
        //   qYZ = [qY0*qZ0, qY0*qZ1, ..., qY3*qZ3]^T.
        //
        // Because RQ and LQ are linear in qZ and qY, respectively, the mapping
        // is bilinear and can be written as a linear map in the 16 monomials.
        arma::mat WC(4, 16, arma::fill::zeros);
        for (int yi = 0; yi < 4; ++yi) {
            Quat eY; eY.zeros(); eY(yi) = 1.0;
            Mat4 LY = LQ(eY);
            for (int zi = 0; zi < 4; ++zi) {
                Quat eZ; eZ.zeros(); eZ(zi) = 1.0;
                Mat4 RZ = RQ(eZ);
                WC.col(4 * yi + zi) = RZ * (LY * qC);
            }
        }
        return WC;
    };

    for (int mask = 0; mask < (1 << m0); ++mask) {
        arma::mat Wtil(4 * m0, 20, arma::fill::zeros);
        for (int k = 0; k < m0; ++k) {
            const SE3& A = data.A[k];
            const SE3& B = data.B[k];
            const SE3& C = data.C[k];

            Quat qA = rotToQuat(A.R);
            Quat qB = rotToQuat(B.R);
            Quat qC = rotToQuat(C.R);

            Mat4 WAB = RQ(qB) * LQ(qA);
            arma::mat WC = buildWC(qC);

            int sgn = ((mask >> k) & 1) ? -1 : +1;
            arma::mat WABC(4, 20, arma::fill::zeros);
            WABC.submat(0, 0, 3, 3) = WAB;
            WABC.submat(0, 4, 3, 19) = -static_cast<double>(sgn) * WC;
            Wtil.submat(4 * k, 0, 4 * k + 3, 19) = WABC;
        }
        arma::mat U, V;
        arma::vec s;
        arma::svd(U, s, V, Wtil);
        double sigma_min = s(s.n_elem - 1);
        if (sigma_min < best_sigma_min) {
            best_sigma_min = sigma_min;
            best_v = V.col(19);
        }
    }

    arma::vec vX = best_v.subvec(0, 3);
    arma::vec vYZ = best_v.subvec(4, 19);
    Quat qX = quatNormalize(toQuat(vX));

    arma::vec qYZ = vYZ / arma::norm(vYZ);

    // Reconstruct a 4x4 matrix Q with Q(i,j) = qY_i qZ_j.
    arma::mat Q(4, 4, arma::fill::zeros);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Q(i, j) = qYZ(4 * i + j);
        }
    }

    // Best rank-1 factorization of Q gives qY and qZ.
    arma::mat Uq, Vq;
    arma::vec sq;
    arma::svd(Uq, sq, Vq, Q);
    Quat qY = quatNormalize(toQuat(Uq.col(0)));
    Quat qZ = quatNormalize(toQuat(Vq.col(0)));
    if (qY(0) < 0.0) {
        qY = -qY;
        qZ = -qZ;
    }

    Mat3 RX = orthonormalizeRot(quatToRot(qX));
    Mat3 RY = orthonormalizeRot(quatToRot(qY));
    Mat3 RZ = orthonormalizeRot(quatToRot(qZ));

    // Translation LS (Eq. 3)
    arma::mat J(3 * N, 9, arma::fill::zeros);
    arma::vec rhs(3 * N, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Mat3 RA = data.A[i].R;
        Mat3 RC = data.C[i].R;
        Vec3 tA = data.A[i].t;
        Vec3 tB = data.B[i].t;
        Vec3 tC = data.C[i].t;

        J.submat(3 * i, 0, 3 * i + 2, 2) = RA;
        J.submat(3 * i, 3, 3 * i + 2, 5) = -I3();
        J.submat(3 * i, 6, 3 * i + 2, 8) = -RY * RC;
        rhs.subvec(3 * i, 3 * i + 2) = -tA - RA * (RX * tB) + RY * tC;
    }
    arma::vec sol = arma::solve(J, rhs);
    Vec3 tX = sol.subvec(0, 2);
    Vec3 tY = sol.subvec(3, 5);
    Vec3 tZ = sol.subvec(6, 8);
    return {orthonormalizeSE3(SE3(RX, tX)), orthonormalizeSE3(SE3(RY, tY)), orthonormalizeSE3(SE3(RZ, tZ))};
}

std::tuple<SE3, SE3, SE3> solveAXB_YCZ_Ma2018(const AXB_YCZData& data, int refine_lm_iters) {
    // Implemented as nonlinear MLE refinement (LM) on the basic residual.
    auto init = solveAXB_YCZ_Wu2016TRO(data);
    ResidualFunction fun = [&](const std::vector<SE3>& vars) { return residualAXB_YCZ_basic(data, vars); };
    LMOptions opts;
    opts.max_iters = std::max(1, refine_lm_iters);
    opts.max_restarts = 1;
    opts.verbose = false;
    std::vector<SE3> seed = {std::get<0>(init), std::get<1>(init), std::get<2>(init)};
    auto sol = solveLMSE3(fun, seed, opts, nullptr);
    return {orthonormalizeSE3(sol[0]), orthonormalizeSE3(sol[1]), orthonormalizeSE3(sol[2])};
}

std::tuple<SE3, SE3, SE3> solveAXB_YCZ_Sui2023(const AXB_YCZData& data, int iters) {
    // Sui et al. 2023: momentum-based gradient descent on the rotational subproblem,
    // followed by a linear least-squares translation solve.
    //
    // We implement the same overall structure but stay in the matrix domain:
    //     min_{R_X,R_Y,R_Z \in SO(3)} \sum_i || R_Ai R_X R_Bi - R_Y R_Ci R_Z ||_F^2.

    auto init = solveAXB_YCZ_Wu2016TRO(data);
    Mat3 RX = std::get<0>(init).R;
    Mat3 RY = std::get<1>(init).R;
    Mat3 RZ = std::get<2>(init).R;

    const int N = static_cast<int>(data.A.size());
    const int maxIters = std::max(1, iters);

    auto veeSkew = [](const Mat3& M) {
        Mat3 S = 0.5 * (M - M.t());
        Vec3 v;
        v(0) = S(2, 1);
        v(1) = S(0, 2);
        v(2) = S(1, 0);
        return v;
    };

    auto costAll = [&](const Mat3& RXc, const Mat3& RYc, const Mat3& RZc) {
        double c = 0.0;
        for (int i = 0; i < N; ++i) {
            Mat3 E = data.A[i].R * RXc * data.B[i].R - RYc * data.C[i].R * RZc;
            c += arma::accu(E % E);
        }
        return c;
    };

    // Momentum parameters.
    Vec3 vX; vX.zeros();
    Vec3 vY; vY.zeros();
    Vec3 vZ; vZ.zeros();
    const double beta = 0.9;
    double alpha0 = 5e-2;
    double decay = 1e-3;

    double cPrev = costAll(RX, RY, RZ);
    for (int iter = 0; iter < maxIters; ++iter) {
        // Full-batch gradients in the tangent space of SO(3).
        Vec3 gX; gX.zeros();
        Vec3 gY; gY.zeros();
        Vec3 gZ; gZ.zeros();

        for (int i = 0; i < N; ++i) {
            const Mat3& RA = data.A[i].R;
            const Mat3& RB = data.B[i].R;
            const Mat3& RC = data.C[i].R;
            Mat3 E = RA * RX * RB - RY * RC * RZ;

            // See derivation in comments above (right-invariant perturbations).
            Mat3 MX = RB * E.t() * RA * RX;
            Mat3 MY = RC * RZ * E.t() * RY;
            Mat3 MZ = E.t() * RY * RC * RZ;
            gX += veeSkew(MX);
            gY += veeSkew(MY);
            gZ += veeSkew(MZ);
        }

        double gnorm = std::sqrt(arma::dot(gX, gX) + arma::dot(gY, gY) + arma::dot(gZ, gZ));
        if (gnorm < 1e-12) {
            break;
        }

        // Momentum update.
        vX = beta * vX + (1.0 - beta) * gX;
        vY = beta * vY + (1.0 - beta) * gY;
        vZ = beta * vZ + (1.0 - beta) * gZ;

        double alpha = alpha0 / (1.0 + decay * static_cast<double>(iter));

        // A simple two-sided check to avoid sign mistakes in the gradient algebra.
        Mat3 RX1 = orthonormalizeRot(RX * so3Exp(alpha * vX));
        Mat3 RY1 = orthonormalizeRot(RY * so3Exp(-alpha * vY));
        Mat3 RZ1 = orthonormalizeRot(RZ * so3Exp(-alpha * vZ));
        double c1 = costAll(RX1, RY1, RZ1);

        Mat3 RX2 = orthonormalizeRot(RX * so3Exp(-alpha * vX));
        Mat3 RY2 = orthonormalizeRot(RY * so3Exp(alpha * vY));
        Mat3 RZ2 = orthonormalizeRot(RZ * so3Exp(alpha * vZ));
        double c2 = costAll(RX2, RY2, RZ2);

        if (c1 <= c2) {
            if (c1 <= cPrev) {
                RX = RX1; RY = RY1; RZ = RZ1; cPrev = c1;
            } else {
                // Backtracking if the step is too aggressive.
                alpha0 *= 0.5;
            }
        } else {
            if (c2 <= cPrev) {
                RX = RX2; RY = RY2; RZ = RZ2; cPrev = c2;
            } else {
                alpha0 *= 0.5;
            }
        }

        if (alpha0 < 1e-6) {
            break;
        }
    }

    // Translation LS (Eq. 3).
    arma::mat J(3 * N, 9, arma::fill::zeros);
    arma::vec rhs(3 * N, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        Mat3 RA = data.A[i].R;
        Mat3 RC = data.C[i].R;
        Vec3 tA = data.A[i].t;
        Vec3 tB = data.B[i].t;
        Vec3 tC = data.C[i].t;

        J.submat(3 * i, 0, 3 * i + 2, 2) = RA;
        J.submat(3 * i, 3, 3 * i + 2, 5) = -I3();
        J.submat(3 * i, 6, 3 * i + 2, 8) = -RY * RC;
        rhs.subvec(3 * i, 3 * i + 2) = -tA - RA * (RX * tB) + RY * tC;
    }
    arma::vec sol = arma::solve(J, rhs);
    Vec3 tX = sol.subvec(0, 2);
    Vec3 tY = sol.subvec(3, 5);
    Vec3 tZ = sol.subvec(6, 8);
    return {orthonormalizeSE3(SE3(RX, tX)), orthonormalizeSE3(SE3(RY, tY)), orthonormalizeSE3(SE3(RZ, tZ))};
}

}  // namespace pem
