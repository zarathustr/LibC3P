#include "problems.h"
#include <cmath>

namespace pem {

static arma::vec flattenMat4(const Mat4& M) {
    arma::vec v(16);
    int idx = 0;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            v(idx++) = M(r,c);
        }
    }
    return v;
}

AXXBData makeAXXBData(const SE3& X_true, int N, double noise) {
    AXXBData data;
    data.A.resize(N);
    data.B.resize(N);

    SE3 Xinv = X_true.inverse();

    for (int i = 0; i < N; ++i) {
        SE3 Ai = randomSE3(0.8, 1.0);
        SE3 Bi = Xinv * Ai * X_true;

        Ai = perturbSE3(Ai, noise, noise);
        Bi = perturbSE3(Bi, noise, noise);

        data.A[i] = Ai;
        data.B[i] = Bi;
    }
    return data;
}

AXYBData makeAXYBData(const SE3& X_true, const SE3& Y_true, int N, double noise) {
    AXYBData data;
    data.A.resize(N);
    data.B.resize(N);

    SE3 Yinv = Y_true.inverse();

    for (int i = 0; i < N; ++i) {
        SE3 Ai = randomSE3(0.8, 1.0);
        SE3 Bi = Yinv * Ai * X_true;

        Ai = perturbSE3(Ai, noise, noise);
        Bi = perturbSE3(Bi, noise, noise);

        data.A[i] = Ai;
        data.B[i] = Bi;
    }
    return data;
}

AXB_YCZData makeAXB_YCZData(const SE3& X_true, const SE3& Y_true, const SE3& Z_true, int N, double noise) {
    AXB_YCZData data;
    data.A.resize(N);
    data.B.resize(N);
    data.C.resize(N);

    for (int i = 0; i < N; ++i) {
        SE3 Ai = randomSE3(0.8, 1.0);
        SE3 Ci = randomSE3(0.8, 1.0);

        SE3 left = Ai * X_true;
        SE3 Binv = left.inverse() * Y_true * Ci * Z_true;
        SE3 Bi = Binv;  // this is B directly, not inverse

        Ai = perturbSE3(Ai, noise, noise);
        Bi = perturbSE3(Bi, noise, noise);
        Ci = perturbSE3(Ci, noise, noise);

        data.A[i] = Ai;
        data.B[i] = Bi;
        data.C[i] = Ci;
    }
    return data;
}

AXBY_ZCWDData makeAXBY_ZCWDData(const SE3& X_true, const SE3& Y_true, const SE3& Z_true, const SE3& W_true, int N, double noise) {
    AXBY_ZCWDData data;
    data.A.resize(N);
    data.B.resize(N);
    data.C.resize(N);
    data.D.resize(N);

    for (int i = 0; i < N; ++i) {
        SE3 Ai = randomSE3(0.8, 1.0);
        SE3 Bi = randomSE3(0.8, 1.0);
        SE3 Ci = randomSE3(0.8, 1.0);

        SE3 left = Ai * X_true * Bi * Y_true;
        SE3 Dinv = (Z_true * Ci * W_true).inverse() * left;
        SE3 Di = Dinv;

        Ai = perturbSE3(Ai, noise, noise);
        Bi = perturbSE3(Bi, noise, noise);
        Ci = perturbSE3(Ci, noise, noise);
        Di = perturbSE3(Di, noise, noise);

        data.A[i] = Ai;
        data.B[i] = Bi;
        data.C[i] = Ci;
        data.D[i] = Di;
    }
    return data;
}

arma::vec residualAXXB_PEM(const AXXBData& data, const std::vector<SE3>& vars) {
    const SE3& X = vars.at(0);
    const SE3 Xinv = X.inverse();

    int N = static_cast<int>(data.A.size());
    double s = std::sqrt(1.0 / static_cast<double>(N));

    arma::vec r(16 * N * 2);
    int idx = 0;

    for (int i = 0; i < N; ++i) {
        Mat4 E1 = (data.A[i] * X).matrix() - (X * data.B[i]).matrix();
        arma::vec v1 = s * flattenMat4(E1);
        r.subvec(idx, idx + 15) = v1;
        idx += 16;

        Mat4 E2 = (data.B[i].inverse() * Xinv).matrix() - (Xinv * data.A[i].inverse()).matrix();
        arma::vec v2 = s * flattenMat4(E2);
        r.subvec(idx, idx + 15) = v2;
        idx += 16;
    }

    return r;
}

arma::vec residualAXYB_PEM(const AXYBData& data, const std::vector<SE3>& vars) {
    const SE3& X = vars.at(0);
    const SE3& Y = vars.at(1);

    const SE3 Xinv = X.inverse();
    const SE3 Yinv = Y.inverse();

    int N = static_cast<int>(data.A.size());
    double s = std::sqrt(1.0 / static_cast<double>(N));

    arma::vec r(16 * N * 2);
    int idx = 0;

    for (int i = 0; i < N; ++i) {
        Mat4 E1 = (data.A[i] * X).matrix() - (Y * data.B[i]).matrix();
        arma::vec v1 = s * flattenMat4(E1);
        r.subvec(idx, idx + 15) = v1;
        idx += 16;

        Mat4 E2 = (data.B[i].inverse() * Yinv).matrix() - (Xinv * data.A[i].inverse()).matrix();
        arma::vec v2 = s * flattenMat4(E2);
        r.subvec(idx, idx + 15) = v2;
        idx += 16;
    }

    return r;
}

arma::vec residualAXB_YCZ_PEM(const AXB_YCZData& data, const std::vector<SE3>& vars) {
    const SE3& X = vars.at(0);
    const SE3& Y = vars.at(1);
    const SE3& Z = vars.at(2);

    const SE3 Xinv = X.inverse();
    const SE3 Yinv = Y.inverse();
    const SE3 Zinv = Z.inverse();

    int N = static_cast<int>(data.A.size());
    double s = std::sqrt(1.0 / static_cast<double>(N));

    arma::vec r(16 * N * 3);
    int idx = 0;

    for (int i = 0; i < N; ++i) {
        const SE3& A = data.A[i];
        const SE3& B = data.B[i];
        const SE3& C = data.C[i];

        Mat4 E1 = (A * X * B).matrix() - (Y * C * Z).matrix();
        r.subvec(idx, idx + 15) = s * flattenMat4(E1);
        idx += 16;

        Mat4 E2 = (B.inverse() * Xinv * A.inverse()).matrix()
                - (Zinv * C.inverse() * Yinv).matrix();
        r.subvec(idx, idx + 15) = s * flattenMat4(E2);
        idx += 16;

        Mat4 E3 = (C * Z * B.inverse()).matrix()
                - (Yinv * A * X).matrix();
        r.subvec(idx, idx + 15) = s * flattenMat4(E3);
        idx += 16;
    }

    return r;
}

arma::vec residualAXBY_ZCWD_PEM(const AXBY_ZCWDData& data, const std::vector<SE3>& vars) {
    const SE3& X = vars.at(0);
    const SE3& Y = vars.at(1);
    const SE3& Z = vars.at(2);
    const SE3& W = vars.at(3);

    const SE3 Xinv = X.inverse();
    const SE3 Yinv = Y.inverse();
    const SE3 Zinv = Z.inverse();
    const SE3 Winv = W.inverse();

    int N = static_cast<int>(data.A.size());
    double s = std::sqrt(1.0 / static_cast<double>(N));

    arma::vec r(16 * N * 2);
    int idx = 0;

    for (int i = 0; i < N; ++i) {
        const SE3& A = data.A[i];
        const SE3& B = data.B[i];
        const SE3& C = data.C[i];
        const SE3& D = data.D[i];

        Mat4 E1 = (A * X * B * Y).matrix()
                - (Z * C * W * D).matrix();
        r.subvec(idx, idx + 15) = s * flattenMat4(E1);
        idx += 16;

        Mat4 E2 = (D.inverse() * Winv * C.inverse() * Zinv).matrix()
                - (Yinv * B.inverse() * Xinv * A.inverse()).matrix();
        r.subvec(idx, idx + 15) = s * flattenMat4(E2);
        idx += 16;
    }

    return r;
}

}  // namespace pem
