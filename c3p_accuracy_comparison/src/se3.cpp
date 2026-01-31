#include "se3.h"
#include <cmath>

namespace pem {

SE3::SE3() {
    R.eye();
    t.zeros();
}

SE3::SE3(const Mat3& R_, const Vec3& t_) : R(R_), t(t_) {}

SE3 SE3::Identity() {
    return SE3(Mat3().eye(), Vec3().zeros());
}

Mat4 SE3::matrix() const {
    Mat4 T;
    T.zeros();
    T.submat(0,0,2,2) = R;
    T.submat(0,3,2,3) = t;
    T(3,3) = 1.0;
    return T;
}

SE3 SE3::inverse() const {
    Mat3 Rt = R.t();
    Vec3 tinv = -Rt * t;
    return SE3(Rt, tinv);
}

SE3 operator*(const SE3& a, const SE3& b) {
    Mat3 R = a.R * b.R;
    Vec3 t = a.R * b.t + a.t;
    return SE3(R, t);
}

Mat3 skew(const Vec3& w) {
    Mat3 W;
    W.zeros();
    W(0,1) = -w(2); W(0,2) =  w(1);
    W(1,0) =  w(2); W(1,2) = -w(0);
    W(2,0) = -w(1); W(2,1) =  w(0);
    return W;
}

Mat3 so3Exp(const Vec3& w) {
    double th = arma::norm(w);
    Mat3 W = skew(w);

    Mat3 R;
    R.eye();

    if (th < 1e-12) {
        R = R + W;
        return R;
    }

    double a = std::sin(th) / th;
    double b = (1.0 - std::cos(th)) / (th * th);
    R = R + a * W + b * (W * W);
    return R;
}

Vec3 so3Log(const Mat3& R) {
    double cos_th = (arma::trace(R) - 1.0) * 0.5;
    cos_th = std::min(1.0, std::max(-1.0, cos_th));
    double th = std::acos(cos_th);

    Vec3 w; w.zeros();
    if (th < 1e-12) {
        return w;
    }

    Mat3 W = (R - R.t()) * (0.5 * th / std::sin(th));
    w(0) = W(2,1);
    w(1) = W(0,2);
    w(2) = W(1,0);
    return w;
}

SE3 se3Exp(const Vec6& xi) {
    Vec3 w = xi.subvec(0,2);
    Vec3 v = xi.subvec(3,5);

    double th = arma::norm(w);
    Mat3 W = skew(w);
    Mat3 R = so3Exp(w);

    Mat3 V;
    V.eye();

    if (th < 1e-12) {
        V = V + 0.5 * W;
    } else {
        double th2 = th * th;
        double th3 = th2 * th;
        V = V
            + (1.0 - std::cos(th)) / th2 * W
            + (th - std::sin(th)) / th3 * (W * W);
    }

    Vec3 t = V * v;
    return SE3(R, t);
}

Mat3 orthonormalizeRot(const Mat3& M) {
    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, arma::mat(M));

    Mat3 R = arma::mat(U * V.t());
    double d = arma::det(R);
    if (d < 0.0) {
        U.col(2) *= -1.0;
        R = arma::mat(U * V.t());
    }
    return R;
}

SE3 orthonormalizeSE3(const SE3& T) {
    SE3 out = T;
    out.R = orthonormalizeRot(T.R);
    return out;
}

SE3 randomSE3(double rot_sigma, double trans_sigma) {
    Vec3 w = rot_sigma * arma::randn<arma::vec>(3);
    Mat3 R = so3Exp(w);
    Vec3 t = trans_sigma * arma::randn<arma::vec>(3);
    return SE3(R, t);
}

Mat3 perturbRot(const Mat3& R, double sigma) {
    Vec3 w = sigma * arma::randn<arma::vec>(3);
    Mat3 dR = so3Exp(w);
    return R * dR;
}

SE3 perturbSE3(const SE3& T, double rot_sigma, double trans_sigma) {
    SE3 out = T;
    out.R = perturbRot(T.R, rot_sigma);
    out.t = T.t + trans_sigma * arma::randn<arma::vec>(3);
    return out;
}

double rotationAngleDeg(const Mat3& R_err) {
    double cos_th = (arma::trace(R_err) - 1.0) * 0.5;
    cos_th = std::min(1.0, std::max(-1.0, cos_th));
    double th = std::acos(cos_th);
    return th * 180.0 / M_PI;
}

double norm3(const Vec3& v) {
    return arma::norm(v);
}

}  // namespace pem
