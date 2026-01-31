#pragma once
#include <armadillo>
#include <iostream>

namespace pem {

using Mat3 = arma::mat::fixed<3,3>;
using Mat4 = arma::mat::fixed<4,4>;
using Vec3 = arma::vec::fixed<3>;
using Vec6 = arma::vec::fixed<6>;

Mat3 skew(const Vec3& w);

Mat3 so3Exp(const Vec3& w);
Vec3 so3Log(const Mat3& R);

struct SE3 {
    Mat3 R;
    Vec3 t;

    SE3();
    SE3(const Mat3& R_, const Vec3& t_);

    static SE3 Identity();

    Mat4 matrix() const;

    SE3 inverse() const;
};

SE3 operator*(const SE3& a, const SE3& b);

SE3 se3Exp(const Vec6& xi);

SE3 orthonormalizeSE3(const SE3& T);
Mat3 orthonormalizeRot(const Mat3& M);

SE3 randomSE3(double rot_sigma, double trans_sigma);
Mat3 perturbRot(const Mat3& R, double sigma);
SE3 perturbSE3(const SE3& T, double rot_sigma, double trans_sigma);

double rotationAngleDeg(const Mat3& R_err);
double norm3(const Vec3& v);

}  // namespace pem
