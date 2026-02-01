#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <cmath>
#include <numeric>

namespace aruco_extrinsic_calib_c3p {

SE3d Inverse(const SE3d& T) {
  cv::Matx33d Rt = T.R.t();
  cv::Vec3d tt = -(Rt * T.t);
  return SE3d(Rt, tt);
}

SE3d Compose(const SE3d& A, const SE3d& B) {
  return SE3d(A.R * B.R, A.R * B.t + A.t);
}

cv::Matx44d ToMat4(const SE3d& T) {
  cv::Matx44d M = cv::Matx44d::eye();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) M(r,c) = T.R(r,c);
    M(r,3) = T.t(r);
  }
  return M;
}

SE3d FromRvecTvec(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);
  cv::Matx33d R;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      R(r,c) = R_cv.at<double>(r,c);
  return SE3d(R, tvec);
}

static inline double Clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

cv::Vec3d LogSO3(const cv::Matx33d& R) {
  // Log map from SO(3) to so(3), returned as rotation vector.
  double tr = R(0,0) + R(1,1) + R(2,2);
  double cos_theta = Clamp((tr - 1.0) * 0.5, -1.0, 1.0);
  double theta = std::acos(cos_theta);

  if (theta < 1e-12) {
    return cv::Vec3d(0,0,0);
  }

  double sin_theta = std::sin(theta);
  cv::Vec3d axis(
      (R(2,1) - R(1,2)) / (2.0 * sin_theta),
      (R(0,2) - R(2,0)) / (2.0 * sin_theta),
      (R(1,0) - R(0,1)) / (2.0 * sin_theta)
  );
  return axis * theta;
}

double RotationAngleDeg(const cv::Matx33d& R) {
  cv::Vec3d w = LogSO3(R);
  double ang = std::sqrt(w.dot(w));
  return ang * 180.0 / M_PI;
}

double RotationErrorDeg(const cv::Matx33d& R_est, const cv::Matx33d& R_ref) {
  cv::Matx33d R_err = R_est * R_ref.t();
  return RotationAngleDeg(R_err);
}

double TranslationError(const cv::Vec3d& t_est, const cv::Vec3d& t_ref) {
  cv::Vec3d d = t_est - t_ref;
  return std::sqrt(d.dot(d));
}

cv::Vec4d RotToQuatWXYZ(const cv::Matx33d& R) {
  // Robust conversion. Returns (w,x,y,z).
  double tr = R(0,0) + R(1,1) + R(2,2);
  double w,x,y,z;

  if (tr > 0.0) {
    double S = std::sqrt(tr + 1.0) * 2.0; // 4w
    w = 0.25 * S;
    x = (R(2,1) - R(1,2)) / S;
    y = (R(0,2) - R(2,0)) / S;
    z = (R(1,0) - R(0,1)) / S;
  } else if ((R(0,0) > R(1,1)) && (R(0,0) > R(2,2))) {
    double S = std::sqrt(1.0 + R(0,0) - R(1,1) - R(2,2)) * 2.0; // 4x
    w = (R(2,1) - R(1,2)) / S;
    x = 0.25 * S;
    y = (R(0,1) + R(1,0)) / S;
    z = (R(0,2) + R(2,0)) / S;
  } else if (R(1,1) > R(2,2)) {
    double S = std::sqrt(1.0 + R(1,1) - R(0,0) - R(2,2)) * 2.0; // 4y
    w = (R(0,2) - R(2,0)) / S;
    x = (R(0,1) + R(1,0)) / S;
    y = 0.25 * S;
    z = (R(1,2) + R(2,1)) / S;
  } else {
    double S = std::sqrt(1.0 + R(2,2) - R(0,0) - R(1,1)) * 2.0; // 4z
    w = (R(1,0) - R(0,1)) / S;
    x = (R(0,2) + R(2,0)) / S;
    y = (R(1,2) + R(2,1)) / S;
    z = 0.25 * S;
  }

  // Normalize
  double n = std::sqrt(w*w + x*x + y*y + z*z);
  if (n < 1e-12) return cv::Vec4d(1,0,0,0);
  return cv::Vec4d(w/n, x/n, y/n, z/n);
}

cv::Matx33d QuatToRotWXYZ(const cv::Vec4d& q) {
  // q = (w,x,y,z)
  double w=q[0], x=q[1], y=q[2], z=q[3];
  double ww=w*w, xx=x*x, yy=y*y, zz=z*z;

  cv::Matx33d R;
  R(0,0) = ww + xx - yy - zz;
  R(0,1) = 2*(x*y - w*z);
  R(0,2) = 2*(x*z + w*y);

  R(1,0) = 2*(x*y + w*z);
  R(1,1) = ww - xx + yy - zz;
  R(1,2) = 2*(y*z - w*x);

  R(2,0) = 2*(x*z - w*y);
  R(2,1) = 2*(y*z + w*x);
  R(2,2) = ww - xx - yy + zz;
  return R;
}

cv::Vec4d AverageQuaternionsWXYZ(const std::vector<cv::Vec4d>& quats_wxyz) {
  // Markley method: eigenvector of sum(q q^T)
  if (quats_wxyz.empty()) return cv::Vec4d(1,0,0,0);

  cv::Matx<double,4,4> A = cv::Matx<double,4,4>::zeros();
  for (auto q : quats_wxyz) {
    // Ensure consistent hemisphere (optional). We'll align to first quaternion.
    // We'll do alignment outside for simplicity (here: if w<0, flip).
    if (q[0] < 0) q *= -1.0;
    cv::Matx<double,4,1> v(q[0], q[1], q[2], q[3]);
    A += v * v.t();
  }

  cv::Mat evals, evecs;
  cv::eigen(cv::Mat(A), evals, evecs); // eigenvalues descending, rows of evecs
  cv::Vec4d q_avg(
    evecs.at<double>(0,0),
    evecs.at<double>(0,1),
    evecs.at<double>(0,2),
    evecs.at<double>(0,3)
  );
  // Normalize
  double n = std::sqrt(q_avg.dot(q_avg));
  if (n < 1e-12) return cv::Vec4d(1,0,0,0);
  return q_avg / n;
}

SE3d AverageSE3(const std::vector<SE3d>& Ts) {
  if (Ts.empty()) return SE3d();

  std::vector<cv::Vec4d> qs;
  qs.reserve(Ts.size());
  cv::Vec3d t_sum(0,0,0);
  for (const auto& T : Ts) {
    qs.push_back(RotToQuatWXYZ(T.R));
    t_sum += T.t;
  }
  cv::Vec4d q_avg = AverageQuaternionsWXYZ(qs);
  cv::Matx33d R_avg = QuatToRotWXYZ(q_avg);
  cv::Vec3d t_avg = t_sum * (1.0 / static_cast<double>(Ts.size()));
  return SE3d(R_avg, t_avg);
}

} // namespace aruco_extrinsic_calib_c3p
