#include "aruco_extrinsic_calib_c3p/handeye_axxb.h"
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>

namespace aruco_extrinsic_calib_c3p {

SE3d SolveAXXB_ParkMartin(const std::vector<SE3d>& A, const std::vector<SE3d>& B) {
  if (A.size() != B.size()) {
    throw std::runtime_error("SolveAXXB_ParkMartin: A and B must have same length");
  }
  if (A.size() < 2) {
    throw std::runtime_error("SolveAXXB_ParkMartin: need at least 2 motion pairs");
  }

  // --- Rotation ---
  // Minimize sum || log(R_Ai) - R * log(R_Bi) ||.
  cv::Matx33d M = cv::Matx33d::zeros();
  for (size_t i = 0; i < A.size(); ++i) {
    cv::Vec3d a = LogSO3(A[i].R);
    cv::Vec3d b = LogSO3(B[i].R);
    // If either is near zero, skip (not informative)
    if (cv::norm(a) < 1e-10 || cv::norm(b) < 1e-10) continue;

    // Outer product a * b^T
    M += cv::Matx33d(
      a[0]*b[0], a[0]*b[1], a[0]*b[2],
      a[1]*b[0], a[1]*b[1], a[1]*b[2],
      a[2]*b[0], a[2]*b[1], a[2]*b[2]
    );
  }

  // NOTE: Avoid the "most vexing parse" on some compilers (e.g. clang) where
  // `cv::SVD svd(cv::Mat(M));` is interpreted as a function declaration.
  // Supplying the flags argument makes this unambiguously a variable.
  cv::SVD svd(cv::Mat(M), cv::SVD::FULL_UV);
  cv::Mat U = svd.u;
  cv::Mat Vt = svd.vt;
  cv::Mat R_cv = U * Vt;

  // Ensure det = +1
  double detR = cv::determinant(R_cv);
  if (detR < 0) {
    cv::Mat D = cv::Mat::eye(3,3,CV_64F);
    D.at<double>(2,2) = -1.0;
    R_cv = U * D * Vt;
  }

  cv::Matx33d R;
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) R(r,c)=R_cv.at<double>(r,c);

  // --- Translation ---
  // (R_Ai - I) t = R * t_Bi - t_Ai
  cv::Mat A_ls(3 * (int)A.size(), 3, CV_64F);
  cv::Mat b_ls(3 * (int)A.size(), 1, CV_64F);

  for (size_t i = 0; i < A.size(); ++i) {
    cv::Matx33d Ra = A[i].R;
    cv::Vec3d ta = A[i].t;
    cv::Vec3d tb = B[i].t;

    cv::Matx33d L = Ra - cv::Matx33d::eye();
    cv::Vec3d rhs = R * tb - ta;

    for (int r = 0; r < 3; ++r) {
      A_ls.at<double>(3*(int)i + r, 0) = L(r,0);
      A_ls.at<double>(3*(int)i + r, 1) = L(r,1);
      A_ls.at<double>(3*(int)i + r, 2) = L(r,2);
      b_ls.at<double>(3*(int)i + r, 0) = rhs(r);
    }
  }

  cv::Mat t_ls;
  cv::solve(A_ls, b_ls, t_ls, cv::DECOMP_SVD);

  cv::Vec3d t(t_ls.at<double>(0,0), t_ls.at<double>(1,0), t_ls.at<double>(2,0));
  return SE3d(R, t);
}

} // namespace aruco_extrinsic_calib_c3p
