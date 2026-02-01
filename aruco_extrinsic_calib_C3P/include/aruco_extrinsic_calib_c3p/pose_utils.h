#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <unordered_map>

namespace aruco_extrinsic_calib_c3p {

struct SE3d {
  cv::Matx33d R;  // rotation
  cv::Vec3d t;    // translation

  SE3d() : R(cv::Matx33d::eye()), t(0,0,0) {}
  SE3d(const cv::Matx33d& R_, const cv::Vec3d& t_) : R(R_), t(t_) {}
};

SE3d Inverse(const SE3d& T);
SE3d Compose(const SE3d& A, const SE3d& B);  // A * B
cv::Matx44d ToMat4(const SE3d& T);

SE3d FromRvecTvec(const cv::Vec3d& rvec, const cv::Vec3d& tvec);
cv::Vec3d LogSO3(const cv::Matx33d& R);

double RotationAngleDeg(const cv::Matx33d& R);
double RotationErrorDeg(const cv::Matx33d& R_est, const cv::Matx33d& R_ref);
double TranslationError(const cv::Vec3d& t_est, const cv::Vec3d& t_ref);

cv::Vec4d RotToQuatWXYZ(const cv::Matx33d& R);
cv::Matx33d QuatToRotWXYZ(const cv::Vec4d& q_wxyz);
cv::Vec4d AverageQuaternionsWXYZ(const std::vector<cv::Vec4d>& quats_wxyz);

SE3d AverageSE3(const std::vector<SE3d>& Ts);

struct CameraModel {
  std::string name;
  std::string rostopic;
  int width = 0;
  int height = 0;
  cv::Matx33d K = cv::Matx33d::eye();
  cv::Vec4d D = cv::Vec4d(0,0,0,0); // k1 k2 p1 p2 (radtan)

  // Kalibr-style extrinsic to previous camera in the chain:
  //   cam(i)/T_cn_cnm1 : cam_i <- cam_(i-1)
  bool has_T_to_prev = false;
  cv::Matx44d T_this_prev = cv::Matx44d::eye();
};

struct CamChain {
  std::vector<CameraModel> cams;

  // Derived chain transforms:
  //   T_cam_cam0[i] : cam_i <- cam0
  // If chain extrinsics are incomplete, has_T_cam0[i] may be false.
  std::vector<bool> has_T_cam0;
  std::vector<cv::Matx44d> T_cam_cam0;

  int size() const { return static_cast<int>(cams.size()); }
};

} // namespace aruco_extrinsic_calib_c3p
