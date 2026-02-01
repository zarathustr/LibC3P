#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace aruco_extrinsic_calib_c3p {

// Stores an estimated set of transforms used by the 4-camera / 4-marker loop:
//
//   A_i X B_i Y = Z C_i W D_i
//
// Conventions used in this project:
//   X_c1_c0 : cam1 <- cam0
//   W_c3_c2 : cam3 <- cam2
//   Y_m0_m2 : marker0 <- marker2
//   Z_m1_m3 : marker1 <- marker3
//
// The indices/IDs are stored only for convenience when overlaying.
struct XYZWEstimate {
  int cam0 = 0;
  int cam1 = 1;
  int cam2 = 2;
  int cam3 = 3;

  int marker0 = 0;
  int marker1 = 1;
  int marker2 = 2;
  int marker3 = 3;

  bool has_X = false;
  bool has_Y = false;
  bool has_Z = false;
  bool has_W = false;

  cv::Matx44d X_c1_c0 = cv::Matx44d::eye();
  cv::Matx44d Y_m0_m2 = cv::Matx44d::eye();
  cv::Matx44d Z_m1_m3 = cv::Matx44d::eye();
  cv::Matx44d W_c3_c2 = cv::Matx44d::eye();
};

// Load an XYZW estimate YAML file.
// Expected keys:
//   cam0,cam1,cam2,cam3
//   marker0,marker1,marker2,marker3
//   X_c1_c0, Y_m0_m2, Z_m1_m3, W_c3_c2 (each 4x4)
// Also supports aliases: X,Y,Z,W.
XYZWEstimate LoadXYZWYaml(const std::string& path);

// Write an XYZW estimate YAML file.
void SaveXYZWYaml(const std::string& path, const XYZWEstimate& est);

}  // namespace aruco_extrinsic_calib_c3p
