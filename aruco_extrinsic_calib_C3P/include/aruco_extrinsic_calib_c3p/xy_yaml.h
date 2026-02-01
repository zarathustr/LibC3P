#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace aruco_extrinsic_calib_c3p {

// Stores an estimated camera-camera transform X and marker-marker transform Y.
// Convention (kept for backward compatibility with earlier tools):
//   X_c1_c0 : cam1 <- cam0
//   Y_m1_m0 : marker1 <- marker0
struct XYEstimate {
  int marker0_id = 0;
  int marker1_id = 1;

  bool has_X = false;
  bool has_Y = false;

  cv::Matx44d X_c1_c0 = cv::Matx44d::eye();
  cv::Matx44d Y_m1_m0 = cv::Matx44d::eye();
};

// Load an XY estimate YAML file written by robot_world_handeye_solver.
// Expected keys:
//   marker0_id, marker1_id
//   X_c1_c0: 4x4
//   Y_m1_m0: 4x4
XYEstimate LoadXYYaml(const std::string& path);

// Write an XY estimate YAML file.
void SaveXYYaml(const std::string& path, const XYEstimate& xy);

}  // namespace aruco_extrinsic_calib_c3p
