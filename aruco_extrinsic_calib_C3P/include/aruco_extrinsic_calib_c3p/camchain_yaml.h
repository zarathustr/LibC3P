#pragma once
#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include <string>

namespace aruco_extrinsic_calib_c3p {

// Load a Kalibr-style camchain.yaml with keys: cam0, cam1, cam2, ...,
// each with intrinsics, distortion_coeffs, resolution, rostopic,
// and optional cam(i)/T_cn_cnm1 (4x4) giving cam_i <- cam_(i-1).
//
// This loader also computes chain.T_cam_cam0[i] if extrinsics are available.
CamChain LoadCamChainYaml(const std::string& path);

// Convenience: get T_cam_i<-cam_j using the chained cam0 transforms.
// Returns identity and sets ok=false if unavailable.
cv::Matx44d GetT_cam_i_cam_j(const CamChain& chain, int cam_i, int cam_j, bool* ok = nullptr);

} // namespace aruco_extrinsic_calib_c3p
