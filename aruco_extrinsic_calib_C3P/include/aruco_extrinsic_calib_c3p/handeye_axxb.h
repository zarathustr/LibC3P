#pragma once
#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include <vector>

namespace aruco_extrinsic_calib_c3p {

// Solve A_i X = X B_i for X in SE(3).
// This implementation uses a Park–Martin style approach:
//  - rotation: solve Wahba problem on log(R) vectors
//  - translation: linear least squares
//
// Requirements: at least 2 motion pairs; better with many.
SE3d SolveAXXB_ParkMartin(const std::vector<SE3d>& A, const std::vector<SE3d>& B);

} // namespace aruco_extrinsic_calib_c3p
