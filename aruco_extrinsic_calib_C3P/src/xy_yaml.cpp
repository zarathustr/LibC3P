#include "aruco_extrinsic_calib_c3p/xy_yaml.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <stdexcept>
#include <string>

namespace aruco_extrinsic_calib_c3p {

namespace {

static cv::Matx44d ParseMat44(const YAML::Node& n, const std::string& name) {
  if (!n || !n.IsSequence() || n.size() != 4) {
    throw std::runtime_error("Expected 4x4 matrix for key '" + name + "'.");
  }
  cv::Matx44d M = cv::Matx44d::eye();
  for (size_t r = 0; r < 4; ++r) {
    const YAML::Node row = n[r];
    if (!row || !row.IsSequence() || row.size() != 4) {
      throw std::runtime_error("Expected 4 values in row " + std::to_string(r) + " for key '" + name + "'.");
    }
    for (size_t c = 0; c < 4; ++c) {
      M((int)r,(int)c) = row[c].as<double>();
    }
  }
  return M;
}

static void WriteMat44(std::ostream& os, const cv::Matx44d& M, const std::string& key) {
  os << key << ":\n";
  for (int r = 0; r < 4; ++r) {
    os << "  - [";
    for (int c = 0; c < 4; ++c) {
      os << M(r,c);
      if (c != 3) os << ", ";
    }
    os << "]\n";
  }
}

}  // namespace

XYEstimate LoadXYYaml(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  XYEstimate out;

  if (root["marker0_id"]) out.marker0_id = root["marker0_id"].as<int>();
  if (root["marker1_id"]) out.marker1_id = root["marker1_id"].as<int>();

  // Support multiple key aliases for convenience.
  if (root["X_c1_c0"]) {
    out.X_c1_c0 = ParseMat44(root["X_c1_c0"], "X_c1_c0");
    out.has_X = true;
  } else if (root["X"]) {
    out.X_c1_c0 = ParseMat44(root["X"], "X");
    out.has_X = true;
  }

  if (root["Y_m1_m0"]) {
    out.Y_m1_m0 = ParseMat44(root["Y_m1_m0"], "Y_m1_m0");
    out.has_Y = true;
  } else if (root["Y"]) {
    out.Y_m1_m0 = ParseMat44(root["Y"], "Y");
    out.has_Y = true;
  }

  if (!out.has_X || !out.has_Y) {
    throw std::runtime_error("XY yaml missing required keys. Need X_c1_c0 and Y_m1_m0 (or X/Y aliases): " + path);
  }
  return out;
}

void SaveXYYaml(const std::string& path, const XYEstimate& xy) {
  std::ofstream os(path);
  if (!os) {
    throw std::runtime_error("Failed to write: " + path);
  }
  os.setf(std::ios::fixed);
  os.precision(12);
  os << "marker0_id: " << xy.marker0_id << "\n";
  os << "marker1_id: " << xy.marker1_id << "\n";
  WriteMat44(os, xy.X_c1_c0, "X_c1_c0");
  WriteMat44(os, xy.Y_m1_m0, "Y_m1_m0");
}

}  // namespace aruco_extrinsic_calib_c3p
