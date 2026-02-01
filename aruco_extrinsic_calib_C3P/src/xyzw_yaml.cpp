#include "aruco_extrinsic_calib_c3p/xyzw_yaml.h"

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
      M((int)r, (int)c) = row[c].as<double>();
    }
  }
  return M;
}

static void WriteMat44(std::ostream& os, const cv::Matx44d& M, const std::string& key) {
  os << key << ":\n";
  for (int r = 0; r < 4; ++r) {
    os << "  - [";
    for (int c = 0; c < 4; ++c) {
      os << M(r, c);
      if (c != 3) os << ", ";
    }
    os << "]\n";
  }
}

}  // namespace

XYZWEstimate LoadXYZWYaml(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  XYZWEstimate out;

  if (root["cam0"]) out.cam0 = root["cam0"].as<int>();
  if (root["cam1"]) out.cam1 = root["cam1"].as<int>();
  if (root["cam2"]) out.cam2 = root["cam2"].as<int>();
  if (root["cam3"]) out.cam3 = root["cam3"].as<int>();

  if (root["marker0"]) out.marker0 = root["marker0"].as<int>();
  if (root["marker1"]) out.marker1 = root["marker1"].as<int>();
  if (root["marker2"]) out.marker2 = root["marker2"].as<int>();
  if (root["marker3"]) out.marker3 = root["marker3"].as<int>();

  // X
  if (root["X_c1_c0"]) {
    out.X_c1_c0 = ParseMat44(root["X_c1_c0"], "X_c1_c0");
    out.has_X = true;
  } else if (root["X"]) {
    out.X_c1_c0 = ParseMat44(root["X"], "X");
    out.has_X = true;
  }

  // W
  if (root["W_c3_c2"]) {
    out.W_c3_c2 = ParseMat44(root["W_c3_c2"], "W_c3_c2");
    out.has_W = true;
  } else if (root["W"]) {
    out.W_c3_c2 = ParseMat44(root["W"], "W");
    out.has_W = true;
  }

  // Y
  if (root["Y_m0_m2"]) {
    out.Y_m0_m2 = ParseMat44(root["Y_m0_m2"], "Y_m0_m2");
    out.has_Y = true;
  } else if (root["Y"]) {
    out.Y_m0_m2 = ParseMat44(root["Y"], "Y");
    out.has_Y = true;
  }

  // Z
  if (root["Z_m1_m3"]) {
    out.Z_m1_m3 = ParseMat44(root["Z_m1_m3"], "Z_m1_m3");
    out.has_Z = true;
  } else if (root["Z"]) {
    out.Z_m1_m3 = ParseMat44(root["Z"], "Z");
    out.has_Z = true;
  }

  if (!out.has_X || !out.has_Y || !out.has_Z || !out.has_W) {
    throw std::runtime_error("XYZW yaml missing required keys (need X,Y,Z,W or X_c1_c0,Y_m0_m2,Z_m1_m3,W_c3_c2): " + path);
  }

  return out;
}

void SaveXYZWYaml(const std::string& path, const XYZWEstimate& est) {
  std::ofstream os(path);
  if (!os) {
    throw std::runtime_error("Failed to write: " + path);
  }
  os.setf(std::ios::fixed);
  os.precision(12);

  os << "cam0: " << est.cam0 << "\n";
  os << "cam1: " << est.cam1 << "\n";
  os << "cam2: " << est.cam2 << "\n";
  os << "cam3: " << est.cam3 << "\n";

  os << "marker0: " << est.marker0 << "\n";
  os << "marker1: " << est.marker1 << "\n";
  os << "marker2: " << est.marker2 << "\n";
  os << "marker3: " << est.marker3 << "\n";

  WriteMat44(os, est.X_c1_c0, "X_c1_c0");
  WriteMat44(os, est.W_c3_c2, "W_c3_c2");
  WriteMat44(os, est.Y_m0_m2, "Y_m0_m2");
  WriteMat44(os, est.Z_m1_m3, "Z_m1_m3");
}

}  // namespace aruco_extrinsic_calib_c3p
