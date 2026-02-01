#include "aruco_extrinsic_calib_c3p/camchain_yaml.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace aruco_extrinsic_calib_c3p {

static cv::Matx33d KFromIntrinsics(double fx, double fy, double cx, double cy) {
  return cv::Matx33d(
    fx, 0,  cx,
    0,  fy, cy,
    0,  0,  1
  );
}

static cv::Matx44d Mat44FromYaml(const YAML::Node& n) {
  if (!n || !n.IsSequence() || n.size() != 4) {
    throw std::runtime_error("Expected 4x4 matrix YAML node");
  }
  cv::Matx44d M;
  for (size_t r = 0; r < 4; ++r) {
    if (!n[r].IsSequence() || n[r].size() != 4) {
      throw std::runtime_error("Expected 4 columns in 4x4 matrix");
    }
    for (size_t c = 0; c < 4; ++c) {
      M(static_cast<int>(r), static_cast<int>(c)) = n[r][c].as<double>();
    }
  }
  return M;
}

static CameraModel LoadOneCam(const std::string& name, const YAML::Node& camNode) {
  CameraModel cam;
  cam.name = name;

  if (camNode["rostopic"]) cam.rostopic = camNode["rostopic"].as<std::string>();
  else cam.rostopic = "";

  auto res = camNode["resolution"];
  cam.width  = res[0].as<int>();
  cam.height = res[1].as<int>();

  auto intr = camNode["intrinsics"];
  cam.K = KFromIntrinsics(intr[0].as<double>(), intr[1].as<double>(), intr[2].as<double>(), intr[3].as<double>());

  auto dist = camNode["distortion_coeffs"];
  if (!dist || !dist.IsSequence() || dist.size() < 4) {
    throw std::runtime_error("distortion_coeffs must have at least 4 elements (k1,k2,p1,p2)");
  }
  cam.D = cv::Vec4d(dist[0].as<double>(), dist[1].as<double>(), dist[2].as<double>(), dist[3].as<double>());

  if (camNode["T_cn_cnm1"]) {
    cam.has_T_to_prev = true;
    cam.T_this_prev = Mat44FromYaml(camNode["T_cn_cnm1"]);
  }

  return cam;
}

static bool IsCamKey(const std::string& k, int* idx_out) {
  // Accept "cam0", "cam1", ...
  if (k.size() < 4) return false;
  if (k.rfind("cam", 0) != 0) return false;
  for (size_t i = 3; i < k.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(k[i]))) return false;
  }
  int idx = std::stoi(k.substr(3));
  if (idx < 0) return false;
  if (idx_out) *idx_out = idx;
  return true;
}

static cv::Matx44d InverseMat44(const cv::Matx44d& T) {
  cv::Matx33d R(
    T(0,0), T(0,1), T(0,2),
    T(1,0), T(1,1), T(1,2),
    T(2,0), T(2,1), T(2,2)
  );
  cv::Vec3d t(T(0,3), T(1,3), T(2,3));
  cv::Matx33d Rt = R.t();
  cv::Vec3d tt = -(Rt * t);
  cv::Matx44d inv = cv::Matx44d::eye();
  inv(0,0)=Rt(0,0); inv(0,1)=Rt(0,1); inv(0,2)=Rt(0,2); inv(0,3)=tt[0];
  inv(1,0)=Rt(1,0); inv(1,1)=Rt(1,1); inv(1,2)=Rt(1,2); inv(1,3)=tt[1];
  inv(2,0)=Rt(2,0); inv(2,1)=Rt(2,1); inv(2,2)=Rt(2,2); inv(2,3)=tt[2];
  return inv;
}

CamChain LoadCamChainYaml(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);

  // Collect camN nodes.
  std::map<int, YAML::Node> nodes;
  for (auto it = root.begin(); it != root.end(); ++it) {
    if (!it->first.IsScalar()) continue;
    std::string key = it->first.as<std::string>();
    int idx = -1;
    if (IsCamKey(key, &idx)) {
      nodes[idx] = it->second;
    }
  }

  if (nodes.empty() || nodes.find(0) == nodes.end()) {
    throw std::runtime_error("camchain.yaml must contain at least 'cam0'");
  }

  // Build chain in index order.
  CamChain chain;
  chain.cams.reserve(nodes.size());

  for (const auto& kv : nodes) {
    int idx = kv.first;
    const YAML::Node& camNode = kv.second;
    const std::string name = "cam" + std::to_string(idx);
    chain.cams.push_back(LoadOneCam(name, camNode));
  }

  // Compute T_cam_cam0 by chaining T_cn_cnm1 (cam_i <- cam_(i-1)).
  const int N = static_cast<int>(chain.cams.size());
  chain.has_T_cam0.assign(N, false);
  chain.T_cam_cam0.assign(N, cv::Matx44d::eye());

  chain.has_T_cam0[0] = true;
  chain.T_cam_cam0[0] = cv::Matx44d::eye();

  for (int i = 1; i < N; ++i) {
    if (!chain.cams[i].has_T_to_prev) {
      chain.has_T_cam0[i] = false;
      continue;
    }
    if (!chain.has_T_cam0[i-1]) {
      chain.has_T_cam0[i] = false;
      continue;
    }
    chain.T_cam_cam0[i] = chain.cams[i].T_this_prev * chain.T_cam_cam0[i-1];
    chain.has_T_cam0[i] = true;
  }

  return chain;
}

cv::Matx44d GetT_cam_i_cam_j(const CamChain& chain, int cam_i, int cam_j, bool* ok) {
  if (ok) *ok = false;
  const int N = chain.size();
  if (cam_i < 0 || cam_j < 0 || cam_i >= N || cam_j >= N) {
    return cv::Matx44d::eye();
  }
  if (chain.has_T_cam0.size() != static_cast<size_t>(N) ||
      chain.T_cam_cam0.size() != static_cast<size_t>(N)) {
    return cv::Matx44d::eye();
  }
  if (!chain.has_T_cam0[cam_i] || !chain.has_T_cam0[cam_j]) {
    return cv::Matx44d::eye();
  }

  const cv::Matx44d& Ti0 = chain.T_cam_cam0[cam_i];  // cam_i <- cam0
  const cv::Matx44d& Tj0 = chain.T_cam_cam0[cam_j];  // cam_j <- cam0

  cv::Matx44d T0j = InverseMat44(Tj0);               // cam0 <- cam_j
  cv::Matx44d Tij = Ti0 * T0j;                       // cam_i <- cam_j

  if (ok) *ok = true;
  return Tij;
}

} // namespace aruco_extrinsic_calib_c3p
