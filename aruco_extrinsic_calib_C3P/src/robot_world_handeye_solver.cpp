#include "aruco_extrinsic_calib_c3p/camchain_yaml.h"
#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include "aruco_extrinsic_calib_c3p/handeye_axxb.h"
#include "aruco_extrinsic_calib_c3p/xy_yaml.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <set>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace {

struct ObsLine {
  uint64_t t_ns = 0;
  int frame_idx = 0;
  int marker_id = -1;
  aruco_extrinsic_calib_c3p::SE3d T_c_m;  // camera <- marker
};

static std::string GetArg(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::string(argv[i+1]);
  }
  return def;
}
static double GetArgD(int argc, char** argv, const std::string& key, double def) {
  std::string s = GetArg(argc, argv, key, "");
  if (s.empty()) return def;
  return std::stod(s);
}
static int GetArgI(int argc, char** argv, const std::string& key, int def) {
  std::string s = GetArg(argc, argv, key, "");
  if (s.empty()) return def;
  return std::stoi(s);
}

static bool ParseCsvLine(const std::string& line, ObsLine& out) {
  // Expected columns:
  // t_ns,frame_idx,marker_id,c0x,c0y,c1x,c1y,c2x,c2y,c3x,c3y,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z
  if (line.empty()) return false;
  if (line.rfind("t_ns", 0) == 0) return false;

  std::vector<std::string> tok;
  tok.reserve(32);
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) tok.push_back(item);
  if (tok.size() < 17) return false;

  out.t_ns = (uint64_t)std::stoull(tok[0]);
  out.frame_idx = std::stoi(tok[1]);
  out.marker_id = std::stoi(tok[2]);

  double rvec_x = std::stod(tok[11]);
  double rvec_y = std::stod(tok[12]);
  double rvec_z = std::stod(tok[13]);
  double tvec_x = std::stod(tok[14]);
  double tvec_y = std::stod(tok[15]);
  double tvec_z = std::stod(tok[16]);

  out.T_c_m = aruco_extrinsic_calib_c3p::FromRvecTvec(cv::Vec3d(rvec_x,rvec_y,rvec_z),
                                                   cv::Vec3d(tvec_x,tvec_y,tvec_z));
  return true;
}

static std::unordered_map<uint64_t, std::unordered_map<int, aruco_extrinsic_calib_c3p::SE3d>>
LoadObsByTimeMarker(const std::string& csv_path) {
  std::ifstream is(csv_path);
  if (!is) throw std::runtime_error("Failed to open " + csv_path);

  std::unordered_map<uint64_t, std::unordered_map<int, aruco_extrinsic_calib_c3p::SE3d>> out;
  std::string line;
  while (std::getline(is, line)) {
    ObsLine o;
    if (!ParseCsvLine(line, o)) continue;
    out[o.t_ns][o.marker_id] = o.T_c_m;
  }
  return out;
}

static std::vector<uint64_t> SortedKeys(
    const std::unordered_map<uint64_t, std::unordered_map<int, aruco_extrinsic_calib_c3p::SE3d>>& m) {
  std::vector<uint64_t> keys;
  keys.reserve(m.size());
  for (const auto& kv : m) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  return keys;
}

static std::vector<std::pair<uint64_t,uint64_t>> SyncTimes(
    const std::vector<uint64_t>& a_ns,
    const std::vector<uint64_t>& b_ns,
    uint64_t tol_ns) {
  std::vector<std::pair<uint64_t,uint64_t>> out;
  size_t j = 0;
  for (size_t i = 0; i < a_ns.size(); ++i) {
    uint64_t t = a_ns[i];
    while (j + 1 < b_ns.size() && b_ns[j+1] < t) j++;
    uint64_t best = 0;
    uint64_t best_dt = (uint64_t)-1;

    auto try_j = [&](size_t jj){
      if (jj >= b_ns.size()) return;
      uint64_t dt = (b_ns[jj] > t) ? (b_ns[jj] - t) : (t - b_ns[jj]);
      if (dt < best_dt) { best_dt = dt; best = b_ns[jj]; }
    };

    if (j < b_ns.size()) try_j(j);
    if (j + 1 < b_ns.size()) try_j(j+1);
    if (j > 0) try_j(j-1);

    if (best_dt <= tol_ns) out.emplace_back(t, best);
  }
  return out;
}

static aruco_extrinsic_calib_c3p::SE3d Mat44ToSE3(const cv::Matx44d& M) {
  cv::Matx33d R(M(0,0),M(0,1),M(0,2),
                M(1,0),M(1,1),M(1,2),
                M(2,0),M(2,1),M(2,2));
  cv::Vec3d t(M(0,3), M(1,3), M(2,3));
  return aruco_extrinsic_calib_c3p::SE3d(R, t);
}

static double Norm3(const cv::Vec3d& v) {
  return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

} // namespace

int main(int argc, char** argv) {
  const std::string calib_path = GetArg(argc, argv, "--calib", "");
  const std::string cam0_csv = GetArg(argc, argv, "--cam0_csv", "");
  const std::string cam1_csv = GetArg(argc, argv, "--cam1_csv", "");
  const std::string out_xy_yaml = GetArg(argc, argv, "--out_xy", "");

  const int cam0_idx = GetArgI(argc, argv, "--cam0_idx", 0);
  const int cam1_idx = GetArgI(argc, argv, "--cam1_idx", 1);

  const int marker0_id = GetArgI(argc, argv, "--marker0_id", 0);
  const int marker1_id = GetArgI(argc, argv, "--marker1_id", 1);

  const double sync_tol_s = GetArgD(argc, argv, "--sync_tol", 0.01);

  const double tag_size_m = GetArgD(argc, argv, "--tag_size", 0.25);
  const double marker_margin_m = GetArgD(argc, argv, "--marker_margin", 0.01);

  const double min_rot_deg = GetArgD(argc, argv, "--min_rot_deg", 0.5);
  const double min_trans_m = GetArgD(argc, argv, "--min_trans_m", 0.002);

  // Optional known grid offset between marker0 and marker1 (in #steps).
  // For a 2x2 board, typical values are (1,0), (0,1), or (1,1).
  const int grid_dx = GetArgI(argc, argv, "--grid_dx", 0);
  const int grid_dy = GetArgI(argc, argv, "--grid_dy", 0);

  if (cam0_csv.empty() || cam1_csv.empty()) {
    std::cerr << R"(Usage:
  robot_world_handeye_solver --calib calib-camchain.yaml \
    --cam0_csv out/cam0_aruco_poses.csv --cam1_csv out/cam1_aruco_poses.csv \
    [--cam0_idx 0] [--cam1_idx 1] \
    [--marker0_id 0] [--marker1_id 1] [--sync_tol 0.01] \
    [--tag_size 0.25] [--marker_margin 0.01] \
    [--min_rot_deg 0.5] [--min_trans_m 0.002] \
    [--grid_dx 0] [--grid_dy 0] [--out_xy out/estimated_XY.yaml]
)";
    return 1;
  }

  // Load calibration (optional) for verifying X against camchain.yaml.
  bool have_X_calib = false;
  aruco_extrinsic_calib_c3p::SE3d X_calib;

  if (!calib_path.empty()) {
    try {
      aruco_extrinsic_calib_c3p::CamChain chain = aruco_extrinsic_calib_c3p::LoadCamChainYaml(calib_path);
      bool ok = false;
      cv::Matx44d Xmat = aruco_extrinsic_calib_c3p::GetT_cam_i_cam_j(chain, cam1_idx, cam0_idx, &ok); // cam1 <- cam0
      if (ok) {
        X_calib = Mat44ToSE3(Xmat);
        have_X_calib = true;
      } else {
        std::cerr << "Warning: calib chain does not provide T_cam" << cam1_idx << "_cam" << cam0_idx << " (or chain incomplete)\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "Warning: failed to load calib: " << e.what() << "\n";
    }
  }

  // Load observations
  auto obs0 = LoadObsByTimeMarker(cam0_csv);
  auto obs1 = LoadObsByTimeMarker(cam1_csv);
  auto t0 = SortedKeys(obs0);
  auto t1 = SortedKeys(obs1);

  uint64_t tol_ns = (uint64_t)std::llround(sync_tol_s * 1e9);
  auto pairs = SyncTimes(t0, t1, tol_ns);

  // Build A_i, B_i (poses) for equation:  A_i X = Y B_i
  // Here:
  //   X = T_c1_c0 (cam1 <- cam0)
  //   Y = T_m1_m0 (marker1 <- marker0)
  //   A_i = T_m1_c1 (marker1 <- cam1) = inv(T_c1_m1)
  //   B_i = T_m0_c0 (marker0 <- cam0) = inv(T_c0_m0)
  std::vector<aruco_extrinsic_calib_c3p::SE3d> A_pose, B_pose;
  A_pose.reserve(pairs.size());
  B_pose.reserve(pairs.size());

  for (const auto& p : pairs) {
    auto it0 = obs0.find(p.first);
    auto it1 = obs1.find(p.second);
    if (it0 == obs0.end() || it1 == obs1.end()) continue;

    auto it_m0 = it0->second.find(marker0_id);
    auto it_m1 = it1->second.find(marker1_id);
    if (it_m0 == it0->second.end()) continue;
    if (it_m1 == it1->second.end()) continue;

    const auto& T_c0_m0 = it_m0->second;
    const auto& T_c1_m1 = it_m1->second;

    auto B_i = aruco_extrinsic_calib_c3p::Inverse(T_c0_m0); // m0 <- c0
    auto A_i = aruco_extrinsic_calib_c3p::Inverse(T_c1_m1); // m1 <- c1

    B_pose.push_back(B_i);
    A_pose.push_back(A_i);
  }

  std::cout << "Synchronized samples usable for AX=YB: " << A_pose.size()
            << " (marker0_id=" << marker0_id << ", marker1_id=" << marker1_id << ")\n";
  if (A_pose.size() < 3) {
    std::cerr << "Not enough samples (need >=3) to form >=2 relative motion pairs.\n";
    return 1;
  }

  // Convert AX=YB into A_rel X = X B_rel using pairs of measurements:
  // From A_i X = Y B_i and A_j X = Y B_j:
  //   (A_j^-1 A_i) X = X (B_j^-1 B_i)
  // We'll use consecutive pairs (j=i-1).
  std::vector<aruco_extrinsic_calib_c3p::SE3d> A_rel, B_rel;
  A_rel.reserve(A_pose.size() - 1);
  B_rel.reserve(B_pose.size() - 1);

  size_t skipped_small = 0;
  for (size_t i = 1; i < A_pose.size(); ++i) {
    auto Aij = aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Inverse(A_pose[i-1]), A_pose[i]);
    auto Bij = aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Inverse(B_pose[i-1]), B_pose[i]);

    const double rotA = aruco_extrinsic_calib_c3p::RotationAngleDeg(Aij.R);
    const double rotB = aruco_extrinsic_calib_c3p::RotationAngleDeg(Bij.R);
    const double transA = Norm3(Aij.t);
    const double transB = Norm3(Bij.t);

    // Skip near-identity motions (they contribute little and can hurt conditioning).
    if (std::max(rotA, rotB) < min_rot_deg && std::max(transA, transB) < min_trans_m) {
      skipped_small++;
      continue;
    }

    A_rel.push_back(Aij);
    B_rel.push_back(Bij);
  }

  std::cout << "Relative motion pairs: " << A_rel.size()
            << " (skipped small motions: " << skipped_small << ")\n";
  if (A_rel.size() < 2) {
    std::cerr << "Not enough informative motion pairs for solving X (need >=2).\n";
    return 1;
  }

  // Solve A_rel * X = X * B_rel  (hand-eye).
  aruco_extrinsic_calib_c3p::SE3d X_est;
  try {
    X_est = aruco_extrinsic_calib_c3p::SolveAXXB_ParkMartin(A_rel, B_rel);
  } catch (const std::exception& e) {
    std::cerr << "SolveAXXB failed: " << e.what() << "\n";
    return 1;
  }

  auto Xm = aruco_extrinsic_calib_c3p::ToMat4(X_est);
  std::cout << "\nEstimated X = T_c1_c0 (cam1 <- cam0):\n"
            << std::setprecision(12)
            << Xm(0,0) << " " << Xm(0,1) << " " << Xm(0,2) << " " << Xm(0,3) << "\n"
            << Xm(1,0) << " " << Xm(1,1) << " " << Xm(1,2) << " " << Xm(1,3) << "\n"
            << Xm(2,0) << " " << Xm(2,1) << " " << Xm(2,2) << " " << Xm(2,3) << "\n"
            << "0 0 0 1\n";

  if (have_X_calib) {
    std::cout << "X vs calib (cam1/T_cn_cnm1): rot_err_deg="
              << aruco_extrinsic_calib_c3p::RotationErrorDeg(X_est.R, X_calib.R)
              << " trans_err_m="
              << aruco_extrinsic_calib_c3p::TranslationError(X_est.t, X_calib.t)
              << "\n";
  }

  // Recover Y from each measurement: Y = A_i X inv(B_i), then average.
  std::vector<aruco_extrinsic_calib_c3p::SE3d> Ys;
  Ys.reserve(A_pose.size());
  for (size_t i = 0; i < A_pose.size(); ++i) {
    auto Yi = aruco_extrinsic_calib_c3p::Compose(
                aruco_extrinsic_calib_c3p::Compose(A_pose[i], X_est),
                aruco_extrinsic_calib_c3p::Inverse(B_pose[i])); // inv(B) = c0 <- m0
    Ys.push_back(Yi);
  }
  auto Y_est = aruco_extrinsic_calib_c3p::AverageSE3(Ys);
  auto Ym = aruco_extrinsic_calib_c3p::ToMat4(Y_est);

  std::cout << "\nEstimated Y = T_m1_m0 (marker1 <- marker0):\n"
            << std::setprecision(12)
            << Ym(0,0) << " " << Ym(0,1) << " " << Ym(0,2) << " " << Ym(0,3) << "\n"
            << Ym(1,0) << " " << Ym(1,1) << " " << Ym(1,2) << " " << Ym(1,3) << "\n"
            << Ym(2,0) << " " << Ym(2,1) << " " << Ym(2,2) << " " << Ym(2,3) << "\n"
            << "0 0 0 1\n";

  // Geometry check for Y based on marker size + margin.
  const double step = tag_size_m + marker_margin_m; // center-to-center between adjacent markers if origins are at centers.
  const double dist_3d = Norm3(Y_est.t);
  const double dist_xy = std::sqrt(Y_est.t[0]*Y_est.t[0] + Y_est.t[1]*Y_est.t[1]);
  const double abs_z = std::abs(Y_est.t[2]);

  std::cout << "\nY translation magnitude: |t|=" << dist_3d
            << " m (xy=" << dist_xy << " m, |z|=" << abs_z << " m)\n";
  std::cout << "Y rotation angle from identity: " << aruco_extrinsic_calib_c3p::RotationAngleDeg(Y_est.R) << " deg\n";
  std::cout << "Assumed adjacent marker step = tag_size + margin = " << tag_size_m << " + " << marker_margin_m
            << " = " << step << " m\n";

  if (grid_dx != 0 || grid_dy != 0) {
    const double exp_dist = step * std::sqrt((double)(grid_dx*grid_dx + grid_dy*grid_dy));
    std::cout << "Expected center distance from (grid_dx,grid_dy)=(" << grid_dx << "," << grid_dy
              << "): " << exp_dist << " m\n";
    std::cout << "Distance error: " << std::abs(dist_xy - exp_dist) << " m\n";
  } else {
    // For a 2x2 board, typical possibilities between distinct markers are:
    // adjacent: step, diagonal: step*sqrt(2).
    const double cand1 = step;
    const double cand2 = step * std::sqrt(2.0);
    const double e1 = std::abs(dist_xy - cand1);
    const double e2 = std::abs(dist_xy - cand2);
    if (e1 <= e2) {
      std::cout << "Nearest expected distance (2x2 assumption): step = " << cand1
                << " m, error=" << e1 << " m\n";
    } else {
      std::cout << "Nearest expected distance (2x2 assumption): step*sqrt(2) = " << cand2
                << " m, error=" << e2 << " m\n";
    }
  }

  std::cout << "\nDone.\n";

  if (!out_xy_yaml.empty()) {
    try {
      aruco_extrinsic_calib_c3p::XYEstimate xy;
      xy.marker0_id = marker0_id;
      xy.marker1_id = marker1_id;
      xy.has_X = true;
      xy.has_Y = true;
      xy.X_c1_c0 = aruco_extrinsic_calib_c3p::ToMat4(X_est);
      xy.Y_m1_m0 = aruco_extrinsic_calib_c3p::ToMat4(Y_est);
      aruco_extrinsic_calib_c3p::SaveXYYaml(out_xy_yaml, xy);
      std::cout << "Wrote estimated X/Y to: " << out_xy_yaml << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to write --out_xy: " << e.what() << "\n";
      return 1;
    }
  }

  return 0;
}
