#include "aruco_extrinsic_calib_c3p/camchain_yaml.h"
#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include "aruco_extrinsic_calib_c3p/handeye_axxb.h"

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
  aruco_extrinsic_calib_c3p::SE3d T_c_m; // camera <- marker
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
  // Columns:
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

static std::vector<uint64_t> SortedKeys(const std::unordered_map<uint64_t, std::unordered_map<int, aruco_extrinsic_calib_c3p::SE3d>>& m) {
  std::vector<uint64_t> keys;
  keys.reserve(m.size());
  for (const auto& kv : m) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  return keys;
}

static std::vector<std::pair<uint64_t,uint64_t>> MatchTimes(
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
  cv::Matx33d R(
    M(0,0), M(0,1), M(0,2),
    M(1,0), M(1,1), M(1,2),
    M(2,0), M(2,1), M(2,2)
  );
  cv::Vec3d t(M(0,3), M(1,3), M(2,3));
  return aruco_extrinsic_calib_c3p::SE3d(R, t);
}

} // namespace

int main(int argc, char** argv) {
  const std::string calib_path = GetArg(argc, argv, "--calib", "");
  const std::string camA_csv   = GetArg(argc, argv, "--camA_csv", "");
  const std::string camB_csv   = GetArg(argc, argv, "--camB_csv", "");
  const int camA_idx           = GetArgI(argc, argv, "--camA_idx", -1);
  const int camB_idx           = GetArgI(argc, argv, "--camB_idx", -1);

  const double sync_tol_s      = GetArgD(argc, argv, "--sync_tol", 0.01);
  int marker_id                = GetArgI(argc, argv, "--marker_id", -1);

  if (camA_csv.empty() || camB_csv.empty()) {
    std::cerr << "Usage:\n"
              << "  verify_extrinsics --camA_csv <csvA> --camB_csv <csvB> [--sync_tol 0.01] [--marker_id -1]\\\n"
              << "    [--calib camchain.yaml --camA_idx <i> --camB_idx <j>]  (for comparison to calib)\n";
    return 2;
  }

  bool have_ref = false;
  aruco_extrinsic_calib_c3p::SE3d T_ref;
  if (!calib_path.empty() && camA_idx >= 0 && camB_idx >= 0) {
    try {
      auto chain = aruco_extrinsic_calib_c3p::LoadCamChainYaml(calib_path);
      bool ok = false;
      cv::Matx44d T = aruco_extrinsic_calib_c3p::GetT_cam_i_cam_j(chain, camB_idx, camA_idx, &ok); // camB <- camA
      if (ok) {
        T_ref = Mat44ToSE3(T);
        have_ref = true;
      } else {
        std::cerr << "Warning: calibration chain does not provide T_cam" << camB_idx << "_cam" << camA_idx << "\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "Warning: failed to load calib: " << e.what() << "\n";
    }
  }

  auto obsA = LoadObsByTimeMarker(camA_csv);
  auto obsB = LoadObsByTimeMarker(camB_csv);
  auto tA = SortedKeys(obsA);
  auto tB = SortedKeys(obsB);

  uint64_t tol_ns = (uint64_t)std::llround(sync_tol_s * 1e9);
  auto pairs = MatchTimes(tA, tB, tol_ns);
  std::cout << "Matched timestamps: " << pairs.size() << " within tol=" << sync_tol_s << " sec\n";

  std::vector<aruco_extrinsic_calib_c3p::SE3d> Xs; // X = T_camB<-camA
  std::unordered_map<int, std::vector<std::pair<aruco_extrinsic_calib_c3p::SE3d, aruco_extrinsic_calib_c3p::SE3d>>> per_id;

  for (const auto& pr : pairs) {
    uint64_t ta = pr.first;
    uint64_t tb = pr.second;
    const auto& ma = obsA.at(ta);
    const auto& mb = obsB.at(tb);

    for (const auto& kv : ma) {
      int id = kv.first;
      auto it = mb.find(id);
      if (it == mb.end()) continue;

      const auto& T_cA_m = kv.second;
      const auto& T_cB_m = it->second;

      // X = T_cB_cA = T_cB_m * inv(T_cA_m)
      auto X = aruco_extrinsic_calib_c3p::Compose(T_cB_m, aruco_extrinsic_calib_c3p::Inverse(T_cA_m));
      Xs.push_back(X);
      per_id[id].push_back({T_cA_m, T_cB_m});
    }
  }

  std::cout << "Common-marker extrinsic samples: " << Xs.size() << "\n";
  if (!Xs.empty()) {
    auto X_avg = aruco_extrinsic_calib_c3p::AverageSE3(Xs);
    auto Xm = aruco_extrinsic_calib_c3p::ToMat4(X_avg);

    std::cout << "Average T_camB<-camA:\n"
              << std::setprecision(12)
              << Xm(0,0) << " " << Xm(0,1) << " " << Xm(0,2) << " " << Xm(0,3) << "\n"
              << Xm(1,0) << " " << Xm(1,1) << " " << Xm(1,2) << " " << Xm(1,3) << "\n"
              << Xm(2,0) << " " << Xm(2,1) << " " << Xm(2,2) << " " << Xm(2,3) << "\n"
              << "0 0 0 1\n";

    if (have_ref) {
      std::cout << "Avg vs calib: rot_err_deg="
                << aruco_extrinsic_calib_c3p::RotationErrorDeg(X_avg.R, T_ref.R)
                << " trans_err_m="
                << aruco_extrinsic_calib_c3p::TranslationError(X_avg.t, T_ref.t)
                << "\n";
    }
  }

  // pick marker for hand-eye
  if (marker_id < 0) {
    size_t best_n = 0;
    int best_id = -1;
    for (const auto& kv : per_id) {
      if (kv.second.size() > best_n) { best_n = kv.second.size(); best_id = kv.first; }
    }
    marker_id = best_id;
  }

  auto it = per_id.find(marker_id);
  if (it == per_id.end() || it->second.size() < 3) {
    std::cerr << "Not enough synchronized samples for hand-eye on marker_id=" << marker_id << "\n";
    return 0;
  }

  const auto& seq = it->second;
  std::vector<aruco_extrinsic_calib_c3p::SE3d> A, B;
  for (size_t i = 0; i + 1 < seq.size(); ++i) {
    // motion of marker in camA / camB between consecutive synchronized samples
    auto Aij = aruco_extrinsic_calib_c3p::Compose(seq[i+1].first, aruco_extrinsic_calib_c3p::Inverse(seq[i].first));
    auto Bij = aruco_extrinsic_calib_c3p::Compose(seq[i+1].second, aruco_extrinsic_calib_c3p::Inverse(seq[i].second));

    // Bij * X = X * Aij  ->  A'=Bij, B'=Aij for AX=XB
    A.push_back(Bij);
    B.push_back(Aij);
  }

  auto X_handeye = aruco_extrinsic_calib_c3p::SolveAXXB_ParkMartin(A, B);
  auto Xhm = aruco_extrinsic_calib_c3p::ToMat4(X_handeye);

  std::cout << "\nHand-eye AX=XB (marker_id=" << marker_id << ", pairs=" << A.size() << ") estimated T_camB<-camA:\n"
            << std::setprecision(12)
            << Xhm(0,0) << " " << Xhm(0,1) << " " << Xhm(0,2) << " " << Xhm(0,3) << "\n"
            << Xhm(1,0) << " " << Xhm(1,1) << " " << Xhm(1,2) << " " << Xhm(1,3) << "\n"
            << Xhm(2,0) << " " << Xhm(2,1) << " " << Xhm(2,2) << " " << Xhm(2,3) << "\n"
            << "0 0 0 1\n";

  if (have_ref) {
    std::cout << "HandEye vs calib: rot_err_deg="
              << aruco_extrinsic_calib_c3p::RotationErrorDeg(X_handeye.R, T_ref.R)
              << " trans_err_m="
              << aruco_extrinsic_calib_c3p::TranslationError(X_handeye.t, T_ref.t)
              << "\n";
  }

  return 0;
}
