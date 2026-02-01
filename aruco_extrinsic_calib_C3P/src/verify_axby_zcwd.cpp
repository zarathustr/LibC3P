#include "aruco_extrinsic_calib_c3p/camchain_yaml.h"
#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include "aruco_extrinsic_calib_c3p/handeye_axxb.h"
#include "aruco_extrinsic_calib_c3p/xyzw_yaml.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
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

static bool FindClosestWithinTol(const std::vector<uint64_t>& keys, uint64_t t, uint64_t tol, uint64_t& best_out) {
  if (keys.empty()) return false;
  // binary search
  auto it = std::lower_bound(keys.begin(), keys.end(), t);
  uint64_t best = 0;
  uint64_t best_dt = (uint64_t)-1;

  auto try_it = [&](decltype(it) jt){
    if (jt == keys.end()) return;
    uint64_t tj = *jt;
    uint64_t dt = (tj > t) ? (tj - t) : (t - tj);
    if (dt < best_dt) { best_dt = dt; best = tj; }
  };

  if (it != keys.end()) try_it(it);
  if (it != keys.begin()) try_it(std::prev(it));

  if (best_dt <= tol) { best_out = best; return true; }
  return false;
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

static double Norm3(const cv::Vec3d& v) {
  return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static void PrintMat44(const cv::Matx44d& T) {
  std::cout << std::setprecision(12)
            << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << "\n"
            << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << "\n"
            << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << "\n"
            << "0 0 0 1\n";
}

} // namespace

int main(int argc, char** argv) {
  const std::string calib_path = GetArg(argc, argv, "--calib", "");
  const std::string csv_dir    = GetArg(argc, argv, "--csv_dir", "");

  const std::string out_xyzw_yaml = GetArg(argc, argv, "--out_xyzw", "");

  const int cam0 = GetArgI(argc, argv, "--cam0", 0);
  const int cam1 = GetArgI(argc, argv, "--cam1", 1);
  const int cam2 = GetArgI(argc, argv, "--cam2", 2);
  const int cam3 = GetArgI(argc, argv, "--cam3", 3);

  const int marker0 = GetArgI(argc, argv, "--marker0", 0);
  const int marker1 = GetArgI(argc, argv, "--marker1", 1);
  const int marker2 = GetArgI(argc, argv, "--marker2", 2);
  const int marker3 = GetArgI(argc, argv, "--marker3", 3);

  const double sync_tol_s = GetArgD(argc, argv, "--sync_tol", 0.01);
  const double min_rot_deg = GetArgD(argc, argv, "--min_rot_deg", 0.5);
  const double min_trans_m = GetArgD(argc, argv, "--min_trans_m", 0.002);

  // Optional geometry check for Y/Z translation magnitudes.
  const double tag_size_m = GetArgD(argc, argv, "--tag_size", 0.25);
  const double marker_margin_m = GetArgD(argc, argv, "--marker_margin", 0.01);

  // Optional expected grid offsets for marker pairs used in Y (m0<-m2) and Z (m1<-m3).
  const int y_grid_dx = GetArgI(argc, argv, "--y_grid_dx", 0);
  const int y_grid_dy = GetArgI(argc, argv, "--y_grid_dy", 0);
  const int z_grid_dx = GetArgI(argc, argv, "--z_grid_dx", 0);
  const int z_grid_dy = GetArgI(argc, argv, "--z_grid_dy", 0);

  if (calib_path.empty() || csv_dir.empty()) {
    std::cerr << "Usage:\n"
              << "  verify_axby_zcwd --calib camchain.yaml --csv_dir out \\\n"
              << "    --cam0 0 --marker0 0 --cam1 1 --marker1 1 --cam2 2 --marker2 2 --cam3 3 --marker3 3 \\\n"
              << "    [--sync_tol 0.01] [--min_rot_deg 0.5] [--min_trans_m 0.002] \\\n"
              << "    [--tag_size 0.25 --marker_margin 0.01] \\\n"
              << "    [--y_grid_dx 0 --y_grid_dy 0 --z_grid_dx 0 --z_grid_dy 0] \\\n"
              << "    [--out_xyzw out/estimated_XYZW.yaml]\n";
    return 2;
  }

  auto chain = aruco_extrinsic_calib_c3p::LoadCamChainYaml(calib_path);

  auto csvPathForCam = [&](int cam_idx) -> std::string {
    std::ostringstream oss;
    oss << csv_dir;
    if (!csv_dir.empty() && csv_dir.back() != '/') oss << "/";
    oss << "cam" << cam_idx << "_aruco_poses.csv";
    return oss.str();
  };

  const std::string csv0 = csvPathForCam(cam0);
  const std::string csv1 = csvPathForCam(cam1);
  const std::string csv2 = csvPathForCam(cam2);
  const std::string csv3 = csvPathForCam(cam3);

  auto obs0 = LoadObsByTimeMarker(csv0);
  auto obs1 = LoadObsByTimeMarker(csv1);
  auto obs2 = LoadObsByTimeMarker(csv2);
  auto obs3 = LoadObsByTimeMarker(csv3);

  auto t0 = SortedKeys(obs0);
  auto t1 = SortedKeys(obs1);
  auto t2 = SortedKeys(obs2);
  auto t3 = SortedKeys(obs3);

  const uint64_t tol_ns = (uint64_t)std::llround(sync_tol_s * 1e9);

  // Build synchronized quadruples using cam0 as reference.
  std::vector<std::array<uint64_t,4>> samples;
  samples.reserve(t0.size());

  for (uint64_t t_ref : t0) {
    uint64_t t_1=0, t_2=0, t_3=0;
    if (!FindClosestWithinTol(t1, t_ref, tol_ns, t_1)) continue;
    if (!FindClosestWithinTol(t2, t_ref, tol_ns, t_2)) continue;
    if (!FindClosestWithinTol(t3, t_ref, tol_ns, t_3)) continue;

    // Require the needed marker IDs to be present.
    auto it0 = obs0.find(t_ref);
    auto it1 = obs1.find(t_1);
    auto it2 = obs2.find(t_2);
    auto it3 = obs3.find(t_3);
    if (it0 == obs0.end() || it1 == obs1.end() || it2 == obs2.end() || it3 == obs3.end()) continue;

    if (it0->second.find(marker0) == it0->second.end()) continue;
    if (it1->second.find(marker1) == it1->second.end()) continue;
    if (it2->second.find(marker2) == it2->second.end()) continue;
    if (it3->second.find(marker3) == it3->second.end()) continue;

    samples.push_back({t_ref, t_1, t_2, t_3});
  }

  std::cout << "Synchronized samples (all 4 cameras + required markers): " << samples.size() << "\n";
  if (samples.size() < 3) {
    std::cerr << "Need >= 3 samples.\n";
    return 1;
  }

  // Build pose sequences A_i,B_i,C_i,D_i for:
  //   A_i X B_i Y = Z C_i W D_i
  // with:
  //   X = T_cam1<-cam0
  //   W = T_cam3<-cam2
  //   Y = T_m0<-m2
  //   Z = T_m1<-m3
  //   A_i = T_m1<-cam1 = inv(T_cam1<-m1)
  //   B_i = T_cam0<-m0
  //   C_i = T_m3<-cam3 = inv(T_cam3<-m3)
  //   D_i = T_cam2<-m2
  std::vector<aruco_extrinsic_calib_c3p::SE3d> A_pose, B_pose, C_pose, D_pose;
  A_pose.reserve(samples.size());
  B_pose.reserve(samples.size());
  C_pose.reserve(samples.size());
  D_pose.reserve(samples.size());

  for (const auto& s : samples) {
    uint64_t ta = s[0];
    uint64_t tb = s[1];
    uint64_t tc = s[2];
    uint64_t td = s[3];

    const auto& m0 = obs0.at(ta).at(marker0); // T_cam0<-m0
    const auto& m1 = obs1.at(tb).at(marker1); // T_cam1<-m1
    const auto& m2 = obs2.at(tc).at(marker2); // T_cam2<-m2
    const auto& m3 = obs3.at(td).at(marker3); // T_cam3<-m3

    B_pose.push_back(m0);
    A_pose.push_back(aruco_extrinsic_calib_c3p::Inverse(m1));
    D_pose.push_back(m2);
    C_pose.push_back(aruco_extrinsic_calib_c3p::Inverse(m3));
  }

  auto build_rel_pairs = [&](const std::vector<aruco_extrinsic_calib_c3p::SE3d>& P,
                             const std::vector<aruco_extrinsic_calib_c3p::SE3d>& Q,
                             bool Q_order_prev_inv_curr, // when true: Q_rel = Q_prev * inv(Q_curr); else inv(prev)*curr
                             std::vector<aruco_extrinsic_calib_c3p::SE3d>& outA,
                             std::vector<aruco_extrinsic_calib_c3p::SE3d>& outB) {
    outA.clear();
    outB.clear();
    size_t skipped = 0;
    for (size_t i = 1; i < P.size(); ++i) {
      auto Arel = aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Inverse(P[i-1]), P[i]);
      aruco_extrinsic_calib_c3p::SE3d Brel;
      if (Q_order_prev_inv_curr) {
        Brel = aruco_extrinsic_calib_c3p::Compose(Q[i-1], aruco_extrinsic_calib_c3p::Inverse(Q[i]));
      } else {
        Brel = aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Inverse(Q[i-1]), Q[i]);
      }

      const double rotA = aruco_extrinsic_calib_c3p::RotationAngleDeg(Arel.R);
      const double rotB = aruco_extrinsic_calib_c3p::RotationAngleDeg(Brel.R);
      const double transA = Norm3(Arel.t);
      const double transB = Norm3(Brel.t);

      if (std::max(rotA, rotB) < min_rot_deg && std::max(transA, transB) < min_trans_m) {
        skipped++;
        continue;
      }

      outA.push_back(Arel);
      outB.push_back(Brel);
    }
    std::cout << "Built " << outA.size() << " motion pairs (skipped small: " << skipped << ")\n";
  };

  // 1) Solve X from: (A_j^-1 A_i) X = X (B_j B_i^-1) -> consecutive => B_rel = B_prev * inv(B_curr)
  std::cout << "\nSolving X (cam" << cam1 << "<-cam" << cam0 << ") ...\n";
  std::vector<aruco_extrinsic_calib_c3p::SE3d> ArelX, BrelX;
  build_rel_pairs(A_pose, B_pose, /*Q_order_prev_inv_curr=*/true, ArelX, BrelX);
  if (ArelX.size() < 2) { std::cerr << "Not enough motion pairs for X.\n"; return 1; }
  auto X_est = aruco_extrinsic_calib_c3p::SolveAXXB_ParkMartin(ArelX, BrelX);

  // 2) Solve W from: (C_j^-1 C_i) W = W (D_j D_i^-1) -> consecutive => D_rel = D_prev * inv(D_curr)
  std::cout << "\nSolving W (cam" << cam3 << "<-cam" << cam2 << ") ...\n";
  std::vector<aruco_extrinsic_calib_c3p::SE3d> CrelW, DrelW;
  build_rel_pairs(C_pose, D_pose, /*Q_order_prev_inv_curr=*/true, CrelW, DrelW);
  if (CrelW.size() < 2) { std::cerr << "Not enough motion pairs for W.\n"; return 1; }
  auto W_est = aruco_extrinsic_calib_c3p::SolveAXXB_ParkMartin(CrelW, DrelW);

  // Compose helpers
  auto Mul = [&](const aruco_extrinsic_calib_c3p::SE3d& A, const aruco_extrinsic_calib_c3p::SE3d& B) {
    return aruco_extrinsic_calib_c3p::Compose(A, B);
  };

  // Build P_i = A_i X B_i and Q_i = C_i W D_i
  std::vector<aruco_extrinsic_calib_c3p::SE3d> P, Q;
  P.reserve(samples.size());
  Q.reserve(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    P.push_back(Mul(Mul(A_pose[i], X_est), B_pose[i]));
    Q.push_back(Mul(Mul(C_pose[i], W_est), D_pose[i]));
  }

  // 3) Solve Y from: (P_j^-1 P_i) Y = Y (Q_j^-1 Q_i) -> consecutive => Q_rel = inv(prev)*curr
  std::cout << "\nSolving Y (m" << marker0 << "<-m" << marker2 << ") ...\n";
  std::vector<aruco_extrinsic_calib_c3p::SE3d> PrelY, QrelY;
  build_rel_pairs(P, Q, /*Q_order_prev_inv_curr=*/false, PrelY, QrelY);
  if (PrelY.size() < 2) { std::cerr << "Not enough motion pairs for Y.\n"; return 1; }
  auto Y_est = aruco_extrinsic_calib_c3p::SolveAXXB_ParkMartin(PrelY, QrelY);

  // 4) Recover Z_i = P_i Y Q_i^-1, then average.
  std::cout << "\nRecovering Z (m" << marker1 << "<-m" << marker3 << ") ...\n";
  std::vector<aruco_extrinsic_calib_c3p::SE3d> Zs;
  Zs.reserve(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    auto Zi = Mul(Mul(P[i], Y_est), aruco_extrinsic_calib_c3p::Inverse(Q[i]));
    Zs.push_back(Zi);
  }
  auto Z_est = aruco_extrinsic_calib_c3p::AverageSE3(Zs);

  // Print results.
  std::cout << "\n=== Estimated transforms ===\n";
  std::cout << "X_est (cam" << cam1 << "<-cam" << cam0 << "):\n";
  PrintMat44(aruco_extrinsic_calib_c3p::ToMat4(X_est));

  std::cout << "\nW_est (cam" << cam3 << "<-cam" << cam2 << "):\n";
  PrintMat44(aruco_extrinsic_calib_c3p::ToMat4(W_est));

  std::cout << "\nY_est (m" << marker0 << "<-m" << marker2 << "):\n";
  PrintMat44(aruco_extrinsic_calib_c3p::ToMat4(Y_est));

  std::cout << "\nZ_est (m" << marker1 << "<-m" << marker3 << "):\n";
  PrintMat44(aruco_extrinsic_calib_c3p::ToMat4(Z_est));

  // Compare camera-camera parts to calib.
  bool okX=false, okW=false;
  cv::Matx44d X_gt = aruco_extrinsic_calib_c3p::GetT_cam_i_cam_j(chain, cam1, cam0, &okX);
  cv::Matx44d W_gt = aruco_extrinsic_calib_c3p::GetT_cam_i_cam_j(chain, cam3, cam2, &okW);

  if (okX) {
    auto Xref = Mat44ToSE3(X_gt);
    std::cout << "\nX vs calib: rot_err_deg="
              << aruco_extrinsic_calib_c3p::RotationErrorDeg(X_est.R, Xref.R)
              << " trans_err_m=" << aruco_extrinsic_calib_c3p::TranslationError(X_est.t, Xref.t) << "\n";
  } else {
    std::cout << "\nX vs calib: unavailable (chain incomplete)\n";
  }
  if (okW) {
    auto Wref = Mat44ToSE3(W_gt);
    std::cout << "W vs calib: rot_err_deg="
              << aruco_extrinsic_calib_c3p::RotationErrorDeg(W_est.R, Wref.R)
              << " trans_err_m=" << aruco_extrinsic_calib_c3p::TranslationError(W_est.t, Wref.t) << "\n";
  } else {
    std::cout << "W vs calib: unavailable (chain incomplete)\n";
  }

  // Residual statistics for the full equation.
  double rms_rot = 0.0, rms_trans = 0.0;
  double max_rot = 0.0, max_trans = 0.0;
  for (size_t i = 0; i < samples.size(); ++i) {
    auto left  = aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Compose(A_pose[i], X_est), B_pose[i]), Y_est);
    auto right = aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Compose(Z_est, C_pose[i]), W_est), D_pose[i]);
    auto err = aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Inverse(left), right);

    double r = aruco_extrinsic_calib_c3p::RotationAngleDeg(err.R);
    double t = Norm3(err.t);
    rms_rot += r*r;
    rms_trans += t*t;
    max_rot = std::max(max_rot, r);
    max_trans = std::max(max_trans, t);
  }
  rms_rot = std::sqrt(rms_rot / (double)samples.size());
  rms_trans = std::sqrt(rms_trans / (double)samples.size());

  std::cout << "\nResiduals of AXBY=ZCWD over " << samples.size() << " samples:\n"
            << "  RMS rot(deg)=" << rms_rot << "  RMS trans(m)=" << rms_trans << "\n"
            << "  Max rot(deg)=" << max_rot << "  Max trans(m)=" << max_trans << "\n";

  // Optional translation magnitude checks for Y and Z.
  const double step = tag_size_m + marker_margin_m;
  auto check_translation = [&](const char* name,
                               const aruco_extrinsic_calib_c3p::SE3d& T,
                               int dx, int dy) {
    const double dist_xy = std::sqrt(T.t[0]*T.t[0] + T.t[1]*T.t[1]);
    std::cout << "\n" << name << " translation |xy| = " << dist_xy << " m, |z|=" << std::abs(T.t[2]) << " m\n";
    std::cout << name << " rotation angle from identity: " << aruco_extrinsic_calib_c3p::RotationAngleDeg(T.R) << " deg\n";
    std::cout << "Assumed step = tag_size + margin = " << tag_size_m << " + " << marker_margin_m << " = " << step << " m\n";
    if (dx != 0 || dy != 0) {
      const double exp = step * std::sqrt((double)(dx*dx + dy*dy));
      std::cout << "Expected distance from (dx,dy)=(" << dx << "," << dy << "): " << exp
                << " m; error=" << std::abs(dist_xy - exp) << " m\n";
    }
  };

  check_translation("Y", Y_est, y_grid_dx, y_grid_dy);
  check_translation("Z", Z_est, z_grid_dx, z_grid_dy);

  if (!out_xyzw_yaml.empty()) {
    try {
      aruco_extrinsic_calib_c3p::XYZWEstimate est;
      est.cam0 = cam0;
      est.cam1 = cam1;
      est.cam2 = cam2;
      est.cam3 = cam3;
      est.marker0 = marker0;
      est.marker1 = marker1;
      est.marker2 = marker2;
      est.marker3 = marker3;
      est.X_c1_c0 = aruco_extrinsic_calib_c3p::ToMat4(X_est);
      est.W_c3_c2 = aruco_extrinsic_calib_c3p::ToMat4(W_est);
      est.Y_m0_m2 = aruco_extrinsic_calib_c3p::ToMat4(Y_est);
      est.Z_m1_m3 = aruco_extrinsic_calib_c3p::ToMat4(Z_est);
      est.has_X = est.has_Y = est.has_Z = est.has_W = true;
      aruco_extrinsic_calib_c3p::SaveXYZWYaml(out_xyzw_yaml, est);
      std::cout << "\nWrote XYZW yaml: " << out_xyzw_yaml << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to write --out_xyzw: " << e.what() << "\n";
    }
  }

  std::cout << "\nDone.\n";
  return 0;
}
