#include "aruco_extrinsic_calib_c3p/camchain_yaml.h"
#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include "aruco_extrinsic_calib_c3p/handeye_axxb.h"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <boost/foreach.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace {

enum class ImageMode {
  AUTO = 0,
  COMPRESSED = 1,
  RAW = 2
};

struct MarkerObs {
  int id = -1;
  std::array<cv::Point2f,4> corners;
  aruco_extrinsic_calib_c3p::SE3d T_c_m; // camera <- marker
  cv::Vec3d rvec, tvec;
};

struct FrameObs {
  double t_sec = 0.0;
  uint64_t t_ns = 0;
  int frame_idx = 0;
  std::vector<MarkerObs> markers;
  std::unordered_map<int, aruco_extrinsic_calib_c3p::SE3d> by_id;
};

static std::string NormalizeTopic(const std::string& t) {
  if (t.empty()) return t;
  if (t[0] == '/') return t;
  return "/" + t;
}

static bool TopicEqualsNormalized(const std::string& a_norm, const std::string& b_norm) {
  return a_norm == b_norm;
}

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
static bool HasFlag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key) return true;
  }
  return false;
}

static std::vector<std::string> SplitComma(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

static std::string EnsureDirSlash(std::string s) {
  if (s.empty()) return "./";
  if (s.back() != '/') s.push_back('/');
  return s;
}

static bool MkdirP(const std::string& out_dir) {
  std::ostringstream cmd;
  cmd << "mkdir -p " << out_dir;
  int ret = std::system(cmd.str().c_str());
  return (ret == 0);
}

static double EstimateFpsFromStamps(const std::vector<ros::Time>& stamps, double fallback) {
  if (stamps.size() < 2) return fallback;
  double t0 = stamps.front().toSec();
  double t1 = stamps.back().toSec();
  double dt = t1 - t0;
  if (dt <= 1e-6) return fallback;
  return (double)(stamps.size() - 1) / dt;
}

static bool TryOpenVideo(cv::VideoWriter& w,
                         const std::string& path_mp4,
                         int width, int height,
                         double fps) {
  cv::Size sz(width, height);

  // 1) avc1
  {
    int fourcc = cv::VideoWriter::fourcc('a','v','c','1');
    if (w.open(path_mp4, fourcc, fps, sz, true)) return true;
  }
  // 2) H264
  {
    int fourcc = cv::VideoWriter::fourcc('H','2','6','4');
    if (w.open(path_mp4, fourcc, fps, sz, true)) return true;
  }
  // 3) mp4v
  {
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    if (w.open(path_mp4, fourcc, fps, sz, true)) return true;
  }
  return false;
}

static void WriteCsvHeader(std::ofstream& os) {
  os << "t_ns,frame_idx,marker_id,"
        "c0x,c0y,c1x,c1y,c2x,c2y,c3x,c3y,"
        "rvec_x,rvec_y,rvec_z,"
        "tvec_x,tvec_y,tvec_z\n";
}

static void AppendMarkerCsv(std::ofstream& os, const FrameObs& f, const MarkerObs& m) {
  os << f.t_ns << "," << f.frame_idx << "," << m.id << ",";
  os << std::fixed << std::setprecision(3)
     << m.corners[0].x << "," << m.corners[0].y << ","
     << m.corners[1].x << "," << m.corners[1].y << ","
     << m.corners[2].x << "," << m.corners[2].y << ","
     << m.corners[3].x << "," << m.corners[3].y << ",";
  os << std::setprecision(9)
     << m.rvec[0] << "," << m.rvec[1] << "," << m.rvec[2] << ","
     << m.tvec[0] << "," << m.tvec[1] << "," << m.tvec[2] << "\n";
}

static bool ExtractStampFromMsg(const rosbag::MessageInstance& m, ImageMode mode, ros::Time& stamp_out) {
  if (mode == ImageMode::COMPRESSED || mode == ImageMode::AUTO) {
    auto msg = m.instantiate<sensor_msgs::CompressedImage>();
    if (msg) {
      stamp_out = msg->header.stamp;
      return true;
    }
  }
  if (mode == ImageMode::RAW || mode == ImageMode::AUTO) {
    auto msg = m.instantiate<sensor_msgs::Image>();
    if (msg) {
      stamp_out = msg->header.stamp;
      return true;
    }
  }
  return false;
}

static cv::Mat DecodeCompressed(const sensor_msgs::CompressedImage& msg) {
  cv::Mat raw(1, (int)msg.data.size(), CV_8UC1, const_cast<uint8_t*>(msg.data.data()));
  cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
  return img;
}

static bool DecodeRawImage(const sensor_msgs::Image& msg, cv::Mat& out_bgr) {
  // Minimal support for common encodings.
  const int w = (int)msg.width;
  const int h = (int)msg.height;
  if (w <= 0 || h <= 0) return false;
  if (msg.data.empty()) return false;

  // Note: msg.step is bytes per row.
  const std::string& enc = msg.encoding;

  if (enc == "bgr8") {
    cv::Mat img(h, w, CV_8UC3, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step);
    out_bgr = img.clone();
    return true;
  }
  if (enc == "rgb8") {
    cv::Mat img(h, w, CV_8UC3, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step);
    cv::cvtColor(img, out_bgr, cv::COLOR_RGB2BGR);
    return true;
  }
  if (enc == "mono8") {
    cv::Mat img(h, w, CV_8UC1, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step);
    cv::cvtColor(img, out_bgr, cv::COLOR_GRAY2BGR);
    return true;
  }
  if (enc == "bgra8") {
    cv::Mat img(h, w, CV_8UC4, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step);
    cv::cvtColor(img, out_bgr, cv::COLOR_BGRA2BGR);
    return true;
  }
  if (enc == "rgba8") {
    cv::Mat img(h, w, CV_8UC4, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step);
    cv::cvtColor(img, out_bgr, cv::COLOR_RGBA2BGR);
    return true;
  }

  // Unsupported encoding.
  return false;
}

static bool DecodeImageFromMsg(const rosbag::MessageInstance& m, ImageMode mode, cv::Mat& out_bgr, ros::Time& stamp_out) {
  if (mode == ImageMode::COMPRESSED || mode == ImageMode::AUTO) {
    auto msg = m.instantiate<sensor_msgs::CompressedImage>();
    if (msg) {
      stamp_out = msg->header.stamp;
      out_bgr = DecodeCompressed(*msg);
      return !out_bgr.empty();
    }
  }
  if (mode == ImageMode::RAW || mode == ImageMode::AUTO) {
    auto msg = m.instantiate<sensor_msgs::Image>();
    if (msg) {
      stamp_out = msg->header.stamp;
      return DecodeRawImage(*msg, out_bgr);
    }
  }
  return false;
}

static void MaybeSwapRB(cv::Mat& img, bool swap_rb) {
  if (!swap_rb) return;
  if (img.empty()) return;
  const int ch = img.channels();
  if (ch == 3) {
    cv::Mat tmp;
    cv::cvtColor(img, tmp, cv::COLOR_BGR2RGB);
    img = tmp;
  } else if (ch == 4) {
    cv::Mat tmp;
    cv::cvtColor(img, tmp, cv::COLOR_BGRA2RGBA);
    img = tmp;
  }
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

// Return rotation error in degrees and translation error in meters, given T_est and T_ref.
static void PrintSE3Error(std::ostream& os,
                          const aruco_extrinsic_calib_c3p::SE3d& T_est,
                          const aruco_extrinsic_calib_c3p::SE3d& T_ref,
                          const std::string& label) {
  double rerr = aruco_extrinsic_calib_c3p::RotationErrorDeg(T_est.R, T_ref.R);
  double terr = aruco_extrinsic_calib_c3p::TranslationError(T_est.t, T_ref.t);
  os << label << ": rotation_err_deg=" << rerr << " translation_err_m=" << terr << "\n";
}

static std::vector<std::pair<size_t,size_t>> MatchFramesByTime(
    const std::vector<FrameObs>& a,
    const std::vector<FrameObs>& b,
    double tol_sec) {
  std::vector<std::pair<size_t,size_t>> pairs;
  size_t j = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    double t = a[i].t_sec;
    while (j + 1 < b.size() && b[j+1].t_sec < t) j++;
    size_t best = (size_t)-1;
    double best_dt = 1e9;

    for (int dj = -1; dj <= 1; ++dj) {
      long jj = (long)j + dj;
      if (jj < 0 || (size_t)jj >= b.size()) continue;
      double dt = std::abs(b[(size_t)jj].t_sec - t);
      if (dt < best_dt) { best_dt = dt; best = (size_t)jj; }
    }
    if (best != (size_t)-1 && best_dt <= tol_sec) {
      pairs.emplace_back(i, best);
    }
  }
  return pairs;
}

static ImageMode ParseImageMode(const std::string& s) {
  if (s == "compressed") return ImageMode::COMPRESSED;
  if (s == "raw") return ImageMode::RAW;
  return ImageMode::AUTO;
}

} // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "extract_aruco_from_bag");

  const std::string bag_path   = GetArg(argc, argv, "--bag", "");
  const std::string calib_path = GetArg(argc, argv, "--calib", "calib-camchain.yaml");
  const std::string out_dir    = EnsureDirSlash(GetArg(argc, argv, "--out_dir", "out"));
  const double tag_size_m      = GetArgD(argc, argv, "--tag_size", 0.25);
  const std::string dict_name  = GetArg(argc, argv, "--dict", "DICT_6X6_250");
  const double sync_tol        = GetArgD(argc, argv, "--sync_tol", 0.01); // seconds
  const double fps_fallback    = GetArgD(argc, argv, "--fps", 0.0); // 0 means auto; fallback=10
  const bool draw_axes         = !HasFlag(argc, argv, "--no_axes");
  const bool swap_rb          = HasFlag(argc, argv, "--swap_rb");

  const int num_cams_arg       = GetArgI(argc, argv, "--num_cams", -1);
  const std::string topics_arg = GetArg(argc, argv, "--topics", "");
  const std::string image_mode_arg = GetArg(argc, argv, "--image_mode", "auto");
  const ImageMode image_mode = ParseImageMode(image_mode_arg);

  if (bag_path.empty()) {
    std::cerr << "Usage:\n"
              << "  extract_aruco_from_bag --bag <bag> --calib <camchain.yaml> [--out_dir out]\\\n"
              << "    [--num_cams N] [--topics t0,t1,...] [--image_mode auto|compressed|raw]\\\n"
              << "    [--tag_size 0.25] [--dict DICT_6X6_250] [--sync_tol 0.01] [--fps 0] [--swap_rb] [--no_axes]\n";
    return 2;
  }
  if (!MkdirP(out_dir)) {
    std::cerr << "Failed to create output dir: " << out_dir << "\n";
    return 3;
  }

  aruco_extrinsic_calib_c3p::CamChain chain = aruco_extrinsic_calib_c3p::LoadCamChainYaml(calib_path);
  if (chain.cams.empty()) {
    std::cerr << "No cameras found in calib file.\n";
    return 3;
  }

  int num_cams = chain.size();
  if (num_cams_arg > 0) num_cams = std::min(num_cams, num_cams_arg);

  // Truncate to requested number of cameras.
  chain.cams.resize(num_cams);
  if (chain.has_T_cam0.size() >= (size_t)num_cams) chain.has_T_cam0.resize(num_cams);
  if (chain.T_cam_cam0.size() >= (size_t)num_cams) chain.T_cam_cam0.resize(num_cams);

  // Override topics if user provided.
  auto topic_list = SplitComma(topics_arg);
  if (!topic_list.empty()) {
    if ((int)topic_list.size() < num_cams) {
      std::cerr << "--topics provided but has fewer entries than --num_cams (" << topic_list.size() << " < " << num_cams << ")\n";
      return 2;
    }
    for (int i = 0; i < num_cams; ++i) {
      chain.cams[i].rostopic = topic_list[i];
    }
  }

  // Ensure all topics are present.
  for (int i = 0; i < num_cams; ++i) {
    if (chain.cams[i].rostopic.empty()) {
      std::cerr << "Camera " << chain.cams[i].name << " has empty rostopic. Provide it in YAML or via --topics.\n";
      return 2;
    }
  }

  // Choose dictionary
  int dict_id = cv::aruco::DICT_6X6_250;
  if (dict_name == "DICT_6X6_100") dict_id = cv::aruco::DICT_6X6_100;
  else if (dict_name == "DICT_6X6_250") dict_id = cv::aruco::DICT_6X6_250;
  else if (dict_name == "DICT_6X6_1000") dict_id = cv::aruco::DICT_6X6_1000;
  else {
    std::cerr << "Unknown --dict " << dict_name << ", using DICT_6X6_250\n";
  }

  cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(dict_id);
  cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

  // --- Pass 1: compute FPS from timestamps only ---
  std::vector<std::vector<ros::Time>> stamps((size_t)num_cams);

  std::vector<std::string> topics;
  topics.reserve(num_cams);
  std::unordered_map<std::string,int> topic_to_idx;
  topic_to_idx.reserve((size_t)num_cams * 2);

  for (int i = 0; i < num_cams; ++i) {
    std::string tn = NormalizeTopic(chain.cams[i].rostopic);
    topics.push_back(tn);
    topic_to_idx[tn] = i;
  }

  {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    BOOST_FOREACH (rosbag::MessageInstance const m, view) {
      const std::string topic_norm = NormalizeTopic(m.getTopic());
      auto it = topic_to_idx.find(topic_norm);
      if (it == topic_to_idx.end()) continue;

      ros::Time stamp;
      if (!ExtractStampFromMsg(m, image_mode, stamp)) continue;
      stamps[(size_t)it->second].push_back(stamp);
    }
    bag.close();
  }

  std::vector<double> fps((size_t)num_cams, 10.0);
  for (int i = 0; i < num_cams; ++i) {
    double fallback = (fps_fallback > 0.0 ? fps_fallback : 10.0);
    fps[(size_t)i] = EstimateFpsFromStamps(stamps[(size_t)i], fallback);
    std::cout << "Estimated FPS: " << chain.cams[i].name << "=" << fps[(size_t)i] << "\n";
  }

  // --- Pass 2: decode + detect + write ---
  std::vector<cv::VideoWriter> writers((size_t)num_cams);
  std::vector<std::ofstream> csvs((size_t)num_cams);

  for (int i = 0; i < num_cams; ++i) {
    const auto& cam = chain.cams[i];

    std::string mp4 = out_dir + cam.name + "_annotated.mp4";
    std::string avi = out_dir + cam.name + "_annotated.avi";

    bool ok = TryOpenVideo(writers[(size_t)i], mp4, cam.width, cam.height, fps[(size_t)i]);
    if (!ok) {
      std::cerr << "MP4 writer open failed for " << cam.name << ", falling back to AVI: " << avi << "\n";
      int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
      if (!writers[(size_t)i].open(avi, fourcc, fps[(size_t)i], cv::Size(cam.width, cam.height), true)) {
        std::cerr << "AVI writer open failed for " << cam.name << " as well.\n";
        return 4;
      }
    }

    csvs[(size_t)i].open(out_dir + cam.name + "_aruco_poses.csv");
    WriteCsvHeader(csvs[(size_t)i]);
  }

  std::vector<std::vector<FrameObs>> frames((size_t)num_cams);
  std::vector<int> frame_idx((size_t)num_cams, 0);

  {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    BOOST_FOREACH (rosbag::MessageInstance const m, view) {
      const std::string topic_norm = NormalizeTopic(m.getTopic());
      auto it = topic_to_idx.find(topic_norm);
      if (it == topic_to_idx.end()) continue;
      int ci = it->second;
      const auto& cam = chain.cams[ci];

      ros::Time stamp;
      cv::Mat img;
      if (!DecodeImageFromMsg(m, image_mode, img, stamp)) continue;
      MaybeSwapRB(img, swap_rb);
      if (img.empty()) continue;

      // Detect markers
      cv::Mat cameraMatrix = cv::Mat(cam.K);
      cv::Mat distCoeffs = (cv::Mat_<double>(1,4) << cam.D[0], cam.D[1], cam.D[2], cam.D[3]);

      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners, rejected;
      cv::aruco::detectMarkers(img, dict, corners, ids, params, rejected);

      std::vector<cv::Vec3d> rvecs, tvecs;
      if (!ids.empty()) {
        cv::aruco::estimatePoseSingleMarkers(corners, tag_size_m, cameraMatrix, distCoeffs, rvecs, tvecs);
        cv::aruco::drawDetectedMarkers(img, corners, ids);
      }

      FrameObs f;
      f.t_sec = stamp.toSec();
      f.t_ns = (uint64_t)stamp.toNSec();
      f.frame_idx = frame_idx[(size_t)ci];

      for (size_t k = 0; k < ids.size(); ++k) {
        MarkerObs mo;
        mo.id = ids[k];
        for (int c = 0; c < 4; ++c) mo.corners[c] = corners[k][c];
        mo.rvec = rvecs[k];
        mo.tvec = tvecs[k];
        mo.T_c_m = aruco_extrinsic_calib_c3p::FromRvecTvec(mo.rvec, mo.tvec);
        f.by_id[mo.id] = mo.T_c_m;
        f.markers.push_back(mo);

        if (draw_axes) {
          cv::aruco::drawAxis(img, cameraMatrix, distCoeffs, mo.rvec, mo.tvec, tag_size_m * 0.5);
        }
      }

      // Overlay timestamp + debug
      {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << f.t_sec;
        cv::putText(img, "t=" + oss.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
        cv::putText(img, cam.name + " markers=" + std::to_string(ids.size()),
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
      }

      // record csv
      for (const auto& mo : f.markers) AppendMarkerCsv(csvs[(size_t)ci], f, mo);

      writers[(size_t)ci].write(img);
      frames[(size_t)ci].push_back(f);
      frame_idx[(size_t)ci] += 1;
    }

    bag.close();
  }

  for (int i = 0; i < num_cams; ++i) {
    std::cout << "Processed frames: " << chain.cams[i].name << "=" << frames[(size_t)i].size() << "\n";
  }

  // --- Extrinsic verification vs cam0 ---
  std::ofstream summary(out_dir + "extrinsic_summary.txt");
  summary << "Extrinsic verification (reference: cam0)\n";
  summary << "num_cams=" << num_cams << " sync_tol=" << sync_tol << " sec\n\n";

  if (num_cams < 2) {
    summary << "Only one camera; skipping extrinsic verification.\n";
    std::cout << "Wrote outputs under: " << out_dir << "\n";
    return 0;
  }

  const auto& frames0 = frames[0];

  for (int k = 1; k < num_cams; ++k) {
    const auto& framesk = frames[(size_t)k];
    summary << "=== " << chain.cams[k].name << " vs cam0 ===\n";

    bool have_calib = false;
    aruco_extrinsic_calib_c3p::SE3d T_calib;
    if (chain.has_T_cam0.size() == (size_t)num_cams && chain.has_T_cam0[(size_t)k]) {
      T_calib = Mat44ToSE3(chain.T_cam_cam0[(size_t)k]); // camk <- cam0
      have_calib = true;
    } else {
      summary << "No chained calibration transform cam" << k << "<-cam0 available (missing T_cn_cnm1?)\n";
    }

    auto pairs = MatchFramesByTime(frames0, framesk, sync_tol);
    summary << "Synced frame pairs within tol: " << pairs.size() << "\n";
    std::cout << chain.cams[k].name << ": synced pairs with cam0: " << pairs.size() << "\n";

    std::string csv_path = out_dir + "extrinsic_common_markers_" + chain.cams[k].name + ".csv";
    std::ofstream csvX(csv_path);
    csvX << "t0_ns,tk_ns,marker_id,"
            "R00,R01,R02,t0,"
            "R10,R11,R12,t1,"
            "R20,R21,R22,t2,"
            "rot_err_deg,trans_err_m\n";

    std::vector<aruco_extrinsic_calib_c3p::SE3d> X_estimates;
    std::unordered_map<int, std::vector<std::pair<aruco_extrinsic_calib_c3p::SE3d, aruco_extrinsic_calib_c3p::SE3d>>> per_id_Tc0_Tck;

    for (const auto& p : pairs) {
      const FrameObs& f0 = frames0[p.first];
      const FrameObs& fk = framesk[p.second];

      // Intersect marker IDs
      for (const auto& kv : f0.by_id) {
        int id = kv.first;
        auto itmk = fk.by_id.find(id);
        if (itmk == fk.by_id.end()) continue;

        const auto& T_c0_m = kv.second;
        const auto& T_ck_m = itmk->second;

        // X = T_ck_c0 = T_ck_m * inv(T_c0_m)
        auto X = aruco_extrinsic_calib_c3p::Compose(T_ck_m, aruco_extrinsic_calib_c3p::Inverse(T_c0_m));
        X_estimates.push_back(X);

        double rerr = 0.0, terr = 0.0;
        if (have_calib) {
          rerr = aruco_extrinsic_calib_c3p::RotationErrorDeg(X.R, T_calib.R);
          terr = aruco_extrinsic_calib_c3p::TranslationError(X.t, T_calib.t);
        }

        csvX << f0.t_ns << "," << fk.t_ns << "," << id << ",";
        csvX << std::setprecision(12)
             << X.R(0,0) << "," << X.R(0,1) << "," << X.R(0,2) << "," << X.t[0] << ","
             << X.R(1,0) << "," << X.R(1,1) << "," << X.R(1,2) << "," << X.t[1] << ","
             << X.R(2,0) << "," << X.R(2,1) << "," << X.R(2,2) << "," << X.t[2] << ","
             << rerr << "," << terr << "\n";

        per_id_Tc0_Tck[id].push_back({T_c0_m, T_ck_m});
      }
    }

    summary << "Common-marker extrinsic samples: " << X_estimates.size() << "\n";

    if (!X_estimates.empty()) {
      auto X_avg = aruco_extrinsic_calib_c3p::AverageSE3(X_estimates);
      auto Xm = aruco_extrinsic_calib_c3p::ToMat4(X_avg);
      summary << "Average T_" << chain.cams[k].name << "_cam0 (from common markers):\n";
      summary << std::setprecision(12)
              << Xm(0,0) << " " << Xm(0,1) << " " << Xm(0,2) << " " << Xm(0,3) << "\n"
              << Xm(1,0) << " " << Xm(1,1) << " " << Xm(1,2) << " " << Xm(1,3) << "\n"
              << Xm(2,0) << " " << Xm(2,1) << " " << Xm(2,2) << " " << Xm(2,3) << "\n"
              << "0 0 0 1\n";
      if (have_calib) {
        PrintSE3Error(summary, X_avg, T_calib, "Avg vs calib(camk<-cam0)");
      }
    }

    // --- Hand-eye AX=XB verification ---
    int marker_id = GetArgI(argc, argv, "--marker_id", -1);
    if (marker_id < 0) {
      size_t best_n = 0;
      int best_id = -1;
      for (const auto& kv : per_id_Tc0_Tck) {
        if (kv.second.size() > best_n) { best_n = kv.second.size(); best_id = kv.first; }
      }
      marker_id = best_id;
    }

    auto itseq = per_id_Tc0_Tck.find(marker_id);
    if (marker_id >= 0 && itseq != per_id_Tc0_Tck.end() && itseq->second.size() >= 3) {
      const auto& seq = itseq->second;
      std::vector<aruco_extrinsic_calib_c3p::SE3d> A, B;
      A.reserve(seq.size()-1);
      B.reserve(seq.size()-1);

      for (size_t i = 0; i + 1 < seq.size(); ++i) {
        const auto& T_c0_i = seq[i].first;
        const auto& T_c0_j = seq[i+1].first;

        const auto& T_ck_i = seq[i].second;
        const auto& T_ck_j = seq[i+1].second;

        auto Aij = aruco_extrinsic_calib_c3p::Compose(T_c0_j, aruco_extrinsic_calib_c3p::Inverse(T_c0_i));
        auto Bij = aruco_extrinsic_calib_c3p::Compose(T_ck_j, aruco_extrinsic_calib_c3p::Inverse(T_ck_i));

        // Bij * X = X * Aij  ->  A'=Bij, B'=Aij for AX=XB
        A.push_back(Bij);
        B.push_back(Aij);
      }

      auto X_handeye = aruco_extrinsic_calib_c3p::SolveAXXB_ParkMartin(A, B);
      auto Xhm = aruco_extrinsic_calib_c3p::ToMat4(X_handeye);

      summary << "\nHand-eye AX=XB (marker_id=" << marker_id << ", motion pairs=" << A.size() << ")\n";
      summary << "Estimated T_" << chain.cams[k].name << "_cam0:\n";
      summary << std::setprecision(12)
              << Xhm(0,0) << " " << Xhm(0,1) << " " << Xhm(0,2) << " " << Xhm(0,3) << "\n"
              << Xhm(1,0) << " " << Xhm(1,1) << " " << Xhm(1,2) << " " << Xhm(1,3) << "\n"
              << Xhm(2,0) << " " << Xhm(2,1) << " " << Xhm(2,2) << " " << Xhm(2,3) << "\n"
              << "0 0 0 1\n";

      if (have_calib) {
        PrintSE3Error(summary, X_handeye, T_calib, "HandEye vs calib(camk<-cam0)");
      }
    } else {
      summary << "\nHand-eye: not enough synchronized samples for marker_id=" << marker_id << "\n";
    }

    summary << "\n";
  }

  std::cout << "Wrote outputs under: " << out_dir << "\n";
  return 0;
}
