#include "aruco_extrinsic_calib_c3p/camchain_yaml.h"
#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include "aruco_extrinsic_calib_c3p/xyzw_yaml.h"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <boost/foreach.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

enum class ImageMode {
  AUTO = 0,
  COMPRESSED = 1,
  RAW = 2
};

static ImageMode ParseImageMode(const std::string& s) {
  if (s == "compressed") return ImageMode::COMPRESSED;
  if (s == "raw") return ImageMode::RAW;
  return ImageMode::AUTO;
}

static std::string NormalizeTopic(const std::string& t) {
  if (t.empty()) return t;
  if (t[0] == '/') return t;
  return "/" + t;
}

static bool TopicEqualsNormalized(const std::string& a_norm, const std::string& b_norm) {
  return a_norm == b_norm;
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

static std::string GetArg(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
  }
  return def;
}

static double GetArgD(int argc, char** argv, const std::string& key, double def) {
  const std::string s = GetArg(argc, argv, key, "");
  if (s.empty()) return def;
  return std::stod(s);
}

static int GetArgI(int argc, char** argv, const std::string& key, int def) {
  const std::string s = GetArg(argc, argv, key, "");
  if (s.empty()) return def;
  return std::stoi(s);
}

static bool HasFlag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key) return true;
  }
  return false;
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
  return cv::imdecode(raw, cv::IMREAD_COLOR);
}

static bool DecodeRawImage(const sensor_msgs::Image& msg, cv::Mat& out_bgr) {
  const int w = (int)msg.width;
  const int h = (int)msg.height;
  if (w <= 0 || h <= 0) return false;
  if (msg.data.empty()) return false;

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
                         int width,
                         int height,
                         double fps) {
  cv::Size sz(width, height);
  // 1) avc1
  {
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    if (w.open(path_mp4, fourcc, fps, sz, true)) return true;
  }
  // 2) H264
  {
    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    if (w.open(path_mp4, fourcc, fps, sz, true)) return true;
  }
  // 3) mp4v
  {
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    if (w.open(path_mp4, fourcc, fps, sz, true)) return true;
  }
  return false;
}

static void SE3ToRvecTvec(const aruco_extrinsic_calib_c3p::SE3d& T,
                          cv::Vec3d& rvec,
                          cv::Vec3d& tvec) {
  cv::Mat R_cv(3, 3, CV_64F);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      R_cv.at<double>(r, c) = T.R(r, c);
  cv::Rodrigues(R_cv, rvec);
  tvec = T.t;
}

static void DrawPoly(cv::Mat& img,
                     const std::vector<cv::Point2f>& pts,
                     const cv::Scalar& color,
                     int thickness) {
  if (pts.size() != 4) return;
  std::vector<cv::Point> pxi;
  pxi.reserve(4);
  for (const auto& p : pts) pxi.emplace_back((int)std::lround(p.x), (int)std::lround(p.y));
  const cv::Point* arr[1] = {pxi.data()};
  int npts[1] = {4};
  cv::polylines(img, arr, npts, 1, true, color, thickness, cv::LINE_AA);
  for (const auto& p : pxi) {
    cv::circle(img, p, 4, color, -1, cv::LINE_AA);
  }
}

static double MeanCornerError(const std::array<cv::Point2f, 4>& a,
                              const std::vector<cv::Point2f>& b) {
  if (b.size() != 4) return 0.0;
  double sum = 0.0;
  for (int i = 0; i < 4; ++i) {
    double dx = a[i].x - b[i].x;
    double dy = a[i].y - b[i].y;
    sum += std::sqrt(dx * dx + dy * dy);
  }
  return sum / 4.0;
}

static std::vector<std::vector<cv::Point3f>> CandidateObjectPoints(double tag_size_m) {
  const float s = static_cast<float>(tag_size_m);
  const float h = s * 0.5f;
  std::vector<std::vector<cv::Point3f>> c;
  c.reserve(4);

  // 0) centered, y-up
  c.push_back({cv::Point3f(-h, +h, 0), cv::Point3f(+h, +h, 0), cv::Point3f(+h, -h, 0), cv::Point3f(-h, -h, 0)});
  // 1) centered, y-down
  c.push_back({cv::Point3f(-h, -h, 0), cv::Point3f(+h, -h, 0), cv::Point3f(+h, +h, 0), cv::Point3f(-h, +h, 0)});
  // 2) top-left origin
  c.push_back({cv::Point3f(0, 0, 0), cv::Point3f(s, 0, 0), cv::Point3f(s, s, 0), cv::Point3f(0, s, 0)});
  // 3) top-left origin alternative
  c.push_back({cv::Point3f(0, 0, 0), cv::Point3f(0, s, 0), cv::Point3f(s, s, 0), cv::Point3f(s, 0, 0)});

  return c;
}

static std::vector<cv::Point3f> SelectBestObjectPoints(
    double tag_size_m,
    const cv::Matx33d& K,
    const cv::Vec4d& D,
    const cv::Vec3d& rvec,
    const cv::Vec3d& tvec,
    const std::array<cv::Point2f, 4>& detected_corners) {
  auto candidates = CandidateObjectPoints(tag_size_m);
  cv::Mat dist(1, 4, CV_64F);
  dist.at<double>(0, 0) = D[0];
  dist.at<double>(0, 1) = D[1];
  dist.at<double>(0, 2) = D[2];
  dist.at<double>(0, 3) = D[3];

  double best_err = 1e18;
  size_t best_idx = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    std::vector<cv::Point2f> proj;
    cv::projectPoints(candidates[i], rvec, tvec, cv::Mat(K), dist, proj);
    double err = MeanCornerError(detected_corners, proj);
    if (err < best_err) {
      best_err = err;
      best_idx = i;
    }
  }

  std::cerr << "Selected marker corner object-point pattern #" << best_idx
            << " (mean self reproj err=" << best_err << " px)" << std::endl;
  return candidates[best_idx];
}

struct Det {
  int id = -1;
  std::array<cv::Point2f, 4> corners;
  cv::Vec3d rvec;
  cv::Vec3d tvec;
  aruco_extrinsic_calib_c3p::SE3d T_c_m;  // camera <- marker
};

static std::unordered_map<int, Det> DetectAndEstimate(
    const cv::Mat& img,
    const cv::Ptr<cv::aruco::Dictionary>& dict,
    const cv::Ptr<cv::aruco::DetectorParameters>& params,
    double tag_size_m,
    const cv::Matx33d& K,
    const cv::Vec4d& D,
    bool draw_axes,
    cv::Mat& draw_img) {
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<std::vector<cv::Point2f>> rejected;

  cv::aruco::detectMarkers(img, dict, corners, ids, params, rejected);

  draw_img = img.clone();
  if (!ids.empty()) cv::aruco::drawDetectedMarkers(draw_img, corners, ids);

  std::unordered_map<int, Det> out;
  if (ids.empty()) return out;

  std::vector<cv::Vec3d> rvecs, tvecs;
  cv::Mat dist(1, 4, CV_64F);
  dist.at<double>(0, 0) = D[0];
  dist.at<double>(0, 1) = D[1];
  dist.at<double>(0, 2) = D[2];
  dist.at<double>(0, 3) = D[3];
  cv::aruco::estimatePoseSingleMarkers(corners, (float)tag_size_m, cv::Mat(K), dist, rvecs, tvecs);

  for (size_t i = 0; i < ids.size(); ++i) {
    Det d;
    d.id = ids[i];
    for (int k = 0; k < 4; ++k) d.corners[k] = corners[i][k];
    d.rvec = rvecs[i];
    d.tvec = tvecs[i];
    d.T_c_m = aruco_extrinsic_calib_c3p::FromRvecTvec(d.rvec, d.tvec);
    out[d.id] = d;
    if (draw_axes) {
      cv::aruco::drawAxis(draw_img, cv::Mat(K), dist, d.rvec, d.tvec, (float)tag_size_m * 0.5f);
    }
  }
  return out;
}

struct Frame {
  ros::Time stamp;
  cv::Mat img;
};

static bool SyncFour(std::array<std::deque<Frame>, 4>& qs,
                     double tol_sec,
                     std::array<Frame, 4>& out) {
  // Need all non-empty
  for (int k = 0; k < 4; ++k) {
    if (qs[k].empty()) return false;
  }

  // Iterate until sync or queues exhausted.
  while (true) {
    bool any_empty = false;
    for (int k = 0; k < 4; ++k) {
      if (qs[k].empty()) any_empty = true;
    }
    if (any_empty) return false;

    double t0 = qs[0].front().stamp.toSec();
    double t1 = qs[1].front().stamp.toSec();
    double t2 = qs[2].front().stamp.toSec();
    double t3 = qs[3].front().stamp.toSec();
    double tmin = std::min(std::min(t0, t1), std::min(t2, t3));
    double tmax = std::max(std::max(t0, t1), std::max(t2, t3));
    if ((tmax - tmin) <= tol_sec) {
      out[0] = qs[0].front();
      out[1] = qs[1].front();
      out[2] = qs[2].front();
      out[3] = qs[3].front();
      for (int k = 0; k < 4; ++k) qs[k].pop_front();
      return true;
    }

    // Drop the oldest frame (min timestamp)
    int k_drop = 0;
    double best = t0;
    if (t1 < best) { best = t1; k_drop = 1; }
    if (t2 < best) { best = t2; k_drop = 2; }
    if (t3 < best) { best = t3; k_drop = 3; }
    qs[k_drop].pop_front();
  }
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "overlay_reprojection_zcwd_from_bag");

  const std::string bag_path = GetArg(argc, argv, "--bag", "");
  const std::string calib_path = GetArg(argc, argv, "--calib", "calib-camchain.yaml");
  const std::string xyzw_path = GetArg(argc, argv, "--xyzw_yaml", "");
  const std::string out_dir = EnsureDirSlash(GetArg(argc, argv, "--out_dir", "out_reproj_zcwd"));

  const double tag_size_m = GetArgD(argc, argv, "--tag_size", 0.25);
  const std::string dict_name = GetArg(argc, argv, "--dict", "DICT_6X6_250");
  const double sync_tol = GetArgD(argc, argv, "--sync_tol", 0.01);
  const double fps_fallback = GetArgD(argc, argv, "--fps", 0.0);
  const bool draw_axes = !HasFlag(argc, argv, "--no_axes");
  const bool swap_rb = HasFlag(argc, argv, "--swap_rb");
  const ImageMode image_mode = ParseImageMode(GetArg(argc, argv, "--image_mode", "auto"));

  if (bag_path.empty() || xyzw_path.empty()) {
    std::cerr << "Usage:\n"
              << "  overlay_reprojection_zcwd_from_bag --bag data.bag --calib camchain.yaml --xyzw_yaml out/estimated_XYZW.yaml \\\n"
              << "    [--out_dir out_reproj_zcwd] [--tag_size 0.25] [--dict DICT_6X6_250] [--sync_tol 0.01] [--fps 0] [--image_mode auto|compressed|raw] [--swap_rb] [--no_axes]\\n"
              << "    [--topics t0,t1,t2,t3]  # override topics for (cam0,cam1,cam2,cam3)\\n";
    return 2;
  }

  if (!MkdirP(out_dir)) {
    std::cerr << "Failed to create output dir: " << out_dir << "\n";
    return 3;
  }

  aruco_extrinsic_calib_c3p::CamChain chain = aruco_extrinsic_calib_c3p::LoadCamChainYaml(calib_path);
  aruco_extrinsic_calib_c3p::XYZWEstimate est = aruco_extrinsic_calib_c3p::LoadXYZWYaml(xyzw_path);

  // Allow overriding cams/markers, but warn: this may not match the transforms stored in the YAML.
  const int cam0 = GetArgI(argc, argv, "--cam0", est.cam0);
  const int cam1 = GetArgI(argc, argv, "--cam1", est.cam1);
  const int cam2 = GetArgI(argc, argv, "--cam2", est.cam2);
  const int cam3 = GetArgI(argc, argv, "--cam3", est.cam3);

  const int marker0 = GetArgI(argc, argv, "--marker0", est.marker0);
  const int marker1 = GetArgI(argc, argv, "--marker1", est.marker1);
  const int marker2 = GetArgI(argc, argv, "--marker2", est.marker2);
  const int marker3 = GetArgI(argc, argv, "--marker3", est.marker3);

  if (cam0 != est.cam0 || cam1 != est.cam1 || cam2 != est.cam2 || cam3 != est.cam3 ||
      marker0 != est.marker0 || marker1 != est.marker1 || marker2 != est.marker2 || marker3 != est.marker3) {
    std::cerr << "Warning: overriding cam/marker IDs relative to those stored in --xyzw_yaml.\n"
              << "  YAML cams:   (" << est.cam0 << "," << est.cam1 << "," << est.cam2 << "," << est.cam3 << ")\n"
              << "  argv cams:   (" << cam0 << "," << cam1 << "," << cam2 << "," << cam3 << ")\n"
              << "  YAML markers:(" << est.marker0 << "," << est.marker1 << "," << est.marker2 << "," << est.marker3 << ")\n"
              << "  argv markers:(" << marker0 << "," << marker1 << "," << marker2 << "," << marker3 << ")\n";
  }

  if (cam0 < 0 || cam1 < 0 || cam2 < 0 || cam3 < 0 || cam0 >= chain.size() || cam1 >= chain.size() || cam2 >= chain.size() || cam3 >= chain.size()) {
    std::cerr << "Invalid camera indices for camchain.yaml (chain size=" << chain.size() << ")\n";
    return 4;
  }

  const auto& c0 = chain.cams.at(cam0);
  const auto& c1 = chain.cams.at(cam1);
  const auto& c2 = chain.cams.at(cam2);
  const auto& c3 = chain.cams.at(cam3);

  // Topic override
  std::array<std::string, 4> topics_norm = {
      NormalizeTopic(c0.rostopic),
      NormalizeTopic(c1.rostopic),
      NormalizeTopic(c2.rostopic),
      NormalizeTopic(c3.rostopic)};

  const std::string topics_override = GetArg(argc, argv, "--topics", "");
  if (!topics_override.empty()) {
    auto tt = SplitComma(topics_override);
    if (tt.size() != 4) {
      std::cerr << "--topics must have exactly 4 comma-separated topics. Got " << tt.size() << "\n";
      return 5;
    }
    for (int k = 0; k < 4; ++k) topics_norm[k] = NormalizeTopic(tt[k]);
  }

  // Dictionary
  int dict_id = cv::aruco::DICT_6X6_250;
  if (dict_name == "DICT_6X6_100") dict_id = cv::aruco::DICT_6X6_100;
  else if (dict_name == "DICT_6X6_250") dict_id = cv::aruco::DICT_6X6_250;
  else if (dict_name == "DICT_6X6_1000") dict_id = cv::aruco::DICT_6X6_1000;
  else {
    std::cerr << "Unknown --dict " << dict_name << ", using DICT_6X6_250\n";
  }
  cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(dict_id);
  cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

  // Convert estimated transforms to SE3d
  auto mat44_to_se3 = [](const cv::Matx44d& M) {
    cv::Matx33d R(M(0, 0), M(0, 1), M(0, 2),
                  M(1, 0), M(1, 1), M(1, 2),
                  M(2, 0), M(2, 1), M(2, 2));
    cv::Vec3d t(M(0, 3), M(1, 3), M(2, 3));
    return aruco_extrinsic_calib_c3p::SE3d(R, t);
  };

  const aruco_extrinsic_calib_c3p::SE3d X = mat44_to_se3(est.X_c1_c0);  // cam1 <- cam0
  const aruco_extrinsic_calib_c3p::SE3d W = mat44_to_se3(est.W_c3_c2);  // cam3 <- cam2
  const aruco_extrinsic_calib_c3p::SE3d Y = mat44_to_se3(est.Y_m0_m2);  // m0 <- m2
  const aruco_extrinsic_calib_c3p::SE3d Z = mat44_to_se3(est.Z_m1_m3);  // m1 <- m3

  // Pass 1: estimate FPS from stamps
  std::array<std::vector<ros::Time>, 4> stamps;
  try {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    std::vector<std::string> topics = {topics_norm[0], topics_norm[1], topics_norm[2], topics_norm[3]};
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    for (const auto& m : view) {
      const std::string topic = NormalizeTopic(m.getTopic());
      ros::Time stamp;
      if (!ExtractStampFromMsg(m, image_mode, stamp)) continue;
      for (int k = 0; k < 4; ++k) {
        if (TopicEqualsNormalized(topic, topics_norm[k])) {
          stamps[k].push_back(stamp);
          break;
        }
      }
    }
    bag.close();
  } catch (const std::exception& e) {
    std::cerr << "Failed to read bag for FPS pass: " << e.what() << "\n";
    return 6;
  }

  const double fallback = (fps_fallback > 0.0) ? fps_fallback : 10.0;
  std::array<double, 4> fps_each = {
      EstimateFpsFromStamps(stamps[0], fallback),
      EstimateFpsFromStamps(stamps[1], fallback),
      EstimateFpsFromStamps(stamps[2], fallback),
      EstimateFpsFromStamps(stamps[3], fallback)};
  const double fps = std::min(std::min(fps_each[0], fps_each[1]), std::min(fps_each[2], fps_each[3]));
  std::cout << "Estimated FPS: [" << fps_each[0] << ", " << fps_each[1] << ", " << fps_each[2] << ", " << fps_each[3]
            << "] -> using " << fps << "\n";

  // Pass 2: iterate, sync, draw, write videos
  std::array<cv::VideoWriter, 4> writers;
  std::array<bool, 4> writer_open = {false, false, false, false};
  std::array<std::string, 4> out_paths = {
      out_dir + "cam" + std::to_string(cam0) + "_zcwd_reproj.mp4",
      out_dir + "cam" + std::to_string(cam1) + "_zcwd_reproj.mp4",
      out_dir + "cam" + std::to_string(cam2) + "_zcwd_reproj.mp4",
      out_dir + "cam" + std::to_string(cam3) + "_zcwd_reproj.mp4"};

  std::vector<cv::Point3f> obj_pts;
  bool obj_pts_inferred = false;

  size_t synced = 0;
  size_t used = 0;

  try {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    std::vector<std::string> topics = {topics_norm[0], topics_norm[1], topics_norm[2], topics_norm[3]};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    std::array<std::deque<Frame>, 4> qs;
    std::array<Frame, 4> synced_frames;

    BOOST_FOREACH (rosbag::MessageInstance const m, view) {
      const std::string topic = NormalizeTopic(m.getTopic());
      cv::Mat img;
      ros::Time stamp;
      if (!DecodeImageFromMsg(m, image_mode, img, stamp)) continue;
      if (img.empty()) continue;
      MaybeSwapRB(img, swap_rb);

      for (int k = 0; k < 4; ++k) {
        if (TopicEqualsNormalized(topic, topics_norm[k])) {
          qs[k].push_back(Frame{stamp, img});
          break;
        }
      }

      while (SyncFour(qs, sync_tol, synced_frames)) {
        synced++;

        const cv::Mat& img0 = synced_frames[0].img;
        const cv::Mat& img1 = synced_frames[1].img;
        const cv::Mat& img2 = synced_frames[2].img;
        const cv::Mat& img3 = synced_frames[3].img;

        // Open videos if needed
        auto open_one = [&](int k, const cv::Mat& img) {
          if (writer_open[k]) return true;
          if (!TryOpenVideo(writers[k], out_paths[k], img.cols, img.rows, fps)) {
            // Fallback AVI
            std::string avi = out_paths[k];
            size_t p = avi.rfind(".mp4");
            if (p != std::string::npos) avi = avi.substr(0, p) + ".avi";
            int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            if (!writers[k].open(avi, fourcc, fps, cv::Size(img.cols, img.rows), true)) {
              return false;
            }
            out_paths[k] = avi;
          }
          writer_open[k] = true;
          return true;
        };

        if (!open_one(0, img0) || !open_one(1, img1) || !open_one(2, img2) || !open_one(3, img3)) {
          std::cerr << "Failed to open video writers. Install FFmpeg/GStreamer or check codecs.\n";
          return 7;
        }

        // Detect
        cv::Mat draw0, draw1, draw2, draw3;
        auto det0 = DetectAndEstimate(img0, dict, params, tag_size_m, c0.K, c0.D, draw_axes, draw0);
        auto det1 = DetectAndEstimate(img1, dict, params, tag_size_m, c1.K, c1.D, draw_axes, draw1);
        auto det2 = DetectAndEstimate(img2, dict, params, tag_size_m, c2.K, c2.D, draw_axes, draw2);
        auto det3 = DetectAndEstimate(img3, dict, params, tag_size_m, c3.K, c3.D, draw_axes, draw3);

        if (det0.find(marker0) == det0.end() || det1.find(marker1) == det1.end() ||
            det2.find(marker2) == det2.end() || det3.find(marker3) == det3.end()) {
          // Write frames with detections only.
          writers[0].write(draw0);
          writers[1].write(draw1);
          writers[2].write(draw2);
          writers[3].write(draw3);
          continue;
        }

        used++;

        // Infer object points ordering once
        if (!obj_pts_inferred) {
          const Det& d = det0.at(marker0);
          obj_pts = SelectBestObjectPoints(tag_size_m, c0.K, c0.D, d.rvec, d.tvec, d.corners);
          obj_pts_inferred = true;
        }

        // Build measured transforms
        const aruco_extrinsic_calib_c3p::SE3d T_c0_m0 = det0.at(marker0).T_c_m;
        const aruco_extrinsic_calib_c3p::SE3d T_c1_m1 = det1.at(marker1).T_c_m;
        const aruco_extrinsic_calib_c3p::SE3d T_c2_m2 = det2.at(marker2).T_c_m;
        const aruco_extrinsic_calib_c3p::SE3d T_c3_m3 = det3.at(marker3).T_c_m;

        const aruco_extrinsic_calib_c3p::SE3d A = aruco_extrinsic_calib_c3p::Inverse(T_c1_m1); // m1 <- c1
        const aruco_extrinsic_calib_c3p::SE3d B = T_c0_m0;                                      // c0 <- m0
        const aruco_extrinsic_calib_c3p::SE3d C = aruco_extrinsic_calib_c3p::Inverse(T_c3_m3); // m3 <- c3
        const aruco_extrinsic_calib_c3p::SE3d D = T_c2_m2;                                      // c2 <- m2

        auto Mul = [&](const aruco_extrinsic_calib_c3p::SE3d& P, const aruco_extrinsic_calib_c3p::SE3d& Q) {
          return aruco_extrinsic_calib_c3p::Compose(P, Q);
        };
        auto Inv = [&](const aruco_extrinsic_calib_c3p::SE3d& T) { return aruco_extrinsic_calib_c3p::Inverse(T); };

        // Predict each measurement using the other three cameras' measurements + estimated X,Y,Z,W
        // A_pred = Z C W D Y^-1 B^-1 X^-1
        const aruco_extrinsic_calib_c3p::SE3d A_pred = Mul(Mul(Mul(Mul(Mul(Mul(Mul(Z, C), W), D), Inv(Y)), Inv(B)), Inv(X)), aruco_extrinsic_calib_c3p::SE3d());
        // The trailing identity above avoids clang "-Wvexing-parse" pitfalls with nested temporaries.

        // B_pred = X^-1 A^-1 Z C W D Y^-1  (A^-1 is T_c1_m1)
        const aruco_extrinsic_calib_c3p::SE3d B_pred = Mul(Mul(Mul(Mul(Mul(Mul(Inv(X), Inv(A)), Z), C), W), D), Inv(Y));

        // C_pred = Z^-1 A X B Y D^-1 W^-1
        const aruco_extrinsic_calib_c3p::SE3d C_pred = Mul(Mul(Mul(Mul(Mul(Mul(Mul(Inv(Z), A), X), B), Y), Inv(D)), Inv(W)), aruco_extrinsic_calib_c3p::SE3d());

        // D_pred = W^-1 C^-1 Z^-1 A X B Y   (C^-1 is T_c3_m3)
        const aruco_extrinsic_calib_c3p::SE3d D_pred = Mul(Mul(Mul(Mul(Mul(Mul(Inv(W), Inv(C)), Inv(Z)), A), X), B), Y);

        const aruco_extrinsic_calib_c3p::SE3d T_c1_m1_pred = Inv(A_pred);
        const aruco_extrinsic_calib_c3p::SE3d T_c0_m0_pred = B_pred;
        const aruco_extrinsic_calib_c3p::SE3d T_c3_m3_pred = Inv(C_pred);
        const aruco_extrinsic_calib_c3p::SE3d T_c2_m2_pred = D_pred;

        auto project_and_draw = [&](cv::Mat& canvas,
                                    const aruco_extrinsic_calib_c3p::SE3d& T_c_m_pred,
                                    const aruco_extrinsic_calib_c3p::CameraModel& cam,
                                    const Det& det_meas,
                                    const std::string& label) {
          cv::Vec3d rvec, tvec;
          SE3ToRvecTvec(T_c_m_pred, rvec, tvec);

          cv::Mat dist(1, 4, CV_64F);
          dist.at<double>(0, 0) = cam.D[0];
          dist.at<double>(0, 1) = cam.D[1];
          dist.at<double>(0, 2) = cam.D[2];
          dist.at<double>(0, 3) = cam.D[3];

          std::vector<cv::Point2f> proj;
          cv::projectPoints(obj_pts, rvec, tvec, cv::Mat(cam.K), dist, proj);
          DrawPoly(canvas, proj, cv::Scalar(0, 0, 255), 2);  // predicted: red

          double err = MeanCornerError(det_meas.corners, proj);
          std::ostringstream oss;
          oss.setf(std::ios::fixed);
          oss << label << " pred_err=" << std::setprecision(2) << err << " px";
          cv::putText(canvas, oss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        };

        project_and_draw(draw0, T_c0_m0_pred, c0, det0.at(marker0), "cam" + std::to_string(cam0) + "/m" + std::to_string(marker0));
        project_and_draw(draw1, T_c1_m1_pred, c1, det1.at(marker1), "cam" + std::to_string(cam1) + "/m" + std::to_string(marker1));
        project_and_draw(draw2, T_c2_m2_pred, c2, det2.at(marker2), "cam" + std::to_string(cam2) + "/m" + std::to_string(marker2));
        project_and_draw(draw3, T_c3_m3_pred, c3, det3.at(marker3), "cam" + std::to_string(cam3) + "/m" + std::to_string(marker3));

        writers[0].write(draw0);
        writers[1].write(draw1);
        writers[2].write(draw2);
        writers[3].write(draw3);
      }
    }
    bag.close();
  } catch (const std::exception& e) {
    std::cerr << "Error during bag processing: " << e.what() << "\n";
    return 8;
  }

  std::cout << "Synchronized sets: " << synced << "\n";
  std::cout << "Used (all 4 markers present): " << used << "\n";
  std::cout << "Wrote:\n";
  for (int k = 0; k < 4; ++k) {
    std::cout << "  " << out_paths[k] << "\n";
  }
  std::cout << "Legend: detected markers=green, reprojection via estimated (X,Y,Z,W)=red\n";
  return 0;
}
