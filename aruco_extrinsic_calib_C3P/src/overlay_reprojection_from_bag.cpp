#include "aruco_extrinsic_calib_c3p/camchain_yaml.h"
#include "aruco_extrinsic_calib_c3p/pose_utils.h"
#include "aruco_extrinsic_calib_c3p/xy_yaml.h"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <boost/foreach.hpp>

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

#include <random>

namespace {

static bool TopicEquals(const std::string& a, const std::string& b) {
  if (a == b) return true;
  if (!a.empty() && a[0] != '/' && ("/" + a) == b) return true;
  if (!b.empty() && b[0] != '/' && ("/" + b) == a) return true;
  return false;
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

static bool MkdirP(const std::string& out_dir) {
  std::ostringstream cmd;
  cmd << "mkdir -p " << out_dir;
  int ret = std::system(cmd.str().c_str());
  return (ret == 0);
}

static std::string EnsureDirSlash(std::string s) {
  if (s.empty()) return "./";
  if (s.back() != '/') s.push_back('/');
  return s;
}

static cv::Mat DecodeCompressed(const sensor_msgs::CompressedImage& msg) {
  cv::Mat raw(1, (int)msg.data.size(), CV_8UC1, const_cast<uint8_t*>(msg.data.data()));
  return cv::imdecode(raw, cv::IMREAD_COLOR);
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

static aruco_extrinsic_calib_c3p::SE3d Mat44ToSE3(const cv::Matx44d& M) {
  cv::Matx33d R(M(0, 0), M(0, 1), M(0, 2),
                M(1, 0), M(1, 1), M(1, 2),
                M(2, 0), M(2, 1), M(2, 2));
  cv::Vec3d t(M(0, 3), M(1, 3), M(2, 3));
  return aruco_extrinsic_calib_c3p::SE3d(R, t);
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

static aruco_extrinsic_calib_c3p::SE3d SampleNoiseSE3(std::mt19937& rng,
                                                   double sigma_rot_rad,
                                                   double sigma_trans_m) {
  std::normal_distribution<double> n_rot(0.0, sigma_rot_rad);
  std::normal_distribution<double> n_tr(0.0, sigma_trans_m);

  cv::Vec3d w(n_rot(rng), n_rot(rng), n_rot(rng));
  cv::Vec3d v(n_tr(rng), n_tr(rng), n_tr(rng));

  cv::Mat rvec(3, 1, CV_64F);
  rvec.at<double>(0, 0) = w[0];
  rvec.at<double>(1, 0) = w[1];
  rvec.at<double>(2, 0) = w[2];

  cv::Mat Rcv;
  cv::Rodrigues(rvec, Rcv);
  cv::Matx33d R;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      R(r, c) = Rcv.at<double>(r, c);

  return aruco_extrinsic_calib_c3p::SE3d(R, v);
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

// Candidate marker object points patterns. We infer the best ordering by minimizing
// the reprojection error of (rvec,tvec) from estimatePoseSingleMarkers against the
// detected corners ordering.
static std::vector<std::vector<cv::Point3f>> CandidateObjectPoints(double tag_size_m) {
  const float s = static_cast<float>(tag_size_m);
  const float h = s * 0.5f;
  std::vector<std::vector<cv::Point3f>> c;
  c.reserve(4);

  // 0) Centered, top-left then clockwise in image (common for ARUCO_CCW_CENTER).
  c.push_back({cv::Point3f(-h, +h, 0), cv::Point3f(+h, +h, 0), cv::Point3f(+h, -h, 0), cv::Point3f(-h, -h, 0)});

  // 1) Centered, top-left then clockwise assuming y-down convention.
  c.push_back({cv::Point3f(-h, -h, 0), cv::Point3f(+h, -h, 0), cv::Point3f(+h, +h, 0), cv::Point3f(-h, +h, 0)});

  // 2) Top-left origin (0,0) order.
  c.push_back({cv::Point3f(0, 0, 0), cv::Point3f(s, 0, 0), cv::Point3f(s, s, 0), cv::Point3f(0, s, 0)});

  // 3) Top-left origin alternative.
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

  std::cerr << "Selected object point pattern #" << best_idx
            << " (mean self reproj err=" << best_err << " px)\n";
  return candidates[best_idx];
}

struct Det {
  int id = -1;
  std::array<cv::Point2f, 4> corners;
  cv::Vec3d rvec;
  cv::Vec3d tvec;
  aruco_extrinsic_calib_c3p::SE3d T_c_m;
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

  // Pose estimation
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

struct ImgMsg {
  ros::Time stamp;
  sensor_msgs::CompressedImage::ConstPtr msg;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "overlay_reprojection_from_bag");

  const std::string bag_path = GetArg(argc, argv, "--bag", "");
  const std::string calib_path = GetArg(argc, argv, "--calib", "calib-camchain.yaml");
  const std::string xy_path = GetArg(argc, argv, "--xy_yaml", "");
  const std::string out_dir = EnsureDirSlash(GetArg(argc, argv, "--out_dir", "out_reproj"));
  const double tag_size_m = GetArgD(argc, argv, "--tag_size", 0.25);
  const double marker_margin_m = GetArgD(argc, argv, "--marker_margin", 0.01);
  const std::string dict_name = GetArg(argc, argv, "--dict", "DICT_6X6_250");
  const double sync_tol = GetArgD(argc, argv, "--sync_tol", 0.01);
  const double fps_fallback = GetArgD(argc, argv, "--fps", 0.0);
  const bool draw_axes = !HasFlag(argc, argv, "--no_axes");
  const bool swap_rb = HasFlag(argc, argv, "--swap_rb");

  // Ground-truth reference reprojection options.
  // If --grid_dx/--grid_dy are not provided, we infer them by rounding the estimated Y translation.
  const int grid_dx_in = GetArgI(argc, argv, "--grid_dx", 9999);
  const int grid_dy_in = GetArgI(argc, argv, "--grid_dy", 9999);

  const double gt_noise_rot_deg = GetArgD(argc, argv, "--gt_noise_rot_deg", 0.0);
  const double gt_noise_trans_m = GetArgD(argc, argv, "--gt_noise_trans_m", 0.0);
  const int gt_noise_seed = GetArgI(argc, argv, "--gt_noise_seed", 1);
  const int gt_noise_on_X = GetArgI(argc, argv, "--gt_noise_on_X", 1);
  const int gt_noise_on_Y = GetArgI(argc, argv, "--gt_noise_on_Y", 1);
  const bool show_ref = !HasFlag(argc, argv, "--no_ref");

  if (bag_path.empty() || xy_path.empty()) {
    std::cerr << "Usage:\n"
              << "  overlay_reprojection_from_bag --bag calib.bag --calib calib-camchain.yaml --xy_yaml out/estimated_XY.yaml \\\n"
              << "    [--out_dir out_reproj] [--tag_size 0.25] [--marker_margin 0.01] [--dict DICT_6X6_250] [--sync_tol 0.01] [--fps 0] [--swap_rb] [--no_axes]\n"
              << "    [--grid_dx INT --grid_dy INT]  # board step offsets (marker1 relative to marker0) in marker-frame axes\n"
              << "    [--gt_noise_rot_deg 0.0 --gt_noise_trans_m 0.0 --gt_noise_seed 1 --gt_noise_on_X 1 --gt_noise_on_Y 1]\n"
              << "    [--no_ref]  # disable ground-truth reference overlay\n";
    return 2;
  }

  if (!MkdirP(out_dir)) {
    std::cerr << "Failed to create output dir: " << out_dir << "\n";
    return 3;
  }

  aruco_extrinsic_calib_c3p::CamChain chain = aruco_extrinsic_calib_c3p::LoadCamChainYaml(calib_path);
  const auto& cam0 = chain.cams.at(0);
  const auto& cam1 = chain.cams.at(1);

  const aruco_extrinsic_calib_c3p::XYEstimate xy = aruco_extrinsic_calib_c3p::LoadXYYaml(xy_path);
  const int marker0_id = GetArgI(argc, argv, "--marker0_id", xy.marker0_id);
  const int marker1_id = GetArgI(argc, argv, "--marker1_id", xy.marker1_id);

  const aruco_extrinsic_calib_c3p::SE3d X = Mat44ToSE3(xy.X_c1_c0);  // cam1 <- cam0
  const aruco_extrinsic_calib_c3p::SE3d Y = Mat44ToSE3(xy.Y_m1_m0);  // m1 <- m0

  // -------- Build a "reference" (ground-truth + optional noise) X_ref and Y_ref --------
  // X_gt comes from camchain.yaml (cam1/T_cn_cnm1), interpreted as cam1 <- cam0.
  aruco_extrinsic_calib_c3p::SE3d X_gt = X;
  if (cam1.has_T_to_prev) {
    X_gt = Mat44ToSE3(cam1.T_this_prev);
  } else {
    std::cerr << "Warning: calib file has no cam1/T_cn_cnm1; reference overlay will use estimated X." << std::endl;
  }

  // Y_gt is derived from the board geometry: two marker frames assumed to be aligned (same print orientation).
  // Let step = tag_size + marker_margin. If marker1 is at (grid_dx,grid_dy) steps from marker0 (expressed in marker0 axes),
  // then Y = T_m1_m0 has translation t = (-grid_dx*step, -grid_dy*step, 0).
  const double step = tag_size_m + marker_margin_m;
  bool grid_provided = (grid_dx_in != 9999) || (grid_dy_in != 9999);
  int grid_dx = 0;
  int grid_dy = 0;
  if (grid_provided) {
    grid_dx = (grid_dx_in != 9999) ? grid_dx_in : 0;
    grid_dy = (grid_dy_in != 9999) ? grid_dy_in : 0;
  } else {
    if (step > 1e-9) {
      grid_dx = (int)std::lround(-Y.t[0] / step);
      grid_dy = (int)std::lround(-Y.t[1] / step);
    }
    std::cout << "Inferred board offsets from estimated Y: grid_dx=" << grid_dx
              << " grid_dy=" << grid_dy << " (step=" << step << " m)" << std::endl;
  }

  aruco_extrinsic_calib_c3p::SE3d Y_gt(cv::Matx33d::eye(), cv::Vec3d(-grid_dx * step, -grid_dy * step, 0.0));

  aruco_extrinsic_calib_c3p::SE3d X_ref = X_gt;
  aruco_extrinsic_calib_c3p::SE3d Y_ref = Y_gt;

  const double sigma_rot_rad = gt_noise_rot_deg * (3.14159265358979323846 / 180.0);
  const double sigma_trans_m = gt_noise_trans_m;
  if (show_ref && (sigma_rot_rad > 0.0 || sigma_trans_m > 0.0)) {
    std::mt19937 rng((uint32_t)gt_noise_seed);
    if (gt_noise_on_X) {
      X_ref = aruco_extrinsic_calib_c3p::Compose(SampleNoiseSE3(rng, sigma_rot_rad, sigma_trans_m), X_gt);
    }
    if (gt_noise_on_Y) {
      Y_ref = aruco_extrinsic_calib_c3p::Compose(SampleNoiseSE3(rng, sigma_rot_rad, sigma_trans_m), Y_gt);
    }
    std::cout << "Reference overlay: GT + noise (rot_sigma_deg=" << gt_noise_rot_deg
              << ", trans_sigma_m=" << gt_noise_trans_m
              << ", seed=" << gt_noise_seed
              << ", on_X=" << gt_noise_on_X
              << ", on_Y=" << gt_noise_on_Y
              << ")" << std::endl;
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

  // Pass 1: estimate FPS
  std::vector<ros::Time> stamps0, stamps1;
  try {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    std::vector<std::string> topics = {cam0.rostopic, cam1.rostopic};
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    for (const auto& m : view) {
      const std::string topic = m.getTopic();
      sensor_msgs::CompressedImage::ConstPtr img = m.instantiate<sensor_msgs::CompressedImage>();
      if (!img) continue;
      if (TopicEquals(topic, cam0.rostopic)) stamps0.push_back(img->header.stamp);
      else if (TopicEquals(topic, cam1.rostopic)) stamps1.push_back(img->header.stamp);
    }
    bag.close();
  } catch (const std::exception& e) {
    std::cerr << "Failed to read bag for FPS pass: " << e.what() << "\n";
    return 4;
  }

  const double fallback = (fps_fallback > 0.0) ? fps_fallback : 10.0;
  const double fps0 = EstimateFpsFromStamps(stamps0, fallback);
  const double fps1 = EstimateFpsFromStamps(stamps1, fallback);
  const double fps = std::min(fps0, fps1);
  std::cout << "Estimated FPS: cam0=" << fps0 << " cam1=" << fps1 << " -> using " << fps << "\n";

  // Pass 2: iterate, sync, draw
  cv::VideoWriter w0, w1;
  bool video_open = false;
  std::vector<cv::Point3f> obj_pts;  // inferred object points order
  bool obj_pts_inferred = false;

  std::string v0_path = out_dir + "cam0_reproj.mp4";
  std::string v1_path = out_dir + "cam1_reproj.mp4";

  size_t pair_count = 0;
  size_t pair_used = 0;

  try {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    std::vector<std::string> topics = {cam0.rostopic, cam1.rostopic};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    std::deque<ImgMsg> q0, q1;
    BOOST_FOREACH (rosbag::MessageInstance const m, view) {
      const std::string topic = m.getTopic();
      sensor_msgs::CompressedImage::ConstPtr img_msg = m.instantiate<sensor_msgs::CompressedImage>();
      if (!img_msg) continue;

      ImgMsg mm;
      mm.stamp = img_msg->header.stamp;
      mm.msg = img_msg;

      if (TopicEquals(topic, cam0.rostopic)) q0.push_back(mm);
      else if (TopicEquals(topic, cam1.rostopic)) q1.push_back(mm);
      else continue;

      // Try sync
      while (!q0.empty() && !q1.empty()) {
        double t0 = q0.front().stamp.toSec();
        double t1 = q1.front().stamp.toSec();
        double dt = t0 - t1;
        if (std::abs(dt) <= sync_tol) {
          pair_count++;

          cv::Mat img0 = DecodeCompressed(*q0.front().msg);
          cv::Mat img1 = DecodeCompressed(*q1.front().msg);
          MaybeSwapRB(img0, swap_rb);
          MaybeSwapRB(img1, swap_rb);

          q0.pop_front();
          q1.pop_front();

          if (img0.empty() || img1.empty()) continue;

          if (!video_open) {
            bool ok_mp4 = TryOpenVideo(w0, v0_path, img0.cols, img0.rows, fps) &&
                          TryOpenVideo(w1, v1_path, img1.cols, img1.rows, fps);
            if (!ok_mp4) {
              // Fallback to AVI (MJPG) if MP4 writer isn't available.
              v0_path = out_dir + "cam0_reproj.avi";
              v1_path = out_dir + "cam1_reproj.avi";
              int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
              cv::Size sz0(img0.cols, img0.rows);
              cv::Size sz1(img1.cols, img1.rows);
              bool ok_avi = w0.open(v0_path, fourcc, fps, sz0, true) && w1.open(v1_path, fourcc, fps, sz1, true);
              if (!ok_avi) {
                std::cerr << "Failed to open video writers (MP4 and AVI). Install FFmpeg/GStreamer or check codecs.\n";
                return 5;
              }
              std::cerr << "Warning: MP4 writer not available; wrote AVI instead.\n";
            }
            video_open = true;
          }

          cv::Mat draw0, draw1;
          auto det0 = DetectAndEstimate(img0, dict, params, tag_size_m, cam0.K, cam0.D, draw_axes, draw0);
          auto det1 = DetectAndEstimate(img1, dict, params, tag_size_m, cam1.K, cam1.D, draw_axes, draw1);

          // Infer object points ordering once, using the first available detection in cam0.
          if (!obj_pts_inferred) {
            if (!det0.empty()) {
              const auto it = det0.begin();
              obj_pts = SelectBestObjectPoints(tag_size_m, cam0.K, cam0.D, it->second.rvec, it->second.tvec, it->second.corners);
              obj_pts_inferred = true;
            } else if (!det1.empty()) {
              const auto it = det1.begin();
              obj_pts = SelectBestObjectPoints(tag_size_m, cam1.K, cam1.D, it->second.rvec, it->second.tvec, it->second.corners);
              obj_pts_inferred = true;
            }
          }

          // Only attempt re-projection when we have both required marker observations.
          auto it_m0_c0 = det0.find(marker0_id);
          auto it_m1_c1 = det1.find(marker1_id);
          if (obj_pts_inferred && it_m0_c0 != det0.end() && it_m1_c1 != det1.end()) {
            const cv::Scalar col_est(0, 0, 255);   // red
            const cv::Scalar col_ref(255, 0, 0);   // blue
            const cv::Scalar col_txt(255, 255, 255);

            // Common inputs
            aruco_extrinsic_calib_c3p::SE3d T_c0_m0 = it_m0_c0->second.T_c_m;
            aruco_extrinsic_calib_c3p::SE3d T_c1_m1 = it_m1_c1->second.T_c_m;

            // Distortion vectors
            cv::Mat dist1(1, 4, CV_64F);
            dist1.at<double>(0, 0) = cam1.D[0];
            dist1.at<double>(0, 1) = cam1.D[1];
            dist1.at<double>(0, 2) = cam1.D[2];
            dist1.at<double>(0, 3) = cam1.D[3];
            cv::Mat dist0(1, 4, CV_64F);
            dist0.at<double>(0, 0) = cam0.D[0];
            dist0.at<double>(0, 1) = cam0.D[1];
            dist0.at<double>(0, 2) = cam0.D[2];
            dist0.at<double>(0, 3) = cam0.D[3];

            // ---------------- cam1 overlay: project marker1 ----------------
            // Estimated: T_c1_m1 = X * T_c0_m0 * inv(Y)
            aruco_extrinsic_calib_c3p::SE3d T_c1_m1_est = aruco_extrinsic_calib_c3p::Compose(
                aruco_extrinsic_calib_c3p::Compose(X, T_c0_m0),
                aruco_extrinsic_calib_c3p::Inverse(Y));
            cv::Vec3d rvec1_est, tvec1_est;
            SE3ToRvecTvec(T_c1_m1_est, rvec1_est, tvec1_est);
            std::vector<cv::Point2f> proj1_est;
            cv::projectPoints(obj_pts, rvec1_est, tvec1_est, cv::Mat(cam1.K), dist1, proj1_est);
            DrawPoly(draw1, proj1_est, col_est, 2);
            double err1_est = MeanCornerError(it_m1_c1->second.corners, proj1_est);

            // Reference (GT + optional noise): T_c1_m1 = X_ref * T_c0_m0 * inv(Y_ref)
            double err1_ref = 0.0;
            if (show_ref) {
              aruco_extrinsic_calib_c3p::SE3d T_c1_m1_ref = aruco_extrinsic_calib_c3p::Compose(
                  aruco_extrinsic_calib_c3p::Compose(X_ref, T_c0_m0),
                  aruco_extrinsic_calib_c3p::Inverse(Y_ref));
              cv::Vec3d rvec1_ref, tvec1_ref;
              SE3ToRvecTvec(T_c1_m1_ref, rvec1_ref, tvec1_ref);
              std::vector<cv::Point2f> proj1_ref;
              cv::projectPoints(obj_pts, rvec1_ref, tvec1_ref, cv::Mat(cam1.K), dist1, proj1_ref);
              DrawPoly(draw1, proj1_ref, col_ref, 2);
              err1_ref = MeanCornerError(it_m1_c1->second.corners, proj1_ref);
            }

            // ---------------- cam0 overlay: project marker0 ----------------
            // Estimated: T_c0_m0 = inv(X) * T_c1_m1 * Y
            aruco_extrinsic_calib_c3p::SE3d T_c0_m0_est = aruco_extrinsic_calib_c3p::Compose(
                aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Inverse(X), T_c1_m1),
                Y);
            cv::Vec3d rvec0_est, tvec0_est;
            SE3ToRvecTvec(T_c0_m0_est, rvec0_est, tvec0_est);
            std::vector<cv::Point2f> proj0_est;
            cv::projectPoints(obj_pts, rvec0_est, tvec0_est, cv::Mat(cam0.K), dist0, proj0_est);
            DrawPoly(draw0, proj0_est, col_est, 2);
            double err0_est = MeanCornerError(it_m0_c0->second.corners, proj0_est);

            // Reference (GT + optional noise): T_c0_m0 = inv(X_ref) * T_c1_m1 * Y_ref
            double err0_ref = 0.0;
            if (show_ref) {
              aruco_extrinsic_calib_c3p::SE3d T_c0_m0_ref = aruco_extrinsic_calib_c3p::Compose(
                  aruco_extrinsic_calib_c3p::Compose(aruco_extrinsic_calib_c3p::Inverse(X_ref), T_c1_m1),
                  Y_ref);
              cv::Vec3d rvec0_ref, tvec0_ref;
              SE3ToRvecTvec(T_c0_m0_ref, rvec0_ref, tvec0_ref);
              std::vector<cv::Point2f> proj0_ref;
              cv::projectPoints(obj_pts, rvec0_ref, tvec0_ref, cv::Mat(cam0.K), dist0, proj0_ref);
              DrawPoly(draw0, proj0_ref, col_ref, 2);
              err0_ref = MeanCornerError(it_m0_c0->second.corners, proj0_ref);
            }

            // Text/legend
            {
              std::ostringstream t;
              t << "err: est=" << std::fixed << std::setprecision(2) << err1_est << "px";
              if (show_ref) t << " ref=" << std::fixed << std::setprecision(2) << err1_ref << "px";
              cv::putText(draw1, t.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, col_txt, 2, cv::LINE_AA);
              cv::putText(draw1, "det=green  est=red  ref=blue", cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, col_txt, 2,
                          cv::LINE_AA);
            }
            {
              std::ostringstream t;
              t << "err: est=" << std::fixed << std::setprecision(2) << err0_est << "px";
              if (show_ref) t << " ref=" << std::fixed << std::setprecision(2) << err0_ref << "px";
              cv::putText(draw0, t.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, col_txt, 2, cv::LINE_AA);
              cv::putText(draw0, "det=green  est=red  ref=blue", cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, col_txt, 2,
                          cv::LINE_AA);
            }

            // If noise is enabled on the reference, show it once per frame (cheap, but helpful for screenshots).
            if (show_ref && (sigma_rot_rad > 0.0 || sigma_trans_m > 0.0)) {
              std::ostringstream n;
              n << "ref noise: rot_sig=" << gt_noise_rot_deg << "deg tr_sig=" << gt_noise_trans_m << "m";
              cv::putText(draw1, n.str(), cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.55, col_txt, 2, cv::LINE_AA);
              cv::putText(draw0, n.str(), cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.55, col_txt, 2, cv::LINE_AA);
            }

            pair_used++;
          }

          // Always label marker IDs used.
          {
            std::ostringstream l0;
            l0 << "marker0_id=" << marker0_id;
            cv::putText(draw0, l0.str(), cv::Point(20, draw0.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            std::ostringstream l1;
            l1 << "marker1_id=" << marker1_id;
            cv::putText(draw1, l1.str(), cv::Point(20, draw1.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
          }

          w0.write(draw0);
          w1.write(draw1);
        } else {
          // Drop the older one.
          if (dt < 0) q0.pop_front();
          else q1.pop_front();
        }
      }
    }
    bag.close();
  } catch (const std::exception& e) {
    std::cerr << "Failed to process bag: " << e.what() << "\n";
    return 6;
  }

  if (video_open) {
    w0.release();
    w1.release();
  }

  std::cout << "Wrote videos:\n"
            << "  " << v0_path << "\n"
            << "  " << v1_path << "\n";
  std::cout << "Synced pairs processed: " << pair_count << ", pairs with reprojection overlay: " << pair_used << "\n";
  return 0;
}
