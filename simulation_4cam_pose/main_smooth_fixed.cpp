// tri_pattern_sim_sequence_autofit_imu_rosbag_fixed.cpp
//
// Fixes requested by user:
//  - DO NOT scale images: intrinsics (fx,fy,cx,cy) and resolution (W,H) remain constant.
//    Auto-fit ONLY changes camera pose (translation + attitude) to keep full board in view.
//  - Fix ffmpeg encoding: remove unsupported "-preset"; use older-compatible flags and fallbacks.
//
// Features:
//  1) Load 3 SVGs -> raster grayscale textures.
//  2) Build tri-orthogonal board: planes z=0 (XY), x=0 (YZ), y=0 (XZ).
//  3) Continuous rig motion + auto-fit that keeps ALL 3 planes fully in view in ALL 4 cameras.
//  4) Render per-camera PNG sequences (lossless; max PNG compression level).
//  5) Encode PNG sequences into MP4 using ffmpeg (system call; robust fallbacks; no scaling filters).
//  6) Generate synthetic IMU (gyro + accel) from rig ground truth.
//  7) Integrate IMU and compare to ground truth (CSV + RMS).
//  8) Save PNG sequences + IMU into ROS1 rosbag (optional, compile with -DWITH_ROS1).
//
// Dependencies (header-only):
//   - nanosvg.h, nanosvgrast.h
//   - stb_image_write.h
//
// Optional ROS1 (compile with -DWITH_ROS1 in a ROS1 environment):
//   - rosbag, sensor_msgs, geometry_msgs
//
// Run:
//   ./sim pattern_xy.svg pattern_yz.svg pattern_xz.svg output_dir/
//
// Notes on "maximum PNG quality":
//   PNG is lossless. We set max compression (level 9) for smaller files without quality loss.
//
// Notes on MP4:
//   We do NOT scale frames. If your ffmpeg/x264 requires even dimensions for yuv420p,
//   ensure IMG_W and IMG_H are even (default here is even).
//   If libx264 isn't available, we fall back to mpeg4 (lossy) OR create a lossless .mkv (FFV1).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <random>

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"
#define NANOSVGRAST_IMPLEMENTATION
#include "nanosvgrast.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <ros/time.h>
#include <rosbag/bag.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>

// ---------------------------
// Constants / helpers
// ---------------------------
static constexpr double PI = 3.14159265358979323846;

static inline double deg2rad(double d) { return d * PI / 180.0; }
static inline double clamp01(double t) { return std::max(0.0, std::min(1.0, t)); }
static inline double lerp(double a, double b, double t) { return a + (b - a) * t; }

static inline double cosineEase(double t01) {
    t01 = clamp01(t01);
    return 0.5 - 0.5 * std::cos(PI * t01); // C1 continuous
}

static std::string zpad(int v, int width) {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << v;
    return ss.str();
}

// POSIX-ish shell single-quote escaping (for paths with spaces).
static std::string shQuote(const std::string &s) {
    std::string out = "'";
    for (char c: s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

// ---------------------------
// Minimal math
// ---------------------------
struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3() = default;

    Vec3(double X, double Y, double Z) : x(X), y(Y), z(Z) {
    }

    Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }

    Vec3 &operator+=(const Vec3 &o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }
};

static inline double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static inline double norm(const Vec3 &a) { return std::sqrt(dot(a, a)); }

static inline Vec3 normalize(const Vec3 &a) {
    double n = norm(a);
    if (n <= 1e-15) return {0, 0, 0};
    return a * (1.0 / n);
}

struct Mat3 {
    // Column-major: columns c0,c1,c2
    Vec3 c0, c1, c2;

    static Mat3 fromCols(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
        Mat3 R;
        R.c0 = a;
        R.c1 = b;
        R.c2 = c;
        return R;
    }

    Vec3 operator*(const Vec3 &v) const {
        return c0 * v.x + c1 * v.y + c2 * v.z;
    }
};

static inline Mat3 transpose(const Mat3 &R) {
    Vec3 r0{R.c0.x, R.c1.x, R.c2.x};
    Vec3 r1{R.c0.y, R.c1.y, R.c2.y};
    Vec3 r2{R.c0.z, R.c1.z, R.c2.z};
    return Mat3::fromCols(r0, r1, r2);
}

static inline Mat3 mul(const Mat3 &A, const Mat3 &B) {
    return Mat3::fromCols(A * B.c0, A * B.c1, A * B.c2);
}

// Camera-frame rotations: x right, y down, z forward
static inline Mat3 rotX_cam(double a) {
    double c = std::cos(a), s = std::sin(a);
    Vec3 col0{1, 0, 0};
    Vec3 col1{0, c, s};
    Vec3 col2{0, -s, c};
    return Mat3::fromCols(col0, col1, col2);
}

static inline Mat3 rotY_cam(double a) {
    double c = std::cos(a), s = std::sin(a);
    Vec3 col0{c, 0, -s};
    Vec3 col1{0, 1, 0};
    Vec3 col2{s, 0, c};
    return Mat3::fromCols(col0, col1, col2);
}

static inline Mat3 rotZ_cam(double a) {
    double c = std::cos(a), s = std::sin(a);
    Vec3 col0{c, s, 0};
    Vec3 col1{-s, c, 0};
    Vec3 col2{0, 0, 1};
    return Mat3::fromCols(col0, col1, col2);
}

// Look-at returning WORLD->CAM rotation (R_wc), camera axes:
// +Z forward, +X right, +Y down
static Mat3 lookAt_Rwc(const Vec3 &camPos_w,
                       const Vec3 &target_w,
                       const Vec3 &worldUp_w = Vec3{0, 0, 1}) {
    Vec3 z_cam_w = normalize(target_w - camPos_w);
    Vec3 worldDown_w = worldUp_w * (-1.0);

    Vec3 y0 = worldDown_w - z_cam_w * dot(worldDown_w, z_cam_w);
    Vec3 y_cam_w = normalize(y0);
    if (norm(y_cam_w) < 1e-9) {
        y_cam_w = Vec3{0, -1, 0};
        y_cam_w = normalize(y_cam_w - z_cam_w * dot(y_cam_w, z_cam_w));
    }

    Vec3 x_cam_w = normalize(cross(y_cam_w, z_cam_w));
    Mat3 R_cw = Mat3::fromCols(x_cam_w, y_cam_w, z_cam_w);
    return transpose(R_cw);
}

// ---------------------------
// Quaternion utilities (IMU)
// ---------------------------
struct Quat {
    double w = 1, x = 0, y = 0, z = 0; // unit quaternion
};

static inline Quat quatNormalize(const Quat &q) {
    double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n <= 1e-15) return {1, 0, 0, 0};
    return {q.w / n, q.x / n, q.y / n, q.z / n};
}

static inline Quat quatConj(const Quat &q) { return {q.w, -q.x, -q.y, -q.z}; }

static inline Quat quatInv(const Quat &q) { return quatConj(q); }

static inline Quat quatMul(const Quat &a, const Quat &b) {
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

static inline double quatDot(const Quat &a, const Quat &b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

static Quat quatSlerp(Quat q0, Quat q1, double t) {
    t = clamp01(t);
    double d = quatDot(q0, q1);
    if (d < 0.0) {
        q1.w = -q1.w;
        q1.x = -q1.x;
        q1.y = -q1.y;
        q1.z = -q1.z;
        d = -d;
    }

    if (d > 0.9995) {
        Quat q{
            q0.w + t * (q1.w - q0.w),
            q0.x + t * (q1.x - q0.x),
            q0.y + t * (q1.y - q0.y),
            q0.z + t * (q1.z - q0.z)
        };
        return quatNormalize(q);
    }

    double theta0 = std::acos(std::clamp(d, -1.0, 1.0));
    double theta = theta0 * t;
    double s0 = std::sin(theta0);
    double a = std::sin(theta0 - theta) / s0;
    double b = std::sin(theta) / s0;

    Quat q{
        a * q0.w + b * q1.w,
        a * q0.x + b * q1.x,
        a * q0.y + b * q1.y,
        a * q0.z + b * q1.z
    };
    return quatNormalize(q);
}

static Quat quatFromMat3(const Mat3 &R) {
    // R is rotation matrix (column-major in Mat3), row-major elements:
    double r00 = R.c0.x, r01 = R.c1.x, r02 = R.c2.x;
    double r10 = R.c0.y, r11 = R.c1.y, r12 = R.c2.y;
    double r20 = R.c0.z, r21 = R.c1.z, r22 = R.c2.z;

    double tr = r00 + r11 + r22;
    Quat q;

    if (tr > 0.0) {
        double S = std::sqrt(tr + 1.0) * 2.0;
        q.w = 0.25 * S;
        q.x = (r21 - r12) / S;
        q.y = (r02 - r20) / S;
        q.z = (r10 - r01) / S;
    } else if ((r00 > r11) && (r00 > r22)) {
        double S = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
        q.w = (r21 - r12) / S;
        q.x = 0.25 * S;
        q.y = (r01 + r10) / S;
        q.z = (r02 + r20) / S;
    } else if (r11 > r22) {
        double S = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
        q.w = (r02 - r20) / S;
        q.x = (r01 + r10) / S;
        q.y = 0.25 * S;
        q.z = (r12 + r21) / S;
    } else {
        double S = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
        q.w = (r10 - r01) / S;
        q.x = (r02 + r20) / S;
        q.y = (r12 + r21) / S;
        q.z = 0.25 * S;
    }
    return quatNormalize(q);
}

static inline Vec3 rotateVec(const Quat &q_wi, const Vec3 &v_i) {
    Quat v{0, v_i.x, v_i.y, v_i.z};
    Quat qi = quatInv(q_wi);
    Quat r = quatMul(quatMul(q_wi, v), qi);
    return {r.x, r.y, r.z};
}

static inline Vec3 rotateVecInv(const Quat &q_wi, const Vec3 &v_w) {
    Quat v{0, v_w.x, v_w.y, v_w.z};
    Quat qi = quatInv(q_wi);
    Quat r = quatMul(quatMul(qi, v), q_wi);
    return {r.x, r.y, r.z};
}

static Quat quatExpRotvec(const Vec3 &r) {
    double a = norm(r);
    if (a < 1e-12) return quatNormalize({1, 0.5 * r.x, 0.5 * r.y, 0.5 * r.z});
    double half = 0.5 * a;
    double s = std::sin(half) / a;
    return quatNormalize({std::cos(half), r.x * s, r.y * s, r.z * s});
}

static Vec3 quatToRotvec(Quat q_rel) {
    q_rel = quatNormalize(q_rel);
    if (q_rel.w < 0) {
        q_rel.w = -q_rel.w;
        q_rel.x = -q_rel.x;
        q_rel.y = -q_rel.y;
        q_rel.z = -q_rel.z;
    }

    double w = std::clamp(q_rel.w, -1.0, 1.0);
    double angle = 2.0 * std::acos(w);
    double s = std::sqrt(std::max(0.0, 1.0 - w * w));

    if (s < 1e-12 || angle < 1e-12) return {2.0 * q_rel.x, 2.0 * q_rel.y, 2.0 * q_rel.z};

    Vec3 axis{q_rel.x / s, q_rel.y / s, q_rel.z / s};
    return axis * angle;
}

static double quatAngleErrorRad(const Quat &q_hat, const Quat &q_gt) {
    Quat q_err = quatMul(quatInv(q_hat), q_gt);
    q_err = quatNormalize(q_err);
    double w = std::abs(std::clamp(q_err.w, -1.0, 1.0));
    return 2.0 * std::acos(w);
}

// ---------------------------
// Distortion (Brown–Conrady 5-param OpenCV style)
// ---------------------------
struct Dist5 {
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;
};

static inline void distortNormalized(double x, double y, const Dist5 &d, double &xd, double &yd) {
    double r2 = x * x + y * y;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double radial = 1.0 + d.k1 * r2 + d.k2 * r4 + d.k3 * r6;

    double x_tan = 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x);
    double y_tan = d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y;

    xd = x * radial + x_tan;
    yd = y * radial + y_tan;
}

static inline void undistortNormalized(double xd, double yd, const Dist5 &d, double &x, double &y) {
    x = xd;
    y = yd;
    for (int i = 0; i < 6; ++i) {
        double x_est, y_est;
        distortNormalized(x, y, d, x_est, y_est);
        x += (xd - x_est);
        y += (yd - y_est);
    }
}

// ---------------------------
// Texture & SVG rasterization
//   IMPORTANT: No padding/cropping. Texture aspect matches SVG aspect.
// ---------------------------
struct TextureGray {
    int w = 0, h = 0;
    std::vector<uint8_t> pix; // grayscale [0..255]
};

struct SvgTexture {
    TextureGray tex;
    double svgW_px = 0;
    double svgH_px = 0;
};

static SvgTexture loadSvgAsGrayTexture_NoPadding(const std::string &svgPath, int maxDim) {
    NSVGimage *svg = nsvgParseFromFile(svgPath.c_str(), "px", 96.0f);
    if (!svg) throw std::runtime_error("Failed to parse SVG: " + svgPath);

    double w = svg->width;
    double h = svg->height;
    if (w <= 0 || h <= 0) {
        nsvgDelete(svg);
        throw std::runtime_error("SVG has invalid dimensions: " + svgPath);
    }

    double scale = double(maxDim) / std::max(w, h);
    int texW = std::max(1, int(std::lround(w * scale)));
    int texH = std::max(1, int(std::lround(h * scale)));

    std::vector<uint8_t> rgba(size_t(texW) * size_t(texH) * 4, 0);

    NSVGrasterizer *rast = nsvgCreateRasterizer();
    if (!rast) {
        nsvgDelete(svg);
        throw std::runtime_error("Failed to create NanoSVG rasterizer");
    }

    // Render SVG to exactly fill texW x texH (no centering/padding offsets)
    nsvgRasterize(rast, svg, 0.0f, 0.0f, float(scale),
                  rgba.data(), texW, texH, texW * 4);

    nsvgDeleteRasterizer(rast);

    TextureGray out;
    out.w = texW;
    out.h = texH;
    out.pix.assign(size_t(texW) * size_t(texH), 255);

    // Alpha blend over white -> grayscale
    for (int y = 0; y < texH; ++y) {
        for (int x = 0; x < texW; ++x) {
            size_t idx = (size_t(y) * texW + size_t(x)) * 4;
            uint8_t r = rgba[idx + 0], g = rgba[idx + 1], b = rgba[idx + 2], a = rgba[idx + 3];
            double alpha = double(a) / 255.0;
            double rr = alpha * double(r) + (1.0 - alpha) * 255.0;
            double gg = alpha * double(g) + (1.0 - alpha) * 255.0;
            double bb = alpha * double(b) + (1.0 - alpha) * 255.0;
            double gray = 0.299 * rr + 0.587 * gg + 0.114 * bb;
            out.pix[size_t(y) * texW + size_t(x)] = uint8_t(std::clamp(int(std::lround(gray)), 0, 255));
        }
    }

    SvgTexture st;
    st.tex = std::move(out);
    st.svgW_px = w;
    st.svgH_px = h;

    nsvgDelete(svg);
    return st;
}

static inline uint8_t sampleBilinear(const TextureGray &t, double x, double y) {
    x = std::clamp(x, 0.0, double(t.w - 1));
    y = std::clamp(y, 0.0, double(t.h - 1));

    int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    int x1 = std::min(x0 + 1, t.w - 1);
    int y1 = std::min(y0 + 1, t.h - 1);
    double ax = x - x0;
    double ay = y - y0;

    auto I = [&](int xx, int yy)-> double {
        return double(t.pix[size_t(yy) * t.w + size_t(xx)]);
    };

    double i00 = I(x0, y0), i10 = I(x1, y0), i01 = I(x0, y1), i11 = I(x1, y1);
    double i0 = (1.0 - ax) * i00 + ax * i10;
    double i1 = (1.0 - ax) * i01 + ax * i11;
    double v = (1.0 - ay) * i0 + ay * i1;
    return uint8_t(std::clamp(int(std::lround(v)), 0, 255));
}

// ---------------------------
// Plane pattern in 3D
// ---------------------------
struct PlanePattern {
    TextureGray tex;
    double width_m = 0, height_m = 0;
    Vec3 origin_w;
    Vec3 udir_w;
    Vec3 vdir_w;
    Vec3 normal_w;

    bool flipU = false;
    bool flipV = true; // common because SVG y is down
};

// ---------------------------
// Camera model
// ---------------------------
struct Camera {
    int W = 0, H = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    Dist5 dist;

    Vec3 C_w;
    Mat3 R_wc; // world->camera
};

// ---------------------------
// Precompute undistorted rays (fast rendering)
// ---------------------------
static std::vector<Vec3> precomputeRaysCam(const Camera &cam) {
    std::vector<Vec3> rays(size_t(cam.W) * size_t(cam.H));
    for (int v = 0; v < cam.H; ++v) {
        for (int u = 0; u < cam.W; ++u) {
            double xd = (double(u) - cam.cx) / cam.fx;
            double yd = (double(v) - cam.cy) / cam.fy;

            double x, y;
            undistortNormalized(xd, yd, cam.dist, x, y);

            rays[size_t(v) * cam.W + size_t(u)] = normalize(Vec3{x, y, 1.0});
        }
    }
    return rays;
}

// ---------------------------
// Rendering (ray-plane intersection + texture sampling)
// ---------------------------
static std::vector<uint8_t> renderFrameGray(const Camera &cam,
                                            const std::vector<Vec3> &raysCam,
                                            const std::vector<PlanePattern> &planes,
                                            uint8_t background = 255) {
    std::vector<uint8_t> img(size_t(cam.W) * size_t(cam.H), background);
    Mat3 R_cw = transpose(cam.R_wc);

    for (int v = 0; v < cam.H; ++v) {
        for (int u = 0; u < cam.W; ++u) {
            const Vec3 &d_cam = raysCam[size_t(v) * cam.W + size_t(u)];
            Vec3 d_w = R_cw * d_cam;

            bool hit = false;
            double bestT = 1e300;
            const PlanePattern *best = nullptr;
            double bestU = 0, bestV = 0;

            for (const auto &pl: planes) {
                double denom = dot(pl.normal_w, d_w);
                if (std::abs(denom) < 1e-12) continue;

                double t = dot(pl.normal_w, (pl.origin_w - cam.C_w)) / denom;
                if (t <= 0.0) continue;

                Vec3 P = cam.C_w + d_w * t;
                Vec3 r = P - pl.origin_w;

                double uu = dot(r, pl.udir_w);
                double vv = dot(r, pl.vdir_w);
                if (uu < 0.0 || uu > pl.width_m || vv < 0.0 || vv > pl.height_m) continue;

                if (t < bestT) {
                    bestT = t;
                    best = &pl;
                    bestU = uu;
                    bestV = vv;
                    hit = true;
                }
            }

            if (!hit || !best) continue;

            double uu = best->flipU ? (best->width_m - bestU) : bestU;
            double vv = best->flipV ? (best->height_m - bestV) : bestV;

            double tx = (uu / best->width_m) * double(best->tex.w - 1);
            double ty = (vv / best->height_m) * double(best->tex.h - 1);

            img[size_t(v) * cam.W + size_t(u)] = sampleBilinear(best->tex, tx, ty);
        }
    }
    return img;
}

// ---------------------------
// PNG writing (lossless). "Maximum quality" == lossless.
// We set maximum compression level (9).
// ---------------------------
struct PngMemWriter {
    std::vector<uint8_t> buf;
};

static void stbWriteToVector(void *context, void *data, int size) {
    auto *w = reinterpret_cast<PngMemWriter *>(context);
    uint8_t *p = reinterpret_cast<uint8_t *>(data);
    w->buf.insert(w->buf.end(), p, p + size);
}

static std::vector<uint8_t> encodeGrayPNGToMemory(int W, int H, const std::vector<uint8_t> &gray) {
    PngMemWriter w;
    w.buf.reserve(size_t(W) * size_t(H) / 2);

    int ok = stbi_write_png_to_func(stbWriteToVector, &w, W, H, 1, gray.data(), W);
    if (!ok) throw std::runtime_error("Failed to encode PNG to memory");
    return w.buf;
}

static void writeGrayPNGFile(const std::string &path, int W, int H, const std::vector<uint8_t> &gray) {
    int ok = stbi_write_png(path.c_str(), W, H, 1, gray.data(), W);
    if (!ok) throw std::runtime_error("Failed to write PNG: " + path);
}

// ---------------------------
// Projection for auto-fit checks
// ---------------------------
static bool projectPoint(const Camera &cam, const Vec3 &Pw, double &u, double &v) {
    Vec3 Pc = cam.R_wc * (Pw - cam.C_w);
    if (Pc.z <= 1e-9) return false;

    double x = Pc.x / Pc.z;
    double y = Pc.y / Pc.z;

    double xd, yd;
    distortNormalized(x, y, cam.dist, xd, yd);

    u = cam.fx * xd + cam.cx;
    v = cam.fy * yd + cam.cy;
    return true;
}

static bool allPointsInView(const Camera &cam,
                            const std::vector<Vec3> &pts_w,
                            double marginPx) {
    double minU = 1e300, maxU = -1e300;
    double minV = 1e300, maxV = -1e300;

    for (const auto &P: pts_w) {
        double u, v;
        if (!projectPoint(cam, P, u, v)) return false;
        minU = std::min(minU, u);
        maxU = std::max(maxU, u);
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
    }

    double u0 = marginPx;
    double u1 = double(cam.W - 1) - marginPx;
    double v0 = marginPx;
    double v1 = double(cam.H - 1) - marginPx;

    return (minU >= u0 && maxU <= u1 && minV >= v0 && maxV <= v1);
}

static std::vector<Vec3> makeBoardCheckPoints(double wXY, double hXY,
                                              double wYZ, double hYZ,
                                              double wXZ, double hXZ,
                                              int gridN = 7) {
    std::vector<Vec3> pts;
    pts.reserve(size_t(3) * size_t(gridN) * size_t(gridN));

    auto gridVals = [&](int n)-> std::vector<double> {
        std::vector<double> t;
        t.reserve(size_t(n));
        if (n <= 1) {
            t.push_back(0.0);
            return t;
        }
        for (int i = 0; i < n; ++i) t.push_back(double(i) / double(n - 1));
        return t;
    };
    auto G = gridVals(gridN);

    // XY (z=0)
    for (double a: G) for (double b: G) pts.emplace_back(a * wXY, b * hXY, 0.0);
    // YZ (x=0)
    for (double a: G) for (double b: G) pts.emplace_back(0.0, a * wYZ, b * hYZ);
    // XZ (y=0)
    for (double a: G) for (double b: G) pts.emplace_back(a * wXZ, 0.0, b * hXZ);

    return pts;
}

// ---------------------------
// Trajectory (continuous translation + base attitude)
// ---------------------------
struct TrajectoryConfig {
    Vec3 center_w;
    double radiusBase = 1.0;
    double radiusAmpFrac = 0.0; // disabled: keep constant distance (no zoom)
    double radiusFreq = 1.0;

    double phi0 = deg2rad(25.0);
    double phi1 = deg2rad(70.0);
    double th0 = deg2rad(18.0);
    double th1 = deg2rad(55.0);

    double wobblePhiAmp = deg2rad(2.0);
    double wobbleThAmp = deg2rad(2.0);
    double wobbleFreq = 2.0;
    double wobblePhase = deg2rad(90.0);

    Vec3 truckAmp_w{0, 0, 0};

    // Deliberate off-centering angles (camera frame)
    double yawAmp = deg2rad(12.0);
    double pitchAmp = deg2rad(8.0);
    double rollAmp = deg2rad(4.0);

    double yawFreq = 1.0;
    double pitchFreq = 1.2;
    double rollFreq = 0.7;

    double yawPhase = 0.0;
    double pitchPhase = deg2rad(60.0);
    double rollPhase = deg2rad(20.0);
};

static Vec3 rigNominalPosition(const TrajectoryConfig &cfg, double t01) {
    // Highly smooth (C∞) motion with CONSTANT radius about cfg.center_w.
    // This explicitly disables any time-varying "zoom" due to changing distance.
    //
    // Position moves on a spherical patch in the +octant.
    double s = cosineEase(t01);

    double phi = lerp(cfg.phi0, cfg.phi1, s)
                 + cfg.wobblePhiAmp * std::sin(2.0 * PI * cfg.wobbleFreq * t01);
    double theta = lerp(cfg.th0, cfg.th1, s)
                   + cfg.wobbleThAmp * std::sin(2.0 * PI * cfg.wobbleFreq * t01 + cfg.wobblePhase);

    double r = cfg.radiusBase; // FIXED radius => constant scale (no dolly/zoom)

    double cth = std::cos(theta);
    Vec3 dir{
        cth * std::cos(phi),
        cth * std::sin(phi),
        std::sin(theta)
    };

    Vec3 pos = cfg.center_w + dir * r;

    // Keep in +octant (in front of trihedral planes)
    pos.x = std::max(pos.x, 1e-6);
    pos.y = std::max(pos.y, 1e-6);
    pos.z = std::max(pos.z, 1e-6);
    return pos;
}

static void desiredOffsets(const TrajectoryConfig &cfg, double t01,
                           double &yaw, double &pitch, double &roll) {
    yaw = cfg.yawAmp * std::sin(2.0 * PI * cfg.yawFreq * t01 + cfg.yawPhase);
    pitch = cfg.pitchAmp * std::sin(2.0 * PI * cfg.pitchFreq * t01 + cfg.pitchPhase);
    roll = cfg.rollAmp * std::sin(2.0 * PI * cfg.rollFreq * t01 + cfg.rollPhase);
}

// ---------------------------
// Ground-truth pose structs + interpolation for IMU sampling
// ---------------------------
struct PoseGT {
    double t = 0;
    Vec3 p_w;
    Quat q_wi; // IMU->WORLD
};

static Vec3 hermitePos(const Vec3 &p0, const Vec3 &v0,
                       const Vec3 &p1, const Vec3 &v1,
                       double dt, double s) {
    double s2 = s * s;
    double s3 = s2 * s;
    double h00 = 2 * s3 - 3 * s2 + 1;
    double h10 = s3 - 2 * s2 + s;
    double h01 = -2 * s3 + 3 * s2;
    double h11 = s3 - s2;
    return p0 * h00 + v0 * (h10 * dt) + p1 * h01 + v1 * (h11 * dt);
}

static PoseGT interpolatePose(const std::vector<PoseGT> &frames,
                              const std::vector<Vec3> &velFrames,
                              double t) {
    if (frames.empty()) return {};
    if (t <= frames.front().t) return frames.front();
    if (t >= frames.back().t) return frames.back();

    size_t i = 0;
    while (i + 1 < frames.size() && !(frames[i].t <= t && t <= frames[i + 1].t)) ++i;
    if (i + 1 >= frames.size()) return frames.back();

    const PoseGT &A = frames[i];
    const PoseGT &B = frames[i + 1];
    double dt = B.t - A.t;
    double s = (dt > 0) ? (t - A.t) / dt : 0.0;
    s = clamp01(s);

    PoseGT out;
    out.t = t;
    out.q_wi = quatSlerp(A.q_wi, B.q_wi, s);

    Vec3 v0 = velFrames[i];
    Vec3 v1 = velFrames[i + 1];
    out.p_w = hermitePos(A.p_w, v0, B.p_w, v1, dt, s);
    return out;
}

// ---------------------------
// Synthetic IMU generation
// ---------------------------
struct ImuSample {
    double t = 0;
    Vec3 gyro_i; // rad/s in IMU frame
    Vec3 accel_i; // m/s^2 specific force in IMU frame
    PoseGT gt;
};

struct ImuNoiseConfig {
    bool enable = false;
    Vec3 gyroBias{0, 0, 0};
    Vec3 accelBias{0, 0, 0};
    double gyroNoiseStd = 0.0;
    double accelNoiseStd = 0.0;
    uint32_t seed = 12345;
};

static std::vector<ImuSample> generateImu(const std::vector<PoseGT> &frames,
                                          double imuHz,
                                          const ImuNoiseConfig &noiseCfg,
                                          std::vector<PoseGT> *outGtAtImu = nullptr) {
    if (frames.size() < 2) return {};

    double t0 = frames.front().t;
    double t1 = frames.back().t;
    double dt = 1.0 / imuHz;

    // Estimate velocities at frame timestamps
    std::vector<Vec3> v(frames.size(), Vec3{0, 0, 0});
    for (size_t i = 0; i < frames.size(); ++i) {
        if (i == 0) {
            double d = frames[1].t - frames[0].t;
            if (d > 0) v[i] = (frames[1].p_w - frames[0].p_w) * (1.0 / d);
        } else if (i + 1 == frames.size()) {
            double d = frames[i].t - frames[i - 1].t;
            if (d > 0) v[i] = (frames[i].p_w - frames[i - 1].p_w) * (1.0 / d);
        } else {
            double d = frames[i + 1].t - frames[i - 1].t;
            if (d > 0) v[i] = (frames[i + 1].p_w - frames[i - 1].p_w) * (1.0 / d);
        }
    }

    // Sample GT at IMU timestamps
    std::vector<PoseGT> gt;
    for (double t = t0; t <= t1 + 1e-12; t += dt) gt.push_back(interpolatePose(frames, v, t));
    if (outGtAtImu) *outGtAtImu = gt;

    size_t N = gt.size();
    std::vector<Vec3> pos(N);
    std::vector<Quat> q(N);
    for (size_t k = 0; k < N; ++k) {
        pos[k] = gt[k].p_w;
        q[k] = gt[k].q_wi;
    }

    std::mt19937 rng(noiseCfg.seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    auto addNoiseVec = [&](const Vec3 &v0, double stddev)-> Vec3 {
        if (!noiseCfg.enable || stddev <= 0.0) return v0;
        return {v0.x + stddev * N01(rng), v0.y + stddev * N01(rng), v0.z + stddev * N01(rng)};
    };

    const Vec3 g_w{0, 0, -9.81};

    // Approx world acceleration by second differences
    std::vector<Vec3> acc_w(N, Vec3{0, 0, 0});
    for (size_t k = 0; k < N; ++k) {
        if (k > 0 && k + 1 < N) {
            acc_w[k] = (pos[k + 1] - pos[k] * 2.0 + pos[k - 1]) * (1.0 / (dt * dt));
        } else if (N >= 3) {
            if (k == 0) acc_w[k] = (pos[2] - pos[1] * 2.0 + pos[0]) * (1.0 / (dt * dt));
            else acc_w[k] = (pos[N - 1] - pos[N - 2] * 2.0 + pos[N - 3]) * (1.0 / (dt * dt));
        }
    }

    std::vector<ImuSample> out(N);
    for (size_t k = 0; k < N; ++k) {
        out[k].t = gt[k].t;
        out[k].gt = gt[k];

        // Gyro: from relative quaternion q_k^{-1} q_{k+1}
        Vec3 omega{0, 0, 0};
        if (k + 1 < N) {
            Quat q_rel = quatMul(quatInv(q[k]), q[k + 1]);
            Vec3 rotvec = quatToRotvec(q_rel);
            omega = rotvec * (1.0 / dt);
        } else if (k > 0) {
            omega = out[k - 1].gyro_i;
        }

        // Accel specific force: f_i = R_iw*(a_w - g_w)
        Vec3 f_i = rotateVecInv(q[k], acc_w[k] - g_w);

        // Bias + noise
        omega = omega + noiseCfg.gyroBias;
        f_i = f_i + noiseCfg.accelBias;
        omega = addNoiseVec(omega, noiseCfg.gyroNoiseStd);
        f_i = addNoiseVec(f_i, noiseCfg.accelNoiseStd);

        out[k].gyro_i = omega;
        out[k].accel_i = f_i;
    }

    return out;
}

// ---------------------------
// IMU integration + verification
// ---------------------------
struct ImuIntegrationResult {
    double rmsAngleDeg = 0.0;
    double rmsPos = 0.0;
};

static ImuIntegrationResult integrateImuAndCompare(const std::vector<ImuSample> &imu,
                                                   double imuHz,
                                                   const std::string &outCsvPath) {
    ImuIntegrationResult res;
    if (imu.size() < 3) return res;

    double dt = 1.0 / imuHz;
    const Vec3 g_w{0, 0, -9.81};

    Quat q_hat = imu[0].gt.q_wi;
    Vec3 p_hat = imu[0].gt.p_w;
    Vec3 v_hat = (imu[1].gt.p_w - imu[0].gt.p_w) * (1.0 / dt);

    std::ofstream csv(outCsvPath);
    csv << "t,angle_err_deg,pos_err_m,px_hat,py_hat,pz_hat,px_gt,py_gt,pz_gt\n";

    double sumAng2 = 0.0, sumPos2 = 0.0;
    int count = 0;

    for (size_t k = 0; k + 1 < imu.size(); ++k) {
        // q_{k+1} = q_k ⊗ Exp(omega*dt)
        Quat dq = quatExpRotvec(imu[k].gyro_i * dt);
        q_hat = quatNormalize(quatMul(q_hat, dq));

        // a_w = R_wi f_i + g
        Vec3 a_w_hat = rotateVec(q_hat, imu[k].accel_i) + g_w;

        p_hat += v_hat * dt + a_w_hat * (0.5 * dt * dt);
        v_hat += a_w_hat * dt;

        const PoseGT &gt = imu[k + 1].gt;
        double angDeg = quatAngleErrorRad(q_hat, gt.q_wi) * 180.0 / PI;
        double posErr = norm(p_hat - gt.p_w);

        sumAng2 += angDeg * angDeg;
        sumPos2 += posErr * posErr;
        count++;

        csv << gt.t << "," << angDeg << "," << posErr << ","
                << p_hat.x << "," << p_hat.y << "," << p_hat.z << ","
                << gt.p_w.x << "," << gt.p_w.y << "," << gt.p_w.z << "\n";
    }

    if (count > 0) {
        res.rmsAngleDeg = std::sqrt(sumAng2 / count);
        res.rmsPos = std::sqrt(sumPos2 / count);
    }
    return res;
}

// ---------------------------
// MP4 encoding via ffmpeg (NO -preset; robust fallbacks)
// ---------------------------
static bool runCommand(const std::string &cmd) {
    int ret = std::system(cmd.c_str());
    return (ret == 0);
}

static void encodeMp4WithFfmpeg_NoPreset(const std::string &outDir, int camIdx, int fps) {
    const char *env = std::getenv("FFMPEG_BIN");
    std::string ffmpeg = env ? std::string(env) : std::string("ffmpeg");

    std::string camDir = outDir + "/cam" + std::to_string(camIdx);
    std::string inPattern = camDir + "/frame_%05d.png";
    std::string mp4Out = outDir + "/cam" + std::to_string(camIdx) + ".mp4";
    std::string mkvFallback = outDir + "/cam" + std::to_string(camIdx) + ".mkv";

    // Use older-compatible "-r" before -i, and "-vcodec" form.
    // Try lossless H.264 first. If libx264 isn't available, fall back.
    std::vector<std::string> cmds;

    // 1) Lossless x264 using CRF=0 (no preset)
    cmds.push_back(
        shQuote(ffmpeg) + " -y -r " + std::to_string(fps) +
        " -i " + shQuote(inPattern) +
        " -vcodec libx264 -crf 0 -pix_fmt yuv420p " + shQuote(mp4Out)
    );

    // 2) Lossless x264 using QP=0 (if CRF unsupported)
    cmds.push_back(
        shQuote(ffmpeg) + " -y -r " + std::to_string(fps) +
        " -i " + shQuote(inPattern) +
        " -vcodec libx264 -qp 0 -pix_fmt yuv420p " + shQuote(mp4Out)
    );

    // 3) High-quality MPEG-4 (lossy fallback, but widely available)
    cmds.push_back(
        shQuote(ffmpeg) + " -y -r " + std::to_string(fps) +
        " -i " + shQuote(inPattern) +
        " -vcodec mpeg4 -q:v 1 " + shQuote(mp4Out)
    );

    // 4) True lossless fallback to MKV (FFV1), if MP4 encoders unavailable
    cmds.push_back(
        shQuote(ffmpeg) + " -y -r " + std::to_string(fps) +
        " -i " + shQuote(inPattern) +
        " -vcodec ffv1 -level 3 " + shQuote(mkvFallback)
    );

    std::cout << "Encoding video for cam" << camIdx << " (no scaling, no preset)...\n";
    for (size_t i = 0; i < cmds.size(); ++i) {
        if (runCommand(cmds[i])) {
            if (i <= 2) std::cout << "  wrote: " << mp4Out << "\n";
            else std::cout << "  wrote (fallback): " << mkvFallback << "\n";
            return;
        }
    }

    std::cout << "  [WARN] All ffmpeg encoding attempts failed.\n";
    std::cout << "  Try running manually (example):\n    "
            << ffmpeg << " -y -r " << fps << " -i " << inPattern
            << " -vcodec libx264 -crf 0 -pix_fmt yuv420p " << mp4Out << "\n";
    std::cout << "  Also check available encoders with:\n    "
            << ffmpeg << " -encoders\n";
}

// ---------------------------
// Main
// ---------------------------
int main(int argc, char **argv) {
    try {
        if (argc < 5) {
            std::cerr << "Usage:\n  " << argv[0]
                    << " pattern_xy.svg pattern_yz.svg pattern_xz.svg output_dir/\n";
            return 1;
        }

        ros::Time::init();

        // Convert sim time (seconds) -> valid ROS time for bags
        auto toRosStamp = [](double t_sec) -> ros::Time {
            // Guarantees stamp >= ros::TIME_MIN even when t_sec == 0
            return ros::TIME_MIN + ros::Duration(t_sec);
        };

        std::string svgXY = argv[1];
        std::string svgYZ = argv[2];
        std::string svgXZ = argv[3];
        std::string outDir = argv[4];

        std::filesystem::create_directories(outDir);

        // -----------------------
        // Tunables
        // -----------------------
        const int TEX_MAXDIM = 2048; // texture resolution limit per SVG (keeps aspect ratio)

        const int FPS = 30;
        const int NUM_FRAMES = 240; // 80 seconds @ 30fps

        // Resolution is fixed => intrinsics hold.
        const int IMG_W = 1280;
        const int IMG_H = 800;

        // IMPORTANT: to avoid any encoder-driven resizing with yuv420p, keep even dimensions.
        if ((IMG_W % 2) || (IMG_H % 2)) {
            std::cerr << "[WARN] IMG_W/IMG_H are odd. Some H.264 encoders may force padding/scaling.\n"
                    << "       Use even IMG_W/IMG_H to guarantee no resizing in MP4.\n";
        }

        const double MARGIN_PX = 20.0;
        const double IMU_HZ = 200.0;

        // Physical scale of SVG:
        constexpr double PX_TO_METERS = (25.4e-3 / 96.0);
        const double PRINT_SCALE = 1.0; // <-- adjust if needed
        const double SVG_TO_METERS = PX_TO_METERS * PRINT_SCALE;

        // PNG: max compression (still lossless)
        stbi_write_png_compression_level = 9;

        // -----------------------
        // Camera intrinsics + distortion (examples; replace with your real intrinsics)
        // Intrinsics are CONSTANT across time => no scaling.
        // -----------------------
        Dist5 dist;
        dist.k1 = -0.22;
        dist.k2 = 0.08;
        dist.p1 = 0.0005;
        dist.p2 = -0.0003;
        dist.k3 = -0.015;

        auto makeCam = [&](double fx, double fy, double cx, double cy)-> Camera {
            Camera c;
            c.W = IMG_W;
            c.H = IMG_H;
            c.fx = fx;
            c.fy = fy;
            c.cx = cx;
            c.cy = cy;
            c.dist = dist;
            return c;
        };

        std::vector<Camera> cams(4);
        cams[0] = makeCam(1100, 1100, IMG_W * 0.50, IMG_H * 0.50);
        cams[1] = makeCam(1080, 1095, IMG_W * 0.52, IMG_H * 0.48);
        cams[2] = makeCam(1120, 1090, IMG_W * 0.49, IMG_H * 0.51);
        cams[3] = makeCam(1090, 1110, IMG_W * 0.50, IMG_H * 0.50);

        // Store intrinsics to assert they never change (intrinsics hold)
        struct Intr {
            double fx, fy, cx, cy;
        };
        std::vector<Intr> intr0(4);
        for (int i = 0; i < 4; ++i) intr0[i] = {cams[i].fx, cams[i].fy, cams[i].cx, cams[i].cy};

        // Rigid 4-camera rig offsets in rig/cam coords (x right, y down, z forward).
        const double BASELINE = 0.16;
        std::vector<Vec3> camOffsets_r = {
            Vec3{-BASELINE * 0.5, -BASELINE * 0.5, 0.0},
            Vec3{+BASELINE * 0.5, -BASELINE * 0.5, 0.0},
            Vec3{-BASELINE * 0.5, +BASELINE * 0.5, 0.0},
            Vec3{+BASELINE * 0.5, +BASELINE * 0.5, 0.0}
        };

        // -----------------------
        // Load SVG textures (no padding) + get SVG sizes in px
        // -----------------------
        SvgTexture stXY = loadSvgAsGrayTexture_NoPadding(svgXY, TEX_MAXDIM);
        SvgTexture stYZ = loadSvgAsGrayTexture_NoPadding(svgYZ, TEX_MAXDIM);
        SvgTexture stXZ = loadSvgAsGrayTexture_NoPadding(svgXZ, TEX_MAXDIM);

        double wXY_m = stXY.svgW_px * SVG_TO_METERS;
        double hXY_m = stXY.svgH_px * SVG_TO_METERS;
        double wYZ_m = stYZ.svgW_px * SVG_TO_METERS;
        double hYZ_m = stYZ.svgH_px * SVG_TO_METERS;
        double wXZ_m = stXZ.svgW_px * SVG_TO_METERS;
        double hXZ_m = stXZ.svgH_px * SVG_TO_METERS;

        double xmax = std::max(wXY_m, wXZ_m);
        double ymax = std::max(hXY_m, wYZ_m);
        double zmax = std::max(hYZ_m, hXZ_m);
        double maxDim = std::max({xmax, ymax, zmax});

        // -----------------------
        // Build planes
        // -----------------------
        std::vector<PlanePattern> planes;
        planes.reserve(3);

        PlanePattern pXY;
        pXY.tex = std::move(stXY.tex);
        pXY.width_m = wXY_m;
        pXY.height_m = hXY_m;
        pXY.origin_w = Vec3{0, 0, 0};
        pXY.udir_w = Vec3{1, 0, 0};
        pXY.vdir_w = Vec3{0, 1, 0};
        pXY.normal_w = normalize(cross(pXY.udir_w, pXY.vdir_w)); // +z
        pXY.flipV = true;
        planes.push_back(std::move(pXY));

        PlanePattern pYZ;
        pYZ.tex = std::move(stYZ.tex);
        pYZ.width_m = wYZ_m;
        pYZ.height_m = hYZ_m;
        pYZ.origin_w = Vec3{0, 0, 0};
        pYZ.udir_w = Vec3{0, 1, 0};
        pYZ.vdir_w = Vec3{0, 0, 1};
        pYZ.normal_w = normalize(cross(pYZ.udir_w, pYZ.vdir_w)); // +x
        pYZ.flipV = true;
        planes.push_back(std::move(pYZ));

        PlanePattern pXZ;
        pXZ.tex = std::move(stXZ.tex);
        pXZ.width_m = wXZ_m;
        pXZ.height_m = hXZ_m;
        pXZ.origin_w = Vec3{0, 0, 0};
        pXZ.udir_w = Vec3{1, 0, 0};
        pXZ.vdir_w = Vec3{0, 0, 1};
        pXZ.normal_w = normalize(cross(pXZ.udir_w, pXZ.vdir_w)); // (0,-1,0) sign not critical
        pXZ.flipV = true;
        planes.push_back(std::move(pXZ));

        // -----------------------
        // Auto-fit check points: require FULL trihedral in view
        // -----------------------
        std::vector<Vec3> checkPts = makeBoardCheckPoints(wXY_m, hXY_m, wYZ_m, hYZ_m, wXZ_m, hXZ_m, 7);

        // -----------------------
        // Trajectory config
        // -----------------------
        TrajectoryConfig traj;
        traj.center_w = Vec3{0.35 * xmax, 0.35 * ymax, 0.35 * zmax};
        traj.radiusBase = 2.8 * maxDim;
        // Disable any extra "truck" translation that changes distance (prevents zoom-like scale changes)
        traj.truckAmp_w = Vec3{0.0, 0.0, 0.0};
        // Disable radial modulation: keep constant distance to target
        traj.radiusAmpFrac = 0.0;


        traj.phi0 = deg2rad(25);
        traj.phi1 = deg2rad(70);
        traj.th0 = deg2rad(18);
        traj.th1 = deg2rad(55);

        // -----------------------
        // Global safety tuning (ONE-TIME, no per-frame zoom):
        // Ensure all 4 cameras keep all 3 planes in view for the whole sequence
        // without changing distance scale over time.
        // -----------------------
        auto sequenceFeasible = [&](const TrajectoryConfig &T) -> bool {
            Vec3 worldUpLocal{0, 0, 1};
            for (int f = 0; f < NUM_FRAMES; ++f) {
                double t01 = (NUM_FRAMES <= 1) ? 0.0 : double(f) / double(NUM_FRAMES - 1);
                Vec3 rigPos_w = rigNominalPosition(T, t01);

                double yawDes, pitchDes, rollDes;
                desiredOffsets(T, t01, yawDes, pitchDes, rollDes);

                Mat3 R_wr_base = lookAt_Rwc(rigPos_w, T.center_w, worldUpLocal);
                Mat3 R_off = mul(rotZ_cam(rollDes),
                                 mul(rotY_cam(yawDes),
                                     rotX_cam(pitchDes)));
                Mat3 R_wr = mul(R_off, R_wr_base);
                Mat3 R_rw = transpose(R_wr);

                for (int i = 0; i < 4; ++i) {
                    Camera c = cams[i]; // copy intrinsics/dist only
                    c.R_wc = R_wr;
                    c.C_w = rigPos_w + (R_rw * camOffsets_r[i]);
                    if (!allPointsInView(c, checkPts, MARGIN_PX)) return false;
                }
            }
            return true;
        };

        int tuneIters = 0;
        for (; tuneIters < 80; ++tuneIters) {
            if (sequenceFeasible(traj)) break;

            // Increase radius a bit (global constant) and reduce offset wobbles a bit.
            traj.radiusBase *= 1.02;
            traj.yawAmp *= 0.98;
            traj.pitchAmp *= 0.98;
            traj.rollAmp *= 0.98;
            traj.wobblePhiAmp *= 0.98;
            traj.wobbleThAmp *= 0.98;
        }

        if (tuneIters >= 80) {
            std::cerr << "[WARN] Global tuning could not guarantee all 3 planes in view for all frames.\n"
                    << "       Consider increasing traj.radiusBase manually.\n";
        } else {
            std::cout << "Global tuning OK in " << tuneIters
                    << " iterations. Fixed radiusBase=" << traj.radiusBase << " m\n";
        }


        // -----------------------
        // Precompute rays
        // -----------------------
        std::vector<std::vector<Vec3> > rays(4);
        for (int i = 0; i < 4; ++i) {
            std::cout << "Precomputing rays for cam " << i << "...\n";
            rays[i] = precomputeRaysCam(cams[i]);
        }

        // -----------------------
        // Output folders + CSVs
        // -----------------------
        for (int i = 0; i < 4; ++i) {
            std::filesystem::create_directories(outDir + "/cam" + std::to_string(i));
        }

        std::vector<std::ofstream> poseCsv(4);
        for (int i = 0; i < 4; ++i) {
            poseCsv[i].open(outDir + "/cam" + std::to_string(i) + "_poses.csv");
            poseCsv[i] << "frame,time_s,Cx,Cy,Cz,"
                    << "R00,R01,R02,R10,R11,R12,R20,R21,R22\n";
        }

        std::ofstream autoCsv(outDir + "/autofit.csv");
        autoCsv << "frame,time_s,distScale,offScale\n";

        std::ofstream rigFramesCsv(outDir + "/rig_frames_gt.csv");
        rigFramesCsv << "frame,time_s,px,py,pz,qw,qx,qy,qz\n";

        // -----------------------
        // Optional ROS1 bag
        // -----------------------

        std::string bagPath = outDir + "/sim.bag";
        rosbag::Bag bag;
        bag.open(bagPath, rosbag::bagmode::Write);
        std::cout << "ROS1 bag open: " << bagPath << "\n";

        // Smooth motion mode: no per-frame auto-fit (no time-varying zoom).
        Vec3 worldUp{0, 0, 1};

        std::vector<PoseGT> rigFramesGT;
        rigFramesGT.reserve(NUM_FRAMES);

        // -----------------------
        // Render sequence
        // -----------------------
        for (int f = 0; f < NUM_FRAMES; ++f) {
            double t01 = (NUM_FRAMES <= 1) ? 0.0 : double(f) / double(NUM_FRAMES - 1);
            double time_s = double(f) / double(FPS);

            // Intrinsics must hold: assert unchanged
            for (int i = 0; i < 4; ++i) {
                if (cams[i].fx != intr0[i].fx || cams[i].fy != intr0[i].fy ||
                    cams[i].cx != intr0[i].cx || cams[i].cy != intr0[i].cy) {
                    throw std::runtime_error("INTERNAL ERROR: Intrinsics changed (image scaling not allowed).");
                }
            }

            // Smooth motion pose: fixed-scale trajectory (no per-frame zoom / auto-fit).
            Vec3 rigPos_w = rigNominalPosition(traj, t01);

            double yawDes, pitchDes, rollDes;
            desiredOffsets(traj, t01, yawDes, pitchDes, rollDes);

            Mat3 R_wr_base = lookAt_Rwc(rigPos_w, traj.center_w, worldUp);

            Mat3 R_off = mul(rotZ_cam(rollDes),
                             mul(rotY_cam(yawDes),
                                 rotX_cam(pitchDes)));

            Mat3 R_wr_final = mul(R_off, R_wr_base);
            Mat3 R_rw_final = transpose(R_wr_final);

            // Parallel cameras on a rig
            for (int i = 0; i < 4; ++i) {
                Vec3 offset_w = R_rw_final * camOffsets_r[i];
                cams[i].C_w = rigPos_w + offset_w;
                cams[i].R_wc = R_wr_final;
            }

            bool ok = true;
            for (int i = 0; i < 4; ++i) {
                if (!allPointsInView(cams[i], checkPts, MARGIN_PX)) {
                    ok = false;
                    break;
                }
            }

            // distScale/offScale are fixed at 1.0 in smooth mode (no time-varying zoom)
            autoCsv << f << "," << time_s << "," << 1.0 << "," << 1.0 << "\n";

            // Rig GT: IMU at rigPos_w, IMU frame aligned to rig/cam frame
            Quat q_wi = quatFromMat3(R_rw_final); // IMU->WORLD

            rigFramesGT.push_back({time_s, rigPos_w, q_wi});
            rigFramesCsv << f << "," << time_s << ","
                    << rigPos_w.x << "," << rigPos_w.y << "," << rigPos_w.z << ","
                    << q_wi.w << "," << q_wi.x << "," << q_wi.y << "," << q_wi.z << "\n";

            std::cout << "Frame " << f << "/" << (NUM_FRAMES - 1)
                    << (ok ? "" : " [WARN board out of view]")
                    << "\n";
            // Render cameras
            for (int i = 0; i < 4; ++i) {
                auto gray = renderFrameGray(cams[i], rays[i], planes, 255);

                std::string fname = outDir + "/cam" + std::to_string(i)
                                    + "/frame_" + zpad(f, 5) + ".png";
                writeGrayPNGFile(fname, cams[i].W, cams[i].H, gray);

                const Mat3 &R = cams[i].R_wc;
                double R00 = R.c0.x, R01 = R.c1.x, R02 = R.c2.x;
                double R10 = R.c0.y, R11 = R.c1.y, R12 = R.c2.y;
                double R20 = R.c0.z, R21 = R.c1.z, R22 = R.c2.z;

                poseCsv[i] << f << "," << time_s << ","
                        << cams[i].C_w.x << "," << cams[i].C_w.y << "," << cams[i].C_w.z << ","
                        << R00 << "," << R01 << "," << R02 << ","
                        << R10 << "," << R11 << "," << R12 << ","
                        << R20 << "," << R21 << "," << R22 << "\n";


                // Store PNG bytes (no scaling) as CompressedImage
                std::vector<uint8_t> pngBytes = encodeGrayPNGToMemory(cams[i].W, cams[i].H, gray);

                sensor_msgs::CompressedImage msg;
                msg.header.stamp = toRosStamp(time_s);
                msg.header.frame_id = "cam" + std::to_string(i);
                msg.format = "png";
                msg.data = std::move(pngBytes);

                bag.write("/cam" + std::to_string(i) + "/image_png", msg.header.stamp, msg);
            }
        }

        // -----------------------
        // Encode MP4 per camera (no preset; no scaling)
        // -----------------------
        for (int i = 0; i < 4; ++i) {
            encodeMp4WithFfmpeg_NoPreset(outDir, i, FPS);
        }

        // -----------------------
        // Generate IMU
        // -----------------------
        ImuNoiseConfig imuNoise;
        imuNoise.enable = false; // set true for noisy IMU
        imuNoise.gyroNoiseStd = 0.002;
        imuNoise.accelNoiseStd = 0.02;

        std::vector<PoseGT> gtAtImu;
        auto imu = generateImu(rigFramesGT, IMU_HZ, imuNoise, &gtAtImu);

        // Save IMU CSV
        std::string imuCsvPath = outDir + "/imu.csv"; {
            std::ofstream csv(imuCsvPath);
            csv << "t,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,"
                    "px_gt,py_gt,pz_gt,qw_gt,qx_gt,qy_gt,qz_gt\n";
            for (const auto &s: imu) {
                csv << s.t << ","
                        << s.gyro_i.x << "," << s.gyro_i.y << "," << s.gyro_i.z << ","
                        << s.accel_i.x << "," << s.accel_i.y << "," << s.accel_i.z << ","
                        << s.gt.p_w.x << "," << s.gt.p_w.y << "," << s.gt.p_w.z << ","
                        << s.gt.q_wi.w << "," << s.gt.q_wi.x << "," << s.gt.q_wi.y << "," << s.gt.q_wi.z
                        << "\n";
            }
        }
        std::cout << "Wrote IMU CSV: " << imuCsvPath << "\n";

        // IMU integration verification
        std::string errCsvPath = outDir + "/imu_integration_error.csv";
        auto integRes = integrateImuAndCompare(imu, IMU_HZ, errCsvPath);
        std::cout << "Wrote IMU integration error CSV: " << errCsvPath << "\n";
        std::cout << "IMU integration RMS angle error (deg): " << integRes.rmsAngleDeg << "\n";
        std::cout << "IMU integration RMS position error (m): " << integRes.rmsPos << "\n";


        // Write IMU + GT pose to bag
        for (const auto &s: imu) {
            ros::Time stamp = toRosStamp(s.t);

            sensor_msgs::Imu m;
            m.header.stamp = stamp;
            m.header.frame_id = "imu";
            m.orientation_covariance[0] = -1.0; // IMU orientation not directly provided here
            m.angular_velocity.x = s.gyro_i.x;
            m.angular_velocity.y = s.gyro_i.y;
            m.angular_velocity.z = s.gyro_i.z;
            m.linear_acceleration.x = s.accel_i.x;
            m.linear_acceleration.y = s.accel_i.y;
            m.linear_acceleration.z = s.accel_i.z;
            bag.write("/imu", stamp, m);

            geometry_msgs::PoseStamped gt;
            gt.header.stamp = stamp;
            gt.header.frame_id = "world";
            gt.pose.position.x = s.gt.p_w.x;
            gt.pose.position.y = s.gt.p_w.y;
            gt.pose.position.z = s.gt.p_w.z;
            gt.pose.orientation.w = s.gt.q_wi.w;
            gt.pose.orientation.x = s.gt.q_wi.x;
            gt.pose.orientation.y = s.gt.q_wi.y;
            gt.pose.orientation.z = s.gt.q_wi.z;
            bag.write("/rig/gt_pose", stamp, gt);
        }
        bag.close();
        std::cout << "Closed ROS bag: " << (outDir + "/sim.bag") << "\n";

        std::cout << "\nDone.\n"
                << "PNG sequences: " << outDir << "/cam*/frame_XXXXX.png\n"
                << "Videos       : " << outDir << "/cam0.mp4 ... cam3.mp4 (or .mkv fallback)\n"
                << "IMU CSV      : " << outDir << "/imu.csv\n"
                << "IMU verify   : " << outDir << "/imu_integration_error.csv\n"
                << "Rig GT       : " << outDir << "/rig_frames_gt.csv\n"
                << "AutoFit CSV  : " << outDir << "/autofit.csv\n"
                << "ROS bag      : " << outDir << "/sim.bag\n"
                << "Intrinsics are held constant; no image scaling is applied.\n";

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
