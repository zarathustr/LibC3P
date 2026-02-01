# aruco_extrinsic_calib_C3P

**ROS1 catkin package name:** `aruco_extrinsic_calib_c3p`  
> Note: ROS package names are required to be lowercase. This repo folder is named `aruco_extrinsic_calib_C3P`, but you should use `rosrun aruco_extrinsic_calib_c3p ...`.

This C++ project (OpenCV **3.4.5** + `opencv_contrib` ArUco) provides:

1. **Multi-camera** extraction of ArUco markers from a ROS1 bag, using **user-defined camera count and topics**.
2. Supports both **`sensor_msgs/CompressedImage`** and **`sensor_msgs/Image`** (raw) topics.
3. Uses intrinsics/distortion and extrinsics from a Kalibr-style `camchain.yaml` (e.g., `cam0`, `cam1`, `cam2`, ...).
4. Estimates all marker poses (`T_camera<-marker`) and records them to CSV.
5. Saves annotated frames to high-resolution **MP4** (falls back to **AVI** if MP4 codecs are unavailable).
6. Preserves the previous **extrinsic verification** scheme (common-marker + hand-eye AX=XB) and extends it to **N cameras**.
7. Adds a dedicated verifier for the **AXBY = ZCWD** calibration loop (4 cameras + 4 markers), solving for **X, Y, Z, W** and comparing camera-camera parts to your `camchain.yaml`.

---

## Build

```bash
cd ~/catkin_ws/src
# place the folder here:
#   aruco_extrinsic_calib_C3P/
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

Dependencies:
- ROS1: `roscpp`, `rosbag`, `sensor_msgs`
- OpenCV 3.4.5 with `aruco` module (`opencv_contrib`)
- `yaml-cpp`

---

## 1) Multi-camera extraction + per-camera MP4 + CSV

```bash
rosrun aruco_extrinsic_calib_c3p extract_aruco_from_bag   --bag calib.bag   --calib calib-camchain.yaml   --out_dir out   --tag_size 0.25   --dict DICT_6X6_250   --sync_tol 0.01   --image_mode auto
```

### Selecting camera count and topics

- Use the first **N** cameras in the YAML:

```bash
--num_cams 4
```

- Override topics explicitly (comma-separated, must match `--num_cams`):

```bash
--topics /cam0/image_raw,/cam1/image_raw,/cam2/image_raw,/cam3/image_raw
```

### Image mode

- `--image_mode auto` (default): try CompressedImage first, then raw Image.
- `--image_mode compressed`: only `sensor_msgs/CompressedImage`
- `--image_mode raw`: only `sensor_msgs/Image`

### Color channel swap

If your saved videos look like **red/blue are swapped** (RGB vs BGR mismatch), add:

```bash
--swap_rb
```

This swaps the red and blue channels **before** detection, drawing, and video writing.


### Outputs

In `out/`:

- `cam0_annotated.mp4` (or `.avi` fallback), `cam1_annotated.mp4`, ...
- `cam0_aruco_poses.csv`, `cam1_aruco_poses.csv`, ...
- `extrinsic_summary.txt`
- `extrinsic_common_markers_cam1.csv`, `extrinsic_common_markers_cam2.csv`, ... (per camera vs cam0)

---

## 2) Pairwise extrinsic verification from CSV

```bash
rosrun aruco_extrinsic_calib_c3p verify_extrinsics   --calib calib-camchain.yaml   --camA_csv out/cam0_aruco_poses.csv --camA_idx 0   --camB_csv out/cam2_aruco_poses.csv --camB_idx 2   --sync_tol 0.01
```

This estimates `T_camB<-camA` from common markers and also via hand-eye AX=XB, and compares to the calibration chain (if available).

---

## 3) AXBY = ZCWD verification (4 cameras + 4 markers)

This solves the loop equation:

`A_i X B_i Y = Z C_i W D_i`

with:
- **measured** `A_i,B_i,C_i,D_i` from per-frame ArUco poses
- **unknown**:
  - `X = T_{cam1<-cam0}`
  - `W = T_{cam3<-cam2}`
  - `Y = T_{m0<-m2}`
  - `Z = T_{m1<-m3}`

Example:

```bash
rosrun aruco_extrinsic_calib_c3p verify_axby_zcwd   --calib calib-camchain.yaml   --csv_dir out   --cam0 0 --marker0 0   --cam1 1 --marker1 1   --cam2 2 --marker2 2   --cam3 3 --marker3 3   --sync_tol 0.01   --tag_size 0.25 --marker_margin 0.01
```

The tool prints estimated `X,Y,Z,W`, residual statistics, and compares `X`/`W` against the `camchain.yaml` camera extrinsics when available.

### Saving the estimated transforms to YAML

Add `--out_xyzw out/estimated_XYZW.yaml`:

```bash
rosrun aruco_extrinsic_calib_c3p verify_axby_zcwd \
  --calib calib-camchain.yaml --csv_dir out \
  --cam0 0 --marker0 0 --cam1 1 --marker1 1 --cam2 2 --marker2 2 --cam3 3 --marker3 3 \
  --sync_tol 0.01 --tag_size 0.25 --marker_margin 0.01 \
  --out_xyzw out/estimated_XYZW.yaml
```

---

## 4) Reprojection overlay MP4 for AXBY = ZCWD (estimated X,Y,Z,W)

This creates **new MP4/AVI videos** where the detected marker corners are shown in **green**, and the marker corners reprojected via the **estimated** (X,Y,Z,W) loop are drawn in **red**.

```bash
rosrun aruco_extrinsic_calib_c3p overlay_reprojection_zcwd_from_bag \
  --bag calib.bag --calib calib-camchain.yaml --xyzw_yaml out/estimated_XYZW.yaml \
  --out_dir out_reproj_zcwd \
  --tag_size 0.25 --dict DICT_6X6_250 --sync_tol 0.01 \
  --image_mode auto
```

Topic override (comma-separated topics for cam0,cam1,cam2,cam3):

```bash
--topics /cam0/image_raw,/cam1/image_raw,/cam2/image_raw,/cam3/image_raw
```

---

## Notes

- Your calibration YAML contains the camera intrinsics/distortion and `T_cn_cnm1` extrinsics (e.g., cam1 relative to cam0).  
- For best hand-eye / AXBY results, include diverse motion (rotation + translation) of the board in view.
