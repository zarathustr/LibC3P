# LibC3P: Generalized Simultaneous Closed-Chain Calibration


<a href="https://arxiv.org/abs/2509.06285"><img src='https://img.shields.io/badge/arXiv-2509.06285-b31b1b' alt='arXiv'></a>[![video](https://img.shields.io/badge/Video-Bilibili-74b9ff?logo=bilibili&logoColor=red)]( https://www.bilibili.com/video/BV1jsHQzCEra/?share_source=copy_web)[![C++](c3p_README/C++-17-green.svg)](https://isocpp.org/)[[![ROS](c3p_README/ROS-Melodic%252FNoetic-orange.svg)](http://wiki.ros.org/)[![GitHub Stars](https://img.shields.io/github/stars/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/issues)

</div>


---

## Introduction

We propose a unified framework for solving **Closed-Chain Calibration Problems ($C^3Ps$)**. This framework generalizes classical problems such as **hand-eye calibration** ($AX=XB$), **robot-world calibration** ($AX=YB$), and extends to high-order problems like $AXB=YCZ$, the novel simultaneous hand-eye/robot-camera/marker-marker calibration $AXBY=ZCWD$ formulation, the Tri-camera non-overlapping calibration. 

The core contribution is the **Physical Ensemble Method (PEM)**, a globally optimal solver that exploits homogeneous symmetry in closed-chain kinematics to improve robustness and accuracy against measurement noise.

### Key Features
* **Unified Formulation:** Solves arbitrary order closed-chain calibration problems ($C^3Ps$) in $SE(3)$.
* **Global Optimality:** Includes solvers based on Gröbner basis, Homotopy Continuation, LMI/SDP, and the proposed PEM (Physical Ensemble Method).
* **High Robustness:** PEM utilizes physically equivalent ensembles to suppress spurious local minima.
* **Diverse Applications:**
    * Classical Hand-Eye ($AX=XB$)
    * Robot-World-Hand-Eye ($AX=YB$)
    * Simultaneous Hand-Eye/Tool-Flange/Robot-Robot ($AXB=YCZ$)
    * **NEW:** Dual-arm/Multi-sensor non-overlapping calibration ($AXBY=ZCWD$)
    * **NEW:** Tri-Camera non-overlapping calibration
* **Multi-Platform Support:** C++, MATLAB, and ROS wrappers.

---

## Methodology & Visuals

### 1. The Concept of $C^3Ps$ 
We generalize calibration tasks into a closed-chain equation:

$$\prod_{j=1}^{P} A_{i,j} X_j = \prod_{j=1}^{Q} Y_j B_{i,j}$$
This covers scenarios from simple point cloud registration to complex multi-robot collaboration.

<div align="center">
    <img src="c3p_README/image-20260123152314904.png" width="40%" alt="Tri-camera Setup">
    <img src="c3p_README/image-20260123152029998.png" width="50%" alt="AXBY=ZCWD Setup">
    <br>
    <em>Left: Tri-camera system (AX=YB variants). Right: Multi-sensor setup (AXBY=ZCWD).</em>
</div>

### 2. Physical Ensemble Method (PEM)
PEM constructs a "Physical Ensemble" from homogeneous symmetries (e.g., inverting the chain, cyclic shifts) to constrain the optimization landscape, ensuring convergence to the global optimum.

<div align="center">
    <img src="c3p_README/pem_opt.jpg" width="100%" alt="PEM Optimization Theory">
    <br>
    <em>Why physical ensemble enhances global solution of $C^3P$s: Left: Solution space significantly reduces after combining objective candidates in physical ensemble; Right: Rigidity guarantees near $\mathrm{SO}(3)$ enables polynomial-friendly approximation to $\mathrm{SO}(3)$ to the maximum extent.</em>
</div>
---

## Prerequisites 

The code has been tested on **macOS** and **Linux (Ubuntu)**.

* **C++ Standard:** C++17
* **ROS:** Melodic / Noetic (for ROS wrappers)
* **MATLAB:** R2023b (for verification scripts)
* **Dependencies:**
    * Eigen3
    * OpenCV (for PnP solvers)
    * [GTSAM](https://gtsam.org/) (optional, for factor graph comparison)
    * [Manopt](https://www.manopt.org/) (optional, for Riemannian optimization comparisons)

---

## Build & Run 

### 1. Clone the repository 
```bash
git clone [https://github.com/zarathustr/LibC3P.git](https://github.com/zarathustr/LibC3P.git)
cd LibC3P
```

### 2. C++ Build

```
cd c3p_accuracy_comparison
mkdir build && cd build
cmake ..
make -j4
./pem_benchmark_mc --mc 500 --N 20 --noise 0.002,0.005,0.01,0.02,0.05 --out mc_results.csv
./pem_c3p_noise_sweep --P 1 --Q 2 --noise 0.005:0.005:0.02 --mc 200 --N 20 --solver PEM --out c3p_p1q2.csv
```

### 3. ROS Build

Copy the package to your catkin workspace `src` folder:

```
cp -r aruco_extrinsic_calib_C3P ~/catkin_ws/src/
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# For 4cam-large-nuc dataset:
rosrun aruco_extrinsic_calib_c3p extract_aruco_from_bag --bag 4cam-large-ust-nautilus-native-0000000000-2026-01-01-06-48-53.bag --calib calib-4cam-large/calib-camchain.yaml --out_dir out --num_cams 4 --topics /sensors/cam0,/sensors/cam1,/sensors/cam2,/sensors/cam3 --tag_size 0.25 --dict DICT_6X6_250 --sync_tol 0.01 --image_mode raw

rosrun aruco_extrinsic_calib_c3p verify_extrinsics --calib calib-4cam-large/calib-camchain.yaml --camA_csv out/cam0_aruco_poses.csv --camA_idx 0 --camB_csv out/cam2_aruco_poses.csv --camB_idx 2  --sync_tol 0.01

rosrun aruco_extrinsic_calib_c3p verify_axby_zcwd --calib calib-4cam-large/calib-camchain.yaml --csv_dir out --cam0 0 --marker0 10 --cam1 1 --marker1 12 --cam2 2 --marker2 13 --cam3 3 --marker3 14 --sync_tol 0.01 --tag_size 0.25 --marker_margin 0.01 --out_xyzw out/estimated_XYZW.yaml

rosrun aruco_extrinsic_calib_c3p overlay_reprojection_zcwd_from_bag --bag 4cam-large-ust-nautilus-native-0000000000-2026-01-01-06-48-53.bag --calib calib-4cam-large/calib-camchain.yaml --xyzw_yaml out/estimated_XYZW.yaml --topics /sensors/cam0,/sensors/cam1,/sensors/cam2,/sensors/cam3 --image_mode raw --sync_tol 0.01 --sync_mode ref0 --out_dir out_reproj_zcwd --swap_rb
```

### 4. Running Examples 

**C++ Simulation:**

```
./build/simulation_4cam_pose
```

**MATLAB Scripts:**

Open MATLAB and navigate to the `verification/` folder. Run `CRLB_verification.m` to verify the Cramér-Rao Lower Bound analysis.

**Python Scripts:**

Redirect to `AutoTight_comparison`:
```
# Python 3.10.19
source ~/miniforge3/etc/profile.d/conda.sh
conda activate constraint_learning

python -m _scripts.run_c3p_se3_axxb_autotight
python -m _scripts.run_c3p_se3_axyb_autotight
python -m _scripts.run_c3p_se3_axb_ycz_autotight
python -m _scripts.run_c3p_se3_axby_zcwd_autotight
```

------

## Datasets

We provide standard datasets for validating multiple practical $C^3Ps$, collected using high-precision hardware synchronization.

- `plane-aruco-2025071801`: Multi-camera Aruco calibration data.
- `4cam-jetson`: Data from NVidia Jetson NX setup.
- `4cam-nuc`: Data from Intel NUC driven 4 camera extrinsic calibration setup.
- `tencent-robot-datasets`: Industrial-robot calibration in Tencent Robotics X.

------

## Citation

If you find this work useful for your research, please cite our paper:

```bash
@article{wu2024generalized,
  title={Simultaneous Closed-Chain Calibration: Generalized Optimization, Global Solutions and Applications},
  author={Wu, Jin and Chen, Xieyuanli and Hu, Xiangcheng and Zhang, Chengxi and Li, Haoang and Jiang, Yi and Ge, Shuzhi Sam and Zhang, Wei and He, Wei},
  journal={Submission to The International Journal of Robotics Research},
  year={2026},
  publisher={Arxiv},
  url={[https://github.com/zarathustr/LibC3P](https://github.com/zarathustr/LibC3P)}
}
```

------

## Issues

For any questions, please open an issue or contact `wujin@ustb.edu.cn` or `xhubd@connect.ust.hk`.

