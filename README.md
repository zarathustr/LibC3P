# LibC3P: Generalized Simultaneous Closed-Chain Calibration


<a href="https://arxiv.org/abs/2509.06285"><img src='https://img.shields.io/badge/arXiv-2509.06285-b31b1b' alt='arXiv'></a>[![video](https://img.shields.io/badge/Video-Bilibili-74b9ff?logo=bilibili&logoColor=red)]( https://www.bilibili.com/video/BV1jsHQzCEra/?share_source=copy_web)[![C++](c3p_README/C++-17-green.svg)](https://isocpp.org/)[[![ROS](c3p_README/ROS-Melodic%252FNoetic-orange.svg)](http://wiki.ros.org/)[![GitHub Stars](https://img.shields.io/github/stars/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/issues)

</div>


---

## Introduction

We propose a unified framework for solving **Closed-Chain Calibration Problems ($C^3Ps$)**. This framework generalizes classical problems such as **hand-eye calibration** ($AX=XB$), **robot-world calibration** ($AX=YB$), and extends to high-order problems like $AXB=YCZ$ and the novel $AXBY=ZCWD$ formulation. 

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
mkdir build && cd build
cmake ..
make -j4
```

### 3. ROS Build

Copy the package to your catkin workspace `src` folder:

```
cp -r LibC3P ~/catkin_ws/src/
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 4. Running Examples 

**C++ Simulation:**

```
./build/simulation_4cam_pose
```

**MATLAB Scripts:**

Open MATLAB and navigate to the `verification/` folder. Run `CRLB_verification.m` to verify the Cramér-Rao Lower Bound analysis.

------

## Datasets

We provide standard datasets for validating multiple practical $C^3Ps$, collected using high-precision hardware synchronization.

- `data/plane-aruco-2025071801`: Multi-camera Aruco calibration data.
- `data/4cam-jetson`: Data from NVidia Jetson NX setup.

------

## Citation 

If you find this work useful for your research, please cite our paper:

代码段

```bash
@article{wu2024generalized,
  title={Generalized Simultaneous Closed-Chain Calibration},
  author={Wu, Jin and Chen, Xieyuanli and Hu, Xiangcheng and Zhang, Chengxi and Li, Haoang and Jiang, Yi and Ge, Shuzhi Sam and Zhang, Wei and He, Wei},
  journal={The International Journal of Robotics Research},
  year={2026},
  publisher={Arxiv},
  url={[https://github.com/zarathustr/LibC3P](https://github.com/zarathustr/LibC3P)}
}
```

------

## Issues

For any questions, please open an issue or contact `eeweiz@ust.hk`.

