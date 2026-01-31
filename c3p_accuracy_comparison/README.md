PEM C3P C++ Simulation Project

This project reproduces the simulation experiments from the provided YALMIP MATLAB scripts
for the following closed-chain calibration types

  AX = XB
  AX = YB
  AXB = YCZ
  AXBY = ZCWD

Instead of YALMIP and MATLAB solvers, the project implements a simple Levenberg–Marquardt
solver on SE(3) using Armadillo for linear algebra.

Build

  mkdir -p build
  cd build
  cmake ..
  cmake --build . -j

Run

  ./pem_sim

Monte-Carlo benchmark with increasing noise

  ./pem_benchmark_mc --mc 200 --N 20 --out mc_results.csv

Noise-sweep evaluation for C3P specializations

  ./pem_c3p_noise_sweep --P 1 --Q 1 --noise 0:0.002:0.02 --mc 200 --N 20 --out c3p_p1q1.csv
  ./pem_c3p_noise_sweep --P 1 --Q 2 --noise 0:0.002:0.02 --mc 200 --N 20 --out c3p_p1q2.csv
  ./pem_c3p_noise_sweep --P 2 --Q 2 --noise 0:0.002:0.02 --mc 200 --N 20 --out c3p_p2q2.csv

The solver option can be set to PEM, Analytical, or all. Analytical is currently
implemented for AX=YB and AXB=YCZ.

The CSV can be visualized with the MATLAB script

  matlab/benchmark_accuracy_c3p_boxplot.m

The program runs all four experiments sequentially and prints

  the ground-truth transforms
  the estimated transforms
  rotation and translation errors
  final PEM objective value

Notes

  The solver uses numerical Jacobians and a multi-start strategy.
  You can change the number of random restarts and noise levels in src/main.cpp.

  The benchmark executable compares PEM against paper-referenced baseline solvers:
  Park1994, Horaud1995, Daniilidis1999, Zhang2017 for AX=XB,
  Dornaika1998, Shah2013, Park2016TIE, Tabb2017 for AX=YB,
  Wu2016TRO, Ma2018, Sui2023 for AXB=YCZ.

  Notes on implementations used in this codebase
  Park1994, Horaud1995, Daniilidis1999, Dornaika1998, Shah2013, and Wu2016TRO
  are implemented as closed-form initializations matching the corresponding
  papers at the algorithmic level.

  Zhang2017, Park2016TIE, Tabb2017, Ma2018, and Sui2023 are implemented as
  iterative refinements on standard least-squares objectives with the paper-named
  initializations when available.
