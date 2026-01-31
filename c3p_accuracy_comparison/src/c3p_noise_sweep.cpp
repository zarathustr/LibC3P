// Noise-sweep evaluation for C3P instances.
//
// This executable focuses on the three C3P specializations requested:
//   P=1, Q=1 : AX = YB
//   P=1, Q=2 : AXB = YCZ
//   P=2, Q=2 : AXBY = ZCWD
//
// Output is a per-trial CSV compatible with box-plot and statistics scripts.
//
// CSV columns:
//   problem,solver,P,Q,noise,trial,rot_err_deg,trans_err

#include <armadillo>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "problems.h"
#include "solvers.h"

using namespace pem;

namespace {

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        out.push_back(item);
    }
    return out;
}

std::vector<double> parseNoiseSpec(const std::string& s) {
    // Supported formats:
    //   "a,b,c"        explicit list
    //   "start:step:end" MATLAB-like range
    //   "x"            single value
    std::vector<double> out;
    if (s.find(',') != std::string::npos) {
        auto items = split(s, ',');
        for (const auto& it : items) {
            if (!it.empty()) out.push_back(std::stod(it));
        }
        return out;
    }
    if (s.find(':') != std::string::npos) {
        auto items = split(s, ':');
        if (items.size() == 3) {
            double start = std::stod(items[0]);
            double step  = std::stod(items[1]);
            double end   = std::stod(items[2]);
            if (step <= 0) {
                return {start};
            }
            for (double x = start; x <= end + 1e-15; x += step) {
                out.push_back(x);
            }
            return out;
        }
        // Fallback: treat as list using ':' as separator
        for (const auto& it : items) {
            if (!it.empty()) out.push_back(std::stod(it));
        }
        return out;
    }
    out.push_back(std::stod(s));
    return out;
}

std::vector<std::string> parseSolverList(const std::string& s) {
    // "all" expands to a minimal set:
    //   AXYB: PEM, Analytical
    //   AXB_YCZ: PEM, Analytical
    //   AXBY_ZCWD: PEM
    if (s == "all") {
        return {"PEM", "Analytical"};
    }
    if (s.find(',') != std::string::npos) {
        return split(s, ',');
    }
    return {s};
}

double rotErrDeg(const Mat3& R_est, const Mat3& R_true) {
    return rotationAngleDeg(R_est.t() * R_true);
}

void writeRow(std::ofstream& f,
              const std::string& problem,
              const std::string& solver,
              int P,
              int Q,
              double noise,
              int trial,
              double rot_err_deg,
              double trans_err) {
    f << problem << ',' << solver << ',' << P << ',' << Q << ',' << noise << ',' << trial << ','
      << rot_err_deg << ',' << trans_err << "\n";
}

void usage() {
    std::cout
        << "Usage: pem_c3p_noise_sweep [options]\n\n"
        << "Options:\n"
        << "  --P <int>            C3P parameter P. Supported: 1 or 2. Default 1\n"
        << "  --Q <int>            C3P parameter Q. Supported: 1 or 2. Default 1\n"
        << "  --N <int>            Number of measurements per trial. Default 20\n"
        << "  --mc <int>           Monte-Carlo trials per noise level. Default 200\n"
        << "  --noise <spec>       Noise spec. Examples: 0,0.002,0.01  or  0:0.002:0.02\n"
        << "  --solver <name(s)>   Solver: PEM, Analytical, or comma list, or all. Default all\n"
        << "  --out <file.csv>     Output CSV file. Default c3p_noise_sweep.csv\n"
        << "  --seed <int>         Base RNG seed. Default 1234\n";
}

}  // namespace

int main(int argc, char** argv) {
    int P = 1;
    int Q = 1;
    int N = 20;
    int MC = 200;
    int seed0 = 1234;
    std::string outCsv = "c3p_noise_sweep.csv";
    std::vector<double> noiseLevels = {0.0, 0.002, 0.005, 0.01, 0.02};
    std::vector<std::string> solvers = {"PEM", "Analytical"};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--P" && i + 1 < argc) {
            P = std::stoi(argv[++i]);
        } else if (a == "--Q" && i + 1 < argc) {
            Q = std::stoi(argv[++i]);
        } else if (a == "--N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (a == "--mc" && i + 1 < argc) {
            MC = std::stoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            seed0 = std::stoi(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            outCsv = argv[++i];
        } else if (a == "--noise" && i + 1 < argc) {
            noiseLevels = parseNoiseSpec(argv[++i]);
        } else if (a == "--solver" && i + 1 < argc) {
            solvers = parseSolverList(argv[++i]);
        } else if (a == "--help") {
            usage();
            return 0;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            usage();
            return 1;
        }
    }

    std::string problem;
    if (P == 1 && Q == 1) {
        problem = "AXYB";
    } else if (P == 1 && Q == 2) {
        problem = "AXB_YCZ";
    } else if (P == 2 && Q == 2) {
        problem = "AXBY_ZCWD";
    } else {
        std::cerr << "Unsupported (P,Q)= (" << P << "," << Q << ")\n";
        std::cerr << "Supported: (1,1), (1,2), (2,2)\n";
        return 1;
    }

    std::ofstream f(outCsv);
    if (!f) {
        std::cerr << "Failed to open output file: " << outCsv << "\n";
        return 1;
    }
    f << "problem,solver,P,Q,noise,trial,rot_err_deg,trans_err\n";

    for (size_t ni = 0; ni < noiseLevels.size(); ++ni) {
        double noise = noiseLevels[ni];
        std::cout << "Noise " << noise << "  (" << (ni + 1) << "/" << noiseLevels.size() << ")\n";

        for (int trial = 0; trial < MC; ++trial) {
            arma::arma_rng::set_seed(static_cast<arma::uword>(seed0)
                                    + static_cast<arma::uword>(trial)
                                    + static_cast<arma::uword>(100000) * static_cast<arma::uword>(ni));

            if (P == 1 && Q == 1) {
                SE3 X_true = randomSE3(0.8, 1.0);
                SE3 Y_true = randomSE3(0.8, 1.0);
                AXYBData data = makeAXYBData(X_true, Y_true, N, noise);

                for (const auto& sname : solvers) {
                    if (sname == "PEM") {
                        auto est = solveAXYB_PEM(data);
                        double r = 0.5 * (rotErrDeg(est.first.R, X_true.R) + rotErrDeg(est.second.R, Y_true.R));
                        double t = 0.5 * (norm3(est.first.t - X_true.t) + norm3(est.second.t - Y_true.t));
                        writeRow(f, problem, sname, P, Q, noise, trial, r, t);
                    } else if (sname == "Analytical") {
                        auto est = solveAXYB_Analytical(data);
                        double r = 0.5 * (rotErrDeg(est.first.R, X_true.R) + rotErrDeg(est.second.R, Y_true.R));
                        double t = 0.5 * (norm3(est.first.t - X_true.t) + norm3(est.second.t - Y_true.t));
                        writeRow(f, problem, sname, P, Q, noise, trial, r, t);
                    } else {
                        std::cerr << "Unsupported solver for AXYB: " << sname << "\n";
                    }
                }
            } else if (P == 1 && Q == 2) {
                SE3 X_true = randomSE3(0.8, 1.0);
                SE3 Y_true = randomSE3(0.8, 1.0);
                SE3 Z_true = randomSE3(0.8, 1.0);
                AXB_YCZData data = makeAXB_YCZData(X_true, Y_true, Z_true, N, noise);

                for (const auto& sname : solvers) {
                    if (sname == "PEM") {
                        auto est = solveAXB_YCZ_PEM(data);
                        const SE3& Xe = std::get<0>(est);
                        const SE3& Ye = std::get<1>(est);
                        const SE3& Ze = std::get<2>(est);
                        double r = (rotErrDeg(Xe.R, X_true.R) + rotErrDeg(Ye.R, Y_true.R) + rotErrDeg(Ze.R, Z_true.R)) / 3.0;
                        double t = (norm3(Xe.t - X_true.t) + norm3(Ye.t - Y_true.t) + norm3(Ze.t - Z_true.t)) / 3.0;
                        writeRow(f, problem, sname, P, Q, noise, trial, r, t);
                    } else if (sname == "Analytical") {
                        auto est = solveAXB_YCZ_Analytical(data);
                        const SE3& Xe = std::get<0>(est);
                        const SE3& Ye = std::get<1>(est);
                        const SE3& Ze = std::get<2>(est);
                        double r = (rotErrDeg(Xe.R, X_true.R) + rotErrDeg(Ye.R, Y_true.R) + rotErrDeg(Ze.R, Z_true.R)) / 3.0;
                        double t = (norm3(Xe.t - X_true.t) + norm3(Ye.t - Y_true.t) + norm3(Ze.t - Z_true.t)) / 3.0;
                        writeRow(f, problem, sname, P, Q, noise, trial, r, t);
                    } else {
                        std::cerr << "Unsupported solver for AXB_YCZ: " << sname << "\n";
                    }
                }
            } else if (P == 2 && Q == 2) {
                SE3 X_true = randomSE3(0.8, 1.0);
                SE3 Y_true = randomSE3(0.8, 1.0);
                SE3 Z_true = randomSE3(0.8, 1.0);
                SE3 W_true = randomSE3(0.8, 1.0);
                AXBY_ZCWDData data = makeAXBY_ZCWDData(X_true, Y_true, Z_true, W_true, N, noise);

                for (const auto& sname : solvers) {
                    if (sname == "PEM") {
                        auto est = solveAXBY_ZCWD_PEM(data);
                        const SE3& Xe = std::get<0>(est);
                        const SE3& Ye = std::get<1>(est);
                        const SE3& Ze = std::get<2>(est);
                        const SE3& We = std::get<3>(est);
                        double r = (rotErrDeg(Xe.R, X_true.R) + rotErrDeg(Ye.R, Y_true.R) + rotErrDeg(Ze.R, Z_true.R) + rotErrDeg(We.R, W_true.R)) / 4.0;
                        double t = (norm3(Xe.t - X_true.t) + norm3(Ye.t - Y_true.t) + norm3(Ze.t - Z_true.t) + norm3(We.t - W_true.t)) / 4.0;
                        writeRow(f, problem, sname, P, Q, noise, trial, r, t);
                    } else if (sname == "Analytical") {
                        // Not implemented for AXBY=ZCWD.
                    } else {
                        std::cerr << "Unsupported solver for AXBY_ZCWD: " << sname << "\n";
                    }
                }
            }
        }
    }

    std::cout << "Done. CSV saved to: " << outCsv << "\n";
    return 0;
}
