// Monte-Carlo benchmark driver.
//
// Produces a CSV with per-trial rotation and translation errors across noise levels.
//
// Expected CSV columns:
//   problem, solver, noise, trial, rot_err_deg, trans_err

#include <armadillo>

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

std::vector<double> parseCommaList(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(std::stod(item));
    }
    return out;
}

void writeRow(std::ofstream& f,
              const std::string& problem,
              const std::string& solver,
              double noise,
              int trial,
              double rot_err_deg,
              double trans_err) {
    f << problem << ',' << solver << ',' << noise << ',' << trial << ','
      << rot_err_deg << ',' << trans_err << "\n";
}

double rotErrDeg(const Mat3& R_est, const Mat3& R_true) {
    return rotationAngleDeg(R_est.t() * R_true);
}

}  // namespace

int main(int argc, char** argv) {
    int MC = 200;
    int N = 20;
    std::string outCsv = "mc_results.csv";
    std::vector<double> noiseLevels = {0.0, 0.002, 0.005, 0.01, 0.02};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mc" && i + 1 < argc) {
            MC = std::stoi(argv[++i]);
        } else if (a == "--N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            outCsv = argv[++i];
        } else if (a == "--noise" && i + 1 < argc) {
            noiseLevels = parseCommaList(argv[++i]);
        } else if (a == "--help") {
            std::cout
                << "Usage: pem_benchmark_mc [--mc 200] [--N 20] [--noise 0,0.002,...] [--out mc_results.csv]\n";
            return 0;
        }
    }

    std::ofstream f(outCsv);
    if (!f) {
        std::cerr << "Failed to open output file: " << outCsv << std::endl;
        return 1;
    }
    f << "problem,solver,noise,trial,rot_err_deg,trans_err\n";

    for (size_t ni = 0; ni < noiseLevels.size(); ++ni) {
        double noise = noiseLevels[ni];
        std::cout << "Noise level " << noise << "  (" << (ni + 1) << "/" << noiseLevels.size() << ")" << std::endl;

        for (int trial = 0; trial < MC; ++trial) {
            arma::arma_rng::set_seed(1234u + static_cast<unsigned>(trial) + 10000u * static_cast<unsigned>(ni));

            // -----------------------------
            // AX = XB
            // -----------------------------
            {
                SE3 X_true = randomSE3(0.8, 1.0);
                AXXBData data = makeAXXBData(X_true, N, noise);

                SE3 X_pem = solveAXXB_PEM(data);
                SE3 X_ana = solveAXXB_Analytical(data);
                SE3 X_park = solveAXXB_Park1994(data);
                SE3 X_horaud = solveAXXB_Horaud1995(data);
                SE3 X_dani = solveAXXB_Daniilidis1999(data);
                SE3 X_zhang = solveAXXB_Zhang2017(data);

                writeRow(f, "AXXB", "Ours-PEM", noise, trial, rotErrDeg(X_pem.R, X_true.R), norm3(X_pem.t - X_true.t));
                writeRow(f, "AXXB", "Analytical", noise, trial, rotErrDeg(X_ana.R, X_true.R), norm3(X_ana.t - X_true.t));
                writeRow(f, "AXXB", "Park1994", noise, trial, rotErrDeg(X_park.R, X_true.R), norm3(X_park.t - X_true.t));
                writeRow(f, "AXXB", "Horaud1995", noise, trial, rotErrDeg(X_horaud.R, X_true.R), norm3(X_horaud.t - X_true.t));
                writeRow(f, "AXXB", "Daniilidis1999", noise, trial, rotErrDeg(X_dani.R, X_true.R), norm3(X_dani.t - X_true.t));
                writeRow(f, "AXXB", "Zhang2017", noise, trial, rotErrDeg(X_zhang.R, X_true.R), norm3(X_zhang.t - X_true.t));
            }

            // -----------------------------
            // AX = YB
            // -----------------------------
            {
                SE3 X_true = randomSE3(0.8, 1.0);
                SE3 Y_true = randomSE3(0.8, 1.0);
                AXYBData data = makeAXYBData(X_true, Y_true, N, noise);

                auto XY_pem = solveAXYB_PEM(data);
                auto XY_ana = solveAXYB_Analytical(data);
                auto XY_dor = solveAXYB_Dornaika1998(data);
                auto XY_shah = solveAXYB_Shah2013(data);
                auto XY_park = solveAXYB_Park2016TIE(data);
                auto XY_tabb = solveAXYB_Tabb2017(data);

                auto meanErr = [&](const std::pair<SE3, SE3>& est) {
                    double r = 0.5 * (rotErrDeg(est.first.R, X_true.R) + rotErrDeg(est.second.R, Y_true.R));
                    double t = 0.5 * (norm3(est.first.t - X_true.t) + norm3(est.second.t - Y_true.t));
                    return std::make_pair(r, t);
                };

                auto e_pem = meanErr(XY_pem);
                auto e_ana = meanErr(XY_ana);
                auto e_dor = meanErr(XY_dor);
                auto e_shah = meanErr(XY_shah);
                auto e_park = meanErr(XY_park);
                auto e_tabb = meanErr(XY_tabb);

                writeRow(f, "AXYB", "Ours-PEM", noise, trial, e_pem.first, e_pem.second);
                writeRow(f, "AXYB", "Analytical", noise, trial, e_ana.first, e_ana.second);
                writeRow(f, "AXYB", "Dornaika1998", noise, trial, e_dor.first, e_dor.second);
                writeRow(f, "AXYB", "Shah2013", noise, trial, e_shah.first, e_shah.second);
                writeRow(f, "AXYB", "Park2016", noise, trial, e_park.first, e_park.second);
                writeRow(f, "AXYB", "Tabb2017", noise, trial, e_tabb.first, e_tabb.second);
            }

            // -----------------------------
            // AXB = YCZ
            // -----------------------------
            {
                SE3 X_true = randomSE3(0.8, 1.0);
                SE3 Y_true = randomSE3(0.8, 1.0);
                SE3 Z_true = randomSE3(0.8, 1.0);
                AXB_YCZData data = makeAXB_YCZData(X_true, Y_true, Z_true, N, noise);

                auto XYZ_pem = solveAXB_YCZ_PEM(data);
                auto XYZ_ana = solveAXB_YCZ_Analytical(data);
                auto XYZ_wu = solveAXB_YCZ_Wu2016TRO(data);
                auto XYZ_ma = solveAXB_YCZ_Ma2018(data);
                auto XYZ_sui = solveAXB_YCZ_Sui2023(data);

                auto meanErr3 = [&](const std::tuple<SE3, SE3, SE3>& est) {
                    const SE3& Xe = std::get<0>(est);
                    const SE3& Ye = std::get<1>(est);
                    const SE3& Ze = std::get<2>(est);
                    double r = (rotErrDeg(Xe.R, X_true.R) + rotErrDeg(Ye.R, Y_true.R) + rotErrDeg(Ze.R, Z_true.R)) / 3.0;
                    double t = (norm3(Xe.t - X_true.t) + norm3(Ye.t - Y_true.t) + norm3(Ze.t - Z_true.t)) / 3.0;
                    return std::make_pair(r, t);
                };

                auto e_pem = meanErr3(XYZ_pem);
                auto e_ana = meanErr3(XYZ_ana);
                auto e_wu = meanErr3(XYZ_wu);
                auto e_ma = meanErr3(XYZ_ma);
                auto e_sui = meanErr3(XYZ_sui);

                writeRow(f, "AXB_YCZ", "Ours-PEM", noise, trial, e_pem.first, e_pem.second);
                writeRow(f, "AXB_YCZ", "Analytical", noise, trial, e_ana.first, e_ana.second);
                writeRow(f, "AXB_YCZ", "Wu2016", noise, trial, e_wu.first, e_wu.second);
                writeRow(f, "AXB_YCZ", "Ma2018", noise, trial, e_ma.first, e_ma.second);
                writeRow(f, "AXB_YCZ", "Sui2023", noise, trial, e_sui.first, e_sui.second);
            }
        }
    }

    std::cout << "Done. Results written to: " << outCsv << std::endl;
    return 0;
}
