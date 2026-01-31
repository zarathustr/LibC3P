#include <iostream>
#include <iomanip>
#include "se3.h"
#include "lm.h"
#include "problems.h"

using namespace pem;

static void printSE3(const std::string& name, const SE3& T) {
    std::cout << name << " =\n";
    std::cout << arma::mat(T.matrix()) << std::endl;
}

static void printError(const std::string& name, const SE3& Test, const SE3& Ttrue) {
    Mat3 Rerr = Test.R * Ttrue.R.t();
    double ang = rotationAngleDeg(Rerr);
    double terr = norm3(Test.t - Ttrue.t);
    std::cout << name << " rotation error deg: " << ang
              << "  translation error: " << terr << std::endl;
}

int main() {
    arma::arma_rng::set_seed(1);

    LMOptions opts;
    opts.max_iters = 80;
    opts.max_restarts = 8;
    opts.verbose = false;

    std::cout << std::setprecision(6) << std::fixed;

    {
        std::cout << "\n=== Experiment AX = XB  PEM ===\n";
        int N = 3;
        double noise = 1e-3;

        SE3 Xtrue = randomSE3(1.0, 1.0);

        AXXBData data = makeAXXBData(Xtrue, N, noise);

        ResidualFunction fun = [&](const std::vector<SE3>& vars) {
            return residualAXXB_PEM(data, vars);
        };

        std::vector<SE3> init = { SE3::Identity() };

        LMSummary sum;
        std::vector<SE3> sol = solveLMSE3(fun, init, opts, &sum);

        printSE3("X_true", Xtrue);
        printSE3("X_est", sol[0]);
        printError("X", sol[0], Xtrue);
        std::cout << "final cost " << sum.final_cost << " iters " << sum.iters << "\n";
    }

    {
        std::cout << "\n=== Experiment AX = YB  PEM ===\n";
        int N = 3;
        double noise = 1e-4;

        double level = 1e-6;
        SE3 Xtrue = orthonormalizeSE3(SE3(orthonormalizeRot(Mat3().eye() + level * arma::randn<arma::mat>(3,3)), arma::randn<arma::vec>(3)));
        SE3 Ytrue = orthonormalizeSE3(SE3(orthonormalizeRot(Mat3().eye() + level * arma::randn<arma::mat>(3,3)), arma::randn<arma::vec>(3)));

        AXYBData data = makeAXYBData(Xtrue, Ytrue, N, noise);

        ResidualFunction fun = [&](const std::vector<SE3>& vars) {
            return residualAXYB_PEM(data, vars);
        };

        std::vector<SE3> init = { SE3::Identity(), SE3::Identity() };

        LMSummary sum;
        std::vector<SE3> sol = solveLMSE3(fun, init, opts, &sum);

        printSE3("X_true", Xtrue);
        printSE3("X_est", sol[0]);
        printError("X", sol[0], Xtrue);

        printSE3("Y_true", Ytrue);
        printSE3("Y_est", sol[1]);
        printError("Y", sol[1], Ytrue);

        std::cout << "final cost " << sum.final_cost << " iters " << sum.iters << "\n";
    }

    {
        std::cout << "\n=== Experiment AXB = YCZ  PEM ===\n";
        int N = 6;
        double noise = 1e-3;

        SE3 Xtrue = randomSE3(1.0, 1.0);
        SE3 Ytrue = randomSE3(1.0, 1.0);
        SE3 Ztrue = randomSE3(1.0, 1.0);

        AXB_YCZData data = makeAXB_YCZData(Xtrue, Ytrue, Ztrue, N, noise);

        ResidualFunction fun = [&](const std::vector<SE3>& vars) {
            return residualAXB_YCZ_PEM(data, vars);
        };

        std::vector<SE3> init = { SE3::Identity(), SE3::Identity(), SE3::Identity() };

        LMSummary sum;
        std::vector<SE3> sol = solveLMSE3(fun, init, opts, &sum);

        printSE3("X_true", Xtrue);
        printSE3("X_est", sol[0]);
        printError("X", sol[0], Xtrue);

        printSE3("Y_true", Ytrue);
        printSE3("Y_est", sol[1]);
        printError("Y", sol[1], Ytrue);

        printSE3("Z_true", Ztrue);
        printSE3("Z_est", sol[2]);
        printError("Z", sol[2], Ztrue);

        std::cout << "final cost " << sum.final_cost << " iters " << sum.iters << "\n";
    }

    {
        std::cout << "\n=== Experiment AXBY = ZCWD  PEM ===\n";
        int N = 15;
        double noise = 1e-3;

        SE3 Xtrue = randomSE3(1.0, 1.0);
        SE3 Ytrue = randomSE3(1.0, 1.0);
        SE3 Ztrue = randomSE3(1.0, 1.0);
        SE3 Wtrue = randomSE3(1.0, 1.0);

        AXBY_ZCWDData data = makeAXBY_ZCWDData(Xtrue, Ytrue, Ztrue, Wtrue, N, noise);

        ResidualFunction fun = [&](const std::vector<SE3>& vars) {
            return residualAXBY_ZCWD_PEM(data, vars);
        };

        std::vector<SE3> init = { SE3::Identity(), SE3::Identity(), SE3::Identity(), SE3::Identity() };

        LMSummary sum;
        std::vector<SE3> sol = solveLMSE3(fun, init, opts, &sum);

        printSE3("X_true", Xtrue);
        printSE3("X_est", sol[0]);
        printError("X", sol[0], Xtrue);

        printSE3("Y_true", Ytrue);
        printSE3("Y_est", sol[1]);
        printError("Y", sol[1], Ytrue);

        printSE3("Z_true", Ztrue);
        printSE3("Z_est", sol[2]);
        printError("Z", sol[2], Ztrue);

        printSE3("W_true", Wtrue);
        printSE3("W_est", sol[3]);
        printError("W", sol[3], Wtrue);

        std::cout << "final cost " << sum.final_cost << " iters " << sum.iters << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
