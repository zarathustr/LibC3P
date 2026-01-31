#pragma once
#include <armadillo>
#include <functional>
#include <vector>
#include "se3.h"

namespace pem {

struct LMOptions {
    int max_iters = 100;
    int max_restarts = 10;
    double fd_eps = 1e-6;
    double lambda0 = 1e-3;
    double lambda_up = 10.0;
    double lambda_down = 0.2;
    double step_tol = 1e-10;
    double grad_tol = 1e-10;
    bool verbose = true;
};

struct LMSummary {
    bool success = false;
    int iters = 0;
    double final_cost = 0.0;
};

using ResidualFunction = std::function<arma::vec(const std::vector<SE3>&)>;

std::vector<SE3> solveLMSE3(
    const ResidualFunction& fun,
    const std::vector<SE3>& init,
    const LMOptions& opts,
    LMSummary* summary_out
);

}  // namespace pem
