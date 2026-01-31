#include "lm.h"
#include <iostream>

namespace pem {

static arma::mat numericalJacobian(
    const ResidualFunction& fun,
    const std::vector<SE3>& vars,
    const arma::vec& r0,
    double eps
) {
    const int K = static_cast<int>(vars.size());
    const int D = 6 * K;
    const int M = static_cast<int>(r0.n_rows);

    arma::mat J(M, D, arma::fill::zeros);

    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < 6; ++d) {
            Vec6 delta; delta.zeros();
            delta(d) = eps;

            std::vector<SE3> vars_pert = vars;
            SE3 dT = se3Exp(delta);
            vars_pert[k] = dT * vars[k];

            arma::vec r1 = fun(vars_pert);
            J.col(6 * k + d) = (r1 - r0) / eps;
        }
    }
    return J;
}

std::vector<SE3> solveLMSE3(
    const ResidualFunction& fun,
    const std::vector<SE3>& init,
    const LMOptions& opts,
    LMSummary* summary_out
) {
    const int K = static_cast<int>(init.size());
    const int D = 6 * K;

    std::vector<SE3> best_vars;
    double best_cost = std::numeric_limits<double>::infinity();
    LMSummary best_sum;

    for (int restart = 0; restart < opts.max_restarts; ++restart) {
        std::vector<SE3> vars = init;

        if (restart > 0) {
            for (int k = 0; k < K; ++k) {
                vars[k] = randomSE3(0.8, 1.0);
            }
        }

        double lambda = opts.lambda0;

        arma::vec r = fun(vars);
        double cost = 0.5 * arma::dot(r, r);

        if (opts.verbose) {
            std::cout << "  restart " << restart << " cost0 " << cost << std::endl;
        }

        LMSummary sum;
        sum.iters = 0;

        for (int iter = 0; iter < opts.max_iters; ++iter) {
            arma::mat J = numericalJacobian(fun, vars, r, opts.fd_eps);

            arma::mat H = J.t() * J;
            arma::vec g = J.t() * r;

            double gnorm = arma::norm(g);
            if (gnorm < opts.grad_tol) {
                sum.success = true;
                sum.iters = iter;
                break;
            }

            arma::mat H_lm = H + lambda * arma::eye(D, D);
            arma::vec delta = arma::solve(H_lm, -g, arma::solve_opts::fast);

            if (arma::norm(delta) < opts.step_tol) {
                sum.success = true;
                sum.iters = iter;
                break;
            }

            std::vector<SE3> vars_new = vars;
            for (int k = 0; k < K; ++k) {
                Vec6 dxi = delta.subvec(6 * k, 6 * k + 5);
                SE3 dT = se3Exp(dxi);
                vars_new[k] = dT * vars[k];
            }

            arma::vec r_new = fun(vars_new);
            double cost_new = 0.5 * arma::dot(r_new, r_new);

            if (cost_new < cost) {
                vars = vars_new;
                r = r_new;
                cost = cost_new;
                lambda = std::max(1e-12, lambda * opts.lambda_down);
            } else {
                lambda = std::min(1e12, lambda * opts.lambda_up);
            }

            if (opts.verbose && (iter % 10 == 0)) {
                std::cout << "    iter " << iter
                          << " cost " << cost
                          << " lambda " << lambda
                          << " gnorm " << gnorm
                          << std::endl;
            }

            sum.iters = iter;
        }

        sum.final_cost = cost;

        if (cost < best_cost) {
            best_cost = cost;
            best_vars = vars;
            best_sum = sum;
        }
    }

    if (summary_out) {
        *summary_out = best_sum;
        summary_out->final_cost = best_cost;
    }
    return best_vars;
}

}  // namespace pem
