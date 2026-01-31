#pragma once
#include <vector>
#include "se3.h"

namespace pem {

struct AXXBData {
    std::vector<SE3> A;
    std::vector<SE3> B;
};

struct AXYBData {
    std::vector<SE3> A;
    std::vector<SE3> B;
};

struct AXB_YCZData {
    std::vector<SE3> A;
    std::vector<SE3> B;
    std::vector<SE3> C;
};

struct AXBY_ZCWDData {
    std::vector<SE3> A;
    std::vector<SE3> B;
    std::vector<SE3> C;
    std::vector<SE3> D;
};

AXXBData makeAXXBData(const SE3& X_true, int N, double noise);
AXYBData makeAXYBData(const SE3& X_true, const SE3& Y_true, int N, double noise);
AXB_YCZData makeAXB_YCZData(const SE3& X_true, const SE3& Y_true, const SE3& Z_true, int N, double noise);
AXBY_ZCWDData makeAXBY_ZCWDData(const SE3& X_true, const SE3& Y_true, const SE3& Z_true, const SE3& W_true, int N, double noise);

arma::vec residualAXXB_PEM(const AXXBData& data, const std::vector<SE3>& vars);
arma::vec residualAXYB_PEM(const AXYBData& data, const std::vector<SE3>& vars);
arma::vec residualAXB_YCZ_PEM(const AXB_YCZData& data, const std::vector<SE3>& vars);
arma::vec residualAXBY_ZCWD_PEM(const AXBY_ZCWDData& data, const std::vector<SE3>& vars);

}  // namespace pem
