#include "math_utils.h"


#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

std::vector<double> polyfit(const std::vector<double>& x_vals, const std::vector<double>& y_vals, int order) {
    Eigen::MatrixXd a(x_vals.size(), order + 1);

    for (int row = 0; row < a.rows(); row++) {
        double val = 1.0;
        double mult = x_vals[row];
        for (int col = 0; col < a.cols(); col++) {
            a(row, col) = val;
            val *= mult;
        }
    }

    Eigen::Map<Eigen::VectorXd> b(const_cast<double*>(y_vals.data()), y_vals.size());

    auto q = a.householderQr();
    Eigen::VectorXd result = q.solve(b);

    // order coefficients as highest power first, e.g same as polyfit in python
    result.reverseInPlace();

    return {result.data(), result.data() + result.size()};
}