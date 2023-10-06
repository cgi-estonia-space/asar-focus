#include "math_utils.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

std::vector<double> Polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree) {
    Eigen::MatrixXd a(x.size(), degree + 1);

    for (int row = 0; row < a.rows(); row++) {
        double val = 1.0;
        double mult = x[row];
        for (int col = 0; col < a.cols(); col++) {
            a(row, col) = val;
            val *= mult;
        }
    }

    Eigen::Map<Eigen::VectorXd> b(const_cast<double*>(y.data()), y.size());

    auto q = a.householderQr();
    Eigen::VectorXd result = q.solve(b);

    // order coefficients as highest power first, e.g same as Polyfit in python
    result.reverseInPlace();

    return {result.data(), result.data() + result.size()};
}
