/*
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c)
 * by CGI Estonia AS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "math_utils.h"

#include "Eigen/Core"
#include "Eigen/Dense"

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
