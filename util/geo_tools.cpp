/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/
#include "geo_tools.h"

#include "Eigen/Dense"

#include "alus_log.h"

GeoPos3D RangeDopplerGeoLocate(Velocity3D velocity, GeoPos3D position, GeoPos3D init_guess, double slant_range) {
    constexpr double a = 6378137.0;
    constexpr double b = 6356752.314245;

    Eigen::Vector3d result = {init_guess.x, init_guess.y, init_guess.z};
    Eigen::Vector3d sensor_vel = {velocity.x, velocity.y, velocity.z};
    Eigen::Vector3d sensor_pos = {position.x, position.y, position.z};

    int iter = 0;
    while (1) {
        Eigen::Matrix3d A(3, 3);

        Eigen::Vector3d pos_vec = result - sensor_pos;

        A.row(0) = sensor_vel;
        A.row(1) = 2.0 * pos_vec;
        A(2, 0) = (2 * result[0]) / (a * a);
        A(2, 1) = (2 * result[1]) / (a * a);
        A(2, 2) = (2 * result[2]) / (b * b);

        Eigen::Vector3d B;
        B[0] = sensor_vel.dot(pos_vec);
        B[1] = pos_vec.dot(pos_vec) - slant_range * slant_range;
        B[2] = pow(result[0] / a, 2.0) + pow(result[1] / a, 2.0) + pow(result[2] / b, 2.0) - 1.0;

        // TODO better comments
        Eigen::Vector3d dx = A.fullPivHouseholderQr().solve(-B);

        result = result + dx;
        iter++;

        if (dx.norm() < 1e-6) {
            // LOGV << "geocode iter = " << iter;
            return {result[0], result[1], result[2]};
        }

        if (iter > 100) {
            LOGE << "failed to converge!";
            exit(1);
        }
    }
}
