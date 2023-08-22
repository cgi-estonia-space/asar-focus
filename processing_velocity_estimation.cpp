#include "processing_velocity_estimation.h"


#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "geo_tools.h"
#include "asar_lvl0_parser.h"

namespace{
double SquareFitVr(const Eigen::VectorXd& xvals, const Eigen::VectorXd& yvals) {
    Eigen::MatrixXd A(xvals.size(), 2);

    A.setZero();

    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (int j = 0; j < xvals.size(); j++) {
        A(j, 1) = xvals(j) * xvals(j);
    }

    auto Q = A.fullPivHouseholderQr();
    auto result = Q.solve(yvals);

    return sqrt(result[1]);
}

double CalcDistance(OrbitStateVector pos, GeoPos3D xyz) {
    double dx = pos.x_pos - xyz.x;
    double dy = pos.y_pos - xyz.y;
    double dz = pos.z_pos - xyz.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace


double EstimateProcessingVelocity(const SARMetadata& metadata) {
    const double PRI = 1 / metadata.pulse_repetition_frequency;
    auto center_time_point = metadata.center_time;
    auto center_point = metadata.center_point;

    const int center_az_idx = metadata.img.azimuth_size / 2;
    const int center_range_idx = metadata.img.range_size / 2;


    auto center_osv = InterpolateOrbit(metadata.osv, center_time_point);

    double min = CalcDistance(center_osv, center_point);
    int min_idx = center_az_idx;
    OrbitStateVector min_pos = center_osv;

    // find the closest Orbit state vector to center point from both directions
    for (int i = center_az_idx + 1; i < metadata.img.azimuth_size; i++) {

        auto idx_time = CalcAzimuthTime(metadata, i);
        auto pos = InterpolateOrbit(metadata.osv, idx_time);

        const double new_dist = CalcDistance(pos, center_point);
        if (new_dist < min) {
            min = new_dist;
            min_idx = i;
            min_pos = pos;
        } else {
            break;
        }
    }

    for (int i = center_az_idx - 1; i > 0; i--) {
        auto idx_time = CalcAzimuthTime(metadata, i);
        auto pos = InterpolateOrbit(metadata.osv, idx_time);
        const double new_dist = CalcDistance(pos, center_point);
        if (new_dist < min) {
            min = new_dist;
            min_idx = i;
            min_pos = pos;
        } else {
            break;
        }
    }

    double R0 = metadata.slant_range_first_sample + (metadata.img.range_size / 2) * metadata.range_spacing;

    const int aperature_size = CalcAperturePixels(metadata, center_range_idx);

    const int N = 9;

    const int step = aperature_size / (N - 1);

    // calculate distance between center point and positisons on the aperture
    // goal is to find data points from real orbit state vector for the hyperbolic range function
    // R^2(n) = R0^2 + Vr^2 * (n)
    // R - slant range across azimuth time points
    // R0 - slant range and closes point
    // Vr - processing / effective radar velocity
    // (n) - relative azimuth time
    Eigen::VectorXd y_vals(N);  // Vr
    Eigen::VectorXd x_vals(N);  // t
    for (int i = 0; i < N; i++) {

        const int az_idx = min_idx +  (i - (N / 2)) * step;
        auto idx_time = CalcAzimuthTime(metadata, az_idx);
        auto osv = InterpolateOrbit(metadata.osv, idx_time);
        double dx = osv.x_pos - center_point.x;
        double dy = osv.y_pos - center_point.y;
        double dz = osv.z_pos - center_point.z;
        double R_square = dx * dx + dy * dy + dz * dz;
        double R0_square = R0 * R0;
        double dt = (min_pos.time - osv.time).total_microseconds() * 1e-6;



        y_vals[i] = R_square - R0_square;
        x_vals[i] = dt;
    }

    // Now we have calculated data, where slant range to center point varies with azimuth time

    // mathematically the vectors now contain data points for the equation
    // y = ax^2 + c, best data fit for a gives us an estimate for Vr^2

    return SquareFitVr(x_vals, y_vals);
}