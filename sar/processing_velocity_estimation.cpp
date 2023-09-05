#include "processing_velocity_estimation.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "sar_metadata.h"
#include "util/geo_tools.h"
#include "util/math_utils.h"

namespace {
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

double EstimateVrAtRange(const SARMetadata& metadata, int az_idx, int range_idx) {
    double R0 = metadata.slant_range_first_sample + range_idx * metadata.range_spacing;
    const int aperture_size = CalcAperturePixels(metadata, range_idx);
    const int N = 9;

    auto center_osv = InterpolateOrbit(metadata.osv, CalcAzimuthTime(metadata, az_idx));

    auto earth_point =
        RangeDopplerGeoLocate({center_osv.x_vel, center_osv.y_vel, center_osv.z_vel},
                              {center_osv.x_pos, center_osv.y_pos, center_osv.z_pos}, metadata.center_point, R0);

    const int step = aperture_size / (N - 1);

    // calculate distance between center point and positions on the aperture
    // goal is to find data points from real orbit state vector for the hyperbolic range function
    // R^2(n) = R0^2 + Vr^2 * (n)
    // R - slant range across azimuth time points
    // R0 - slant range and closes point
    // Vr - processing / effective radar velocity
    // (n) - relative azimuth time
    Eigen::VectorXd y_vals(N);  // Vr
    Eigen::VectorXd x_vals(N);  // t
    for (int i = 0; i < N; i++) {
        const int aperture_az_idx = az_idx + (i - (N / 2)) * step;
        ;
        auto idx_time = CalcAzimuthTime(metadata, aperture_az_idx);
        auto osv = InterpolateOrbit(metadata.osv, idx_time);
        double dx = osv.x_pos - earth_point.x;
        double dy = osv.y_pos - earth_point.y;
        double dz = osv.z_pos - earth_point.z;
        double R_square = dx * dx + dy * dy + dz * dz;
        double R0_square = R0 * R0;
        double dt = (center_osv.time - osv.time).total_microseconds() * 1e-6;

        y_vals[i] = R_square - R0_square;
        x_vals[i] = dt;
    }

    // Now we have calculated data, where slant range to center point varies with azimuth time

    // mathematically the vectors now contain data points for the equation
    // y = ax^2 + c, the best data fit for a gives us an estimate for Vr^2
    // Now we have calculated data, where slant range to center point varies with azimuth time

    return SquareFitVr(x_vals, y_vals);
}

}  // namespace

std::vector<double> EstimateProcessingVelocity(const SARMetadata& metadata) {
    constexpr int N_Vr_CALC = 10;  // TODO investigate
    constexpr int POLY_ORDER = 2;

    const int az_idx = metadata.img.azimuth_size / 2;  // TODO How much does Vr vary in azimuth direction?
    const int range_start = 0;
    const int range_end = (metadata.img.range_size - metadata.chirp.n_samples);

    const int step = (range_end - range_start) / N_Vr_CALC;

    std::vector<double> Vr_results;
    std::vector<double> idx_vec;
    for (int range_idx = range_start; range_idx < range_end; range_idx += step) {
        double Vr = EstimateVrAtRange(metadata, az_idx, range_idx);
        Vr_results.push_back(Vr);
        idx_vec.push_back(range_idx);
    }

    auto Vr_poly = Polyfit(idx_vec, Vr_results, POLY_ORDER);

    printf("Vr result(range sample - m/s):\n");
    for (int i = 0; i < N_Vr_CALC; i++) {
        if (i && (i % 4) == 0) {
            printf("\n");
        }
        printf("(%5d - %5.2f ) ", static_cast<int>(idx_vec.at(i)), Vr_results.at(i));
    }

    printf("\nFitted polynomial\n");
    for (double e : Vr_poly) {
        printf("%g ", e);
    }
    printf("\n");

    return Vr_poly;
}