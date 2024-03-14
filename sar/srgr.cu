#include "srgr.cuh"

#include <fmt/format.h>

#include "alus_log.h"
#include "checks.h"
#include "cuda_util.h"
#include "geo_tools.h"
#include "math_utils.h"
#include "orbit_state_vector.h"
#include "plot.h"

struct KernelArgs {
    float grsr_poly[5];
    int out_x_size;
    int in_x_size;
    int y_size;
    float slant_range_spacing;
    float ground_spacing;
};

template <int N>
__device__ inline float Polyval(float (&p)[N], float x) {
    float r = 0.0f;
    for (auto e : p) {
        r *= x;
        r += e;
    }
    return r;
}

__global__ void SrGrInterpolation(const float* data_in, KernelArgs args, float* data_out) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int out_idx = y * args.out_x_size + x;

    const float ground_distance = x * args.ground_spacing;

    const float range_distance = Polyval(args.grsr_poly, ground_distance);
    const float range_idx = range_distance / args.slant_range_spacing;
    const int range_start = static_cast<int>(range_idx);
    const float dx = range_idx - range_start;
    // linear interpolation... for now
    const float val0 = data_in[y * args.in_x_size + range_start];
    const float val1 = data_in[y * args.in_x_size + range_start + 1];
    const float interp = val0 + dx * (val1 - val0);

    data_out[out_idx] = interp;
}

void SlantRangeToGroundRangeInterpolation(SARMetadata& sar_meta, const float* d_slant_in, float* d_gr_out) {
    constexpr int N = 100;
    const int range_size = sar_meta.img.range_size;
    auto az_time = CalcAzimuthTime(sar_meta, sar_meta.img.azimuth_size / 2);
    auto osv = InterpolateOrbit(sar_meta.osv, az_time);
    const Velocity3D vel = {osv.x_vel, osv.y_vel, osv.z_vel};
    const GeoPos3D pos = {osv.x_pos, osv.y_pos, osv.z_pos};
    auto first_pos = RangeDopplerGeoLocate(vel, pos, sar_meta.center_point, sar_meta.slant_range_first_sample);
    std::vector<double> ground_distances;
    std::vector<double> slant_ranges;

    // generate a mapping of slant -> ground range
    for (int i = 0; i < N; i++) {
        const double R = CalcSlantRange(sar_meta, i * (range_size / 100));
        const auto ground_point = RangeDopplerGeoLocate(vel, pos, sar_meta.center_point, R);
        const double dx = ground_point.x - first_pos.x;
        const double dy = ground_point.y - first_pos.y;
        const double dz = ground_point.z - first_pos.z;
        const double distance = sqrt(dx * dx + dy * dy + dz * dz);
        slant_ranges.push_back(R - sar_meta.slant_range_first_sample);
        ground_distances.push_back(distance);
    }

    constexpr float ground_spacing = 12.5f;  // move to argument, for now IMP mode uses fixed 12.5m

    const int output_range_size = ground_distances.back() / ground_spacing;

    // fit the GR <-> SR conversion into a polynomial, to be used for interpolation
    std::vector<double> grsr_poly = Polyfit(ground_distances, slant_ranges, 4);

    std::string coeff_str = "[";
    for(const auto& e : grsr_poly) {
        coeff_str += fmt::format("{} ", e);
    }
    coeff_str += " ]";
    LOGI << "GRSR polynomial = " << coeff_str;

    KernelArgs args = {};
    args.ground_spacing = ground_spacing;
    args.slant_range_spacing = sar_meta.range_spacing;
    args.in_x_size = sar_meta.img.range_size;
    args.out_x_size = output_range_size;
    args.y_size = sar_meta.img.azimuth_size;

    sar_meta.srgr.applied = true;
    sar_meta.srgr.grsr_poly = grsr_poly;

    CHECK_BOOL(std::size(args.grsr_poly) == grsr_poly.size());
    std::copy(grsr_poly.begin(), grsr_poly.end(), args.grsr_poly);

    dim3 block_size(16, 16);
    dim3 grid_size((args.out_x_size + 15) / 16, (args.y_size + 15) / 16);

    SrGrInterpolation<<<grid_size, block_size>>>(d_slant_in, args, d_gr_out);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
    // Plot({"/tmp/srgr.html", "SRGR", "ground", "slant", {{"...", ground_distances, slant_ranges}}});

    sar_meta.range_spacing = ground_spacing;
    sar_meta.img.range_size = output_range_size;
}