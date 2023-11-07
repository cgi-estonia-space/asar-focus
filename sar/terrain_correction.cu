//
// Created by priit on 11/6/23.
//

#include "terrain_correction.cuh"

#include "terrain_correction.cuh"

#include "util/include/plot.h"

#include "fmt/format.h"

struct Vec3d {
    double x;
    double y;
    double z;
};

#if 0
namespace WGS84 {                                        // NOLINT
constexpr double A = 6378137.0;                          // m
constexpr double B = 6356752.3142451794975639665996337;  // 6356752.31424518; // m
constexpr double FLAT_EARTH_COEF = 1.0 / ((A - B) / A);  // 298.257223563;
constexpr double E2 = 2.0 / FLAT_EARTH_COEF - 1.0 / (FLAT_EARTH_COEF * FLAT_EARTH_COEF);
// constexpr double EP2 = E2 / (1 - E2);
}  // namespace WGS84

constexpr double DTOR = M_PI / 180.0;
// constexpr double RTOD = 180 / M_PI;
#endif
inline __device__ Vec3d Geo2xyzWgs84Device(double latitude, double longitude, double altitude) {
    double const lat = latitude * DTOR;
    double const lon = longitude * DTOR;

    double sin_lat;
    double cos_lat;
    sincos(lat, &sin_lat, &cos_lat);

    double const sinLat = sin_lat;

    double const N = (WGS84::A / sqrt(1.0 - WGS84::E2 * sinLat * sinLat));
    double const NcosLat = (N + altitude) * cos_lat;

    double sin_lon;
    double cos_lon;
    sincos(lon, &sin_lon, &cos_lon);

    Vec3d r = {};
    r.x = NcosLat * cos_lon;
    r.y = NcosLat * sin_lon;
    r.z = (N + altitude - WGS84::E2 * N) * sinLat;
    return r;
}

struct OrbitStateVectorCalc {
    double time_point;
    double x_pos;
    double y_pos;
    double z_pos;
    double x_vel;
    double y_vel;
    double z_vel;
};

// kargs pwd

struct KernelArgs {
    double lon_start;
    double lat_start;
    double pixel_spacing_y;
    double pixel_spacing_x;
    double first_line_utc;
    double line_time_interval;
    double wavelength;

    double range_spacing;
    double slant_range_first_sample;

    Vec3d* sensor_position;
    Vec3d* sensor_velocity;

    DEM dem;

    OrbitStateVectorCalc* osv;
    int n_osv;
};

inline __device__ __host__ double getDopplerFrequency(Vec3d earthPoint, Vec3d sensorPosition, Vec3d sensorVelocity,
                                                      double wavelength) {
    const auto xDiff = earthPoint.x - sensorPosition.x;
    const auto yDiff = earthPoint.y - sensorPosition.y;
    const auto zDiff = earthPoint.z - sensorPosition.z;
    const auto distance = sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);

    return 2.0 * (sensorVelocity.x * xDiff + sensorVelocity.y * yDiff + sensorVelocity.z * zDiff) /
           (distance * wavelength);
}

inline __device__ double GetEarthPointZeroDopplerTimeImpl(double first_line_utc, double line_time_interval,
                                                          double wavelength, Vec3d earth_point, int n_azimuth,
                                                          const Vec3d* sensor_position, const Vec3d* sensor_velocity,
                                                          bool ) {
    // binary search is used in finding the zero doppler time
    int lower_bound = 0;
    int upper_bound = n_azimuth - 1;
    auto lower_bound_freq =
        getDopplerFrequency(earth_point, sensor_position[lower_bound], sensor_velocity[lower_bound], wavelength);
    auto upper_bound_freq =
        getDopplerFrequency(earth_point, sensor_position[upper_bound], sensor_velocity[upper_bound], wavelength);


    if (std::abs(lower_bound_freq) < 1.0) {
        return first_line_utc + lower_bound * line_time_interval;
    } else if (std::abs(upper_bound_freq) < 1.0) {
        return first_line_utc + upper_bound * line_time_interval;
    } else if (lower_bound_freq * upper_bound_freq > 0.0) {
        return NAN;
    }

    // start binary search
    double mid_freq;
    while (upper_bound - lower_bound > 1) {
        const auto mid = (int)((lower_bound + upper_bound) / 2.0);
        mid_freq = sensor_velocity[mid].x * (earth_point.x - sensor_position[mid].x) +
                   sensor_velocity[mid].y * (earth_point.y - sensor_position[mid].y) +
                   sensor_velocity[mid].z * (earth_point.z - sensor_position[mid].z);

        if (mid_freq * lower_bound_freq > 0.0) {
            lower_bound = mid;
            lower_bound_freq = mid_freq;
        } else if (mid_freq * upper_bound_freq > 0.0) {
            upper_bound = mid;
            upper_bound_freq = mid_freq;
        } else if (mid_freq == 0.0) {
            return first_line_utc + mid * line_time_interval;
        }
    }

    const auto y0 =
        lower_bound - lower_bound_freq * (upper_bound - lower_bound) / (upper_bound_freq - lower_bound_freq);
    return first_line_utc + y0 * line_time_interval;
}

inline __device__ __host__ Vec3d GetPosition(double time, OrbitStateVectorCalc* vectors, int n_osv, bool ) {
    const int nv{8};
    const int vectorsSize = n_osv;
    // TODO: This should be done once.
    const double dt =
        (vectors[vectorsSize - 1].time_point - vectors[0].time_point) / static_cast<double>(vectorsSize - 1);

    int i0;
    int iN;
    if (vectorsSize <= nv) {
        i0 = 0;
        iN = static_cast<int>(vectorsSize - 1);
    } else {
        i0 = std::max((int)((time - vectors[0].time_point) / dt) - nv / 2 + 1, 0);
        iN = std::min(i0 + nv - 1, vectorsSize - 1);
        i0 = (iN < vectorsSize - 1 ? i0 : iN - nv + 1);
    }

    // i0 = 0;
    // iN = n_osv;
    Vec3d result{0, 0, 0};
    for (int i = 0; i <= n_osv; ++i) {
        auto const orbI = vectors[i];

        double weight = 1;
        for (int j = i0; j <= iN; ++j) {
            if (j != i) {
                double const time2 = vectors[j].time_point;
                weight *= (time - time2) / (orbI.time_point - time2);
            }
        }
        result.x += weight * orbI.x_pos;
        result.y += weight * orbI.y_pos;
        result.z += weight * orbI.z_pos;
    }
    return result;
}

__device__ float GetAltitude(double latitude, double longitude, DEM dem, bool) {
    const double dem_spacing = 5 / 6000.0;
    if (latitude < dem.y_orig - 5.0 || latitude > dem.y_orig) {
        return NAN;
    }
    if (longitude < dem.x_orig || longitude > dem.x_orig + 5.0) {
        return NAN;
    }

    int x = std::round((longitude - dem.x_orig) / dem_spacing);
    int y = std::round((dem.y_orig - latitude) / dem_spacing);

    int16_t val = dem.d_data[x + y * 6000];
    if (val == -32768) {
        return NAN;
    }
    return val;
}

__device__ float ToIntens(cufftComplex val) { return val.x * val.x + val.y * val.y; }

__global__ void TerrainCorrectionKernel(float* out, int out_x_size, int out_y_size, const cufftComplex* in,
                                        int in_range_sz, int in_range_stride, int in_az_size, KernelArgs args) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x > out_x_size || y > out_y_size) {
        return;
    }

    bool debug = false;

    if (x == out_x_size / 2 && y == out_y_size / 2) {
        debug = true;
    }

    const int out_idx = x + y * out_x_size;

    // const int data_idx = y * range_stride + x;

    // find output pixel coordinates
    const double lat = args.lat_start + y * args.pixel_spacing_y + 0.5 * args.pixel_spacing_y;
    const double lon = args.lon_start + x * args.pixel_spacing_x + 0.5 * args.pixel_spacing_x;

    // elevation

    float altitude = GetAltitude(lat, lon, args.dem, debug);

    if (std::isnan(altitude)) {
        out[out_idx] = 0.0f;
        return;
    }

    // lat lon -> XYZ

    auto earth_point = Geo2xyzWgs84Device(lat, lon, altitude);

    // find azimuth time

    double az_time =
        GetEarthPointZeroDopplerTimeImpl(args.first_line_utc, args.line_time_interval, args.wavelength, earth_point,
                                         in_az_size, args.sensor_position, args.sensor_velocity, debug);
    // azimuth idx
    double az_idx = (az_time - args.first_line_utc) / args.line_time_interval;


    if (std::isnan(az_idx) || az_idx < 0.0 || az_idx >= in_az_size) {
        out[out_idx] = 0.0f;
        return;
    }

    // slant range


    Vec3d sat_pos = GetPosition(az_time, args.osv, args.n_osv, debug);


    // range idx
    double dx = earth_point.x - sat_pos.x;
    double dy = earth_point.y - sat_pos.y;
    double dz = earth_point.z - sat_pos.z;
    double slant_range = sqrt(dx * dx + dy * dy + dz * dz);
    double rg_idx = (slant_range - args.slant_range_first_sample) / args.range_spacing;


    if (rg_idx >= in_range_sz || rg_idx < 0.0) {
        out[out_idx] = 0.0f;
        return;
    }

    // interpolate

    int x_i = (int)rg_idx;
    int y_i = (int)az_idx;

    float val0 = ToIntens(in[y_i * in_range_stride + x_i]);
    float val1 = ToIntens(in[y_i * in_range_stride + x_i + 1]);
    float val2 = ToIntens(in[(y_i + 1) * in_range_stride + x_i]);
    float val3 = ToIntens(in[(y_i + 1) * in_range_stride + x_i + 1]);

    float interp_x = (float)rg_idx - x_i;
    float interp_y = (float)az_idx - y_i;

    float int_r0 = val0 + interp_x * (val1 - val0);
    float int_r1 = val2 + interp_x * (val3 - val2);
    float data = int_r0 + interp_y * (int_r1 - int_r0);

    // write result

    // out[out_idx] = 10.0f * logf(data);
    out[out_idx] = data;  // * 0.0f + 1.0f;
}

TCResult RDTerrainCorrection(const DevicePaddedImage& d_img, DEM dem, const SARMetadata& sar_meta) {
    std::vector<OrbitStateVectorCalc> h_osv(sar_meta.osv.size());

    for (size_t i = 0; i < h_osv.size(); i++) {
        h_osv[i].time_point = (sar_meta.osv[i].time - sar_meta.osv.front().time).total_microseconds() * 1e-6;
        h_osv[i].x_pos = sar_meta.osv[i].x_pos;
        h_osv[i].y_pos = sar_meta.osv[i].y_pos;
        h_osv[i].z_pos = sar_meta.osv[i].z_pos;
        h_osv[i].x_vel = sar_meta.osv[i].x_vel;
        h_osv[i].y_vel = sar_meta.osv[i].y_vel;
        h_osv[i].z_vel = sar_meta.osv[i].z_vel;
    }

    std::vector<Vec3d> h_pos(sar_meta.img.azimuth_size);
    std::vector<Vec3d> h_vel(sar_meta.img.azimuth_size);

    for (size_t i = 0; i < h_pos.size(); i++) {
        auto osv = InterpolateOrbit(sar_meta.osv, CalcAzimuthTime(sar_meta, i));
        h_pos[i] = {osv.x_pos, osv.y_pos, osv.z_pos};
        h_vel[i] = {osv.x_vel, osv.y_vel, osv.z_vel};
    }

    Vec3d* d_pos;
    cudaMalloc(&d_pos, sizeof(Vec3d) * h_pos.size());
    Vec3d* d_vel;
    cudaMalloc(&d_vel, sizeof(Vec3d) * h_vel.size());

    cudaMemcpy(d_pos, h_pos.data(), sizeof(Vec3d) * h_pos.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel.data(), sizeof(Vec3d) * h_vel.size(), cudaMemcpyHostToDevice);

    OrbitStateVectorCalc* d_osv;
    cudaMalloc(&d_osv, sizeof(OrbitStateVectorCalc) * h_osv.size());
    cudaMemcpy(d_osv, h_osv.data(), sizeof(OrbitStateVectorCalc) * h_osv.size(), cudaMemcpyHostToDevice);

    [[maybe_unused]] KernelArgs args = {};

    (void)d_img;

    args.line_time_interval = 1 / sar_meta.pulse_repetition_frequency;
    args.first_line_utc = (sar_meta.first_line_time - sar_meta.osv.front().time).total_microseconds() * 1e-6;
    args.wavelength = sar_meta.wavelength;
    args.slant_range_first_sample = sar_meta.slant_range_first_sample;
    args.range_spacing = sar_meta.range_spacing;
    args.dem = dem;

    std::vector<double> lats;
    std::vector<double> lons;

    std::vector<int> az_idx = {0, 0, sar_meta.img.azimuth_size - 1, sar_meta.img.azimuth_size - 1};
    std::vector<int> rg_idx = {0, sar_meta.img.range_size - 1, 0, sar_meta.img.range_size - 1};

    std::string corner_info = "SLC Corner points: ";
    for (int i = 0; i < 4; i++) {
        const int range_samp = rg_idx[i];
        double R = sar_meta.slant_range_first_sample + sar_meta.range_spacing * range_samp;
        auto osv = InterpolateOrbit(sar_meta.osv, CalcAzimuthTime(sar_meta, az_idx[i]));
        auto xyz = RangeDopplerGeoLocate({osv.x_vel, osv.y_vel, osv.z_vel}, {osv.x_pos, osv.y_pos, osv.z_pos},
                                         sar_meta.center_point, R);
        auto llh = xyz2geoWGS84(xyz);
        lats.push_back(llh.latitude);
        lons.push_back(llh.longitude);
        corner_info += fmt::format("{{{} , {}}} ", llh.longitude, llh.latitude);
    }

    LOGI << corner_info;

    double lat_max = *std::max_element(lats.begin(), lats.end());
    double lat_min = *std::min_element(lats.begin(), lats.end());
    double lon_max = *std::max_element(lons.begin(), lons.end());
    double lon_min = *std::min_element(lons.begin(), lons.end());

    LOGI << fmt::format("geobox = {} {} {} {}", lat_max, lat_min, lon_max, lon_min);

    double pixel_spacing_in_meter = 20;
    double pixel_spacing_in_degree = pixel_spacing_in_meter / WGS84::A * (180.0 / M_PI);

    double dif_lat = std::fabs(lat_max - lat_min);
    double dif_lon = std::fabs(lon_max - lon_min);

    int y_size = dif_lat / pixel_spacing_in_degree;
    int x_size = dif_lon / pixel_spacing_in_degree;

    size_t total = (size_t)x_size * y_size;

    float* d_result;
    cudaMalloc(&d_result, sizeof(float) * total);

    LOGI << fmt::format("Terrain correction output dimensions = {} {}, pixel spacing = {} m", x_size, y_size, pixel_spacing_in_meter);

    args.lat_start = lat_max;
    args.lon_start = lon_min;
    args.pixel_spacing_x = pixel_spacing_in_degree;
    args.pixel_spacing_y = -pixel_spacing_in_degree;

    args.sensor_position = d_pos;
    args.sensor_velocity = d_vel;
    args.osv = d_osv;
    args.n_osv = sar_meta.osv.size();

    dim3 block_size(16, 16);
    dim3 grid_size((x_size + 15) / 16, (y_size + 15) / 16);

    TerrainCorrectionKernel<<<grid_size, block_size>>>(d_result, x_size, y_size, d_img.Data(), d_img.XSize(),
                                                       d_img.XStride(), d_img.YSize(), args);

    TCResult ret = {};
    ret.y_size = y_size;
    ret.x_size = x_size;
    ret.x_start = lon_min;
    ret.y_start = lat_max;
    ret.x_spacing = pixel_spacing_in_degree;
    ret.y_spacing = -pixel_spacing_in_degree;
    ret.d_data = d_result;
    return ret;
}