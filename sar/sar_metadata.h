#pragma once

#include <boost/date_time/posix_time/posix_time.hpp>
#include <vector>

#include "sar/orbit_state_vector.h"
#include "util/geo_tools.h"
#include "util/math_utils.h"

struct ImgDimensions {
    int range_size;
    int range_stride;
    int azimuth_size;
    int azimuth_stride;
};

struct ChirpInfo {
    double range_sampling_rate;
    double pulse_duration;
    int n_samples;
    double pulse_bandwidth;
    double Kr;  // Linear FM rate of transmitted pulse
};

struct SARResults {
    double dc_i;
    double dc_q;
    double iq_gain;
    double phase_error;                         // degrees
    std::vector<double> Vr_poly;                // fitted to range index
    std::vector<double> doppler_centroid_poly;  // fitted to range index
};

struct SARMetadata {
    ChirpInfo chirp;
    ImgDimensions img;
    std::string polarisation;
    std::vector<uint32_t> slant_range_times;
    std::vector<uint32_t> left_range_offsets;
    boost::posix_time::ptime first_orbit_time;
    double orbit_interval;
    std::vector<OrbitStateVector> osv;
    double pulse_repetition_frequency;
    double azimuth_bandwidth_fraction;
    double carrier_frequency;
    double wavelength;
    double platform_velocity;
    double range_spacing;
    double azimuth_spacing;
    double slant_range_first_sample;
    double slatn_range_first_time;
    boost::posix_time::ptime first_line_time;
    boost::posix_time::ptime center_time;

    GeoPos3D center_point;
    size_t total_raw_samples;

    SARResults results;
};

inline boost::posix_time::ptime CalcAzimuthTime(const SARMetadata& meta, int azimuth_idx) {
    uint32_t us = (1 / meta.pulse_repetition_frequency) * azimuth_idx * 1e6;
    return meta.first_line_time + boost::posix_time::microseconds(us);
}

inline double CalcR0(const SARMetadata& metadata, int range_pixel) {
    return metadata.slant_range_first_sample + range_pixel * metadata.range_spacing;
}

inline double CalcVr(const SARMetadata& metadata, int range_pixel) {
    const auto& p = metadata.results.Vr_poly;
    return Polyval(p.data(), p.size(), range_pixel);
}

inline double CalcDopplerCentroid(const SARMetadata& metadata, int range_pixel) {
    const auto& p = metadata.results.doppler_centroid_poly;
    return Polyval(p.data(), p.size(), range_pixel);
}

inline double CalcKa(const SARMetadata& metadata, int range_pixel) {
    constexpr double SOL = 299792458;
    const double Vr = CalcVr(metadata, range_pixel);
    const double R0 = CalcR0(metadata, range_pixel);
    return (2 * metadata.carrier_frequency * Vr * Vr) / (SOL * R0);
}

inline int CalcAperturePixels(const SARMetadata& metadata, int range_pixel) {
    double Ka = CalcKa(metadata, range_pixel);
    double fmax = (CalcDopplerCentroid(metadata, range_pixel) +
                   (metadata.pulse_repetition_frequency * metadata.azimuth_bandwidth_fraction) / 2);
    double pixels = (fmax / Ka) * metadata.pulse_repetition_frequency;

    return std::round(pixels);
}