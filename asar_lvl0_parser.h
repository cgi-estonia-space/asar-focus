#pragma once

#include <vector>

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time.hpp>

#include <complex>
#include <variant>
#include <map>

#include "envisat_types.h"
#include "geo_tools.h"
#include "orbit_state_vector.h"

struct ImgDimensions
{
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
    double coefficient[5];
};


struct ImgFormat {
    int range_size;
    int azimuth_size;
    int data_line_offset;
    int record_length;
};

struct SARResults {
    size_t total_samples;
    double dc_i;
    double dc_q;
    double iq_gain;
    double quad_depart;
    double Vr;
    double doppler_centroid;
};


struct ASARMetadata
{
    std::string lvl0_file_name;
    std::string product_name;
    std::string instrument_file;
    std::string acquistion_station;
    std::string processing_station;
    boost::posix_time::ptime sensing_start;
    boost::posix_time::ptime sensing_stop;
    boost::posix_time::ptime first_line_time;
    boost::posix_time::ptime last_line_time;

    std::string swath;
    std::string polarization;

    double two_way_slant_range_time;

    uint32_t first_swst_code;
    uint32_t last_swst_code;
    uint32_t swst_changes;
    uint32_t pri_code;
    uint32_t tx_pulse_len_code;
    uint32_t tx_bw_code;
    uint32_t echo_win_len_code;
    uint32_t up_code;
    uint32_t down_code;
    uint32_t resamp_code;
    uint32_t beam_adj_code;
    uint32_t beam_set_num_code;
    uint32_t tx_monitor_code;

    uint8_t swst_rank;

    uint64_t first_sbt;
    mjd first_mjd;
    uint64_t last_sbt;
    mjd last_mjd;


    double start_nadir_lat;
    double start_nadir_lon;
    double stop_nadir_lat;
    double stop_nadir_lon;
    bool ascending;
};




struct SARMetadata {
    ChirpInfo chirp;
    ImgDimensions img;
    std::string polarisation;
    std::vector<uint32_t> slant_range_times;
    std::vector<uint32_t> left_range_offsets;
    boost::posix_time::ptime first_orbit_time;
    double orbit_interval;
    std::vector<OrbitInfo> orbit_state_vectors;
    std::vector<OrbitStateVector> osv;
    double pulse_repetition_frequency;
    double azimuth_bandwidth_fraction;
    double carrier_frequency;
    double wavelength;
    OrbitInfo first_position;
    double platform_velocity;
    double range_spacing;
    double azimuth_spacing;
    double slant_range_first_sample;
    double slatn_range_first_time;
    boost::posix_time::ptime first_line_time;
    boost::posix_time::ptime center_time;

    GeoPos3D center_point;

    SARResults results;

};

inline boost::posix_time::ptime CalcAzimuthTime(const SARMetadata& meta, int azimuth_idx)
{
    uint32_t us = (1/meta.pulse_repetition_frequency) * azimuth_idx * 1e6;
    printf("az idx = %d , us = %u\n", azimuth_idx, us);
    return meta.first_line_time + boost::posix_time::microseconds(us);
}

inline double CalcR0(const SARMetadata& metadata, int range_pixel) {
    return metadata.slant_range_first_sample + range_pixel * metadata.range_spacing;
}

inline double CalcKa(const SARMetadata& metadata, int range_pixel) {
    constexpr double SOL = 299792458;
    const double Vr = metadata.results.Vr;
    const double R0 = CalcR0(metadata, range_pixel);
    return (2 * metadata.carrier_frequency * Vr * Vr) / (SOL * R0);
}

inline int CalcAperturePixels(const SARMetadata& metadata, int range_pixel) {
    double Ka = CalcKa(metadata, range_pixel);
    double fmax = (metadata.results.doppler_centroid +
            (metadata.pulse_repetition_frequency * metadata.azimuth_bandwidth_fraction) / 2);
    double pixels = (fmax / Ka) * metadata.pulse_repetition_frequency;

    return std::round(pixels);
}

void ParseIMFile(const std::vector<char>& file_data, const char* aux_pax, SARMetadata& sar_meta, ASARMetadata& asar_meta,  std::vector<std::complex<float>>& img_data);




