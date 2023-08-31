#pragma once

#include <string_view>
#include <vector>

#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include <complex>
#include <map>
#include <variant>

#include "doris_orbit.h"
#include "envisat_types.h"
#include "sar/orbit_state_vector.h"
#include "sar/sar_metadata.h"
#include "util/geo_tools.h"
#include "util/math_utils.h"

/**
 * Envisat LVL0 IM file parser, ex ASA_IM__0PNPDE20040111_085939_000000182023_00179_09752_2307.N1
 * Extracts RAW echo data and relevant metadata needed for SAR focussing
 */

struct ASARMetadata {
    std::string lvl0_file_name;
    std::string product_name;
    std::string instrument_file;
    std::string configuration_file;
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

void ParseIMFile(const std::vector<char> &file_data, const char *aux_pax, SARMetadata &sar_meta,
                 ASARMetadata &asar_meta, std::vector<std::complex<float>> &img_data,
                 alus::dorisorbit::Parsable &orbit_source);
