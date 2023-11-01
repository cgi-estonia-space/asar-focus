/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */
#pragma once

#include <string_view>
#include <vector>

#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include <complex>
#include <map>
#include <variant>

#include "asar_constants.h"
#include "doris_orbit.h"
#include "envisat_aux_file.h"
#include "envisat_types.h"
#include "geo_tools.h"
#include "math_utils.h"
#include "sar/orbit_state_vector.h"
#include "sar/sar_metadata.h"

/**
 * Envisat LVL0 IM file parser, ex ASA_IM__0PNPDE20040111_085939_000000182023_00179_09752_2307.N1
 * Extracts RAW echo data and relevant metadata needed for SAR focussing
 */

struct ASARMetadata {
    std::string lvl0_file_name;
    std::string product_name;
    std::string instrument_file;
    std::string configuration_file;
    std::string orbit_dataset_name;
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

    struct {
        //            SetChar(phase, "PHASE", 'X');
        //            SetAc(cycle, "CYCLE", 0);
        //            SetAs(rel_orbit, "REL_ORBIT", 0);
        //            SetAs(abs_orbit, "ABS_ORBIT", 0);
        //            SetStr(state_vector_time, "STATE_VECTOR_TIME", "");
        //            strcpy((char*)&delta_ut1[0], "DELTA_UT1=+.281903<s>");  // TODO
        //            delta_ut1[sizeof(delta_ut1) - 1] = '\n';
        //            SetAdo73(x_position, "X_POSITION", 0.0, "<m>");
        //            SetAdo73(y_position, "Y_POSITION", 0.0, "<m>");
        //            SetAdo73(z_position, "Z_POSITION", 0.0, "<m>");
        //            SetAdo46(x_velocity, "X_VELOCITY", 0.0, "<m/s>");
        //            SetAdo46(y_velocity, "Y_VELOCITY", 0.0, "<m/s>");
        //            SetAdo46(z_velocity, "Z_VELOCITY", 0.0, "<m/s>");
        //            SetStr(vector_source, "VECTOR_SOURCE", "PC");
        std::string orbit_name;
        std::string processing_centre;
        std::string processing_time;
        char phase;
        boost::posix_time::ptime state_vector_time;
        std::string vector_source;
        uint32_t cycle;
        uint32_t rel_orbit;
        uint32_t abs_orbit;
        double delta_ut1;
        double x_position;
        double y_position;
        double z_position;
        double x_velocity;
        double y_velocity;
        double z_velocity;
    } orbit_metadata;

    struct {
        std::string echo_method;
        std::string echo_ratio;
        std::string init_cal_method;
        std::string init_cal_ratio;
        std::string per_cal_method;
        std::string per_cal_ratio;
        std::string noise_method;
        std::string noise_ratio;
    } compression_metadata;
};

namespace alus::asar::envformat {

void ParseLevel0Header(const std::vector<char>& file_data, ASARMetadata& asar_meta);
void ParseLevel0Packets(const std::vector<char>& file_data, SARMetadata& sar_meta, ASARMetadata& asar_meta,
                        std::vector<std::complex<float>>& img_data,
                        alus::asar::specification::ProductTypes product_type, InstrumentFile& ins_file,
                        boost::posix_time::ptime packets_start_filter = boost::posix_time::not_a_date_time,
                        boost::posix_time::ptime packets_stop_filter = boost::posix_time::not_a_date_time);

}  // namespace alus::asar::envformat