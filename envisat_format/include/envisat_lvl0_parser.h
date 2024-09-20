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

#include <cstdint>
#include <map>
#include <string_view>
#include <variant>
#include <vector>

#include <cufft.h>

#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include "asar_constants.h"
#include "doris_orbit.h"
#include "envisat_aux_file.h"
#include "envisat_types.h"
#include "geo_tools.h"
#include "math_utils.h"
#include "sar/orbit_state_vector.h"
#include "sar/sar_metadata.h"

struct ASARMetadata {
    std::string lvl0_file_name;
    std::string product_name;
    alus::asar::specification::ProductTypes target_product_type;
    std::string instrument_file;
    std::string configuration_file;
    std::string external_calibration_file;
    std::string external_characterization_file;
    std::string orbit_dataset_name;
    std::string acquistion_station;
    std::string processing_station;
    char phase;
    uint32_t rel_orbit;
    uint32_t abs_orbit;
    boost::posix_time::ptime sensing_start;
    boost::posix_time::ptime sensing_stop;
    boost::posix_time::ptime first_line_time;
    boost::posix_time::ptime last_line_time;
    bool product_err{false};

    std::string swath;
    int swath_idx;
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

    struct {
        fl thresh_chirp_broadening;
        fl thresh_chirp_sidelobe;
        fl thresh_chirp_islr;
        fl thresh_input_mean;
        fl exp_input_mean;
        fl thresh_input_std_dev;
        fl exp_input_std_dev;
        fl thresh_dop_cen;
        fl thresh_dop_amb;
        fl thresh_output_mean;
        fl exp_output_mean;
        fl thresh_output_std_dev;
        fl exp_output_std_dev;
        fl thresh_input_missing_lines;
        fl thresh_input_gaps;
    } summary_quality;
};

namespace alus::asar::envformat {

int SwathIdx(const std::string& swath);

struct ForecastMeta {
    size_t packet_start_offset_bytes;
    mjd isp_sensing_time;
};

struct CommonPacketMetadata {
    mjd sensing_time;
    uint16_t swst_code;
    uint16_t pri_code;
    uint64_t onboard_time;
    uint16_t sample_count;                    // Count of measurement I/Q samples, not equal to bytes.
    uint16_t measurement_array_length_bytes;  // Including any block information etc...

    struct {
        uint8_t chirp_pulse_bw_code;
        uint8_t upconverter_raw;
        uint8_t downconverter_raw;
        uint16_t tx_pulse_code;
        uint8_t antenna_beam_set_no;
        uint8_t beam_adj_delta;
        uint16_t resampling_factor;
    } asar;
};

struct RawSampleMeasurements {
    std::unique_ptr<uint8_t[]> raw_samples;
    size_t single_entry_length;  // Length in bytes of the maximum sample measurement record, including block id etc.
    size_t entries_total;        // How many lines in azimuth direction
    size_t max_samples;          // Maximum samples that the entries will consist. Can fluctuate.
    size_t total_samples;        // Total IQ samples collected over all of the range and azimuth directions.
    size_t no_of_product_errors_compensated;
};

DSD_lvl0 ParseSphAndGetMdsr(ASARMetadata& asar_meta, const SARMetadata& sar_meta, const std::vector<char>& file_data);

/* Current silent contract is that every packet of the returned list of ForecastMeta shall be parsed by the
 * ParseLevel0Packets. That also means that any duplicates and/or missing packets shall be compensated already
 * during FetchMeta. Parsing packets would just respect the offsets. But it can throw if there are discrepancies,
 * which would indicate programming error. Current FetchMeta is simple byte offsets starting from MDSR (not L0 ds)
 * and sensing time. Later more fields could be added and parsed to implement more comprehensive parsing scheme.
 */
std::vector<ForecastMeta> FetchMeta(const std::vector<char>& file_data, boost::posix_time::ptime packets_start_filter,
                                    boost::posix_time::ptime packets_stop_filter,
                                    specification::ProductTypes product_type, size_t& packets_before_start,
                                    size_t& packets_after_stop);

void ParseLevel0Header(const std::vector<char>& file_data, ASARMetadata& asar_meta);

RawSampleMeasurements ParseLevel0Packets(const std::vector<char>& file_data, size_t mdsr_offset_bytes,
                                         const std::vector<ForecastMeta>& entries_to_be_parsed,
                                         alus::asar::specification::ProductTypes product_type, InstrumentFile& ins_file,
                                         std::vector<CommonPacketMetadata>& common_metadata);

void ParseConfFile(const ConfigurationFile& con_file, SARMetadata& sar_meta, ASARMetadata& asar_meta);

}  // namespace alus::asar::envformat