/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "envisat_mph_sph_parser.h"

#include "envisat_lvl0_parser.h"

#include <sstream>

#include "fmt/format.h"

#include "alus_log.h"
#include "asar_constants.h"
#include "envisat_aux_file.h"
#include "envisat_im_parse.h"
#include "envisat_mph_sph_str_utils.h"
#include "ers_im_parse.h"
#include "sar/orbit_state_vector.h"

namespace {

double NadirLLParse(const std::string& str) {
    auto val = std::stol(str.substr(3, 8));
    if (str.at(0) == '-') {
        val = -val;
    }
    return val * 1e-6;
}

}  // namespace

namespace alus::asar::envformat {

int SwathIdx(const std::string& swath) {
    if (swath.size() == 3 && swath[0] == 'I' && swath[1] == 'S') {
        char idx = swath[2];
        if (idx >= '1' && idx <= '7') {
            return idx - '1';
        }
    }
    throw std::runtime_error("MDSR contains swath '" + swath + "' which is not supported or invalid.");
}

DSD_lvl0 ParseSphAndGetMdsr(ASARMetadata& asar_meta, const SARMetadata& sar_meta, const std::vector<char>& file_data) {
    LOGD << "Product name = " << asar_meta.product_name;
    LOGD << "Input SENSING_START=" << asar_meta.sensing_start;
    LOGD << "Input SENSING_END=" << asar_meta.sensing_stop;
    if (sar_meta.osv.size() < 8) {
        throw std::runtime_error(
            "Atleast 8 OSVs are required in order to accomplish precise interpolation, currently " +
            std::to_string(sar_meta.osv.size()) +
            " are supplied. Check OSV sources and supply one which has more overlap.");
    }
    LOGD << "ORBIT vectors start " << sar_meta.osv.front().time;
    LOGD << "ORBIT vectors end " << sar_meta.osv.back().time;
    if (asar_meta.sensing_start < sar_meta.osv.front().time || asar_meta.sensing_stop > sar_meta.osv.back().time) {
        throw std::runtime_error("Given OSV source timespan does not cover dataset");
    }

    ProductHeader sph = {};
    sph.Load(file_data.data() + MPH_SIZE, SPH_SIZE);

    // LOGV << "SPH =";
    // sph.PrintValues();

    asar_meta.start_nadir_lat = NadirLLParse(sph.Get("START_LAT"));
    asar_meta.start_nadir_lon = NadirLLParse(sph.Get("START_LONG"));
    asar_meta.stop_nadir_lat = NadirLLParse(sph.Get("STOP_LAT"));
    asar_meta.stop_nadir_lon = NadirLLParse(sph.Get("STOP_LONG"));

    asar_meta.ascending = asar_meta.start_nadir_lat < asar_meta.stop_nadir_lat;

    asar_meta.swath = sph.Get("SWATH");

    asar_meta.polarization = sph.Get("TX_RX_POLAR");

    auto dsds = ExtractDSDs(sph);
    if (dsds.size() < 1) {
        throw std::runtime_error("Given input dataset does not contain echos.");
    }

    return dsds.front();
}

DSD_lvl0 ParseMdsr(const std::vector<char>& file_data) {
    ProductHeader sph = {};
    sph.Load(file_data.data() + MPH_SIZE, SPH_SIZE);
    auto dsds = ExtractDSDs(sph);
    return dsds.front();
}

std::vector<ForecastMeta> FetchMeta(const std::vector<char>& file_data, boost::posix_time::ptime packets_start_filter,
                                    boost::posix_time::ptime packets_stop_filter,
                                    specification::ProductTypes product_type, size_t& packets_before_start,
                                    size_t& packets_after_stop) {
    const auto mdsr = ParseMdsr(file_data);
    if (product_type == specification::ASA_IM0) {
        return FetchEnvisatL0ImForecastMeta(file_data, mdsr, packets_start_filter, packets_stop_filter,
                                            packets_before_start, packets_after_stop);
    } else if (product_type == specification::SAR_IM0) {
        return FetchErsL0ImForecastMeta(file_data, mdsr, packets_start_filter, packets_stop_filter,
                                        packets_before_start, packets_after_stop);
    } else {
        throw std::invalid_argument("Unsupported product type supplied for level 0 packet parsing.");
    }
}

void ParseLevel0Header(const std::vector<char>& file_data, ASARMetadata& asar_meta) {
    ProductHeader mph = {};

    mph.Load(file_data.data(), MPH_SIZE);

    // LOGV << "Lvl1MPH =";
    // mph.PrintValues();

    asar_meta.product_name = mph.Get("PRODUCT");
    asar_meta.sensing_start = StrToPtime(mph.Get("SENSING_START"));
    asar_meta.sensing_stop = StrToPtime(mph.Get("SENSING_STOP"));
    asar_meta.acquistion_station = mph.Get("ACQUISITION_STATION");
    asar_meta.processing_station = mph.Get("PROC_CENTER");
    asar_meta.rel_orbit = std::stoul(mph.Get("REL_ORBIT"));
    asar_meta.abs_orbit = std::stoul(mph.Get("ABS_ORBIT"));
    asar_meta.phase = mph.Get("PHASE").front();
    asar_meta.cycle = std::stol(mph.Get("CYCLE"));
    asar_meta.utc_sbt_time = StrToPtime(mph.Get("UTC_SBT_TIME"));
    asar_meta.sat_binary_time = std::stoll(mph.Get("SAT_BINARY_TIME"));
    asar_meta.clock_step = std::stoll(mph.Get("CLOCK_STEP"));
    asar_meta.leap_utc = StrToPtime(mph.Get("LEAP_UTC"));
    asar_meta.leap_sign = std::stol(mph.Get("LEAP_SIGN"));
    asar_meta.leap_err = mph.Get("LEAP_ERR").front();
    asar_meta.product_err = mph.Get("PRODUCT_ERR").front() == '0' ? false : true;
}

RawSampleMeasurements ParseLevel0Packets(const std::vector<char>& file_data, size_t mdsr_offset_bytes,
                                         const std::vector<ForecastMeta>& entries_to_be_parsed,
                                         alus::asar::specification::ProductTypes product_type, InstrumentFile& ins_file,
                                         std::vector<CommonPacketMetadata>& common_metadata) {
    if (entries_to_be_parsed.size() != common_metadata.size()) {
        throw std::runtime_error(
            fmt::format("Programming error detected - the packet count {} which are to be parsed does not equal given "
                        "metadata entry count {}",
                        entries_to_be_parsed.size(), common_metadata.size()));
    }

    if (product_type == specification::ASA_IM0) {
        return ParseEnvisatLevel0ImPackets(file_data, mdsr_offset_bytes, entries_to_be_parsed, ins_file,
                                           common_metadata);
    } else if (product_type == specification::SAR_IM0) {
        return ParseErsLevel0ImPackets(file_data, mdsr_offset_bytes, entries_to_be_parsed, ins_file, common_metadata);
    } else {
        throw std::invalid_argument("Unsupported product type supplied for level 0 packet parsing.");
    }
}

void ParseConfFile(const ConfigurationFile& con_file, SARMetadata& sar_meta, ASARMetadata& asar_meta) {
    {
        const char* rg_window = reinterpret_cast<const char*>(&con_file.range_win_type_ims.window_type[0]);

        if (strncmp(rg_window, "NONE", 4) == 0) {
            sar_meta.chirp.apply_window = false;
        } else if (strncmp(rg_window, "HAMMING", 7) == 0) {
            sar_meta.chirp.apply_window = true;
            // NOTE alpha does not make sense with hamming? It should be fixed to 0.54?
            sar_meta.chirp.alpha = con_file.range_win_type_ims.window_coeff;
        } else {
            LOGW << "Defaulting range HAMMING WINDOW unknown = "
                 << std::string(rg_window, std::size(con_file.range_win_type_ims.window_type));
            sar_meta.chirp.apply_window = true;
            sar_meta.chirp.alpha = 0.75;
        }
    }

    /* TODO default to no az window for now, needs further study in range_doppler_algorithm.cu
    {
        const char* az_window = reinterpret_cast<const char*>(&con_file.az_win_type_ims.window_type[0]);

        if (strncmp(az_window, "NONE", 4) == 0) {
            sar_meta.azimuth_window = false;
        } else if (strncmp(az_window, "HAMMING", 7) == 0) {
            sar_meta.azimuth_window = true;
            sar_meta.azimuth_window_alpha = con_file.az_win_type_ims.window_coeff;
        } else {
            LOGW << "Defaulting azimuth HAMMING WINDOW, unknown = "
                 << std::string(az_window, std::size(con_file.az_win_type_ims.window_type));
            sar_meta.chirp.apply_window = true;
            sar_meta.azimuth_window_alpha = 0.75;
        }
    }
    */

    // TODO - Per product type?
    const float az_bw = con_file.tot_azimuth_bandw_ims[asar_meta.swath_idx];
    sar_meta.azimuth_bandwidth_fraction = az_bw / sar_meta.pulse_repetition_frequency;

    LOGD << "processed azimuth bandwidth " << az_bw << " Hz , fraction = " << sar_meta.azimuth_bandwidth_fraction;

    auto& sq = asar_meta.summary_quality;
    sq.thresh_chirp_broadening = con_file.thresh_chirp_broadening;
    sq.thresh_chirp_sidelobe = con_file.thresh_chirp_sidelobe;
    sq.thresh_chirp_islr = con_file.thresh_chirp_islr;
    sq.thresh_input_mean = con_file.thresh_input_mean;
    sq.exp_input_mean = con_file.input_mean;
    sq.thresh_input_std_dev = con_file.thresh_input_std_dev;
    sq.exp_input_std_dev = con_file.input_std_dev;
    sq.thresh_dop_cen = con_file.thresh_dop_cen;
    sq.thresh_dop_amb = con_file.thresh_dop_amb;
    sq.thresh_output_mean = con_file.thresh_output_mean;
    if (asar_meta.target_product_type == specification::ProductTypes::ASA_IMS) {
        sq.exp_output_mean = con_file.exp_im_mean;
    } else if (asar_meta.target_product_type == specification::ProductTypes::ASA_IMP) {
        sq.exp_output_mean = con_file.exp_imp_mean;
    }
    sq.thresh_output_std_dev = con_file.thresh_output_std_dev;
    if (asar_meta.target_product_type == specification::ProductTypes::ASA_IMS) {
        sq.exp_output_std_dev = con_file.exp_im_std_dev;
    } else if (asar_meta.target_product_type == specification::ProductTypes::ASA_IMP) {
        sq.exp_output_std_dev = con_file.exp_imp_std_dev;
    }
    sq.thresh_input_missing_lines = con_file.thresh_missing_lines;
    sq.thresh_input_gaps = con_file.thresh_gaps;
    sq.lines_per_gaps = con_file.lines_per_gap;
}

}  // namespace alus::asar::envformat
