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
        return {};
        // return ParseErsLevel0ImPackets(file_data, mdsr, sar_meta, asar_meta, d_parsed_packets, ins_file,
        // packets_start_filter,
        //                         packets_stop_filter);
    } else {
        throw std::invalid_argument("Unsupported product type supplied for level 0 packet parsing.");
    }
}

// void ParseLevel0Packets(const std::vector<char>& file_data, SARMetadata& sar_meta, ASARMetadata& asar_meta,
//                         cufftComplex** d_parsed_packets, alus::asar::specification::ProductTypes product_type,
//                         InstrumentFile& ins_file, boost::posix_time::ptime packets_start_filter,
//                         boost::posix_time::ptime packets_stop_filter) {
//     if (product_type == specification::ASA_IM0) {
//         const auto& mdsr = ParseSphAndGetMdsr(asar_meta, sar_meta, file_data);
//         ParseEnvisatLevel0ImPackets(file_data, mdsr, sar_meta, asar_meta, d_parsed_packets, ins_file,
//                                     packets_start_filter, packets_stop_filter);
//     } else if (product_type == specification::SAR_IM0) {
//         const auto& mdsr = ParseSphAndGetMdsr(asar_meta, sar_meta, file_data);
//         ParseErsLevel0ImPackets(file_data, mdsr, sar_meta, asar_meta, d_parsed_packets, ins_file,
//         packets_start_filter,
//                                 packets_stop_filter);
//     } else {
//         throw std::invalid_argument("Unsupported product type supplied for level 0 packet parsing.");
//     }
// }

}  // namespace alus::asar::envformat
