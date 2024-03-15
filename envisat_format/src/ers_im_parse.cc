/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "ers_im_parse.h"

#include <stdexcept>
#include <string>

#include "fmt/format.h"

#include "alus_log.h"
#include "cuda_algorithm.h"
#include "cuda_util.h"
#include "envisat_format_kernels.h"
#include "envisat_utils.h"
#include "ers_env_format.h"
#include "ers_sbt.h"
#include "parse_util.h"

#define DEBUG_PACKETS 0
#define HANDLE_DIAGNOSTICS 1

namespace {

template <class T>
[[nodiscard]] const uint8_t* CopyBSwapPOD(T& dest, const uint8_t* src) {
    memcpy(&dest, src, sizeof(T));
    dest = bswap(dest);
    return src + sizeof(T);
}

struct EchoMeta {
    mjd isp_sensing_time;

    uint16_t isp_length;  // Part of the FEP annotation, but this is the only useful field, others are static fillers.

    uint32_t data_record_number;
    // IDHT header
    uint8_t packet_counter;
    uint8_t subcommutation_counter;

    // Auxiliary data and replica/calibration pulses
    uint64_t onboard_time;
    uint8_t activity_task;
    uint8_t sample_flags;
    uint32_t image_format_counter;
    uint16_t pri_code;
    uint16_t swst_code;
};

#if DEBUG_PACKETS

struct ErsFepAndPacketMetadata {
    EchoMeta echo_meta;
};

[[maybe_unused]] void DebugErsPackets(const std::vector<ErsFepAndPacketMetadata> dm, size_t limit,
                                      std::ostream& out_stream) {
    out_stream << "isp_sense,isp_sense_micros,isp_length,data_rec_no,packet_counter,subcomm_counter,onboard_time,"
               << "activity_task,image_format_counter,pri_code,swst_code" << std::endl;
    for (size_t i{}; i < limit; i++) {
        const auto& dmi = dm.at(i).echo_meta;

        const auto isp_sensing_time_ptime = MjdToPtime(dmi.isp_sensing_time);
        // clang-format off
        out_stream << boost::posix_time::to_simple_string(isp_sensing_time_ptime) << ","
                   << isp_sensing_time_ptime.time_of_day().total_microseconds() << ","
                   << dmi.isp_length << "," << dmi.data_record_number << "," << (uint32_t)dmi.packet_counter  << ","
                   << (uint32_t)dmi.subcommutation_counter << "," << dmi.onboard_time << ","
                   << dmi.activity_task << "," << dmi.image_format_counter << "," << dmi.pri_code << ","
                   << dmi.swst_code << std::endl;
        // clang-format on
    }
}

#endif

inline void FetchUint32(const uint8_t* array_start, uint32_t& var) {
    [[maybe_unused]] const auto discard = CopyBSwapPOD(var, array_start);
}

#if HANDLE_DIAGNOSTICS
constexpr auto DR_NO_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint32_t, 32>()};
constexpr auto PACKET_COUNTER_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint8_t, 8>()};
constexpr uint8_t SUBCOMMUTATION_COUNTER_MAX{48};
constexpr auto IMAGE_FORMAT_COUNTER_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint32_t, 24>()};
constexpr auto SBT_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint32_t, 32>()};

void InitializeDataRecordNo(const uint8_t* packet_start, uint32_t& dr_no) {
    FetchUint32(packet_start + alus::asar::envformat::ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES +
                    alus::asar::envformat::ers::highrate::FEP_ANNOTATION_SIZE_BYTES,
                dr_no);
    if (dr_no != 0) {
        dr_no--;
    } else {
        dr_no = DR_NO_MAX;
    }
}

void FetchSatelliteBinaryTime(const uint8_t* packet_start, uint32_t& sbt) {
    /*
         * The update of that binary counter shall occur every 4th PRI.
         * The time relation to the echo data in the format is as following:
         * the transfer of the ICU on-board time to the auxiliary memory
         * occurs t2 before the RF transmit pulse as depicted in
         * Fig. 4.4.2.4.6-3. The least significant bit is equal to 1/256 sec.
         *
         * ER-IS-ESA-GS-0002  - pg. 4.4 - 24
     */
    FetchUint32(packet_start + alus::asar::envformat::ers::highrate::SBT_OFFSET_BYTES, sbt);
}

void InitializeCounters(const uint8_t* packets_start, uint32_t& dr_no, uint8_t& packet_counter,
                        uint8_t& subcommutation_counter, uint32_t& image_format_counter) {
    InitializeDataRecordNo(packets_start, dr_no);
    // 36 for packet counter
    packet_counter = packets_start[alus::asar::envformat::ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES +
                                   alus::asar::envformat::ers::highrate::FEP_ANNOTATION_SIZE_BYTES +
                                   alus::asar::envformat::ers::highrate::DATA_RECORD_NUMBER_SIZE_BYTES];
    if (packet_counter == 0) {
        packet_counter = PACKET_COUNTER_MAX;
    } else {
        packet_counter--;
    }

    // 37 for subcommutation counter.
    subcommutation_counter = packets_start[alus::asar::envformat::ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES +
                                           alus::asar::envformat::ers::highrate::FEP_ANNOTATION_SIZE_BYTES +
                                           alus::asar::envformat::ers::highrate::DATA_RECORD_NUMBER_SIZE_BYTES + 1];
    if (subcommutation_counter == 0) {
        subcommutation_counter = SUBCOMMUTATION_COUNTER_MAX;
    } else {
        subcommutation_counter--;
    };

    FetchUint32(packets_start + alus::asar::envformat::ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES +
                    alus::asar::envformat::ers::highrate::FEP_ANNOTATION_SIZE_BYTES +
                    alus::asar::envformat::ers::highrate::DATA_RECORD_NUMBER_SIZE_BYTES + 18,
                image_format_counter);
    if (image_format_counter == 0) {
        image_format_counter = IMAGE_FORMAT_COUNTER_MAX;
    } else {
        image_format_counter--;
    }
}

// oos - out of sequence
struct PacketParseDiagnostics {
    size_t dr_no_oos;            // No. of times gap happened.
    size_t dr_no_total_missing;  // Total sum of all gaps' value.
    size_t packet_counter_oos;
    size_t packer_counter_total_missing;
    size_t packet_counter_changes;
    size_t subcommutation_counter_oos;
    size_t subcommutation_counter_total_missing;
    size_t image_format_counter_oos;
    size_t image_format_counter_total_missing;
};

#endif

inline void FetchPriCode(const uint8_t* start_array, uint16_t& var) {
    var |= static_cast<uint16_t>(start_array[0]) << 8;
    var |= start_array[1];
}

// This is deprecated, will be removed along with the initial parsing flow.
inline void CalculatePrf(uint16_t pri_code, double& pri, double& prf) {
    // ISCE2 uses 210.943006e-9, but
    // ERS-1-Satellite-to-Ground-Segment-Interface-Specification_01.09.1993_v2D_ER-IS-ESA-GS-0001_EECF05
    // note 5 in 4.4:28 specifies 210.94 ns, which draws more similar results to PF-ERS results.
    pri = (pri_code + 2.0) * 210.94e-9;
    prf = 1 / pri;

    if (prf < 1640 || prf > 1720) {
        LOGW << "PRF value '" << prf << "' is out of the specification (ERS-1-Satellite-to-Ground-Segment-Interface-"
             << "Specification_01.09.1993_v2D_ER-IS-ESA-GS-0001_EECF05) range [1640, 1720] Hz" << std::endl;
    }
}

inline const uint8_t* FetchPacketFrom(const uint8_t* start, alus::asar::envformat::ForecastMeta& meta, uint32_t& dr_no,
                                      uint32_t& sbt) {
    // https://earth.esa.int/eogateway/documents/20142/37627/ERS-products-specification-with-Envisat-format.pdf
    // https://asf.alaska.edu/wp-content/uploads/2019/03/ers_ceos.pdf and ER-IS-ESA-GS-0002 mixed between three.
    auto it = CopyBSwapPOD(meta.isp_sensing_time, start);
    it += 12;  // No MJD for ERS in FEP.
    uint16_t isp_length;
    it = CopyBSwapPOD(isp_length, it);
    InitializeDataRecordNo(start, dr_no);
    FetchSatelliteBinaryTime(start, sbt);

    // ISP size - 1.
    if (isp_length != 11465) {
        throw std::runtime_error("FEP annotation's ISP length shall be 11465. Check the dataset, current one has " +
                                 std::to_string(isp_length));
    }

    return start + alus::asar::envformat::ers::highrate::MDSR_SIZE_BYTES;
}

inline void TransferMetadata(const EchoMeta& echo_meta, alus::asar::envformat::CommonPacketMetadata& common_meta) {
    common_meta.sensing_time = echo_meta.isp_sensing_time;
    common_meta.onboard_time = echo_meta.onboard_time;
    common_meta.pri_code = echo_meta.pri_code;
    common_meta.swst_code = echo_meta.swst_code;
    common_meta.sample_count = alus::asar::envformat::ers::highrate::MEASUREMENT_DATA_SAMPLE_COUNT;
    common_meta.measurement_array_length_bytes = alus::asar::envformat::ers::highrate::MEASUREMENT_DATA_SIZE_BYTES;
}

}  // namespace

namespace alus::asar::envformat {

std::vector<ForecastMeta> FetchErsL0ImForecastMeta(const std::vector<char>& file_data, const DSD_lvl0& mdsr,
                                                   boost::posix_time::ptime packets_start_filter,
                                                   boost::posix_time::ptime packets_stop_filter,
                                                   size_t& packets_before_start, size_t& packets_after_stop) {
    if (mdsr.dsr_size != ers::highrate::MDSR_SIZE_BYTES) {
        throw std::runtime_error("Expected DSR to be " + std::to_string(ers::highrate::MDSR_SIZE_BYTES) +
                                 " bytes, actual - " + std::to_string(mdsr.dsr_size) +
                                 ". This is a failsafe abort, check the dataset.");
    }
    if (file_data.size() - mdsr.ds_offset != mdsr.num_dsr * mdsr.dsr_size) {
        throw std::runtime_error("Actual packets BLOB array (" + std::to_string(file_data.size() - mdsr.ds_offset) +
                                 " bytes) is not equal to dataset defined packets (" +
                                 std::to_string(mdsr.num_dsr * mdsr.dsr_size) + " bytes).");
    }

    std::vector<ForecastMeta> forecast_meta{};
    size_t packets_before_requested_start{};
    size_t packets_after_requested_stop{};
    const auto* const packets_start = reinterpret_cast<const uint8_t*>(file_data.data() + mdsr.ds_offset);
    const uint8_t* next_packet = reinterpret_cast<const uint8_t*>(file_data.data() + mdsr.ds_offset);
    const auto start_mjd = PtimeToMjd(packets_start_filter);
    const auto stop_mjd = PtimeToMjd(packets_stop_filter);

    size_t dr_no_oos{};
    size_t dr_no_total_missing{};
    uint32_t last_dr_no;
    InitializeDataRecordNo(next_packet, last_dr_no);
    uint32_t last_sbt;
    uint32_t sbt_repeat{0};
    FetchSatelliteBinaryTime(next_packet, last_sbt);

    size_t i{};
    for (; i < mdsr.num_dsr; i++) {
        ForecastMeta current_meta;
        uint32_t dr_no;
        current_meta.packet_start_offset_bytes = next_packet - packets_start;
        uint32_t sbt;
        next_packet = FetchPacketFrom(next_packet, current_meta, dr_no, sbt);
        const auto sbt_delta = parseutil::CounterGap<uint32_t, SBT_MAX>(last_sbt, sbt);
        if (sbt_delta == 0) {
            sbt_repeat++;
        } else {
            sbt_repeat = 0;
        }
        bool overflow = (sbt < last_sbt ? true : false);
        ers::AdjustIspSensingTime(current_meta.isp_sensing_time, sbt, sbt_repeat, overflow);

        const auto dr_no_gap = parseutil::CounterGap<uint32_t, DR_NO_MAX>(last_dr_no, dr_no);
        // When gap is 0, this is not handled - could be massive rollover or duplicate or the first packet.
        if (dr_no_gap > 1) {
            if (forecast_meta.size() < 1) {
                throw std::runtime_error(
                    "Implementation error detected - data record no. discontinuity detected, but no packets have been "
                    "fetched yet in order to fill packets");
            }

            const auto packets_to_fill = dr_no_gap - 1;
            dr_no_oos++;
            dr_no_total_missing += packets_to_fill;
            // 900 taken from
            // https://github.com/gmtsar/gmtsar/blob/master/preproc/ERS_preproc/ers_line_fixer/ers_line_fixer.c#L434
            // For ISCE2 this is 30000 which seems to be wayyyy too biig.
            if (packets_to_fill > 900) {
                throw std::runtime_error(
                    fmt::format("There are {} packets missing, which is more than threshold 900 in order to continue. "
                                "It was detected between {}th and {}th packets.",
                                packets_to_fill, i - 1, i));
            }
            const auto& previous_meta = forecast_meta.back();
            // TODO - sensing times should be recalculated as well.
            for (size_t pf{}; pf < packets_to_fill; pf++) {
                forecast_meta.push_back(previous_meta);
            }

            if (previous_meta.isp_sensing_time < start_mjd) {
                packets_before_requested_start += packets_to_fill;
            }

            if (previous_meta.isp_sensing_time > stop_mjd) {
                packets_after_requested_stop += packets_to_fill;
            }
        } else {
            if (current_meta.isp_sensing_time < start_mjd) {
                packets_before_requested_start++;
            }

            if (current_meta.isp_sensing_time > stop_mjd) {
                packets_after_requested_stop++;
            }
        }

        // In case the packets_after_stop is equal to 0. Delete one at the back.
        if (packets_after_requested_stop > packets_after_stop) {
            packets_after_requested_stop--;
            break;
        }

        last_dr_no = dr_no;
        forecast_meta.push_back(current_meta);
    }

    if (packets_before_requested_start > packets_before_start) {
        forecast_meta.erase(forecast_meta.begin(),
                            forecast_meta.begin() + (packets_before_requested_start - packets_before_start));
        packets_before_requested_start -= (packets_before_requested_start - packets_before_start);
        if (packets_before_start == 0) {
            auto it = forecast_meta.begin();
            while (it != forecast_meta.end()) {
                const auto sensing_time = it->isp_sensing_time;
                if (sensing_time < start_mjd) {
                    forecast_meta.erase(it);
                    it = forecast_meta.begin();
                } else {
                    break;
                }
            }
        }
    }

    packets_before_start = packets_before_requested_start;
    packets_after_stop = packets_after_requested_stop;

    LOGD << "Data record number discontinuities/total missing - " << dr_no_oos << "/" << dr_no_total_missing
         << " these were inserted as to be parsed/duplicated during prefetch";

    return forecast_meta;
}

RawSampleMeasurements ParseErsLevel0ImPackets(const std::vector<char>& file_data, size_t mdsr_offset_bytes,
                                              const std::vector<ForecastMeta>& entries_to_be_parsed, InstrumentFile&,
                                              std::vector<CommonPacketMetadata>& common_metadata) {
    const uint8_t* const packets_start = reinterpret_cast<const uint8_t*>(file_data.data()) + mdsr_offset_bytes;
    const uint8_t* it = packets_start + entries_to_be_parsed.front().packet_start_offset_bytes;
    const auto num_dsr = entries_to_be_parsed.size();

    // std::vector is chosen IF actual packets should exceed the num_dsr.
    const auto alloc_bytes_for_raw_samples = num_dsr * ers::highrate::MEASUREMENT_DATA_SIZE_BYTES;
    LOGD << "Reserving " << alloc_bytes_for_raw_samples / (1 << 20) << "MiB for raw samples ("
         << ers::highrate::MEASUREMENT_DATA_SIZE_BYTES << "x" << num_dsr << ")";

    RawSampleMeasurements raw_measurements{};
    raw_measurements.max_samples = ers::highrate::MEASUREMENT_DATA_SAMPLE_COUNT;
    raw_measurements.total_samples = ers::highrate::MEASUREMENT_DATA_SAMPLE_COUNT * num_dsr;
    raw_measurements.single_entry_length = ers::highrate::MEASUREMENT_DATA_SIZE_BYTES;
    raw_measurements.entries_total = num_dsr;
    raw_measurements.raw_samples = std::unique_ptr<uint8_t[]>(new uint8_t[alloc_bytes_for_raw_samples]);
    auto echoes_raw = raw_measurements.raw_samples.get();

#if DEBUG_PACKETS
    std::vector<ErsFepAndPacketMetadata> ers_dbg_meta;
    std::ofstream debug_stream("/tmp/" + asar_meta.product_name + ".debug.csv");
#endif
    // Data rec no. is used to detect missing lines as is in ISCE2. Image format counter seems to be suitable as well.
    uint32_t last_dr_no{};
    uint8_t last_packet_counter{};
    uint8_t last_subcommutation_counter{};
    uint32_t last_image_format_counter{};
    InitializeCounters(it, last_dr_no, last_packet_counter, last_subcommutation_counter, last_image_format_counter);
    uint32_t last_sbt{};
    FetchSatelliteBinaryTime(it, last_sbt);
#if HANDLE_DIAGNOSTICS
    PacketParseDiagnostics diagnostics{};
#endif

    size_t packet_index{};
    // https://earth.esa.int/eogateway/documents/20142/37627/ERS-products-specification-with-Envisat-format.pdf
    // https://asf.alaska.edu/wp-content/uploads/2019/03/ers_ceos.pdf and ER-IS-ESA-GS-0002 mixed between three.
    for (const auto& entry : entries_to_be_parsed) {
        it = packets_start + entry.packet_start_offset_bytes;
        EchoMeta echo_meta = {};

        uint32_t sbt;
        FetchSatelliteBinaryTime(it, sbt);
        it = CopyBSwapPOD(echo_meta.isp_sensing_time, it);
        bool overflow = (sbt < last_sbt ? true : false);
        // sbt_repeat is not handled, because no mechanism could be developed.
        // SBT stays constant for some period of packets due to being updated after every 4th PRI.
        // When dividing SBT update time period with PRI no sensible period did develop between packets,
        // hence this is not used right now. There would be minimal range lines skipped/included due to this anyway.
        ers::AdjustIspSensingTime(echo_meta.isp_sensing_time, sbt, 0, overflow);
        last_sbt = sbt;

        if (echo_meta.isp_sensing_time != entry.isp_sensing_time) {
            throw std::runtime_error(fmt::format(
                "Discrepancy between packet list '{}' and actual parsed packet sensing time '{}' at packet index {}",
                to_simple_string(MjdToPtime(entry.isp_sensing_time)),
                to_simple_string(MjdToPtime(echo_meta.isp_sensing_time)), packet_index));
        }

        // FEP annotation only consists of useful ISP length for ERS.
        it += 12;  // No MJD for ERS.
        it = CopyBSwapPOD(echo_meta.isp_length, it);
        it += 6;  // No CRC or correction count for ERS.

        // ISP size - 1.
        if (echo_meta.isp_length != 11465) {
            throw std::runtime_error("FEP annotation's ISP length shall be 11465. Check the dataset, current one has " +
                                     std::to_string(echo_meta.isp_length) + " for packet no " +
                                     std::to_string(packet_index + 1));
        }
        it = CopyBSwapPOD(echo_meta.data_record_number, it);

        // ER-IS-ESA-GS-0002 4.4.2.4.5
        /*
         * It is a binary counter which is incremented each time a new source packet of general IDHT header data is
         * transmitted (about every 1 second). It is reset to zero with power-on of the IDHT-DCU.
         * The first format transmitted after power-on will show the value "1". Bit O is defined as MSB. bit 7 as LSB.
         */
        echo_meta.packet_counter = it[0];
        /*
         * binary counter counting the 8 byte segments from 1 to 48; it is reset for each new source packet.
         * Bit o is defined as MSB, bit 7 as LSB.
         */
        echo_meta.subcommutation_counter = it[1];
        it += 2;

        /*
         * These segments ot 8 bytes contain the subcommutated IDHT General Header source packet.
         */
        it += 8;
        // Table 4.4.2.6-2 in ER-IS-ESA-GS-0002
        if (it[0] != 0xAA) {
            throw std::runtime_error(fmt::format(
                "ERS data packet's auxiliary section shall start with 0xAA, instead {:#x} was found for packet no {}",
                it[0], packet_index + 1));
        }
        it += 1;
        // OBRC - On-Board Range Compressed | OGRC - On-Ground Range Compressed - See ER-IS-EPO-GS-0201
        it += 1;  // bit 0 - OGRC/OBRC flag (0/1 respectively). Bits 1-4 Orbit number (0-15 -> 1-16)
        /*
         * The update of that binary counter shall occur every 4th PRI.
         * The time relation to the echo data in the format is as following:
         * the transfer of the ICU on-board time to the auxiliary memory
         * occurs t2 before the RF transmit pulse as depicted in
         * Fig. 4.4.2.4.6-3. The least significant bit is equal to 1/256 sec.
         *
         * ER-IS-ESA-GS-0002  - pg. 4.4 - 24
         */
        echo_meta.onboard_time = sbt;
        it += 4;
        /*
         * This word shall define the activity task within the mode of
         * operation, accompanied by the validity flag bits and the first
         * sample flag bits. The update of the activity task word is controlled by the 4th PRI interrupt.
         * ER-IS-ESA-GS-0002  - pg. 4.4 - 25
         *
         * Pg. 4.4-25
         * Activity means generation of noise, echo, calibration or replica data.
         *
         * MSB  LSB
         * 10001000 - Noise; no calibration
         * 10011001 - No echo; Cal. drift (EM only)
         * 10101001 - Echo; Cal. drift
         * 10101010 - Echo; Replica
         * 10100000 - Echo; no Replica (because of OBRC)
         */
        echo_meta.activity_task = it[0];
        /*
         * ...continued from activity task's pg. 4.4-25
         * Bit
         * 0 - echo invalid/valid (0/1)
         * 1 - Cal. data/Replica data invalid/valid (0/1)
         * 2 - Is noise
         * 3 - Is Cal/Repl.
         * 4 - Is echo
         * ... spare bits
         *
         * The bits 2 to 4 (noise/cal/repl/echo) of byte 23 can only be set or reset with a 4 x PRI interval,
         * therefore the start of a new activity is flagged for the first four formats.
         */
        echo_meta.sample_flags = it[1];
        it += 2;
        // Updated every format. Reset at the beginning of a transmission.
        it = CopyBSwapPOD(echo_meta.image_format_counter, it);
        it = CopyBSwapPOD(echo_meta.swst_code, it);
        it = CopyBSwapPOD(echo_meta.pri_code, it);
        // Next item - Cal. S/S attenuation select - ER-IS-ESA-GS-0002 Table 4.4.2.4.6-2
        // Drift calibration data
        // Lets skip all of that
        it += 194 + 10;

        const auto dr_no_gap = parseutil::CounterGap<uint32_t, DR_NO_MAX>(last_dr_no, echo_meta.data_record_number);
        // When gap is 0, it has been corrected by the previously fetched metadata.
        if (dr_no_gap == 0) {
            diagnostics.dr_no_oos++;
            diagnostics.dr_no_total_missing++;
        }
        last_dr_no = echo_meta.data_record_number;
#if HANDLE_DIAGNOSTICS
        const auto packet_counter_gap =
            parseutil::CounterGap<uint8_t, PACKET_COUNTER_MAX>(last_packet_counter, echo_meta.packet_counter);
        // Packet counter increments something like after 1679 packets?? based on IM L0 observation. When this happens
        // subcommutation counter also resets to 1.
        if (packet_counter_gap > 1) {
            diagnostics.packet_counter_oos++;
            diagnostics.packer_counter_total_missing += (packet_counter_gap - 1);
        }
        if (packet_counter_gap == 1) {
            diagnostics.packet_counter_changes++;
        }
        last_packet_counter = echo_meta.packet_counter;

        const auto subcommutation_gap = parseutil::CounterGap<uint8_t, SUBCOMMUTATION_COUNTER_MAX>(
            last_subcommutation_counter, echo_meta.subcommutation_counter);
        // When gap is 0, this is not handled - could be massive rollover or duplicate.
        // Also the counter does not start from 0.
        if (subcommutation_gap > 1 && echo_meta.subcommutation_counter != 1 && packet_counter_gap != 1) {
            diagnostics.subcommutation_counter_oos++;
            diagnostics.subcommutation_counter_total_missing += (subcommutation_gap - 1);
        }
        last_subcommutation_counter = echo_meta.subcommutation_counter;

        const auto image_format_counter_gap = parseutil::CounterGap<uint32_t, IMAGE_FORMAT_COUNTER_MAX>(
            last_image_format_counter, echo_meta.image_format_counter);
        if (image_format_counter_gap > 1) {
            diagnostics.image_format_counter_oos++;
            diagnostics.image_format_counter_total_missing += (image_format_counter_gap - 1);
        }
        last_image_format_counter = echo_meta.image_format_counter;
#endif

        std::memcpy(echoes_raw + packet_index * raw_measurements.single_entry_length, it,
                    raw_measurements.single_entry_length);
        TransferMetadata(echo_meta, common_metadata.at(packet_index));

        packet_index++;
#if DEBUG_PACKETS
        ErsFepAndPacketMetadata meta{};
        meta.echo_meta = echo_meta;
        ers_dbg_meta.push_back(meta);
#endif
    }

#if DEBUG_PACKETS
    DebugErsPackets(ers_dbg_meta, ers_dbg_meta.size(), debug_stream);
#endif

    if (diagnostics.dr_no_oos > 0) {
        raw_measurements.no_of_product_errors_compensated = diagnostics.dr_no_oos;
    }

    LOGD << "Data record number discontinuities/total missing - " << diagnostics.dr_no_oos << "/"
         << diagnostics.dr_no_total_missing << " - were detected during parsing.";

#if HANDLE_DIAGNOSTICS
    LOGD << "Packet counter field diagnostics - " << diagnostics.packet_counter_oos << " "
         << diagnostics.packer_counter_total_missing << " changes " << diagnostics.packet_counter_changes;
    LOGD << "Subcommutation counter field diagnostics - " << diagnostics.subcommutation_counter_oos << " "
         << diagnostics.subcommutation_counter_total_missing;
    LOGD << "Image format counter diagnostics - " << diagnostics.image_format_counter_oos << " "
         << diagnostics.image_format_counter_total_missing;
#endif

    return raw_measurements;
}

void ParseErsLevel0ImPackets(const std::vector<char>& file_data, const DSD_lvl0& mdsr, SARMetadata& sar_meta,
                             ASARMetadata& asar_meta, cufftComplex** d_parsed_packets, InstrumentFile& ins_file,
                             boost::posix_time::ptime packets_start_filter,
                             boost::posix_time::ptime packets_stop_filter) {
    const uint8_t* it = reinterpret_cast<const uint8_t*>(file_data.data()) + mdsr.ds_offset;
    if (mdsr.dsr_size != ers::highrate::MDSR_SIZE_BYTES) {
        throw std::runtime_error("Expected DSR to be " + std::to_string(ers::highrate::MDSR_SIZE_BYTES) +
                                 " bytes, actual - " + std::to_string(mdsr.dsr_size) +
                                 ". This is a failsafe abort, check the dataset.");
    }
    if (file_data.size() - mdsr.ds_offset != mdsr.num_dsr * mdsr.dsr_size) {
        throw std::runtime_error("Actual packets BLOB array (" + std::to_string(file_data.size() - mdsr.ds_offset) +
                                 " bytes) is not equal to dataset defined packets (" +
                                 std::to_string(mdsr.num_dsr * mdsr.dsr_size) + " bytes).");
    }
    // std::vector is chosen IF actual packets should exceed the num_dsr.
    std::vector<EchoMeta> echoes;
    echoes.reserve(mdsr.num_dsr);
    const auto alloc_bytes_for_raw_samples = mdsr.num_dsr * ers::highrate::MEASUREMENT_DATA_SIZE_BYTES;
    LOGD << "Reserving " << alloc_bytes_for_raw_samples / (1 << 20) << "MiB for raw samples ("
         << ers::highrate::MEASUREMENT_DATA_SIZE_BYTES << "x" << mdsr.num_dsr << ")";
    std::unique_ptr<uint8_t[]> echoes_raw(new uint8_t[alloc_bytes_for_raw_samples]);
#if DEBUG_PACKETS
    std::vector<ErsFepAndPacketMetadata> ers_dbg_meta;
    std::ofstream debug_stream("/tmp/" + asar_meta.product_name + ".debug.csv");
#endif
    // Data rec no. is used to detect missing lines as is in ISCE2. Image format counter seems to be suitable as well.
    uint32_t last_dr_no{};
    uint8_t last_packet_counter{};
    uint8_t last_subcommutation_counter{};
    uint32_t last_image_format_counter{};
    InitializeCounters(it, last_dr_no, last_packet_counter, last_subcommutation_counter, last_image_format_counter);
#if HANDLE_DIAGNOSTICS
    PacketParseDiagnostics diagnostics{};
#endif
    uint16_t first_pri_code{};
    FetchPriCode(it + 60, first_pri_code);
    double first_pri_us{};
    CalculatePrf(first_pri_code, first_pri_us, sar_meta.pulse_repetition_frequency);
    first_pri_us *= 10e5;

    //    mjd rolling_isp_sensing_time{};
    //    [[maybe_unused]] const auto discard_it = CopyBSwapPOD(rolling_isp_sensing_time, it);
    //    mjd stash_isp_change = rolling_isp_sensing_time;

    const auto start_filter_mjd = PtimeToMjd(packets_start_filter);
    const auto end_filter_mjd = PtimeToMjd(packets_stop_filter);

    // https://earth.esa.int/eogateway/documents/20142/37627/ERS-products-specification-with-Envisat-format.pdf
    // https://asf.alaska.edu/wp-content/uploads/2019/03/ers_ceos.pdf and ER-IS-ESA-GS-0002 mixed between three.
    for (size_t i = 0; i < mdsr.num_dsr; i++) {
        EchoMeta echo_meta = {};
#if HANDLE_DIAGNOSTICS
        const auto* it_start = it;
#endif
        it = CopyBSwapPOD(echo_meta.isp_sensing_time, it);
        //        if (echo_meta.isp_sensing_time == stash_isp_change) {
        //            echo_meta.isp_sensing_time = MjdAddMicros(rolling_isp_sensing_time, first_pri_us);
        //        } else {
        //            stash_isp_change = echo_meta.isp_sensing_time;
        //        }
        //        rolling_isp_sensing_time = echo_meta.isp_sensing_time;
        // FEP annotation only consists of useful ISP length for ERS.
        it += 12;  // No MJD for ERS.
        it = CopyBSwapPOD(echo_meta.isp_length, it);
        it += 6;  // No CRC or correction count for ERS.
        if (echo_meta.isp_sensing_time < start_filter_mjd) {
            it = it_start + ers::highrate::MDSR_SIZE_BYTES;
            // 32 offset
            FetchUint32(it_start + ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES +
                            ers::highrate::FEP_ANNOTATION_SIZE_BYTES,
                        last_dr_no);
#if HANDLE_DIAGNOSTICS
            // 36 offset (part of IDHT)
            last_packet_counter =
                it_start[ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES + ers::highrate::FEP_ANNOTATION_SIZE_BYTES +
                         ers::highrate::DATA_RECORD_NUMBER_SIZE_BYTES];
            // 37 offset (part of IDHT)
            last_subcommutation_counter =
                it_start[ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES + ers::highrate::FEP_ANNOTATION_SIZE_BYTES +
                         ers::highrate::DATA_RECORD_NUMBER_SIZE_BYTES + 1];
            // 54 offset
            FetchUint32(it_start + ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES +
                            ers::highrate::FEP_ANNOTATION_SIZE_BYTES + ers::highrate::DATA_RECORD_NUMBER_SIZE_BYTES +
                            18,
                        last_image_format_counter);
#endif
            continue;
        }

        if (echo_meta.isp_sensing_time > end_filter_mjd) {
            break;
        }

        // ISP size - 1.
        if (echo_meta.isp_length != 11465) {
            throw std::runtime_error("FEP annotation's ISP length shall be 11465. Check the dataset, current one has " +
                                     std::to_string(echo_meta.isp_length) + " for packet no " + std::to_string(i + 1));
        }
        it = CopyBSwapPOD(echo_meta.data_record_number, it);

        // ER-IS-ESA-GS-0002 4.4.2.4.5
        /*
         * It is a binary counter which is incremented each time a new source packet of general IDHT header data is
         * transmitted (about every 1 second). It is reset to zero with power-on of the IDHT-DCU.
         * The first format transmitted after power-on will show the value "1". Bit O is defined as MSB. bit 7 as LSB.
         */
        echo_meta.packet_counter = it[0];
        /*
         * binary counter counting the 8 byte segments from 1 to 48; it is reset for each new source packet.
         * Bit o is defined as MSB, bit 7 as LSB.
         */
        echo_meta.subcommutation_counter = it[1];
        it += 2;

        /*
         * These segments ot 8 bytes contain the subcommutated IDHT General Header source packet.
         */
        it += 8;
        // Table 4.4.2.6-2 in ER-IS-ESA-GS-0002
        if (it[0] != 0xAA) {
            throw std::runtime_error(fmt::format(
                "ERS data packet's auxiliary section shall start with 0xAA, instead {:#x} was found for packet no {}",
                it[0], i + 1));
        }
        it += 1;
        // OBRC - On-Board Range Compressed | OGRC - On-Ground Range Compressed - See ER-IS-EPO-GS-0201
        it += 1;  // bit 0 - OGRC/OBRC flag (0/1 respectively). Bits 1-4 Orbit number (0-15 -> 1-16)
        /*
         * The update of that binary counter shall occur every 4th PRI.
         * The time relation to the echo data in the format is as following:
         * the transfer of the ICU on-board time to the auxiliary memory
         * occurs t2 before the RF transmit pulse as depicted in
         * Fig. 4.4.2.4.6-3. The last significant bit is equal to 1/256 sec.
         *
         * ER-IS-ESA-GS-0002  - pg. 4.4 - 24
         */
        echo_meta.onboard_time = 0;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[0]) << 24;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[1]) << 16;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[2]) << 8;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[3]) << 0;
        it += 4;
        /*
         * This word shall define the activity task within the mode of
         * operation, accompanied by the validity flag bits and the first
         * sample flag bits. The update of the activity task word is controlled by the 4th PRI interrupt.
         * ER-IS-ESA-GS-0002  - pg. 4.4 - 25
         *
         * Pg. 4.4-25
         * Activity means generation of noise, echo, calibration or replica data.
         *
         * MSB  LSB
         * 10001000 - Noise; no calibration
         * 10011001 - No echo; Cal. drift (EM only)
         * 10101001 - Echo; Cal. drift
         * 10101010 - Echo; Replica
         * 10100000 - Echo; no Replica (because of OBRC)
         */
        echo_meta.activity_task = it[0];
        /*
         * ...continued from activity task's pg. 4.4-25
         * Bit
         * 0 - echo invalid/valid (0/1)
         * 1 - Cal. data/Replica data invalid/valid (0/1)
         * 2 - Is noise
         * 3 - Is Cal/Repl.
         * 4 - Is echo
         * ... spare bits
         *
         * The bits 2 to 4 (noise/cal/repl/echo) of byte 23 can only be set or reset with a 4 x PRI interval,
         * therefore the start of a new activity is flagged for the first four formats.
         */
        echo_meta.sample_flags = it[1];
        it += 2;
        // Updated every format. Reset at the beginning of a transmission.
        it = CopyBSwapPOD(echo_meta.image_format_counter, it);
        it = CopyBSwapPOD(echo_meta.swst_code, it);
        it = CopyBSwapPOD(echo_meta.pri_code, it);
        // Next item - Cal. S/S attenuation select - ER-IS-ESA-GS-0002 Table 4.4.2.4.6-2
        // Drift calibration data
        // Lets skip all of that
        it += 194 + 10;

        const auto dr_no_gap = parseutil::CounterGap<uint32_t, DR_NO_MAX>(last_dr_no, echo_meta.data_record_number);
        // When gap is 0, this is not handled - could be massive rollover or duplicate.
        if (dr_no_gap > 1) {
            diagnostics.dr_no_oos++;
            diagnostics.dr_no_total_missing += (dr_no_gap - 1);
            const auto packets_to_fill = dr_no_gap - 1;
            // 900 taken from
            // https://github.com/gmtsar/gmtsar/blob/master/preproc/ERS_preproc/ers_line_fixer/ers_line_fixer.c#L434
            // For ISCE2 this is 30000 which seems to be wayyyy too biig.
            if (packets_to_fill > 900) {
                throw std::runtime_error(
                    fmt::format("There are {} packets missing, which is more than threshold 900 in order to continue. "
                                "It was detected between {}th and {}th packets.",
                                packets_to_fill, i - 1, i));
            }
            for (size_t pf{}; pf < packets_to_fill; pf++) {
                std::memcpy(echoes_raw.get() + echoes.size() * ers::highrate::MEASUREMENT_DATA_SIZE_BYTES,
                            echoes_raw.get() + (echoes.size() - 1) * ers::highrate::MEASUREMENT_DATA_SIZE_BYTES,
                            ers::highrate::MEASUREMENT_DATA_SIZE_BYTES);
                echoes.push_back(echoes.back());
            }
        }
        last_dr_no = echo_meta.data_record_number;
#if HANDLE_DIAGNOSTICS
        const auto packet_counter_gap =
            parseutil::CounterGap<uint8_t, PACKET_COUNTER_MAX>(last_packet_counter, echo_meta.packet_counter);
        // Packet counter increments something like after 1679 packets?? based on IM L0 observation. When this happens
        // subcommutation counter also resets to 1.
        if (packet_counter_gap > 1) {
            diagnostics.packet_counter_oos++;
            diagnostics.packer_counter_total_missing += (packet_counter_gap - 1);
        }
        if (packet_counter_gap == 1) {
            diagnostics.packet_counter_changes++;
        }
        last_packet_counter = echo_meta.packet_counter;

        const auto subcommutation_gap = parseutil::CounterGap<uint8_t, SUBCOMMUTATION_COUNTER_MAX>(
            last_subcommutation_counter, echo_meta.subcommutation_counter);
        // When gap is 0, this is not handled - could be massive rollover or duplicate.
        // Also the counter does not start from 0.
        if (subcommutation_gap > 1 && echo_meta.subcommutation_counter != 1 && packet_counter_gap != 1) {
            diagnostics.subcommutation_counter_oos++;
            diagnostics.subcommutation_counter_total_missing += (subcommutation_gap - 1);
        }
        last_subcommutation_counter = echo_meta.subcommutation_counter;

        const auto image_format_counter_gap = parseutil::CounterGap<uint32_t, IMAGE_FORMAT_COUNTER_MAX>(
            last_image_format_counter, echo_meta.image_format_counter);
        if (image_format_counter_gap > 1) {
            diagnostics.image_format_counter_oos++;
            diagnostics.image_format_counter_total_missing += (image_format_counter_gap - 1);
        }
        last_image_format_counter = echo_meta.image_format_counter;
#endif

        std::memcpy(echoes_raw.get() + echoes.size() * ers::highrate::MEASUREMENT_DATA_SIZE_BYTES, it,
                    ers::highrate::MEASUREMENT_DATA_SIZE_BYTES);
        it += ers::highrate::MEASUREMENT_DATA_SIZE_BYTES;
        echoes.push_back(std::move(echo_meta));
#if DEBUG_PACKETS
        ErsFepAndPacketMetadata meta{};
        meta.echo_meta = echo_meta;
        ers_dbg_meta.push_back(meta);
#endif
    }

    uint16_t min_swst = UINT16_MAX;
    uint16_t max_swst = 0;
    size_t max_samples = ers::highrate::MEASUREMENT_DATA_SAMPLE_COUNT;
    uint16_t swst_changes = 0;
    uint16_t prev_swst = echoes.front().swst_code;
    const int swst_multiplier{4};
    int swath_idx = SwathIdx(asar_meta.swath);
    asar_meta.swath_idx = swath_idx;
    LOGD << "Swath = " << asar_meta.swath << " idx = " << swath_idx;

#if DEBUG_PACKETS
    DebugErsPackets(ers_dbg_meta, ers_dbg_meta.size(), debug_stream);
#endif

    asar_meta.first_swst_code = echoes.front().swst_code;
    asar_meta.last_swst_code = echoes.back().swst_code;
    asar_meta.tx_bw_code = 0;
    asar_meta.up_code = 0;
    asar_meta.down_code = 0;
    asar_meta.pri_code = echoes.front().pri_code;
    asar_meta.tx_pulse_len_code = 0;
    asar_meta.beam_set_num_code = 0;
    asar_meta.beam_adj_code = 0;
    asar_meta.resamp_code = 0;
    asar_meta.first_mjd = echoes.front().isp_sensing_time;
    asar_meta.first_sbt = echoes.front().onboard_time;
    asar_meta.last_mjd = echoes.back().isp_sensing_time;
    asar_meta.last_sbt = echoes.back().onboard_time;

    std::vector<float> v;
    for (auto& e : echoes) {
        if (prev_swst != e.swst_code) {
            if ((e.swst_code < 500) || (e.swst_code > 1500) || ((prev_swst - e.swst_code) % 22 != 0)) {
                e.swst_code = prev_swst;
                continue;
            }
            prev_swst = e.swst_code;
            swst_changes++;
        }
        min_swst = std::min(min_swst, e.swst_code);
        max_swst = std::max(max_swst, e.swst_code);
    }

    asar_meta.swst_changes = swst_changes;

    asar_meta.first_line_time = MjdToPtime(echoes.front().isp_sensing_time);
    asar_meta.sensing_start = asar_meta.first_line_time;
    asar_meta.last_line_time = MjdToPtime(echoes.back().isp_sensing_time);
    asar_meta.sensing_stop = asar_meta.last_line_time;

    double pulse_bw = 0;

    sar_meta.carrier_frequency = ins_file.flp.radar_frequency;
    sar_meta.chirp.range_sampling_rate = ins_file.flp.radar_sampling_rate;

    sar_meta.chirp.pulse_bandwidth = pulse_bw;

    sar_meta.chirp.Kr = ins_file.flp.im_chirp[swath_idx].phase[2] * 2;  // TODO Envisat format multiply by 2?
    sar_meta.chirp.pulse_duration = ins_file.flp.im_chirp[swath_idx].duration;
    sar_meta.chirp.n_samples = std::round(sar_meta.chirp.pulse_duration / (1 / sar_meta.chirp.range_sampling_rate));

    sar_meta.chirp.pulse_bandwidth = sar_meta.chirp.Kr * sar_meta.chirp.pulse_duration;
    LOGD << fmt::format("Chirp Kr = {}, n_samp = {}", sar_meta.chirp.Kr, sar_meta.chirp.n_samples);
    LOGD << fmt::format("tx pulse calc dur = {}",
                        asar_meta.tx_pulse_len_code * (1 / sar_meta.chirp.range_sampling_rate));
    LOGD << fmt::format("Nominal chirp duration = {}", sar_meta.chirp.pulse_duration);
    LOGD << fmt::format("Calculated chirp BW = {} , meta bw = {}", sar_meta.chirp.pulse_bandwidth, pulse_bw);

    const size_t range_samples = max_samples + swst_multiplier * (max_swst - min_swst);
    LOGD << fmt::format("range samples = {}, minimum range padded size = {}", range_samples,
                        range_samples + sar_meta.chirp.n_samples);

    constexpr double c = 299792458;

    constexpr uint32_t im_idx = 0;
    const uint32_t n_pulses_swst = ins_file.fbp.mode_timelines[im_idx].r_values[swath_idx];

    LOGD << "n PRI before SWST = " << n_pulses_swst;

    asar_meta.swst_rank = n_pulses_swst;

    {
        const auto delay_time = ins_file.flp.range_gate_bias;
        // ISCE2 uses 210.943006e-9, but
        // ERS-1-Satellite-to-Ground-Segment-Interface-Specification_01.09.1993_v2D_ER-IS-ESA-GS-0001_EECF05
        // note 5 in 4.4:28 specifies 210.94 ns, which draws more similar results to PF-ERS results.
        const auto pri = (echoes.front().pri_code + 2.0) * 210.94e-9;
        sar_meta.pulse_repetition_frequency = 1 / pri;

        if (sar_meta.pulse_repetition_frequency < 1640 || sar_meta.pulse_repetition_frequency > 1720) {
            LOGW << "PRF value '" << sar_meta.pulse_repetition_frequency << "'"
                 << "is out of the specification (ERS-1-Satellite-to-Ground-Segment-Interface-"
                 << "Specification_01.09.1993_v2D_ER-IS-ESA-GS-0001_EECF05) range [1640, 1720] Hz" << std::endl;
        }
        asar_meta.two_way_slant_range_time =
            n_pulses_swst * pri + min_swst * swst_multiplier / sar_meta.chirp.range_sampling_rate - delay_time;
    }

    LOGD << "PRF " << sar_meta.pulse_repetition_frequency << " PRI " << 1 / sar_meta.pulse_repetition_frequency;
    sar_meta.slant_range_first_sample = asar_meta.two_way_slant_range_time * c / 2;
    sar_meta.wavelength = c / sar_meta.carrier_frequency;
    sar_meta.range_spacing = c / (2 * sar_meta.chirp.range_sampling_rate);

    LOGD << fmt::format("Carrier frequency: {} Range sampling rate = {}", sar_meta.carrier_frequency,
                        sar_meta.chirp.range_sampling_rate);
    LOGD << fmt::format("Slant range to first sample = {} m two way slant time {} ns",
                        sar_meta.slant_range_first_sample, asar_meta.two_way_slant_range_time * 1e9);

    sar_meta.platform_velocity = CalcVelocity(sar_meta.osv[sar_meta.osv.size() / 2]);
    // Ground speed ~12% less than platform velocity. 4.2.1 from "Digital processing of SAR Data"
    // TODO should it be calculated more precisely?
    const double Vg = sar_meta.platform_velocity * 0.88;
    sar_meta.results.Vr_poly = {0, 0, sqrt(Vg * sar_meta.platform_velocity)};
    LOGD << "platform velocity = " << sar_meta.platform_velocity << ", initial Vr = " << CalcVr(sar_meta, 0);
    sar_meta.azimuth_spacing = Vg * (1 / sar_meta.pulse_repetition_frequency);

    sar_meta.img.range_size = range_samples;
    sar_meta.img.azimuth_size = echoes.size();

    const auto range_az_total_items = sar_meta.img.range_size * sar_meta.img.azimuth_size;
    const auto range_az_total_bytes = range_az_total_items * sizeof(cufftComplex);
    CHECK_CUDA_ERR(cudaMalloc(d_parsed_packets, range_az_total_bytes));
    constexpr auto CLEAR_VALUE = NAN;
    constexpr cufftComplex CLEAR_VALUE_COMPLEX{CLEAR_VALUE, CLEAR_VALUE};
    static_assert(sizeof(CLEAR_VALUE_COMPLEX) == sizeof(CLEAR_VALUE) * 2);
    cuda::algorithm::Fill(*d_parsed_packets, range_az_total_items, CLEAR_VALUE_COMPLEX);

    std::vector<uint16_t> swst_codes;
    swst_codes.reserve(echoes.size());
    for (size_t y = 0; y < echoes.size(); y++) {
        swst_codes.push_back(echoes.at(y).swst_code);
        sar_meta.total_raw_samples += ers::highrate::MEASUREMENT_DATA_SAMPLE_COUNT;
    }
    envformat::ConvertErsImSamplesToComplex(echoes_raw.get(), ers::highrate::MEASUREMENT_DATA_SAMPLE_COUNT,
                                            echoes.size(), swst_codes.data(), min_swst, *d_parsed_packets,
                                            sar_meta.img.range_size);
    // TODO init guess handling? At the moment just a naive guess from nadir point
    double init_guess_lat = (asar_meta.start_nadir_lat + asar_meta.stop_nadir_lat) / 2;
    double init_guess_lon = asar_meta.start_nadir_lon;
    if (asar_meta.ascending) {
        init_guess_lon += 1.0 + 2.0 * swath_idx;
        if (init_guess_lon >= 180.0) {
            init_guess_lon -= 360;
        }
    } else {
        init_guess_lon -= 1.0 + 2.0 * swath_idx;
        if (init_guess_lon <= 180.0) {
            init_guess_lon += 360.0;
        }
    }

    LOGD << fmt::format("nadirs start {} {} - stop  {} {}", asar_meta.start_nadir_lat, asar_meta.start_nadir_lon,
                        asar_meta.stop_nadir_lat, asar_meta.stop_nadir_lon);

    LOGD << fmt::format("guess = {} {}", init_guess_lat, init_guess_lon);

    auto init_xyz = Geo2xyzWgs84(init_guess_lat, init_guess_lon, 0);

    double center_s = (1 / sar_meta.pulse_repetition_frequency) * sar_meta.img.azimuth_size / 2;
    auto center_time = asar_meta.sensing_start + boost::posix_time::microseconds(static_cast<uint32_t>(center_s * 1e6));

    // TODO investigate fast time effect on geolocation
    double slant_range_center = sar_meta.slant_range_first_sample +
                                ((sar_meta.img.range_size - sar_meta.chirp.n_samples) / 2) * sar_meta.range_spacing;

    auto osv = InterpolateOrbit(sar_meta.osv, center_time);

    sar_meta.center_point = RangeDopplerGeoLocate({osv.x_vel, osv.y_vel, osv.z_vel}, {osv.x_pos, osv.y_pos, osv.z_pos},
                                                  init_xyz, slant_range_center);
    sar_meta.center_time = center_time;
    sar_meta.first_line_time = asar_meta.sensing_start;
    sar_meta.azimuth_bandwidth_fraction = 0.8f;
    auto llh = xyz2geoWGS84(sar_meta.center_point);
    LOGD << "center point = " << llh.latitude << " " << llh.longitude;

    LOGD << "Data record number discontinuities/total missing - " << diagnostics.dr_no_oos << "/"
         << diagnostics.dr_no_total_missing;

    if (diagnostics.dr_no_oos > 0) {
        asar_meta.product_err = true;
    }

#if HANDLE_DIAGNOSTICS
    LOGD << "Packet counter field diagnostics - " << diagnostics.packet_counter_oos << " "
         << diagnostics.packer_counter_total_missing << " changes " << diagnostics.packet_counter_changes;
    LOGD << "Subcommutation counter field diagnostics - " << diagnostics.subcommutation_counter_oos << " "
         << diagnostics.subcommutation_counter_total_missing;
    LOGD << "Image format counter diagnostics - " << diagnostics.image_format_counter_oos << " "
         << diagnostics.image_format_counter_total_missing;
#endif
}

}  // namespace alus::asar::envformat
