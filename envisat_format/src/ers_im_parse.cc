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
#include "envisat_utils.h"
#include "ers_env_format.h"
#include "parse_util.h"

#define DEBUG_PACKETS 0
#define HANDLE_DIAGNOSTICS 1

namespace {

void FillFbaqMeta(ASARMetadata& asar_meta) {
    // Nothing for ERS
    asar_meta.compression_metadata.echo_method = "NONE";
    asar_meta.compression_metadata.echo_ratio = "";
    asar_meta.compression_metadata.init_cal_method = "NONE";
    asar_meta.compression_metadata.init_cal_ratio = "";
    asar_meta.compression_metadata.noise_method = "NONE";
    asar_meta.compression_metadata.noise_ratio = "";
    asar_meta.compression_metadata.per_cal_method = "NONE";
    asar_meta.compression_metadata.per_cal_ratio = "";
}

template <class T>
[[nodiscard]] const uint8_t* CopyBSwapPOD(T& dest, const uint8_t* src) {
    memcpy(&dest, src, sizeof(T));
    dest = bswap(dest);
    return src + sizeof(T);
}

int SwathIdx(const std::string& swath) {
    if (swath.size() == 3 && swath[0] == 'I' && swath[1] == 'S') {
        char idx = swath[2];
        if (idx >= '1' && idx <= '7') {
            return idx - '1';
        }
    }
    throw std::runtime_error("MDSR contains swath '" + swath + "' which is not supported or invalid.");
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
    uint16_t activity_task;
    uint32_t image_format_counter;
    uint16_t pri_code;
    uint16_t swst_code;

    std::vector<std::complex<float>> raw_data;
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
constexpr auto IMAGE_FORMAT_COUNTER_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint32_t, 32>()};

void InitializeCounters(const uint8_t* packets_start, uint32_t& dr_no, uint8_t& packet_counter,
                        uint8_t& subcommutation_counter, uint32_t& image_format_counter) {
    FetchUint32(packets_start + alus::asar::envformat::ers::highrate::PROCESSOR_ANNOTATION_ISP_SIZE_BYTES +
                    alus::asar::envformat::ers::highrate::FEP_ANNOTATION_SIZE_BYTES,
                dr_no);
    if (dr_no != 0) {
        dr_no--;
    } else {
        dr_no = DR_NO_MAX;
    }
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

}  // namespace

namespace alus::asar::envformat {

void ParseErsLevel0ImPackets(const std::vector<char>& file_data, const DSD_lvl0& mdsr, SARMetadata& sar_meta,
                             ASARMetadata& asar_meta, std::vector<std::complex<float>>& img_data,
                             InstrumentFile& ins_file, boost::posix_time::ptime packets_start_filter,
                             boost::posix_time::ptime packets_stop_filter) {
    FillFbaqMeta(asar_meta);
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
    std::vector<EchoMeta> echoes;
    echoes.reserve(mdsr.num_dsr);
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
    // https://asf.alaska.edu/wp-content/uploads/2019/03/ers_ceos.pdf mixed between two.
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

        echo_meta.packet_counter = it[0];
        echo_meta.subcommutation_counter = it[1];
        it += 2;

        it += 8;  // IDHT General header source packet
        if (it[0] != 0xAA) {
            throw std::runtime_error(fmt::format(
                "ERS data packet's auxiliary section shall start with 0xAA, instead {:#x} was found for packet no {}",
                it[0], i + 1));
        }
        it += 1;
        it += 1;  // OGRC/OBRC flag and Orbit ID code
        echo_meta.onboard_time = 0;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[0]) << 24;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[1]) << 16;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[2]) << 8;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[3]) << 0;
        it += 4;
        it = CopyBSwapPOD(echo_meta.activity_task, it);
        it = CopyBSwapPOD(echo_meta.image_format_counter, it);
        it = CopyBSwapPOD(echo_meta.swst_code, it);
        it = CopyBSwapPOD(echo_meta.pri_code, it);
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

        echo_meta.raw_data.reserve(ers::highrate::MEASUREMENT_DATA_SIZE_BYTES);
        // uint64_t i_avg_cumulative{0};
        // uint64_t q_avg_cumulative{0};
        for (size_t r_i{0}; r_i < 5616; r_i++) {
            uint8_t i_sample = it[r_i * 2 + 0];
            // i_avg_cumulative += i_sample;
            uint8_t q_sample = it[r_i * 2 + 1];
            // q_avg_cumulative += q_sample;
            echo_meta.raw_data.emplace_back(static_cast<float>(i_sample), static_cast<float>(q_sample));
        }
        //            double i_avg = i_avg_cumulative / 5616.0;
        //            double q_avg = q_avg_cumulative / 5616.0;
        //            if (i_avg > 16.0 || i_avg < 15.0) {
        //                LOGD << "average for i at MSDR no. " << i << " is OOL " << i_avg;
        //            }
        //            if (q_avg > 16.0 || q_avg < 15.0) {
        //                LOGD << "average for q at MSDR no. " << i << " is OOL " << q_avg;
        //            }
        it += 11232;
        echoes.push_back(std::move(echo_meta));
#if DEBUG_PACKETS
        ErsFepAndPacketMetadata meta{};
        meta.echo_meta = echo_meta;
        ers_dbg_meta.push_back(meta);
#endif
    }

    uint16_t min_swst = UINT16_MAX;
    uint16_t max_swst = 0;
    size_t max_samples = 0;
    uint16_t swst_changes = 0;
    uint16_t prev_swst = echoes.front().swst_code;
    const int swst_multiplier{4};
    int swath_idx = SwathIdx(asar_meta.swath);
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
        max_samples = std::max(max_samples, e.raw_data.size());
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

    img_data.clear();
    img_data.resize(sar_meta.img.range_size * sar_meta.img.azimuth_size, {NAN, NAN});

    sar_meta.total_raw_samples = 0;

    for (size_t y = 0; y < echoes.size(); y++) {
        const auto& e = echoes[y];
        size_t idx = y * range_samples;
        idx += swst_multiplier * (e.swst_code - min_swst);
        const size_t n_samples = e.raw_data.size();
        memcpy(&img_data[idx], e.raw_data.data(), n_samples * 8);
        sar_meta.total_raw_samples += n_samples;
    }

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