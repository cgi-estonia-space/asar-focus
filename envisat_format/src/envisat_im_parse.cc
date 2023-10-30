/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "envisat_im_parse.h"

#include <stdexcept>
#include <string>

#include "fmt/format.h"

#include "alus_log.h"
#include "envisat_utils.h"
#include "parse_util.h"

#define DEBUG_PACKETS 0
#define HANDLE_DIAGNOSTICS 1

namespace {
void FillFbaqMeta(ASARMetadata& asar_meta) {
    asar_meta.compression_metadata.echo_method = "FBAQ";
    asar_meta.compression_metadata.echo_ratio = "4/8";
    asar_meta.compression_metadata.init_cal_method = "NONE";  // TBD
    asar_meta.compression_metadata.init_cal_ratio = "";       // TBD
    asar_meta.compression_metadata.noise_method = "NONE";     // TBD
    asar_meta.compression_metadata.noise_ratio = "";          // TBD
    asar_meta.compression_metadata.per_cal_method = "NONE";   // TBD
    asar_meta.compression_metadata.per_cal_ratio = "";        // TBD
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
    uint8_t mode_id;
    uint64_t onboard_time;
    uint32_t mode_count;
    uint8_t antenna_beam_set_no;
    uint8_t comp_ratio;
    bool echo_flag;
    bool noise_flag;
    bool cal_flag;
    bool cal_type;
    uint16_t cycle_count;

    uint16_t pri_code;
    uint16_t swst_code;
    uint16_t echo_window_code;
    uint8_t upconverter_raw;
    uint8_t downconverter_raw;
    bool tx_pol;
    bool rx_pol;
    uint8_t cal_row_number;
    uint16_t tx_pulse_code;
    uint8_t beam_adj_delta;
    uint8_t chirp_pulse_bw_code;
    uint8_t aux_tx_monitor_level;
    uint16_t resampling_factor;

    std::vector<std::complex<float>> raw_data;
};

int FBAQ4Idx(int block, int idx) {
    switch (idx) {
        case 0b1111:
            idx = 0;
            break;
        case 0b1110:
            idx = 1;
            break;
        case 0b1101:
            idx = 2;
            break;
        case 0b1100:
            idx = 3;
            break;
        case 0b1011:
            idx = 4;
            break;
        case 0b1010:
            idx = 5;
            break;
        case 0b1001:
            idx = 6;
            break;
        case 0b1000:
            idx = 7;
            break;
        case 0b0000:
            idx = 8;
            break;
        case 0b0001:
            idx = 9;
            break;
        case 0b0010:
            idx = 10;
            break;
        case 0b0011:
            idx = 11;
            break;
        case 0b0100:
            idx = 12;
            break;
        case 0b0101:
            idx = 13;
            break;
        case 0b0110:
            idx = 14;
            break;
        case 0b0111:
            idx = 15;
            break;
    }

    return 256 * idx + block;
}

#if DEBUG_PACKETS

constexpr auto DEBUG_ECHOS_ONLY{false};

struct EnvisatFepAndPacketHeader {
    FEPAnnotations fep_annotations;
    uint16_t packet_id;
    uint16_t sequence_control;
    uint16_t packet_length;
    uint16_t datafield_length;
    EchoMeta echo_meta;
};

[[maybe_unused]] void DebugEnvisatPackets(const std::vector<EnvisatFepAndPacketHeader> dm, size_t limit,
                                          std::ostream& out_stream) {
    std::stringstream stream;
    stream << "mjd_fep,mjd_fep_micros,isp_length,crc_error_count,correction_count,spare,"
           << "packet_id,sequence_control,packet_length,datafield_length,"
           << "isp_sense,isp_sense_micros,mode_id,onboard_time,mode_count,antenna_beam_set_no,comp_ratio,echo_flag,"
           << "noise_flag,cal_flag,"
           << "cal_type,cycle_count,pri_code,swst_code,echo_window_code,upconverter_raw,downconverter_raw,tx_pol,"
           << "rx_pol,cal_row_number,tx_pulse_code,beam_adj_delta,chirp_pulse_bw_code,aux_tx_monitor_level,"
           << "resampling_factor" << std::endl;
    out_stream << stream.str();
    for (size_t i{}; i < limit; i++) {
        stream.str("");
        const auto& emi = dm.at(i).echo_meta;
        const auto& dmi = dm.at(i);
        std::string mjd_fep_str{"INVALID"};
        long mjd_fep_micros{};
        mjd mjd_fep = dmi.fep_annotations.mjd_time;
        try {
            const auto mjd_fep_ptime = MjdToPtime(mjd_fep);
            mjd_fep_str = boost::posix_time::to_simple_string(mjd_fep_ptime);
            mjd_fep_micros = mjd_fep_ptime.time_of_day().total_microseconds();
        } catch (const std::out_of_range& e) {
            mjd_fep_str = fmt::format("{}-{}-{}", mjd_fep.days, mjd_fep.seconds, mjd_fep.micros);
        }

        const auto isp_sensing_time_ptime = MjdToPtime(emi.isp_sensing_time);
        // clang-format off
        stream << mjd_fep_str << "," << mjd_fep_micros << "," << dmi.fep_annotations.isp_length << ","
               << dmi.fep_annotations.crcErrorCnt << "," << dmi.fep_annotations.correctionCnt << ","
               << dmi.fep_annotations.spare << ","
               << dmi.packet_id << "," << dmi.sequence_control << "," << dmi.packet_length << ","
               << dmi.datafield_length << ","
               << boost::posix_time::to_simple_string(isp_sensing_time_ptime) << ","
               << isp_sensing_time_ptime.time_of_day().total_microseconds() << ","
               << fmt::format("{:#x},", emi.mode_id) << emi.onboard_time << "," << emi.mode_count << ","
               << fmt::format("{:#x},{:#x},{},{},{},{}", emi.antenna_beam_set_no, emi.comp_ratio,
                              emi.echo_flag, emi.noise_flag, emi.cal_flag, emi.cal_type) << ","
               << emi.cycle_count << "," << emi.pri_code << "," << emi.swst_code << "," << emi.echo_window_code << ","
               << fmt::format("{:#x},{:#x},{},{}", emi.upconverter_raw, emi.downconverter_raw, emi.tx_pol, emi.rx_pol)
               << "," << (uint32_t)emi.cal_row_number << "," << emi.tx_pulse_code << ","
               << fmt::format("{:#x},{:#x},{:#x}",
                              emi.beam_adj_delta, emi.chirp_pulse_bw_code, emi.aux_tx_monitor_level) << ","
               << emi.resampling_factor << std::endl;
        // clang-format on
        out_stream << stream.str();
    }
}

#endif

inline void FetchModePacketCount(const uint8_t* start_array, uint32_t& var) {
    var = 0;
    var |= static_cast<uint32_t>(start_array[0]) << 16;
    var |= static_cast<uint32_t>(start_array[1]) << 8;
    var |= static_cast<uint32_t>(start_array[2]) << 0;
}

inline void FetchCyclePacketCount(const uint8_t* start_array, uint16_t& var) {
    var = 0;
    var |= (start_array[0] & 0x0F) << 8;
    var |= start_array[1] << 0;
}

#if HANDLE_DIAGNOSTICS

constexpr auto SEQUENCE_CONTROL_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint16_t, 14>()};
constexpr auto MODE_PACKET_COUNT_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint32_t, 24>()};
constexpr auto CYCLE_PACKET_COUNT_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint16_t, 12>()};

void InitializeCounters(const uint8_t* packets_start, uint16_t& sequence_control, uint32_t& mode_packet_count,
                        uint16_t& cycle_packet_count) {
    [[maybe_unused]] const auto discarding =
        CopyBSwapPOD(sequence_control,
                     packets_start + sizeof(EchoMeta::isp_sensing_time) + sizeof(FEPAnnotations) + sizeof(uint16_t));
    sequence_control = sequence_control & 0x3FFF;
    if (sequence_control != 0) {
        sequence_control = SEQUENCE_CONTROL_MAX;
    } else {
        sequence_control--;
    }
    // 48 mode start
    FetchModePacketCount(packets_start + 48, mode_packet_count);
    if (mode_packet_count == 0) {
        mode_packet_count = MODE_PACKET_COUNT_MAX;
    } else {
        mode_packet_count--;
    }
    // 52 cycle-start
    FetchCyclePacketCount(packets_start + 52, cycle_packet_count);
    // For cycle_packet_count value 0 is special (i.e. marking reset when packet header changes)
    // and gap 0 is not implemented.
    if (cycle_packet_count != 0) {
        cycle_packet_count--;
    }
}

// oos - out of sequence
struct PacketParseDiagnostics {
    size_t sequence_control_oos;            // No. of times gap happened.
    size_t sequence_control_total_missing;  // Total sum of all gaps' value.
    size_t mode_packet_count_oos;
    size_t mode_packet_total_missing;
    size_t cycle_packet_count_oos;
    size_t cycle_packet_total_missing;
    size_t calibration_packet_count;
};
#endif
}  // namespace

namespace alus::asar::envformat {

void ParseEnvisatLevel0ImPackets(const std::vector<char>& file_data, const DSD_lvl0& mdsr, SARMetadata& sar_meta,
                                 ASARMetadata& asar_meta, std::vector<std::complex<float>>& img_data,
                                 InstrumentFile& ins_file, boost::posix_time::ptime packets_start_filter,
                                 boost::posix_time::ptime packets_stop_filter) {
    FillFbaqMeta(asar_meta);
    sar_meta.carrier_frequency = ins_file.flp.radar_frequency;
    sar_meta.chirp.range_sampling_rate = ins_file.flp.radar_sampling_rate;
    std::vector<EchoMeta> echoes;  // Via MDSR - the count of data records -> reserve()?
    const uint8_t* it = reinterpret_cast<const uint8_t*>(file_data.data()) + mdsr.ds_offset;
#if DEBUG_PACKETS
    std::vector<EnvisatFepAndPacketHeader> envisat_dbg_meta;
    std::ofstream debug_stream("/tmp/" + asar_meta.product_name + ".debug.csv");
#endif
#if HANDLE_DIAGNOSTICS
    PacketParseDiagnostics diagnostics{};
    uint16_t last_sequence_control;
    uint32_t last_mode_packet_count;
    uint16_t last_cycle_packet_count;
    InitializeCounters(it, last_sequence_control, last_mode_packet_count, last_cycle_packet_count);
#endif
    const auto start_filter_mjd = PtimeToMjd(packets_start_filter);
    const auto end_filter_mjd = PtimeToMjd(packets_stop_filter);
    for (size_t i = 0; i < mdsr.num_dsr; i++) {
        EchoMeta echo_meta = {};
#if HANDLE_DIAGNOSTICS
        const auto* it_start = it;
#endif
        // Source: ENVISAT-1 ASAR INTERPRETATION OF SOURCE PACKET DATA
        // PO-TN-MMS-SR-0248
        it = CopyBSwapPOD(echo_meta.isp_sensing_time, it);
        FEPAnnotations fep;
        it = CopyBSwapPOD(fep, it);

        // This is actually application process ID plus some bits, needs to be refactored, more info - PO-ID-DOR-SY-0032
        uint16_t packet_id;
        it = CopyBSwapPOD(packet_id, it);

        uint16_t sequence_control;
        it = CopyBSwapPOD(sequence_control, it);

        [[maybe_unused]] uint8_t segmentation_flags = sequence_control >> 14;
        // Wrap around sequence counter for the number of packets minus one transmitted for the Application Process ID
        // to which the source data applies. Thus the first packet in the data stream has a Source Sequence Count of 0.
        sequence_control = sequence_control & 0x3FFF;  // It is 14 bit rolling value

        uint16_t packet_length;
        it = CopyBSwapPOD(packet_length, it);

        uint16_t datafield_length;
        it = CopyBSwapPOD(datafield_length, it);

        // 28 bytes from mode_id until block loop. 29 is datafield_length.
        // Exclude also 32 bytes above (FEP, ISP sensing time).
        if (echo_meta.isp_sensing_time < start_filter_mjd) {
#if HANDLE_DIAGNOSTICS
            last_sequence_control = sequence_control;
            FetchModePacketCount(it_start + 48, last_mode_packet_count);
            FetchCyclePacketCount(it_start + 52, last_cycle_packet_count);
#endif
            it += (fep.isp_length - datafield_length + 28);
            continue;
        }
        if (echo_meta.isp_sensing_time > end_filter_mjd) {
            break;
        }

        if (datafield_length != 29) {
            throw std::runtime_error(
                fmt::format("DSR nr = {} ({}) parsing error. Date Field Header Length should be 29 - is {}", i,
                            mdsr.num_dsr, datafield_length));
        }

        it++;
        echo_meta.mode_id = *it;
        it++;

        if (echo_meta.mode_id != 0x54) {
            throw std::runtime_error(
                fmt::format("Expected only IM mode packets, but encountered mode ID {:#x}", echo_meta.mode_id));
        }

        // Wrap around counter indicating the time at which the first sampled window in
        // the source data was acquired. The lsb represents an interval of 15.26 µs nominal.
        echo_meta.onboard_time = 0;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[0]) << 32;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[1]) << 24;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[2]) << 16;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[3]) << 8;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[4]) << 0;

        it += 5;
        it++;  // spare byte after 40 bit onboard time

        // Wrap around sequence counter for the number of packets minus one transmitted for the Mode ID to which
        // the source data applies. Thus the first packet in the data stream has a Mode Packet Count of 0.
        FetchModePacketCount(it, echo_meta.mode_count);
        it += 3;

        echo_meta.antenna_beam_set_no = *it >> 2;
        echo_meta.comp_ratio = *it & 0x3;
        it++;

        echo_meta.echo_flag = *it & 0x80;
        echo_meta.noise_flag = *it & 0x40;
        echo_meta.cal_flag = *it & 0x20;
        echo_meta.cal_type = *it & 0x10;
        // Sequence counter for the number of packets minus one transmitted since the last commanded data field change.
        // Thus the first packet in the data stream following a commanded change in data field header has a
        // Mode Packet Count of 0.
        FetchCyclePacketCount(it, echo_meta.cycle_count);
        it += 2;

        it = CopyBSwapPOD(echo_meta.pri_code, it);
        it = CopyBSwapPOD(echo_meta.swst_code, it);
        it = CopyBSwapPOD(echo_meta.echo_window_code, it);

        {
            uint16_t data;
            it = CopyBSwapPOD(data, it);
            echo_meta.upconverter_raw = data >> 12;
            echo_meta.downconverter_raw = (data >> 7) & 0x1F;
            echo_meta.tx_pol = data & 0x40;
            echo_meta.rx_pol = data & 0x20;
            echo_meta.cal_row_number = data & 0x1F;
        }
        {
            uint16_t data = -1;
            it = CopyBSwapPOD(data, it);
            echo_meta.tx_pulse_code = data >> 6;
            echo_meta.beam_adj_delta = data & 0x3F;
        }

        echo_meta.chirp_pulse_bw_code = it[0];
        echo_meta.aux_tx_monitor_level = it[1];
        it += 2;

        it = CopyBSwapPOD(echo_meta.resampling_factor, it);

#if HANDLE_DIAGNOSTICS
        const auto seq_ctrl_gap =
            parseutil::CounterGap<uint16_t, SEQUENCE_CONTROL_MAX>(last_sequence_control, sequence_control);
        // Value 0 denotes reset.
        // When seq_ctrl_gap is 0, this is not handled - could be massive rollover or duplicate.
        if (seq_ctrl_gap > 1) {
            diagnostics.sequence_control_oos++;
            diagnostics.sequence_control_total_missing += (seq_ctrl_gap - 1);
        }
        last_sequence_control = sequence_control;

        const auto mode_packet_gap =
            parseutil::CounterGap<uint32_t, MODE_PACKET_COUNT_MAX>(last_mode_packet_count, echo_meta.mode_count);
        // When gap is 0, this is not handled - could be massive rollover or duplicate.
        if (mode_packet_gap > 1) {
            diagnostics.mode_packet_count_oos++;
            diagnostics.mode_packet_total_missing += (mode_packet_gap - 1);
        }
        last_mode_packet_count = echo_meta.mode_count;

        const auto cycle_packet_gap =
            parseutil::CounterGap<uint16_t, CYCLE_PACKET_COUNT_MAX>(last_cycle_packet_count, echo_meta.cycle_count);
        // When gap is 0, this is not handled - could be massive rollover or duplicate.
        if (cycle_packet_gap > 1) {
            diagnostics.cycle_packet_count_oos++;
            diagnostics.cycle_packet_total_missing += cycle_packet_gap;
        }
        last_cycle_packet_count = echo_meta.cycle_count;
#endif

        size_t data_len = packet_length - 29;

        if (echo_meta.echo_flag) {
            echo_meta.raw_data.reserve(echo_meta.echo_window_code);
            size_t n_blocks = data_len / 64;
            for (size_t bi = 0; bi < n_blocks; bi++) {
                const uint8_t* block_data = it + bi * 64;
                uint8_t block_id = block_data[0];
                for (size_t j = 0; j < 63; j++) {
                    uint8_t i_codeword = block_data[1 + j] >> 4;
                    uint8_t q_codeword = block_data[1 + j] & 0xF;

                    float i_samp = ins_file.fbp.i_LUT_fbaq4[FBAQ4Idx(block_id, i_codeword)];
                    float q_samp = ins_file.fbp.q_LUT_fbaq4[FBAQ4Idx(block_id, q_codeword)];
                    echo_meta.raw_data.emplace_back(i_samp, q_samp);
                }
            }
            // Deal with the remainder
            size_t remainder = data_len % 64;
            const uint8_t* block_data = it + n_blocks * 64;
            uint8_t block_id = block_data[0];
            for (size_t j = 0; j < remainder; j++) {
                uint8_t i_codeword = block_data[1 + j] >> 4;
                uint8_t q_codeword = block_data[1 + j] & 0xF;

                float i_samp = ins_file.fbp.i_LUT_fbaq4[FBAQ4Idx(block_id, i_codeword)];
                float q_samp = ins_file.fbp.q_LUT_fbaq4[FBAQ4Idx(block_id, q_codeword)];
                echo_meta.raw_data.emplace_back(i_samp, q_samp);
            }

            echoes.push_back(std::move(echo_meta));
        } else if (echo_meta.cal_flag) {
            if (echoes.size() > 0) {
                auto copy_echo = echoes.back();
                const auto prf = (sar_meta.chirp.range_sampling_rate / copy_echo.pri_code);
                const auto micros_to_add = static_cast<ul>((1 / prf) * 10e5);
                copy_echo.isp_sensing_time = MjdAddMicros(copy_echo.isp_sensing_time, micros_to_add);
                echoes.push_back(copy_echo);
            }
#if HANDLE_DIAGNOSTICS
            last_cycle_packet_count = 0;  // Special value to mark reset;
            diagnostics.calibration_packet_count++;
#endif
        }

#if DEBUG_PACKETS
        if ((!DEBUG_ECHOS_ONLY && echo_meta.cal_flag) || echo_meta.echo_flag) {
            EnvisatFepAndPacketHeader d;
            d.fep_annotations = fep;
            d.datafield_length = datafield_length;
            d.packet_id = packet_id;
            d.packet_length = packet_length;
            d.sequence_control = sequence_control;
            d.echo_meta = echo_meta;
            if (echo_meta.cal_flag) {
                d.echo_meta.isp_sensing_time = echoes.back().isp_sensing_time;
            }
            envisat_dbg_meta.push_back(d);
        }
#endif

        it += (fep.isp_length - 29);
    }

    uint16_t min_swst = UINT16_MAX;
    uint16_t max_swst = 0;
    size_t max_samples = 0;
    uint16_t swst_changes = 0;
    uint16_t prev_swst = echoes.front().swst_code;
    const int swst_multiplier{1};

#if DEBUG_PACKETS
    DebugEnvisatPackets(envisat_dbg_meta, envisat_dbg_meta.size(), debug_stream);
#endif

    asar_meta.first_swst_code = echoes.front().swst_code;
    asar_meta.last_swst_code = echoes.back().swst_code;
    asar_meta.tx_bw_code = echoes.front().chirp_pulse_bw_code;
    asar_meta.up_code = echoes.front().upconverter_raw;
    asar_meta.down_code = echoes.front().downconverter_raw;
    asar_meta.pri_code = echoes.front().pri_code;
    asar_meta.tx_pulse_len_code = echoes.front().tx_pulse_code;
    asar_meta.beam_set_num_code = echoes.front().antenna_beam_set_no;
    asar_meta.beam_adj_code = echoes.front().beam_adj_delta;
    asar_meta.resamp_code = echoes.front().resampling_factor;
    asar_meta.first_mjd = echoes.front().isp_sensing_time;
    asar_meta.first_sbt = echoes.front().onboard_time;
    asar_meta.last_mjd = echoes.back().isp_sensing_time;
    asar_meta.last_sbt = echoes.back().onboard_time;

    std::vector<float> v;
    for (auto& e : echoes) {
        if (prev_swst != e.swst_code) {
            prev_swst = e.swst_code;
            swst_changes++;
        }
        min_swst = std::min(min_swst, e.swst_code);
        max_swst = std::max(max_swst, e.swst_code);
        max_samples = std::max(max_samples, e.raw_data.size());
    }

    int swath_idx = SwathIdx(asar_meta.swath);
    LOGD << "Swath = " << asar_meta.swath << " idx = " << swath_idx;

    asar_meta.swst_changes = swst_changes;

    asar_meta.first_line_time = MjdToPtime(echoes.front().isp_sensing_time);
    asar_meta.sensing_start = asar_meta.first_line_time;
    asar_meta.last_line_time = MjdToPtime(echoes.back().isp_sensing_time);
    asar_meta.sensing_stop = asar_meta.last_line_time;

    double pulse_bw = 16e6 / 255 * echoes.front().chirp_pulse_bw_code;

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

    // TODO Range gate bias must be included in this calculation
    sar_meta.pulse_repetition_frequency = sar_meta.chirp.range_sampling_rate / echoes.front().pri_code;
    asar_meta.two_way_slant_range_time =
        (min_swst + n_pulses_swst * echoes.front().pri_code) * (1 / sar_meta.chirp.range_sampling_rate);

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
#if HANDLE_DIAGNOSTICS
    LOGD << "Diagnostic summary of the packets";
    LOGD << "Calibration packets - " << diagnostics.calibration_packet_count;
    LOGD << "Seq ctrl [oos total] " << diagnostics.sequence_control_oos << " "
         << diagnostics.sequence_control_total_missing;
    LOGD << "Mode packet [oos total] " << diagnostics.mode_packet_count_oos << " "
         << diagnostics.mode_packet_total_missing;
    LOGD << "Cycle packet [oos total] " << diagnostics.cycle_packet_count_oos << " "
         << diagnostics.cycle_packet_total_missing;
#endif
}
}  // namespace alus::asar::envformat