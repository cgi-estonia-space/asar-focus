/*
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c)
 * by CGI Estonia AS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "envisat_im_parse.h"

#include <stdexcept>
#include <string>

#include "fmt/format.h"

#include "alus_log.h"
#include "cuda_algorithm.h"
#include "cuda_util.h"
#include "envisat_format_kernels.h"
#include "envisat_utils.h"
#include "parse_util.h"

#define DEBUG_PACKETS 0
#define HANDLE_DIAGNOSTICS 1

namespace {
void FillFbaqMetaAsar(ASARMetadata& asar_meta) {
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
    size_t raw_samples;
};
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

inline size_t ForecastMaxBlockDataLength(const uint8_t* mdsr_start, size_t count) {
    const uint8_t* rolling_start = mdsr_start;
    size_t parsed_records{};
    size_t echo_packets{};
    uint16_t max_block_data_length{0};
    while (parsed_records < count) {
        rolling_start += 12;  // MJD
        FEPAnnotations fep;
        rolling_start = CopyBSwapPOD(fep, rolling_start);
        rolling_start += 4;  // packet ID, sequence control.

        uint16_t packet_length;
        rolling_start = CopyBSwapPOD(packet_length, rolling_start);
        uint16_t datafield_length;
        rolling_start = CopyBSwapPOD(datafield_length, rolling_start);
        if (datafield_length != 29) {
            throw std::runtime_error(
                fmt::format("DSR sequence = {} parsing error. Date Field Header Length should be 29 - is {}",
                            parsed_records + 1, datafield_length));
        }

        // ??, MODE ID, Onboard time, spare, mode count, comp/beam
        bool is_echo = rolling_start[12] & 0x80;
        if (is_echo) {
            const uint16_t block_data_len = packet_length - 29;
            max_block_data_length = std::max(max_block_data_length, block_data_len);
            echo_packets++;
        }
        rolling_start += fep.isp_length - datafield_length + 28;
        parsed_records++;
    }

    if (echo_packets < 10) {
        throw std::runtime_error(
            "Could not gather atleast 10 echo packets in order to forecast data block lengths - invalid dataset?");
    }

    return max_block_data_length;
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
    if (sequence_control == 0) {
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
    size_t packet_data_blocks_min;
    size_t packet_data_blocks_max;
};
#endif

inline const uint8_t* FetchPacketFrom(const uint8_t* start, alus::asar::envformat::ForecastMeta& meta, bool& is_echo,
                                      bool& is_cal) {
    // Source: ENVISAT-1 ASAR INTERPRETATION OF SOURCE PACKET DATA
    // PO-TN-MMS-SR-0248
    auto it = CopyBSwapPOD(meta.isp_sensing_time, start);
    FEPAnnotations fep;
    it = CopyBSwapPOD(fep, it);
    it += sizeof(uint16_t);  // Packet ID
    it += sizeof(uint16_t);  // Sequence control
    uint16_t packet_length;
    it = CopyBSwapPOD(packet_length, it);
    uint16_t datafield_length;
    it = CopyBSwapPOD(datafield_length, it);

    if (datafield_length != 29) {
        throw std::runtime_error(
            fmt::format("Fetching metadat for ENVISAT IM parse error. Date Field Header Length should be 29 - is {}",
                        datafield_length));
    }

    const auto next_one = it + (fep.isp_length - datafield_length + 28);

    uint8_t echo_cal_flags = it[12];
    is_echo = echo_cal_flags & 0x80;
    is_cal = echo_cal_flags & 0x20;

    return next_one;
}

inline void TransferMetadata(const EchoMeta& echo_meta, uint16_t measurement_bytes_length,
                             alus::asar::envformat::CommonPacketMetadata& common_meta) {
    common_meta.sensing_time = echo_meta.isp_sensing_time;
    common_meta.onboard_time = echo_meta.onboard_time;
    common_meta.pri_code = echo_meta.pri_code;
    common_meta.swst_code = echo_meta.swst_code;
    common_meta.sample_count = echo_meta.raw_samples;
    common_meta.measurement_array_length_bytes = measurement_bytes_length;

    common_meta.asar.chirp_pulse_bw_code = echo_meta.chirp_pulse_bw_code;
    common_meta.asar.upconverter_raw = echo_meta.upconverter_raw;
    common_meta.asar.downconverter_raw = echo_meta.downconverter_raw;
    common_meta.asar.tx_pulse_code = echo_meta.tx_pulse_code;
    common_meta.asar.antenna_beam_set_no = echo_meta.antenna_beam_set_no;
    common_meta.asar.beam_adj_delta = echo_meta.beam_adj_delta;
    common_meta.asar.resampling_factor = echo_meta.resampling_factor;
}

}  // namespace

namespace alus::asar::envformat {

std::vector<ForecastMeta> FetchEnvisatL0ImForecastMeta(const std::vector<char>& file_data, const DSD_lvl0& mdsr,
                                                       boost::posix_time::ptime packets_start_filter,
                                                       boost::posix_time::ptime packets_stop_filter,
                                                       size_t& packets_before_start, size_t& packets_after_stop) {
    std::vector<ForecastMeta> forecast_meta{};
    size_t packets_before_requested_start{};
    size_t packets_after_requested_stop{};
    const auto* const packets_start = reinterpret_cast<const uint8_t*>(file_data.data() + mdsr.ds_offset);
    const uint8_t* next_packet = reinterpret_cast<const uint8_t*>(file_data.data() + mdsr.ds_offset);
    const auto start_mjd = PtimeToMjd(packets_start_filter);
    const auto stop_mjd = PtimeToMjd(packets_stop_filter);

    bool echo{false};
    bool cal{false};
    size_t i{};
    for (; i < mdsr.num_dsr; i++) {
        ForecastMeta current_meta;
        current_meta.packet_start_offset_bytes = next_packet - packets_start;
        next_packet = FetchPacketFrom(next_packet, current_meta, echo, cal);

        // LOGD << i << " " << MjdToPtime(current_meta.isp_sensing_time);

        if (forecast_meta.size() == 0 && !echo) {
            continue;
        }

        if (!echo && !cal) {
            continue;
        }

        if (current_meta.isp_sensing_time < start_mjd) {
            packets_before_requested_start++;
        }

        if (current_meta.isp_sensing_time > stop_mjd) {
            packets_after_requested_stop++;
        }

        // In case the packets_after_stop is equal to 0. Delete one at the back.
        if (packets_after_requested_stop > packets_after_stop) {
            packets_after_requested_stop--;
            break;
        }

        forecast_meta.push_back(current_meta);
    }

    // std::ofstream debug_stream("/tmp/fetch_meta.debug.csv");
    //    for (size_t j{}; j < 10; j++) {
    //        LOGD << j << " " << MjdToPtime(forecast_meta.at(j).isp_sensing_time);
    //    }
    //
    //    LOGD << "FROM the back";
    //    for (size_t j{forecast_meta.size() - 10}; j < forecast_meta.size(); j++) {
    //        LOGD << j << " " << MjdToPtime(forecast_meta.at(j).isp_sensing_time);
    //    }

    if (packets_before_requested_start > packets_before_start) {
        forecast_meta.erase(forecast_meta.begin(),
                            forecast_meta.begin() + (packets_before_requested_start - packets_before_start));
        packets_before_requested_start -= (packets_before_requested_start - packets_before_start);

        // Make sure that the first packet is not an echo!
        auto forecast_meta_it = forecast_meta.begin();
        while (forecast_meta_it != forecast_meta.end()) {
            ForecastMeta meta;
            bool is_echo{false};
            bool is_cal{false};
            [[maybe_unused]] auto discard =
                FetchPacketFrom(packets_start + forecast_meta_it->packet_start_offset_bytes, meta, is_echo, is_cal);
            if (!is_echo) {
                if (!cal) {
                    throw std::runtime_error(
                        "Programming error detected - only echo and calibration packets should have been fetched.");
                }
                forecast_meta.erase(forecast_meta_it);
                packets_before_requested_start--;
                forecast_meta_it = forecast_meta.begin();
            } else {
                break;
            }
        }
    }

    packets_before_start = packets_before_requested_start;
    packets_after_stop = packets_after_requested_stop;

    return forecast_meta;
}

RawSampleMeasurements ParseEnvisatLevel0ImPackets(const std::vector<char>& file_data, size_t mdsr_offset_bytes,
                                                  const std::vector<ForecastMeta>& entries_to_be_parsed,
                                                  InstrumentFile& ins_file,
                                                  std::vector<CommonPacketMetadata>& common_metadata) {
    const uint8_t* const packets_start = reinterpret_cast<const uint8_t*>(file_data.data()) + mdsr_offset_bytes;
    const uint8_t* it = packets_start + entries_to_be_parsed.front().packet_start_offset_bytes;
    const auto num_dsr = entries_to_be_parsed.size();

    LOGD << MjdToPtime(entries_to_be_parsed.front().isp_sensing_time);
    LOGD << entries_to_be_parsed.front().packet_start_offset_bytes;

    const auto range_sampling_rate = ins_file.flp.radar_sampling_rate;
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
    diagnostics.packet_data_blocks_min = UINT64_MAX;
    diagnostics.packet_data_blocks_max = 0;
#endif

    const auto max_forecasted_sample_blocks_bytes = ForecastMaxBlockDataLength(it, num_dsr);
    LOGD << "Forecasted maximum sample blocks bytes/length per range " << max_forecasted_sample_blocks_bytes;
    const auto compressed_sample_blocks_with_length_single_range_item_bytes =
        max_forecasted_sample_blocks_bytes + 2;  // uint16_t for length
    const auto alloc_bytes_for_sample_blocks_with_length =
        num_dsr * compressed_sample_blocks_with_length_single_range_item_bytes;
    LOGD << "Reserving " << alloc_bytes_for_sample_blocks_with_length / (1 << 20)
         << "MiB for raw sample blocks buffer including prepending 2 byte length marker ("
         << compressed_sample_blocks_with_length_single_range_item_bytes << "x" << num_dsr << ")";
    RawSampleMeasurements raw_measurements{};
    raw_measurements.single_entry_length = max_forecasted_sample_blocks_bytes;
    raw_measurements.entries_total = num_dsr;
    raw_measurements.raw_samples = std::unique_ptr<uint8_t[]>(new uint8_t[alloc_bytes_for_sample_blocks_with_length]);
    auto compressed_sample_blocks_buffer = raw_measurements.raw_samples.get();

    size_t max_samples{};
    size_t& total_samples = raw_measurements.total_samples;
    size_t compressed_sample_measurement_sets_stored_in_buffer{};
    size_t packet_index{};
    for (const auto& entry : entries_to_be_parsed) {
        it = packets_start + entry.packet_start_offset_bytes;
        EchoMeta echo_meta = {};

        // Source: ENVISAT-1 ASAR INTERPRETATION OF SOURCE PACKET DATA
        // PO-TN-MMS-SR-0248
        it = CopyBSwapPOD(echo_meta.isp_sensing_time, it);

        if (echo_meta.isp_sensing_time != entry.isp_sensing_time) {
            throw std::runtime_error(fmt::format(
                "Discrepancy between packet list '{}' and actual parsed packet sensing time '{}' at packet index {}",
                to_simple_string(MjdToPtime(entry.isp_sensing_time)),
                to_simple_string(MjdToPtime(echo_meta.isp_sensing_time)), packet_index));
        }

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

        if (datafield_length != 29) {
            throw std::runtime_error(
                fmt::format("DSR index = {} parsing error. Date Field Header Length should be 29 - is {}", packet_index,
                            datafield_length));
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
        // echo_window_code specifies how many samples, without block ID bytes e.g. data_len - block count or
        // packet_length - 29 - block_count
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
#if HANDLE_DIAGNOSTICS
            diagnostics.packet_data_blocks_min = std::min(diagnostics.packet_data_blocks_min, data_len);
            diagnostics.packet_data_blocks_max = std::max(diagnostics.packet_data_blocks_max, data_len);
#endif
            uint16_t sample_blocks_length = static_cast<uint16_t>(data_len);
            const auto sample_blocks_offset_bytes =
                (packet_index * compressed_sample_blocks_with_length_single_range_item_bytes);
            const auto sample_blocks_buffer_start = compressed_sample_blocks_buffer + sample_blocks_offset_bytes;
            memcpy(sample_blocks_buffer_start, &sample_blocks_length, sizeof(sample_blocks_length));
            memcpy(sample_blocks_buffer_start + 2, it, data_len);
            // First item is the count of whole blocks times 63 samples which is included in block, lastly is added
            // any samples from remainder block that does not consist full 63 samples
            const auto current_samples =
                static_cast<size_t>(((sample_blocks_length / 64) * 63) + (sample_blocks_length % 64) -
                                    ((sample_blocks_length % 64) > 0 ? 1 : 0));
            max_samples = std::max(max_samples, current_samples);
            echo_meta.raw_samples = current_samples;
            total_samples += current_samples;
            compressed_sample_measurement_sets_stored_in_buffer++;

            TransferMetadata(echo_meta, sample_blocks_length, common_metadata.at(packet_index));
        } else if (echo_meta.cal_flag) {
            if (packet_index > 0) {
                auto copy_echo = common_metadata.at(packet_index - 1);
                const auto prf = (range_sampling_rate / copy_echo.pri_code);
                const auto micros_to_add = static_cast<ul>((1 / prf) * 10e5);
                copy_echo.sensing_time = MjdAddMicros(copy_echo.sensing_time, micros_to_add);
                memcpy(compressed_sample_blocks_buffer +
                           packet_index * compressed_sample_blocks_with_length_single_range_item_bytes,
                       compressed_sample_blocks_buffer +
                           (packet_index - 1) * compressed_sample_blocks_with_length_single_range_item_bytes,
                       compressed_sample_blocks_with_length_single_range_item_bytes);
                total_samples += copy_echo.sample_count;
                compressed_sample_measurement_sets_stored_in_buffer++;
                common_metadata.at(packet_index) = copy_echo;
            }
#if HANDLE_DIAGNOSTICS
            last_cycle_packet_count = 0;  // Special value to mark reset;
            diagnostics.calibration_packet_count++;
#endif
        } else {
            throw std::runtime_error(
                "Programming error detected - only echo and calibration packets should have been fetched.");
        }

        packet_index++;

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
    }

#if DEBUG_PACKETS
    DebugEnvisatPackets(envisat_dbg_meta, envisat_dbg_meta.size(), debug_stream);
#endif

    const auto forecasted_max_samples = (max_forecasted_sample_blocks_bytes / 64) * 63 +
                                        (max_forecasted_sample_blocks_bytes % 64) -
                                        ((max_forecasted_sample_blocks_bytes % 64) > 0 ? 1 : 0);
    if (max_samples != forecasted_max_samples) {
        throw std::runtime_error(fmt::format("Post-parse max samples {} do not equal with forecast {}", max_samples,
                                             forecasted_max_samples));
    }

    raw_measurements.max_samples = max_samples;

    if (entries_to_be_parsed.size() != compressed_sample_measurement_sets_stored_in_buffer) {
        throw std::runtime_error(
            fmt::format("Programming error detected, there were {} packets fetch to be stored but only {} compressed "
                        "sample measurements copied",
                        entries_to_be_parsed.size(), compressed_sample_measurement_sets_stored_in_buffer));
    }

#if HANDLE_DIAGNOSTICS
    LOGD << "Diagnostic summary of the packets";
    LOGD << "Calibration packets - " << diagnostics.calibration_packet_count;
    LOGD << "Seq ctrl [oos total] " << diagnostics.sequence_control_oos << " "
         << diagnostics.sequence_control_total_missing;
    LOGD << "Mode packet [oos total] " << diagnostics.mode_packet_count_oos << " "
         << diagnostics.mode_packet_total_missing;
    LOGD << "Cycle packet [oos total] " << diagnostics.cycle_packet_count_oos << " "
         << diagnostics.cycle_packet_total_missing;
    LOGD << "MIN|MAX datablock lengths in bytes " << diagnostics.packet_data_blocks_min << "|"
         << diagnostics.packet_data_blocks_max;
#endif

    return raw_measurements;
}

void ParseEnvisatLevel0ImPackets(const std::vector<char>& file_data, const DSD_lvl0& mdsr, SARMetadata& sar_meta,
                                 ASARMetadata& asar_meta, cufftComplex** d_parsed_packets, InstrumentFile& ins_file,
                                 boost::posix_time::ptime packets_start_filter,
                                 boost::posix_time::ptime packets_stop_filter) {
    FillFbaqMetaAsar(asar_meta);
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
    diagnostics.packet_data_blocks_min = UINT64_MAX;
    diagnostics.packet_data_blocks_max = 0;
#endif
    const auto start_filter_mjd = PtimeToMjd(packets_start_filter);
    const auto end_filter_mjd = PtimeToMjd(packets_stop_filter);

    const auto max_forecasted_sample_blocks_bytes = ForecastMaxBlockDataLength(it, mdsr.num_dsr);
    LOGD << "Forecasted maximum sample blocks bytes/length per range " << max_forecasted_sample_blocks_bytes;
    const auto compressed_sample_blocks_with_length_single_range_item_bytes =
        max_forecasted_sample_blocks_bytes + 2;  // uint16_t for length
    const auto alloc_bytes_for_sample_blocks_with_length =
        mdsr.num_dsr * compressed_sample_blocks_with_length_single_range_item_bytes;
    LOGD << "Reserving " << alloc_bytes_for_sample_blocks_with_length / (1 << 20)
         << "MiB for raw sample blocks buffer including prepending 2 byte length marker ("
         << compressed_sample_blocks_with_length_single_range_item_bytes << "x" << mdsr.num_dsr << ")";
    std::unique_ptr<uint8_t[]> compressed_sample_blocks_buffer(new uint8_t[alloc_bytes_for_sample_blocks_with_length]);

    size_t max_samples{};
    size_t compressed_sample_measurements_stored_in_buf{};
    sar_meta.total_raw_samples = 0;

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
        // echo_window_code specifies how many samples, without block ID bytes e.g. data_len - block count or
        // packet_length - 29 - block_count
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
#if HANDLE_DIAGNOSTICS
            diagnostics.packet_data_blocks_min = std::min(diagnostics.packet_data_blocks_min, data_len);
            diagnostics.packet_data_blocks_max = std::max(diagnostics.packet_data_blocks_max, data_len);
#endif
            uint16_t sample_blocks_length = static_cast<uint16_t>(data_len);
            const auto sample_blocks_offset_bytes =
                (echoes.size() * compressed_sample_blocks_with_length_single_range_item_bytes);
            const auto sample_blocks_buffer_start = compressed_sample_blocks_buffer.get() + sample_blocks_offset_bytes;
            memcpy(sample_blocks_buffer_start, &sample_blocks_length, sizeof(sample_blocks_length));
            memcpy(sample_blocks_buffer_start + 2, it, data_len);
            compressed_sample_measurements_stored_in_buf++;
            // First item is the count of whole blocks times 63 samples which is included in block, lastly is added
            // any samples from remainder block that does not consist full 63 samples
            const auto current_samples =
                static_cast<size_t>(((sample_blocks_length / 64) * 63) + (sample_blocks_length % 64) -
                                    ((sample_blocks_length % 64) > 0 ? 1 : 0));
            max_samples = std::max(max_samples, current_samples);
            echo_meta.raw_samples = current_samples;
            sar_meta.total_raw_samples += current_samples;

            echoes.push_back(std::move(echo_meta));
        } else if (echo_meta.cal_flag) {
            if (echoes.size() > 0) {
                auto copy_echo = echoes.back();
                const auto prf = (sar_meta.chirp.range_sampling_rate / copy_echo.pri_code);
                const auto micros_to_add = static_cast<ul>((1 / prf) * 10e5);
                copy_echo.isp_sensing_time = MjdAddMicros(copy_echo.isp_sensing_time, micros_to_add);
                memcpy(compressed_sample_blocks_buffer.get() +
                           echoes.size() * compressed_sample_blocks_with_length_single_range_item_bytes,
                       compressed_sample_blocks_buffer.get() +
                           (echoes.size() - 1) * compressed_sample_blocks_with_length_single_range_item_bytes,
                       compressed_sample_blocks_with_length_single_range_item_bytes);
                compressed_sample_measurements_stored_in_buf++;
                sar_meta.total_raw_samples += copy_echo.raw_samples;
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

    for (auto& e : echoes) {
        if (prev_swst != e.swst_code) {
            prev_swst = e.swst_code;
            swst_changes++;
        }
        min_swst = std::min(min_swst, e.swst_code);
        max_swst = std::max(max_swst, e.swst_code);
    }

    const auto forecasted_max_samples = (max_forecasted_sample_blocks_bytes / 64) * 63 +
                                        (max_forecasted_sample_blocks_bytes % 64) -
                                        ((max_forecasted_sample_blocks_bytes % 64) > 0 ? 1 : 0);
    if (max_samples != forecasted_max_samples) {
        LOGW << "Post-parse max samples " << max_samples << " do not equal with forecast " << forecasted_max_samples;
    }

    if (echoes.size() != compressed_sample_measurements_stored_in_buf) {
        throw std::runtime_error(
            fmt::format("Programming error detected, there were {} packets' metadata stored but only {} compressed "
                        "sample measurements copied",
                        echoes.size(), compressed_sample_measurements_stored_in_buf));
    }

    int swath_idx = SwathIdx(asar_meta.swath);
    asar_meta.swath_idx = swath_idx;
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
    LOGD << "SWST changes " << swst_changes << " MIN|MAX SWST " << min_swst << "|" << max_swst;

    asar_meta.swst_rank = n_pulses_swst;

    // TODO Range gate bias must be included in this calculation
    sar_meta.pulse_repetition_frequency = sar_meta.chirp.range_sampling_rate / echoes.front().pri_code;
    sar_meta.line_time_interval = 1 / sar_meta.pulse_repetition_frequency;
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
        const auto& e = echoes[y];
        swst_codes.push_back(e.swst_code);
    }

    std::vector<float> i_lut(4096);
    memcpy(i_lut.data(), ins_file.fbp.i_LUT_fbaq4, 4096 * sizeof(float));
    std::vector<float> q_lut(4096);
    memcpy(q_lut.data(), ins_file.fbp.q_LUT_fbaq4, 4096 * sizeof(float));
    ConvertAsarImBlocksToComplex(compressed_sample_blocks_buffer.get(),
                                 compressed_sample_blocks_with_length_single_range_item_bytes, echoes.size(),
                                 swst_codes.data(), min_swst, *d_parsed_packets, sar_meta.img.range_size, i_lut.data(),
                                 q_lut.data(), 4096);

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
    LOGD << "MIN|MAX datablock lengths in bytes " << diagnostics.packet_data_blocks_min << "|"
         << diagnostics.packet_data_blocks_max;
#endif
}
}  // namespace alus::asar::envformat
