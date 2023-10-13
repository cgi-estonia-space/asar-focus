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

#include "fmt/format.h"

#include "alus_log.h"
#include "asar_constants.h"
#include "envisat_aux_file.h"
#include "envisat_mph_sph_str_utils.h"
#include "ers_aux_file.h"
#include "sar/orbit_state_vector.h"

namespace {
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

double NadirLLParse(const std::string& str) {
    auto val = std::stol(str.substr(3, 8));
    if (str.at(0) == '-') {
        val = -val;
    }
    return val * 1e-6;
}

int SwathIdx(const std::string& swath) {
    if (swath.size() == 3 && swath[0] == 'I' && swath[1] == 'S') {
        char idx = swath[2];
        if (idx >= '1' && idx <= '7') {
            return idx - '1';
        }
    }
    ERROR_EXIT(swath + " = unknown swath");
}
}  // namespace

void ParseIMFile(const std::vector<char>& file_data, const char* aux_path, SARMetadata& sar_meta,
                 ASARMetadata& asar_meta, std::vector<std::complex<float>>& img_data,
                 alus::dorisorbit::Parsable& orbit_source) {
    ProductHeader mph = {};

    mph.Load(file_data.data(), MPH_SIZE);

    // LOGV << "Lvl1MPH =";
    // mph.PrintValues();

    asar_meta.product_name = mph.Get("PRODUCT");

    alus::asar::specification::ProductTypes product_type =
        alus::asar::specification::GetProductTypeFrom(asar_meta.product_name);

    if (product_type == alus::asar::specification::ProductTypes::UNIDENTIFIED) {
        ERROR_EXIT("This product is not supported - " + asar_meta.product_name);
    }

    asar_meta.sensing_start = StrToPtime(mph.Get("SENSING_START"));
    asar_meta.sensing_stop = StrToPtime(mph.Get("SENSING_STOP"));

    asar_meta.first_line_time = asar_meta.sensing_start;
    asar_meta.last_line_time = asar_meta.sensing_stop;

    sar_meta.osv = orbit_source.CreateOrbitInfo(asar_meta.sensing_start, asar_meta.sensing_stop);

    if (sar_meta.osv.size() < 8) {
        std::string msg = "Wrong date on orbit file?\nOSV start/end = ";
        if (sar_meta.osv.size() > 1) {
            msg += boost::posix_time::to_simple_string(sar_meta.osv.front().time) + " / ";
            msg += boost::posix_time::to_simple_string(sar_meta.osv.back().time);
        }
        msg += " Sensing start = " + boost::posix_time::to_simple_string(asar_meta.sensing_start) + "\n";
        ERROR_EXIT(msg);
    }

    LOGD << "Product name = " << asar_meta.product_name;
    LOGD << "SENSING START " << asar_meta.sensing_start;
    LOGD << "SENSIND END " << asar_meta.sensing_stop;
    LOGD << "ORBIT vectors start " << sar_meta.osv.front().time;
    LOGD << "ORBIT vectors end " << sar_meta.osv.back().time;
    CHECK_BOOL(asar_meta.sensing_start >= sar_meta.osv.front().time &&
               asar_meta.sensing_stop <= sar_meta.osv.back().time);

    InstrumentFile ins_file = {};
    ConfigurationFile conf_file = {};
    if (product_type == alus::asar::specification::ProductTypes::ASA_IM0) {
        FindINSFile(aux_path, asar_meta.sensing_start, ins_file, asar_meta.instrument_file);
        FindCONFile(aux_path, asar_meta.sensing_start, conf_file, asar_meta.configuration_file);
    } else if (product_type == alus::asar::specification::ProductTypes::SAR_IM0) {
        alus::asar::envisat_format::FindINSFile(aux_path, asar_meta.sensing_start, ins_file, asar_meta.instrument_file);
        alus::asar::envisat_format::FindCONFile(aux_path, asar_meta.sensing_start, conf_file,
                                                asar_meta.configuration_file);
    } else {
        LOGE << "WAT?";
        exit(1);
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

    asar_meta.product_name = mph.Get("PRODUCT");
    asar_meta.swath = sph.Get("SWATH");

    int swath_idx = SwathIdx(asar_meta.swath);
    LOGD << "Swath = " << asar_meta.swath << " idx = " << swath_idx;

    asar_meta.acquistion_station = mph.Get("ACQUISITION_STATION");
    asar_meta.processing_station = mph.Get("PROC_CENTER");
    asar_meta.polarization = sph.Get("TX_RX_POLAR");

    auto dsds = ExtractDSDs(sph);
    if (dsds.size() < 1) {
        ERROR_EXIT("Expected DSD to contain echos, none found");
    }

    auto mdsr = dsds.front();
    const uint8_t* it = reinterpret_cast<const uint8_t*>(file_data.data()) + mdsr.ds_offset;

    int swst_multiplier{1};
    if (product_type == alus::asar::specification::ProductTypes::SAR_IM0) {
        swst_multiplier = 4;
        // Nothing for ERS
        asar_meta.compression_metadata.echo_method = "NONE";
        asar_meta.compression_metadata.echo_ratio = "";
        asar_meta.compression_metadata.init_cal_method = "NONE";
        asar_meta.compression_metadata.init_cal_ratio = "";
        asar_meta.compression_metadata.noise_method = "NONE";
        asar_meta.compression_metadata.noise_ratio = "";
        asar_meta.compression_metadata.per_cal_method = "NONE";
        asar_meta.compression_metadata.per_cal_ratio = "";
    } else if (product_type == alus::asar::specification::ProductTypes::ASA_IM0) {
        asar_meta.compression_metadata.echo_method = "FBAQ";
        asar_meta.compression_metadata.echo_ratio = "4/8";
        asar_meta.compression_metadata.init_cal_method = "NONE";  // TBD
        asar_meta.compression_metadata.init_cal_ratio = "";       // TBD
        asar_meta.compression_metadata.noise_method = "NONE";     // TBD
        asar_meta.compression_metadata.noise_ratio = "";          // TBD
        asar_meta.compression_metadata.per_cal_method = "NONE";   // TBD
        asar_meta.compression_metadata.per_cal_ratio = "";        // TBD
    }

    std::vector<EchoMeta> echos;
    for (size_t i = 0; i < mdsr.num_dsr; i++) {
        EchoMeta echo_meta = {};
        // Source: ENVISAT-1 ASAR INTERPRETATION OF SOURCE PACKET DATA
        // PO-TN-MMS-SR-0248
        it = CopyBSwapPOD(echo_meta.isp_sensing_time, it);

        FEPAnnotations fep;
        it = CopyBSwapPOD(fep, it);

        if (product_type == alus::asar::specification::ProductTypes::ASA_IM0) {
            uint16_t packet_id;
            it = CopyBSwapPOD(packet_id, it);

            uint16_t sequence_control;
            it = CopyBSwapPOD(sequence_control, it);

            uint16_t packet_length;
            it = CopyBSwapPOD(packet_length, it);

            uint16_t datafield_length;
            it = CopyBSwapPOD(datafield_length, it);

            if (datafield_length != 29) {
                ERROR_EXIT(fmt::format("DSR nr = {} ({}) parsing error. Date Field Header Length should be 29 - is {}",
                                       i, mdsr.num_dsr, datafield_length));
            }

            it++;
            echo_meta.mode_id = *it;
            it++;

            if (echo_meta.mode_id != 0x54) {
                LOGE << "mode id = " << std::hex << echo_meta.mode_id << "- not IM mode!";
                exit(1);
            }
            echo_meta.onboard_time = 0;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[0]) << 32;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[1]) << 24;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[2]) << 16;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[3]) << 8;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[4]) << 0;

            it += 5;
            it++;  // spare byte after 40 bit onboard time

            echo_meta.mode_count = 0;
            echo_meta.mode_count |= static_cast<uint32_t>(it[0]) << 16;
            echo_meta.mode_count |= static_cast<uint32_t>(it[1]) << 8;
            echo_meta.mode_count |= static_cast<uint32_t>(it[2]) << 0;
            it += 3;

            echo_meta.antenna_beam_set_no = *it >> 2;
            echo_meta.comp_ratio = *it & 0x3;
            it++;

            echo_meta.echo_flag = *it & 0x80;
            echo_meta.noise_flag = *it & 0x40;
            echo_meta.cal_flag = *it & 0x20;
            echo_meta.cal_type = *it & 0x10;
            echo_meta.cycle_count = 0;
            echo_meta.cycle_count |= (it[0] & 0xF) << 8;
            echo_meta.cycle_count |= it[1] << 0;
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

            size_t data_len = packet_length - 29;

            if (echo_meta.echo_flag) {
                echo_meta.raw_data.reserve(echo_meta.echo_window_code);
                size_t n_blocks = data_len / 64;
                for (size_t i = 0; i < n_blocks; i++) {
                    const uint8_t* block_data = it + i * 64;
                    uint8_t block_id = block_data[0];
                    for (size_t j = 0; j < 63; j++) {
                        uint8_t i_codeword = block_data[1 + j] >> 4;
                        uint8_t q_codeword = block_data[1 + j] & 0xF;

                        float i_samp = ins_file.fbp.i_LUT_fbaq4[FBAQ4Idx(block_id, i_codeword)];
                        float q_samp = ins_file.fbp.q_LUT_fbaq4[FBAQ4Idx(block_id, q_codeword)];
                        echo_meta.raw_data.emplace_back(i_samp, q_samp);
                    }
                }

                echos.push_back(std::move(echo_meta));
            }

            it += (fep.isp_length - 29);
        } else if (product_type == alus::asar::specification::ProductTypes::SAR_IM0) {
            static uint32_t last_data_record_no{0};
            uint32_t dr_no{};
            it = CopyBSwapPOD(dr_no, it);
            if (last_data_record_no == 0) {
                last_data_record_no = dr_no - 1;
            }
            //            if (last_data_record_no + 1 != dr_no) {
            //                LOGD << "There are discrepancies between ERS data packets. Last no. " <<
            //                last_data_record_nr << " current no. " << dr_no;
            //            }

            last_data_record_no = dr_no;

            // uint8_t packet_counter = it[0];
            // uint8_t subcommutation_counter = it[1];
            it += 2;

            it += 8;
            if (it[0] != 0xAA) {
                std::cerr << "ERS data packet's auxiliary section shall start with 0xAA, instead " << std::hex << it[0]
                          << " was found" << std::endl;
                exit(1);
            }
            it += 1;
            it += 1;  // OGRC/OBRC flag and Orbit ID code
            echo_meta.onboard_time = 0;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[0]) << 24;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[1]) << 16;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[2]) << 8;
            echo_meta.onboard_time |= static_cast<uint64_t>(it[3]) << 0;
            it += 4;
            it += 6;  // activity task and image format counter;
            it = CopyBSwapPOD(echo_meta.swst_code, it);
            it = CopyBSwapPOD(echo_meta.pri_code, it);
            // https://earth.esa.int/eogateway/documents/20142/37627/ERS-products-specification-with-Envisat-format.pdf
            // https://asf.alaska.edu/wp-content/uploads/2019/03/ers_ceos.pdf mixed between two.
            it += 194 + 10;
            echo_meta.raw_data.reserve(11232);
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
            echos.push_back(std::move(echo_meta));
        } else {
            std::cerr << "WAT?" << std::endl;
            exit(1);
        }
    }

    uint16_t min_swst = UINT16_MAX;
    uint16_t max_swst = 0;
    size_t max_samples = 0;
    uint16_t swst_changes = 0;
    uint16_t prev_swst = echos.front().swst_code;

    asar_meta.first_swst_code = echos.front().swst_code;
    asar_meta.last_swst_code = echos.back().swst_code;
    asar_meta.tx_bw_code = echos.front().chirp_pulse_bw_code;
    asar_meta.up_code = echos.front().upconverter_raw;
    asar_meta.down_code = echos.front().downconverter_raw;
    asar_meta.pri_code = echos.front().pri_code;
    asar_meta.tx_pulse_len_code = echos.front().tx_pulse_code;
    asar_meta.beam_set_num_code = echos.front().antenna_beam_set_no;
    asar_meta.beam_adj_code = echos.front().beam_adj_delta;
    asar_meta.resamp_code = echos.front().resampling_factor;
    asar_meta.first_mjd = echos.front().isp_sensing_time;
    asar_meta.first_sbt = echos.front().onboard_time;
    asar_meta.last_mjd = echos.back().isp_sensing_time;
    asar_meta.last_sbt = echos.back().onboard_time;

    std::vector<float> v;
    if (product_type == alus::asar::specification::ProductTypes::ASA_IM0) {
        for (auto& e : echos) {
            if (prev_swst != e.swst_code) {
                prev_swst = e.swst_code;
                swst_changes++;
            }
            min_swst = std::min(min_swst, e.swst_code);
            max_swst = std::max(max_swst, e.swst_code);
            max_samples = std::max(max_samples, e.raw_data.size());
        }
    } else if (product_type == alus::asar::specification::ProductTypes::SAR_IM0) {
        for (auto& e : echos) {
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
    } else {
        std::cerr << "WTF?!" << std::endl;
        exit(1);
    }

    asar_meta.swst_changes = swst_changes;

    double pulse_bw = 16e6 / 255 * echos.front().chirp_pulse_bw_code;

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

    // TODO Range gate bias must be included in this calculation
    if (product_type == alus::asar::specification::ProductTypes::ASA_IM0) {
        sar_meta.pulse_repetition_frequency = sar_meta.chirp.range_sampling_rate / echos.front().pri_code;
        asar_meta.two_way_slant_range_time =
            (min_swst + n_pulses_swst * echos.front().pri_code) * (1 / sar_meta.chirp.range_sampling_rate);
    } else if (product_type == alus::asar::specification::ProductTypes::SAR_IM0) {
        const auto delay_time = ins_file.flp.range_gate_bias;
        // ISCE2 uses 210.943006e-9, but
        // ERS-1-Satellite-to-Ground-Segment-Interface-Specification_01.09.1993_v2D_ER-IS-ESA-GS-0001_EECF05
        // note 5 in 4.4:28 specifies 210.94 ns, which draws more similar results to PF-ERS results.
        const auto pri = (echos.front().pri_code + 2.0) * 210.94e-9;
        sar_meta.pulse_repetition_frequency = 1 / pri;

        if (sar_meta.pulse_repetition_frequency < 1640 || sar_meta.pulse_repetition_frequency > 1720) {
            LOGW << "PRF value '" << sar_meta.pulse_repetition_frequency << "'"
                 << "is out of the specification (ERS-1-Satellite-to-Ground-Segment-Interface-"
                 << "Specification_01.09.1993_v2D_ER-IS-ESA-GS-0001_EECF05) range [1640, 1720] Hz" << std::endl;
        }

        asar_meta.two_way_slant_range_time =
            n_pulses_swst * pri + min_swst * swst_multiplier / sar_meta.chirp.range_sampling_rate - delay_time;
    } else {
        LOGE << "WTF? " << __FUNCTION__ << " " << __LINE__ << std::endl;
        exit(1);
    }

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
    sar_meta.img.azimuth_size = echos.size();

    img_data.clear();
    img_data.resize(sar_meta.img.range_size * sar_meta.img.azimuth_size, {NAN, NAN});

    sar_meta.total_raw_samples = 0;

    for (size_t y = 0; y < echos.size(); y++) {
        const auto& e = echos[y];
        size_t idx = y * range_samples;
        idx += swst_multiplier * (e.swst_code - min_swst);
        const size_t n_samples = e.raw_data.size();
        memcpy(&img_data[idx], e.raw_data.data(), n_samples * 8);
        sar_meta.total_raw_samples += n_samples;
    }

    // TODO these two always seem to differ?
//    auto first_mjd = MjdToPtime(echos.front().isp_sensing_time);
//    LOGD << fmt::format("Sensing start = {} First ISP = {} Diff = {} us",
//                        to_iso_extended_string(asar_meta.sensing_start), to_iso_extended_string(first_mjd),
//                        (asar_meta.sensing_start - first_mjd).total_microseconds());

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
}
