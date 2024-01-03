/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "main_flow.h"

#include <optional>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "fmt/format.h"

#include "alus_log.h"
#include "cuda_algorithm.h"
#include "date_time_util.h"
#include "envisat_format_kernels.h"
#include "envisat_utils.h"
#include "ers_aux_file.h"
#include "img_output.h"
#include "plot.h"
#include "globe_plot.h"
#include "sar/iq_correction.cuh"
#include "status_assembly.h"

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

void FillFbaqMetaSar(ASARMetadata& asar_meta) {
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
}  // namespace

namespace alus::asar::mainflow {

void CheckAndLimitSensingStartEnd(boost::posix_time::ptime& product_sensing_start,
                                  boost::posix_time::ptime& product_sensing_stop,
                                  std::optional<std::string_view> user_sensing_start,
                                  std::optional<std::string_view> user_sensing_end) {
    if (!user_sensing_start.has_value()) {
        return;
    }

    const auto user_start = alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff(std::string(user_sensing_start.value()));
    if (user_start < product_sensing_start) {
        LOGE << "User specified sensing start '" << boost::posix_time::to_simple_string(user_start)
             << "' is earlier than product's packets - " << boost::posix_time::to_simple_string(product_sensing_start);
        exit(alus::asar::status::EXIT_CODE::INVALID_ARGUMENT);
    }

    if (user_start > product_sensing_stop) {
        LOGE << "User specified sensing start '" << boost::posix_time::to_simple_string(user_start)
             << "' is later than product's sensing end time - "
             << boost::posix_time::to_simple_string(product_sensing_stop);
        exit(alus::asar::status::EXIT_CODE::INVALID_ARGUMENT);
    }

    product_sensing_start = user_start;

    if (!user_sensing_end.has_value()) {
        return;
    }

    const auto user_end = alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff(std::string(user_sensing_end.value()));
    if (user_end < user_start) {
        LOGE << "Invalid sensing end time specified, because it is earlier than start.";
        exit(alus::asar::status::EXIT_CODE::INVALID_ARGUMENT);
    }

    if (user_end > product_sensing_stop) {
        LOGE << "User specified sensing end '" << boost::posix_time::to_simple_string(user_end)
             << "' is later than product's sensing end time - "
             << boost::posix_time::to_simple_string(product_sensing_stop);
        exit(alus::asar::status::EXIT_CODE::INVALID_ARGUMENT);
    }

    product_sensing_stop = user_end;
}

specification::ProductTypes TryDetermineProductType(std::string_view product_name) {
    specification::ProductTypes product_type = specification::GetProductTypeFrom(product_name);
    if (product_type == specification::ProductTypes::UNIDENTIFIED) {
        LOGE << "This product is not supported - " << product_name;
        exit(alus::asar::status::EXIT_CODE::UNSUPPORTED_DATASET);
    }

    return product_type;
}

void TryFetchOrbit(alus::dorisorbit::Parsable& orbit_source, ASARMetadata& asar_meta, SARMetadata& sar_meta) {
    sar_meta.osv = orbit_source.CreateOrbitInfo(asar_meta.sensing_start, asar_meta.sensing_stop);

    if (sar_meta.osv.size() < 8) {
        LOGE << "Could not fetch enough orbit state vectors from input '"
             << orbit_source.GetL1ProductMetadata().orbit_name << "' atleast 8 required.";
        if (sar_meta.osv.size() > 0) {
            LOGE << "First OSV date - " << boost::posix_time::to_simple_string(sar_meta.osv.front().time);
            LOGE << "Last OSV date - " << boost::posix_time::to_simple_string(sar_meta.osv.back().time);
        }
        exit(alus::asar::status::EXIT_CODE::INVALID_DATASET);
    }
}

void FetchAuxFiles(InstrumentFile& ins_file, ConfigurationFile& conf_file, envformat::aux::ExternalCalibration& xca,
                   ASARMetadata& asar_meta, specification::ProductTypes product_type, std::string_view aux_path) {
    if (product_type == alus::asar::specification::ProductTypes::ASA_IM0) {
        FindINSFile(std::string(aux_path), asar_meta.sensing_start, ins_file, asar_meta.instrument_file);
        FindCONFile(std::string(aux_path), asar_meta.sensing_start, conf_file, asar_meta.configuration_file);
        envformat::aux::GetXca(aux_path, asar_meta.sensing_start, xca, asar_meta.external_calibration_file);
    } else if (product_type == alus::asar::specification::ProductTypes::SAR_IM0) {
        alus::asar::envisat_format::FindINSFile(std::string(aux_path), asar_meta.sensing_start, ins_file,
                                                asar_meta.instrument_file);
        alus::asar::envisat_format::FindCONFile(std::string(aux_path), asar_meta.sensing_start, conf_file,
                                                asar_meta.configuration_file);
        envformat::aux::GetXca(aux_path, asar_meta.sensing_start, xca, asar_meta.external_calibration_file);
    } else {
        LOGE << "Implementation error for this type of product while trying to fetch auxiliary files";
        exit(alus::asar::status::EXIT_CODE::ASSERT_FAILURE);
    }
}

void FormatResults(DevicePaddedImage& img, char* dest_space, size_t record_header_size, float calibration_constant) {
    envformat::ConditionResults(img, dest_space, record_header_size, calibration_constant);
}

void StorePlots(std::string output_path, std::string product_name, const SARMetadata& sar_metadata,
                const std::vector<std::complex<float>>& chirp) {
    {
        PlotArgs plot_args = {};
        plot_args.out_path = output_path + "/" + product_name + "_Vr.html";
        plot_args.x_axis_title = "range index";
        plot_args.y_axis_title = "Vr(m/s)";
        plot_args.data.resize(1);
        auto& line = plot_args.data[0];
        line.line_name = "Vr";
        PolyvalRange(sar_metadata.results.Vr_poly, 0, sar_metadata.img.range_size, line.x, line.y);
        Plot(plot_args);
    }
    {
        PlotArgs plot_args = {};
        plot_args.out_path = output_path + "/" + product_name + "_chirp.html";
        plot_args.x_axis_title = "nth sample";
        plot_args.y_axis_title = "Amplitude";
        plot_args.data.resize(2);
        auto& i = plot_args.data[0];
        auto& q = plot_args.data[1];
        std::vector<double> n_samp;
        int cnt = 0;
        for (auto iq : chirp) {
            i.y.push_back(iq.real());
            i.x.push_back(cnt);
            q.y.push_back(iq.imag());
            q.x.push_back(cnt);
            cnt++;
        }

        i.line_name = "I";
        q.line_name = "Q";
        Plot(plot_args);
    }
    {
        PlotArgs plot_args = {};
        plot_args.out_path = output_path + "/" + product_name + "_dc.html";
        plot_args.x_axis_title = "range index";
        plot_args.y_axis_title = "Hz";
        plot_args.data.resize(1);
        auto& line = plot_args.data[0];
        line.line_name = "Doppler centroid";
        PolyvalRange(sar_metadata.results.doppler_centroid_poly, 0, sar_metadata.img.range_size, line.x, line.y);
        Plot(plot_args);
    }
    {
        PlotGlobe(sar_metadata, output_path + "/" + product_name + "_globe.html");
    }
}

void StoreIntensity(std::string output_path, std::string product_name, std::string postfix,
                    const DevicePaddedImage& dev_padded_img) {
    std::string path = output_path + "/";
    path += product_name + "_" + postfix + ".tif";
    WriteIntensityPaddedImg(dev_padded_img, path.c_str());
}

void AssembleMetadataFrom(std::vector<envformat::CommonPacketMetadata>& parsed_meta, ASARMetadata& asar_meta,
                          SARMetadata& sar_meta, InstrumentFile& ins_file, size_t max_raw_samples_at_range,
                          size_t total_raw_samples, alus::asar::specification::ProductTypes product_type) {
    uint16_t min_swst = UINT16_MAX;
    uint16_t max_swst = 0;
    uint16_t swst_changes = 0;
    uint16_t prev_swst = parsed_meta.front().swst_code;
    const auto instrument = specification::GetInstrumentFrom(product_type);
    const int swst_multiplier{instrument == specification::Instrument::ASAR ? 1 : 4};

    sar_meta.carrier_frequency = ins_file.flp.radar_frequency;
    sar_meta.chirp.range_sampling_rate = ins_file.flp.radar_sampling_rate;

    asar_meta.first_swst_code = parsed_meta.front().swst_code;
    asar_meta.last_swst_code = parsed_meta.back().swst_code;
    if (instrument == specification::Instrument::ASAR) {
        asar_meta.tx_bw_code = parsed_meta.front().asar.chirp_pulse_bw_code;
        asar_meta.up_code = parsed_meta.front().asar.upconverter_raw;
        asar_meta.down_code = parsed_meta.front().asar.downconverter_raw;
        asar_meta.tx_pulse_len_code = parsed_meta.front().asar.tx_pulse_code;
        asar_meta.beam_set_num_code = parsed_meta.front().asar.antenna_beam_set_no;
        asar_meta.beam_adj_code = parsed_meta.front().asar.beam_adj_delta;
        asar_meta.resamp_code = parsed_meta.front().asar.resampling_factor;
        FillFbaqMetaAsar(asar_meta);
    }

    if (instrument == specification::Instrument::SAR) {
        asar_meta.tx_bw_code = 0;
        asar_meta.up_code = 0;
        asar_meta.down_code = 0;
        asar_meta.tx_pulse_len_code = 0;
        asar_meta.beam_set_num_code = 0;
        asar_meta.beam_adj_code = 0;
        asar_meta.resamp_code = 0;
        FillFbaqMetaSar(asar_meta);
    }

    asar_meta.pri_code = parsed_meta.front().pri_code;
    asar_meta.first_mjd = parsed_meta.front().sensing_time;
    asar_meta.first_sbt = parsed_meta.front().onboard_time;
    asar_meta.last_mjd = parsed_meta.back().sensing_time;
    asar_meta.last_sbt = parsed_meta.back().onboard_time;

    if (instrument == specification::Instrument::ASAR) {
        for (const auto& e : parsed_meta) {
            if (prev_swst != e.swst_code) {
                prev_swst = e.swst_code;
                swst_changes++;
            }
            min_swst = std::min(min_swst, e.swst_code);
            max_swst = std::max(max_swst, e.swst_code);
        }
    }

    if (instrument == specification::Instrument::SAR) {
        for (auto& e : parsed_meta) {
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
    }

    int swath_idx = envformat::SwathIdx(asar_meta.swath);
    LOGD << "Swath = " << asar_meta.swath << " idx = " << swath_idx;

    asar_meta.swst_changes = swst_changes;

    asar_meta.first_line_time = MjdToPtime(parsed_meta.front().sensing_time);
    asar_meta.sensing_start = asar_meta.first_line_time;
    asar_meta.last_line_time = MjdToPtime(parsed_meta.back().sensing_time);
    asar_meta.sensing_stop = asar_meta.last_line_time;

    // TODO - Pulse bandwidth thing is crazy since it is recalculated couple of rows below.
    double pulse_bw{};
    if (instrument == specification::Instrument::ASAR) {
        pulse_bw = 16e6 / 255 * parsed_meta.front().asar.chirp_pulse_bw_code;
        sar_meta.chirp.pulse_bandwidth = pulse_bw;
    }

    if (instrument == specification::Instrument::SAR) {
        pulse_bw = 0;
        sar_meta.chirp.pulse_bandwidth = 0;  // TODO - Has worked thus far, but is it correct?
    }

    sar_meta.chirp.Kr = ins_file.flp.im_chirp[swath_idx].phase[2] * 2;  // TODO Envisat format multiply by 2?
    sar_meta.chirp.pulse_duration = ins_file.flp.im_chirp[swath_idx].duration;
    sar_meta.chirp.n_samples = std::round(sar_meta.chirp.pulse_duration / (1 / sar_meta.chirp.range_sampling_rate));

    sar_meta.chirp.pulse_bandwidth = sar_meta.chirp.Kr * sar_meta.chirp.pulse_duration;
    LOGD << fmt::format("Chirp Kr = {}, n_samp = {}", sar_meta.chirp.Kr, sar_meta.chirp.n_samples);
    LOGD << fmt::format("tx pulse calc dur = {}",
                        asar_meta.tx_pulse_len_code * (1 / sar_meta.chirp.range_sampling_rate));
    LOGD << fmt::format("Nominal chirp duration = {}", sar_meta.chirp.pulse_duration);
    LOGD << fmt::format("Calculated chirp BW = {} , meta bw = {}", sar_meta.chirp.pulse_bandwidth, pulse_bw);

    sar_meta.total_raw_samples = total_raw_samples;
    const size_t range_samples = max_raw_samples_at_range + swst_multiplier * (max_swst - min_swst);
    LOGD << fmt::format("range samples = {}, minimum range padded size = {}", range_samples,
                        range_samples + sar_meta.chirp.n_samples);

    constexpr double c = 299792458;

    constexpr uint32_t im_idx = 0;
    const uint32_t n_pulses_swst = ins_file.fbp.mode_timelines[im_idx].r_values[swath_idx];

    LOGD << "n PRI before SWST = " << n_pulses_swst;
    LOGD << "SWST changes " << swst_changes << " MIN|MAX SWST " << min_swst << "|" << max_swst;

    asar_meta.swst_rank = n_pulses_swst;

    if (instrument == specification::Instrument::ASAR) {
        // TODO Range gate bias must be included in this calculation
        sar_meta.pulse_repetition_frequency = sar_meta.chirp.range_sampling_rate / parsed_meta.front().pri_code;
        asar_meta.two_way_slant_range_time =
            (min_swst + n_pulses_swst * parsed_meta.front().pri_code) * (1 / sar_meta.chirp.range_sampling_rate);
    }

    if (instrument == specification::Instrument::SAR) {
        const auto delay_time = ins_file.flp.range_gate_bias;
        // ISCE2 uses 210.943006e-9, but
        // ERS-1-Satellite-to-Ground-Segment-Interface-Specification_01.09.1993_v2D_ER-IS-ESA-GS-0001_EECF05
        // note 5 in 4.4:28 specifies 210.94 ns, which draws more similar results to PF-ERS results.
        const auto pri = (parsed_meta.front().pri_code + 2.0) * 210.94e-9;
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
    sar_meta.img.azimuth_size = parsed_meta.size();

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

void ConvertRawSampleSetsToComplex(const envformat::RawSampleMeasurements& raw_measurements,
                                   const std::vector<envformat::CommonPacketMetadata>& parsed_meta,
                                   const SARMetadata& sar_meta, const InstrumentFile& ins_file,
                                   specification::ProductTypes product_type, cufftComplex** d_converted_measurements) {
    const auto range_az_total_items = sar_meta.img.range_size * sar_meta.img.azimuth_size;
    const auto range_az_total_bytes = range_az_total_items * sizeof(cufftComplex);
    CHECK_CUDA_ERR(cudaMalloc(d_converted_measurements, range_az_total_bytes));
    constexpr auto CLEAR_VALUE = NAN;
    constexpr cufftComplex CLEAR_VALUE_COMPLEX{CLEAR_VALUE, CLEAR_VALUE};
    static_assert(sizeof(CLEAR_VALUE_COMPLEX) == sizeof(CLEAR_VALUE) * 2);
    cuda::algorithm::Fill(*d_converted_measurements, range_az_total_items, CLEAR_VALUE_COMPLEX);

    std::vector<uint16_t> swst_codes;
    uint16_t min_swst{UINT16_MAX};
    swst_codes.reserve(parsed_meta.size());
    if (product_type == specification::ProductTypes::ASA_IM0) {
        for (const auto& pm : parsed_meta) {
            swst_codes.push_back(pm.swst_code);
            min_swst = std::min(min_swst, pm.swst_code);
        }
    } else if (product_type == specification::ProductTypes::SAR_IM0) {
        uint16_t prev_swst = parsed_meta.front().swst_code;
        for (const auto& pm : parsed_meta) {
            const auto swst_code = pm.swst_code;
            swst_codes.push_back(swst_code);
            if ((swst_code < 500) || (swst_code > 1500) || ((prev_swst - swst_code) % 22 != 0)) {
                throw std::runtime_error("Detected invalid SWST code " + std::to_string(swst_code) +
                                         " which should have been corrected before " + std::string(__FUNCTION__));
            }
            min_swst = std::min(min_swst, swst_code);
            prev_swst = swst_code;
        }
    } else {
        throw std::runtime_error("Unsupported product type for " + std::string(__FUNCTION__));
    }

    if (product_type == specification::ProductTypes::ASA_IM0) {
        std::vector<float> i_lut(4096);
        memcpy(i_lut.data(), ins_file.fbp.i_LUT_fbaq4, 4096 * sizeof(float));
        std::vector<float> q_lut(4096);
        memcpy(q_lut.data(), ins_file.fbp.q_LUT_fbaq4, 4096 * sizeof(float));
        // Single entry length does not include 2 bytes of block data byte length.
        envformat::ConvertAsarImBlocksToComplex(
            raw_measurements.raw_samples.get(), raw_measurements.single_entry_length + 2,
            raw_measurements.entries_total, swst_codes.data(), min_swst, *d_converted_measurements,
            sar_meta.img.range_size, i_lut.data(), q_lut.data(), 4096);
    } else if (product_type == specification::ProductTypes::SAR_IM0) {
        envformat::ConvertErsImSamplesToComplex(raw_measurements.raw_samples.get(), raw_measurements.max_samples,
                                                raw_measurements.entries_total, swst_codes.data(), min_swst,
                                                *d_converted_measurements, sar_meta.img.range_size);
    } else {
        throw std::runtime_error("Unsupported product type for " + std::string(__FUNCTION__));
    }
}

}  // namespace alus::asar::mainflow