/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */
#include "envisat_lvl1_writer.h"

#include <filesystem>

#include "alus_log.h"
#include "asar_constants.h"
#include "envisat_mph_sph_str_utils.h"
#include "geo_tools.h"
#include "sar/sar_metadata.h"

namespace {
template <class T>
void append_packed_struct(std::vector<uint8_t>& buffer, const T& data) {
    static_assert(alignof(T) == 1 && std::is_pod_v<T>);
    auto begin = reinterpret_cast<const uint8_t*>(&data);
    auto end = begin + sizeof(data);
    buffer.insert(buffer.end(), begin, end);
}

void FillMainProcessingParams(const SARMetadata& sar_meta, const ASARMetadata& asar_meta,
                              MainProcessingParametersADS& out) {
    const bool is_slc = alus::asar::specification::IsSLCProduct(asar_meta.target_product_type);
    {
        auto& gs = out.general_summary;
        gs.first_zero_doppler_time = PtimeToMjd(asar_meta.sensing_start);
        gs.last_zero_doppler_time = PtimeToMjd(asar_meta.sensing_stop);
        memcpy(gs.swath_num, asar_meta.swath.data(), 3);
        gs.range_spacing = sar_meta.range_spacing;
        gs.azimuth_spacing = sar_meta.azimuth_spacing;
        gs.line_time_interval = sar_meta.line_time_interval;
        gs.num_output_lines = sar_meta.img.azimuth_size;
        gs.num_samples_per_line = sar_meta.img.range_size;
        memcpy(gs.data_type, is_slc ? "SWORD" : "UWORD", 5);
        gs.num_range_lines_per_burst = 0;
        gs.time_diff_zero_doppler = 0;  // TODO azimuth cut
    }

    {
        auto& ips = out.image_processing_summary;
        ips.dop_cen_flag = 1;
        ips.srgr_flag = sar_meta.srgr.applied;
        ips.look_sum_flag = !is_slc;
        ips.detected_flag = !is_slc;
        ips.range_spread_comp_flag = !is_slc;
    }
    {
        auto& rd = out.raw_data_analysis[0];
        rd.calc_i_bias = sar_meta.results.dc_i;
        rd.calc_q_bias = sar_meta.results.dc_q;
        rd.calc_gain = sar_meta.results.iq_gain;
        rd.calc_quad = sar_meta.results.phase_error;
        // todo
    }

    {
        auto& dl = out.downlink_header;
        dl.first_obt1[0] = asar_meta.first_sbt >> 8;
        dl.first_obt1[1] = asar_meta.first_sbt & 0xFF;
        dl.first_mjd_1 = asar_meta.first_mjd;

        dl.swst_code[0] = asar_meta.first_swst_code;
        dl.last_swst_code[0] = asar_meta.last_swst_code;
        dl.pri_code[0] = asar_meta.pri_code;
        dl.tx_pulse_len_code[0] = asar_meta.tx_pulse_len_code;
        dl.tx_bw_code[0] = asar_meta.tx_bw_code;
        dl.echo_win_len_code[0] = asar_meta.echo_win_len_code;
        dl.up_code[0] = asar_meta.up_code;
        dl.down_code[0] = asar_meta.down_code;
        dl.resamp_code[0] = asar_meta.down_code;
        dl.resamp_code[0] = asar_meta.resamp_code;
        dl.beam_adj_code[0] = asar_meta.beam_adj_code;
        dl.beam_set_num_code[0] = asar_meta.beam_set_num_code;
        dl.tx_monitor_code[0] = asar_meta.tx_monitor_code;

        // todo errors
        const double dt = 1 / sar_meta.chirp.range_sampling_rate;

        dl.swst_value[0] = asar_meta.first_swst_code * dt;
        dl.last_swst_value[0] = asar_meta.first_swst_code * dt;
        dl.swst_changes[0] = asar_meta.swst_changes;
        dl.prf_value[0] = sar_meta.pulse_repetition_frequency;
        dl.tx_pulse_len_value[0] = asar_meta.tx_pulse_len_code * dt;
        dl.tx_pulse_bw_value[0] = (static_cast<double>(asar_meta.tx_bw_code) / 255) * 16e6;
        dl.echo_win_len_value[0] = asar_meta.echo_win_len_code * dt;
        dl.up_value[0] = UpconverterLevelConversion(asar_meta.up_code);
        dl.tx_monitor_value[0] = AuxTxMonitorConversion(asar_meta.tx_monitor_code);

        // todo other values conversion???

        dl.rank[0] = asar_meta.swst_rank;
    }

    {
        auto& r = out.range_processing_information;
        r.range_ref = alus::asar::specification::REFERENCE_RANGE;
        r.range_samp_rate = sar_meta.chirp.range_sampling_rate;
        r.radar_freq = sar_meta.carrier_frequency;
        r.num_looks_range = 1;
        // r.filter_window has been prefilled.
        // r.filter_coef_range has been prefilled.

        if (sar_meta.chirp.apply_window) {
            r.filter_coef_range = sar_meta.chirp.alpha;
            CopyStrPad(r.filter_range, "HAMMING");
        } else {
            r.filter_coef_range = 0.0;
            CopyStrPad(r.filter_range, "NONE");
        }

        r.look_bw_range[0] = out.downlink_header.tx_pulse_bw_value[0];
        r.tot_bw_range[0] = out.downlink_header.tx_pulse_bw_value[0];

        r.nominal_chirp[0].nom_chirp_amp[0] = 1.0;
        r.nominal_chirp[0].nom_chirp_phs[2] =
            sar_meta.chirp.Kr / 2.0;  // TODO why is the nominal chirp div 2 in envisat files?
    }

    {
        auto& az = out.azimuth_processing_information;
        // Prefilled - az.num_lines_proc
        az.num_look_az = 1;
        az.to_bw_az = sar_meta.pulse_repetition_frequency * sar_meta.azimuth_bandwidth_fraction;
        CopyStrPad(az.filter_az, "NONE");

        if (sar_meta.chirp.apply_window) {
            az.filter_coef_az = sar_meta.azimuth_window_alpha;
            CopyStrPad(az.filter_az, "HAMMING");
        } else {
            az.filter_coef_az = 0.0;
            CopyStrPad(az.filter_az, "NONE");
        }

        // refit Vr polynomial to a Ka polynomial
        // internal Vr polynimal is fitted to range index, Envisat format uses two way slant range time from first
        // TODO range cutoff due to RCMC has a small effect on this
        std::vector<double> Ka_vec;
        std::vector<double> t_vec;
        constexpr int N = 10;
        double t0 = asar_meta.two_way_slant_range_time;
        for (int i = 0; i < N; i++) {
            int range_idx = (sar_meta.img.range_size * i) / N;
            double Vr = CalcVr(sar_meta, range_idx);
            double R0 = CalcSlantRange(sar_meta, range_idx);
            double Ka = -2 * (Vr * Vr) / (sar_meta.wavelength * R0);
            double two_way_slant_time = 2 * (R0 / SOL);
            Ka_vec.push_back(Ka);
            t_vec.push_back(two_way_slant_time - t0);
        }

        auto poly = Polyfit(t_vec, Ka_vec, 2);

        az.az_fm_rate[0] = poly.at(2);
        az.az_fm_rate[1] = poly.at(1);
        az.az_fm_rate[2] = poly.at(0);
        az.ax_fm_origin = t0 * 1e9;
        az.dop_amb_conf = 0.0f;
    }

    {
        auto& comp = out.data_compression_information;
        CopyStrPad(comp.echo_comp, asar_meta.compression_metadata.echo_method);
        CopyStrPad(comp.echo_comp_ratio, asar_meta.compression_metadata.echo_ratio);
        CopyStrPad(comp.init_cal_comp, asar_meta.compression_metadata.init_cal_method);
        CopyStrPad(comp.init_cal_ratio, asar_meta.compression_metadata.init_cal_ratio);
        CopyStrPad(comp.per_cal_comp, asar_meta.compression_metadata.per_cal_method);
        CopyStrPad(comp.per_cal_ratio, asar_meta.compression_metadata.per_cal_ratio);
        CopyStrPad(comp.noise_comp, asar_meta.compression_metadata.noise_method);
        CopyStrPad(comp.noise_comp_ratio, asar_meta.compression_metadata.noise_ratio);
    }

    {
        for (int i = 0; i < 5; i++) {
            int az_idx = (sar_meta.img.azimuth_size * i) / 4;
            auto time = CalcAzimuthTime(sar_meta, az_idx);
            auto iosv = InterpolateOrbit(sar_meta.osv, time);

            out.orbit_state_vectors[i].state_vect_time = PtimeToMjd(time);
            out.orbit_state_vectors[i].x_pos = iosv.x_pos * 1e2;
            out.orbit_state_vectors[i].y_pos = iosv.y_pos * 1e2;
            out.orbit_state_vectors[i].z_pos = iosv.z_pos * 1e2;
            out.orbit_state_vectors[i].x_vel = iosv.x_vel * 1e5;
            out.orbit_state_vectors[i].y_vel = iosv.y_vel * 1e5;
            out.orbit_state_vectors[i].z_vel = iosv.z_vel * 1e5;
        }
    }
}

void FillSummaryQuality(const SARMetadata& sar_meta, const ASARMetadata& asar_meta, SummaryQualityADS& sq) {
    sq = {};
    const auto& meta_sq = asar_meta.summary_quality;
    sq.zero_doppler_time = PtimeToMjd(sar_meta.first_line_time);
    sq.thresh_chirp_broadening = meta_sq.thresh_chirp_broadening;
    sq.thresh_chirp_sidelobe = meta_sq.thresh_chirp_sidelobe;
    sq.thresh_chirp_islr = meta_sq.thresh_chirp_islr;
    sq.thresh_input_mean = meta_sq.thresh_input_mean;
    sq.thresh_input_mean = meta_sq.thresh_input_mean;
    sq.exp_input_mean = meta_sq.exp_input_mean;
    sq.thresh_input_std_dev = meta_sq.thresh_input_std_dev;
    sq.exp_input_std_dev = meta_sq.exp_input_std_dev;

    sq.thresh_dop_cen = meta_sq.thresh_dop_cen;
    sq.thresh_dop_amb = meta_sq.thresh_dop_amb;
    sq.thresh_output_mean = meta_sq.thresh_output_mean;
    sq.exp_output_mean = meta_sq.exp_output_mean;
    sq.thresh_output_std_dev = meta_sq.thresh_output_std_dev;
    sq.exp_output_std_dev = meta_sq.exp_output_std_dev;
    sq.thresh_input_missing_lines = meta_sq.thresh_input_missing_lines;
    sq.thresh_input_gaps = meta_sq.thresh_input_gaps;
    sq.lines_per_gap = meta_sq.lines_per_gaps;

    sq.input_mean[0] = sar_meta.results.dc_i;
    sq.input_mean[1] = sar_meta.results.dc_q;
    sq.input_std_dev[0] = sar_meta.results.stddev_i;
    sq.input_std_dev[1] = sar_meta.results.stddev_q;

    sq.output_mean[0] = 1;
    sq.output_mean[1] = 2;
    sq.output_std_dev[0] = 3;
    sq.output_std_dev[1] = 4;

    CopyStr(sq.swath_nr, asar_meta.swath);

    if ((fabs(sq.input_mean[0] - sq.exp_input_mean)) > sq.thresh_input_mean) {
        sq.input_mean_flag = 1;
    }
    if ((fabs(sq.input_mean[1] - sq.exp_input_mean)) > sq.thresh_input_mean) {
        sq.input_mean_flag = 1;
    }

    if ((fabs(sq.input_std_dev[0] - sq.exp_input_std_dev)) > sq.thresh_input_std_dev) {
        sq.input_std_dev_flag = 1;
    }
    if ((fabs(sq.input_std_dev[1] - sq.exp_input_std_dev)) > sq.thresh_input_std_dev) {
        sq.input_std_dev_flag = 1;
    }

    // TODO
    // sq.output_mean = ...
    // sq.output_std_dev = ...
}

void FillGeoLocationAds(int az_idx, int az_last, const SARMetadata& sar_meta, GeoLocationADSR& geo) {
    auto first = CalcAzimuthTime(sar_meta, az_idx);
    auto last = CalcAzimuthTime(sar_meta, az_last);
    const int n = std::size(geo.first_line_tie_points.lats);
    geo.first_zero_doppler_time = PtimeToMjd(first);
    geo.last_zero_doppler_time = PtimeToMjd(last);
    geo.line_num = az_idx;
    geo.num_lines = az_last - az_idx;

    geo.sub_sat_track = 0.0;  // TODO
    const int range_size = sar_meta.img.range_size;
    for (int i = 0; i < n; i++) {
        const int range_samp = i * range_size / (n - 1);
        double R = CalcSlantRange(sar_meta, range_samp);
        if (sar_meta.srgr.applied) {
            const double ground_range = range_samp * sar_meta.range_spacing;
            const double slant_from_first =
                Polyval(sar_meta.srgr.grsr_poly.data(), sar_meta.srgr.grsr_poly.size(), ground_range);
            R = sar_meta.slant_range_first_sample + slant_from_first;
        }
        constexpr double c = 299792458;
        double two_way_slant_time = 2 * (R / c) * 1e9;
        {
            auto osv = InterpolateOrbit(sar_meta.osv, first);  // TODO OSV fast time interpolation
            auto xyz = RangeDopplerGeoLocate({osv.x_vel, osv.y_vel, osv.z_vel}, {osv.x_pos, osv.y_pos, osv.z_pos},
                                             sar_meta.center_point, R);
            geo.first_line_tie_points.samp_numbers[i] = range_samp + 1;
            auto llh = xyz2geoWGS84(xyz);
            geo.first_line_tie_points.lats[i] = llh.latitude * 1e6;
            geo.first_line_tie_points.longs[i] = llh.longitude * 1e6;
            geo.first_line_tie_points.slant_range_times[i] = two_way_slant_time;
            geo.first_line_tie_points.angles[i] = CalcIncidenceAngle(xyz, {osv.x_pos, osv.y_pos, osv.z_pos});
        }

        {
            auto osv = InterpolateOrbit(sar_meta.osv, last);
            auto xyz = RangeDopplerGeoLocate({osv.x_vel, osv.y_vel, osv.z_vel}, {osv.x_pos, osv.y_pos, osv.z_pos},
                                             sar_meta.center_point, R);
            geo.last_line_tie_points.samp_numbers[i] = range_samp + 1;
            auto llh = xyz2geoWGS84(xyz);
            geo.last_line_tie_points.lats[i] = llh.latitude * 1e6;
            geo.last_line_tie_points.longs[i] = llh.longitude * 1e6;
            geo.last_line_tie_points.slant_range_times[i] = two_way_slant_time;
            geo.last_line_tie_points.angles[i] = CalcIncidenceAngle(xyz, {osv.x_pos, osv.y_pos, osv.z_pos});
        }
    }
}
}  // namespace

std::vector<uint8_t> ConstructEnvisatFileHeader(EnvisatSubFiles& header_files, const SARMetadata& sar_meta,
                                                const ASARMetadata& asar_meta, const MDS& mds,
                                                std::string_view software_ver) {
    static_assert(__builtin_offsetof(EnvisatSubFiles, main_processing_params) == 7516);

    std::string out_name = asar_meta.product_name;
    const bool is_detected = alus::asar::specification::IsDetectedProduct(asar_meta.target_product_type);
    const bool is_slc = !is_detected;
    out_name.at(6) = is_slc ? 'S' : 'P';
    out_name.at(8) = '1';

    uint32_t file_offset = 0;

    {
        auto& mph = header_files.mph;
        mph.SetDefaults();

        mph.Set_PRODUCT(out_name);
        mph.Set_PROC_STAGE('N');
        if (asar_meta.product_name.substr(0, 3) == "ASA") {
            // https://earth.esa.int/eogateway/documents/20142/37627/PO-RS-507316_4_C_Envisat_Product_Spec_Vol8.pdf/
            mph.Set_REF_DOC("PO-RS-MDA-GS-2009_4/C");
        } else if (asar_meta.product_name.substr(0, 3) == "SAR") {
            // https://earth.esa.int/eogateway/documents/20142/37627/ERS-products-specification-with-Envisat-format.pdf
            mph.Set_REF_DOC("PX-SP-50-9105_3/1");
        } else {
            mph.Set_REF_DOC("TBD.pdf");
        }

        auto now = boost::posix_time::microsec_clock::universal_time();

        mph.SetDataAcqusitionProcessingInfo(asar_meta.acquistion_station, asar_meta.processing_station, PtimeToStr(now),
                                            std::string(software_ver));

        mph.SetSensingStartStop(PtimeToStr(asar_meta.sensing_start), PtimeToStr(asar_meta.sensing_stop));
        mph.SetOrbitInfo(asar_meta);
        mph.SetSbt(asar_meta);
        mph.SetLeap(asar_meta);
        mph.Set_PRODUCT_ERR(asar_meta.product_err ? '1' : '0');

        // set tot size later

        size_t tot = sizeof(header_files.mph) + sizeof(header_files.sph);
        tot += sizeof(header_files.summary_quality);
        tot += sizeof(header_files.dop_centroid_coeffs);
        // TODO chirp params
        // tot += sizeof(header_files.chirp_params);
        tot += sizeof(header_files.geolocation_grid);
        tot += sizeof(header_files.main_processing_params);
        size_t n_datasets = 5;
        if (is_detected) {
            tot += sizeof(header_files.srgr_params);
            n_datasets++;  // + srgr
        }
        tot += static_cast<size_t>(mds.n_records) * mds.record_size;
        mph.SetProductSizeInformation(tot, sizeof(header_files.sph), 18, n_datasets);
        file_offset += sizeof(mph);
    }
    {
        auto& sph = header_files.sph;
        sph.SetDefaults();

        sph.SetHeader(is_slc ? "Image Mode SLC Image" : "Image Mode Precision Image", 0, 1, 1);

        sph.SetLineTimeInfo(PtimeToStr(asar_meta.first_line_time), PtimeToStr(asar_meta.last_line_time));

        {
            auto osv = InterpolateOrbit(sar_meta.osv, sar_meta.first_line_time);
            // todo fast time adjustment
            Velocity3D vel = {osv.x_vel, osv.y_vel, osv.z_vel};
            GeoPos3D pos = {osv.x_pos, osv.y_pos, osv.z_pos};
            double r_near = sar_meta.slant_range_first_sample;
            double r_mid = r_near + sar_meta.range_spacing * sar_meta.img.range_size / 2;
            double r_far = r_near + sar_meta.range_spacing * sar_meta.img.range_size;
            auto near = RangeDopplerGeoLocate(vel, pos, sar_meta.center_point, r_near);
            auto mid = RangeDopplerGeoLocate(vel, pos, sar_meta.center_point, r_mid);
            auto far = RangeDopplerGeoLocate(vel, pos, sar_meta.center_point, r_far);
            sph.SetFirstPosition(xyz2geoWGS84(near), xyz2geoWGS84(mid), xyz2geoWGS84(far));
        }

        {
            auto osv = InterpolateOrbit(sar_meta.osv, asar_meta.last_line_time);
            // todo fast time adjustment
            Velocity3D vel = {osv.x_vel, osv.y_vel, osv.z_vel};
            GeoPos3D pos = {osv.x_pos, osv.y_pos, osv.z_pos};
            double r_near = sar_meta.slant_range_first_sample;
            double r_mid = r_near + sar_meta.range_spacing * sar_meta.img.range_size / 2;
            double r_far = r_near + sar_meta.range_spacing * sar_meta.img.range_size;
            auto near = RangeDopplerGeoLocate(vel, pos, sar_meta.center_point, r_near);
            auto mid = RangeDopplerGeoLocate(vel, pos, sar_meta.center_point, r_mid);
            auto far = RangeDopplerGeoLocate(vel, pos, sar_meta.center_point, r_far);
            sph.SetLastPosition(xyz2geoWGS84(near), xyz2geoWGS84(mid), xyz2geoWGS84(far));
        }

        sph.SetProductInfo1(asar_meta.swath, asar_meta.ascending ? "ASCENDING" : "DESCENDING",
                            is_slc ? "COMPLEX" : "DETECTED", "RAN/DOP");
        if (asar_meta.compression_metadata.echo_ratio.length() > 0) {
            sph.SetProductInfo2(
                asar_meta.polarization, "",
                asar_meta.compression_metadata.echo_method + asar_meta.compression_metadata.echo_ratio.front());
        } else {
            sph.SetProductInfo2(asar_meta.polarization, "", asar_meta.compression_metadata.echo_method);
        }
        sph.SetProductInfo3(is_slc ? 1 : 4, 1);
        sph.SetProductInfo4(sar_meta.range_spacing, sar_meta.azimuth_spacing, sar_meta.line_time_interval);

        const int pix_size = is_slc ? 4 : 2;
        if (sar_meta.img.range_size * pix_size != mds.record_size - 17) {
            throw std::runtime_error(
                "Range value for SAR metadata does not match MDS record size - this is an implementation error");
        }
        sph.SetProductInfo5(sar_meta.img.range_size, is_slc ? "SWORD" : "UWORD");

        header_files.main_processing_params.general_summary.num_samples_per_line = sar_meta.img.range_size;

        file_offset += sizeof(sph);

        {
            constexpr size_t size = sizeof(header_files.summary_quality);
            static_assert(size == 170);
            sph.dsds[0].SetInternalDSD("MDS1 SQ ADS", 'A', file_offset, 1, size);
            file_offset += size;

            FillSummaryQuality(sar_meta, asar_meta, header_files.summary_quality);
            header_files.summary_quality.BSwap();
        }

        { sph.dsds[1].SetEmptyDSD("MDS2 SQ ADS", 'A'); }

        {
            constexpr size_t size = sizeof(header_files.main_processing_params);
            static_assert(size == 2009);
            sph.dsds[2].SetInternalDSD("MAIN PROCESSING PARAMS ADS", 'A', file_offset, 1, size);
            file_offset += size;

            FillMainProcessingParams(sar_meta, asar_meta, header_files.main_processing_params);
            header_files.main_processing_params.BSwap();
        }

        {
            constexpr size_t size = sizeof(header_files.dop_centroid_coeffs);
            static_assert(size == 55);
            sph.dsds[3].SetInternalDSD("DOP CENTROID COEFFS ADS", 'A', file_offset, 1, size);
            file_offset += size;

            // convert DC polynomial to Envisat format
            //  polynomial internally fitted to range sample index, Envisat format uses two way slant range time,
            //  starting from first sample
            const double dt = 1 / sar_meta.chirp.range_sampling_rate;
            const auto& poly = sar_meta.results.doppler_centroid_poly;
            auto& dc = header_files.dop_centroid_coeffs;
            dc.zero_doppler_time = PtimeToMjd(sar_meta.first_line_time);
            dc.slant_range_time = asar_meta.two_way_slant_range_time * 1e9;
            dc.dop_coef[0] = poly.at(2);
            dc.dop_coef[1] = poly.at(1) / dt;
            dc.dop_coef[2] = poly.at(0) / (dt * dt);
            dc.BSwap();
        }

        if (is_slc) {
            sph.dsds[4].SetEmptyDSD("SR GR ADS", 'A');
        } else {
            constexpr size_t size = sizeof(header_files.srgr_params);
            static_assert(size == 55);
            sph.dsds[4].SetInternalDSD("SR GR ADS", 'A', file_offset, 1, size);
            file_offset += size;

            auto& srgr = header_files.srgr_params;
            srgr.zero_doppler_time = PtimeToMjd(sar_meta.first_line_time);
            srgr.slant_range_time = asar_meta.two_way_slant_range_time * 1e9;
            srgr.ground_range_origin = 0.0;
            // the GR to SR polynomial in Envisat spec is real value(aka 800KM+), the sar_meta version is from 0
            srgr.srgr_coeffs[0] = sar_meta.srgr.grsr_poly.at(4) + sar_meta.slant_range_first_sample;
            srgr.srgr_coeffs[1] = sar_meta.srgr.grsr_poly.at(3);
            srgr.srgr_coeffs[2] = sar_meta.srgr.grsr_poly.at(2);
            srgr.srgr_coeffs[3] = sar_meta.srgr.grsr_poly.at(1);
            srgr.srgr_coeffs[4] = sar_meta.srgr.grsr_poly.at(0);
            srgr.BSwap();
        }

        {
            sph.dsds[5].SetEmptyDSD("CHIRP PARAMS ADS", 'A');

            /* Removed file, because it is all zeroes in the current version anyway, TODO implementation
            constexpr size_t size = sizeof(header_files.chirp_params);
            static_assert(size == 1483);
            sph.dsds[5].SetInternalDSD("CHIRP PARAMS ADS", 'A', file_offset, 1, size);
            file_offset += size;
             */
        }

        sph.dsds[6].SetEmptyDSD("MDS1 ANTENNA ELEV PATT ADS", 'A');
        sph.dsds[7].SetEmptyDSD("MDS2 ANTENNA ELEV PATT ADS", 'A');

        {
            constexpr size_t size = sizeof(header_files.geolocation_grid[0]);
            size_t n = std::size(header_files.geolocation_grid);

            static_assert(size == 521);

            int step = sar_meta.img.azimuth_size / (n - 1);
            for (size_t i = 0; i < n; i++) {
                int az_idx = i * sar_meta.img.azimuth_size / (n - 1);
                FillGeoLocationAds(az_idx, az_idx + step - 1, sar_meta, header_files.geolocation_grid[i]);
                header_files.geolocation_grid[i].BSwap();
            }
            sph.dsds[8].SetInternalDSD("GEOLOCATION GRID ADS", 'A', file_offset, n, size);

            file_offset += (n * size);
        }

        sph.dsds[9].SetEmptyDSD("MAP PROJECTION GADS", 'G');
        {
            sph.dsds[10].SetInternalDSD("MDS1", 'M', file_offset, mds.n_records, mds.record_size);
            // MDS1
        }

        sph.dsds[11].SetEmptyDSD("MDS2", 'M');
        sph.dsds[12].SetReferenceDSD("LEVEL 0 PRODUCT", asar_meta.product_name);
        sph.dsds[13].SetReferenceDSD("ASAR PROCESSOR CONFIG", asar_meta.configuration_file);
        sph.dsds[14].SetReferenceDSD("INSTRUMENT CHARACTERIZATION", asar_meta.instrument_file);
        if (asar_meta.external_characterization_file.empty()) {
            sph.dsds[15].SetEmptyDSD("EXTERNAL CHARACTERIZATION", 'R');
        } else {
            sph.dsds[15].SetReferenceDSD("EXTERNAL_CHARACTERIZATION", asar_meta.external_characterization_file);
        }
        sph.dsds[16].SetReferenceDSD("EXTERNAL CALIBRATION", asar_meta.external_calibration_file);
        sph.dsds[17].SetReferenceDSD("ORBIT STATE VECTOR 1", asar_meta.orbit_dataset_name);
    }

    std::vector<uint8_t> out_buffer;
    append_packed_struct(out_buffer, header_files.mph);
    append_packed_struct(out_buffer, header_files.sph);
    append_packed_struct(out_buffer, header_files.summary_quality);
    append_packed_struct(out_buffer, header_files.main_processing_params);
    append_packed_struct(out_buffer, header_files.dop_centroid_coeffs);
    if (is_detected) {
        append_packed_struct(out_buffer, header_files.srgr_params);
    }
    // TODO chirp params...
    // append_packed_struct(out_buffer, header_files.chirp_params);
    append_packed_struct(out_buffer, header_files.geolocation_grid);

    CHECK_BOOL(out_buffer.size() == file_offset);
    return out_buffer;
}