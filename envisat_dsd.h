#pragma once

#include <cstdint>
#include "bswap_util.h"

#include "envisat_utils.h"

using sc = int8_t;
using uc = uint8_t;
using ss = int16_t;
using us = uint16_t;
using sl = int32_t;
using ul = uint32_t;
using sd = int64_t;
using ud = uint64_t;
using fl = float;
using do_ = double; // do is already a keyword...


struct mjd {
    sl days;
    ul seconds;
    ul micros;
};

inline mjd bswap(mjd in) {
    in.days = bswap(in.days);
    in.seconds = bswap(in.seconds);
    in.micros = bswap(in.micros);
    return in;
}

static_assert(sizeof(mjd) == 12);


struct __attribute__((packed)) GeneralSummary {
    mjd first_zero_doppler_time;
    uc attach_flag;
    mjd last_zero_doppler_time;
    uc work_order_id[12];
    fl time_diff;
    uc swath_num[3];
    fl range_spacing;
    fl azimuth_spacing;
    fl line_time_interval;
    ul num_output_lines;
    ul num_samples_per_line;
    uc data_type[5];
    ul num_range_lines_per_burst;
    fl time_diff_zero_doppler;
    uc spare_1[43];

    void BSwap() {
        first_zero_doppler_time = bswap(first_zero_doppler_time);
        //bswap(attach_flag);
        last_zero_doppler_time = bswap(last_zero_doppler_time);
        //bswap(work_order_id[12]);
        time_diff = bswap(time_diff);
        //bswap(swath_num[3]);
        range_spacing = bswap(range_spacing);
        azimuth_spacing = bswap(azimuth_spacing);
        line_time_interval = bswap(line_time_interval);
        num_output_lines = bswap(num_output_lines);
        num_samples_per_line = bswap(num_samples_per_line);
        //bswap(data_type[5]);
        num_range_lines_per_burst = bswap(num_range_lines_per_burst);
        time_diff_zero_doppler = bswap(time_diff_zero_doppler);
    }

    void SetDefaults() {
        memset(this, 0, sizeof(*this));
        first_zero_doppler_time = {1, 1, 1};
        attach_flag = 0;
        last_zero_doppler_time = {1, 15, 1};
        CopyStrPad(work_order_id, "123");
        CopyStrPad(swath_num, "IS2");
        CopyStrPad(data_type, "SWORD");


    }
};

static_assert(sizeof(GeneralSummary) == 120);

struct ImageProcessingSummary {
    uc data_analysis_flag;
    uc ant_elev_corr_flag;
    uc chirp_extract_flag;
    uc srgr_flag;
    uc dop_cen_flag;
    uc dop_amb_flag;
    uc range_spread_comp_flag;
    uc detected_flag;
    uc look_sum_flag;
    uc rms_equal_flag;
    uc ant_scal_flag;
    uc vga_com_echo_flag;
    uc vga_com_cal_flag;
    uc vga_com_nom_time_flag;
    uc gm_range_comp_inverse_filter_flag;
    uc spare_2[6];

    void BSwap() {
        // no op
    }
};

static_assert(sizeof(ImageProcessingSummary) == 21);

struct __attribute__((packed)) RawDataAnalysisInformation {
    ul num_gaps;
    ul num_missing_lines;
    ul range_samp_skip;
    ul range_lines_skip;
    fl calc_i_bias;
    fl calc_q_bias;
    fl calc_i_std_dev;
    fl calc_q_std_dev;
    fl calc_gain;
    fl calc_quad;
    fl i_bias_max;
    fl i_bias_min;
    fl q_bias_max;
    fl q_bias_min;
    fl gain_min;
    fl gain_max;
    fl quad_min;
    fl quad_max;
    uc i_bias_flag;
    uc q_bias_flag;
    uc gain_flag;
    uc quad_flag;
    fl used_i_bias;
    fl used_q_bias;
    fl used_gain;
    fl used_quad;

    void BSwap() {
        num_gaps = bswap(num_gaps);
        num_missing_lines = bswap(num_missing_lines);
        range_samp_skip = bswap(range_samp_skip);
        range_lines_skip = bswap(range_lines_skip);
        calc_i_bias = bswap(calc_i_bias);
        calc_q_bias = bswap(calc_q_bias);
        calc_i_std_dev = bswap(calc_i_std_dev);
        calc_q_std_dev = bswap(calc_q_std_dev);
        calc_gain = bswap(calc_gain);
        calc_quad = bswap(calc_quad);
        i_bias_max = bswap(i_bias_max);
        i_bias_min = bswap(i_bias_min);
        q_bias_max = bswap(q_bias_max);
        q_bias_min = bswap(q_bias_min);
        gain_min = bswap(gain_min);
        gain_max = bswap(gain_max);
        quad_min = bswap(quad_min);
        quad_max = bswap(quad_max);
        //i_bias_flag = bswap(uc i_bias_flag);
        //q_bias_flag = bswap(uc q_bias_flag);
        //iq_gain_significance = bswap(iq_gain_significance);
        //iq_quadrature_departure_significan = bswap(iq_quadrature_departure_significa);
        used_i_bias = bswap(used_i_bias);
        used_q_bias = bswap(used_q_bias);
        used_gain = bswap(used_gain);
        used_quad = bswap(used_quad);
    }
};

static_assert(sizeof(RawDataAnalysisInformation) == 92);

struct __attribute__((packed)) InformationFromDownlinkHeader {
    ul first_obt1[2];
    mjd first_mjd_1;
    ul first_obt_2[2];
    mjd first_mjd_2;
    us swst_code[5];
    us last_swst_code[5];
    us pri_code[5];
    us tx_pulse_len_code[5];
    us tx_bw_code[5];
    us echo_win_len_code[5];
    us up_code[5];
    us down_code[5];
    us resamp_code[5];
    us beam_adj_code[5];
    us beam_set_num_code[5];
    us tx_monitor_code[5];
    uc spare4[60];
    ul num_err_swst;
    ul num_err_pri;
    ul num_err_tx_pulse_len;
    ul num_err_pulse_bw;
    ul num_err_echo_win_len;
    ul num_err_up; // ers diff name
    ul num_err_down; // ers diff name
    ul num_err_resamp;
    ul num_err_beam_adj;
    ul num_err_beam_set_num;
    uc spare_5[26];
    fl swst_value[5];
    fl last_swst_value[5];
    ul swst_changes[5];
    fl prf_value[5];
    fl tx_pulse_len_value[5];
    fl tx_pulse_bw_value[5];
    fl echo_win_len_value[5];
    fl up_value[5];
    fl down_value[5];
    fl resamp_value[5];
    fl beam_adj_value[5];
    us beam_set_value[5];
    fl tx_monitor_value[5];
    ul rank[5]; // present in ASAR handbook, but not in ERS spec?
    uc spare_6[62];

    void BSwap() {
        BSWAP_ARR(first_obt1);
        first_mjd_1 = bswap(first_mjd_1);
        BSWAP_ARR(first_obt_2);
        first_mjd_2 = bswap(first_mjd_2);
        BSWAP_ARR(swst_code);
        BSWAP_ARR(last_swst_code);
        BSWAP_ARR(pri_code);
        BSWAP_ARR(tx_pulse_len_code);
        BSWAP_ARR(tx_bw_code);
        BSWAP_ARR(echo_win_len_code);
        BSWAP_ARR(up_code);
        BSWAP_ARR(down_code);
        BSWAP_ARR(resamp_code);
        BSWAP_ARR(beam_adj_code);
        BSWAP_ARR(beam_set_num_code);
        BSWAP_ARR(tx_monitor_code);

        num_err_swst = bswap(num_err_swst);
        num_err_pri = bswap(num_err_pri);
        num_err_tx_pulse_len = bswap(num_err_tx_pulse_len);
        num_err_pulse_bw = bswap(num_err_pulse_bw);
        num_err_echo_win_len = bswap(num_err_echo_win_len);
        num_err_up = bswap(num_err_up);
        num_err_down = bswap(num_err_down);
        num_err_resamp = bswap(num_err_resamp);
        num_err_beam_adj = bswap(num_err_beam_adj);
        num_err_beam_set_num = bswap(num_err_beam_set_num);

        BSWAP_ARR(swst_value);
        BSWAP_ARR(last_swst_value);
        BSWAP_ARR(swst_changes);
        BSWAP_ARR(prf_value);
        BSWAP_ARR(tx_pulse_len_value);
        BSWAP_ARR(tx_pulse_bw_value);
        BSWAP_ARR(echo_win_len_value);
        BSWAP_ARR(up_value);
        BSWAP_ARR(down_value);
        BSWAP_ARR(resamp_value);
        BSWAP_ARR(beam_adj_value);
        BSWAP_ARR(beam_set_value);
        BSWAP_ARR(tx_monitor_value);
        BSWAP_ARR(rank);
    }
};

static_assert(__builtin_offsetof(InformationFromDownlinkHeader, num_err_swst) == 220);
static_assert(__builtin_offsetof(InformationFromDownlinkHeader, swst_value) == 286);
static_assert(__builtin_offsetof(InformationFromDownlinkHeader, spare_5) == 617 - 357);
//static_assert(__builtin_offsetof(InformationFromDownlinkHeader, spare2) == 893-357);

static_assert(sizeof(InformationFromDownlinkHeader) == 618);


struct NominalChirp {
    float nom_chirp_amp[4];
    float nom_chirp_phs[4];
};

struct __attribute__((packed)) RangeProcessingInformation {
    ul first_proc_range_samp;
    fl range_ref;
    fl range_samp_rate;
    fl radar_freq;
    us num_looks_range;
    uc filter_range[7];
    fl filter_coef_range;
    fl look_bw_range[5];
    fl tot_bw_range[5];
    NominalChirp nominal_chirp[5];
    uc spare_7[60];

    void SetDefaults() {
        CopyStrPad(filter_range, "NONE");
    }

    void BSwap() {
        first_proc_range_samp = bswap(first_proc_range_samp);
        range_ref = bswap(range_ref);
        range_samp_rate = bswap(range_samp_rate);
        radar_freq = bswap(radar_freq);
        num_looks_range = bswap(num_looks_range);
        filter_coef_range = bswap(filter_coef_range);
        BSWAP_ARR(look_bw_range);
        BSWAP_ARR(tot_bw_range);
        for (size_t i = 0; i < 5; i++) {
            BSWAP_ARR(nominal_chirp[i].nom_chirp_amp);
            BSWAP_ARR(nominal_chirp[i].nom_chirp_phs);
        }
    }
};

static_assert(sizeof(RangeProcessingInformation) == 289);

struct __attribute__((packed)) AzimuthProcessingInformation {
    ul num_lines_proc;
    us num_look_az;
    fl look_bw_az;
    fl to_bw_az;
    uc filter_az[7];
    fl filter_coef_az;
    fl az_fm_rate[3];
    fl ax_fm_origin;
    fl dop_amb_conf;
    uc spare[68];

    void SetDefaults() {
        CopyStrPad(filter_az, "NONE");
    }

    void BSwap() {
        num_lines_proc = bswap(num_lines_proc);
        num_look_az = bswap(num_look_az);
        look_bw_az = bswap(look_bw_az);
        to_bw_az = bswap(to_bw_az);
        filter_coef_az = bswap(filter_coef_az);
        BSWAP_ARR(az_fm_rate);
        ax_fm_origin = bswap(ax_fm_origin);
        dop_amb_conf = bswap(dop_amb_conf);
    }
};

static_assert(sizeof(AzimuthProcessingInformation) == 113);

struct CalibrationInformation {
    fl processor_scaling_factor1;
    fl external_calibration_scaling_factor1;
    fl processor_scaling_factor2;
    fl external_calibration_scaling_factor2;
    fl noise_power_correction_factor[5];
    fl nr_lines_noise_correction_calc[5];
    uc spare_9[64];

    void BSwap() {
        processor_scaling_factor1 = bswap(processor_scaling_factor1);
        external_calibration_scaling_factor1 = bswap(external_calibration_scaling_factor1);
        processor_scaling_factor2 = bswap(processor_scaling_factor2);
        external_calibration_scaling_factor2 = bswap(external_calibration_scaling_factor2);
        BSWAP_ARR(noise_power_correction_factor);
        BSWAP_ARR(nr_lines_noise_correction_calc);
    }
};

static_assert(sizeof(CalibrationInformation) == 120);

struct OtherProcessingInformation {
    uc spare_10[12];
    fl output_mean1;
    fl output_imag_mean1;
    fl output_stddev1;
    fl output_imag_std_dev1;
    fl output_mean2;
    fl output_imag_mean2;
    fl output_stddev2;
    fl output_imag_std_dev2;
    fl average_height_above_ellipsoid_used;
    uc spare_11[48];

    void BSwap() {
        output_mean1 = bswap(output_mean1);
        output_imag_mean1 = bswap(output_imag_mean1);
        output_stddev1 = bswap(output_stddev1);
        output_imag_std_dev1 = bswap(output_imag_std_dev1);
        output_mean2 = bswap(output_mean2);
        output_imag_mean2 = bswap(output_imag_mean2);
        output_stddev2 = bswap(output_stddev2);
        output_imag_std_dev2 = bswap(output_imag_std_dev2);
        average_height_above_ellipsoid_used = bswap(average_height_above_ellipsoid_used);
    }
};

static_assert(sizeof(OtherProcessingInformation) == 96);

struct DataCompressionInformation {
    uc echo_comp[4];
    uc echo_comp_ratio[3];
    uc init_cal_comp[4];
    uc init_cal_ratio[3];
    uc per_cal_comp[4];
    uc per_cal_ratio[3];
    uc noise_comp[4];
    uc noise_comp_ratio[3];
    uc spare_12[64];


    void SetDefaults() {
        CopyStrPad(echo_comp, "NONE");
        CopyStrPad(init_cal_comp, "NONE");
        CopyStrPad(per_cal_comp, "NONE");
        CopyStrPad(noise_comp, "NONE");
    }

    void BSwap() {}
};

static_assert(sizeof(DataCompressionInformation) == 92);


struct ScanSARSpecificInformation {
    uc RFU[16 + 16 + 20 + 12];
    uc spare_13[16];


    void BSwap() {}
};

static_assert(sizeof(ScanSARSpecificInformation) == 80);

struct Osv {
    mjd state_vect_time;
    sl x_pos;
    sl y_pos;
    sl z_pos;
    sl x_vel;
    sl y_vel;
    sl z_vel;

    void BSwap() {
        state_vect_time = bswap(state_vect_time);
        x_pos = bswap(x_pos);
        y_pos = bswap(y_pos);
        z_pos = bswap(z_pos);
        x_vel = bswap(x_vel);
        y_vel = bswap(y_vel);
        z_vel = bswap(z_vel);
    }
};

static_assert(sizeof(Osv) == 36);


struct __attribute__((packed)) MainProcessingParametersADS {
    GeneralSummary general_summary;
    ImageProcessingSummary image_processing_summary;
    RawDataAnalysisInformation raw_data_analysis[2];
    uc spare3[32];
    InformationFromDownlinkHeader downlink_header;
    RangeProcessingInformation range_processing_information;
    AzimuthProcessingInformation azimuth_processing_information;
    CalibrationInformation calibration_information;
    OtherProcessingInformation other_processing_information;
    DataCompressionInformation data_compression_information;
    ScanSARSpecificInformation scansar_rfu;
    Osv orbit_state_vectors[5];
    uc spare[64];

    void BSwap() {
        general_summary.BSwap();
        image_processing_summary.BSwap();
        raw_data_analysis[0].BSwap();
        raw_data_analysis[1].BSwap();
        downlink_header.BSwap();
        range_processing_information.BSwap();
        azimuth_processing_information.BSwap();
        calibration_information.BSwap();
        other_processing_information.BSwap();
        data_compression_information.BSwap();
        scansar_rfu.BSwap();
        for (size_t i = 0; i < 5; i++) {
            orbit_state_vectors[i].BSwap();
        }
    }


    void SetDefaults() {
        memset(this, 0, sizeof(*this));
        general_summary.SetDefaults();
        range_processing_information.SetDefaults();
        azimuth_processing_information.SetDefaults();
        data_compression_information.SetDefaults();
    }
};

static_assert(__builtin_offsetof(MainProcessingParametersADS, raw_data_analysis) == 141);
static_assert(__builtin_offsetof(MainProcessingParametersADS, downlink_header) == 357);
static_assert(__builtin_offsetof(InformationFromDownlinkHeader, prf_value) == 346);
static_assert(__builtin_offsetof(InformationFromDownlinkHeader, down_value) == 446);
static_assert(__builtin_offsetof(InformationFromDownlinkHeader, spare_6) == 556);
static_assert(__builtin_offsetof(MainProcessingParametersADS, range_processing_information) == 975);
static_assert(__builtin_offsetof(MainProcessingParametersADS, azimuth_processing_information) == 1264);
static_assert(__builtin_offsetof(MainProcessingParametersADS, calibration_information) == 1377);
static_assert(__builtin_offsetof(MainProcessingParametersADS, other_processing_information) == 1497);
static_assert(__builtin_offsetof(MainProcessingParametersADS, data_compression_information) == 1593);
static_assert(__builtin_offsetof(MainProcessingParametersADS, scansar_rfu) == 1685);
static_assert(__builtin_offsetof(MainProcessingParametersADS, orbit_state_vectors) == 1765);
static_assert(sizeof(MainProcessingParametersADS) == 2009);


struct __attribute__((packed)) SummaryQualityADS {
    mjd zero_doppler_time;
    uc attach_flag;
    uc input_mean_flag;
    uc input_std_dev_flag;
    uc input_gaps_flag;
    uc input_missing_lines_flag;
    uc dop_cen_flag;
    uc dop_amb_flag;
    uc output_mean_flag;
    uc output_std_dev_flag;
    uc chirp_flag;
    uc missing_data_sets_flag;
    uc invalid_downlink_flag;
    uc spare_1[7];
    fl thresh_chirp_broadening;
    fl thresh_chirp_sidelobe;
    fl threst_chirp_islr;
    fl thresh_input_mean;
    fl exp_input_mean;
    fl thresh_input_std_dev;
    fl exp_input_std_dev;
    fl thresh_dop_cen;
    fl threst_dop_amb;
    fl thresh_output_mean;
    fl exp_output_mean;
    fl thresh_output_std_dev;
    fl exp_output_std_dev;
    fl thresh_input_missing_lines;
    fl thresh_input_gaps;
    ul lines_per_gap;
    uc spare_2[15];
    fl input_mean[2];
    fl input_std_dev[2];
    fl num_gaps;
    fl num_missing_lines;
    fl output_mean[2];
    fl output_std_dev[2];
    ul tot_errors;
    uc spare_3[16];

    void SetDefaults() {
        memset(this, 0, sizeof(*this));
    }

    void BSwap() {
        zero_doppler_time = bswap(zero_doppler_time);
        thresh_chirp_broadening - bswap(thresh_chirp_broadening);
        thresh_chirp_sidelobe - bswap(thresh_chirp_sidelobe);
        threst_chirp_islr - bswap(threst_chirp_islr);
        thresh_input_mean - bswap(thresh_input_mean);
        exp_input_mean - bswap(exp_input_mean);
        thresh_input_std_dev - bswap(thresh_input_std_dev);
        exp_input_std_dev - bswap(exp_input_std_dev);
        thresh_dop_cen - bswap(thresh_dop_cen);
        threst_dop_amb - bswap(threst_dop_amb);
        thresh_output_mean - bswap(thresh_output_mean);
        exp_output_mean - bswap(exp_output_mean);
        thresh_output_std_dev - bswap(thresh_output_std_dev);
        exp_output_std_dev - bswap(exp_output_std_dev);
        thresh_input_missing_lines - bswap(thresh_input_missing_lines);
        thresh_input_gaps - bswap(thresh_input_gaps);
        lines_per_gap - bswap(lines_per_gap);

        BSWAP_ARR(input_mean);
        BSWAP_ARR(input_std_dev);
        num_gaps = bswap(num_gaps);
        num_missing_lines = bswap(num_gaps);
        BSWAP_ARR(output_mean);
        BSWAP_ARR(output_std_dev);
        tot_errors = bswap(tot_errors);

    }
};

static_assert(__builtin_offsetof(SummaryQualityADS, spare_1) == 24);
static_assert(__builtin_offsetof(SummaryQualityADS, spare_2) == 95);
static_assert(sizeof(SummaryQualityADS) == 170);


struct __attribute__((packed)) DopplerCentroidParameters {
    mjd zero_doppler_time;
    uc attach_flag;
    fl slant_range_time;
    fl dop_coef[5];
    fl dop_conf;
    uc dop_conf_below_thresh_falg;
    us delta_dopp_coeff[5];
    uc spare_1[3];

    void SetDefaults() {
        memset(this, 0, sizeof(*this));
    }

    void BSwap() {
        zero_doppler_time = bswap(zero_doppler_time);
        slant_range_time = bswap(slant_range_time);
        BSWAP_ARR(dop_coef);
        dop_conf = bswap(dop_conf);
        BSWAP_ARR(delta_dopp_coeff);
    }
};


static_assert(sizeof(DopplerCentroidParameters) == 55);


struct CalPulseInfo {
    fl max_cal[3];
    fl avg_cal[3];
    fl avg_val_1a;
    fl phs_cal[4];
};

static_assert(sizeof(CalPulseInfo) == 44);

struct __attribute__((packed)) ChirpParameters {
    mjd zero_doppler_time;
    uc attach_flag;
    uc swath[3];
    uc polar[3];
    fl chirp_width;
    fl chirp_sidelobe;
    fl chirp_islr;
    fl chirp_peak_loc;
    fl re_chirp_power;
    fl elev_chirp_power;
    uc chirp_quality_flag;
    fl ref_chirp_power;
    uc normialization_source[7];
    uc spare_1[4];
    CalPulseInfo cal_pulse_info[32];
    uc spare_2[16];

    void SetDefaults()
    {
        memset(this, 0, sizeof(*this));
        CopyStrPad(swath, "V/V");
    }

    void BSwap() {
        zero_doppler_time = bswap(zero_doppler_time);
        chirp_width = bswap(chirp_width);
        chirp_sidelobe = bswap(chirp_sidelobe);
        chirp_islr = bswap(chirp_islr);
        chirp_peak_loc = bswap(chirp_peak_loc);
        re_chirp_power = bswap(re_chirp_power);
        elev_chirp_power = bswap(elev_chirp_power);
        ref_chirp_power = bswap(ref_chirp_power);

        for (size_t i = 0; i < 32; i++) {
            BSWAP_ARR(cal_pulse_info[i].max_cal);
            BSWAP_ARR(cal_pulse_info[i].avg_cal);
            BSWAP_ARR(cal_pulse_info[i].max_cal)
            cal_pulse_info[i].avg_val_1a = bswap(cal_pulse_info[i].avg_val_1a);
            BSWAP_ARR(cal_pulse_info[i].phs_cal);
        }
    }
};

static_assert(__builtin_offsetof(ChirpParameters, spare_1) == 55);
static_assert(sizeof(ChirpParameters) == 1483);


struct TiePoints {
    ul samp_numbers[11];
    fl slange_range_times[11];
    fl angles[11];
    fl lats[11];
    fl longs[11];
};

static_assert(sizeof(TiePoints) == 220);

struct __attribute__((packed)) GeoLocationADSR {
    mjd first_zero_doppler_time;
    uc attach_flag;
    ul line_num;
    ul num_lines;
    fl sub_sat_track;
    TiePoints first_line_tie_points;
    uc spare_1[22];
    mjd last_zero_doppler_time;
    TiePoints last_line_tie_points;
    uc spare_2[22];


    void SetDefaults()
    {
        memset(this, 0, sizeof(*this));
    }

    void BSwap() {
        first_zero_doppler_time = bswap(first_zero_doppler_time);

        line_num = bswap(line_num);
        num_lines = bswap(num_lines);
        sub_sat_track = bswap(sub_sat_track);


        BSWAP_ARR(first_line_tie_points.samp_numbers);
        BSWAP_ARR(first_line_tie_points.slange_range_times);
        BSWAP_ARR(first_line_tie_points.angles);
        BSWAP_ARR(first_line_tie_points.lats);
        BSWAP_ARR(first_line_tie_points.longs);

        last_zero_doppler_time = bswap(last_zero_doppler_time);

        BSWAP_ARR(last_line_tie_points.samp_numbers);
        BSWAP_ARR(last_line_tie_points.slange_range_times);
        BSWAP_ARR(last_line_tie_points.angles);
        BSWAP_ARR(last_line_tie_points.lats);
        BSWAP_ARR(last_line_tie_points.longs);
    }
};

static_assert(sizeof(GeoLocationADSR) == 521);