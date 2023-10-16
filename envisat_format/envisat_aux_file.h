/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */
#pragma once

#include <filesystem>

#include "boost/date_time/posix_time/posix_time.hpp"

#include "bswap_util.h"

/**
 * Parser for ASA_CON and ASA_INS files, auxiliary files used for metadata extractions
 */

struct NominalChirpInstrument {
    fl amplitude[4];
    fl phase[4];
    fl duration;
};

inline NominalChirpInstrument bswap(NominalChirpInstrument in) {
    BSWAP_ARR(in.amplitude);
    BSWAP_ARR(in.phase);
    in.duration = bswap(in.duration);
    return in;
}

struct FixedLifetimeParameters {
    mjd time_of_creation;
    ul dsr_length;
    fl radar_frequency;
    fl radar_sampling_rate;
    fl offset_freq_wave_cal;
    fl amp_phase_cal_pulse[4 * 128];
    fl amp_phase_cal_pulse_im[12 * 896];
    fl amp_phase_cal_pulse_ws[8 * 640];

    NominalChirpInstrument im_chirp[7];
    NominalChirpInstrument ap_chirp[7];
    NominalChirpInstrument wv_chirp[7];
    NominalChirpInstrument ws_chirp[5];
    NominalChirpInstrument gm_chirp[5];
    fl antenn_az_pattern_is1[101];
    fl antenn_az_pattern_is2[101];
    fl antenn_az_pattern_is3[101];
    fl antenn_az_pattern_is4[101];
    fl antenn_az_pattern_is5[101];
    fl antenn_az_pattern_is6[101];
    fl antenn_az_pattern_is7[101];
    fl two_way_az_pattern_ss1[101];
    fl range_gate_bias;
    fl range_gate_bias_gm;
    fl LUT_I_adc[255];
    fl LUT_Q_adc[255];
    uc spare[648];

    void BSwap() {
        time_of_creation = bswap(time_of_creation);
        dsr_length = bswap(dsr_length);
        radar_frequency = bswap(radar_frequency);
        radar_sampling_rate = bswap(radar_sampling_rate);
        offset_freq_wave_cal = bswap(offset_freq_wave_cal);
        BSWAP_ARR(amp_phase_cal_pulse);
        BSWAP_ARR(amp_phase_cal_pulse_im);
        BSWAP_ARR(amp_phase_cal_pulse_ws);

        BSWAP_ARR(im_chirp);
        BSWAP_ARR(ap_chirp);
        BSWAP_ARR(wv_chirp);
        BSWAP_ARR(ws_chirp);
        BSWAP_ARR(gm_chirp);
        BSWAP_ARR(antenn_az_pattern_is1);
        BSWAP_ARR(antenn_az_pattern_is2);
        BSWAP_ARR(antenn_az_pattern_is3);
        BSWAP_ARR(antenn_az_pattern_is4);
        BSWAP_ARR(antenn_az_pattern_is5);
        BSWAP_ARR(antenn_az_pattern_is6);
        BSWAP_ARR(antenn_az_pattern_is7);
        BSWAP_ARR(two_way_az_pattern_ss1);
        range_gate_bias = bswap(range_gate_bias);
        range_gate_bias_gm = bswap(range_gate_bias_gm);
        BSWAP_ARR(LUT_I_adc);
        BSWAP_ARR(LUT_Q_adc);
    }
};

static_assert(sizeof(FixedLifetimeParameters) == 72608);

struct SwathConfTable {
    us nr_sample_windows_for_echo[7];
    us nr_sample_windows_for_init_cal[7];
    us nr_sample_windows_for_periodic_cal[7];
    us nr_sample_windows_for_noise[7];
    fl resampling_factor[7];

    void BSwap() {
        BSWAP_ARR(nr_sample_windows_for_echo);
        BSWAP_ARR(nr_sample_windows_for_init_cal);
        BSWAP_ARR(nr_sample_windows_for_periodic_cal);
        BSWAP_ARR(nr_sample_windows_for_noise);
        BSWAP_ARR(resampling_factor);
    }
};

static_assert(sizeof(SwathConfTable) == 84);

struct ModeTimelines {
    us swath_nums[7];
    us m_values[7];
    us r_values[7];
    us g_values[7];
};

inline ModeTimelines bswap(ModeTimelines in) {
    BSWAP_ARR(in.swath_nums);
    BSWAP_ARR(in.m_values);
    BSWAP_ARR(in.r_values);
    BSWAP_ARR(in.g_values);
    return in;
}

static_assert(sizeof(ModeTimelines) == 56);

struct __attribute__((packed)) FixedBaselineParameters {
    fl i_LUT_8bit[256];
    fl q_LUT_8bit[256];
    fl i_LUT_fbaq4[4096];
    fl i_LUT_fbaq3[2048];
    fl i_LUT_fbaq2[1024];
    fl q_LUT_fbaq4[4096];
    fl q_LUT_fbaq3[2048];
    fl q_LUT_fbaq2[1024];
    fl no_adc_fbaq4[4096];
    fl no_adc_fbaq3[2048];
    fl no_adc_fbaq2[1024];
    fl sign_mag_i_LUT[16];
    fl sign_mag_q_LUT[16];
    uc def_conf[160];
    SwathConfTable swath_conf_table[5];
    us nr_periodic_cal_windows_ec;
    us nr_periodic_cal_windows_ms;
    us def_swath_id_table[70];
    us beam_set_num_wm;
    us beam_set_num_ec;
    us beam_set_num_ms;
    us cal_row_sequence_table[32];
    ModeTimelines mode_timelines[5];
    us m_value;
    uc spare[44];
    fl ref_elev_angle[8];
    uc spare_2[64];

    void BSwap() {
        BSWAP_ARR(i_LUT_8bit);
        BSWAP_ARR(q_LUT_8bit);
        BSWAP_ARR(i_LUT_fbaq4);
        BSWAP_ARR(i_LUT_fbaq3);
        BSWAP_ARR(i_LUT_fbaq2);
        BSWAP_ARR(q_LUT_fbaq4);
        BSWAP_ARR(q_LUT_fbaq3);
        BSWAP_ARR(q_LUT_fbaq2);
        BSWAP_ARR(no_adc_fbaq4);
        BSWAP_ARR(no_adc_fbaq3);
        BSWAP_ARR(no_adc_fbaq2);
        BSWAP_ARR(sign_mag_i_LUT);
        BSWAP_ARR(sign_mag_q_LUT);
        // BSWAP_ARR(def_conf);
        swath_conf_table->BSwap();
        nr_periodic_cal_windows_ec = bswap(nr_periodic_cal_windows_ec);
        nr_periodic_cal_windows_ms = bswap(nr_periodic_cal_windows_ms);
        BSWAP_ARR(def_swath_id_table);
        beam_set_num_wm = bswap(beam_set_num_wm);
        beam_set_num_ec = bswap(beam_set_num_ec);
        beam_set_num_ms = bswap(beam_set_num_ms);
        BSWAP_ARR(cal_row_sequence_table);
        BSWAP_ARR(mode_timelines);
        m_value = bswap(m_value);

        BSWAP_ARR(ref_elev_angle);
    }
};

static_assert(sizeof(FixedBaselineParameters) == 89408);

struct __attribute__((__packed__, aligned(4))) AdditionalParams {
    fl ground_cmplx_amp[8 * 128];
    uc spare[5120];

    fl fixed_swst_offset_calp2;

    fl im_temp;
    do_ droop_im[8];
    fl ap_temp;
    do_ droop_ap[8];
    fl ws_temp;
    do_ droop_ws[8];
    fl gm_temp;
    do_ droop_gm[8];
    fl wv_temp;
    do_ droop_wv[8];

    // char tmp[68*5];
    uc spare2[72];
};

static_assert(sizeof(AdditionalParams) == 9632);

struct InstrumentFile {
    FixedLifetimeParameters flp;
    FixedBaselineParameters fbp;
    AdditionalParams ap;

    void BSwap() {
        flp.BSwap();
        fbp.BSwap();
        // ap.BSwap();
    }
};

static_assert(__builtin_offsetof(InstrumentFile, ap) == 162016);
static_assert(sizeof(InstrumentFile) == 171648);

struct Norm {
    uc perform_norm;
    uc norm_type[7];
};

static_assert(sizeof(Norm) == 8);

struct __attribute__((packed)) FilterWindow {
    uc window_type[7];
    fl window_coeff;
};

static_assert(sizeof(FilterWindow) == 11);

struct __attribute__((packed)) ConfigurationFile {
    mjd dsr_time;
    ul dsr_length;
    fl thresh_chirp_broadening;
    fl thresh_chirp_sidelobe;
    fl thresh_chirp_islr;
    fl thresh_input_mean;
    fl thresh_input_std_dev;
    fl thresh_dop_cen;
    fl thresh_dop_amb;
    fl thresh_output_mean;
    fl thresh_output_std_dev;
    fl thresh_missing_lines;
    fl thresh_gaps;
    uc spare_1[64];

    ul lines_per_gap;
    fl exp_im_mean;
    fl exp_im_std_dev;
    fl exp_ap_mean;
    fl exp_ap_std_dev;
    fl exp_imp_mean;
    fl exp_imp_std_dev;
    fl exp_app_mean;
    fl exp_app_std_dev;
    fl exp_imm_mean;
    fl exp_imm_std_dev;
    fl exp_apm_mean;
    fl exp_apm_std_dev;
    fl exp_wsm_mean;
    fl exp_wsm_std_dev;
    fl exp_gm1_mean;
    fl exp_gm1_std_dev;
    fl input_mean;
    fl input_std_dev;
    fl look_conf_tresh[2];
    fl inter_look_conf_thresh;
    fl az_cutoff_thresh;
    fl az_cutoff_iterations_thresh;
    fl phs_peak_thresh;
    fl phs_cross_thresh;
    uc spare_2[64];

    uc apply_gain_corr_flag;
    uc apply_gc_cal_p2_flag;
    uc apply_gc_delay_cal_p2_flag;
    uc spare_3[65];

    fl chirp_val_im[14];
    fl chirp_val_ap[28];
    fl chirp_val_wv[14];
    fl chirp_val_ws[10];
    fl chirp_val_gm[10];
    fl rep_eng_thresh_im;
    fl rep_eng_thresh_ap;
    fl rep_eng_thresh_wv;
    fl rep_eng_thresh_ws;
    fl rep_eng_thresh_gm;
    Norm norm_im;
    Norm norm_ap;
    Norm norm_wv;
    Norm norm_ws;
    Norm norm_gm;
    fl rep_re_cutoff_im;
    fl rep_re_cutoff_ap;
    fl rep_re_cutoff_wv;
    fl rep_re_cutoff_ws;
    fl rep_re_cutoff_gm;
    fl cal_pulse_apl_thres_im;
    fl cal_pulse_apl_thres_ap;
    fl cal_pulse_apl_thres_wv;
    fl cal_pulse_apl_thres_ws;
    fl cal_pulse_apl_thres_gm;
    uc spare_4[256];

    // 312
    fl process_gain_imp[7];
    fl process_gain_ims[7];
    fl process_gain_img[7];
    fl process_gain_imm[7];
    fl process_gain_app[7];
    fl process_gain_aps[7];
    fl process_gain_apg[7];
    fl process_gain_apm[7];
    fl process_gain_wv[7];
    fl process_gain_wss[5];
    fl process_gain_wsm[5];
    fl process_gain_gmone[5];

    // 284
    fl range_look_bandw_imp[7];
    fl range_look_bandw_ims[7];
    fl range_look_bandw_img[7];
    fl range_look_bandw_imm[7];
    fl range_look_bandw_app[7];
    fl range_look_bandw_aps[7];
    fl range_look_bandw_apg[7];
    fl range_look_bandw_apm[7];
    fl range_look_bandw_wss[5];
    fl range_look_bandw_wsm[5];
    fl range_look_bandw_gmone[5];

    // 312
    fl total_range_bandw_gain_imp[7];
    fl total_range_bandw_gain_ims[7];
    fl total_range_bandw_gain_img[7];
    fl total_range_bandw_gain_imm[7];
    fl total_range_bandw_gain_app[7];
    fl total_range_bandw_gain_aps[7];
    fl total_range_bandw_gain_apg[7];
    fl total_range_bandw_gain_apm[7];
    fl total_range_bandw_gain_wv[7];
    fl total_range_bandw_gain_wss[5];
    fl total_range_bandw_gain_wsm[5];
    fl total_range_bandw_gain_gmone[5];

    // 284
    ul num_range_looks_imp[7];
    ul num_range_looks_ims[7];
    ul num_range_looks_img[7];
    ul num_range_looks_imm[7];
    ul num_range_looks_app[7];
    ul num_range_looks_aps[7];
    ul num_range_looks_apg[7];
    ul num_range_looks_apm[7];
    ul num_range_looks_wss[5];
    ul num_range_looks_wsm[5];
    ul num_range_looks_gmone[5];

    fl azimuth_look_bandw_imp[7];
    fl azimuth_look_bandw_ims[7];
    fl azimuth_look_bandw_img[7];
    fl azimuth_look_bandw_aps[7];

    fl tot_azimuth_bandw_imp[7];
    fl tot_azimuth_bandw_ims[7];
    fl tot_azimuth_bandw_img[7];
    fl tot_azimuth_bandw_aps[7];
    fl tot_azimuth_bandw_wv[7];

    ul num_azimuth_looks_imp[7];
    ul num_azimuth_looks_ims[7];
    ul num_azimuth_looks_img[7];
    ul num_azimuth_looks_imm[7];
    ul num_azimuth_looks_app[7];
    ul num_azimuth_looks_aps[7];
    ul num_azimuth_looks_apg[7];
    ul num_azimuth_looks_apm[7];
    ul num_azimuth_looks_wss;
    ul num_azimuth_looks_wsm;
    ul num_azimuth_looks_gmone;

    uc descallop_flag_imm;
    uc descallop_flag_app;
    uc descallop_flag_apg;
    uc descallop_flag_apm;
    uc spare_5;

    uc descallop_flag_wsm;
    uc descallop_flag_gmone;
    uc const_snr_filter_flag_imm;
    uc const_snr_filter_flag_app;
    uc const_snr_filter_flag_apg;
    uc const_snr_filter_flag_apm;
    uc spare_6;

    uc const_snr_filter_flag_wsm;
    uc const_snr_filter_flag_gmone;

    FilterWindow az_win_type_imp;
    FilterWindow az_win_type_ims;
    FilterWindow az_win_type_img;
    FilterWindow az_win_type_imm;
    FilterWindow az_win_type_app;
    FilterWindow az_win_type_aps;
    FilterWindow az_win_type_apg;
    FilterWindow az_win_type_apm;
    FilterWindow az_win_type_wv;
    FilterWindow az_win_type_wss;
    FilterWindow az_win_type_wsm;
    FilterWindow az_win_type_gmone;

    FilterWindow range_win_type_imp;
    FilterWindow range_win_type_ims;
    FilterWindow range_win_type_img;
    FilterWindow range_win_type_imm;
    FilterWindow range_win_type_app;
    FilterWindow range_win_type_aps;
    FilterWindow range_win_type_apg;
    FilterWindow range_win_type_apm;
    FilterWindow range_win_type_wv;
    FilterWindow range_win_type_wss;
    FilterWindow range_win_type_wsm;
    FilterWindow range_win_type_gmone;

    fl thresh_terr_height_std_dev_imp;  // asar handbook 16 bit int, Vol08_ASAR_4C_20120120.pdf -> float
    fl thresh_terr_height_std_dev_ims;
    fl thresh_terr_height_std_dev_img;
    fl thresh_terr_height_std_dev_imm;
    fl thresh_terr_height_std_dev_app;
    fl thresh_terr_height_std_dev_aps;
    fl thresh_terr_height_std_dev_apg;
    fl thresh_terr_height_std_dev_apm;
    fl thresh_terr_height_std_dev_wss;
    fl thresh_terr_height_std_dev_wsm;
    fl thresh_terr_height_std_dev_gmone;
    uc noise_sub_flag_app;  // not present in asar handbook
    uc noise_sub_flag_apm;
    uc noise_sub_flag_apg;
    uc noise_sub_flag_wsm;
    fl azimuth_noise_factor_app[7];
    fl azimuth_noise_factor_apm[7];
    fl azimuth_noise_factor_apg[7];
    fl azimuth_noise_factor_wsm[5];
    uc spare_7[872];

    uc dop_cen_est_flag_imp;
    uc dop_cen_est_flag_ims;
    uc dop_cen_est_flag_img;
    uc dop_cen_est_flag_imm;
    uc dop_cen_est_flag_app;
    uc dop_cen_est_flag_aps;
    uc dop_cen_est_flag_apg;
    uc dop_cen_est_flag_apm;
    uc dop_cen_est_flag_wv;
    uc dop_cen_est_flag_wss;
    uc dop_cen_est_flag_wsm;
    uc dop_cen_est_flag_gmone;

    uc per_dop_amb_est_flag_imp;
    uc per_dop_amb_est_flag_ims;
    uc per_dop_amb_est_flag_img;
    uc per_dop_amb_est_flag_imm;
    uc per_dop_amb_est_flag_app;
    uc per_dop_amb_est_flag_aps;
    uc per_dop_amb_est_flag_apg;
    uc per_dop_amb_est_flag_apm;
    uc per_dop_amb_est_flag_wv;
    uc per_dop_amb_est_flag_wss;
    uc per_dop_amb_est_flag_wsm;
    uc per_dop_amb_est_flag_gmone;

    uc dop_refine_flag_app;
    uc dop_refine_flag_aps;
    uc dop_refine_flag_apg;
    uc dop_refine_flag_apm;
    uc dop_refine_flag_wss;
    uc dop_refine_flag_wsm;

    uc per_dop_grid_est_flag_imp;
    uc per_dop_grid_est_flag_ims;
    uc per_dop_grid_est_flag_imm;
    uc per_dop_grid_est_flag_app;
    uc per_dop_grid_est_flag_aps;
    uc per_dop_grid_est_flag_apm;
    uc per_dop_grid_est_flag_wss;
    uc per_dop_grid_est_flag_wsm;
    uc per_dop_grid_est_flag_gmone;

    uc spare_8[55];

    void BSwap() {
        dsr_time = bswap(dsr_time);
        dsr_length = bswap(dsr_length);
        thresh_chirp_broadening = bswap(thresh_chirp_broadening);
        thresh_chirp_sidelobe = bswap(thresh_chirp_sidelobe);
        thresh_chirp_islr = bswap(thresh_chirp_islr);
        thresh_input_mean = bswap(thresh_input_mean);
        thresh_input_std_dev = bswap(thresh_input_std_dev);
        thresh_dop_cen = bswap(thresh_dop_cen);
        thresh_dop_amb = bswap(thresh_dop_amb);
        thresh_output_mean = bswap(thresh_output_mean);
        thresh_output_std_dev = bswap(thresh_output_std_dev);
        thresh_missing_lines = bswap(thresh_missing_lines);
        thresh_gaps = bswap(thresh_gaps);

        lines_per_gap = bswap(lines_per_gap);
        exp_im_mean = bswap(exp_im_mean);
        exp_im_std_dev = bswap(exp_im_std_dev);
        exp_ap_mean = bswap(exp_ap_mean);
        exp_ap_std_dev = bswap(exp_ap_std_dev);
        exp_imp_mean = bswap(exp_imp_mean);
        exp_imp_std_dev = bswap(exp_imp_std_dev);
        exp_app_mean = bswap(exp_app_mean);
        exp_app_std_dev = bswap(exp_app_std_dev);
        exp_imm_mean = bswap(exp_imm_mean);
        exp_imm_std_dev = bswap(exp_imm_std_dev);
        exp_apm_mean = bswap(exp_apm_mean);
        exp_apm_std_dev = bswap(exp_apm_std_dev);
        exp_wsm_mean = bswap(exp_wsm_mean);
        exp_wsm_std_dev = bswap(exp_wsm_std_dev);
        exp_gm1_mean = bswap(exp_gm1_mean);
        exp_gm1_std_dev = bswap(exp_gm1_std_dev);
        input_mean = bswap(input_mean);
        input_std_dev = bswap(input_std_dev);
        BSWAP_ARR(look_conf_tresh);
        inter_look_conf_thresh = bswap(inter_look_conf_thresh);
        az_cutoff_thresh = bswap(az_cutoff_thresh);
        az_cutoff_iterations_thresh = bswap(az_cutoff_iterations_thresh);
        phs_peak_thresh = bswap(phs_peak_thresh);
        phs_cross_thresh = bswap(phs_cross_thresh);

        // Below fields are char arrays, not need to be swapped
        // uc spare_2[64];
        //        uc apply_gain_corr_flag;
        //        uc apply_gc_cal_p2_flag;
        //        uc apply_gc_delay_cal_p2_flag;
        //        uc spare_3[65];

        BSWAP_ARR(chirp_val_im);
        BSWAP_ARR(chirp_val_ap);
        BSWAP_ARR(chirp_val_wv);
        BSWAP_ARR(chirp_val_ws);
        BSWAP_ARR(chirp_val_gm);
        rep_eng_thresh_im = bswap(rep_eng_thresh_im);
        rep_eng_thresh_ap = bswap(rep_eng_thresh_ap);
        rep_eng_thresh_wv = bswap(rep_eng_thresh_wv);
        rep_eng_thresh_ws = bswap(rep_eng_thresh_ws);
        rep_eng_thresh_gm = bswap(rep_eng_thresh_gm);

        rep_re_cutoff_im = bswap(rep_re_cutoff_im);
        rep_re_cutoff_ap = bswap(rep_re_cutoff_ap);
        rep_re_cutoff_wv = bswap(rep_re_cutoff_wv);
        rep_re_cutoff_ws = bswap(rep_re_cutoff_ws);
        rep_re_cutoff_gm = bswap(rep_re_cutoff_gm);
        cal_pulse_apl_thres_im = bswap(cal_pulse_apl_thres_im);
        cal_pulse_apl_thres_ap = bswap(cal_pulse_apl_thres_ap);
        cal_pulse_apl_thres_wv = bswap(cal_pulse_apl_thres_wv);
        cal_pulse_apl_thres_ws = bswap(cal_pulse_apl_thres_ws);
        cal_pulse_apl_thres_gm = bswap(cal_pulse_apl_thres_gm);

        BSWAP_ARR(process_gain_imp);
        BSWAP_ARR(process_gain_ims);
        BSWAP_ARR(process_gain_img);
        BSWAP_ARR(process_gain_imm);
        BSWAP_ARR(process_gain_app);
        BSWAP_ARR(process_gain_aps);
        BSWAP_ARR(process_gain_apg);
        BSWAP_ARR(process_gain_apm);
        BSWAP_ARR(process_gain_wv);
        BSWAP_ARR(process_gain_wss);
        BSWAP_ARR(process_gain_wsm);
        BSWAP_ARR(process_gain_gmone);

        // 284
        BSWAP_ARR(range_look_bandw_imp);
        BSWAP_ARR(range_look_bandw_ims);
        BSWAP_ARR(range_look_bandw_img);
        BSWAP_ARR(range_look_bandw_imm);
        BSWAP_ARR(range_look_bandw_app);
        BSWAP_ARR(range_look_bandw_aps);
        BSWAP_ARR(range_look_bandw_apg);
        BSWAP_ARR(range_look_bandw_apm);
        BSWAP_ARR(range_look_bandw_wss);
        BSWAP_ARR(range_look_bandw_wsm);
        BSWAP_ARR(range_look_bandw_gmone);
        // 312
        BSWAP_ARR(total_range_bandw_gain_imp);
        BSWAP_ARR(total_range_bandw_gain_ims);
        BSWAP_ARR(total_range_bandw_gain_img);
        BSWAP_ARR(total_range_bandw_gain_imm);
        BSWAP_ARR(total_range_bandw_gain_app);
        BSWAP_ARR(total_range_bandw_gain_aps);
        BSWAP_ARR(total_range_bandw_gain_apg);
        BSWAP_ARR(total_range_bandw_gain_apm);
        BSWAP_ARR(total_range_bandw_gain_wv);
        BSWAP_ARR(total_range_bandw_gain_wss);
        BSWAP_ARR(total_range_bandw_gain_wsm);
        BSWAP_ARR(total_range_bandw_gain_gmone);

        // 284
        BSWAP_ARR(num_range_looks_imp);
        BSWAP_ARR(num_range_looks_ims);
        BSWAP_ARR(num_range_looks_img);
        BSWAP_ARR(num_range_looks_imm);
        BSWAP_ARR(num_range_looks_app);
        BSWAP_ARR(num_range_looks_aps);
        BSWAP_ARR(num_range_looks_apg);
        BSWAP_ARR(num_range_looks_apm);
        BSWAP_ARR(num_range_looks_wss);
        BSWAP_ARR(num_range_looks_wsm);
        BSWAP_ARR(num_range_looks_gmone);

        BSWAP_ARR(azimuth_look_bandw_imp);
        BSWAP_ARR(azimuth_look_bandw_ims);
        BSWAP_ARR(azimuth_look_bandw_img);
        BSWAP_ARR(azimuth_look_bandw_aps);

        BSWAP_ARR(tot_azimuth_bandw_imp);
        BSWAP_ARR(tot_azimuth_bandw_ims);
        BSWAP_ARR(tot_azimuth_bandw_img);
        BSWAP_ARR(tot_azimuth_bandw_aps);
        BSWAP_ARR(tot_azimuth_bandw_wv);

        BSWAP_ARR(num_azimuth_looks_imp);
        BSWAP_ARR(num_azimuth_looks_ims);
        BSWAP_ARR(num_azimuth_looks_img);
        BSWAP_ARR(num_azimuth_looks_imm);
        BSWAP_ARR(num_azimuth_looks_app);
        BSWAP_ARR(num_azimuth_looks_aps);
        BSWAP_ARR(num_azimuth_looks_apg);
        BSWAP_ARR(num_azimuth_looks_apm);
        num_azimuth_looks_wss = bswap(num_azimuth_looks_wss);
        num_azimuth_looks_wsm = bswap(num_azimuth_looks_wsm);
        num_azimuth_looks_gmone = bswap(num_azimuth_looks_gmone);

        az_win_type_imp.window_coeff = bswap(az_win_type_imp.window_coeff);
        az_win_type_ims.window_coeff = bswap(az_win_type_ims.window_coeff);
        az_win_type_img.window_coeff = bswap(az_win_type_img.window_coeff);
        az_win_type_imm.window_coeff = bswap(az_win_type_imm.window_coeff);
        az_win_type_app.window_coeff = bswap(az_win_type_app.window_coeff);
        az_win_type_aps.window_coeff = bswap(az_win_type_aps.window_coeff);
        az_win_type_apg.window_coeff = bswap(az_win_type_apg.window_coeff);
        az_win_type_apm.window_coeff = bswap(az_win_type_apm.window_coeff);
        az_win_type_wv.window_coeff = bswap(az_win_type_wv.window_coeff);
        az_win_type_wss.window_coeff = bswap(az_win_type_wss.window_coeff);
        az_win_type_wsm.window_coeff = bswap(az_win_type_wsm.window_coeff);
        az_win_type_gmone.window_coeff = bswap(az_win_type_gmone.window_coeff);

        range_win_type_imp.window_coeff = bswap(range_win_type_imp.window_coeff);
        range_win_type_ims.window_coeff = bswap(range_win_type_ims.window_coeff);
        range_win_type_img.window_coeff = bswap(range_win_type_img.window_coeff);
        range_win_type_imm.window_coeff = bswap(range_win_type_imm.window_coeff);
        range_win_type_app.window_coeff = bswap(range_win_type_app.window_coeff);
        range_win_type_aps.window_coeff = bswap(range_win_type_aps.window_coeff);
        range_win_type_apg.window_coeff = bswap(range_win_type_apg.window_coeff);
        range_win_type_apm.window_coeff = bswap(range_win_type_apm.window_coeff);
        range_win_type_wv.window_coeff = bswap(range_win_type_wv.window_coeff);
        range_win_type_wss.window_coeff = bswap(range_win_type_wss.window_coeff);
        range_win_type_wsm.window_coeff = bswap(range_win_type_wsm.window_coeff);
        range_win_type_gmone.window_coeff = bswap(range_win_type_gmone.window_coeff);

        thresh_terr_height_std_dev_imp = bswap(thresh_terr_height_std_dev_imp);
        thresh_terr_height_std_dev_ims = bswap(thresh_terr_height_std_dev_ims);
        thresh_terr_height_std_dev_img = bswap(thresh_terr_height_std_dev_img);
        thresh_terr_height_std_dev_imm = bswap(thresh_terr_height_std_dev_imm);
        thresh_terr_height_std_dev_app = bswap(thresh_terr_height_std_dev_app);
        thresh_terr_height_std_dev_aps = bswap(thresh_terr_height_std_dev_aps);
        thresh_terr_height_std_dev_apg = bswap(thresh_terr_height_std_dev_apg);
        thresh_terr_height_std_dev_apm = bswap(thresh_terr_height_std_dev_apm);
        thresh_terr_height_std_dev_wss = bswap(thresh_terr_height_std_dev_wss);
        thresh_terr_height_std_dev_wsm = bswap(thresh_terr_height_std_dev_wsm);
        thresh_terr_height_std_dev_gmone = bswap(thresh_terr_height_std_dev_gmone);

        BSWAP_ARR(azimuth_noise_factor_app);
        BSWAP_ARR(azimuth_noise_factor_apm);
        BSWAP_ARR(azimuth_noise_factor_apg);
        BSWAP_ARR(azimuth_noise_factor_wsm);
    }
};

static_assert(offsetof(ConfigurationFile, spare_1) == 60);
static_assert(offsetof(ConfigurationFile, spare_2) == 228);
static_assert(offsetof(ConfigurationFile, spare_3) == 295);
static_assert(offsetof(ConfigurationFile, spare_4) == 764);
static_assert(offsetof(ConfigurationFile, process_gain_imp) == 1020);
static_assert(offsetof(ConfigurationFile, range_look_bandw_imp) == 1332);
static_assert(offsetof(ConfigurationFile, num_range_looks_imp) == 1928);
static_assert(offsetof(ConfigurationFile, num_azimuth_looks_imp) == 2464);
static_assert(offsetof(ConfigurationFile, az_win_type_imp) == 2714);
static_assert(offsetof(ConfigurationFile, thresh_terr_height_std_dev_imp) == 2978);

static_assert(offsetof(ConfigurationFile, spare_7) == 3130);

static_assert(sizeof(ConfigurationFile) == 4096);

void FindINSFile(std::string aux_root, boost::posix_time::ptime start, InstrumentFile& ins_file, std::string& filename);
void FindCONFile(std::string aux_root, boost::posix_time::ptime start, ConfigurationFile& ins_file,
                 std::string& filename);
