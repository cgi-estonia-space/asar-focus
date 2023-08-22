#pragma once

#include <filesystem>
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "asar_lvl0_parser.h"

#include "bswap_util.h"


struct NominalChirpInstrument
{
    fl amplitude[4];
    fl phase[4];
    fl duration;
};

inline NominalChirpInstrument bswap(NominalChirpInstrument in)
{
    BSWAP_ARR(in.amplitude);
    BSWAP_ARR(in.phase);
    in.duration = bswap(in.duration);
    return in;
}

struct FixedLifetimeParameters
{
    mjd time_of_creation;
    ul dsr_length;
    fl radar_frequency;
    fl radar_sampling_rate;
    fl offset_freq_wave_cal;
    fl amp_phase_cal_pulse[4 * 128];
    fl amp_phase_cal_pulse_im[12 * 896];
    fl amp_phase_cal_pulse_ws[8* 640];

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


    void BSwap()
    {
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

struct SwathConfTable
{
    us nr_sample_windows_for_echo[7];
    us nr_sample_windows_for_init_cal[7];
    us nr_sample_windows_for_periodic_cal[7];
    us nr_sample_windows_for_noise[7];
    fl resampling_factor[7];

    void BSwap()
    {
        BSWAP_ARR(nr_sample_windows_for_echo);
        BSWAP_ARR(nr_sample_windows_for_init_cal);
        BSWAP_ARR(nr_sample_windows_for_periodic_cal);
        BSWAP_ARR(nr_sample_windows_for_noise);
        BSWAP_ARR(resampling_factor);
    }
};

static_assert(sizeof(SwathConfTable) == 84);

struct ModeTimelines
{
    us swath_nums[7];
    us m_values[7];
    us r_values[7];
    us g_values[7];

};

inline ModeTimelines bswap(ModeTimelines in)
{
    BSWAP_ARR(in.swath_nums);
    BSWAP_ARR(in.m_values);
    BSWAP_ARR(in.r_values);
    BSWAP_ARR(in.g_values);
    return in;
}

static_assert(sizeof(ModeTimelines) == 56);

struct __attribute__((packed)) FixedBaselineParameters
{
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

    void BSwap()
    {
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
        //BSWAP_ARR(def_conf);
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



struct __attribute__((__packed__, aligned(4))) AdditionalParams
{
    fl ground_cmplx_amp[8*128];
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

    //char tmp[68*5];
    uc spare2[72];

};

static_assert(sizeof(AdditionalParams) == 9632);

struct InstrumentFile
{
    FixedLifetimeParameters flp;
    FixedBaselineParameters fbp;
    AdditionalParams ap;


    void BSwap()
    {
        flp.BSwap();
        fbp.BSwap();
        //ap.BSwap();
    }
};

static_assert(__builtin_offsetof(InstrumentFile, ap) == 162016);
static_assert(sizeof(InstrumentFile) == 171648);





void FindInsFile(std::string path, boost::posix_time::ptime start, InstrumentFile& ins_file, std::string& filename);

