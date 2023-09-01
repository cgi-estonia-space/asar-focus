

#pragma once

#include <string>

#include <string.h>

#include "envisat_lvl0_parser.h"
#include "envisat_lvl1_ads.h"
#include "envisat_mph_sph_str_utils.h"
#include "util/checks.h"


/**
 *  Memory layout struct for Envisat LVL1 MPH, intended for file writing
 */

struct Lvl1MPH {
    uc product[73];
    uc proc_state[13];
    uc ref_doc[34];
    uc spare_1[41];
    uc acquistion_station[43];
    uc proc_center[21];
    uc proc_time[40];
    uc software_ver[30];
    uc spare_2[41];
    uc sensing_start[44];
    uc sensing_stop[43];
    uc spare_3[41];
    uc phase[8];
    uc cycle[11];
    uc rel_orbit[17];
    uc abs_orbit[17];
    uc state_vector_time[48];
    uc delta_ut1[22];
    uc x_position[27];
    uc y_position[27];
    uc z_position[27];
    uc x_velocity[29];
    uc y_velocity[29];
    uc z_velocity[29];
    uc vector_source[19];
    uc spare_4[41];
    uc utc_sbt_time[43];
    uc sat_binary_time[28];
    uc clock_step[27];
    uc spare_5[33];
    uc leap_utc[39];
    uc leap_sign[15];
    uc leap_err[11];
    uc spare_6[41];
    uc product_err[14];
    uc tot_size[38];
    uc sph_size[28];
    uc num_dsd[20];
    uc dsd_size[28];
    uc num_data_sets[26];
    uc spare_7[41];

    void SetDefaults() {
        memset(this, ' ', sizeof(*this));
        LastNewline(spare_1);
        LastNewline(spare_2);
        LastNewline(spare_3);
        LastNewline(spare_4);
        LastNewline(spare_5);
        LastNewline(spare_6);
        LastNewline(spare_7);
    }

    void Set_PRODUCT(std::string val) { SetStr(product, "PRODUCT", val); }

    void Set_PROC_STAGE(char val) { SetChar(proc_state, "PROC_STAGE", val); }

    void Set_REF_DOC(std::string val) { SetStr(ref_doc, "REF_DOC", val); }

    void SetDataAcqusitionProcessingInfo(std::string acq_station, std::string proc_center, std::string proc_time,
                                         std::string software_version) {
        SetStr(acquistion_station, "ACQUISITION_STATION", acq_station);
        SetStr(this->proc_center, "PROC_CENTER", proc_center);
        SetStr(this->proc_time, "PROC_TIME", proc_time);
        SetStr(software_ver, "SOFTWARE_VER", software_version);
    }

    void SetSensingStartStop(std::string start, std::string stop) {
        SetStr(sensing_start, "SENSING_START", start);
        SetStr(sensing_stop, "SENSING_STOP", stop);
    }

    void SetOrbitInfo(const ASARMetadata& md) {
        SetChar(phase, "PHASE", md.orbit_metadata.phase);
        SetAc(cycle, "CYCLE", md.orbit_metadata.cycle);
        SetAs(rel_orbit, "REL_ORBIT", md.orbit_metadata.rel_orbit);
        SetAs(abs_orbit, "ABS_ORBIT", md.orbit_metadata.abs_orbit);
        SetStr(state_vector_time, "STATE_VECTOR_TIME", PtimeToStr(md.orbit_metadata.state_vector_time));

        strcpy((char*)&delta_ut1[0], "DELTA_UT1=+.281903<s>");  // TODO
        delta_ut1[sizeof(delta_ut1) - 1] = '\n';
        SetAdo73(x_position, "X_POSITION", md.orbit_metadata.x_position, "<m>");
        SetAdo73(y_position, "Y_POSITION", md.orbit_metadata.y_position, "<m>");
        SetAdo73(z_position, "Z_POSITION", md.orbit_metadata.z_position, "<m>");
        SetAdo46(x_velocity, "X_VELOCITY", md.orbit_metadata.x_velocity, "<m/s>");
        SetAdo46(y_velocity, "Y_VELOCITY", md.orbit_metadata.y_velocity, "<m/s>");
        SetAdo46(z_velocity, "Z_VELOCITY", md.orbit_metadata.z_velocity, "<m/s>");
        SetStr(vector_source, "VECTOR_SOURCE", md.orbit_metadata.vector_source);
    }

    void Set_SBT_Defaults() {
        SetStr(utc_sbt_time, "UTC_SBT_TIME", "");
        SetAl(sat_binary_time, "SAT_BINARY_TIME", 0);
        SetAl(clock_step, "CLOCK_STEP", 0, "<ps>");
    }

    void Set_LEAP_Defaults() {
        SetStr(leap_utc, "LEAP_UTC", "");
        SetAc(leap_sign, "LEAP_SIGN", 0);
        SetChar(leap_err, "LEAP_ERR", '0');
    }

    void Set_PRODUCT_ERR(char val) { SetChar(product_err, "PRODUCT_ERR", val); }

    void SetProductSizeInformation(size_t total_size, size_t sph_sz, size_t n_dsd, size_t n_data_sets) {
        SetAd(tot_size, "TOT_SIZE", total_size, "<bytes>");
        SetAl(sph_size, "SPH_SIZE", sph_sz, "<bytes>");
        SetAl(num_dsd, "NUM_DSD", n_dsd);
        SetAl(dsd_size, "DSD_SIZE", 280, "<bytes>");
        SetAl(num_data_sets, "NUM_DATA_SETS", n_data_sets);
    }
};

static_assert(__builtin_offsetof(Lvl1MPH, acquistion_station) == 161);
static_assert(__builtin_offsetof(Lvl1MPH, software_ver) == 265);
static_assert(__builtin_offsetof(Lvl1MPH, phase) == 464);
static_assert(__builtin_offsetof(Lvl1MPH, utc_sbt_time) == 815);
static_assert(__builtin_offsetof(Lvl1MPH, leap_utc) == 946);
static_assert(sizeof(Lvl1MPH) == 1247);