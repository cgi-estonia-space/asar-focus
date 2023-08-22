#pragma once

#include <string>


#include "envisat_ph.h"

#include "geo_tools.h"


struct DSD {
    uc ds_name[39];
    uc ds_type[10];
    uc filename[74];
    uc ds_offset[39];
    uc ds_size[37];
    uc num_dsr[20];
    uc dsr_size[28];
    uc spare[33];


    void SetDSName(std::string name) {
        SetStr(ds_name, "DS_NAME", name);
    }

    void SetType(char type) {
        SetChar(ds_type, "DS_TYPE", type);
    }

    void SetFileName(std::string name) {
        SetStr(filename, "FILENAME", name);
    }

    void SetDSOffset(size_t offset) {
        SetAd(ds_offset, "DS_OFFSET", offset, "<bytes>");
    }

    void SetDSSize(size_t size) {
        SetAd(ds_size, "DS_SIZE", size, "<bytes>");
    }

    void SetNumDSR(size_t num) {
        SetAl(num_dsr, "NUM_DSR", num);
    }

    void SetDSRSize(size_t size) {
        SetAl(dsr_size, "DSR_SIZE", size, "<bytes>");
    }

    void SetInternalDSD(std::string name, char type, size_t offset, size_t n, size_t size) {
        SetDSName(name);
        SetType(type);
        SetFileName("");
        SetDSOffset(offset);
        SetDSSize(n * size);
        SetNumDSR(n);
        SetDSRSize(size);
        ClearSpare(spare);
    }

    void SetEmptyDSD(std::string name, char type) {
        SetDSName(name);
        SetType(type);
        SetFileName("NOT USED");
        SetDSOffset(0);
        SetDSSize(0);
        SetNumDSR(0);
        SetDSSize(0);
        SetDSRSize(0);
        ClearSpare(spare);
    }

    void SetReferenceDSD(std::string name, std::string ext_name)
    {
        SetDSName(name);
        SetType('R');
        SetFileName(ext_name);
        SetDSOffset(0);
        SetDSSize(0);
        SetNumDSR(0);
        SetDSSize(0);
        SetDSRSize(0);
        ClearSpare(spare);
    }
};


static_assert(sizeof(DSD) == 280);


struct Lvl1SPH {
    //uc spare
    uc sph_descriptor[46];
    uc stripline_continuity_indicator[36];
    uc slice_position[20];
    uc num_slices[16];


    // product time information

    uc first_line_time[46];
    uc last_line_time[45];
    // product positioning information

    uc first_near_lat[15 + 22];
    uc first_near_long[16 + 22];
    uc first_mid_lat[14 + 22];
    uc first_mid_long[15 + 22];
    uc first_far_lat[14 + 22];
    uc first_far_long[15 + 22];

    uc last_near_lat[14 + 22];
    uc last_near_long[15 + 22];
    uc last_mid_lat[13 + 22];
    uc last_mid_long[14 + 22];
    uc last_far_lat[13 + 22];
    uc last_far_long[14 + 22];


    uc spare_1[36];
    uc swath[12];
    uc pass[18];
    uc sample_type[23];
    uc algorithm[20];
    uc mds1_tx_rx_polar[23];
    uc mds2_tx_rx_polar[23];
    uc compression[20];
    uc azimuth_looks[19];
    uc range_looks[17];
    uc range_spacing[33];
    uc azimuth_spacing[35];
    uc line_time_interval[38];
    uc line_length[28];
    uc data_type[18];
    uc spare_2[51];
    DSD dsds[18];


    void SetHeader(std::string descriptor, int continuity_indicator, int slice_pos, int n_slice) {
        SetStr(sph_descriptor, "SPH_DESCRIPTOR", descriptor);
        SetAc(stripline_continuity_indicator, "STRIPLINE_CONTINUITY_INDICATOR", continuity_indicator);
        SetAc(slice_position, "SLICE_POSITION", slice_pos);
        SetAc(num_slices, "NUM_SLICES", n_slice);
    }


    void SetLineTimeInfo(std::string first_line_arg, std::string last_line_arg)
    {
        SetStr(first_line_time, "FIRST_LINE_TIME", first_line_arg);
        SetStr(last_line_time, "LAST_LINE_TIME", last_line_arg);
    }


    void SetFirstPosition(GeoPosLLH near, GeoPosLLH mid, GeoPosLLH far)
    {
        auto conv = [](double val){ return static_cast<int32_t>(1e6*val); };
        SetAl(first_near_lat, "FIRST_NEAR_LAT", conv(near.latitude), "<10-6degN>");
        SetAl(first_near_long, "FIRST_NEAR_LONG", conv(near.longitude), "<10-6degE>");
        SetAl(first_mid_lat, "FIRST_MID_LAT", conv(mid.latitude), "<10-6degN>");
        SetAl(first_mid_long, "FIRST_MID_LONG", conv(mid.longitude), "<10-6degE>");
        SetAl(first_far_lat, "FIRST_FAR_LAT", conv(far.latitude), "<10-6degN>");
        SetAl(first_far_long, "FIRST_FAR_LONG", conv(far.longitude), "<10-6degE>");
    }

    void SetLastPosition(GeoPosLLH near, GeoPosLLH mid, GeoPosLLH far)
    {
        auto conv = [](double val){ return static_cast<int32_t>(1e6*val); };
        SetAl(last_near_lat, "LAST_NEAR_LAT", conv(near.latitude), "<10-6degN>");
        SetAl(last_near_long, "LAST_NEAR_LONG", conv(near.longitude), "<10-6degE>");
        SetAl(last_mid_lat, "LAST_MID_LAT", conv(mid.latitude), "<10-6degN>");
        SetAl(last_mid_long, "LAST_MID_LONG", conv(mid.longitude), "<10-6degE>");
        SetAl(last_far_lat, "LAST_FAR_LAT", conv(far.latitude), "<10-6degN>");
        SetAl(last_far_long, "LAST_FAR_LONG", conv(far.longitude), "<10-6degE>");
    }


    void SetProductInfo1(std::string swath, std::string pass, std::string sample_type, std::string algorithm_arg)
    {
        SetStr(this->swath, "SWATH", swath);
        SetStr(this->pass, "PASS", pass);
        SetStr(this->sample_type, "SAMPLE_TYPE", sample_type);
        SetStr(this->algorithm, "ALGORITHM", algorithm_arg);
    }

    void SetProductInfo2(std::string mds1_polar, std::string mds2_polar, std::string compression_arg)
    {
        SetStr(mds1_tx_rx_polar, "MDS1_TX_RX_POLAR", mds1_polar);
        SetStr(mds2_tx_rx_polar, "MDS2_TX_RX_POLAR", mds2_polar);
        SetStr(compression, "COMPRESSION", compression_arg);
    }

    void SetProductInfo3(int azimuth_looks_arg, int range_looks_arg)
    {
        SetAc(azimuth_looks, "AZIMUTH_LOOKS", azimuth_looks_arg);
        SetAc(range_looks, "RANGE_LOOKS", range_looks_arg);
    }

    void SetProductInfo4(float range_spacing_arg, float azimuth_spacing_arg, float line_time_interval_arg)
    {
        SetAfl(range_spacing, "RANGE_SPACING", range_spacing_arg, "<m>");
        SetAfl(azimuth_spacing, "AZIMUTH_SPACING", azimuth_spacing_arg, "<m>");
        SetAfl(line_time_interval, "LINE_TIME_INTERVAL", line_time_interval_arg, "<s>");
    }

    void SetProductInfo5(int line_length_arg, std::string data_type_arg)
    {
        SetAs(line_length, "LINE_LENGTH", line_length_arg, "<samples>");
        SetStr(data_type, "DATA_TYPE", data_type_arg);
    }

    void SetDefaults() {
        memset(this, ' ', sizeof(*this));
        ClearSpare(spare_1);
        ClearSpare(spare_2);
    }
};

static_assert(__builtin_offsetof(Lvl1SPH, slice_position) == 82);
static_assert(__builtin_offsetof(Lvl1SPH, num_slices) == 102);
static_assert(__builtin_offsetof(Lvl1SPH, last_line_time) == 164);
static_assert(__builtin_offsetof(Lvl1SPH, first_near_lat) == 209);
static_assert(__builtin_offsetof(Lvl1SPH, swath) == 681);
static_assert(__builtin_offsetof(Lvl1SPH, sample_type) == 711);
static_assert(__builtin_offsetof(Lvl1SPH, mds1_tx_rx_polar) == 754);
static_assert(__builtin_offsetof(Lvl1SPH, azimuth_looks) == 820);
static_assert(__builtin_offsetof(Lvl1SPH, range_spacing) == 856);
static_assert(sizeof(Lvl1SPH) == 6099);
