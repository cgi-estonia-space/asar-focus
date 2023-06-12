#pragma once

#include "envisat_ph.h"
#include "envisat_sph.h"
#include "envisat_mph.h"
#include "envisat_dsd.h"

#include "asar_metadata.h"


struct __attribute__((packed)) ERS_IM_LVL1
{
    MPH mph;
    Lvl1SPH sph;
    SummaryQualityADS summary_quality;
    MainProcessingParametersADS main_processing_params;
    DopplerCentroidParameters dop_centroid_coeffs;
    ChirpParameters chirp_params;
    GeoLocationADSR geolocation_grid[12];
};


struct MDS
{
    int n_records;
    int record_size;


    char* buf = nullptr;


    ~MDS()
    {
        delete[] buf;
    }
};



// 1247
// 6099
// 170
// 2009
// 55
// 1483
// 512
static_assert(sizeof(ERS_IM_LVL1));







void WriteLvl1(SARMetadata& el, MDS& mds);

