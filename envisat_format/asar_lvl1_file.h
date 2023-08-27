#pragma once

#include "envisat_lvl1_dsd.h"
#include "envisat_mph.h"
#include "envisat_ph.h"
#include "envisat_sph.h"

#include "asar_lvl0_parser.h"

struct __attribute__((packed)) EnvisatIMS {
    MPH mph;
    Lvl1SPH sph;
    SummaryQualityADS summary_quality;
    MainProcessingParametersADS main_processing_params;
    DopplerCentroidParameters dop_centroid_coeffs;
    ChirpParameters chirp_params;
    GeoLocationADSR geolocation_grid[12];
};

struct MDS {
    int n_records;
    int record_size;

    char* buf = nullptr;

    ~MDS() { delete[] buf; }
};

void WriteLvl1(const SARMetadata& sar_meta, const ASARMetadata& asar_meta, MDS& mds);
