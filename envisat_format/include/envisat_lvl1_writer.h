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

#include <string_view>

#include "envisat_lvl1_ads.h"
#include "envisat_lvl1_mph.h"
#include "envisat_lvl1_sph.h"
#include "envisat_mph_sph_str_utils.h"

#include "envisat_lvl0_parser.h"

struct __attribute__((packed)) EnvisatSubFiles {
    Lvl1MPH mph;
    Lvl1SPH sph;
    SummaryQualityADS summary_quality;
    MainProcessingParametersADS main_processing_params;
    DopplerCentroidParameters dop_centroid_coeffs;
    ChirpParameters chirp_params;
    GeoLocationADSR geolocation_grid[12];
    SrGrADSR srgr_params;
};

struct MDS {
    int n_records = 0;
    int record_size = 0;
    char* buf = nullptr;

    MDS() = default;
    ~MDS() { delete[] buf; }
    MDS(const MDS&) = delete;
    MDS& operator=(const MDS&) = delete;
};

std::vector<uint8_t> ConstructEnvisatFileHeader(EnvisatSubFiles& ims, const SARMetadata& sar_meta, const ASARMetadata& asar_meta, const MDS& mds,
                  std::string_view software_ver);