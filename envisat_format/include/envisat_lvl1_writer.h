/*
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c)
 * by CGI Estonia AS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

std::vector<uint8_t> ConstructEnvisatFileHeader(EnvisatSubFiles& ims, const SARMetadata& sar_meta,
                                                const ASARMetadata& asar_meta, const MDS& mds,
                                                std::array<double, 4> statistics, std::string_view software_ver);