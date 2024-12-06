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

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "envisat_mph_sph_parser.h"
#include "sar/orbit_state_vector.h"

namespace alus::dorisorbit {

    class Parsable final {
    public:

        struct L1ProductMetadata {
//            SetChar(phase, "PHASE", 'X');
//            SetAc(cycle, "CYCLE", 0);
//            SetAs(rel_orbit, "REL_ORBIT", 0);
//            SetAs(abs_orbit, "ABS_ORBIT", 0);
//            SetStr(state_vector_time, "STATE_VECTOR_TIME", "");
//            strcpy((char*)&delta_ut1[0], "DELTA_UT1=+.281903<s>");  // TODO
//            delta_ut1[sizeof(delta_ut1) - 1] = '\n';
//            SetAdo73(x_position, "X_POSITION", 0.0, "<m>");
//            SetAdo73(y_position, "Y_POSITION", 0.0, "<m>");
//            SetAdo73(z_position, "Z_POSITION", 0.0, "<m>");
//            SetAdo46(x_velocity, "X_VELOCITY", 0.0, "<m/s>");
//            SetAdo46(y_velocity, "Y_VELOCITY", 0.0, "<m/s>");
//            SetAdo46(z_velocity, "Z_VELOCITY", 0.0, "<m/s>");
//            SetStr(vector_source, "VECTOR_SOURCE", "PC");
            std::string orbit_name;
            std::string processing_centre;
            std::string processing_time;
            char phase;
            boost::posix_time::ptime state_vector_time;
            std::string vector_source;
            uint32_t cycle;
            uint32_t rel_orbit;
            uint32_t abs_orbit;
            double delta_ut1;
            double x_position;
            double y_position;
            double z_position;
            double x_velocity;
            double y_velocity;
            double z_velocity;
        };

        static Parsable TryCreateFrom(std::string_view file_or_folder);

        const std::vector<OrbitStateVector> &
        CreateOrbitInfo(boost::posix_time::ptime start, boost::posix_time::ptime stop); // AssembleOsvFor()

        const L1ProductMetadata& GetL1ProductMetadata() const { return l1_product_metadata_; }

        ~Parsable() = default;

    private:

        static Parsable CreateFromFile(std::string_view filename);
        static Parsable CreateFromDirectoryListing(std::string_view directory);

        Parsable() = delete;

        Parsable(ProductHeader mph, ProductHeader sph, std::string dsd_records);

        void UpdateMetadataForL1Product(boost::posix_time::ptime start, boost::posix_time::ptime stop);

        struct EntryListing {
            boost::posix_time::ptime start;
            boost::posix_time::ptime end;
            std::filesystem::path file_path;
        };
        Parsable(std::vector<EntryListing> listing);

        // From https://earth.esa.int/eogateway/documents/20142/37627/Readme-file-for-Envisat-DORIS-POD.pdf chapter 2
        enum QualityFlag : unsigned {
            PRECISE_ADJUSTED = 3,
            PRECISE_MANOEUVRE = 4,
            PRECISE_TRACKING_GAP = 5,
            PRECISE_LESS_THAN_24_HRS = 6,
            PRELIMINARY_1_TO_2_DAYS = 7,
            PRELIMINARY_OVER_2_DAYS_OR_MANOEUVRE = 8
        };

        struct PointEntryInfo {
            uint32_t absolute_orbit;
            double ut1_delta;
            QualityFlag quality;
        };

        ProductHeader mph_;
        ProductHeader sph_;
        std::string dsd_records_;
        std::vector<OrbitStateVector> osv_;
        std::vector<PointEntryInfo> osv_metadata_;
        std::vector<EntryListing> listing_;

        L1ProductMetadata l1_product_metadata_;
    };
}
