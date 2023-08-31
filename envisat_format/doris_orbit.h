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

#include <string>
#include <string_view>
#include <vector>

#include "envisat_mph_sph_parser.h"
#include "sar/orbit_state_vector.h"

namespace alus::dorisorbit {

    class Parsable final {
    public:
        static Parsable TryCreateFrom(std::string_view filename);

        const std::vector<OrbitStateVector> &
        CreateOrbitInfo(boost::posix_time::ptime start, boost::posix_time::ptime stop); // AssembleOsvFor()

        std::string GetDatasetName() const;

        std::string GetAbsoluteOrbit() const;

        ~Parsable() = default;

    private:

        Parsable() = delete;

        Parsable(ProductHeader mph, ProductHeader sph, std::string dsd_records);

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
            std::string absolute_orbit;
            std::string ut1_delta;
            QualityFlag quality;
        };

        ProductHeader _mph;
        ProductHeader _sph;
        std::string _dsd_records;
        std::vector<OrbitStateVector> _osv;
        std::vector<PointEntryInfo> _osv_metadata;

    };
}
