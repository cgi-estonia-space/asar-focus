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

    std::vector<OrbitStateVector> CreateOrbitInfo() const;

    ~Parsable() = default;

private:

    Parsable() = delete;
    Parsable(ProductHeader mph, ProductHeader sph, std::string dsd_records);

    ProductHeader _mph;
    ProductHeader _sph;
    std::string _dsd_records;
};

}