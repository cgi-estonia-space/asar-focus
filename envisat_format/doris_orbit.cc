/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "doris_orbit.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>
#include <sstream>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include "envisat_mph_sph_parser.h"

namespace {

// Following are defined in https://earth.esa.int/eogateway/documents/20142/37627/Readme-file-for-Envisat-DORIS-POD.pdf
// chapter 2 "Doris POD data"
    constexpr size_t ENVISAT_DORIS_MPH_LENGTH_BYTES{1247};
    constexpr size_t ENVISAT_DORIS_SPH_LENGTH_BYTES{378};
    constexpr size_t ENVISAT_DORIS_MDS_LENGTH_BYTES{204981};
    constexpr size_t ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES{206606};
    constexpr size_t ENVISAT_DORIS_MDSR_LENGTH_BYTES{129};
    constexpr size_t ENVISAT_DORIS_MDSR_COUNT{1589};
    constexpr size_t ENVISAT_DORIS_MDSR_ITEM_COUNT{11};  // Date and time are considered separate
    static_assert(ENVISAT_DORIS_MDSR_COUNT * ENVISAT_DORIS_MDSR_LENGTH_BYTES == ENVISAT_DORIS_MDS_LENGTH_BYTES);
    static_assert(ENVISAT_DORIS_MPH_LENGTH_BYTES + ENVISAT_DORIS_SPH_LENGTH_BYTES + ENVISAT_DORIS_MDS_LENGTH_BYTES ==
                  ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES);
//constexpr std::string_view ENVISAT_DORIS_TIMESTAMP_PATTERN{"%d-%m-%Y %H:%M:%S"};
    constexpr std::string_view ENVISAT_DORIS_TIMESTAMP_PATTERN{"%d-%b-%Y %H:%M:%S%f"};
}  // namespace

namespace alus::dorisorbit {

    Parsable::Parsable(ProductHeader mph, ProductHeader sph, std::string dsd_records) : _mph{std::move(mph)},
                                                                                        _sph{std::move(sph)},
                                                                                        _dsd_records{std::move(
                                                                                                dsd_records)} {
    }

    std::vector<OrbitStateVector> Parsable::CreateOrbitInfo() const {
        std::vector<OrbitStateVector> orbits;

        std::istringstream record_stream(_dsd_records);
        std::string record;
        // std::locale will take an ownership over this one.
        const auto *timestamp_styling = new boost::posix_time::time_input_facet(ENVISAT_DORIS_TIMESTAMP_PATTERN.data());
        const auto locale = std::locale(std::locale::classic(), timestamp_styling);
        while (std::getline(record_stream, record)) {
            std::vector<std::string> items;
            boost::split(items, record, boost::is_any_of(" \n"), boost::token_compress_on);
            if (items.size() != ENVISAT_DORIS_MDSR_ITEM_COUNT) {
                std::cerr << "MDSR COUNT is not a standard one for orbit file" << std::endl;
                exit(1);
            }
            const auto item_datetime = items.front() + " " + items.at(1);//.substr(0, 8);
            std::stringstream stream(item_datetime);
            stream.imbue(locale);
            boost::posix_time::ptime date;
            stream >> date;

            if (date.is_not_a_date_time()) {
                std::cerr << "Unparseable date '" + item_datetime + "' for format: " +
                             std::string(ENVISAT_DORIS_TIMESTAMP_PATTERN) << std::endl;
                exit(1);
            }
            OrbitStateVector a;
            a.time = date;
            a.x_pos = strtod(items.at(4).c_str(), nullptr);
            a.y_pos = strtod(items.at(5).c_str(), nullptr);
            a.z_pos = strtod(items.at(6).c_str(), nullptr);
            a.x_vel = strtod(items.at(7).c_str(), nullptr);
            a.y_vel = strtod(items.at(8).c_str(), nullptr);
            a.z_vel = strtod(items.at(9).c_str(), nullptr);
            orbits.push_back(a);
        }

        return orbits;
    }

    Parsable Parsable::TryCreateFrom(std::string_view filename) {
        size_t file_sz = std::filesystem::file_size(filename.data());
        if (file_sz != ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES) {
            std::cerr << "Expected DORIS orbit file to be " << ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES
                      << " bytes, but actual file size is " << file_sz << " bytes" << std::endl;
        }

        std::fstream in_stream;
        in_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        in_stream.open(filename.data(), std::ios::in);

        std::vector<char> data(file_sz);
        in_stream.read(data.data(), file_sz);
        in_stream.close();

        ProductHeader mph = {};
        mph.Load(data.data(), ENVISAT_DORIS_MPH_LENGTH_BYTES);
        ProductHeader sph = {};
        sph.Load(data.data() + ENVISAT_DORIS_MPH_LENGTH_BYTES, ENVISAT_DORIS_SPH_LENGTH_BYTES);

        std::string dsd_records(data.cbegin() + ENVISAT_DORIS_MPH_LENGTH_BYTES + ENVISAT_DORIS_SPH_LENGTH_BYTES,
                                data.cend());

        return Parsable(std::move(mph), std::move(sph), std::move(dsd_records));
    }

}  // namespace alus::dorisorbit
