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

#include "envisat_format/envisat_mph_sph_parser.h"
#include "util/filesystem_util.h"

namespace {

// Following are defined in https://earth.esa.int/eogateway/documents/20142/37627/Readme-file-for-Envisat-DORIS-POD.pdf
// chapter 2 "Doris POD data"
    constexpr size_t ENVISAT_DORIS_FILENAME_LENGTH{61}; // DOR_VOR_AXVF-P20120424_120200_20040101_215528_20040103_002328
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
    constexpr std::string_view ENVISAT_DORIS_TIMESTAMP_PATTERN{"%d-%b-%Y %H:%M:%S%f"};
    // DOR_VOR_AXVF-P20120424_120200_20040101_215528_20040103_002328
    constexpr std::string_view ENVISAT_DORIS_TIMESTAMP_FILENAME_PATTERN{"%Y%M%d_%H%M%S"};

    boost::posix_time::ptime ParseDate(const std::string &date_string, std::locale format) {
        std::stringstream stream(date_string);
        stream.imbue(format);
        boost::posix_time::ptime date;
        stream >> date;

        return date;
    }
}  // namespace

namespace alus::dorisorbit {

    Parsable::Parsable(ProductHeader mph, ProductHeader sph, std::string dsd_records) : _mph{std::move(mph)},
                                                                                        _sph{std::move(sph)},
                                                                                        _dsd_records{std::move(
                                                                                                dsd_records)} {
    }

    Parsable::Parsable(std::vector<EntryListing> listing) : _listing{std::move(listing)} {}

    const std::vector<OrbitStateVector> &
    Parsable::CreateOrbitInfo(boost::posix_time::ptime start, boost::posix_time::ptime stop) {

        if (_listing.size() > 0) {
            const auto listing = _listing; // Make a copy for possible instantiation.
            bool found{false};
            for (const auto &l: listing) {
                if (l.start < start && l.end > stop) {
                    *this = CreateFromFile(l.file_path.c_str());
                    this->_listing = listing;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Could not find orbit file entry that would cover sensing time spane of the dataset"
                          << std::endl;
                exit(1);
            }
        }

        _osv.clear();
        _osv_metadata.clear();

        const auto delta_minutes = boost::posix_time::minutes(5);
        const auto start_delta = start - delta_minutes;
        const auto end_delta = stop + delta_minutes;

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

            if (date < start_delta || date > end_delta) {
                continue;
            }

            OrbitStateVector a;
            a.time = date;
            a.x_pos = strtod(items.at(4).c_str(), nullptr);
            a.y_pos = strtod(items.at(5).c_str(), nullptr);
            a.z_pos = strtod(items.at(6).c_str(), nullptr);
            a.x_vel = strtod(items.at(7).c_str(), nullptr);
            a.y_vel = strtod(items.at(8).c_str(), nullptr);
            a.z_vel = strtod(items.at(9).c_str(), nullptr);
            _osv.push_back(a);
            PointEntryInfo pei;
            pei.absolute_orbit = items.at(3);
            pei.ut1_delta = items.at(2);
            pei.quality = static_cast<QualityFlag>(strtoul(items.at(10).c_str(), nullptr, 10));
            _osv_metadata.push_back(pei);
        }

        return _osv;
    }

    Parsable Parsable::CreateFromFile(std::string_view filename) {
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

    Parsable Parsable::CreateFromDirectoryListing(std::string_view directory) {
        const auto listings = alus::util::filesystem::GetFileListingRecursively(directory);

        std::vector<EntryListing> doris_listing;
        const auto *timestamp_styling = new boost::posix_time::time_input_facet(
                ENVISAT_DORIS_TIMESTAMP_FILENAME_PATTERN.data());
        const auto locale = std::locale(std::locale::classic(), timestamp_styling);
        for (const auto &l: listings) {
            const auto string_name = l.filename().string();
            if (string_name.length() != ENVISAT_DORIS_FILENAME_LENGTH) {
                continue;
            }

            if (std::filesystem::file_size(l) != ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES) {
                continue;
            }

            std::vector<std::string> items;
            boost::split(items, string_name, boost::is_any_of("_"), boost::token_compress_on);
            // DOR_VOR_AXVF-P20120424_120200_20040101_215528_20040103_002328
            if (items.size() != 8) {
                continue;
            }

            if (items.front() != "DOR" || items.at(1) != "VOR" || items.at(2).length() != 14 ||
                items.at(3).length() != 6 || items.at(4).length() != 8 || items.at(5).length() != 6 ||
                items.at(6).length() != 8 || items.at(7).length() != 6) {
                continue;
            }

            const auto start_date = ParseDate(items.at(4) + "_" + items.at(5), locale);
            const auto end_date = ParseDate(items.at(6) + "_" + items.at(7), locale);

            if (start_date.is_not_a_date_time() || end_date.is_not_a_date_time()) {
                std::cerr << "Unparseable date from DORIS orbit file '" + string_name + "' for format: " +
                             std::string(ENVISAT_DORIS_TIMESTAMP_FILENAME_PATTERN) << std::endl;
                exit(1);
            }

            doris_listing.push_back({start_date, end_date, l});
        }

        return Parsable(std::move(doris_listing));
    }

    Parsable Parsable::TryCreateFrom(std::string_view file_or_folder) {
        if (std::filesystem::is_regular_file(file_or_folder)) {
            return CreateFromFile(file_or_folder);
        } else if (std::filesystem::is_directory(file_or_folder)) {
            return CreateFromDirectoryListing(file_or_folder);
        } else {
            std::cerr << "Unidentified orbit source - " << file_or_folder << std::endl;
            exit(1);
        }
    }

}  // namespace alus::dorisorbit
