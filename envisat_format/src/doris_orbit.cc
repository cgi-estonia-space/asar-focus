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

#include "doris_orbit.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <locale>
#include <numeric>
#include <sstream>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include "envisat_mph_sph_parser.h"
#include "checks.h"
#include "filesystem_util.h"

namespace {

// Following are defined in https://earth.esa.int/eogateway/documents/20142/37627/Readme-file-for-Envisat-DORIS-POD.pdf
// chapter 2 "Doris POD data"
// Currently known ERS DORIS orbit files to be used 'POD_REAPER_v1_071991_062003' are having extra HHMMSS in filename
// and has more MDSR items, all other aspects are technically similar and therefore no distinction is made for other
// properties like MPH and SPH sizes.
constexpr size_t DORIS_FILENAME_LENGTH_A{61};  // DOR_VOR_AXVF-P20120424_120200_20040101_215528_20040103_002328
constexpr size_t DORIS_FILENAME_LENGTH_B{68};  // DOR_VOR_AXVF-P19950503_210000_19950503_210000_210000_19950505_030000
constexpr size_t ENVISAT_DORIS_MPH_LENGTH_BYTES{1247};
constexpr size_t ENVISAT_DORIS_SPH_LENGTH_BYTES{378};
constexpr size_t ENVISAT_DORIS_MDS_LENGTH_BYTES{204981};
constexpr size_t ERS_DORIS_MDS_LENGTH_BYTES{232329};
constexpr size_t ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES{206606};
constexpr size_t ERS_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES{233954};
constexpr size_t ENVISAT_DORIS_MDSR_LENGTH_BYTES{129};
constexpr size_t ENVISAT_DORIS_MDSR_COUNT{1589};
constexpr size_t ERS_DORIS_MDSR_COUNT{1801};
constexpr size_t ENVISAT_DORIS_MDSR_ITEM_COUNT{11};  // Date and time are considered separate
static_assert(ENVISAT_DORIS_MDSR_COUNT * ENVISAT_DORIS_MDSR_LENGTH_BYTES == ENVISAT_DORIS_MDS_LENGTH_BYTES);
static_assert(ENVISAT_DORIS_MPH_LENGTH_BYTES + ENVISAT_DORIS_SPH_LENGTH_BYTES + ENVISAT_DORIS_MDS_LENGTH_BYTES ==
              ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES);
static_assert(ERS_DORIS_MDSR_COUNT * ENVISAT_DORIS_MDSR_LENGTH_BYTES == ERS_DORIS_MDS_LENGTH_BYTES);
static_assert(ENVISAT_DORIS_MPH_LENGTH_BYTES + ENVISAT_DORIS_SPH_LENGTH_BYTES + ERS_DORIS_MDS_LENGTH_BYTES ==
              ERS_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES);
constexpr std::string_view ENVISAT_DORIS_TIMESTAMP_PATTERN{"%d-%b-%Y %H:%M:%S%f"};
// DOR_VOR_AXVF-P20120424_120200_20040101_215528_20040103_002328
constexpr std::string_view ENVISAT_DORIS_TIMESTAMP_FILENAME_PATTERN{"%Y%m%d_%H%M%S"};

boost::posix_time::ptime ParseDate(const std::string& date_string, std::locale format) {
    std::stringstream stream(date_string);
    stream.imbue(format);
    boost::posix_time::ptime date;
    stream >> date;

    return date;
}
}  // namespace

namespace alus::dorisorbit {

Parsable::Parsable(ProductHeader mph, ProductHeader sph, std::string dsd_records)
    : mph_{std::move(mph)}, sph_{std::move(sph)}, dsd_records_{std::move(dsd_records)} {}

Parsable::Parsable(std::vector<EntryListing> listing) : listing_{std::move(listing)} {}

const std::vector<OrbitStateVector>& Parsable::CreateOrbitInfo(boost::posix_time::ptime start,
                                                               boost::posix_time::ptime stop) {
    if (listing_.size() > 0) {
        const auto listing = listing_;  // Make a copy for possible instantiation.
        bool found{false};
        for (const auto& l : listing) {
            if (l.start < start && l.end > stop) {
                *this = CreateFromFile(l.file_path.c_str());
                this->listing_ = listing;
                found = true;
                break;
            }
        }
        if (!found) {
            ERROR_EXIT("Could not find orbit file entry that would cover sensing time span of the dataset");
        }
    }

    osv_.clear();
    osv_metadata_.clear();

    constexpr size_t ORBIT_DELTA_MINUTE_COUNT{5};
    const auto delta_minutes = boost::posix_time::minutes(ORBIT_DELTA_MINUTE_COUNT);
    const auto start_delta = start - delta_minutes;
    const auto end_delta = stop + delta_minutes;

    std::istringstream record_stream(dsd_records_);
    std::string record;
    // std::locale will take an ownership over this one.
    const auto* timestamp_styling = new boost::posix_time::time_input_facet(ENVISAT_DORIS_TIMESTAMP_PATTERN.data());
    const auto locale = std::locale(std::locale::classic(), timestamp_styling);
    while (std::getline(record_stream, record)) {
        std::vector<std::string> items;
        boost::split(items, record, boost::is_any_of(" \n"), boost::token_compress_on);
        if (items.size() != ENVISAT_DORIS_MDSR_ITEM_COUNT) {
            ERROR_EXIT("MDSR COUNT is not a standard one for orbit file");
        }
        const auto item_datetime = items.front() + " " + items.at(1);  //.substr(0, 8);
        auto date = ParseDate(item_datetime, locale);
        if (date.is_not_a_date_time()) {
            ERROR_EXIT("Unparseable date '" + item_datetime +
                       "' for format: " + std::string(ENVISAT_DORIS_TIMESTAMP_PATTERN));
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
        osv_.push_back(a);
        PointEntryInfo pei;
        pei.absolute_orbit = strtoul(items.at(3).c_str(), nullptr, 10);
        pei.ut1_delta = strtod(items.at(2).c_str(), nullptr);
        pei.quality = static_cast<QualityFlag>(strtoul(items.at(10).c_str(), nullptr, 10));
        osv_metadata_.push_back(pei);
    }

    if (osv_.size() < ORBIT_DELTA_MINUTE_COUNT * 2) {
        ERROR_EXIT("There were less than required " + std::to_string(ORBIT_DELTA_MINUTE_COUNT * 2) +
                   " entries parsed from orbit file (over a timespan of " +
                   std::to_string(ORBIT_DELTA_MINUTE_COUNT * 2) + " minutes), probably a corrupt orbit dataset " +
                   "or implementation error.");
    }

    UpdateMetadataForL1Product(start, stop);

    return osv_;
}

Parsable Parsable::CreateFromFile(std::string_view filename) {
    const auto file_size = std::filesystem::file_size(filename);
    if (file_size != ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES &&
        file_size != ERS_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES) {
        ERROR_EXIT("Expected DORIS orbit file to be " +
                   std::to_string(ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES) + " bytes for ENVISAT or " +
                   std::to_string(ERS_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES) + " bytes for ERS, but actual " +
                   "file size is " + std::to_string(file_size) + " bytes");
    }

    std::fstream in_stream;
    in_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    in_stream.open(filename.data(), std::ios::in);

    std::vector<char> data(file_size);
    in_stream.read(data.data(), file_size);
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
    const auto* timestamp_styling =
        new boost::posix_time::time_input_facet(ENVISAT_DORIS_TIMESTAMP_FILENAME_PATTERN.data());
    const auto locale = std::locale(std::locale::classic(), timestamp_styling);
    for (const auto& l : listings) {
        const auto string_name = l.filename().string();
        const auto filename_length = string_name.length();
        if (filename_length != DORIS_FILENAME_LENGTH_A && filename_length != DORIS_FILENAME_LENGTH_B) {
            continue;
        }

        const auto file_size = std::filesystem::file_size(l);
        if (file_size != ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES &&
            file_size != ERS_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES) {
            continue;
        }

        std::vector<std::string> items;
        boost::split(items, string_name, boost::is_any_of("_"), boost::token_compress_on);
        // DOR_VOR_AXVF-P20120424_120200_20040101_215528_20040103_002328
        // DOR_VOR_AXVF-P19950503_210000_19950503_210000_210000_19950505_030000
        const auto item_count = items.size();
        if (item_count < 8 || item_count > 9) {
            continue;
        }

        auto end_date_index_yymmdd{6u};
        auto end_date_index_hhmmss{7u};
        if (item_count == 8) {
            if (items.front() != "DOR" || items.at(1) != "VOR" || items.at(2).length() != 14 ||
                items.at(3).length() != 6 || items.at(4).length() != 8 || items.at(5).length() != 6 ||
                items.at(6).length() != 8 || items.at(7).length() != 6) {
                continue;
            }
        }

        if (item_count == 9) {
            if (items.front() != "DOR" || items.at(1) != "VOR" || items.at(2).length() != 14 ||
                items.at(3).length() != 6 || items.at(4).length() != 8 || items.at(5).length() != 6 ||
                items.at(6).length() != 6 || items.at(7).length() != 8 || items.at(8).length() != 6) {
                continue;
            }
            end_date_index_yymmdd += 1;
            end_date_index_hhmmss += 1;
        }

        const auto start_date = ParseDate(items.at(4) + "_" + items.at(5), locale);
        const auto end_date =
            ParseDate(items.at(end_date_index_yymmdd) + "_" + items.at(end_date_index_hhmmss), locale);

        if (start_date.is_not_a_date_time() || end_date.is_not_a_date_time()) {
            ERROR_EXIT("Unparseable date from DORIS orbit file '" + string_name +
                       "' for format: " + std::string(ENVISAT_DORIS_TIMESTAMP_FILENAME_PATTERN));
        }

        doris_listing.push_back({start_date, end_date, l});
    }

    if (doris_listing.size() == 0) {
        throw std::runtime_error("Could not determine any valid Doris Orbit files out of the " +
                                 std::string(directory));
    }
    return Parsable(std::move(doris_listing));
}

Parsable Parsable::TryCreateFrom(std::string_view file_or_folder) {
    if (std::filesystem::is_regular_file(file_or_folder)) {
        return CreateFromFile(file_or_folder);
    } else if (std::filesystem::is_directory(file_or_folder)) {
        return CreateFromDirectoryListing(file_or_folder);
    } else {
        ERROR_EXIT("Unidentified orbit source - " + std::string(file_or_folder));
    }
}

void Parsable::UpdateMetadataForL1Product(boost::posix_time::ptime start, boost::posix_time::ptime stop) {
    (void)stop;
    bool found{false};
    const auto filtered_osv_count = osv_.size();
    for (size_t i{}; i < filtered_osv_count; i++) {
        const auto& osv = osv_.at(i);
        // Based on observed results of the blueprints, the last state vector that is before start-stop is fetched.
        if (osv.time > start - boost::posix_time::minutes(1)) {
            l1_product_metadata_.delta_ut1 = osv_metadata_.at(i).ut1_delta;
            l1_product_metadata_.abs_orbit = osv_metadata_.at(i).absolute_orbit;

            l1_product_metadata_.state_vector_time = osv.time;
            l1_product_metadata_.x_position = osv.x_pos;
            l1_product_metadata_.y_position = osv.y_pos;
            l1_product_metadata_.z_position = osv.z_pos;
            l1_product_metadata_.x_velocity = osv.x_vel;
            l1_product_metadata_.y_velocity = osv.y_vel;
            l1_product_metadata_.z_velocity = osv.z_vel;

            found = true;
            break;
        }
    }

    if (!found) {
        ERROR_EXIT("Failed to construct L1 product metadata from orbit source, probably corrupt orbit file.");
    }

    l1_product_metadata_.orbit_name = mph_.Get("PRODUCT");
    const auto vector_source_long = sph_.Get("DS_NAME");
    if (vector_source_long.find("DORIS PRECISE") != std::string::npos) {
        l1_product_metadata_.vector_source = "DP";
    } else if (vector_source_long.find("REAPER ORBIT") != std::string::npos) {
        l1_product_metadata_.vector_source = "RO";
    } else {
        l1_product_metadata_.vector_source = "??";
    }
    l1_product_metadata_.phase = mph_.Get("PHASE").front();
    l1_product_metadata_.cycle = strtoul(mph_.Get("CYCLE").c_str(), nullptr, 10);
}

}  // namespace alus::dorisorbit
