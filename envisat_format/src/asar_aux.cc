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
#include "asar_aux.h"

#include <filesystem>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include "checks.h"
#include "date_time_util.h"
#include "filesystem_util.h"

namespace {
constexpr std::string_view PROCESSOR_CONFIGURATION_ID{"CON"};
constexpr std::string_view INSTRUMENT_CHARACTERIZATION_ID{"INS"};
constexpr std::string_view EXTERNAL_CALIBRATION_ID{"XCA"};
constexpr std::string_view EXTERNAL_CHARACTERIZATION_ID{"XCH"};

std::string_view DetermineDatasetIdFrom(alus::asar::aux::Type t) {
    switch (t) {
        case alus::asar::aux::Type::EXTERNAL_CALIBRATION:
            return EXTERNAL_CALIBRATION_ID;
        case alus::asar::aux::Type::EXTERNAL_CHARACTERIZATION:
            return EXTERNAL_CHARACTERIZATION_ID;
        case alus::asar::aux::Type::PROCESSOR_CONFIGURATION:
            return PROCESSOR_CONFIGURATION_ID;
        case alus::asar::aux::Type::INSTRUMENT_CHARACTERIZATION:
            return INSTRUMENT_CHARACTERIZATION_ID;
        default:
            ERROR_EXIT("Should not reach here");
    }
}

constexpr std::string_view DATE_PATTERN{"%Y%m%d_%H%M%S"};
}  // namespace

namespace alus::asar::aux {

std::string GetPathFrom(std::string aux_root, boost::posix_time::ptime start, Type t) {
    const auto file_list = util::filesystem::GetFileListingRecursively(aux_root);

    const auto id_string = DetermineDatasetIdFrom(t);

    const auto* timestamp_styling = new boost::posix_time::time_input_facet(DATE_PATTERN.data());
    const auto locale = std::locale(std::locale::classic(), timestamp_styling);
    std::string satellite_id{};
    for (const auto& f : file_list) {
        if (f.has_extension()) {
            continue;
        }
        std::vector<std::string> items;
        boost::split(items, f.filename().string(), boost::is_any_of("_"), boost::token_compress_on);

        if (items.size() != 8) {
            continue;
        }

        const auto& current_sat_id = items.front();
        if (current_sat_id != "ASA" && current_sat_id != "ER1" && current_sat_id != "ER2") {
            continue;
        }

        if (satellite_id.empty()) {
            satellite_id = items.front();
        }

        if (current_sat_id != satellite_id) {
            throw std::runtime_error("There are mixed satellite " + std::string(id_string) +
                                     " auxiliary files in the supplied aux folder - " + aux_root);
        }

        if (items.at(1) != id_string) {
            continue;
        }

        const auto start_str = items.at(4) + " " + items.at(5);
        auto start_date = util::date_time::ParseDate(start_str, locale);
        if (start_date.is_not_a_date_time()) {
            ERROR_EXIT("Unparseable date '" + start_str + "' for format: " + std::string(DATE_PATTERN));
        }

        const auto end_str = items.at(6) + " " + items.at(7);
        auto end_date = util::date_time::ParseDate(end_str, locale);
        if (end_date.is_not_a_date_time()) {
            ERROR_EXIT("Unparseable date '" + end_str + "' for format: " + std::string(DATE_PATTERN));
        }

        if (start > start_date && start < end_date) {
            return f.string();
        }
    }

    return {};
}
}  // namespace alus::asar::aux