/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/
#include "envisat_format/asar_aux.h"

#include <filesystem>
#include <iostream>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include "util/checks.h"
#include "util/date_time_util.h"
#include "util/filesystem_util.h"

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
}

namespace alus::asar::aux {

    std::string GetPathFrom(std::string aux_root, boost::posix_time::ptime start, Type t) {
        const auto file_list = util::filesystem::GetFileListingRecursively(aux_root);

        const auto id_string = DetermineDatasetIdFrom(t);

        const auto *timestamp_styling = new boost::posix_time::time_input_facet(DATE_PATTERN.data());
        const auto locale = std::locale(std::locale::classic(), timestamp_styling);
        std::string satellite_id{};
        for (const auto &f: file_list) {
            std::vector<std::string> items;
            boost::split(items, f.filename().string(), boost::is_any_of("_"), boost::token_compress_on);

            if (items.size() != 8) {
                continue;
            }

            if (satellite_id.empty()) {
                satellite_id = items.front();
            }

            const auto &current_sat_id = items.front();
            if (current_sat_id != satellite_id) {
                ERROR_EXIT("There are mixed satellite auxiliary files in the supplied aux folder - " + aux_root);
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
}