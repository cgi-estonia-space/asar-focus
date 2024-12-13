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

#include "args.h"

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options.hpp>

namespace {
std::optional<std::string_view> OptionalString(const std::string& str) {
    if (str.empty()) {
        return std::nullopt;
    }

    return str;
}

}  // namespace

namespace alus::asar {

namespace po = boost::program_options;

void Args::Construct() {
    visible_args_.add_options()("help,h", po::bool_switch()->default_value(false), "Print help");

    // clang-format off
    visible_args_.add_options()
    ("input,i", po::value<std::string>(&ds_path_)->required(), "Level 0 ERS or ENVISAT dataset")
    ("aux", po::value<std::string>(&aux_path_)->required(), "Auxiliary files folder path")
    ("orb", po::value<std::string>(&orbit_path_)->required(), "Doris Orbit file or folder path")
    ("output,o", po::value<std::string>(&output_path_)->required(), "Destination path for focussed dataset")
    ("type,t", po::value<std::string>(&focussed_product_type_)->required(), "Focussed product type")
    ("sensing_start", po::value<std::string>(&process_sensing_start_),
        "Sensing start time in format '%Y%m%d_%H%M%S%F' e.g. 20040229_212504912000. "
        "If \"sensing_stop\" is not specified, product type based sensing duration is used or until packets last.")
    ("sensing_stop", po::value<std::string>(&process_sensing_end_),
        "Sensing stop time in format '%Y%m%d_%H%M%S%F' e.g. 20040229_212504912000. "
        " Requires \"sensing_start\" time to be specified.")
    ("log", po::value<std::string>(&log_level_arg_)->default_value("verbose"),
        "Log level, one of the following - verbose|debug|info|warning|error");

    hidden_args_.add_options()
        ("plot", po::bool_switch(&plots_on_)->default_value(false))
        ("intensity", po::bool_switch(&store_intensity_)->default_value(false));

    args_.add(visible_args_).add(hidden_args_);
    // clang-format on
}

void Args::Check() {
    boost::program_options::notify(vm_);

    if (vm_.count("sensing_stop") && !vm_.count("sensing_start")) {
        throw std::invalid_argument("The argument 'sensing_stop' requires 'sensing_start' to be define.");
    }

    log_level_ = TryFetchLogLevelFrom(log_level_arg_);
}

log::Level Args::TryFetchLogLevelFrom(std::string_view level_arg) {
    constexpr std::array<std::string_view, 5> ALLOWED_LEVELS{"verbose", "debug", "info", "warning", "error"};
    constexpr std::array<log::Level, 5> LOG_LEVELS{log::Level::VERBOSE, log::Level::DEBUG, log::Level::INFO,
                                                   log::Level::WARNING, log::Level::ERROR};

    size_t level_index{0};
    for (const auto level_str : ALLOWED_LEVELS) {
        if (boost::iequals(level_arg, level_str)) {
            return LOG_LEVELS.at(level_index);
        }
        level_index++;
    }

    throw std::invalid_argument("'" + std::string(level_arg) +
                                "' is not a valid log level. Valid ones are - verbose|debug|info|warning|error");
}

Args::Args(const std::vector<char*>& args) {
    if (args.size() == 0) {
        throw std::logic_error("Programming error when supplying arguments to parser - arg array length is 0.");
    }
    Construct();
    po::store(po::parse_command_line(static_cast<int>(args.size()), args.data(), args_), vm_);
    const auto argc = args.size();
    const auto check_help = [](const char* a) { return strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0; };
    const auto has_help = std::find_if(args.cbegin(), args.cend(), check_help) != args.end();
    if (argc > 1 && !has_help) {
        Check();
    } else {
        precheck_help = true;
    }
}

bool Args::IsHelpRequested() const { return precheck_help; }

std::string Args::GetHelp() const {
    std::stringstream help;
    help << visible_args_;
    return help.str();
}

std::optional<std::string_view> Args::GetProcessSensingStart() const { return OptionalString(process_sensing_start_); }

std::optional<std::string_view> Args::GetProcessSensingEnd() const { return OptionalString(process_sensing_end_); }

}  // namespace alus::asar
