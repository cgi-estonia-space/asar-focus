/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "args.h"

#include <sstream>
#include <string>

#include <boost/program_options.hpp>

#include <iostream>
namespace {
std::optional<std::string_view> OptionalString(const std::string& str) {
    if (str.empty()) {
        return std::nullopt;
    }

    return str;
}

[[maybe_unused]] void DebugArgs(boost::program_options::variables_map& vm) {
    std::cout << "help/h " << vm.count("help") << "/" << vm.count("h") << std::endl;
    std::cout << "aux " << vm.count("aux") << std::endl;
    std::cout << "orb " << vm.count("orb") << std::endl;
    std::cout << "output/o " << vm.count("output") << "/" << vm.count("o") << std::endl;
    std::cout << "sensing_start " << vm.count("sensing_start") << std::endl;
}

}  // namespace

namespace alus::asar {

namespace po = boost::program_options;

void Args::Construct() {
    args_.add_options()("help,h", po::bool_switch()->default_value(false), "Print help");

    // clang-format off
    args_.add_options()
    ("input,i", po::value<std::string>(&ds_path_)->required(), "Level 0 ERS or ENVISAT dataset")
    ("aux", po::value<std::string>(&aux_path_)->required(), "Auxiliary files folder path")
    ("orb", po::value<std::string>(&orbit_path_)->required(), "Doris Orbit file or folder path")
    ("output,o", po::value<std::string>(&output_path_)->required(), "Destination path for focussed dataset")
    ("type,t", po::value<std::string>(&focussed_product_type_)->required(), "Focussed product type")
    ("sensing_start", po::value<std::string>(&process_sensing_start_),
        "Sensing start time <IN FORMAT>. If \"sensing_stop\" is not specified, product type based sensing duration is "
        "added to start for the end time or until packets last.")
    ("sensing_stop", po::value<std::string>(&process_sensing_end_),
        "Sensing stop time <IN FORMAT>. Requires \"sensing_start\" time to be specified.");
}

void Args::Check() {
    boost::program_options::notify(vm_);

    if (vm_.count("sensing_stop") && !vm_.count("sensing_start")) {
        throw std::invalid_argument("The argument 'sensing_stop' requires 'sensing_start' to be define.");
    }
}

Args::Args(const std::vector<char *> &args) {
    Construct();
    po::store(po::parse_command_line(static_cast<int>(args.size()), args.data(), args_), vm_);
    const auto argc = args.size();
    const auto check_help = [](const char* a) { return strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0; };
    const auto has_help = std::find_if(args.cbegin(), args.cend(), check_help) != args.end();
    if (argc != 0 && !has_help) {
        Check();
    } else {
        precheck_help = true;
    }
}

bool Args::IsHelpRequested() const {
    return precheck_help;
}

std::string Args::GetHelp() const {
    std::stringstream help;
    help << args_;
    return help.str();
}

std::optional<std::string_view> Args::GetProcessSensingStart() const {
    return OptionalString(process_sensing_start_);
}

std::optional<std::string_view> Args::GetProcessSensingEnd() const {
    return OptionalString(process_sensing_end_);
}

}
