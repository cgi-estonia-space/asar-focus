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

namespace {
std::optional<std::string_view> OptionalString(const std::string& str) {
    if (str.empty()) {
        return std::nullopt;
    }

    return str;
}
}

namespace alus::asar {

std::optional<std::string_view> Args::GetProcessSensingStart() const {
    return OptionalString(process_sensing_start_);
}

std::optional<std::string_view> Args::GetProcessSensingEnd() const {
    return OptionalString(process_sensing_end_);
}

}
