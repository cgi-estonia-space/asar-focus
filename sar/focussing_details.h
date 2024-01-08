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

#include <cstddef>
#include <string>

namespace alus::sar::focus {

struct RcmcParameters {
    size_t window_size;
    double window_coef_range;
    std::string window_name;
};

RcmcParameters GetRcmcParameters();

}  // namespace alus::sar::focus