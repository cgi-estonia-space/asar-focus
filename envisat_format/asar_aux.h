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

#include <boost/date_time/posix_time/ptime.hpp>

namespace alus::asar::aux {

    enum class Type {
        PROCESSOR_CONFIGURATION,
        INSTRUMENT_CHARACTERIZATION,
        EXTERNAL_CALIBRATION,
        EXTERNAL_CHARACTERIZATION
    };

    std::string GetPathFrom(std::string aux_root, boost::posix_time::ptime ds_start, Type t);
}