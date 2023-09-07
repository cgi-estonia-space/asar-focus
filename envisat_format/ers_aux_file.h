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

#include <boost/date_time/posix_time/posix_time.hpp>

#include "envisat_format/envisat_aux_file.h"

namespace alus::asar::envisat_format {
    void FindINSFile(std::string aux_root, boost::posix_time::ptime start, InstrumentFile& ins_file,
                     std::string& filename);
}