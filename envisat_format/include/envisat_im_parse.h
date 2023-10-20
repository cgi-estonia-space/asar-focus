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

#include <complex>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "envisat_lvl0_parser.h"

namespace alus::asar::envformat {

void ParseEnvisatLevel0ImPackets(const std::vector<char>& file_data, const DSD_lvl0& mdsr, SARMetadata& sar_meta,
                                 ASARMetadata& asar_meta, std::vector<std::complex<float>>& img_data,
                                 InstrumentFile& ins_file,
                                 boost::posix_time::ptime packets_start_filter = boost::posix_time::not_a_date_time,
                                 boost::posix_time::ptime packets_stop_filter = boost::posix_time::not_a_date_time);

}