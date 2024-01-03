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

#include <vector>

#include <cufft.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "envisat_lvl0_parser.h"

namespace alus::asar::envformat {

std::vector<ForecastMeta> FetchErsL0ImForecastMeta(const std::vector<char>& file_data, const DSD_lvl0& mdsr,
                                                   boost::posix_time::ptime packets_start_filter,
                                                   boost::posix_time::ptime packets_stop_filter,
                                                   size_t& packets_before_start, size_t& packets_after_stop);

RawSampleMeasurements ParseErsLevel0ImPackets(const std::vector<char>& file_data, size_t mdsr_offset_bytes,
                                              const std::vector<ForecastMeta>& entries_to_be_parsed,
                                              InstrumentFile& ins_file,
                                              std::vector<CommonPacketMetadata>& common_metadata);

void ParseErsLevel0ImPackets(const std::vector<char>& file_data, const DSD_lvl0& mdsr, SARMetadata& sar_meta,
                             ASARMetadata& asar_meta, cufftComplex** d_parsed_packets, InstrumentFile& ins_file,
                             boost::posix_time::ptime packets_start_filter = boost::posix_time::not_a_date_time,
                             boost::posix_time::ptime packets_stop_filter = boost::posix_time::not_a_date_time);

}  // namespace alus::asar::envformat