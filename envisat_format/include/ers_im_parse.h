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