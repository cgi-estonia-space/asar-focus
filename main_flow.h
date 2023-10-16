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

#include "envisat_format/asar_constants.h"

#include <optional>
#include <string_view>

#include "boost/date_time/posix_time/posix_time.hpp"

#include "envisat_format/doris_orbit.h"
#include "envisat_format/envisat_aux_file.h"
#include "envisat_format/envisat_lvl0_parser.h"
#include "sar/sar_metadata.h"

namespace alus::asar::mainflow {

void CheckAndLimitSensingStartEnd(boost::posix_time::ptime& product_sensing_start,
                                  boost::posix_time::ptime& product_sensing_stop,
                                  std::optional<std::string_view> user_sensing_start,
                                  std::optional<std::string_view> user_sensing_end);
specification::ProductTypes TryDetermineProductType(std::string_view product_name);

void TryFetchOrbit(alus::dorisorbit::Parsable& orbit_source, ASARMetadata& asar_meta, SARMetadata& sar_meta);

void FetchAuxFiles(InstrumentFile& ins_file, ConfigurationFile& conf_file, ASARMetadata& asar_meta,
                   specification::ProductTypes product_type, std::string_view aux_path);

}  // namespace alus::asar::mainflow
