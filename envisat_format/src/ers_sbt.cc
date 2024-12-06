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

#include "ers_sbt.h"

#include <cstdint>
#include <stdexcept>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "date_time_util.h"
#include "envisat_types.h"
#include "envisat_utils.h"
#include "parse_util.h"
#include "patc.h"

namespace {

bool patc_cor_set{false};
alus::asar::envformat::aux::Patc patc_cor{};
boost::posix_time::ptime patc_time_point;

constexpr auto SBT_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint32_t, 32>()};

}  // namespace

namespace alus::asar::envformat::ers {

void RegisterPatc(const aux::Patc& patc) {
    if (patc.epoch_1950 < 0) {
        throw std::runtime_error("Supplied PATC has negative epoch for days since 1950.");
    }

    if (patc.milliseconds < 0) {
        throw std::runtime_error("Supplied PATC has negative epoch for milliseconds.");
    }

    const auto epoch = alus::util::date_time::YYYYMMDD("19500101");

    // Days between 19500101 and 19910101
    if (patc.epoch_1950 < 14975) {
        const auto err_date = epoch + boost::posix_time::time_duration(patc.epoch_1950 * 24, 0, 0, 0);
        throw std::runtime_error(
            "Supplied PATC epoch days has value that is earlier than year 1990. Resulting date - " +
            to_simple_string(err_date));
    }

    patc_cor = patc;
    patc_time_point = epoch + boost::posix_time::time_duration(patc.epoch_1950 * 24, 0, 0, 0);
    patc_time_point += boost::posix_time::milliseconds(patc.milliseconds);
    patc_cor_set = true;
}

void AdjustIspSensingTime(mjd& isp_sensing_time, uint32_t sbt, uint32_t, bool overflow) {
    if (!patc_cor_set) {
        return;
    }

    boost::posix_time::ptime adjusted_ptime;
    int64_t delta_count{};
    if (sbt < patc_cor.sbt_counter && !overflow) {
        delta_count = static_cast<int64_t>(sbt) - patc_cor.sbt_counter;
    }
    else {
        delta_count = parseutil::CounterGap<uint32_t, SBT_MAX>(patc_cor.sbt_counter, sbt);
    }
    const int64_t nanoseconds_diff = static_cast<int64_t>(patc_cor.sbt_period) * delta_count;
    const auto microseconds_val = static_cast<int64_t>(nanoseconds_diff / 1000);
    adjusted_ptime = patc_time_point + boost::posix_time::microseconds(microseconds_val);

    isp_sensing_time = PtimeToMjd(adjusted_ptime);
}

}  // namespace alus::asar::envformat::ers
